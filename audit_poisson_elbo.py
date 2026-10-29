"""Audit Poisson ELBO accounting on a saved checkpoint without updating it."""
import argparse
import json
import math
from unittest import mock

import torch

from distributions import Poisson
from poisson_sampling import PoissonSampler
import utils


def reduce_vars(tensor):
    return tensor.flatten(1).sum(1)


class PoissonAudit:
    def __init__(self, sampler='native', seed=1):
        if sampler not in {'native', 'cpu', 'fresh'}:
            raise ValueError('Unknown audit sampler: %s' % sampler)
        self.sampler = sampler
        self.backend_sampler = PoissonSampler(sampler, seed)
        self.reset()

    def reset(self):
        self.records = {}
        self.groups = []

    def __enter__(self):
        audit = self
        original_log = Poisson.log_p
        original_kl = Poisson.kl

        def log_p(dist, z):
            value = original_log(dist, z)
            audit.records[id(dist)] = (z.detach(), value.detach())
            return value

        def kl(q, p):
            value = original_kl(q, p)
            z, logq = audit.records[id(q)]
            zp, logp = audit.records[id(p)]
            if not torch.equal(z, zp):
                raise RuntimeError('q and p were evaluated on different samples.')
            z = z.double()
            qr, pr = q.rate.double(), p.rate.double()
            ratio = qr.log() - pr.log()
            exact_kl = qr * ratio + pr - qr
            exact_mc = z * ratio + pr - qr
            audit.groups.append({
                'analytic64': reduce_vars(exact_kl),
                'mc64': reduce_vars(exact_mc),
                'residual64': reduce_vars((z - qr) * ratio),
                'residual_variance': reduce_vars(qr * ratio.square()),
                'kl_numeric_error': reduce_vars(value).double() - reduce_vars(exact_kl),
                'density_numeric_error': reduce_vars(logq.double() - logp.double()) - reduce_vars(exact_mc),
                'count_residual': reduce_vars(z - qr),
                'count_variance': reduce_vars(qr),
            })
            return value

        def sample(dist, *args, **kwargs):
            if kwargs.get('estimator') != 'exact':
                raise RuntimeError('Poisson audit requires eval exact sampling.')
            return audit.backend_sampler(dist.rate), None

        self.patches = [mock.patch.object(Poisson, 'log_p', log_p),
                        mock.patch.object(Poisson, 'kl', kl),
                        mock.patch.object(Poisson, 'sample', sample)]
        for patch in self.patches:
            patch.start()
        return self

    def __exit__(self, *exc):
        for patch in reversed(self.patches):
            patch.stop()


def evaluate_batch(model, x, samples, audit):
    weights, values, per_group = [], [], []
    with torch.no_grad():
        for _ in range(samples):
            audit.reset()
            logits, logq, logp, kl_all, _ = model(x)
            recon = utils.reconstruction_loss(model.decoder_output(logits), x, crop=model.crop_output)
            if len(audit.groups) != len(kl_all):
                raise RuntimeError('Not all latent groups were audited.')
            analytic = torch.stack([k.double() for k in kl_all]).sum(0)
            mc = logq.double() - logp.double()
            weights.append(-recon.double() - mc)
            values.append(torch.stack([recon.double(), analytic, mc], dim=1))
            per_group.append(torch.stack([torch.stack(list(g.values()), dim=1)
                                          for g in audit.groups], dim=1))
    weights = torch.stack(weights, dim=1)
    values = torch.stack(values, dim=1)
    groups = torch.stack(per_group, dim=1)
    iw_nll = -(torch.logsumexp(weights, dim=1) - math.log(samples))
    mc_nelbo = -weights.mean(1)
    if (iw_nll - mc_nelbo).max().item() > 1e-8:
        raise RuntimeError('Same-sample Jensen inequality failed.')
    return values, groups, iw_nll


def run(options):
    # Load only a checkpoint supplied by the caller; compatible with torch 1.6.
    try:
        checkpoint = torch.load(options.checkpoint, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(options.checkpoint, map_location='cpu')
    args = checkpoint['args']
    if args.latent_distribution != 'poisson' or args.num_nf != 0:
        raise ValueError('This audit requires all-Poisson latents and num_nf=0.')
    torch.manual_seed(options.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(options.seed)
    from model import AutoEncoder
    import datasets
    args.data = options.data or args.data
    args.distributed = False
    args.batch_size = options.batch_size
    model = AutoEncoder(args, None, checkpoint.get('arch_instance') or
                        utils.get_arch_cells(args.arch_instance))
    incompatible = model.load_state_dict(checkpoint['state_dict'], strict=False)
    # Conv2D's weight_normalized is a derived, non-parameter tensor.
    bad_missing = [k for k in incompatible.missing_keys if not k.endswith('weight_normalized')]
    bad_extra = [k for k in incompatible.unexpected_keys if not k.endswith('weight_normalized')]
    if bad_missing or bad_extra:
        raise RuntimeError('Checkpoint mismatch: %s %s' % (bad_missing, bad_extra))
    for layer in model.modules():
        if hasattr(layer, 'init_done'):
            layer.init_done = True
    model = model.to(options.device).eval()
    _, loader, _ = datasets.get_loaders(args)
    totals, group_totals, count, images, nll_sum = None, None, 0, 0, 0.
    with PoissonAudit(options.sampler, seed=options.seed) as audit:
        for batch_index, batch in enumerate(loader):
            if options.max_batches and batch_index >= options.max_batches:
                break
            x = utils.pre_process(batch[0].to(options.device), args.num_x_bits)
            values, groups, nll = evaluate_batch(model, x, options.samples, audit)
            v = values.sum((0, 1)).cpu()
            g = groups.sum((0, 1)).cpu()
            totals = v if totals is None else totals + v
            group_totals = g if group_totals is None else group_totals + g
            count += x.size(0) * options.samples
            images += x.size(0)
            nll_sum += nll.sum().item()
            print('audited batch %d, images %d' % (batch_index, images), flush=True)
    recon, kl, mc = (totals / count).tolist()
    rows = []
    keys = list(audit.groups[0])
    for index, row in enumerate(group_totals):
        r = {key: row[i].item() / count for i, key in enumerate(keys)}
        r['group'] = index
        r['residual_se'] = math.sqrt(max(row[keys.index('residual_variance')].item(), 0.)) / count
        r['count_residual_se'] = math.sqrt(max(row[keys.index('count_variance')].item(), 0.)) / count
        rows.append(r)
    variance = group_totals[:, keys.index('residual_variance')].sum().item()
    report = {'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
              'device': options.device, 'sampler': options.sampler, 'seed': options.seed,
              'checkpoint_epoch': checkpoint.get('epoch'), 'images': images,
              'samples_per_image': options.samples, 'recon': recon,
              'analytic_nelbo': recon + kl, 'mc_nelbo': recon + mc,
              'iw_nll': nll_sum / images, 'mc_minus_analytic': mc - kl,
              'total_kl_numeric_error': kl - sum(r['analytic64'] for r in rows),
              'total_log_ratio_numeric_error': mc - sum(r['mc64'] for r in rows),
              'poisson_residual64': sum(r['residual64'] for r in rows),
              'poisson_residual_se': math.sqrt(max(variance, 0.)) / count,
              'groups': rows}
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--max-batches', type=int, default=10, help='0: entire validation set')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--sampler', choices=['native', 'cpu', 'fresh'], default='native')
    parser.add_argument('--seed', type=int, default=1)
    options = parser.parse_args()
    if options.samples < 1 or options.batch_size < 1 or options.max_batches < 0:
        parser.error('samples/batch-size must be positive; max-batches must be nonnegative')
    run(options)
