#!/usr/bin/env python3
"""TensorBoard gradient diagnostics for adaptive Poisson CTS and smooth STE.

Pass the usual train.py options after this script's --grad-* options. The
monitor expects one GPU, one scale, two 4x4 Poisson ST groups, and no flows.
It supports the Fisher and Adamax optimizers and smooth STE on or off.

grad_positions/current_batch/ keeps [C,H,W] and logs signed scalars for
batch_mean and sample_0, plus batch_mean_abs. Sample indices denote current
batch slots. Gradients are for the batch-mean training objectives, unscaled
from AMP. See docs/poisson_smooth_ste.md for tags, commands, and resume usage.
"""
import argparse
import copy
import inspect
import math
import sys
from pathlib import Path


class GradientMonitor:
    def __init__(self, every=100, hist_every=1000, position_images=1, summary=False):
        self.every = every
        self.hist_every = hist_every
        self.position_images = position_images
        self.summary = summary
        self.resume_path = None
        self.resume_checkpoint = None
        self.first_training_step = True
        self.active = False
        self.records = {}
        self.nodes = {}
        self.activations = {}
        self.sampled = []
        self.handles = []
        self.model = None
        self.patched_class = None

    def install(self, poisson_class):
        """Capture exact graph tensors without recomputing the sampler."""
        monitor = self
        self.patched_class = poisson_class
        self.original_init = poisson_class.__init__
        self.original_sample = poisson_class.sample
        self.original_arrival_counts = poisson_class._arrival_counts

        def initialize(dist, log_rate, *args, **kwargs):
            monitor.original_init(dist, log_rate, *args, **kwargs)
            if monitor.active:
                monitor.records[id(dist)] = {
                    'dist': dist, 'raw': log_rate, 'rate': dist.rate,
                }

        def arrival_counts(dist):
            hard_count, soft_count = monitor.original_arrival_counts(dist)
            if monitor.active:
                monitor.records[id(dist)]['soft_count'] = soft_count
            return hard_count, soft_count

        def sample(dist, *args, **kwargs):
            result = monitor.original_sample(dist, *args, **kwargs)
            if monitor.active:
                record = monitor.records.get(id(dist))
                if record is None:
                    raise RuntimeError('Poisson sample has no captured distribution.')
                record['z'] = result[0]
                monitor.sampled.append(record)
            return result

        poisson_class.__init__ = initialize
        poisson_class.sample = sample
        poisson_class._arrival_counts = arrival_counts

    def close(self):
        self.active = False
        self.clear_graph()
        for handle in self.handles:
            handle.remove()
        self.handles = []
        if self.patched_class is not None:
            self.patched_class.__init__ = self.original_init
            self.patched_class.sample = self.original_sample
            self.patched_class._arrival_counts = self.original_arrival_counts
            self.patched_class = None

    def clear_graph(self):
        self.records.clear()
        self.nodes.clear()
        self.activations.clear()
        self.sampled.clear()

    def bind(self, model):
        import torch
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
            raise ValueError('Gradient monitor currently supports exactly one GPU/process.')
        if (model.latent_distribution != 'poisson' or
                model.poisson_gradient_estimator != 'straight_through' or
                model.num_latent_scales != 1 or list(model.groups_per_scale) != [2] or
                list(model.z0_size[-2:]) != [4, 4] or model.num_flows != 0):
            raise ValueError('Expected Poisson ST, one scale, two 4x4 groups, num_nf=0. '
                             'For MNIST use num_preprocess_blocks=3 (32 -> 16 -> 8 -> 4).')
        self.model = model
        if not self.summary:
            return
        for name, module in model.named_modules():
            kind = type(module).__name__
            if name in ('enc0.0', 'enc0.2', 'dec_sampler.0.0', 'image_conditional.0'):
                self.handles.append(module.register_forward_hook(self.make_hook(name, 'elu')))
            elif kind == 'FactorizedReduce':
                self.handles.append(module.register_forward_hook(self.make_hook(name, 'swish_skip')))
            elif kind == 'SyncBatchNormSwish':
                self.handles.append(module.register_forward_hook(self.make_hook(name, 'fused_swish')))

    def make_hook(self, name, kind):
        def hook(module, inputs, output):
            if self.active:
                base = 'activation/' + name.replace('.', '/')
                self.nodes[base + '/output'] = output
                if kind != 'fused_swish':
                    self.nodes[base + '/input'] = inputs[0]
                    self.activations[base] = (kind, inputs[0])
        return hook

    def begin(self, model, writer, step, scaler, images=None):
        if self.model is None:
            self.bind(model)
        self.clear_graph()
        self.step = int(step)
        self.active = self.first_training_step or self.step % self.every == 0
        self.first_training_step = False
        if not self.active:
            return
        self.writer = writer
        self.scale = float(scaler.get_scale())
        if images is not None:
            for index in range(min(self.position_images, images.size(0))):
                self.writer.add_image('grad_positions/current_batch/sample_%d/input' % index,
                                      images[index].detach().float().cpu(), self.step)

    def scalar(self, name, value):
        value = float(value)
        if math.isfinite(value):
            self.writer.add_scalar('grad_monitor/' + name, value, self.step)

    def stats(self, name, tensor):
        import torch
        if not self.summary:
            return
        value = tensor.detach().double().reshape(-1)
        finite = torch.isfinite(value)
        self.scalar(name + '/nonfinite_fraction', (~finite).double().mean().item())
        value = value[finite]
        if value.numel() == 0:
            return
        self.scalar(name + '/mean', value.mean().item())
        self.scalar(name + '/mean_abs', value.abs().mean().item())
        self.scalar(name + '/rms', (value * value).mean().sqrt().item())
        self.scalar(name + '/l2', value.norm().item())
        self.scalar(name + '/max_abs', value.abs().max().item())
        self.scalar(name + '/zero_fraction', (value == 0).float().mean().item())
        if self.hist_every and self.step % self.hist_every == 0:
            self.writer.add_histogram('grad_monitor/' + name + '/histogram',
                                      value.cpu(), self.step)

    def positions(self, name, tensor):
        """One scalar per channel/row/column; keep batch and image identities explicit."""
        import torch
        value = tensor.detach().float().cpu()
        if value.dim() != 4:
            raise ValueError('Position logging expects a [B,C,H,W] tensor: ' + name)
        views = [('batch_mean', value.mean(dim=0)),
                 ('batch_mean_abs', value.abs().mean(dim=0))]
        views.extend(('sample_%d' % index, value[index])
                     for index in range(min(self.position_images, value.size(0))))
        for view, grid in views:
            fraction = (~torch.isfinite(grid)).float().mean().item()
            self.scalar('positions/' + view + '/' + name + '/nonfinite_fraction', fraction)
            for channel, plane in enumerate(grid.tolist()):
                for row, entries in enumerate(plane):
                    for column, item in enumerate(entries):
                        if math.isfinite(item):
                            self.writer.add_scalar(
                                'grad_positions/current_batch/%s/%s/c%02d/h%d_w%d' %
                                (view, name, channel, row, column), item, self.step)

    def probe(self, objective, tensors):
        import torch
        if not objective.requires_grad:
            return [None] * len(tensors)
        # autograd.grad returns gradients without accumulating parameter .grad.
        gradients = torch.autograd.grad(objective.float() * self.scale, tensors,
                                        retain_graph=True, allow_unused=True)
        return [None if grad is None else grad.detach().float() / self.scale
                for grad in gradients]

    def report(self, recon, weighted_kl, kl_all, total):
        import torch
        if not self.active:
            return
        if len(self.sampled) != 2 or any(list(r['z'].shape[-2:]) != [4, 4]
                                        for r in self.sampled):
            raise RuntimeError('Expected exactly two sampled 4x4 posterior groups.')
        for index, record in enumerate(self.sampled):
            name = 'q' + str(index)
            for key in ('raw', 'rate', 'z'):
                self.nodes[name + '/' + key] = record[key]
                self.stats('state/' + name + '/' + key, record[key])
                self.positions('posterior/' + name + '/' + key, record[key])
            dist = record['dist']
            base = 'sampling/' + name + '/'
            self.scalar(base + 'cts_num_terms', dist.last_cts_num_terms)
            self.scalar(base + 'cts_horizon', dist.last_cts_horizon)
            cts_weight = (dist._smooth_ste_weight() if dist.smooth_ste else
                          torch.ones_like(record['rate']))
            local = self.probe(record['z'].float().sum(), [record['rate'], record['raw']])
            if dist.smooth_ste:
                original_g, = self.probe(record['soft_count'].float().sum(), [record['rate']])
            else:
                original_g = local[0]
            for key, value in (('cts_weight', cts_weight), ('ste_weight', 1. - cts_weight),
                               ('cts_dz_drate', original_g), ('dz_drate', local[0]),
                               ('dz_draw', local[1])):
                if value is not None:
                    self.stats(base + key, value)
                    self.positions(base + key, value)
        priors = [r for r in self.records.values() if 'z' not in r and r['raw'].requires_grad]
        if len(priors) != 1:
            raise RuntimeError('Expected one trainable conditional prior p1.')
        for key in ('raw', 'rate'):
            self.nodes['p1/' + key] = priors[0][key]
            self.stats('state/p1/' + key, priors[0][key])
            self.positions('prior/p1/' + key, priors[0][key])
        for name, record in zip(('q0', 'q1', 'p1'), self.sampled + priors):
            raw = record['raw'].detach().float()
            slope = 1. - torch.tanh(raw / 5.).pow(2)
            self.stats('state/' + name + '/soft_clamp_slope', slope)
            self.scalar('state/' + name + '/soft_clamp_near_zero_fraction',
                        (slope.abs() < 1e-3).float().mean().item())
        for name, (kind, tensor) in self.activations.items():
            value = tensor.detach().float()
            if kind == 'elu':
                slope = torch.exp(torch.clamp(value, max=0.))
            else:
                sigmoid = torch.sigmoid(value)
                slope = sigmoid * (1. + value * (1. - sigmoid))
            self.stats('slope/' + name, slope)
            self.scalar('slope/' + name + '/near_zero_fraction', (slope.abs() < 1e-3).float().mean().item())
        named = [(name, value) for name, value in self.nodes.items() if value.requires_grad]
        names, tensors = zip(*named)
        objectives = [('recon', recon.mean()), ('weighted_kl', weighted_kl.mean()),
                      ('raw_kl0', kl_all[0].mean()), ('raw_kl1', kl_all[1].mean()),
                      ('total', total)]
        saved = {}
        for source, objective in objectives:
            self.scalar('loss/' + source, objective.detach().float().item())
            gradients = self.probe(objective, tensors)
            for name, gradient in zip(names, gradients):
                tag = source + '/' + name
                self.scalar(tag + '/connected', gradient is not None)
                if gradient is not None:
                    self.stats(tag, gradient)
                    if name.startswith(('q0/', 'q1/', 'p1/')):
                        self.positions('gradient/' + tag, gradient)
                    if source in ('recon', 'weighted_kl') and name in ('q0/raw', 'q1/raw'):
                        saved[(source, name)] = gradient.double().reshape(-1)
        for name in ('q0/raw', 'q1/raw'):
            a = saved.get(('recon', name))
            b = saved.get(('weighted_kl', name))
            valid = a is not None and b is not None
            if valid:
                denominator = a.norm() * b.norm()
                valid = bool(torch.isfinite(denominator).item() and denominator.item() > 0.)
            self.scalar('comparison/' + name + '/cosine_defined', valid)
            if valid:
                self.scalar('comparison/' + name + '/recon_kl_cosine', torch.dot(a, b).item() / denominator.item())
        self.clear_graph()

    def after_backward(self, model, scaler):
        import torch
        if not self.active:
            return
        scale = float(scaler.get_scale())
        self.scalar('amp/scale', scale)
        if self.summary:
            blocks = {}
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    block = name.split('.')[0]
                    blocks.setdefault(block, []).append(parameter.grad.detach().float().reshape(-1) / scale)
            for block, gradients in blocks.items():
                self.stats('parameter/' + block, torch.cat(gradients))

    def after_step(self, optimizer):
        if not self.active:
            return
        for name, value in getattr(optimizer, 'last_stats', {}).items():
            self.scalar('fisher/' + name, value)
        underlying = getattr(self.writer, 'writer', self.writer)
        if hasattr(underlying, 'flush'):
            underlying.flush()
        self.active = False

    @staticmethod
    def read_checkpoint(path):
        import torch
        kwargs = {'map_location': 'cpu'}
        # Modern torch defaults to weights_only=True; NVAE stores argparse.Namespace.
        if 'weights_only' in inspect.signature(torch.load).parameters:
            kwargs['weights_only'] = False
        return torch.load(str(path), **kwargs)

    def configure_resume(self, path):
        self.resume_path = Path(path).resolve()
        checkpoint = self.read_checkpoint(self.resume_path)
        required = {'args', 'state_dict', 'optimizer', 'scheduler', 'grad_scalar',
                    'epoch', 'global_step'}
        missing = required.difference(checkpoint)
        if missing:
            raise ValueError('Incomplete NVAE checkpoint; missing ' + ', '.join(sorted(missing)))
        saved = checkpoint['args']
        if int(getattr(saved, 'num_process_per_node', 1)) != 1 or int(getattr(saved, 'num_proc_node', 1)) != 1:
            raise ValueError('Resume requires a single-GPU checkpoint.')
        self.resume_checkpoint = checkpoint
        print('Resume checkpoint: %s; completed epochs=%d; global_step=%d' %
              (self.resume_path, checkpoint['epoch'], checkpoint['global_step']), flush=True)

    def resume_args(self, current):
        if self.resume_checkpoint is None:
            return current
        saved = copy.deepcopy(self.resume_checkpoint['args'])
        # Apply saved values over parser defaults so older saved args remain usable.
        vars(current).update(vars(saved))
        run_dir = self.resume_path.parent
        if self.resume_path.name != 'checkpoint.pt' or not run_dir.name.startswith('eval-'):
            raise ValueError('--grad-resume requires the run directory eval-*/checkpoint.pt.')
        current.root = str(run_dir.parent)
        current.save = run_dir.name[len('eval-'):]
        current.cont_training = True
        current.reset_optimizer = False
        current.global_rank = 0
        current.local_rank = 0
        current.node_rank = 0
        current.distributed = False
        if self.resume_checkpoint['epoch'] >= current.epochs:
            raise ValueError('Checkpoint has already completed its configured training epochs.')
        return current

    def load_checkpoint(self, path):
        if self.resume_checkpoint is None:
            return self.read_checkpoint(path)
        if Path(path).resolve() != self.resume_path:
            raise ValueError('The restored run path does not match --grad-resume.')
        checkpoint = self.resume_checkpoint
        self.resume_checkpoint = None
        return checkpoint

    def make_writer(self, factory, args):
        writer = factory(args.global_rank, args.save)
        if self.resume_checkpoint is not None and args.global_rank == 0:
            from tensorboardX import SummaryWriter
            writer.writer.close()
            writer.writer = SummaryWriter(logdir=args.save, flush_secs=20,
                                         purge_step=int(self.resume_checkpoint['global_step']))
        return writer


def self_test():
    """Exercise the real Poisson graph without the GPU training entry point."""
    import torch
    from distributions import Poisson

    class Writer:
        def __init__(self):
            self.values = {}
        def add_scalar(self, name, value, step):
            self.values[name] = value
        def add_histogram(self, *args):
            pass

    class Scaler:
        def get_scale(self):
            return 1024.

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.raw0 = torch.nn.Parameter(torch.full((2, 1, 4, 4), 5. * math.atanh(math.log(0.1) / 5.)))
            self.raw1 = torch.nn.Parameter(torch.full((2, 1, 4, 4), 0.1))
            self.prior_bias = torch.nn.Parameter(torch.zeros(2, 1, 4, 4))
            self.latent_distribution = 'poisson'
            self.poisson_gradient_estimator = 'straight_through'
            self.num_latent_scales = 1
            self.groups_per_scale = [2]
            self.z0_size = [1, 4, 4]
            self.num_flows = 0
        def forward(self):
            q0 = Poisson(self.raw0, smooth_ste=True)
            z0, _ = q0.sample(estimator='straight_through')
            p0 = Poisson(torch.zeros_like(z0))
            p1 = Poisson(self.prior_bias + 0.02 * z0)
            q1 = Poisson(self.raw1 + 0.1 * z0, smooth_ste=True)
            z1, _ = q1.sample(estimator='straight_through')
            kl = [q0.kl(p0).sum((1, 2, 3)), q1.kl(p1).sum((1, 2, 3))]
            recon = ((z0 + z1 - 2.) ** 2).sum((1, 2, 3))
            weighted = 0.3 * (0.7 * kl[0] + 1.3 * kl[1])
            return recon, weighted, kl, (recon + weighted).mean()

    model, writer, scaler = Model(), Writer(), Scaler()
    torch.manual_seed(123)
    baseline = model()
    baseline[-1].backward()
    expected = [p.grad.clone() for p in model.parameters()]
    expected_rng = torch.get_rng_state().clone()
    model.zero_grad()
    monitor = GradientMonitor(1, 0, summary=True)
    monitor.install(Poisson)
    try:
        torch.manual_seed(123)
        monitor.begin(model, writer, 0, scaler)
        result = model()
        reference_rate = monitor.sampled[1]['rate'].detach().clone()
        reference_gradient = monitor.probe(result[0].mean(), [monitor.sampled[1]['rate']])[0]
        monitor.report(*result)
        for row in range(4):
            for column in range(4):
                suffix = '/c00/h%d_w%d' % (row, column)
                prefix = 'grad_positions/current_batch/'
                assert writer.values[prefix + 'sample_0/posterior/q1/rate' + suffix] == reference_rate[0, 0, row, column].item()
                assert writer.values[prefix + 'sample_0/gradient/recon/q1/rate' + suffix] == reference_gradient[0, 0, row, column].item()
                assert writer.values[prefix + 'batch_mean_abs/gradient/recon/q1/rate' + suffix] == reference_gradient[:, 0, row, column].abs().mean().item()
        assert all(p.grad is None or not (p.grad != 0).any().item() for p in model.parameters())
        (result[-1] * scaler.get_scale()).backward()
        monitor.after_backward(model, scaler)
        monitor.after_step(None)
        for parameter, reference in zip(model.parameters(), expected):
            assert torch.allclose(parameter.grad / scaler.get_scale(), reference)
        assert torch.equal(torch.get_rng_state(), expected_rng)
        for name in ('recon/q0/raw/rms', 'recon/q1/raw/rms', 'raw_kl1/q0/raw/rms',
                     'sampling/q0/dz_drate/rms', 'parameter/raw0/rms'):
            assert math.isfinite(writer.values['grad_monitor/' + name]), name
        assert writer.values['grad_monitor/raw_kl0/q1/raw/connected'] == 0
        prefix = 'grad_positions/current_batch/sample_0/'
        suffix = '/c00/h0_w0'
        values = writer.values
        rate = values[prefix + 'posterior/q0/rate' + suffix]
        w = values[prefix + 'sampling/q0/cts_weight' + suffix]
        g = values[prefix + 'sampling/q0/cts_dz_drate' + suffix]
        blended = values[prefix + 'sampling/q0/dz_drate' + suffix]
        assert math.isclose(w, rate ** 4 / (rate ** 4 + 0.35 ** 4), rel_tol=1e-5)
        assert math.isclose(blended, (1. - w) + w * g, rel_tol=1e-5)
        raw = values[prefix + 'posterior/q0/raw' + suffix]
        chain = rate * (1. - math.tanh(raw / 5.) ** 2)
        assert math.isclose(values[prefix + 'sampling/q0/dz_draw' + suffix],
                            blended * chain, rel_tol=1e-5)
        assert values['grad_monitor/sampling/q0/cts_num_terms'] > 0.
        assert values['grad_monitor/sampling/q0/cts_horizon'] >= 1.

        # Missing nonfinite points must be distinguished from actual zeros.
        monitor.positions('nonfinite_test', torch.tensor([[[[float('nan'), 0.]]]]))
        assert prefix + 'nonfinite_test/c00/h0_w0' not in values
        assert values[prefix + 'nonfinite_test/c00/h0_w1'] == 0.
        assert values['grad_monitor/positions/sample_0/nonfinite_test/nonfinite_fraction'] == 0.5

        monitor.every = 100
        monitor.begin(model, writer, 2, scaler)
        model()
        assert not monitor.records and not monitor.nodes
        print('PASS: adaptive CTS, smooth STE and tanh derivatives, AMP unscaling, '
              'unchanged gradients/RNG, per-position logging, and nonfinite filtering.')
    finally:
        monitor.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--grad-repo', default='.')
    parser.add_argument('--grad-every', type=int, default=100)
    parser.add_argument('--grad-hist-every', type=int, default=1000)
    parser.add_argument('--grad-position-images', type=int, default=1)
    parser.add_argument('--grad-summary', action='store_true')
    parser.add_argument('--grad-resume', help='Resume eval-*/checkpoint.pt using its saved training args')
    parser.add_argument('--grad-checkpoint-info', action='store_true')
    parser.add_argument('--grad-self-test', action='store_true')
    parser.add_argument('--grad-help', action='store_true')
    options, training_args = parser.parse_known_args()
    if options.grad_help:
        parser.print_help()
        return
    if options.grad_every < 1 or options.grad_hist_every < 0:
        parser.error('--grad-every must be positive; --grad-hist-every must be >= 0.')
    if options.grad_hist_every and options.grad_hist_every % options.grad_every:
        parser.error('--grad-hist-every must be 0 or a multiple of --grad-every.')
    if options.grad_position_images < 0:
        parser.error('--grad-position-images must be >= 0.')
    if options.grad_resume and training_args:
        parser.error('With --grad-resume, training settings come from checkpoint. Use only --grad-* flags.')
    if options.grad_checkpoint_info and not options.grad_resume:
        parser.error('--grad-checkpoint-info requires --grad-resume.')
    repo = Path(options.grad_repo).resolve()
    sys.path.insert(0, str(repo))
    if options.grad_self_test:
        self_test()
        return
    monitor = GradientMonitor(options.grad_every, options.grad_hist_every,
                              options.grad_position_images, options.grad_summary)
    if options.grad_resume:
        monitor.configure_resume(options.grad_resume)
        if options.grad_checkpoint_info:
            return
    from distributions import Poisson
    from train import run_cli

    monitor.install(Poisson)
    try:
        run_cli(training_args, grad_monitor=monitor)
    finally:
        monitor.close()


if __name__ == '__main__':
    main()
