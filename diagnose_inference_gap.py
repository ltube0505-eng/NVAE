# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for NVAE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

"""Diagnose inference gaps in trained all-Poisson NVAE checkpoints.

The diagnostic freezes the trained model and locally refines one additive
log-rate correction per example and latent group.  It then evaluates both the
amortized and refined posteriors with exact Poisson samples.  The resulting
amortization improvement is a conservative lower bound because the local
family only adds prefix-independent corrections to the hierarchical encoder.
"""

import argparse
import json
import math
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

import datasets
import utils
from model import AutoEncoder


def parse_checkpoints(values):
    checkpoints = OrderedDict()
    for value in values:
        if '=' not in value:
            raise ValueError('checkpoint must use LABEL=PATH syntax: %s' % value)
        label, path = value.split('=', 1)
        label, path = label.strip(), path.strip()
        if not label or not path:
            raise ValueError('checkpoint must use non-empty LABEL=PATH syntax: %s' % value)
        if label in checkpoints:
            raise ValueError('duplicate checkpoint label: %s' % label)
        checkpoints[label] = path
    return checkpoints


def parse_iw_checkpoints(value, max_samples):
    checkpoints = sorted(set(int(item.strip()) for item in value.split(',') if item.strip()))
    if not checkpoints or checkpoints[0] < 1:
        raise ValueError('--iw_checkpoints must contain positive integers.')
    if checkpoints[-1] > max_samples:
        raise ValueError('--iw_checkpoints cannot exceed --iw_samples.')
    if checkpoints[-1] != max_samples:
        checkpoints.append(max_samples)
    return checkpoints


def importance_statistics(log_weights, checkpoints):
    """Return per-example IW-NLL values and weight-degeneracy diagnostics.

    Args:
        log_weights: tensor with shape [num_importance_samples, batch].
        checkpoints: increasing sample counts no larger than the first axis.
    """
    if log_weights.dim() != 2:
        raise ValueError('log_weights must have shape [samples, batch].')
    num_samples = log_weights.size(0)
    if checkpoints[-1] > num_samples:
        raise ValueError('importance checkpoint exceeds available samples.')

    nll = OrderedDict()
    for sample_count in checkpoints:
        estimate = torch.logsumexp(log_weights[:sample_count], dim=0) - math.log(sample_count)
        nll[str(sample_count)] = -estimate

    normalizer = torch.logsumexp(log_weights, dim=0)
    log_ess = 2. * normalizer - torch.logsumexp(2. * log_weights, dim=0)
    ess_fraction = torch.exp(log_ess) / float(num_samples)
    max_weight_fraction = torch.exp(torch.max(log_weights, dim=0)[0] - normalizer)
    return nll, ess_fraction, max_weight_fraction


def classify_gap(metrics, dominance_ratio=0.6, ess_threshold=0.01,
                 iw_stability_tolerance=0.5, refinement_tolerance=0.05):
    """Classify the dominant source while exposing inconclusive cases."""
    if metrics['refined_ess_fraction'] < ess_threshold or \
            abs(metrics['last_iw_tightening']) > iw_stability_tolerance:
        return 'inconclusive_importance_sampling'

    if metrics['surrogate_drop'] > refinement_tolerance and \
            metrics['amortization_gap'] <= refinement_tolerance:
        return 'surrogate_mismatch'

    total_gap = metrics['total_inference_gap']
    if total_gap <= refinement_tolerance:
        return 'no_material_inference_gap'

    amortization_share = metrics['amortization_gap'] / total_gap
    residual_share = metrics['residual_gap'] / total_gap
    if amortization_share >= dominance_ratio:
        return 'amortization_dominated'
    if residual_share >= dominance_ratio:
        return 'posterior_family_dominated_proxy'
    return 'mixed'


class PosteriorRateOffsets(object):
    """Per-example additive corrections attached to Poisson encoder heads."""

    def __init__(self, model, x):
        self.model = model
        self.enabled = False
        self.handles = []
        captured = [None] * len(model.enc_sampler)

        def capture(index):
            def hook(module, inputs, output):
                del module, inputs
                captured[index] = output.detach()
            return hook

        temporary = [module.register_forward_hook(capture(index))
                     for index, module in enumerate(model.enc_sampler)]
        model.eval()
        with torch.no_grad():
            model(x)
        for handle in temporary:
            handle.remove()
        if any(output is None for output in captured):
            raise RuntimeError('failed to capture every posterior head output.')

        self.offsets = nn.ParameterList([
            nn.Parameter(output[:, :output.size(1) // 2].new_zeros(
                output.size(0), output.size(1) // 2, output.size(2), output.size(3)))
            for output in captured
        ])
        self.base_batch_size = x.size(0)
        for index, module in enumerate(model.enc_sampler):
            self.handles.append(module.register_forward_hook(self._offset_hook(index)))

    def _offset_hook(self, index):
        def hook(module, inputs, output):
            del module, inputs
            if not self.enabled:
                return output
            offset = self.offsets[index]
            if output.size(0) % self.base_batch_size != 0:
                raise RuntimeError('expanded batch is incompatible with local posterior offsets.')
            repeats = output.size(0) // self.base_batch_size
            expanded = offset.repeat(repeats, 1, 1, 1)
            correction = torch.cat([expanded, torch.zeros_like(expanded)], dim=1)
            return output + correction
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def set_refinement_mode(model):
    """Enable training-time latent sampling without updating BN statistics."""
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) or \
                'BatchNorm' in module.__class__.__name__:
            module.eval()


def strict_nelbo(model, x):
    logits, _, _, kl_all, _ = model(x)
    output = model.decoder_output(logits)
    reconstruction = utils.reconstruction_loss(output, x, crop=model.crop_output)
    kl = torch.sum(torch.stack(kl_all, dim=0), dim=0)
    return reconstruction + kl


def refine_offsets(model, x, refiner, steps, learning_rate, mc_samples, grad_clip):
    if model.poisson_gradient_estimator not in {'straight_through', 'relaxed'}:
        raise ValueError('local pathwise refinement supports ST and relaxed checkpoints only.')
    optimizer = torch.optim.Adam(refiner.offsets.parameters(), lr=learning_rate)
    losses = []
    refiner.enabled = True
    set_refinement_mode(model)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = 0.
        for _ in range(mc_samples):
            loss = loss + torch.mean(strict_nelbo(model, x)) / float(mc_samples)
        loss.backward()
        if grad_clip > 0.:
            torch.nn.utils.clip_grad_norm_(refiner.offsets.parameters(), grad_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    return losses


def evaluate_posterior(model, x, refiner, enabled, num_samples, chunk_size,
                       iw_checkpoints, seed):
    refiner.enabled = enabled
    model.eval()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    nelbos, log_weights = [], []
    remaining = num_samples
    with torch.no_grad():
        while remaining > 0:
            current = min(chunk_size, remaining)
            expanded_x = x.repeat(current, 1, 1, 1)
            logits, log_q, log_p, kl_all, _ = model(expanded_x)
            output = model.decoder_output(logits)
            reconstruction = utils.reconstruction_loss(
                output, expanded_x, crop=model.crop_output)
            kl = torch.sum(torch.stack(kl_all, dim=0), dim=0)
            nelbos.append((reconstruction + kl).view(current, x.size(0)))
            log_weights.append((-reconstruction - log_q + log_p).view(current, x.size(0)))
            remaining -= current

    nelbos = torch.cat(nelbos, dim=0)
    log_weights = torch.cat(log_weights, dim=0)
    nll, ess_fraction, max_weight_fraction = importance_statistics(
        log_weights, iw_checkpoints)
    return {
        'nelbo': nelbos.mean(dim=0),
        'nll': nll,
        'ess_fraction': ess_fraction,
        'max_weight_fraction': max_weight_fraction,
    }


def load_model(path, device):
    checkpoint = torch.load(path, map_location='cpu')
    model_args = checkpoint['args']
    if not hasattr(model_args, 'ada_groups'):
        model_args.ada_groups = False
    if not hasattr(model_args, 'min_groups_per_scale'):
        model_args.min_groups_per_scale = 1
    if not hasattr(model_args, 'num_mixture_dec'):
        model_args.num_mixture_dec = 10
    architecture = utils.get_arch_cells(model_args.arch_instance)
    model = AutoEncoder(model_args, None, architecture)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    # Checkpoints already contain initialized convolution weights. Unlike the
    # original evaluate.py, this single-GPU diagnostic does not create a DDP
    # process group, so data-dependent initialization must not run again.
    for module in model.modules():
        if hasattr(module, 'init_done'):
            module.init_done = True
    model.to(device)
    model.eval()
    if model.latent_distribution != 'poisson':
        raise ValueError('diagnostic currently requires an all-Poisson checkpoint: %s' % path)
    if model.poisson_gradient_estimator not in {'straight_through', 'relaxed'}:
        raise ValueError('checkpoint is not ST or relaxed: %s' % path)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, model_args, checkpoint.get('epoch', None)


def cache_validation_batches(model_args, data_path, batch_size, max_batches, seed):
    model_args.data = data_path
    model_args.batch_size = batch_size
    model_args.distributed = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    _, valid_queue, _ = datasets.get_loaders(model_args)
    batches = []
    for index, batch in enumerate(valid_queue):
        if index >= max_batches:
            break
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        batches.append(utils.pre_process(x, model_args.num_x_bits).cpu())
    if not batches:
        raise RuntimeError('validation loader produced no batches.')
    return batches


def aggregate_batch_results(batch_results, iw_checkpoints, args):
    total_examples = sum(result['batch_size'] for result in batch_results)

    def weighted(name):
        return sum(result[name] * result['batch_size'] for result in batch_results) / total_examples

    aggregated = OrderedDict()
    aggregated['num_examples'] = total_examples
    aggregated['amortized_nelbo'] = weighted('amortized_nelbo')
    aggregated['refined_nelbo'] = weighted('refined_nelbo')
    aggregated['amortization_gap'] = aggregated['amortized_nelbo'] - aggregated['refined_nelbo']
    aggregated['surrogate_start'] = weighted('surrogate_start')
    aggregated['surrogate_end'] = weighted('surrogate_end')
    aggregated['surrogate_drop'] = aggregated['surrogate_start'] - aggregated['surrogate_end']
    aggregated['amortized_ess_fraction'] = weighted('amortized_ess_fraction')
    aggregated['refined_ess_fraction'] = weighted('refined_ess_fraction')
    aggregated['amortized_max_weight_fraction'] = weighted('amortized_max_weight_fraction')
    aggregated['refined_max_weight_fraction'] = weighted('refined_max_weight_fraction')

    aggregated['amortized_nll_iw'] = OrderedDict()
    aggregated['refined_nll_iw'] = OrderedDict()
    for sample_count in iw_checkpoints:
        key = str(sample_count)
        aggregated['amortized_nll_iw'][key] = weighted('amortized_nll_' + key)
        aggregated['refined_nll_iw'][key] = weighted('refined_nll_' + key)

    max_key = str(iw_checkpoints[-1])
    reference_nll = aggregated['refined_nll_iw'][max_key]
    aggregated['residual_gap'] = aggregated['refined_nelbo'] - reference_nll
    aggregated['total_inference_gap'] = aggregated['amortized_nelbo'] - reference_nll
    if len(iw_checkpoints) > 1:
        previous_key = str(iw_checkpoints[-2])
        aggregated['last_iw_tightening'] = \
            aggregated['refined_nll_iw'][previous_key] - reference_nll
    else:
        aggregated['last_iw_tightening'] = 0.
    aggregated['diagnosis'] = classify_gap(
        aggregated,
        dominance_ratio=args.dominance_ratio,
        ess_threshold=args.ess_threshold,
        iw_stability_tolerance=args.iw_stability_tolerance,
        refinement_tolerance=args.refinement_tolerance)
    return aggregated


def diagnose_checkpoint(label, path, batches, args, iw_checkpoints, device):
    model, model_args, epoch = load_model(path, device)
    batch_results = []
    for batch_index, cpu_x in enumerate(batches):
        x = cpu_x.to(device)
        refiner = PosteriorRateOffsets(model, x)
        try:
            refiner.enabled = False
            base = evaluate_posterior(
                model, x, refiner, False, args.iw_samples, args.iw_chunk_size,
                iw_checkpoints, args.seed + batch_index)
            losses = refine_offsets(
                model, x, refiner, args.local_steps, args.local_lr,
                args.local_mc_samples, args.local_grad_clip)
            refined = evaluate_posterior(
                model, x, refiner, True, args.iw_samples, args.iw_chunk_size,
                iw_checkpoints, args.seed + batch_index)
        finally:
            refiner.close()

        result = {
            'batch_size': x.size(0),
            'amortized_nelbo': float(base['nelbo'].mean().cpu()),
            'refined_nelbo': float(refined['nelbo'].mean().cpu()),
            'surrogate_start': float(np.mean(losses[:min(10, len(losses))])),
            'surrogate_end': float(np.mean(losses[-min(10, len(losses)):])),
            'amortized_ess_fraction': float(base['ess_fraction'].mean().cpu()),
            'refined_ess_fraction': float(refined['ess_fraction'].mean().cpu()),
            'amortized_max_weight_fraction': float(base['max_weight_fraction'].mean().cpu()),
            'refined_max_weight_fraction': float(refined['max_weight_fraction'].mean().cpu()),
        }
        for sample_count in iw_checkpoints:
            key = str(sample_count)
            result['amortized_nll_' + key] = float(base['nll'][key].mean().cpu())
            result['refined_nll_' + key] = float(refined['nll'][key].mean().cpu())
        batch_results.append(result)
        print('[%s] batch %d/%d: exact NELBO %.4f -> %.4f' % (
            label, batch_index + 1, len(batches), result['amortized_nelbo'],
            result['refined_nelbo']))

    metrics = aggregate_batch_results(batch_results, iw_checkpoints, args)
    metrics['checkpoint'] = path
    metrics['epoch'] = epoch
    metrics['gradient_estimator'] = model.poisson_gradient_estimator
    metrics['local_family'] = 'per-example per-group additive Poisson log-rate offsets'
    metrics['dataset'] = model_args.dataset
    return metrics


def print_summary(results, iw_samples):
    print('\nInference-gap diagnosis (nats/example)')
    header = ('label', 'estimator', 'NELBO(q)', 'NELBO(q*)', 'amort gap',
              'residual', 'NLL-%d' % iw_samples, 'ESS/K', 'diagnosis')
    print('{:<12} {:<16} {:>10} {:>11} {:>10} {:>10} {:>10} {:>8}  {}'.format(*header))
    for label, metrics in results.items():
        print('{:<12} {:<16} {:>10.4f} {:>11.4f} {:>10.4f} {:>10.4f} '
              '{:>10.4f} {:>8.4f}  {}'.format(
                  label, metrics['gradient_estimator'], metrics['amortized_nelbo'],
                  metrics['refined_nelbo'], metrics['amortization_gap'],
                  metrics['residual_gap'], metrics['refined_nll_iw'][str(iw_samples)],
                  metrics['refined_ess_fraction'], metrics['diagnosis']))


def main(args):
    positive_arguments = {
        '--batch_size': args.batch_size,
        '--max_batches': args.max_batches,
        '--local_steps': args.local_steps,
        '--local_mc_samples': args.local_mc_samples,
        '--iw_samples': args.iw_samples,
        '--iw_chunk_size': args.iw_chunk_size,
    }
    for name, value in positive_arguments.items():
        if value < 1:
            raise ValueError('%s must be positive.' % name)
    checkpoints = parse_checkpoints(args.checkpoints)
    iw_checkpoints = parse_iw_checkpoints(args.iw_checkpoints, args.iw_samples)
    device = torch.device(args.device)

    first_model, first_args, _ = load_model(next(iter(checkpoints.values())), device)
    batches = cache_validation_batches(
        first_args, args.data, args.batch_size, args.max_batches, args.seed)
    del first_model

    results = OrderedDict()
    for label, path in checkpoints.items():
        results[label] = diagnose_checkpoint(
            label, path, batches, args, iw_checkpoints, device)
    print_summary(results, args.iw_samples)

    payload = OrderedDict()
    payload['configuration'] = vars(args)
    payload['notes'] = [
        'All reported NELBO values use beta=1, no KL balancing, and exact Poisson evaluation samples.',
        'amortization_gap is a lower bound because local refinement uses additive log-rate offsets.',
        'residual_gap is only a posterior-family proxy when IW-NLL is stable and refined ESS/K is adequate.',
    ]
    payload['results'] = results
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(args.output, 'w') as output_file:
        json.dump(payload, output_file, indent=2)
    print('\nSaved diagnostics to %s' % args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Poisson NVAE inference-gap diagnostic')
    parser.add_argument('--checkpoints', nargs='+', required=True,
                        help='trained models as LABEL=PATH entries, e.g. st=/path/checkpoint.pt')
    parser.add_argument('--data', required=True, help='dataset root')
    parser.add_argument('--output', default='inference_gap_diagnostics.json')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_batches', type=int, default=5)
    parser.add_argument('--local_steps', type=int, default=100)
    parser.add_argument('--local_lr', type=float, default=0.05)
    parser.add_argument('--local_mc_samples', type=int, default=1)
    parser.add_argument('--local_grad_clip', type=float, default=10.)
    parser.add_argument('--iw_samples', type=int, default=1000)
    parser.add_argument('--iw_chunk_size', type=int, default=10)
    parser.add_argument('--iw_checkpoints', default='1,10,100,1000')
    parser.add_argument('--dominance_ratio', type=float, default=0.6)
    parser.add_argument('--ess_threshold', type=float, default=0.01)
    parser.add_argument('--iw_stability_tolerance', type=float, default=0.5,
                        help='maximum allowed last IW-NLL decrease in nats/example')
    parser.add_argument('--refinement_tolerance', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=1)
    main(parser.parse_args())
