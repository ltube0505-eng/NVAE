"""Multi-seed Poisson marginal, independence and conditional-sampling probes.

No checkpoint or distributed process group is used. Compatible with torch 1.6.
"""
import argparse
import json
import math
from pathlib import Path

import torch


class Sampler:
    def __init__(self, backend, seed):
        self.backend = backend
        self.seed_source = torch.Generator().manual_seed(seed + 100003)

    def __call__(self, rate):
        if self.backend == 'native':
            return torch.poisson(rate)
        if self.backend == 'cpu':
            cpu_dtype = torch.float64 if rate.dtype == torch.float64 else torch.float32
            return torch.poisson(rate.detach().to(device='cpu', dtype=cpu_dtype)).to(rate)
        if self.backend == 'fresh':
            # A diagnostic: separate seed for each call, drawn on CPU. It is
            # not a validated workaround until the target GPU passes probes.
            seed = int(torch.randint(0, 2**63 - 1, (1,), generator=self.seed_source).item())
            generator = torch.Generator(device=rate.device).manual_seed(seed)
            return torch.poisson(rate, generator=generator)
        raise ValueError('Unknown backend: %s' % self.backend)


class PairStats:
    def __init__(self):
        self.n = 0
        self.sx = self.sy = self.sxx = self.syy = self.sxy = self.equal = 0.

    def add(self, x, y):
        x, y = x.reshape(-1), y.reshape(-1)
        self.n += x.numel()
        self.sx += x.sum().item()
        self.sy += y.sum().item()
        self.sxx += x.square().sum().item()
        self.syy += y.square().sum().item()
        self.sxy += (x * y).sum().item()
        self.equal += (x == y).sum().item()

    def result(self):
        mx, my = self.sx / self.n, self.sy / self.n
        vx = max(self.sxx / self.n - mx * mx, 0.)
        vy = max(self.syy / self.n - my * my, 0.)
        denom = math.sqrt(vx * vy)
        return {'pairs': self.n, 'pearson': (self.sxy / self.n - mx * my) / denom
                if denom else None, 'equal_fraction': self.equal / self.n,
                'iid_correlation_se_approx': 1. / math.sqrt(self.n)}


def equality_probability(rate):
    p = math.exp(-rate)
    value = p * p
    for k in range(1, int(rate + 15 * math.sqrt(rate) + 50)):
        p *= rate / k
        value += p * p
    return value


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def marginal_probe(backend, device, shape, rate, calls, seed, dtype):
    seed_all(seed)
    sample = Sampler(backend, seed)
    lam = torch.full(shape, rate, device=device, dtype=dtype)
    coord = {lag: PairStats() for lag in (1, 16, 64, 320, 1280) if lag < lam.numel()}
    across_images, across_calls = PairStats(), PairStats()
    n = total = square = zero = 0
    previous = None
    call_means = []
    for _ in range(calls):
        z = sample(lam).detach().cpu().double()
        if not torch.isfinite(z).all() or (z < 0).any() or not torch.equal(z, z.round()):
            raise RuntimeError('Invalid count values.')
        flat = z.reshape(-1)
        n += z.numel()
        total += flat.sum().item()
        square += flat.square().sum().item()
        zero += (flat == 0).sum().item()
        call_means.append(z.mean().item())
        for lag, stats in coord.items():
            stats.add(flat[:-lag], flat[lag:])
        across_images.add(z[:-1], z[1:])
        if previous is not None:
            across_calls.add(previous, z)
        previous = z
    mean = total / n
    variance = square / n - mean * mean
    return {'kind': 'marginal', 'backend': backend, 'seed': seed, 'shape': list(shape),
            'rate': rate, 'dtype': str(dtype), 'draws': n, 'mean': mean,
            'variance': variance, 'fano': variance / mean if mean else None,
            'zero_fraction': zero / n, 'expected_zero_fraction': math.exp(-rate),
            'mean_iid_se': math.sqrt(rate / n),
            'expected_independent_equal_fraction': equality_probability(rate),
            'within_flat_lags': {str(k): v.result() for k, v in coord.items()},
            'across_images': across_images.result(), 'across_calls': across_calls.result(),
            'call_means': call_means}


def conditional_probe(backend, device, shape, calls, seed, dtype, upsample=False):
    seed_all(seed)
    sample = Sampler(backend, seed)
    prefix_rate = torch.full(shape, 10., device=device, dtype=dtype)
    residual_sum = variance_sum = count_sum = rate_sum = nimages = 0.
    block_means = []
    for _ in range(calls):
        prefix = sample(prefix_rate)
        if upsample:
            prefix = prefix.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        # Later rate is a deterministic function of an earlier random sample.
        qrate = torch.where(prefix > 10., torch.full_like(prefix, 28.),
                            torch.full_like(prefix, 0.1))
        z = sample(qrate)
        q, z = qrate.detach().cpu().double(), z.detach().cpu().double()
        weight = q.log()  # prior rate fixed at 1
        residual = ((z - q) * weight).flatten(1).sum(1)
        residual_sum += residual.sum().item()
        variance_sum += (q * weight.square()).sum().item()
        count_sum += (z - q).sum().item()
        rate_sum += q.sum().item()
        nimages += shape[0]
        block_means.append(residual.mean().item())
    mean = residual_sum / nimages
    iid_se = math.sqrt(variance_sum) / nimages
    # Independent-call approximation, useful when within-call coordinates are
    # correlated. Cross-call dependence can invalidate this estimate too.
    blocks = torch.tensor(block_means, dtype=torch.float64)
    block_se = blocks.std(unbiased=True).item() / math.sqrt(calls)
    return {'kind': 'conditional', 'backend': backend, 'seed': seed,
            'prefix_shape': list(shape), 'upsample': upsample, 'dtype': str(dtype),
            'images': int(nimages), 'weighted_residual_mean': mean,
            'conditional_iid_se': iid_se, 'iid_z_score': mean / iid_se if iid_se else None,
            'call_block_se': block_se, 'call_block_means': block_means,
            'count_residual_mean': count_sum / nimages,
            'count_conditional_iid_se': math.sqrt(rate_sum) / nimages}


def execute(options):
    device = torch.device(options.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not visible. Check CUDA_VISIBLE_DEVICES and nvidia-smi.')
    report = {'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
              'device': str(device), 'gpu_name': torch.cuda.get_device_name(device)
              if device.type == 'cuda' else None,
              'note': 'IID standard errors are reference values, not valid guarantees under dependence. '
                      'Inspect multiple seeds; many correlation tests entail multiple comparisons.',
              'results': [], 'errors': []}
    for backend in options.backends.split(','):
        try:
            for seed in options.seeds:
                for size in options.sizes:
                    shape = (options.batch_size, 20, size, size)
                    for rate in options.rates:
                        row = marginal_probe(backend, device, shape, rate, options.calls,
                                             seed, options.dtype)
                        report['results'].append(row)
                        print('marginal backend=%s seed=%d H=%d rate=%g mean=%.5f '
                              'Fano=%.5f call_corr=%s' %
                              (backend, seed, size, rate, row['mean'], row['fano'],
                               row['across_calls']['pearson']), flush=True)
                    for upsample in ([False, True] if size == 4 else [False]):
                        row = conditional_probe(backend, device, shape, options.calls,
                                                seed, options.dtype, upsample)
                        report['results'].append(row)
                        print('conditional backend=%s seed=%d H=%d up=%s residual=%.5f '
                              'iid_se=%.5f block_se=%.5f' %
                              (backend, seed, size, upsample, row['weighted_residual_mean'],
                               row['conditional_iid_se'], row['call_block_se']), flush=True)
        except (RuntimeError, TypeError) as error:
            # Fresh CUDA generators might not be supported by an old build.
            # Preserve partial results and make that explicit, without fallback.
            report['errors'].append({'backend': backend, 'error': str(error)})
            print('backend %s ERROR: %s' % (backend, error), flush=True)
    path = Path(options.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False))
    print('Saved full report:', path, flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--calls', type=int, default=32)
    parser.add_argument('--seeds', default='1,2,3')
    parser.add_argument('--rates', default='0.01,1,3,10,28,30')
    parser.add_argument('--sizes', default='4,8')
    parser.add_argument('--backends', default='native,cpu,fresh')
    parser.add_argument('--dtype', choices=['float32', 'float64'], default='float32')
    parser.add_argument('--out', required=True)
    options = parser.parse_args()
    options.seeds = [int(v) for v in options.seeds.split(',')]
    options.rates = [float(v) for v in options.rates.split(',')]
    options.sizes = [int(v) for v in options.sizes.split(',')]
    options.dtype = getattr(torch, options.dtype)
    if options.calls < 2 or options.batch_size < 2 or not options.seeds or not options.rates \
            or not options.sizes or any(v <= 0 or not math.isfinite(v) for v in options.rates) \
            or any(v < 1 for v in options.sizes) \
            or any(v not in ('native', 'cpu', 'fresh') for v in options.backends.split(',')):
        parser.error('Use calls/batch-size >= 2, positive finite rates/sizes, valid backends and nonempty lists.')
    execute(options)
