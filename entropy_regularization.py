"""Poisson posterior entropy and the L_alpha training schedule.

Compatible with Python 3.7 / PyTorch 1.6. No sampling cap is used to
normalize or truncate the probability distribution defining the entropy.
"""

import math

import torch
from torch.autograd.function import once_differentiable


def entropy_alpha(step, total_steps, initial=0.1, anneal_portion=0.5, start_portion=0.2):
    """Hold a, cosine-decay between start/end progress, then stay at zero.

    anneal_portion is the END progress, not the duration of the decay.
    """
    if not math.isfinite(initial) or not 0. <= initial < 1.:
        raise ValueError('poisson_entropy_alpha_init must be in [0, 1).')
    if not math.isfinite(anneal_portion) or not 0. < anneal_portion <= 1.:
        raise ValueError('poisson_entropy_anneal_portion must be in (0, 1].')
    if not math.isfinite(start_portion) or not 0. <= start_portion < anneal_portion:
        raise ValueError('poisson_entropy_anneal_start_portion must be in [0, anneal_portion).')
    if total_steps < 1 or step < 0:
        raise ValueError('total_steps must be positive and step nonnegative.')
    # Using total_steps - 1 guarantees alpha=0 on the last optimizer step,
    # including when anneal_portion=1. A one-step run uses alpha=0.
    end = anneal_portion * (total_steps - 1)
    start = start_portion * (total_steps - 1)
    if end <= 0. or step >= end or initial == 0.:
        return 0.
    if step <= start:
        return initial
    return initial * 0.5 * (1. + math.cos(math.pi * (step - start) / (end - start)))


class _PoissonEntropy(torch.autograd.Function):
    """Numerical entropy series with O(number of rates) saved backward state.

    First derivatives only. Summing the derivative alongside the entropy
    avoids retaining a support-by-latent autograd graph for every group.
    """

    @staticmethod
    def forward(ctx, rate, max_count):
        log_rate = rate.log()
        entropy = torch.zeros_like(rate)
        derivative = torch.zeros_like(rate)
        shape = (-1,) + (1,) * rate.dim()
        # Chunk the count dimension to bound temporary memory as well.
        for start in range(0, max_count + 1, 16):
            k = torch.arange(start, min(start + 16, max_count + 1),
                             device=rate.device, dtype=rate.dtype).view(shape)
            log_p = k * log_rate - rate - torch.lgamma(k + 1.)
            probability = log_p.exp()
            entropy -= (probability * log_p).sum(dim=0)
            derivative -= (probability * (log_p + 1.) *
                           (k / rate - 1.)).sum(dim=0)
        ctx.save_for_backward(derivative)
        return entropy

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        derivative, = ctx.saved_tensors
        return grad_output * derivative, None


def poisson_entropy(rate, rate_upper_bound):
    """Elementwise H(Pois(rate)) in nats, using a negligible-tail series.

    The caller guarantees 0 < rate <= rate_upper_bound. The support covers
    the configured rate bound plus 12 standard deviations and 32 counts;
    it is independent of the relaxed/ST sampler's max_count. Probabilities
    are NOT renormalized. Float16 inputs are promoted to float32.
    """
    if not math.isfinite(rate_upper_bound) or rate_upper_bound <= 0.:
        raise ValueError('rate_upper_bound must be finite and positive.')
    upper = int(math.ceil(rate_upper_bound +
                          12. * math.sqrt(rate_upper_bound + 1.) + 32.))
    work_rate = rate if rate.dtype == torch.float64 else rate.float()
    return _PoissonEntropy.apply(work_rate, upper)

