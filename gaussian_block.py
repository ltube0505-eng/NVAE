"""Single-layer Gaussian parameter upscale and exact 2x2 block selection."""

import torch
from torch import nn
from torch.distributions import Categorical

from distributions import Normal


def zero_insert_2x(x):
    """Double each spatial axis; the original entries occupy even coordinates."""
    out = x.new_zeros(x.size(0), x.size(1), 2 * x.size(2), 2 * x.size(3))
    out[:, :, ::2, ::2] = x
    return out


def blockify(x):
    """[B,C,2H,2W] -> [B,C,H,W,4], ordered 00, 01, 10, 11."""
    b, c, hh, ww = x.shape
    if hh % 2 or ww % 2:
        raise ValueError('Block selection requires even spatial dimensions.')
    return x.reshape(b, c, hh // 2, 2, ww // 2, 2).permute(
        0, 1, 2, 4, 3, 5).reshape(b, c, hh // 2, ww // 2, 4)


class ParameterUpscale(nn.Module):
    """Two C-channel Gaussian fields, plus one optional shared logit field."""

    def __init__(self, channels, selector=False):
        super(ParameterUpscale, self).__init__()
        self.mean = nn.ConvTranspose2d(channels, channels, 3, stride=1, padding=1)
        self.log_sigma = nn.ConvTranspose2d(channels, channels, 3, stride=1, padding=1)
        self.selector = nn.ConvTranspose2d(2 * channels, 1, 3, stride=1, padding=1) \
            if selector else None
        # A zero coarse prior starts with the same Normal convention as NVAE.
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.log_sigma.bias)
        if self.selector is not None:
            nn.init.zeros_(self.selector.bias)

    def forward(self, param):
        up = zero_insert_2x(param)
        mean, log_sigma = torch.chunk(up, 2, dim=1)
        return (self.mean(mean).float(), self.log_sigma(log_sigma).float(),
                None if self.selector is None else self.selector(up).float())


class BlockGaussian:
    """A mixture of four C-dimensional diagonal Gaussians per spatial block.

    One categorical index is shared by all C channels in a block. q and p use
    the SAME prior logits. ``kl`` deliberately returns the requested unweighted
    component sum, which upper-bounds their marginal-mixture KL.
    """

    def __init__(self, mean, log_sigma, logits, temp=1.):
        self.components = Normal(blockify(mean.float()), blockify(log_sigma.float()), temp=temp)
        self.log_weights = torch.log_softmax(blockify(logits.float()), dim=-1)

    def sample(self):
        indices = Categorical(logits=self.log_weights.squeeze(1)).sample()
        index = indices[:, None, :, :, None].expand(
            -1, self.components.mu.size(1), -1, -1, 1)
        # Select the distribution parameters BEFORE the Gaussian draw.
        mean = self.components.mu.gather(-1, index).squeeze(-1)
        sigma = self.components.sigma.gather(-1, index).squeeze(-1)
        z = mean + sigma * torch.randn_like(mean)
        log_selection = self.log_weights.squeeze(1).gather(
            -1, indices.unsqueeze(-1)).squeeze(-1).sum(dim=[1, 2])
        return z, log_selection

    def log_p(self, z):
        # Shared index: sum over C INSIDE logsumexp, then sum spatial blocks
        # using the existing model interface. This is the marginal density.
        component_log_p = self.components.log_p(z.float().unsqueeze(-1)).sum(dim=1, keepdim=True)
        return torch.logsumexp(self.log_weights + component_log_p, dim=-1)

    def kl(self, other):
        return self.components.kl(other.components).sum(dim=-1)


class GaussianBlockLatent(nn.Module):
    def __init__(self, z_size, baseline_decay=0.9):
        super(GaussianBlockLatent, self).__init__()
        c, h, w = z_size
        self.posterior_upscale = ParameterUpscale(c)
        self.prior_upscale = ParameterUpscale(c, selector=True)
        self.prior_param = nn.Parameter(torch.zeros(1, 2 * c, h, w))
        self.baseline_decay = baseline_decay
        self.register_buffer('reconstruction_baseline', torch.zeros(()))
        self.register_buffer('baseline_ready', torch.tensor(False))

    def prior(self, batch_size, temp=1.):
        if temp <= 0:
            raise ValueError('Gaussian sampling temperature must be positive.')
        mean, log_sigma, logits = self.prior_upscale(self.prior_param)
        return BlockGaussian(mean.expand(batch_size, -1, -1, -1),
                             log_sigma.expand(batch_size, -1, -1, -1),
                             logits.expand(batch_size, -1, -1, -1), temp=temp)

    def forward(self, posterior_param):
        prior_mean, prior_log_sigma, logits = self.prior_upscale(self.prior_param)
        batch_size = posterior_param.size(0)
        logits = logits.expand(batch_size, -1, -1, -1)
        mean, log_sigma, _ = self.posterior_upscale(posterior_param)
        posterior = BlockGaussian(mean, log_sigma, logits)
        prior = BlockGaussian(prior_mean.expand_as(mean), prior_log_sigma.expand_as(log_sigma), logits)
        return posterior, prior

    def selector_correction(self, reconstruction, log_selection):
        """Zero-value REINFORCE term for the exact categorical draw.

        The baseline uses ONLY previous minibatches, avoiding same-batch bias.
        Gaussian parameters and decoder retain their ordinary pathwise gradient.
        The unweighted component KL has no dependence on the categorical draw.
        """
        baseline = self.reconstruction_baseline.detach().clone()
        score = (reconstruction.detach().float() - baseline) * log_selection
        with torch.no_grad():
            current = reconstruction.detach().float().mean()
            if bool(self.baseline_ready.item()):
                self.reconstruction_baseline.mul_(self.baseline_decay).add_(
                    current * (1. - self.baseline_decay))
            else:
                self.reconstruction_baseline.copy_(current)
                self.baseline_ready.fill_(True)
        return score - score.detach()
