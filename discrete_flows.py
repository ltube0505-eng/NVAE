"""Posterior flows on the *unbounded* nonnegative integer lattice.

Each scalar map swaps adjacent counts. Strict autoregressive masks make the
gate independent of the count being changed, so inversion is sequential and
exact. The sigmoid is only a biased ST backward surrogate, never a PMF.
"""

import math

import torch
import torch.nn as nn

from neural_ar_operations import ARConv2d, ARInvertedResidual, ELUConv


def validate_counts(z):
    if not bool(torch.all(torch.isfinite(z) & (z >= 0) & (z == torch.floor(z)))):
        raise ValueError('Discrete Poisson flows require finite nonnegative integer counts.')


def adjacent_swap(z, gate, offset=0, validate=False):
    """Swap (offset, offset+1), (offset+2, offset+3), ...; fix z<offset.

    ``gate`` must be binary and independent of the value/parity being changed.
    Validation is optional to avoid a device synchronization in every group.
    """
    if offset not in (0, 1):
        raise ValueError('Pair offset must be 0 or 1.')
    if validate:
        validate_counts(z)
        if not bool(torch.all((gate == 0) | (gate == 1))):
            raise ValueError('Swap gates must be binary.')
    active = (z >= offset).to(z.dtype)
    direction = active * (1. - 2. * torch.remainder(z - offset, 2.))
    return z + gate.to(z.dtype) * direction


class PoissonSwapCellAR(nn.Module):
    """A triangular bijection with an encoder-conditioned binary gate.

    Spatial order is raster (reverse raster for mirror=True), then increasing
    channel index within each pixel. No BN/dropout is used in the conditioner.
    """

    def __init__(self, num_z, num_ftr, mirror=False, offset=0, temperature=1.):
        super(PoissonSwapCellAR, self).__init__()
        if offset not in (0, 1):
            raise ValueError('Pair offset must be 0 or 1.')
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Poisson flow temperature must be finite and positive.')
        self.mirror = mirror
        self.offset = offset
        self.temperature = temperature
        self.conv = ARInvertedResidual(num_z, num_ftr, ex=6, mirror=mirror)
        self.gate = ELUConv(self.conv.hidden_dim, num_z, kernel_size=1,
                           masked=True, zero_diag=False,
                           weight_init_coeff=0.1, mirror=mirror)
        # ftr is fixed given x and the preceding *groups*, so this unmasked
        # feature projection cannot violate the within-group triangular order.
        self.feature = ARConv2d(num_ftr, num_z, kernel_size=1, masked=False)
        with torch.no_grad():
            self.gate.conv_0.bias.fill_(-2.)
            self.feature.log_weight_norm.add_(math.log(0.1))

    def gate_logits(self, z, ftr):
        return self.gate(self.conv(torch.log1p(z), ftr)) + self.feature(ftr)

    def forward(self, z, ftr):
        # Keep the integer-valued result exact under mixed precision.
        if z.dtype in (torch.float16, torch.bfloat16):
            z = z.float()
        logits = self.gate_logits(z, ftr).to(z.dtype)
        hard_gate = (logits >= 0.).to(z.dtype)
        hard_z = adjacent_swap(z.detach(), hard_gate, self.offset)
        if self.training:
            soft_gate = torch.sigmoid(logits / self.temperature)
            # Freeze parity/boundaries in the backward surrogate. A continuous
            # interpolation here is not claimed to be an invertible flow.
            direction = adjacent_swap(z.detach(), torch.ones_like(z), self.offset) - z.detach()
            surrogate = z + soft_gate * direction
            new_z = hard_z + (surrogate - surrogate.detach())
        else:
            new_z = hard_z
        # Zero *discrete mass correction*, not a continuous Jacobian formula.
        return new_z, torch.zeros_like(new_z)

    @torch.no_grad()
    def inverse(self, z, ftr):
        """Reference exact inverse; sequential and intended for verification.

        Forward sampling/ELBO evaluation already knows the base sample and
        does not need this expensive inverse. Keep the same ftr and parameters.
        """
        validate_counts(z)
        base = torch.zeros_like(z)
        _, channels, height, width = z.shape
        rows = range(height - 1, -1, -1) if self.mirror else range(height)
        cols = range(width - 1, -1, -1) if self.mirror else range(width)
        for row in rows:
            for col in cols:
                for channel in range(channels):
                    logits = self.gate_logits(base, ftr)[:, channel, row, col]
                    gate = (logits >= 0.).to(z.dtype)
                    base[:, channel, row, col] = adjacent_swap(
                        z[:, channel, row, col], gate, self.offset)
        return base


class PairedPoissonSwapAR(nn.Module):
    """One num_nf block: even-pair raster and odd-pair reverse raster maps."""

    def __init__(self, num_z, num_ftr, temperature=1.):
        super(PairedPoissonSwapAR, self).__init__()
        self.cell1 = PoissonSwapCellAR(num_z, num_ftr, mirror=False,
                                      offset=0, temperature=temperature)
        self.cell2 = PoissonSwapCellAR(num_z, num_ftr, mirror=True,
                                      offset=1, temperature=temperature)

    def forward(self, z, ftr):
        z, correction1 = self.cell1(z, ftr)
        z, correction2 = self.cell2(z, ftr)
        return z, correction1 + correction2

    def inverse(self, z, ftr):
        return self.cell1.inverse(self.cell2.inverse(z, ftr), ftr)
