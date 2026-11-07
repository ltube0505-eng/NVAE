"""Damped diagonal pullback of the Poisson output Fisher (Martens, 2020).

This is a surrogate output metric, not the full marginal-likelihood Fisher.
See docs/poisson_diagonal_fisher.md for the hierarchy and ST approximation.
"""
import math

import torch
import torch.distributed as dist
from torch.optim import Optimizer


class PoissonDiagonalFisher(Optimizer):
    def __init__(self, params, lr=1e-3, damping=1e-2, decay=0.95,
                 probes=1, step_budget=1e-2, weight_decay=0.):
        for name, value in [('lr', lr), ('damping', damping),
                            ('step_budget', step_budget), ('weight_decay', weight_decay)]:
            if not math.isfinite(value) or value < 0:
                raise ValueError('%s must be finite and nonnegative' % name)
        if damping == 0 or step_budget == 0:
            raise ValueError('damping and step_budget must be positive')
        if not 0 <= decay < 1 or not isinstance(probes, int) or probes < 1:
            raise ValueError('decay must be in [0, 1) and probes a positive integer')
        super().__init__(params, dict(lr=lr, damping=damping, decay=decay,
                                     probes=probes, step_budget=step_budget,
                                     weight_decay=weight_decay))
        self._pending = None
        self.last_stats = {}

    def estimate_curvature(self, log_rates, distributed=False):
        """Estimate diag(J^T diag(rate) J)/batch with independent sign probes.

        log_rates are the differentiable post-tanh log rates from this forward.
        Neither probes nor rate weights are differentiated. Squaring happens
        before rank averaging. This consumes no .grad buffers and retains the
        loss graph for the ordinary backward pass.
        """
        self._pending = None
        outputs = [value for value in log_rates if value.requires_grad]
        if not outputs:
            raise ValueError('No differentiable Poisson outputs were collected')
        if len(self.param_groups) != 1:
            raise ValueError('PoissonDiagonalFisher supports one parameter group')
        group = self.param_groups[0]
        params = [p for p in group['params'] if p.requires_grad]
        batch = outputs[0].shape[0]
        if batch == 0 or any(value.shape[0] != batch for value in outputs):
            raise ValueError('Poisson outputs must share a nonempty batch dimension')
        diagonal = [torch.zeros_like(p) for p in params]
        connected = [False] * len(params)
        # Isolate probes from the RNG stream used by the next CTS/data forward.
        devices = sorted({p.device.index for p in params if p.is_cuda})
        with torch.random.fork_rng(devices=devices):
            for _ in range(group['probes']):
                weights = []
                for output in outputs:
                    signs = torch.empty_like(output).bernoulli_(0.5).mul_(2).sub_(1)
                    weights.append(signs * torch.exp(0.5 * output.detach()) / math.sqrt(batch))
                gradients = torch.autograd.grad(outputs, params, grad_outputs=weights,
                                                retain_graph=True, allow_unused=True)
                for index, gradient in enumerate(gradients):
                    if gradient is not None:
                        connected[index] = True
                        diagonal[index].add_(gradient.detach().pow(2) / group['probes'])
        if distributed:
            # All ranks use the same graph topology and equal local batch size.
            for value in diagonal:
                dist.all_reduce(value)
                value.div_(dist.get_world_size())
        # Parameters outside the latent-rate graph (e.g. the image output head)
        # receive an identity metric, avoiding an unintended 1/damping gain.
        for index, is_connected in enumerate(connected):
            if not is_connected:
                diagonal[index].fill_(1.)
        if any(not torch.isfinite(value).all().item() for value in diagonal):
            raise FloatingPointError('Nonfinite Poisson Fisher estimate; no update applied')
        self._pending = list(zip(params, diagonal))

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('Closures are not supported')
        if self._pending is None:
            raise RuntimeError('Call estimate_curvature on the current forward before step')
        pending, self._pending = self._pending, None
        group = self.param_groups[0]
        # Validate everything before modifying parameters or the running metric.
        if any(p.grad is not None and not torch.isfinite(p.grad).all().item()
               for p, _ in pending):
            raise FloatingPointError('Nonfinite loss gradient; no update applied')
        updates = []
        quadratic = pending[0][0].new_zeros((), dtype=torch.float64)
        for param, estimate in pending:
            state = self.state[param]
            old = state.get('fisher')
            fisher = estimate if old is None else group['decay'] * old + (1 - group['decay']) * estimate
            if param.grad is None:
                updates.append((param, fisher, None))
                continue
            gradient = param.grad.detach() + group['weight_decay'] * param.detach()
            metric = fisher + group['damping']
            update = -group['lr'] * gradient / metric
            if not torch.isfinite(update).all().item():
                raise FloatingPointError('Nonfinite Fisher direction; no update applied')
            quadratic.add_(0.5 * (metric.double() * update.double().pow(2)).sum())
            updates.append((param, fisher, update))
        if not torch.isfinite(quadratic).item():
            raise FloatingPointError('Nonfinite quadratic step cost; no update applied')
        scale = min(1., math.sqrt(group['step_budget'] / max(quadratic.item(), 1e-30)))
        for param, fisher, update in updates:
            self.state[param]['fisher'] = fisher
            if update is not None:
                param.add_(update, alpha=scale)
        self.last_stats = dict(step_scale=scale, quadratic_step=quadratic.item() * scale ** 2)

    def load_state_dict(self, state_dict):
        if not state_dict.get('param_groups') or 'damping' not in state_dict['param_groups'][0]:
            raise ValueError('Cannot load Adamax state into PoissonDiagonalFisher; use --reset_optimizer')
        super().load_state_dict(state_dict)
        self._pending = None
