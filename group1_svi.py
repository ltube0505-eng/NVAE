"""SA-VAE momentum refinement with replayed finite-difference reverse mode.

Algorithm reference: harvardnlp/sa-vae, optim_n2n.py (forward_mom/backward_mom).
This implementation recomputes base gradients instead of storing n copies of
all model gradients. Only the local variational tensor is updated in forward.
"""
from contextlib import contextmanager

import torch
from torch.nn.modules.batchnorm import _BatchNorm


def batchnorm_snapshot(model):
    """Freeze inner normalization across samples, steps, and backward replays."""
    snapshot = []
    for module in model.modules():
        if isinstance(module, _BatchNorm):
            if module.running_mean is None or module.running_var is None:
                raise ValueError('Per-sample SVI requires BatchNorm running statistics.')
            snapshot.append((module, module.running_mean.detach().clone(),
                             module.running_var.detach().clone()))
    return tuple(snapshot)


@contextmanager
def fixed_batchnorm(snapshot):
    """Use constant running stats, while retaining affine/input gradients."""
    previous = [(m, m.training, m.running_mean, m.running_var)
                for m, _, _ in snapshot]
    try:
        for module, mean, var in snapshot:
            module.training = False
            module.running_mean = mean
            module.running_var = var
        yield
    finally:
        for module, training, mean, var in previous:
            module.training = training
            module.running_mean = mean
            module.running_var = var


def _rng_state(device):
    return (torch.get_rng_state(), torch.cuda.get_rng_state(device)
            if device.type == 'cuda' else None)


@contextmanager
def replay_rng(state, device):
    devices = [device.index] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.set_rng_state(state[0])
        if state[1] is not None:
            torch.cuda.set_rng_state(state[1], device)
        yield


@contextmanager
def temporary_model_state(model):
    """Inner passes/replays must not advance BN or replace outer graph caches."""
    buffers = [(m, name, value) for m in model.modules()
               for name, value in m._buffers.items() if value is not None]
    weights = [(m, m.weight_normalized) for m in model.modules()
               if hasattr(m, 'weight_normalized')]
    gradient_context = model._gradient_context
    try:
        for module, name, value in buffers:
            module._buffers[name] = value.clone()
        yield
    finally:
        for module, name, value in buffers:
            module._buffers[name] = value
        for module, weight in weights:
            module.weight_normalized = weight
        model._gradient_context = gradient_context


def _grads(objective, value, features, parameters, all_inputs):
    local = value.detach().requires_grad_(True)
    features = tuple(f.detach().requires_grad_(True) for f in features)
    targets = (local,) + features + parameters if all_inputs else (local,)
    with torch.enable_grad():
        loss = objective(local, features)
        grads = torch.autograd.grad(loss, targets, allow_unused=True)
    return tuple(torch.zeros_like(t) if g is None else g.detach()
                 for t, g in zip(targets, grads))


class _MomentumRefinement(torch.autograd.Function):
    @staticmethod
    def forward(ctx, objective, steps, lr, momentum, eps, feature_count, initial, *inputs):
        ctx.objective = objective
        ctx.lr, ctx.momentum, ctx.eps = lr, momentum, eps
        ctx.feature_count = feature_count
        ctx.save_for_backward(*inputs)
        features, parameters = inputs[:feature_count], inputs[feature_count:]
        value = initial.detach()
        velocity = torch.zeros_like(value)
        ctx.trajectory = []
        for _ in range(steps):
            state = _rng_state(value.device)
            ctx.trajectory.append((value, state))
            grad, = _grads(objective, value, features, parameters, False)
            velocity = momentum * velocity - grad
            value = value + lr * velocity
            if not torch.isfinite(value).all():
                raise FloatingPointError('Non-finite group1 SVI update; reduce --group1_svi_lr.')
        return value

    @staticmethod
    def backward(ctx, grad_output):
        inputs = ctx.saved_tensors
        features = inputs[:ctx.feature_count]
        parameters = inputs[ctx.feature_count:]
        value_grad = grad_output
        velocity_grad = torch.zeros_like(value_grad)
        indirect_grads = [torch.zeros_like(t) for t in inputs]
        # Same adjoint recursion and one-sided finite difference as OptimN2N.
        for value, state in reversed(ctx.trajectory):
            velocity_grad = velocity_grad + ctx.lr * value_grad
            with replay_rng(state, value.device):
                base = _grads(ctx.objective, value, features, parameters, True)
            with replay_rng(state, value.device):
                shifted = _grads(ctx.objective, value + ctx.eps * velocity_grad,
                                 features, parameters, True)
            hvp = [(b - a) / ctx.eps for a, b in zip(base, shifted)]
            value_grad = value_grad - hvp[0]
            for i, mixed in enumerate(hvp[1:]):
                indirect_grads[i] = indirect_grads[i] - mixed
            velocity_grad = ctx.momentum * velocity_grad
        return (None,) * 6 + (value_grad,) + tuple(indirect_grads)


def refine_momentum(objective, initial, features, parameters, steps, lr,
                    momentum=0.5, eps=1e-5):
    if steps < 0 or lr <= 0 or not 0 <= momentum < 1 or eps <= 0:
        raise ValueError('Require steps >= 0, lr > 0, 0 <= momentum < 1, eps > 0.')
    if steps == 0:
        return initial
    return _MomentumRefinement.apply(objective, steps, lr, momentum, eps,
                                     len(features), initial, *features, *parameters)
