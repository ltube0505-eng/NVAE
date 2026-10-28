"""Exact Poisson forward samples and rate-STE through the whole hierarchy."""
import argparse
import ast
from pathlib import Path
from unittest import mock

import pytest
import torch

from distributions import Poisson
from neural_operations import ScaleBoundaryCell
from test_scale_last_sample import make_model


def training_defaults():
    # Exercise the actual CLI parser without importing dataset/FID dependencies
    # or starting CUDA/distributed training from train.py's main block.
    source = Path(__file__).resolve().parents[1] / 'train.py'
    tree = ast.parse(source.read_text())
    main = tree.body[-1]
    statements = []
    for statement in main.body:
        if isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == 'args'
                for target in statement.targets):
            break
        statements.append(statement)
    namespace = {'argparse': argparse}
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['parser'].parse_args([])


def test_new_training_defaults_select_all_poisson_ste_bridge():
    args = training_defaults()
    assert args.latent_distribution == 'poisson'
    assert args.poisson_gradient_estimator == 'straight_through'
    assert args.scale_connection == 'last_sample'
    assert args.num_nf == 0


def test_ste_exact_forward_and_weighted_rate_gradient():
    raw_log_rate = torch.tensor([-1.5, -0.2, 0.8, 1.7], requires_grad=True)
    dist = Poisson(raw_log_rate, max_count=1)
    torch.manual_seed(71)
    expected = torch.poisson(dist.rate.detach())
    torch.manual_seed(71)
    with mock.patch.object(dist, '_arrival_counts', side_effect=AssertionError('arrival path')):
        z, _ = dist.sample(estimator='straight_through')
    assert torch.equal(z.detach(), expected)
    assert torch.equal(z.detach(), z.detach().round())
    assert (z >= 0).all()
    weights = torch.tensor([1.0, -2.0, 0.5, 3.0])
    rate_grad, raw_grad = torch.autograd.grad((z * weights).sum(), (dist.rate, raw_log_rate),
                                             retain_graph=True)
    expected_raw, = torch.autograd.grad((dist.rate * weights).sum(), raw_log_rate)
    assert torch.equal(rate_grad, weights)
    assert torch.allclose(raw_grad, expected_raw)
    assert torch.isfinite(raw_grad).all()


def test_ste_does_not_truncate_counts_or_use_relaxation_settings():
    raw = torch.full((4096,), 3.)
    small = Poisson(raw, max_count=1, relaxation_temperature=0.01)
    large = Poisson(raw, max_count=128, relaxation_temperature=2.)
    torch.manual_seed(73)
    a, _ = small.sample(estimator='straight_through')
    torch.manual_seed(73)
    b, _ = large.sample(estimator='straight_through')
    assert torch.equal(a, b)
    assert (a > small.max_count).any()
    assert abs(a.mean().item() - small.rate[0].item()) < 0.5


@pytest.mark.parametrize('adaptive', [False, True])
def test_every_group_is_poisson_and_receives_reconstruction_gradient(adaptive):
    torch.manual_seed(79)
    defaults = training_defaults()
    model = make_model(latent_distribution=defaults.latent_distribution,
                       poisson_gradient_estimator=defaults.poisson_gradient_estimator,
                       scale_connection=defaults.scale_connection,
                       num_latent_scales=3, num_groups_per_scale=4,
                       ada_groups=adaptive).train()
    samples, boundaries = [], []
    original = model._sample_latent

    def record(dist, training_sample=False):
        assert isinstance(dist, Poisson)
        assert training_sample
        z, auxiliary = original(dist, training_sample)
        z.retain_grad()
        samples.append(z)
        return z, auxiliary

    def record_boundary(module, inputs):
        boundaries.append(inputs[0])

    handles = [m.register_forward_pre_hook(record_boundary) for m in model.modules()
               if isinstance(m, ScaleBoundaryCell)]
    try:
        with mock.patch.object(model, '_sample_latent', side_effect=record), \
                mock.patch.object(model, '_latent_from_param', wraps=model._latent_from_param) as heads, \
                mock.patch.object(model, '_fixed_prior', wraps=model._fixed_prior) as fixed:
            logits, log_q, log_p, kl_all, _ = model(torch.rand(2, 1, 32, 32))
        total = sum(model.groups_per_scale)
        assert len(samples) == len(kl_all) == total
        assert heads.call_count == 2 * total - 1  # q for all groups, p for all but z0
        assert fixed.call_count == 1
        assert all(model._latent_kind(i) == 'poisson' for i in range(total))
        assert torch.isfinite(log_q).all() and torch.isfinite(log_p).all()
        assert all(torch.isfinite(kl).all() for kl in kl_all)
        assert all(torch.equal(z.detach(), z.detach().round()) for z in samples)
        # No KL term: gradients must actually traverse the discrete STE samples.
        logits.square().mean().backward()
        assert len(boundaries) == 2
        for z in samples + boundaries:
            assert z.grad is not None and torch.isfinite(z.grad).all()
            assert z.grad.abs().sum() > 0
        for head in model.enc_sampler:
            grads = [p.grad for p in head.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            assert sum(g.abs().sum() for g in grads) > 0
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize('prior', [False, True])
def test_eval_and_prior_sampling_are_exact_poisson_for_every_group(prior):
    model = make_model(latent_distribution='poisson').eval()
    original = Poisson.sample
    calls = []

    def record(dist, *args, **kwargs):
        assert kwargs['estimator'] == 'exact'
        z, auxiliary = original(dist, *args, **kwargs)
        calls.append(z)
        return z, auxiliary

    with mock.patch.object(Poisson, 'sample', record), torch.no_grad():
        logits = model.sample(2, 1.) if prior else model(torch.rand(2, 1, 32, 32))[0]
    assert len(calls) == sum(model.groups_per_scale)
    assert all(torch.equal(z, z.round()) and (z >= 0).all() for z in calls)
    assert logits.shape == (2, 1, 32, 32)
    assert torch.isfinite(logits).all()
