import ast
import math
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from distributions import Poisson


def raw_rate(rates, dtype=torch.float64):
    return torch.tensor([5. * math.atanh(math.log(r) / 5.) for r in rates],
                        dtype=dtype, requires_grad=True)


@pytest.mark.parametrize('estimator', ['straight_through', 'relaxed'])
def test_forward_and_rate_jacobian_match_fixed_arrival_formula(estimator):
    raw = raw_rate([0.01, 0.1, 0.35, 1., 30., 100.])
    q = Poisson(raw, smooth_ste=True)
    # Choose fixed base exponential draws, never differentiate their values.
    base = torch.tensor([0.02, 0.1, 0.3, 0.6, 1.2, 200.], dtype=raw.dtype)
    def arrivals(dist, shape):
        return base[:, None] / dist.rate
    with patch.object(q, '_cts_plan', return_value=(1.5, len(base))), \
            patch.object(torch.distributions.Exponential, 'rsample', arrivals):
        z, _ = q.sample(estimator=estimator)
    times = base.cumsum(0)[:, None] / q.rate.detach()
    sigmoid = torch.sigmoid((1. - times) / 0.1)
    original_g = (times / (0.1 * q.rate.detach()) * sigmoid * (1. - sigmoid)).sum(0)
    rate = q.rate.detach()
    w = rate.pow(4) / (rate.pow(4) + 0.35 ** 4)
    expected_g = 1. - w + w * original_g
    expected_forward = ((times <= 1.).to(raw.dtype).sum(0)
                        if estimator == 'straight_through' else sigmoid.sum(0))
    assert torch.allclose(z.detach(), expected_forward, rtol=1e-12, atol=1e-12)
    actual_g = torch.autograd.grad(z.sum(), q.rate, retain_graph=True)[0]
    assert torch.allclose(actual_g, expected_g, rtol=1e-11, atol=1e-12)
    raw_g = torch.autograd.grad(z.sum(), raw)[0]
    chain = rate * (1. - torch.tanh(raw.detach() / 5.).pow(2))
    assert torch.allclose(raw_g, expected_g * chain, rtol=1e-11, atol=1e-12)
    # In particular w(scale)=1/2, with no derivative of w entering backward.
    assert abs(w[2].item() - 0.5) < 1e-12


@pytest.mark.parametrize('estimator', ['straight_through', 'relaxed'])
def test_sample_paths_rng_and_variance_contraction(estimator):
    raw = raw_rate([0.1] * 4096)
    original = Poisson(raw)
    blended = Poisson(raw, smooth_ste=True)
    torch.manual_seed(123)
    z0, _ = original.sample(estimator=estimator)
    rng = torch.random.get_rng_state()
    g0 = torch.autograd.grad(z0.sum(), original.rate)[0]
    torch.manual_seed(123)
    z1, _ = blended.sample(estimator=estimator)
    g1 = torch.autograd.grad(z1.sum(), blended.rate)[0]
    assert torch.equal(z0.detach(), z1.detach())
    assert torch.equal(rng, torch.random.get_rng_state())
    w = 0.1 ** 4 / (0.1 ** 4 + 0.35 ** 4)
    assert torch.allclose(g1, 1. - w + w * g0)
    assert torch.allclose(g1.var(), w ** 2 * g0.var())
    assert abs(g1.mean().item() - (1. - w + w * g0.mean().item())) < 1e-12


def test_default_backward_and_density_kl_are_unchanged():
    raw = raw_rate([0.01, 0.35, 1., 100.])
    q = Poisson(raw)
    assert not q.smooth_ste
    torch.manual_seed(7)
    hard, soft = q._arrival_counts()
    torch.manual_seed(7)
    z, _ = q.sample(estimator='straight_through')
    assert torch.equal(z.detach(), hard)
    assert torch.equal(torch.autograd.grad(z.sum(), q.rate)[0],
                       torch.autograd.grad(soft.sum(), q.rate)[0])
    blend = Poisson(raw, smooth_ste=True)
    prior = Poisson(torch.zeros_like(raw))
    assert torch.equal(q.log_p(hard), blend.log_p(hard))
    assert torch.equal(q.kl(prior), blend.kl(prior))
    kl_g = torch.autograd.grad(blend.kl(prior).sum(), blend.rate)[0]
    assert torch.allclose(kl_g, (blend.rate / prior.rate).log())


@pytest.mark.parametrize('estimator', ['exact', 'reinforce', 'obbvi'])
def test_exact_and_score_samples_are_unchanged(estimator):
    raw = raw_rate([0.01, 0.35, 1., 100.])
    torch.manual_seed(19)
    z0, _ = Poisson(raw).sample(estimator=estimator, proposal_tau=3.)
    torch.manual_seed(19)
    z1, _ = Poisson(raw, smooth_ste=True).sample(estimator=estimator, proposal_tau=3.)
    assert torch.equal(z0, z1)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_weight_is_finite_at_large_rates_in_low_precision(dtype):
    q = Poisson(raw_rate([0.01, 0.35, 100.], dtype=dtype), smooth_ste=True)
    surrogate = q._smooth_ste_surrogate(2. * q.rate)
    g = torch.autograd.grad(surrogate.sum(), q.rate)[0]
    assert surrogate.dtype == dtype
    assert torch.isfinite(surrogate).all() and torch.isfinite(g).all()
    assert g[0] < 1.01 and g[-1] > 1.99


@pytest.mark.parametrize('name', ['ste_rate_scale', 'ste_power'])
@pytest.mark.parametrize('value', [0., -1., float('nan'), float('inf')])
def test_invalid_blend_parameters(name, value):
    with pytest.raises(ValueError, match=name):
        Poisson(torch.zeros(1), **{name: value})


def test_empty_tensor_with_blending():
    raw = torch.empty(0, requires_grad=True)
    z, _ = Poisson(raw, smooth_ste=True).sample(estimator='straight_through')
    z.sum().backward()
    assert raw.grad.shape == raw.shape


def test_hierarchical_fisher_uses_blended_ancestor_jacobian():
    from poisson_fisher import PoissonDiagonalFisher
    raw = torch.nn.Parameter(raw_rate([0.1]).detach().reshape(1, 1))
    q = Poisson(raw, smooth_ste=True)
    # A far-future first arrival makes the original path derivative vanish.
    soft = q.rate * 0.
    with patch.object(q, '_arrival_counts', return_value=(torch.zeros_like(raw), soft)):
        z, _ = q.sample(estimator='straight_through')
    child = Poisson(0.7 * z + 0.2)
    opt = PoissonDiagonalFisher([raw])
    opt.estimate_curvature([child.log_rate])
    rate = q.rate.detach()
    w = rate.pow(4) / (rate.pow(4) + 0.35 ** 4)
    parent_chain = rate * (1. - torch.tanh(raw.detach() / 5.).pow(2))
    child_chain = 0.7 * (1. - torch.tanh((0.7 * z.detach() + 0.2) / 5.).pow(2))
    expected = child.rate.detach() * (child_chain * (1. - w) * parent_chain).pow(2)
    assert torch.allclose(opt._pending[0][1], expected, rtol=1e-12, atol=1e-12)
    assert expected.item() > 0.


def test_cli_defaults_and_model_config_checkpoint_compatibility():
    import argparse
    import utils
    from model import AutoEncoder
    from test_mixed_model import _args

    # Execute the actual CLI definitions without GPU/dataset startup imports.
    tree = ast.parse(Path('train.py').read_text())
    parser = argparse.ArgumentParser()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and
                isinstance(node.value.func, ast.Attribute) and
                isinstance(node.value.func.value, ast.Name) and
                node.value.func.value.id == 'parser' and
                node.value.func.attr == 'add_argument' and node.value.args and
                isinstance(node.value.args[0], ast.Constant) and
                node.value.args[0].value.startswith('--poisson_')):
            exec(compile(ast.Module(body=[node], type_ignores=[]), 'train.py', 'exec'),
                 {'parser': parser})
    assert parser.parse_args([]).poisson_smooth_ste is False
    config = parser.parse_args(['--poisson_smooth_ste', '--poisson_ste_rate_scale', '0.5',
                               '--poisson_ste_power', '2'])
    args = _args()
    legacy = AutoEncoder(args, None, utils.get_arch_cells('res_elu'))
    for key, value in vars(config).items():
        setattr(args, key, value)
    model = AutoEncoder(args, None, utils.get_arch_cells('res_elu'))
    model.load_state_dict(legacy.state_dict(), strict=True)
    assert not legacy.poisson_smooth_ste
    param = torch.zeros(1, 4, 2, 2)
    for dist in [model._latent_from_param(0, param), model._fixed_prior(0, param[:, :2])]:
        assert dist.smooth_ste and dist.ste_rate_scale == 0.5 and dist.ste_power == 2.


@pytest.mark.parametrize('kind,estimator', [('normal', 'straight_through'),
                                         ('poisson', 'reinforce'), ('poisson', 'obbvi')])
def test_model_rejects_unused_blending(kind, estimator):
    import utils
    from model import AutoEncoder
    from test_mixed_model import _args
    args = _args()
    args.latent_distribution = kind
    args.poisson_gradient_estimator = estimator
    args.poisson_smooth_ste = True
    with pytest.raises(ValueError, match='poisson_smooth_ste'):
        AutoEncoder(args, None, utils.get_arch_cells('res_elu'))
