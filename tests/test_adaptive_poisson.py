import math
from unittest.mock import patch

import pytest
import torch

from distributions import Poisson


def _raw_log_rate(rate):
    return math.log(rate)


def test_rate_above_old_cap_has_nonzero_gradient_and_consistent_density():
    raw = torch.tensor([5., 8.], dtype=torch.float64, requires_grad=True)
    q = Poisson(raw)
    assert (q.rate > 30.).all()
    expected = torch.exp(raw)
    assert torch.allclose(q.rate, expected)
    counts = torch.tensor([40., 100.], dtype=torch.float64)
    assert torch.allclose(q.log_p(counts), torch.distributions.Poisson(q.rate).log_prob(counts))
    p = Poisson(torch.zeros_like(raw))
    assert torch.allclose(q.kl(p), torch.distributions.kl_divergence(
        torch.distributions.Poisson(q.rate), torch.distributions.Poisson(p.rate)))
    q.rate.sum().backward()
    assert torch.allclose(raw.grad, expected)


def test_legacy_caps_are_ignored():
    raw = torch.tensor([_raw_log_rate(100.)])
    with pytest.warns(DeprecationWarning, match='ignored'):
        q = Poisson(raw, max_count=1, max_rate=1.)
    assert q.rate.item() > 99.
    torch.manual_seed(90)
    z, _ = q.sample(estimator='straight_through')
    assert z.item() > 64.


def test_plan_adapts_to_rate_temperature_and_tolerance():
    def plan(rate, temperature=0.1, tolerance=1e-4):
        return Poisson(torch.tensor([_raw_log_rate(rate)], dtype=torch.float64),
                       relaxation_temperature=temperature,
                       cts_tail_tolerance=tolerance)._cts_plan()

    assert plan(1.)[1] < 64 < plan(100.)[1]
    assert plan(30., 0.1)[1] < plan(30., 1.)[1]
    assert plan(30., tolerance=1e-3)[1] < plan(30., tolerance=1e-6)[1]
    for rate in (0.01, 1., 30., 100., 140.):
        for temperature in (0.05, 0.1, 1.):
            horizon, _ = plan(rate, temperature)
            expected_tail = rate * temperature * math.log1p(
                math.exp((1. - horizon) / temperature))
            assert horizon >= 1.
            assert expected_tail <= 1e-4 * (1. + 1e-12)


def test_insufficient_budget_extends_same_arrivals_and_keeps_gradients():
    raw = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    q = Poisson(raw)

    def deterministic_arrivals(dist, sample_shape):
        return torch.full(sample_shape + dist.rate.shape, 0.25,
                          dtype=dist.rate.dtype) / dist.rate

    # Force four chunks, including arrivals exactly at t=1 and H=1.5.
    with patch.object(q, '_cts_plan', return_value=(1.5, 2)), \
            patch.object(torch.distributions.Exponential, 'rsample', deterministic_arrivals):
        hard, soft = q._arrival_counts()
    assert q.last_cts_num_terms == 8
    assert hard.item() == 4.
    arrivals = torch.arange(1., 9., dtype=raw.dtype).unsqueeze(1) * 0.25 / q.rate
    expected = torch.sigmoid((1. - arrivals) / 0.1).sum(dim=0)
    assert torch.allclose(soft, expected)
    actual_grad = torch.autograd.grad(soft.sum(), raw, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.sum(), raw)[0]
    assert torch.allclose(actual_grad, expected_grad)
    assert (actual_grad > 0.).all()


@pytest.mark.parametrize('legacy_alias', [False, True])
def test_st_exact_forward_identity_rate_gradient_and_raw_chain_rule(legacy_alias):
    raw = torch.tensor([-12., -1., 0., 5., 8.], dtype=torch.float64, requires_grad=True)
    q = Poisson(raw)
    torch.manual_seed(42)
    expected = torch.poisson(q.rate.detach())
    torch.manual_seed(42)
    with patch.object(q, '_arrival_counts', side_effect=AssertionError('STE must bypass CTS')):
        st, _ = q.sample(relaxed=True) if legacy_alias else q.sample(estimator='straight_through')
    assert torch.equal(st.detach(), expected)
    assert torch.equal(st.detach(), st.detach().round())
    # A nonconstant upstream gradient checks the full vector-Jacobian product.
    weights = torch.tensor([2., -3., 0.5, 0., 4.], dtype=raw.dtype)
    loss = (weights * st).sum()
    rate_grad = torch.autograd.grad(loss, q.rate, retain_graph=True)[0]
    raw_grad = torch.autograd.grad(loss, raw)[0]
    assert torch.equal(rate_grad, weights)
    assert torch.allclose(raw_grad, weights * q.rate.detach())
    assert q.last_cts_num_terms is None and q.last_cts_horizon is None


def test_st_is_independent_of_cts_settings():
    raw = torch.tensor([-2., 0., 4.], requires_grad=True)
    results = []
    for temperature, tolerance in [(0.05, 1e-2), (1., 1e-6)]:
        q = Poisson(raw, relaxation_temperature=temperature, cts_tail_tolerance=tolerance)
        torch.manual_seed(17)
        z, _ = q.sample(estimator='straight_through')
        results.append((z.detach(), torch.autograd.grad(z.sum(), raw)[0]))
    assert all(torch.equal(a, b) for a, b in zip(*results))


def test_relaxed_still_uses_cts():
    raw = torch.tensor([-1., 0., 3.], requires_grad=True)
    q = Poisson(raw)
    torch.manual_seed(42)
    _, soft = q._arrival_counts()
    torch.manual_seed(42)
    relaxed, _ = q.sample(estimator='relaxed')
    assert torch.equal(relaxed, soft)
    actual = torch.autograd.grad(relaxed.sum(), raw, retain_graph=True)[0]
    expected = torch.autograd.grad(soft.sum(), raw)[0]
    assert torch.equal(actual, expected)
    assert torch.isfinite(actual).all() and (actual > 0.).all()


@pytest.mark.parametrize('estimator', ['straight_through', 'exact', 'reinforce'])
def test_sampling_modes_match_poisson_moments_without_caps(estimator):
    torch.manual_seed(2718)
    n = 12000
    for rate in (0.1, 1., 30., 80., 200.):
        q = Poisson(torch.full((n,), math.log(rate)))
        z, _ = q.sample(estimator=estimator)
        actual_rate = q.rate[0].item()
        assert torch.equal(z, z.round())
        assert abs(z.mean().item() - actual_rate) < 6. * math.sqrt(actual_rate / n)
        assert abs(z.var().item() - actual_rate) < 0.15 * actual_rate
        if rate >= 80.:
            assert z.mean().item() > 64.


def test_hard_counts_match_poisson_moments_above_64():
    torch.manual_seed(2718)
    n = 12000
    for rate in (0.1, 1., 30., 80., 140.):
        q = Poisson(torch.full((n,), _raw_log_rate(rate)))
        hard, _ = q._arrival_counts()
        actual_rate = q.rate[0].item()
        assert abs(hard.mean().item() - actual_rate) < 6. * math.sqrt(actual_rate / n)
        assert abs(hard.var().item() - actual_rate) < 0.15 * actual_rate
        if rate >= 80.:
            assert hard.mean().item() > 64.


@pytest.mark.parametrize('tolerance', [0., -1., 1., float('nan'), float('inf')])
def test_invalid_tail_tolerance(tolerance):
    with pytest.raises(ValueError, match='cts_tail_tolerance'):
        Poisson(torch.zeros(1), cts_tail_tolerance=tolerance)


@pytest.mark.parametrize('temperature', [0., -1., float('nan'), float('inf')])
def test_invalid_temperature(temperature):
    with pytest.raises(ValueError, match='relaxation_temperature'):
        Poisson(torch.zeros(1), relaxation_temperature=temperature)


@pytest.mark.parametrize('raw', [float('nan'), float('inf'), -float('inf'), 100., -110.])
def test_invalid_rate_fails_without_silent_clamping(raw):
    with pytest.raises(ValueError, match='finite and positive'):
        Poisson(torch.tensor([raw], dtype=torch.float32))


def test_empty_rate_tensor():
    raw = torch.empty(0, requires_grad=True)
    q = Poisson(raw)
    z, _ = q.sample(estimator='straight_through')
    z.sum().backward()
    assert z.shape == raw.shape and q.last_cts_num_terms is None
