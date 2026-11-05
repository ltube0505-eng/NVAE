import math
from unittest.mock import patch

import pytest
import torch

from distributions import Poisson


def _raw_log_rate(rate):
    # Invert the retained smooth log-rate parameterization for test inputs.
    return 5. * math.atanh(math.log(rate) / 5.)


def test_rate_above_old_cap_has_nonzero_gradient_and_consistent_density():
    raw = torch.tensor([5., 8.], dtype=torch.float64, requires_grad=True)
    q = Poisson(raw)
    assert (q.rate > 30.).all()
    expected = torch.exp(5. * torch.tanh(raw / 5.))
    assert torch.allclose(q.rate, expected)
    counts = torch.tensor([40., 100.], dtype=torch.float64)
    assert torch.allclose(q.log_p(counts), torch.distributions.Poisson(q.rate).log_prob(counts))
    p = Poisson(torch.zeros_like(raw))
    assert torch.allclose(q.kl(p), torch.distributions.kl_divergence(
        torch.distributions.Poisson(q.rate), torch.distributions.Poisson(p.rate)))
    q.rate.sum().backward()
    assert (raw.grad > 0.).all()


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


def test_st_and_relaxed_share_samples_and_backward_path():
    raw = torch.tensor([0., 5., 8.], requires_grad=True)
    q = Poisson(raw)
    torch.manual_seed(42)
    hard, soft = q._arrival_counts()
    torch.manual_seed(42)
    st, _ = q.sample(estimator='straight_through')
    torch.manual_seed(42)
    relaxed, _ = q.sample(estimator='relaxed')
    assert torch.equal(st.detach(), hard)
    assert torch.equal(st.detach(), st.detach().round())
    assert torch.equal(relaxed, soft)
    st_grad = torch.autograd.grad(st.sum(), raw, retain_graph=True)[0]
    soft_grad = torch.autograd.grad(soft.sum(), raw)[0]
    assert torch.equal(st_grad, soft_grad)
    assert torch.isfinite(st_grad).all() and (st_grad > 0.).all()


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


def test_nonfinite_rate_fails_instead_of_looping():
    q = Poisson(torch.tensor([float('nan')]))
    with pytest.raises(ValueError, match='finite'):
        q.sample(estimator='straight_through')


def test_empty_rate_tensor():
    raw = torch.empty(0, requires_grad=True)
    q = Poisson(raw)
    z, _ = q.sample(estimator='straight_through')
    z.sum().backward()
    assert z.shape == raw.shape and q.last_cts_num_terms == 0
