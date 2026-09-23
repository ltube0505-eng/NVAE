import torch

from distributions import Gamma, Poisson


def test_relaxed_poisson_has_integer_forward_values_and_finite_gradients():
    torch.manual_seed(7)
    log_rate_q = torch.zeros(2, 3, 4, 4, requires_grad=True)
    log_rate_p = torch.full_like(log_rate_q, -0.2)
    q = Poisson(log_rate_q, relaxation_temperature=0.2, max_count=32, max_rate=10.)
    p = Poisson(log_rate_p, relaxation_temperature=0.2, max_count=32, max_rate=10.)

    z, _ = q.sample(relaxed=True)
    assert torch.equal(z.detach(), z.detach().round())

    loss = z.mean() + q.kl(p).mean()
    loss.backward()
    assert torch.isfinite(log_rate_q.grad).all()


def test_continuous_relaxation_and_straight_through_are_distinct():
    torch.manual_seed(13)
    log_rate = torch.zeros(4, 8, requires_grad=True)
    q = Poisson(log_rate, relaxation_temperature=0.4, max_count=32, max_rate=10.)

    relaxed, _ = q.sample(estimator='relaxed')
    straight_through, _ = q.sample(estimator='straight_through')

    assert not torch.equal(relaxed.detach(), relaxed.detach().round())
    assert torch.equal(straight_through.detach(), straight_through.detach().round())
    (relaxed.mean() + straight_through.mean()).backward()
    assert torch.isfinite(log_rate.grad).all()


def test_obbvi_poisson_proposal_and_importance_identity():
    q = Poisson(torch.tensor([0.2]).log(), max_rate=30.)
    taus = (1., 3.)
    counts = torch.arange(0., 40.)

    log_q = q.log_p(counts)
    log_mixture = q.proposal_mixture_log_p(counts, taus)
    mixture = torch.exp(log_mixture)
    weights = torch.exp(log_q - log_mixture)

    # Because tau=1 is one of two mixture components, q/m <= 2.
    assert torch.max(weights) <= 2. + 1e-6
    # E_m[(q/m) z] = E_q[z] = rate (up to negligible tail truncation).
    estimate = torch.sum(mixture * weights * counts)
    assert torch.allclose(estimate, q.rate.squeeze(), atol=1e-5, rtol=1e-5)


def test_poisson_analytic_kl_matches_torch():
    log_rate_q = torch.tensor([[-0.7, 0.2, 1.1]])
    log_rate_p = torch.tensor([[0.4, -0.3, 0.8]])
    q = Poisson(log_rate_q, max_rate=30.)
    p = Poisson(log_rate_p, max_rate=30.)
    expected = torch.distributions.kl_divergence(
        torch.distributions.Poisson(q.rate), torch.distributions.Poisson(p.rate))
    assert torch.allclose(q.kl(p), expected, atol=1e-6, rtol=1e-6)


def test_gamma_sample_and_analytic_kl_have_finite_gradients():
    torch.manual_seed(11)
    log_shape_q = torch.tensor([[0.2, -0.4]], requires_grad=True)
    log_rate_q = torch.tensor([[-0.1, 0.7]], requires_grad=True)
    q = Gamma(log_shape_q, log_rate_q)
    p = Gamma(torch.tensor([[0.5, 0.1]]), torch.tensor([[0.3, -0.2]]))

    z, _ = q.sample()
    expected = torch.distributions.kl_divergence(
        torch.distributions.Gamma(q.shape, q.rate),
        torch.distributions.Gamma(p.shape, p.rate))
    assert torch.all(z > 0.)
    assert torch.allclose(q.kl(p), expected, atol=1e-5, rtol=1e-5)

    loss = z.mean() + q.kl(p).mean()
    loss.backward()
    assert torch.isfinite(log_shape_q.grad).all()
    assert torch.isfinite(log_rate_q.grad).all()
