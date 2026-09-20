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
