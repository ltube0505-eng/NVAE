import pytest
import torch

from group1_svi import refine_momentum, temporary_model_state
from test_mixed_model import _poisson_model
import utils


def test_momentum_adjoint_matches_exact_unroll_with_feature_and_model_paths():
    torch.manual_seed(2)
    initial = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
    feature = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
    weight = torch.tensor(0.7, dtype=torch.double, requires_grad=True)

    def objective(local, features):
        noise = torch.randn_like(local)
        return ((weight * local + features[0] - noise) ** 2).mean()

    state = torch.get_rng_state()
    out = refine_momentum(objective, initial, (feature,), (weight,), 3, 0.1, 0.5, 1e-6)
    forward_state = torch.get_rng_state()
    loss = (out * feature).square().sum() + weight * out.sum()
    actual = torch.autograd.grad(loss, (initial, feature, weight))
    assert torch.equal(torch.get_rng_state(), forward_state)
    torch.set_rng_state(state)
    expected_out = initial
    velocity = torch.zeros_like(initial)
    for _ in range(3):
        g, = torch.autograd.grad(objective(expected_out, (feature,)), expected_out,
                                create_graph=True)
        velocity = 0.5 * velocity - g
        expected_out = expected_out + 0.1 * velocity
    expected_loss = (expected_out * feature).square().sum() + weight * expected_out.sum()
    expected = torch.autograd.grad(expected_loss, (initial, feature, weight))
    assert torch.allclose(out, expected_out, atol=1e-12)
    for a, b in zip(actual, expected):
        assert torch.allclose(a, b, atol=2e-6, rtol=2e-6)


def test_zero_steps_is_identity():
    x = torch.tensor(1., requires_grad=True)
    assert refine_momentum(lambda *_: None, x, (), (), 0, 0.1) is x


@pytest.mark.parametrize('estimator', ['straight_through', 'relaxed'])
def test_only_group1_refined_all_groups_resampled_and_outer_gradients(estimator):
    torch.manual_seed(1)
    model = _poisson_model(estimator).train()
    model.group1_svi_steps = 2
    model.group1_svi_lr = 0.05
    calls = []
    original = model._latent_from_param

    def record(group, param, *a, **kw):
        calls.append(group)
        return original(group, param, *a, **kw)

    model._latent_from_param = record
    x = torch.bernoulli(torch.full((2, 1, 32, 32), 0.2))
    encoder_before = model.enc_sampler[0].weight.detach().clone()
    decoder_before = model.image_conditional[1].weight.detach().clone()
    opt = torch.optim.Adamax(model.parameters(), lr=1e-3)
    logits, _, _, kls, _ = model(x)
    # One q for group0, then p and q for each other group, for n+1 passes.
    assert calls == [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5] * 3
    loss = (utils.reconstruction_loss(model.decoder_output(logits), x, crop=True)
            + sum(kls)).mean()
    assert len(kls) == 6
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
    for group in range(6):
        assert model.enc_sampler[group].weight.grad.abs().sum() > 0
    opt.step()
    assert not torch.equal(encoder_before, model.enc_sampler[0].weight)
    assert not torch.equal(decoder_before, model.image_conditional[1].weight)


def test_validation_refines_under_no_grad_then_samples_exactly():
    model = _poisson_model().eval()
    model.group1_svi_steps = 2
    modes = []
    sample = model._sample_latent

    def record(dist, training_sample=False):
        modes.append(training_sample)
        return sample(dist, training_sample)

    model._sample_latent = record
    with torch.no_grad():
        result = model(torch.zeros(2, 1, 32, 32))
    assert modes == [True] * 12 + [False] * 6
    assert not result[0].requires_grad
    assert all(p.grad is None for p in model.parameters())


def test_inner_passes_do_not_advance_batchnorm_buffers():
    model = _poisson_model()
    model.extra_bn = torch.nn.BatchNorm2d(1)
    original = model.extra_bn.running_mean
    with temporary_model_state(model):
        model.extra_bn(torch.ones(2, 1, 2, 2))
        assert model.extra_bn.num_batches_tracked == 1
    assert model.extra_bn.running_mean is original
    assert model.extra_bn.num_batches_tracked == 0
    assert torch.equal(original, torch.zeros_like(original))


def test_inner_objective_is_only_reconstruction_plus_first_kl():
    torch.manual_seed(7)
    model = _poisson_model().train()
    model.group1_svi_steps = 1
    x = torch.bernoulli(torch.full((2, 1, 32, 32), 0.2))
    param, ftr, cells, features = model._encode(x)
    initial, unused = param.chunk(2, 1)
    initial = initial.detach().requires_grad_(True)
    state = torch.get_rng_state()
    logits, _, _, kl_all, _ = model._decode(torch.cat((initial, unused), 1),
                                          ftr, cells, features)
    local = (utils.reconstruction_loss(model.decoder_output(logits), x, True)
             + kl_all[0]).mean()
    g, = torch.autograd.grad(local, initial)
    expected = initial.detach() - model.group1_svi_lr * g
    seen = []
    decode = model._decode

    def record(param0, *a, **kw):
        seen.append(param0[:, :model.num_latent_per_group].detach())
        return decode(param0, *a, **kw)

    model._decode = record
    torch.set_rng_state(state)
    model(x)
    assert len(seen) == 2
    assert torch.allclose(seen[-1], expected, atol=1e-6)


def test_zero_steps_matches_original_path_and_prior_sampling_ignores_svi():
    model = _poisson_model().eval()
    x = torch.zeros(2, 1, 32, 32)
    with torch.no_grad():
        torch.manual_seed(19)
        original = model._decode(*model._encode(x))
        torch.manual_seed(19)
        disabled = model(x)
        assert torch.equal(original[0], disabled[0])
        torch.manual_seed(20)
        prior = model.sample(2, 1.)
        model.group1_svi_steps = 3
        torch.manual_seed(20)
        assert torch.equal(prior, model.sample(2, 1.))
