from unittest.mock import patch

import pytest
import torch

from distributions import Normal
from test_gaussian_block import _args, _model


def test_learned_prior_elbo_gradients_optimizer_and_round_trip():
    torch.manual_seed(21)
    args = _args(False, learnable_prior=True)
    model = _model(args).train()
    x = torch.bernoulli(torch.full((2, 1, 32, 32), 0.5))
    posterior = []
    hook = model.enc_sampler[0].register_forward_hook(
        lambda module, inputs, output: posterior.append(output))
    logits, _, log_p, kl_all, _ = model(x)
    hook.remove()
    q = model._latent_from_param(0, posterior[0])
    p = model._first_prior(q.mu)
    # The training KL must use the learned distribution, with image sharing.
    assert torch.allclose(kl_all[0], q.kl(p).sum([1, 2, 3]))
    assert model.prior_param.shape == (1, 4, 16, 16)
    assert torch.equal(p.mu[0], p.mu[1])
    fixed = model._fixed_prior(0, q.mu)
    assert torch.equal(p.mu, fixed.mu)
    assert torch.equal(p.sigma, fixed.sigma)
    optimizer = torch.optim.Adamax(model.parameters(), lr=0.01)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, x) + kl_all[0].mean()
    loss.backward()
    # This also protects the manual distributed reducer against unused params.
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
    assert (model.prior_param.grad[:, :2].abs().sum() > 0)
    assert (model.prior_param.grad[:, 2:].abs().sum() > 0)
    optimizer.step()
    assert torch.count_nonzero(model.prior_param) > 0
    restored = _model(args)
    restored.load_state_dict(model.state_dict(), strict=True)
    assert torch.equal(restored.prior_param, model.prior_param)
    restored_optimizer = torch.optim.Adamax(restored.parameters(), lr=0.01)
    restored_optimizer.load_state_dict(optimizer.state_dict())
    assert torch.equal(restored_optimizer.state[restored.prior_param]['exp_avg'],
                       optimizer.state[model.prior_param]['exp_avg'])
    assert torch.isfinite(log_p).all()


def test_training_density_and_generation_use_same_nonzero_prior():
    model = _model(_args(False, learnable_prior=True)).eval()
    with torch.no_grad():
        model.prior_param[:, :2].fill_(0.7)
        model.prior_param[:, 2:].fill_(-0.4)
    distributions, samples, decoder_inputs = [], [], []
    original_log_p = Normal.log_p

    def record_log_p(dist, z):
        distributions.append(dist)
        samples.append(z)
        return original_log_p(dist, z)

    with torch.no_grad(), patch.object(Normal, 'log_p', record_log_p):
        _, _, log_p, _, _ = model(torch.rand(2, 1, 32, 32))
    expected = model._latent_from_param(0, model.prior_param.expand(2, -1, -1, -1))
    assert torch.equal(distributions[-1].mu, expected.mu)
    assert torch.allclose(log_p, original_log_p(expected, samples[-1]).sum([1, 2, 3]))

    hook = model.stem_decoder.register_forward_pre_hook(
        lambda module, inputs: decoder_inputs.append(inputs[0]))

    def noiseless_sample(dist):
        assert torch.equal(dist.mu, expected.mu)
        assert torch.allclose(dist.sigma, expected.sigma * 0.5)
        return dist.mu, torch.zeros_like(dist.mu)

    with torch.no_grad(), patch.object(Normal, 'sample', noiseless_sample):
        generated = model.sample(2, 0.5)
    hook.remove()
    assert torch.equal(decoder_inputs[-1], expected.mu)
    assert generated.shape == (2, 1, 32, 32)


def test_legacy_fixed_prior_checkpoint_stays_fixed():
    args = _args(False)
    original = _model(args)
    assert not hasattr(args, 'learnable_prior')
    assert not original.learnable_prior
    assert 'prior_param' not in original.state_dict()
    restored = _model(_args(False, learnable_prior=False))
    restored.load_state_dict(original.state_dict(), strict=True)
    reference = torch.zeros(3, *restored.z0_size)
    p = restored._first_prior(reference)
    assert torch.equal(p.mu, torch.zeros_like(p.mu))
    assert torch.allclose(p.sigma, torch.full_like(p.sigma, 1.01))


@pytest.mark.parametrize('kwargs', [dict(num_latent_scales=2), dict(num_groups_per_scale=2),
                                   dict(latent_distribution='poisson'), dict(gaussian_block_upscale=True)])
def test_invalid_learnable_prior_configs_rejected(kwargs):
    with pytest.raises(ValueError, match='learnable_prior'):
        _model(_args(False, learnable_prior=True, **kwargs))
