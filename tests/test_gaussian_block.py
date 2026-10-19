from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from gaussian_block import BlockGaussian, GaussianBlockLatent, blockify, zero_insert_2x
from model import AutoEncoder
import utils


def test_spatial_order_and_stride_one_upscale():
    x = torch.arange(24.).reshape(1, 1, 4, 6)
    assert torch.equal(blockify(x)[0, 0, 0, 0], torch.tensor([0., 1., 6., 7.]))
    assert torch.equal(blockify(x)[0, 0, 1, 2], torch.tensor([16., 17., 22., 23.]))
    x.requires_grad_()
    up = zero_insert_2x(x)
    assert up.shape == (1, 1, 8, 12)
    assert torch.equal(up[:, :, ::2, ::2], x)
    assert torch.sum(up[:, :, 1::2, :] != 0) == 0
    up.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))
    layer = GaussianBlockLatent([3, 4, 6])
    mean, scale, logits = layer.prior_upscale(layer.prior_param)
    assert mean.shape == scale.shape == (1, 3, 8, 12)
    assert logits.shape == (1, 1, 8, 12)
    assert layer.prior_upscale.mean.stride == (1, 1)


def test_selector_is_blockwise_softmax_and_shared_across_channels():
    # Nonuniform probabilities; practically noiseless, distinguishable channels.
    torch.manual_seed(4)
    n = 16000
    means = torch.tensor([0., 1., 2., 3., -1., -2., -3., -4.]).reshape(1, 2, 2, 2)
    logits = torch.tensor([0.1, 0.2, 0.3, 0.4]).log().reshape(1, 1, 2, 2)
    q = BlockGaussian(means.expand(n, -1, -1, -1),
                      torch.full((n, 2, 2, 2), -100.), logits.expand(n, -1, -1, -1))
    z, log_selection = q.sample()
    index = z[:, 0, 0, 0].unsqueeze(-1).sub(q.components.mu[0, 0, 0, 0]).abs().argmin(-1)
    freq = torch.bincount(index, minlength=4).float() / n
    assert torch.allclose(freq, logits.exp().flatten(), atol=0.015)
    expected_second = q.components.mu[0, 1, 0, 0][index]
    assert (z[:, 1, 0, 0] - expected_second).abs().max() < 0.15
    assert torch.allclose(log_selection, logits.flatten()[index])


def test_density_matches_explicit_joint_channel_mixture():
    torch.manual_seed(8)
    mean, scale = torch.randn(2, 3, 4, 6), torch.randn(2, 3, 4, 6)
    logits = torch.randn(2, 1, 4, 6)
    q = BlockGaussian(mean, scale, logits)
    z, _ = q.sample()
    components = torch.distributions.Normal(q.components.mu, q.components.sigma)
    densities = components.log_prob(z.unsqueeze(-1)).sum(1, keepdim=True).exp()
    expected = (densities * q.log_weights.exp()).sum(-1).log()
    assert torch.allclose(q.log_p(z), expected, atol=2e-5)


def test_requested_kl_sum_and_weighted_joint_upper_bound():
    torch.manual_seed(10)
    logits = torch.randn(2, 1, 4, 6)
    q = BlockGaussian(torch.randn(2, 3, 4, 6), torch.randn(2, 3, 4, 6), logits)
    p = BlockGaussian(torch.randn(2, 3, 4, 6), torch.randn(2, 3, 4, 6), logits)
    components = torch.distributions.kl_divergence(
        torch.distributions.Normal(q.components.mu, q.components.sigma),
        torch.distributions.Normal(p.components.mu, p.components.sigma))
    assert torch.allclose(q.kl(p), components.sum(-1), atol=2e-5)
    weighted_joint_kl = (components * q.log_weights.exp()).sum(-1)
    assert torch.all(q.kl(p) >= weighted_joint_kl - 1e-5)
    assert torch.allclose(q.kl(q), torch.zeros_like(q.kl(q)), atol=1e-6)


def test_selector_score_gradient_equals_enumerated_expected_reward_gradient():
    # One block, exact enumeration of its categorical expectation.
    logits = torch.tensor([0.1, -0.3, 0.5, 0.8], requires_grad=True)
    rewards = torch.tensor([2., 4., -1., 3.])
    probs = logits.softmax(-1)
    exact = torch.autograd.grad((probs * rewards).sum(), logits, retain_graph=True)[0]
    layer = GaussianBlockLatent([1, 1, 1])
    layer.reconstruction_baseline.fill_(1.5)
    layer.baseline_ready.fill_(True)
    correction = layer.selector_correction(rewards, logits.log_softmax(-1))
    assert torch.equal(correction, torch.zeros_like(correction))
    score_grad = torch.autograd.grad((probs.detach() * correction).sum(), logits)[0]
    assert torch.allclose(score_grad, exact, atol=1e-6)


def _args(enabled=True, **kwargs):
    values = dict(dataset='mnist', use_se=False, res_dist=True, num_x_bits=8,
                  latent_distribution='normal', gaussian_block_upscale=enabled,
                  num_latent_scales=1, num_groups_per_scale=1, num_latent_per_group=2,
                  ada_groups=False, min_groups_per_scale=1,
                  num_channels_enc=4, num_channels_dec=4,
                  num_preprocess_blocks=1, num_preprocess_cells=1,
                  num_postprocess_blocks=1, num_postprocess_cells=1,
                  num_cell_per_cond_enc=1, num_cell_per_cond_dec=1,
                  num_mixture_dec=1, num_nf=0)
    values.update(kwargs)
    return SimpleNamespace(**values)


def _model(args=None):
    model = AutoEncoder(args or _args(), None, utils.get_arch_cells('res_elu'))
    for module in model.modules():
        if hasattr(module, 'init_done'):
            module.init_done = True
    return model


def test_model_reconstruction_prior_gradients_and_checkpoint():
    torch.manual_seed(13)
    model = _model().train()
    with torch.no_grad():
        model.gaussian_block.prior_param.normal_(0., 0.1)
    samples, decoder_inputs = [], []
    original_sample = BlockGaussian.sample

    def record_sample(dist):
        result = original_sample(dist)
        samples.append(result[0])
        return result

    hook = model.stem_decoder.register_forward_pre_hook(
        lambda module, inputs: decoder_inputs.append(inputs[0]))
    x = torch.bernoulli(torch.full((2, 1, 32, 32), 0.5))
    with patch.object(BlockGaussian, 'sample', record_sample):
        logits, log_q, log_p, kl_all, kl_diag = model(x)
        assert decoder_inputs[-1] is samples[-1]
        assert logits.shape == x.shape
        assert log_q.shape == log_p.shape == kl_all[0].shape == (2,)
        assert kl_diag[0].shape == (2,)
        reconstruction = utils.reconstruction_loss(model.decoder_output(logits), x, crop=True)
        loss = (reconstruction + kl_all[0] + model.gaussian_block_selector_correction(reconstruction)).mean()
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None, name
                assert torch.isfinite(parameter.grad).all(), name
        for module in (model.enc_sampler, model.gaussian_block.posterior_upscale,
                       model.gaussian_block.prior_upscale.selector, model.stem_decoder):
            assert sum(p.grad.abs().sum() for p in module.parameters()) > 0
        model.eval()
        with torch.no_grad():
            generated = model.sample(3, 1.)
            assert decoder_inputs[-1] is samples[-1]
            assert generated.shape == (3, 1, 32, 32)
    hook.remove()
    restored = _model()
    restored.load_state_dict(model.state_dict(), strict=True)
    assert bool(restored.gaussian_block.baseline_ready)


@pytest.mark.parametrize('kwargs', [dict(num_latent_scales=2), dict(num_groups_per_scale=2),
                                   dict(num_nf=1), dict(latent_distribution='poisson')])
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError, match='gaussian_block_upscale'):
        _model(_args(**kwargs))


def test_old_args_and_state_dict_remain_compatible():
    args = _args(False)
    del args.gaussian_block_upscale
    original = _model(args)
    restored = _model(_args(False))
    restored.load_state_dict(original.state_dict(), strict=True)
    assert not any('gaussian_block' in name for name in original.state_dict())
    assert original(torch.rand(2, 1, 32, 32))[0].shape == (2, 1, 32, 32)
