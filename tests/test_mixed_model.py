from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch.distributions.bernoulli import Bernoulli

from model import AutoEncoder
import utils
from distributions import Poisson


def _args():
    return SimpleNamespace(
        dataset='mnist',
        use_se=False,
        res_dist=True,
        num_x_bits=8,
        latent_distribution='mixed_poisson_gamma',
        poisson_relaxation_temperature=0.2,
        poisson_max_count=32,
        poisson_max_rate=10.,
        num_latent_scales=2,
        num_groups_per_scale=4,
        num_latent_per_group=2,
        ada_groups=True,
        min_groups_per_scale=2,
        num_channels_enc=4,
        num_channels_dec=4,
        num_preprocess_blocks=0,
        num_preprocess_cells=1,
        num_cell_per_cond_enc=1,
        num_postprocess_blocks=0,
        num_postprocess_cells=1,
        num_cell_per_cond_dec=1,
        num_mixture_dec=1,
        num_nf=0,
    )


def _model():
    model = AutoEncoder(_args(), None, utils.get_arch_cells('res_elu'))
    # Data-dependent Conv2D initialization assumes a distributed process group;
    # it is unrelated to this CPU topology/gradient smoke test.
    for module in model.modules():
        if hasattr(module, 'init_done'):
            module.init_done = True
    return model


def _poisson_model(estimator='straight_through'):
    args = _args()
    args.latent_distribution = 'poisson'
    args.poisson_gradient_estimator = estimator
    args.obbvi_taus = '1.0,3.0'
    args.obbvi_num_samples = 2
    args.reinforce_num_samples = 1
    args.score_baseline_decay = 0.9
    model = AutoEncoder(args, None, utils.get_arch_cells('res_elu'))
    for module in model.modules():
        if hasattr(module, 'init_done'):
            module.init_done = True
    return model


def test_group_mapping_and_dataset_default_head():
    model = _model()
    assert model.groups_per_scale == [4, 2]
    assert [model._latent_kind(i) for i in range(6)] == [
        'poisson', 'poisson', 'gamma', 'gamma', 'gamma', 'gamma']

    model.train()
    logits, log_q, log_p, kl_all, kl_diag = model(torch.rand(2, 1, 32, 32))
    # MNIST keeps the original one-logit Bernoulli reconstruction head.
    assert tuple(logits.shape) == (2, 1, 32, 32)
    assert isinstance(model.decoder_output(logits), Bernoulli)
    assert tuple(log_q.shape) == (2,)
    assert tuple(log_p.shape) == (2,)
    assert len(kl_all) == 6
    assert len(kl_diag) == 6
    assert all(torch.isfinite(kl).all() for kl in kl_all)


def test_exact_prior_sampling_path():
    model = _model().eval()
    with torch.no_grad():
        logits = model.sample(num_samples=2, t=1.)
    assert tuple(logits.shape) == (2, 1, 32, 32)
    assert torch.isfinite(logits).all()


def test_all_poisson_maps_every_group_to_poisson():
    model = _poisson_model()
    assert [model._latent_kind(i) for i in range(6)] == ['poisson'] * 6


def test_adaptive_cts_config_reaches_posterior_and_prior_with_legacy_args():
    # _args mimics old checkpoint arguments, including the old caps and no
    # new tolerance. Loading must use the new default without a hard rate cap.
    model = _poisson_model()
    param = torch.full((1, 4, 2, 2), 8.)
    q = model._latent_from_param(0, param)
    p = model._fixed_prior(0, param[:, :2])
    assert torch.allclose(q.rate, torch.exp(5. * torch.tanh(param[:, :2] / 5.)))
    assert torch.equal(p.rate, torch.ones_like(p.rate))
    assert q.cts_tail_tolerance == p.cts_tail_tolerance == 1e-4
    model.poisson_cts_tail_tolerance = 1e-6
    q = model._latent_from_param(0, param)
    p = model._fixed_prior(0, param[:, :2])
    assert q.cts_tail_tolerance == p.cts_tail_tolerance == 1e-6


def test_all_poisson_pathwise_estimators_have_finite_gradients():
    for estimator in ('relaxed', 'straight_through'):
        model = _poisson_model(estimator).train()
        logits, _, _, kl_all, _ = model(torch.rand(2, 1, 32, 32))
        loss = logits.mean() + torch.stack(kl_all).mean()
        loss.backward()
        assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                   for parameter in model.enc_sampler.parameters())


def test_reinforce_model_builds_finite_score_surrogate():
    model = _poisson_model('reinforce').train()
    x = torch.bernoulli(torch.full((2, 1, 32, 32), 0.5))
    logits, _, _, _, _ = model(x)
    recon = -model.decoder_output(logits).log_prob(x)[:, :, 2:30, 2:30].sum(dim=[1, 2, 3])
    baseline = model.score_baseline(recon)
    objective, sampled_nelbo, weight = model.score_function_objective(recon, 1., baseline)

    assert torch.equal(weight, torch.ones_like(weight))
    assert torch.isfinite(objective).all()
    objective.mean().backward()
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.enc_sampler.parameters())
    model.update_score_baseline([sampled_nelbo])
    assert model._score_baseline_value is not None


def test_obbvi_model_uses_bounded_dmis_weight():
    model = _poisson_model('obbvi').train()
    model.set_obbvi_component(1)
    x = torch.bernoulli(torch.full((2, 1, 32, 32), 0.5))
    logits, _, _, _, _ = model(x)
    recon = -model.decoder_output(logits).log_prob(x)[:, :, 2:30, 2:30].sum(dim=[1, 2, 3])
    objective, _, weight = model.score_function_objective(
        recon, 1., model.score_baseline(recon))

    assert torch.isfinite(objective).all()
    assert torch.isfinite(weight).all()
    assert torch.max(weight) <= len(model.obbvi_taus) + 1e-5


def _vanilla_model(blocks=3, cells=2, latent_distribution='poisson', **overrides):
    args = _args()
    args.latent_distribution = latent_distribution
    args.num_latent_scales = args.num_groups_per_scale = 1
    args.ada_groups = False
    args.min_groups_per_scale = 1
    args.num_preprocess_blocks = args.num_postprocess_blocks = blocks
    args.num_preprocess_cells = args.num_postprocess_cells = cells
    vars(args).update(overrides)
    model = AutoEncoder(args, None, utils.get_arch_cells('res_elu'))
    for module in model.modules():
        if hasattr(module, 'init_done'):
            module.init_done = True
    return model


def test_vanilla_fixed_gaussian_reconstruction_and_sampling():
    variance = 0.04
    model = _vanilla_model(image_likelihood='gaussian_fixed', gaussian_variance=variance)
    x = torch.rand(2, 1, 32, 32)
    logits, log_q, log_p, kl_all, _ = model(x)
    output = model.decoder_output(logits)
    expected = torch.distributions.Normal(logits.sigmoid(), variance ** 0.5)
    expected_recon = -expected.log_prob(x)[:, :, 2:30, 2:30].sum(dim=(1, 2, 3))
    recon = utils.reconstruction_loss(output, x, crop=model.crop_output)
    assert torch.allclose(recon, expected_recon)
    assert torch.allclose(utils.log_iw(output, x, log_q, log_p, crop=model.crop_output),
                          -expected_recon + log_p - log_q)
    (recon + kl_all[0]).mean().backward()
    assert torch.isfinite(model.image_conditional[-1].weight.grad).all()
    assert torch.isfinite(model.prior_log_rate.grad).all()
    with torch.no_grad():
        generated_logits = model.eval().sample(2, t=1.)
        rng = torch.get_rng_state().clone()
        sample = model.image_from_logits(generated_logits)
        assert torch.equal(sample, generated_logits.sigmoid())
        assert torch.equal(torch.get_rng_state(), rng)
        assert torch.equal(model.image_from_logits(generated_logits, t=0.7), sample)
    assert sample.shape == x.shape
    assert ((0. <= sample) & (sample <= 1.)).all()


def test_bernoulli_image_rendering_still_uses_mean():
    model = _vanilla_model()
    logits = torch.randn(2, 1, 32, 32)
    assert torch.equal(model.image_from_logits(logits), logits.sigmoid())


def test_mnist_transforms_preserve_grayscale_only_for_fixed_gaussian():
    from PIL import Image
    from datasets import _data_transforms_mnist

    image = Image.new('L', (28, 28), 128)
    args = SimpleNamespace(image_likelihood='gaussian_fixed')
    for transform in _data_transforms_mnist(args):
        pixels = transform(image)
        assert pixels.shape == (1, 32, 32)
        assert torch.allclose(pixels[:, 2:30, 2:30], torch.full((1, 28, 28), 128. / 255.))
        assert torch.count_nonzero(pixels[:, :2, :]) == 0
    for transform in _data_transforms_mnist(SimpleNamespace()):
        pixels = transform(image)
        assert ((pixels == 0.) | (pixels == 1.)).all()


@pytest.mark.parametrize('blocks,cells', [(2, 3), (3, 2)])
def test_vanilla_topology_and_prior_initialization(blocks, cells):
    model = _vanilla_model(blocks, cells)
    assert model.vanilla_vae
    assert len(model.enc_tower) == len(model.dec_tower) == 0
    assert len(model.pre_process) == len(model.post_process) == blocks * cells
    assert sum(c.cell_type == 'down_pre' for c in model.pre_process) == blocks
    assert sum(c.cell_type == 'up_post' for c in model.post_process) == blocks
    assert model.prior_log_rate.shape == (1, 2, 32 // 2 ** blocks, 32 // 2 ** blocks)
    reference = torch.zeros([3] + model.z0_size)
    prior = model._prior0(reference)
    fixed = model._fixed_prior(0, reference)
    assert torch.equal(prior.rate, fixed.rate)
    assert torch.equal(prior.log_p(reference), fixed.log_p(reference))


def test_vanilla_forward_prior_gradient_and_optimizer_update():
    model = _vanilla_model().train()
    q = Poisson(torch.full([2] + model.z0_size, 0.7, requires_grad=True))
    optimizer = torch.optim.Adamax(model.parameters(), lr=0.01)
    with patch.object(model, '_latent_from_param', return_value=q), \
            patch.object(model, '_sample_latent', wraps=model._sample_latent) as sampler:
        logits, _, log_p, kl_all, _ = model(torch.rand(2, 1, 32, 32))
    assert logits.shape == (2, 1, 32, 32)
    assert len(kl_all) == 1
    assert sampler.call_count == 1
    # At initialization the analytic KL is q*log(q) + 1 - q.
    expected_kl = (q.rate * q.log_rate + 1. - q.rate).sum(dim=[1, 2, 3])
    assert torch.allclose(kl_all[0], expected_kl)
    assert torch.isfinite(log_p).all()
    kl_all[0].mean().backward()
    expected_grad = (1. - q.rate.detach()).mean(dim=0, keepdim=True)
    assert torch.allclose(model.prior_log_rate.grad, expected_grad)
    optimizer.step()
    assert (model.prior_log_rate > 0.).all()


def test_vanilla_learned_prior_used_by_forward_sampling_and_checkpoint():
    model = _vanilla_model().eval()
    with torch.no_grad():
        model.prior_log_rate.copy_(torch.linspace(-0.5, 1., model.prior_log_rate.numel())
                                   .reshape_as(model.prior_log_rate))
    reference = torch.zeros([2] + model.z0_size)
    expected_rate = (5. * torch.tanh(model.prior_log_rate / 5.)).exp().expand_as(reference)
    # Keep posterior samples deterministic so log_p can be checked independently.
    z = torch.ones_like(reference)
    with patch.object(model, '_sample_latent', return_value=(z, None)):
        _, _, log_p, _, _ = model(torch.rand(2, 1, 32, 32))
    expected_log_p = torch.distributions.Poisson(expected_rate).log_prob(z).sum(dim=[1, 2, 3])
    assert torch.allclose(log_p, expected_log_p)
    with torch.no_grad(), patch('torch.poisson', wraps=torch.poisson) as sampler:
        logits = model.sample(2, t=1.)
    assert logits.shape == (2, 1, 32, 32)
    assert torch.isfinite(logits).all()
    sampler.assert_called_once()
    assert torch.equal(sampler.call_args[0][0], expected_rate)
    restored = _vanilla_model()
    restored.load_state_dict(model.state_dict(), strict=True)
    assert torch.equal(restored._prior0(reference).rate, expected_rate)


def test_other_topologies_and_normal_prior_remain_fixed():
    for model in (_poisson_model(), _model(), _vanilla_model(latent_distribution='normal')):
        assert model.prior_log_rate is None
        assert 'prior_log_rate' not in model.state_dict()
        reference = torch.zeros([2] + model.z0_size)
        assert torch.equal(model._prior0(reference).log_p(reference),
                           model._fixed_prior(0, reference).log_p(reference))



@pytest.mark.parametrize('raw_prior', [-8., -1., 2., 8.])
def test_learned_prior_soft_cap_kl_chain_rule(raw_prior):
    model = _vanilla_model()
    with torch.no_grad():
        model.prior_log_rate.fill_(raw_prior)
    reference = torch.zeros([2] + model.z0_size)
    q = Poisson(torch.full_like(reference, 0.7))
    p = model._prior0(reference)
    p_expected = (5. * torch.tanh(model.prior_log_rate / 5.)).exp()
    assert torch.equal(p.rate, p_expected.expand_as(reference))
    q.kl(p).sum(dim=[1, 2, 3]).mean().backward()
    expected = ((p.rate - q.rate).mean(dim=0, keepdim=True)
                * (1. - torch.tanh(model.prior_log_rate / 5.).pow(2)))
    assert torch.allclose(model.prior_log_rate.grad, expected, atol=1e-6)
