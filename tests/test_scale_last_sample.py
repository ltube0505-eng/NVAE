"""CPU tests of scale boundaries, conditional independence and gradients."""
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from model import AutoEncoder
from neural_operations import ScaleBoundaryCell
import utils


def make_model(**overrides):
    args = dict(
        dataset='mnist', use_se=False, res_dist=True, num_x_bits=8,
        latent_distribution='normal', scale_connection='last_sample',
        poisson_relaxation_temperature=0.2, poisson_max_count=32,
        poisson_max_rate=10., num_latent_scales=2, num_groups_per_scale=2,
        num_latent_per_group=2, ada_groups=False, min_groups_per_scale=1,
        num_channels_enc=4, num_channels_dec=4, num_preprocess_blocks=0,
        num_preprocess_cells=1, num_cell_per_cond_enc=1,
        num_postprocess_blocks=0, num_postprocess_cells=1,
        num_cell_per_cond_dec=1, num_mixture_dec=1, num_nf=0)
    args.update(overrides)
    if args['scale_connection'] is None:
        del args['scale_connection']
    # The legacy AR mask constructor unconditionally calls Tensor.cuda().
    # Override device placement only; run the real flow math on the CPU.
    with mock.patch.object(torch.Tensor, 'cuda', lambda tensor, *a, **kw: tensor):
        model = AutoEncoder(SimpleNamespace(**args), None, utils.get_arch_cells('res_elu'))
    # Avoid distributed data initialization, as in test_mixed_model.py.
    for module in model.modules():
        if hasattr(module, 'init_done'):
            module.init_done = True
    return model


def fixed_trajectory(model, changes=None, prior=False, x_value=0.):
    """Intervene on sampled values, leaving all real p/q heads in operation."""
    changes = changes or {}
    index = [0]
    priors = []
    handles = [head.register_forward_hook(
        lambda module, inputs, output: priors.append(output.detach().clone()))
        for head in model.dec_sampler]
    original = model._sample_latent

    def sample(dist, training_sample=False):
        i = index[0]
        index[0] += 1
        return torch.full_like(dist.mu, 0.1 * (i + 1) + changes.get(i, 0.)), None

    model._sample_latent = sample
    try:
        with torch.no_grad():
            logits = model.sample(2, 1.) if prior else model(
                torch.full((2, 1, 32, 32), x_value))[0]
    finally:
        model._sample_latent = original
        for handle in handles:
            handle.remove()
    return logits, priors


@pytest.mark.parametrize('prior', [False, True])
def test_bridge_uses_only_terminal_sample_and_only_for_first_group(prior):
    torch.manual_seed(31)
    model = make_model().eval()  # groups [2, 2] in generation order
    output, priors = fixed_trajectory(model, prior=prior)
    early_output, early_priors = fixed_trajectory(model, {0: 3.}, prior=prior)
    terminal_output, terminal_priors = fixed_trajectory(model, {1: 3.}, prior=prior)

    # Change the nonterminal group while holding the terminal sample fixed:
    # it changes its same-scale successor's prior, not the next scale's prior.
    assert not torch.allclose(priors[0], early_priors[0])
    assert torch.equal(priors[1], early_priors[1])
    # Changing the boundary sample changes the next scale's first prior.
    assert not torch.allclose(priors[1], terminal_priors[1])
    # Once the next first sample is fixed, no old-scale condition persists.
    assert torch.equal(priors[2], terminal_priors[2])
    # The independent image decoder intentionally sees ALL scales' samples.
    assert not torch.allclose(output, early_output)
    assert not torch.allclose(output, terminal_output)
    changed_output, _ = fixed_trajectory(model, {2: 3.}, prior=prior)
    assert not torch.allclose(output, changed_output)


def test_intervention_detects_original_feature_bypass():
    torch.manual_seed(31)
    model = make_model(scale_connection='feature').eval()
    output, _ = fixed_trajectory(model)
    changed_output, _ = fixed_trajectory(model, {1: 3.})
    assert not torch.allclose(output, changed_output)


def test_forward_and_generation_share_the_same_prior_and_likelihood():
    model = make_model(num_latent_scales=3).eval()
    output, priors = fixed_trajectory(model)
    sampled_output, sampled_priors = fixed_trajectory(model, prior=True)
    assert torch.equal(output, sampled_output)
    assert all(torch.equal(p, s) for p, s in zip(priors, sampled_priors))
    # Every group changes the likelihood, with all other samples held fixed.
    for group in range(6):
        intervened, _ = fixed_trajectory(model, {group: 3.})
        assert not torch.allclose(output, intervened)


@pytest.mark.parametrize('config,total', [
    ({}, 4),
    ({'num_latent_scales': 3}, 6),
    ({'num_latent_scales': 3, 'num_groups_per_scale': 4, 'ada_groups': True}, 7),
    ({'num_groups_per_scale': 1}, 2),
    ({'num_preprocess_blocks': 1, 'num_postprocess_blocks': 1}, 4),
    ({'latent_distribution': 'poisson'}, 4),
    ({'latent_distribution': 'mixed_poisson_gamma', 'num_groups_per_scale': 4,
      'ada_groups': True, 'min_groups_per_scale': 2}, 6),
    ({'num_nf': 1}, 4),
    ({'res_dist': False}, 4),
    ({'num_cell_per_cond_dec': 0}, 4),
])
def test_shapes_all_kl_terms_and_backward(config, total):
    torch.manual_seed(17)
    model = make_model(**config).train()
    x = torch.bernoulli(torch.full((2, 1, 32, 32), 0.5))
    logits, log_q, log_p, kl_all, kl_diag = model(x)
    assert logits.shape == x.shape
    assert log_q.shape == log_p.shape == (2,)
    assert len(kl_all) == len(kl_diag) == total
    loss = -model.decoder_output(logits).log_prob(x).mean() + torch.stack(kl_all).mean()
    assert torch.isfinite(loss)
    loss.backward()
    for boundary in (m for m in model.modules() if isinstance(m, ScaleBoundaryCell)):
        for parameter in boundary.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
    with torch.no_grad():
        generated = model.eval().sample(2, 1.)
    assert generated.shape == x.shape
    assert torch.isfinite(generated).all()


def test_reconstruction_gradient_reaches_previous_scale_sample():
    torch.manual_seed(19)
    model = make_model(num_latent_scales=3).train()
    samples = []

    def retain_sample(module, inputs):
        inputs[0].retain_grad()
        samples.append(inputs[0])

    handles = [m.register_forward_pre_hook(retain_sample)
               for m in model.modules() if isinstance(m, ScaleBoundaryCell)]
    try:
        logits = model(torch.rand(2, 1, 32, 32))[0]
        logits.square().mean().backward()  # reconstruction only, no KL
        assert len(samples) == 2
        for sample in samples:
            assert sample.grad is not None
            assert torch.isfinite(sample.grad).all()
            assert sample.grad.abs().sum() > 0
    finally:
        for handle in handles:
            handle.remove()


def test_flow_boundary_receives_actual_post_flow_sample():
    model = make_model(num_nf=1).eval()
    flow_samples, boundary_samples, image_samples = [], [], []
    handles = [cell.register_forward_hook(
        lambda module, inputs, output: flow_samples.append(output[0]))
        for cell in model.nf_cells]
    handles += [m.register_forward_pre_hook(
        lambda module, inputs: boundary_samples.append(inputs[0]))
        for m in model.modules() if isinstance(m, ScaleBoundaryCell)]
    handles += [cell.register_forward_pre_hook(
        lambda module, inputs: image_samples.append(inputs[1]))
        for cell in model.image_dec_tower if cell.cell_type == 'combiner_dec']
    try:
        with torch.no_grad():
            model(torch.rand(2, 1, 32, 32))
        assert boundary_samples[0] is flow_samples[1]
        assert len(image_samples) == len(flow_samples) == 4
        assert all(a is b for a, b in zip(image_samples, flow_samples))
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize('estimator', ['reinforce', 'obbvi'])
def test_score_function_training_keeps_full_joint_density(estimator):
    model = make_model(latent_distribution='poisson',
                       poisson_gradient_estimator=estimator, obbvi_num_samples=2).train()
    x = torch.bernoulli(torch.full((2, 1, 32, 32), 0.5))
    logits, log_q, log_p, kl_all, _ = model(x)
    recon = -model.decoder_output(logits).log_prob(x).sum(dim=[1, 2, 3])
    objective, nelbo, weight = model.score_function_objective(recon, 1., recon.new_zeros(()))
    assert len(kl_all) == 4
    assert torch.allclose(nelbo, (recon + log_q - log_p).detach())
    assert torch.isfinite(objective).all() and torch.isfinite(weight).all()
    objective.mean().backward()


def test_independent_initial_parameters_survive_checkpoint_round_trip():
    model = make_model(num_latent_scales=3).eval()
    seeds = [model.prior_ftr0] + [m.initial_feature for m in model.modules()
                                if isinstance(m, ScaleBoundaryCell)]
    assert len(seeds) == 3
    assert len({seed.data_ptr() for seed in seeds}) == 3
    assert all(seed.requires_grad for seed in seeds)
    restored = make_model(num_latent_scales=3).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    a, _ = fixed_trajectory(model)
    b, _ = fixed_trajectory(restored)
    c, _ = fixed_trajectory(model)
    assert torch.equal(a, b) and torch.equal(a, c)


@pytest.mark.parametrize('scales,groups', [(2, 2), (1, 2), (1, 1)])
def test_missing_flag_preserves_legacy_architecture(scales, groups):
    torch.manual_seed(23)
    old = make_model(scale_connection=None, num_latent_scales=scales,
                     num_groups_per_scale=groups).eval()
    torch.manual_seed(23)
    explicit = make_model(scale_connection='feature', num_latent_scales=scales,
                          num_groups_per_scale=groups).eval()
    explicit.load_state_dict(old.state_dict(), strict=True)
    a, _ = fixed_trajectory(old)
    b, _ = fixed_trajectory(explicit)
    assert torch.equal(a, b)
    if scales == 1:
        single = make_model(num_latent_scales=scales, num_groups_per_scale=groups).eval()
        single.load_state_dict(old.state_dict(), strict=True)
        c, _ = fixed_trajectory(single)
        assert torch.equal(a, c)


def test_invalid_connection_is_rejected():
    with pytest.raises(ValueError, match='Unknown scale connection'):
        make_model(scale_connection='typo')


@pytest.mark.parametrize('adaptive', [False, True])
def test_every_group_has_direct_reconstruction_gradient(adaptive):
    torch.manual_seed(43)
    model = make_model(num_latent_scales=3, num_groups_per_scale=4,
                       ada_groups=adaptive).eval()
    latents = []
    for scale, groups in enumerate(reversed(model.groups_per_scale)):
        size = model.z0_size[1] * 2 ** scale
        latents.extend(torch.randn(2, model.num_latent_per_group, size, size,
                                   requires_grad=True) for _ in range(groups))
    s = model._image_features_from_latents(latents)
    for cell in model.post_process:
        s = cell(s)
    model.image_conditional(s).square().mean().backward()
    for z in latents:
        assert z.grad is not None and torch.isfinite(z.grad).all()
        assert z.grad.abs().sum() > 0
    for p in model.image_dec_tower.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    assert model.image_ftr0.grad is not None


def test_image_tower_does_not_feed_back_into_latent_conditionals():
    model = make_model().eval()
    latent_ids = {id(p) for p in model.dec_tower.parameters()}
    assert not latent_ids.intersection(id(p) for p in model.image_dec_tower.parameters())
    output, priors = fixed_trajectory(model)
    with torch.no_grad():
        model.image_ftr0.add_(5.)
    changed, changed_priors = fixed_trajectory(model)
    assert not torch.allclose(output, changed)
    assert all(torch.equal(a, b) for a, b in zip(priors, changed_priors))


def test_image_decoder_has_no_deterministic_x_bypass():
    model = make_model().eval()
    a, _ = fixed_trajectory(model, x_value=0.)
    b, _ = fixed_trajectory(model, x_value=1.)
    assert torch.equal(a, b)


def test_training_regularizer_and_gradient_average_allow_discarded_features():
    model = make_model(num_groups_per_scale=1).train()
    logits, _, _, kl_all, _ = model(torch.rand(2, 1, 32, 32))
    # As with AR masks above, bypass only hardcoded CUDA placement in the
    # legacy spectral regularizer; its calculation and backward remain real.
    with mock.patch.object(torch.Tensor, 'cuda', lambda tensor, *a, **kw: tensor):
        regularizer = model.spectral_norm_parallel() + model.batchnorm_loss()
    loss = logits.square().mean() + torch.stack(kl_all).mean() + 0.01 * regularizer
    loss.backward()
    assert any(p.grad is None for p in model.parameters())
    gradients = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    assert all(torch.isfinite(grad).all() for grad in gradients)
    with mock.patch.object(utils.dist, 'get_world_size', return_value=2), \
            mock.patch.object(utils.dist, 'all_reduce', side_effect=lambda t, op: t.mul_(2)) as reduce:
        utils.average_gradients(model.parameters(), is_distributed=True)
    assert reduce.call_count == len(gradients)
    actual = [p.grad for p in model.parameters() if p.grad is not None]
    assert all(torch.equal(a, b) for a, b in zip(gradients, actual))
    torch.optim.Adamax(model.parameters(), lr=1e-3).step()
