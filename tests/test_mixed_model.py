from types import SimpleNamespace

import torch
from torch.distributions.bernoulli import Bernoulli

from model import AutoEncoder
import utils


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
