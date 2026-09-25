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
    args.obbvi_baseline_samples = 2
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


def test_obbvi_replays_prefix_and_uses_group_weight(monkeypatch):
    model = _poisson_model('obbvi').train()
    x = torch.bernoulli(torch.full((2, 1, 32, 32), 0.5))
    model(x)
    prefix = [z.detach().clone() for z in model._gradient_context['samples']]
    sample_calls = []
    original_sample = model._sample_latent

    def record_sample(dist, training_sample=False, proposal_group=False):
        sample_calls.append(proposal_group)
        return original_sample(dist, training_sample, proposal_group)

    monkeypatch.setattr(model, '_sample_latent', record_sample)
    for objective in ('sampled', 'analytic_kl'):
        model.obbvi_objective = objective
        for group in (0, 2):
            sample_calls.clear()
            model.set_obbvi_component(1)
            logits, _, _, _, _ = model(x, prefix_samples=prefix, proposal_group=group)
            context = model._gradient_context
            assert sample_calls == [True] + [False] * (model.num_groups - group - 1)
            assert all(torch.equal(context['samples'][j], prefix[j]) for j in range(group))
            assert len(context['group_log_q']) == model.num_groups
            assert len(context['group_kl']) == model.num_groups
            recon = -model.decoder_output(logits).log_prob(x)[:, :, 2:30, 2:30].sum(dim=[1, 2, 3])
            coefficients = recon.new_tensor([1.7] + [0.6] * (model.num_groups - 1))
            score, signal, weight = model.conditional_score_objective(
                recon, 1., model.score_baseline(recon), group,
                group_coeffs=coefficients)
            if objective == 'analytic_kl':
                suffix = sum(coefficients[j] * context['group_kl'][j]
                             for j in range(group + 1, model.num_groups))
            else:
                suffix = sum(coefficients[j] * (context['group_log_q'][j] -
                                                context['group_log_p'][j])
                             for j in range(group, model.num_groups))
            assert torch.allclose(signal, (recon + suffix).detach())
            expected = torch.exp((context['group_log_q'][group] -
                                  context['proposal_log_m']).float()).detach()
            assert torch.allclose(weight, expected)
            assert torch.isfinite(score).all() and torch.isfinite(signal).all()
            assert torch.isfinite(weight).all()
            assert torch.max(weight) <= len(model.obbvi_taus) + 1e-5
            score.mean().backward()
            assert any(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in model.enc_sampler.parameters())
            model.zero_grad()


def test_obbvi_analytic_kl_direct_gradient_without_reconstruction():
    model = _poisson_model('obbvi').train()
    x = torch.rand(2, 1, 32, 32)
    model(x)
    first_kl = model._gradient_context['group_kl'][0].mean()
    first_kl.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.enc_sampler[0].parameters())


def test_conditional_poisson_baseline_uses_pilot_score_and_dmis_weights():
    signals = [torch.tensor([2., 8.]), torch.tensor([10., 4.])]
    weights = [torch.tensor([1., 2.]), torch.tensor([2., 1.])]
    score_norms = [torch.tensor([4., 0.]), torch.tensor([1., 0.])]
    baseline = AutoEncoder.conditional_poisson_baseline(signals, weights, score_norms)
    assert torch.allclose(baseline, torch.tensor([6., 0.]))
    assert not baseline.requires_grad


def test_conditional_poisson_score_norm_uses_selected_group_rate():
    model = _poisson_model('obbvi').train()
    x = torch.rand(2, 1, 32, 32)
    model(x)
    prefix = [z.detach() for z in model._gradient_context['samples']]
    model.set_obbvi_component(0)
    model(x, prefix_samples=prefix, proposal_group=1)
    context = model._gradient_context
    expected = ((context['samples'][1] - context['proposal_rate']) ** 2).sum(dim=[1, 2, 3])
    assert torch.allclose(model.conditional_poisson_score_norm(1), expected)


def test_obbvi_suffix_recomputes_conditional_parameters(monkeypatch):
    model = _poisson_model('obbvi').train()
    x = torch.rand(2, 1, 32, 32)
    model(x)
    prefix = [z.detach().clone() for z in model._gradient_context['samples']]
    original_sample = model._sample_latent
    observed_rates = []
    forced_count = [0.]

    def forced_proposal(dist, training_sample=False, proposal_group=False):
        if proposal_group:
            z = torch.full_like(dist.rate, forced_count[0])
            return z, torch.stack([dist.proposal_log_p(z, tau)
                                   for tau in model.obbvi_taus], dim=0)
        observed_rates.append(dist.rate.detach().clone())
        return original_sample(dist, training_sample, proposal_group)

    monkeypatch.setattr(model, '_sample_latent', forced_proposal)
    model(x, prefix_samples=prefix, proposal_group=2)
    first_suffix_rate = observed_rates[0]
    observed_rates.clear()
    forced_count[0] = 5.
    model(x, prefix_samples=prefix, proposal_group=2)
    assert not torch.allclose(first_suffix_rate, observed_rates[0])
