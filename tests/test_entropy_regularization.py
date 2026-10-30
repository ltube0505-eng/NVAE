"""Numerical objective/gradient checks, including hierarchical score paths."""
import math
from types import SimpleNamespace

import pytest
import torch

from entropy_regularization import entropy_alpha, poisson_entropy
from distributions import Poisson
from model import AutoEncoder


def test_schedule_endpoints_monotonicity_and_resume():
    values = [entropy_alpha(s, 1001) for s in range(1001)]
    assert values[0] == 0.1
    assert values[250] == pytest.approx(0.05)
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert all(v == 0. for v in values[500:])
    assert [entropy_alpha(s, 1001) for s in range(321, 1001)] == values[321:]
    assert entropy_alpha(1000, 1001, anneal_portion=1.) == 0.
    assert entropy_alpha(0, 1) == 0.
    assert entropy_alpha(0, 1001, initial=0.) == 0.


@pytest.mark.parametrize('initial,portion', [(-0.1, .5), (1., .5), (float('nan'), .5),
                                           (.1, 0.), (.1, 1.1), (.1, float('nan'))])
def test_invalid_schedule_rejected(initial, portion):
    with pytest.raises(ValueError):
        entropy_alpha(0, 100, initial, portion)


def test_entropy_series_and_gradient_against_long_support():
    # Includes the soft-clamped minimum and maximum possible model rates.
    rate = torch.tensor([math.exp(-5.), .1, 1., 5., 30., math.exp(5.)],
                        dtype=torch.float64, requires_grad=True)
    entropy = poisson_entropy(rate, math.exp(5.))
    k = torch.arange(0., 600., dtype=torch.float64).view(-1, 1)
    log_p = torch.distributions.Poisson(rate).log_prob(k)
    reference = -(log_p.exp() * log_p).sum(0)
    expected_grad, = torch.autograd.grad(reference.sum(), rate)
    actual_grad, = torch.autograd.grad(entropy.sum(), rate)
    assert torch.allclose(entropy, reference, rtol=1e-10, atol=1e-10)
    assert torch.allclose(actual_grad, expected_grad, rtol=1e-9, atol=1e-10)
    assert (actual_grad > 0.).all()
    assert torch.autograd.gradcheck(lambda r: poisson_entropy(r, 150.), (rate,))


def test_entropy_float32_half_and_sampler_cap_independence():
    rates = torch.tensor([.007, .1, 1., 30., 148.], requires_grad=True)
    result = poisson_entropy(rates, 150.)
    reference = poisson_entropy(rates.detach().double(), 150.)
    assert torch.allclose(result.double(), reference, atol=5e-5, rtol=2e-5)
    result.sum().backward()
    assert torch.isfinite(rates.grad).all()
    half = rates.detach().half().requires_grad_()
    h = poisson_entropy(half, 150.)
    assert h.dtype == torch.float32
    h.sum().backward()
    assert torch.isfinite(half.grad).all()
    q1 = Poisson(torch.tensor([0., 2.]), max_count=1)
    q2 = Poisson(torch.tensor([0., 2.]), max_count=128)
    assert torch.equal(q1.entropy(), q2.entropy())


@pytest.mark.parametrize('estimator', ['reinforce', 'obbvi'])
@pytest.mark.parametrize('alpha', [0., .1, .5])
@pytest.mark.parametrize('beta', [.0001, .2, 1.])
def test_hierarchical_score_gradient_matches_enumerated_objective(estimator, alpha, beta):
    # q(z1)*q(z2|z1): downstream entropy depends on BOTH parameters. This
    # catches missing ancestor score terms and missing O-BBVI weighting.
    parameters = torch.tensor([-.3, -.2], dtype=torch.float64, requires_grad=True)
    prior_log_rate = torch.tensor(.1, dtype=torch.float64, requires_grad=True)
    k = torch.arange(30., dtype=torch.float64)
    z1 = k.view(-1, 1).expand(30, 30)
    z2 = k.view(1, -1).expand(30, 30)
    r1 = parameters[0].exp()
    r2 = (parameters[1] + .12 * z1).exp()
    log_q = (torch.distributions.Poisson(r1).log_prob(z1) +
             torch.distributions.Poisson(r2).log_prob(z2))
    log_p = (torch.distributions.Poisson(prior_log_rate.exp()).log_prob(z1) +
             torch.distributions.Poisson(prior_log_rate.exp()).log_prob(z2))
    recon = .03 * (z1 + z2 - 2.) ** 2
    expected = (log_q.exp() * (recon + (beta - alpha) * log_q - beta * log_p)).sum()
    expected_grads = torch.autograd.grad(expected, (parameters, prior_log_rate), retain_graph=True)

    context = dict(log_q=log_q, log_p=log_p)
    if estimator == 'obbvi':
        proposal_logs = []
        for tau in (1., 3.):
            proposal_logs.append(torch.distributions.Poisson(r1 ** (1. / tau)).log_prob(z1) +
                                 torch.distributions.Poisson(r2 ** (1. / tau)).log_prob(z2))
        log_proposal = torch.logsumexp(torch.stack(proposal_logs), 0) - math.log(2.)
        context['log_proposal'] = log_proposal
        sampling_probability = log_proposal.detach().exp()
    else:
        sampling_probability = log_q.detach().exp()
    stub = SimpleNamespace(_gradient_context=context, poisson_gradient_estimator=estimator)
    surrogate, signal, weight = AutoEncoder.score_function_objective(
        stub, recon, beta, .7, entropy_alpha=alpha)
    actual_grads = torch.autograd.grad((sampling_probability * surrogate).sum(),
                                      (parameters, prior_log_rate))
    for actual, expected_grad in zip(actual_grads, expected_grads):
        assert torch.allclose(actual, expected_grad, atol=2e-6, rtol=2e-5)
    assert torch.allclose((sampling_probability * weight * signal).sum(),
                          expected.detach(), atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize('estimator', ['relaxed', 'straight_through', 'reinforce', 'obbvi'])
def test_model_entropy_objective_backward(estimator):
    from test_mixed_model import _poisson_model
    model = _poisson_model(estimator).train()
    model.collect_posterior_entropy = estimator in ('relaxed', 'straight_through')
    x = torch.bernoulli(torch.full((2, 1, 32, 32), .5))
    logits, _, _, kl_all, _ = model(x)
    recon = -model.decoder_output(logits).log_prob(x).sum(dim=[1, 2, 3])
    if model.collect_posterior_entropy:
        h = model.posterior_entropy()
        assert h.shape == (2,) and torch.isfinite(h).all() and (h > 0.).all()
        loss = (recon + sum(kl_all) + .1 * h).mean()
    else:
        loss = model.score_function_objective(recon, 1., 0., entropy_alpha=.1)[0].mean()
    loss.backward()
    for sampler in model.enc_sampler:
        grads = [p.grad for p in sampler.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0. for g in grads)


def test_zero_alpha_preserves_score_surrogate():
    context = dict(log_q=torch.tensor([-2.], requires_grad=True),
                   log_p=torch.tensor([-3.], requires_grad=True))
    model = SimpleNamespace(_gradient_context=context, poisson_gradient_estimator='reinforce')
    a = AutoEncoder.score_function_objective(model, torch.tensor([4.]), 1., 0.)
    b = AutoEncoder.score_function_objective(model, torch.tensor([4.]), 1., 0., entropy_alpha=0.)
    assert all(torch.equal(left, right) for left, right in zip(a, b))


@pytest.mark.parametrize('estimator', ['relaxed', 'straight_through', 'reinforce', 'obbvi'])
@pytest.mark.parametrize('step', [0, 99, 499, 1099])
@pytest.mark.parametrize('objective', ['elbo_alpha', 'nvae_alpha'])
def test_training_loop_kl_policy_and_entropy(monkeypatch, tmp_path, estimator, step, objective):
    # Load the actual training function without importing GPU/data/FID-only
    # entrypoint dependencies. Only CUDA transport and norm regularizers are
    # replaced; model, objective, optimizer and backward remain real.
    import ast
    import contextlib
    from pathlib import Path
    import numpy as np
    import utils
    from rate_monitor import gather_group1_rates, log_rate_distribution
    from test_mixed_model import _poisson_model

    tree = ast.parse((Path(__file__).resolve().parents[1] / 'train.py').read_text())
    tree.body = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == 'train']
    args = SimpleNamespace(poisson_objective=objective, num_x_bits=8,
                           learning_rate=.001, distributed=False,
                           kl_anneal_portion=.3, kl_const_portion=.0001,
                           kl_const_coeff=.0001, num_total_iter=2001,
                           poisson_entropy_alpha_init=.1,
                           poisson_entropy_anneal_portion=.5,
                           save=str(tmp_path),
                           weight_decay_norm_anneal=False, weight_decay_norm=0.)
    namespace = dict(torch=torch, np=np, utils=utils, args=args,
                     gather_group1_rates=gather_group1_rates,
                     log_rate_distribution=log_rate_distribution,
                     entropy_alpha=entropy_alpha, autocast=contextlib.nullcontext)
    exec(compile(tree, 'train.py', 'exec'), namespace)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **kw: self)
    model = _poisson_model(estimator)
    monkeypatch.setattr(model, 'spectral_norm_parallel', lambda: torch.tensor(0.))
    monkeypatch.setattr(model, 'batchnorm_loss', lambda: torch.tensor(0.))
    optimizer = torch.optim.SGD(model.parameters(), lr=.00001)
    scaler = SimpleNamespace(scale=lambda loss: loss, step=lambda opt: opt.step(), update=lambda: None)
    logged = {}
    histograms = []
    writer = SimpleNamespace(add_scalar=lambda name, value, s: logged.update({
        name: float(value.detach() if isinstance(value, torch.Tensor) else value)}),
        add_histogram=lambda *a, **kw: histograms.append((a, kw)))
    logger = SimpleNamespace(info=lambda *a: None)
    x = torch.bernoulli(torch.full((2, 1, 32, 32), .5))
    original_balancer = utils.kl_balancer
    balance_calls = []

    def record_balance(*a, **kw):
        result = original_balancer(*a, **kw)
        balance_calls.append((kw.get('kl_balance'), result[1]))
        return result

    monkeypatch.setattr(utils, 'kl_balancer', record_balance)
    _, end_step = namespace['train']([(x, None)], model, optimizer, scaler, step, 0, writer, logger)
    assert end_step == step + 1
    assert list(tmp_path.iterdir()) == []
    if step in (0, 499):
        assert len(histograms) == 1
        assert histograms[0][0][2] == step + 1
        np.testing.assert_array_equal(histograms[0][0][1],
                                      model.group1_posterior_rates.numpy().reshape(-1))
    else:
        assert histograms == []
    if step == 0:
        assert logged['posterior_rate/group_001/num_rates'] == model.group1_posterior_rates.numel()
        return  # The parent's loss diagnostics are written only every 100 steps.
    expected_beta = (1. if objective == 'elbo_alpha' else
                     utils.kl_coeff(step, .3 * 2001, .0001 * 2001, .0001))
    assert logged['kl_coeff/coeff'] == expected_beta
    expect_balance = objective == 'nvae_alpha' and estimator in ('relaxed', 'straight_through')
    assert balance_calls[0][0] == expect_balance
    if expect_balance and expected_beta < 1.:
        assert not torch.equal(balance_calls[0][1], torch.ones_like(balance_calls[0][1]))
    assert logged['train/entropy_alpha'] == entropy_alpha(step, 2001)
    assert logged['train/negative_elbo_alpha'] == pytest.approx(
        logged['train/nelbo_iter'] + logged['train/entropy_penalty_nats'], rel=2e-5)
    if step > 1000:
        assert logged['train/entropy_penalty_nats'] == 0.
        assert not model.collect_posterior_entropy
    else:
        assert logged['train/entropy_penalty_nats'] > 0.


@pytest.mark.parametrize('estimator', ['relaxed', 'straight_through', 'reinforce', 'obbvi'])
@pytest.mark.parametrize('step', [99, 1099])
def test_zero_entropy_recovers_parent_loss_and_gradients(monkeypatch, estimator, step):
    import ast
    import contextlib
    from pathlib import Path
    import numpy as np
    import utils
    from test_mixed_model import _poisson_model

    tree = ast.parse((Path(__file__).resolve().parents[1] / 'train.py').read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train']
    args = SimpleNamespace(poisson_objective='nvae', num_x_bits=8,
                           learning_rate=.001, distributed=False,
                           kl_anneal_portion=.3, kl_const_portion=.0001,
                           kl_const_coeff=.0001, num_total_iter=2001,
                           poisson_entropy_alpha_init=0., poisson_entropy_anneal_portion=.5,
                           poisson_rate_monitor_every=0,
                           weight_decay_norm_anneal=True, weight_decay_norm=.01,
                           weight_decay_norm_init=10.)
    namespace = dict(torch=torch, np=np, utils=utils, args=args,
                     entropy_alpha=entropy_alpha, autocast=contextlib.nullcontext)
    exec(compile(tree, 'train.py', 'exec'), namespace)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **kw: self)
    model = _poisson_model(estimator)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    monkeypatch.setattr(model, 'spectral_norm_parallel', lambda: torch.tensor(2.))
    monkeypatch.setattr(model, 'batchnorm_loss', lambda: torch.tensor(3.))
    x = torch.bernoulli(torch.full((2, 1, 32, 32), .5))
    runs = []
    for objective in ('nvae', 'nvae_alpha'):
        args.poisson_objective = objective
        model.load_state_dict(initial)
        model._score_baseline_value = None
        optimizer = torch.optim.SGD(model.parameters(), lr=.00001)
        scaler = SimpleNamespace(scale=lambda loss: loss, step=lambda opt: opt.step(), update=lambda: None)
        logged = {}
        writer = SimpleNamespace(add_scalar=lambda name, value, s: logged.update({
            name: float(value.detach() if isinstance(value, torch.Tensor) else value)}))
        with torch.random.fork_rng():
            torch.manual_seed(812)
            namespace['train']([(x, None)], model, optimizer, scaler, step, 0,
                               writer, SimpleNamespace(info=lambda *a: None))
        runs.append((logged, {k: v.clone() for k, v in model.state_dict().items()},
                     {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}))
    for key in ('train/negative_elbo_alpha', 'train/nelbo_iter', 'kl_coeff/coeff', 'train/norm_coeff'):
        assert runs[0][0][key] == runs[1][0][key]
    for section in (1, 2):
        assert runs[0][section].keys() == runs[1][section].keys()
        for key in runs[0][section]:
            assert torch.equal(runs[0][section][key], runs[1][section][key]), key

