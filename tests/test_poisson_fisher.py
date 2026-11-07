import copy

import pytest
import torch

from poisson_fisher import PoissonDiagonalFisher


def log_rate(raw):
    return 5 * torch.tanh(raw / 5)


def test_scalar_fisher_and_step_match_analytic_tanh_metric():
    raw = torch.nn.Parameter(torch.tensor([[-4., 0., 4.]], dtype=torch.double))
    opt = PoissonDiagonalFisher([raw], lr=0.01, damping=0.01, step_budget=10.)
    eta = log_rate(raw)
    rate = eta.exp()
    expected_fisher = rate.detach() * (1 - torch.tanh(raw.detach() / 5).pow(2)).pow(2)
    before = raw.detach().clone()
    rng = torch.random.get_rng_state()
    opt.estimate_curvature([eta])
    assert raw.grad is None
    assert torch.equal(rng, torch.random.get_rng_state())
    loss = (rate * eta - rate + 1).sum()  # analytic KL to Poisson(1)
    loss.backward()
    expected = before - 0.01 * raw.grad / (expected_fisher + 0.01)
    opt.step()
    assert torch.allclose(opt.state[raw]['fisher'], expected_fisher)
    assert torch.allclose(raw, expected)


def test_shared_parameter_fisher_is_batch_mean_not_squared_mean():
    torch.manual_seed(4)
    raw = torch.nn.Parameter(torch.tensor(0.4, dtype=torch.double))
    inputs = torch.tensor([[-2.], [1.], [3.]], dtype=torch.double)
    eta = log_rate(raw * inputs)
    opt = PoissonDiagonalFisher([raw], probes=4000)
    expected = (eta.detach().exp() *
                (inputs * (1 - torch.tanh(raw.detach() * inputs / 5).pow(2))).pow(2)).mean()
    opt.estimate_curvature([eta])
    assert torch.allclose(opt._pending[0][1], expected, rtol=0.06)


def test_budget_bounds_actual_step_and_decoder_uses_identity():
    raw = torch.nn.Parameter(torch.tensor([[0.]], dtype=torch.double))
    decoder = torch.nn.Parameter(torch.tensor([1.], dtype=torch.double))
    opt = PoissonDiagonalFisher([raw, decoder], lr=100., step_budget=0.001)
    eta = log_rate(raw)
    opt.estimate_curvature([eta])
    (eta.exp().sum() + decoder.sum()).backward()
    before = [p.detach().clone() for p in (raw, decoder)]
    opt.step()
    quadratic = sum((0.5 * (opt.state[p]['fisher'] + 0.01) * (p - old).pow(2)).sum()
                    for p, old in zip((raw, decoder), before))
    assert quadratic.item() <= 0.001 * (1 + 1e-10)
    assert opt.last_stats['step_scale'] < 1
    assert opt.state[decoder]['fisher'].item() == 1


def test_resume_continues_ema_and_rejects_adamax_state():
    raw = torch.nn.Parameter(torch.tensor([[0.2]], dtype=torch.double))
    opt = PoissonDiagonalFisher([raw])
    def update(parameter, optimizer):
        eta = log_rate(parameter)
        optimizer.estimate_curvature([eta])
        optimizer.zero_grad()
        eta.exp().sum().backward()
        optimizer.step()
    update(raw, opt)
    clone = torch.nn.Parameter(raw.detach().clone())
    restored = PoissonDiagonalFisher([clone])
    restored.load_state_dict(copy.deepcopy(opt.state_dict()))
    update(raw, opt)
    update(clone, restored)
    assert torch.equal(raw, clone)
    assert torch.equal(opt.state[raw]['fisher'], restored.state[clone]['fisher'])
    with pytest.raises(ValueError, match='reset_optimizer'):
        restored.load_state_dict(torch.optim.Adamax([clone]).state_dict())


def test_nonfinite_gradient_does_not_change_parameters_or_fisher():
    raw = torch.nn.Parameter(torch.tensor([[1.]], dtype=torch.double))
    opt = PoissonDiagonalFisher([raw])
    opt.estimate_curvature([log_rate(raw)])
    raw.grad = torch.full_like(raw, float('nan'))
    with pytest.raises(FloatingPointError):
        opt.step()
    assert raw.item() == 1.
    assert not opt.state


@pytest.mark.parametrize('smooth_ste', [False, True])
def test_poisson_model_forward_backward_and_eval_collection(smooth_ste):
    from test_mixed_model import _poisson_model
    model = _poisson_model()
    model.poisson_smooth_ste = smooth_ste
    model.collect_poisson_fisher = True
    model.train()
    opt = PoissonDiagonalFisher(model.parameters(), lr=1e-4)
    logits, _, _, kls, _ = model(torch.rand(2, 1, 32, 32))
    assert len(model.poisson_fisher_log_rates) == 2 * sum(model.groups_per_scale) - 1
    opt.estimate_curvature(model.poisson_fisher_log_rates)
    (logits.pow(2).mean() + sum(kl.mean() for kl in kls)).backward()
    opt.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    model.eval()
    with torch.no_grad():
        model(torch.rand(2, 1, 32, 32))
    assert model.poisson_fisher_log_rates == []


@pytest.mark.parametrize('smooth_ste', [False, True])
def test_training_entry_uses_fisher_before_backward(monkeypatch, smooth_ste):
    # Execute the real train() body without importing GPU/FID/dataset startup.
    import ast
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import Mock
    import numpy as np
    import utils
    from test_mixed_model import _poisson_model

    tree = ast.parse(Path('train.py').read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'train')
    module = ast.Module(body=[function], type_ignores=[])
    namespace = dict(torch=torch, np=np, utils=utils, autocast=torch.cuda.amp.autocast,
                     PoissonDiagonalFisher=PoissonDiagonalFisher,
                     args=SimpleNamespace(num_x_bits=8, distributed=False,
                                          learning_rate=1e-3, kl_anneal_portion=0.3,
                                          num_total_iter=10, kl_const_portion=1e-4,
                                          kl_const_coeff=1e-4, weight_decay_norm_anneal=False,
                                          weight_decay_norm=0.))
    exec(compile(module, 'train.py', 'exec'), namespace)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda tensor, *args, **kwargs: tensor)
    model = _poisson_model()
    model.poisson_smooth_ste = smooth_ste
    model.collect_poisson_fisher = True
    monkeypatch.setattr(model, 'spectral_norm_parallel', lambda: torch.tensor(0.))
    monkeypatch.setattr(model, 'batchnorm_loss', lambda: torch.tensor(0.))
    opt = PoissonDiagonalFisher(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    value, step = namespace['train']([(torch.empty(2, 1, 32, 32).bernoulli_(0.5), torch.zeros(2))],
                                     model, opt, scaler, 1, 0, Mock(), Mock())
    assert step == 2 and torch.isfinite(value)
    assert opt.state and opt.last_stats['quadratic_step'] <= 0.01 * (1 + 1e-10)
    assert model.poisson_fisher_log_rates == []
