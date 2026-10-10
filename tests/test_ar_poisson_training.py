from types import SimpleNamespace

import pytest
import torch

import utils
from test_ar_poisson import model_args, new_model


@pytest.mark.parametrize('mode', ['capped64', 'exact64'])
@pytest.mark.parametrize('objective', ['standard', 'nvae'])
def test_actual_training_and_iw_validation_loop(mode, objective, monkeypatch):
    pytest.importorskip('torchvision')
    pytest.importorskip('lmdb')
    import train

    torch.manual_seed(9)
    args = model_args(mode=mode)
    args.ar_poisson_objective = objective
    args.distributed = False
    args.num_total_iter = 1000
    args.kl_anneal_portion = 0.3
    args.kl_const_portion = 0.01
    args.kl_const_coeff = 0.01
    args.weight_decay_norm_anneal = False
    args.weight_decay_norm = 0.01
    args.learning_rate = 1e-4
    monkeypatch.setattr(train, 'args', args, raising=False)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **kw: self)
    model = new_model(args)
    if objective == 'standard':
        def forbid():
            raise AssertionError('Standard NELBO must not evaluate norm or BN penalties.')
        monkeypatch.setattr(model, 'spectral_norm_parallel', forbid)
        monkeypatch.setattr(model, 'batchnorm_loss', forbid)
    else:
        # Exercise the option and metric separation without CUDA power iteration.
        monkeypatch.setattr(model, 'spectral_norm_parallel', lambda: torch.tensor(2.))
        monkeypatch.setattr(model, 'batchnorm_loss', lambda: torch.tensor(3.))
    recorded, balances, scalars = [], [], {}
    x = torch.bernoulli(torch.rand(2, 1, 32, 32))
    def capture(module, inputs, output):
        recon = utils.reconstruction_loss(model.decoder_output(output[0]), inputs[0], crop=model.crop_output)
        recorded.append((recon + sum(output[3])).detach().mean())
    handle = model.register_forward_hook(capture)
    original_balancer = utils.kl_balancer
    def balance(*a, **kw):
        balances.append(kw)
        return original_balancer(*a, **kw)
    monkeypatch.setattr(utils, 'kl_balancer', balance)
    writer = SimpleNamespace(add_scalar=lambda name, value, step: scalars.update({name: value}))
    logger = SimpleNamespace(info=lambda *a: None)
    optimizer = torch.optim.Adamax(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler('cpu', enabled=False)
    old = model.top_prior.raw.detach().clone()
    nelbo, step = train.train([(x, torch.zeros(2))], model, optimizer, scaler,
                              99, 0, writer, logger)
    assert len(recorded) == args.ar_poisson_mc_samples and step == 100
    assert torch.allclose(nelbo, torch.stack(recorded).mean())
    assert torch.allclose(scalars['train/nelbo_iter'], nelbo)
    assert balances[0]['kl_balance'] == (objective == 'nvae')
    assert not torch.equal(old, model.top_prior.raw)
    if objective == 'standard':
        assert torch.allclose(scalars['train/objective_iter'], nelbo)
        assert scalars['kl_coeff/coeff'] == 1.
    else:
        assert not torch.allclose(scalars['train/objective_iter'], nelbo)
        assert scalars['kl_coeff/coeff'] < 1.
    handle.remove()
    weights = {n: p.detach().clone() for n, p in model.named_parameters()}
    iw_nll, valid_nelbo = train.test([(x, torch.zeros(2))], model, num_samples=3,
                                    args=args, logging=logger)
    assert torch.isfinite(iw_nll) and torch.isfinite(valid_nelbo)
    assert all(torch.equal(p, weights[n]) for n, p in model.named_parameters())
