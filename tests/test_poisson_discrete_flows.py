import itertools
import math

import pytest
import torch

from discrete_flows import adjacent_swap, PoissonSwapCellAR, PairedPoissonSwapAR
from distributions import Poisson
from model import AutoEncoder
from test_mixed_model import _args
import utils


@pytest.mark.parametrize('offset', [0, 1])
@pytest.mark.parametrize('gate_value', [0, 1])
def test_scalar_map_is_an_involution_on_unbounded_counts(offset, gate_value):
    # Odd offset explicitly fixes zero; values far above max_count remain valid.
    z = torch.tensor(list(range(100)) + [1000000, 1000001], dtype=torch.float64)
    gate = torch.full_like(z, gate_value)
    output = adjacent_swap(z, gate, offset, validate=True)
    assert (output >= 0).all()
    assert torch.equal(output, output.floor())
    assert torch.unique(output).numel() == output.numel()
    assert torch.equal(adjacent_swap(output, gate, offset), z)
    assert (output - z).abs().max() <= 1
    if offset == 1:
        assert output[0] == 0


def test_invalid_counts_and_nonbinary_gates_are_rejected():
    for value in [-1., 0.25, float('inf'), float('nan')]:
        with pytest.raises(ValueError, match='nonnegative integer'):
            adjacent_swap(torch.tensor([value]), torch.ones(1), validate=True)
    with pytest.raises(ValueError, match='binary'):
        adjacent_swap(torch.ones(1), torch.tensor([0.5]), validate=True)


@pytest.mark.parametrize('mirror', [False, True])
def test_neural_gate_has_strict_autoregressive_dependency(mirror):
    torch.manual_seed(4)
    cell = PoissonSwapCellAR(2, 3, mirror=mirror).double()
    z = torch.rand(1, 2, 2, 2, dtype=torch.float64, requires_grad=True)
    ftr = torch.randn(1, 3, 2, 2, dtype=torch.float64)
    jac = torch.autograd.functional.jacobian(lambda v: cell.gate_logits(v, ftr), z)
    jac = jac.reshape(8, 8)
    rows = list(range(2))
    cols = list(range(2))
    if mirror:
        rows.reverse()
        cols.reverse()
    order = [channel * 4 + row * 2 + col
             for row in rows for col in cols for channel in range(2)]
    ordered = jac[order][:, order]
    assert torch.equal(torch.triu(ordered), torch.zeros_like(ordered))
    assert torch.tril(ordered, diagonal=-1).abs().sum() > 0


@pytest.mark.parametrize('mirror,offset', [(False, 0), (True, 1)])
def test_neural_cell_round_trip_and_st_gradients(mirror, offset):
    torch.manual_seed(5)
    cell = PoissonSwapCellAR(2, 3, mirror=mirror, offset=offset).double().train()
    # Force nonidentity transformations rather than only checking initialization.
    cell.gate.conv_0.bias.data.fill_(0.5)
    z = torch.randint(0, 20, (3, 2, 2, 2)).double().requires_grad_()
    ftr = torch.randn(3, 3, 2, 2, dtype=torch.float64, requires_grad=True)
    output, correction = cell(z, ftr)
    assert torch.equal(output, output.floor())
    assert (output >= 0).all()
    assert torch.equal(correction, torch.zeros_like(output))
    assert not torch.equal(output, z)
    assert torch.equal(cell.inverse(output, ftr), z)
    assert torch.equal(cell(cell.inverse(z, ftr), ftr)[0], z)
    weights = torch.arange(1, output.numel() + 1).reshape_as(output).double()
    (output * weights).sum().backward()
    assert torch.isfinite(z.grad).all()
    assert torch.isfinite(ftr.grad).all()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in cell.parameters())


def _nontrivial_flow():
    flow = PairedPoissonSwapAR(2, 1).double().eval()
    for parameter in flow.parameters():
        parameter.data.fill_(0.2)
    for cell in [flow.cell1, flow.cell2]:
        for module in cell.modules():
            if hasattr(module, 'log_weight_norm'):
                module.log_weight_norm.data.zero_()
        cell.gate.conv_0.bias.data.fill_(-0.1)
    return flow


def test_enumerated_joint_pmf_inverse_kl_and_importance_identity():
    flow = _nontrivial_flow()
    counts = torch.tensor(list(itertools.product(range(25), repeat=2)), dtype=torch.float64)
    base = counts.reshape(-1, 2, 1, 1)
    ftr = torch.zeros(len(base), 1, 1, 1, dtype=torch.float64)
    output, correction = flow(base, ftr)
    assert torch.unique(output.reshape(-1, 2), dim=0).shape[0] == len(base)
    restored = flow.inverse(output, ftr)
    assert torch.equal(restored, base)
    assert torch.equal(flow(flow.inverse(base, ftr), ftr)[0], base)
    assert not torch.equal(output, base)
    assert torch.equal(correction, torch.zeros_like(correction))

    q = torch.distributions.Poisson(torch.tensor([0.7, 1.4], dtype=torch.float64))
    p = torch.distributions.Poisson(torch.tensor([1.1, 1.8], dtype=torch.float64))
    log_q_base = q.log_prob(counts).sum(-1)
    log_q_flow = q.log_prob(restored.reshape(-1, 2)).sum(-1)
    assert torch.equal(log_q_base, log_q_flow)
    mass = log_q_flow.exp()
    assert abs(mass.sum().item() - 1.) < 1e-12
    flat_output = output.reshape(-1, 2)
    mean = (mass[:, None] * flat_output).sum(0)
    covariance = (mass * (flat_output[:, 0] - mean[0]) * (flat_output[:, 1] - mean[1])).sum()
    assert abs(covariance.item()) > 1e-3  # dependent output from independent base

    log_p = p.log_prob(flat_output).sum(-1)
    direct_kl = (mass * (log_q_flow - log_p)).sum()
    base_space_kl = (log_q_base.exp() * (log_q_base - log_p)).sum()
    assert torch.allclose(direct_kl, base_space_kl, atol=1e-12, rtol=0)
    assert direct_kl >= 0
    # Bijective images cover virtually all p mass at this numerical cutoff.
    assert abs((mass * (log_p - log_q_flow).exp()).sum().item() - 1.) < 1e-12


def test_strict_poisson_log_p_has_zero_mass_outside_support():
    q = Poisson(torch.zeros(1))
    values = torch.tensor([-1., -0.2, 1.4, float('nan'), float('inf'), 0., 2.])
    scores = q.log_p(values, strict=True)
    assert (torch.isinf(scores[:5]) & (scores[:5] < 0)).all()
    assert torch.isfinite(scores[5:]).all()
    expected = torch.distributions.Poisson(q.rate).log_prob(values[5:])
    assert torch.equal(scores[5:], expected)
    assert torch.isfinite(q.log_p(torch.tensor([1.4]))).all()  # old relaxed extension


def test_tail_completed_st_counts_follow_poisson_even_with_tiny_arrival_budget():
    torch.manual_seed(12)
    log_rate = torch.full((50000,), math.log(10.), requires_grad=True)
    q = Poisson(log_rate, max_count=2, max_rate=30.)
    z, _ = q.sample(estimator='straight_through', exact_forward=True)
    rate = q.rate[0].item()
    assert torch.equal(z, z.round())
    assert z.max() > q.max_count
    assert abs(z.mean().item() - rate) < 0.08
    assert abs(z.var().item() - rate) < 0.25
    expected_pmf = torch.distributions.Poisson(q.rate[0].detach()).log_prob(torch.arange(20.)).exp()
    empirical = torch.bincount(z.detach().long(), minlength=20)[:20].float() / len(z)
    assert (expected_pmf - empirical).abs().max() < 0.006
    z.mean().backward()
    assert torch.isfinite(log_rate.grad).all()


def _flow_model_args(dataset='mnist', num_nf=1):
    args = _args()
    args.dataset = dataset
    args.latent_distribution = 'poisson'
    args.poisson_gradient_estimator = 'straight_through'
    args.num_nf = num_nf
    if dataset != 'mnist':
        args.num_mixture_dec = 10
        args.num_preprocess_blocks = 1
        args.num_postprocess_blocks = 1
        args.num_preprocess_cells = 2
        args.num_postprocess_cells = 2
        args.num_cell_per_cond_enc = 2
        args.num_cell_per_cond_dec = 2
        if dataset == 'cifar10':
            args.num_latent_scales = 1
            args.num_groups_per_scale = 30
            args.ada_groups = False
        else:
            args.num_latent_scales = 3
            args.num_groups_per_scale = 20
            args.min_groups_per_scale = 1
    return args


def _new_model(args):
    model = AutoEncoder(args, None, utils.get_arch_cells('res_elu'))
    for module in model.modules():
        if hasattr(module, 'init_done'):
            module.init_done = True
    return model


@pytest.mark.parametrize('dataset,num_nf', [('mnist', 1), ('cifar10', 2), ('celeba_64', 1)])
def test_full_model_flow_density_gradient_generation_and_checkpoint(dataset, num_nf):
    torch.manual_seed(20)
    model = _new_model(_flow_model_args(dataset, num_nf)).train()
    seen = []
    handles = [cell.register_forward_hook(lambda module, inputs, output: seen.append(output[0].detach()))
               for cell in model.nf_cells]
    channels = 1 if dataset == 'mnist' else 3
    size = 64 if dataset == 'celeba_64' else 32
    x = torch.rand(2, channels, size, size)
    if dataset == 'mnist':
        x = torch.bernoulli(x)
    logits, log_q, log_p, kl_all, _ = model(x)
    assert len(seen) == sum(model.groups_per_scale) * num_nf
    if dataset == 'cifar10':
        assert model.groups_per_scale == [30]
    if dataset == 'celeba_64':
        assert model.groups_per_scale == [20, 10, 5]
    assert all((z >= 0).all() and torch.equal(z, z.round()) for z in seen)
    assert torch.isfinite(log_q).all() and torch.isfinite(log_p).all()
    assert torch.allclose(torch.stack(kl_all).sum(0), log_q - log_p, atol=0.01, rtol=1e-5)
    recon = utils.reconstruction_loss(model.decoder_output(logits), x, crop=model.crop_output)
    (recon + torch.stack(kl_all).sum(0)).mean().backward()
    flow_grads = [p.grad for p in model.nf_cells.parameters() if p.grad is not None]
    assert flow_grads and all(torch.isfinite(g).all() for g in flow_grads)
    assert any(g.abs().sum() > 0 for g in flow_grads)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    for handle in handles:
        handle.remove()
    model.eval()
    restored = _new_model(_flow_model_args(dataset, num_nf)).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    assert not any(name.endswith('.mask') for name in model.state_dict())
    with torch.no_grad():
        torch.manual_seed(21)
        original = model(x)
        torch.manual_seed(21)
        copy = restored(x)
        assert torch.equal(original[0], copy[0])
        assert torch.equal(original[1], copy[1])
        # Generative sampling uses the Poisson prior, not posterior flows.
        def forbid_posterior_flow(module, inputs):
            raise AssertionError('Posterior flow entered generative path.')
        handles = [cell.register_forward_pre_hook(forbid_posterior_flow) for cell in model.nf_cells]
        assert torch.isfinite(model.sample(2, 1.)).all()
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize('estimator', ['relaxed', 'reinforce', 'obbvi'])
def test_unsupported_flow_estimators_fail_explicitly(estimator):
    args = _flow_model_args()
    args.poisson_gradient_estimator = estimator
    with pytest.raises(ValueError, match='require straight_through'):
        _new_model(args)


def test_mixed_flow_still_rejected_and_gaussian_flow_still_works():
    args = _args()
    args.num_nf = 1
    with pytest.raises(ValueError, match='Mixed Poisson/Gamma'):
        _new_model(args)
    args.latent_distribution = 'normal'
    model = _new_model(args).eval()
    with torch.no_grad():
        assert torch.isfinite(model(torch.rand(2, 1, 32, 32))[0]).all()


@pytest.mark.skipif(not hasattr(torch, 'autocast'), reason='CPU autocast unavailable in old torch')
def test_mixed_precision_forward_remains_exactly_integer():
    flow = PairedPoissonSwapAR(2, 3).train()
    for cell in [flow.cell1, flow.cell2]:
        cell.gate.conv_0.bias.data.fill_(0.5)
    counts = torch.tensor([0., 1., 2., 3., 128., 129., 254., 255.]).reshape(1, 2, 2, 2)
    ftr = torch.zeros(1, 3, 2, 2)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        output, _ = flow(counts.to(torch.bfloat16), ftr)
    assert output.dtype == torch.float32
    assert torch.equal(output, output.floor())
    assert (output >= 0).all()
    assert not torch.equal(output, counts)
    output.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in flow.parameters())
