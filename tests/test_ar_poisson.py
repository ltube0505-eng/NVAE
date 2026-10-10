import argparse
import math

import pytest
import torch

from model import AutoEncoder
from recognition_ar_poisson import (
    RecognitionScale1, ARPoissonGroup, IndependentPoissonGroup, ar_group_result,
    positive_rate, poisson_log_pmf, poisson_kl, poisson_survival,
    capped_poisson_kl, capped_poisson_log_q, cts_st, mixed_nelbo)
import utils


def model_args(dataset='mnist', mode='capped64', vanilla=False):
    return argparse.Namespace(
        dataset=dataset, use_se=False, res_dist=True, num_x_bits=8,
        latent_distribution='ar_poisson', poisson_gradient_estimator='straight_through',
        poisson_max_count=64, poisson_relaxation_temperature=0.1,
        ar_poisson_cts_mode=mode, ar_embed_dim=8, ar_num_heads=2, ar_num_layers=1,
        ar_poisson_mc_samples=2,
        num_latent_scales=1 if vanilla else 2, num_groups_per_scale=1 if vanilla else 3,
        num_latent_per_group=1, ada_groups=False, min_groups_per_scale=1,
        num_channels_enc=4, num_channels_dec=4, num_preprocess_blocks=2,
        num_preprocess_cells=1, num_cell_per_cond_enc=1, num_postprocess_blocks=2,
        num_postprocess_cells=1, num_cell_per_cond_dec=1,
        num_mixture_dec=1 if dataset == 'mnist' else 10, num_nf=0)


def new_model(args):
    model = AutoEncoder(args, None, utils.get_arch_cells('res_elu'))
    for module in model.modules():
        if hasattr(module, 'init_done'):
            module.init_done = True  # Skip unrelated distributed data initialization.
    return model


def assert_grad(parameters):
    grads = [p.grad for p in parameters if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_patch_layout_and_padding_follow_nhwc():
    rec = RecognitionScale1((3, 5, 7), (2, 2), 4, 8, 2, 1)
    image = torch.arange(105.).reshape(1, 3, 5, 7)
    padded = torch.nn.functional.pad(image, (0, 1, 1, 2))
    expected = torch.stack([
        padded[0, :, h:h+4, w:w+4].permute(1, 2, 0).reshape(-1)
        for h in [0, 4] for w in [0, 4]])
    assert torch.equal(rec.patchify(image)[0], expected)


def test_cached_logits_match_full_causal_logits_and_prefix_gradients():
    torch.manual_seed(1)
    rec = RecognitionScale1((1, 4, 4), (2, 2), 6, 8, 2, 2).train()
    x = torch.randn(2, 1, 4, 4, requires_grad=True)
    memory = rec.encode_image(x)
    out = rec.sample_memory(memory, tau=0.7)
    full = rec.teacher_logits_memory(memory, out['y_st'])
    assert torch.allclose(out['logits'], full, atol=2e-6, rtol=2e-5)
    for k, v in out['self_kv']:
        assert k.shape == v.shape == (2, 2, 6, 4)
        assert k.requires_grad and v.requires_grad
    changed = out['y_st'].detach().clone()
    changed[:, 2:] = changed[:, 2:].roll(1, -1)
    altered = rec.teacher_logits_memory(memory, changed)
    # Output t uses only tokens before t; changing z_2 cannot affect outputs <= 2.
    assert torch.allclose(full[:, :3], altered[:, :3], atol=2e-6, rtol=2e-5)
    first_k = out['self_kv'][0][0]
    assert first_k.grad_fn is not None
    # A later probability must backpropagate through earlier sampled ST embeddings.
    grad_y = torch.autograd.grad(out['logits'][:, -1, 1].sum(),
                                 out['self_kv'][0][0], retain_graph=True)[0]
    assert grad_y[:, :, 1:-1].abs().sum() > 0
    out['logits'][:, -1, 1].sum().backward()
    assert_grad(rec.embedding.parameters())
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_ar_uses_untempered_categorical_and_ordinary_poisson_pmf():
    torch.manual_seed(2)
    rec = RecognitionScale1((1, 4, 4), (2, 2), 4, 8, 2, 1).eval()
    x = torch.randn(2, 1, 4, 4)
    torch.manual_seed(3)
    cold = rec(x, tau=0.1)
    torch.manual_seed(3)
    hot = rec(x, tau=2.)
    torch.manual_seed(3)
    hard = rec(x, tau=None)
    assert torch.equal(cold['z'], hot['z']) and torch.equal(cold['z'], hard['z'])
    assert torch.equal(cold['y_st'], torch.nn.functional.one_hot(cold['z'], 64).float())
    assert torch.allclose(cold['log_q_i'], cold['log_q_i_hard'])
    raw = torch.full((2, 1, 2, 2), math.log(80.), requires_grad=True)
    rate = positive_rate(raw)
    result = ar_group_result(cold, rate, (1, 2, 2))
    expected = torch.distributions.Poisson(rate.flatten(1)).log_prob(cold['z']).sum(-1)
    assert torch.allclose(result['log_p'], expected, atol=1e-5)
    assert torch.allclose(result['kl'], result['log_q'] - result['log_p'])
    result['kl'].mean().backward()
    assert torch.isfinite(raw.grad).all() and raw.grad.abs().sum() > 0


def test_capped_kl_log_q_and_gradients_against_enumerated_mass():
    rq = torch.tensor([0.01, 1., 32., 64., 120.], dtype=torch.double, requires_grad=True)
    rp = torch.tensor([0.3, 2., 28., 75., 100.], dtype=torch.double, requires_grad=True)
    k = torch.arange(512., dtype=torch.double)
    full_q = torch.distributions.Poisson(rq[:, None]).log_prob(k).exp()
    mass = torch.cat([full_q[:, :64], full_q[:, 64:].sum(-1, keepdim=True)], -1)
    log_p = torch.distributions.Poisson(rp[:, None]).log_prob(k[:65])
    expected = (mass * (mass.clamp_min(torch.finfo(torch.double).tiny).log() - log_p)).sum(-1)
    actual = capped_poisson_kl(rq, rp)
    assert torch.allclose(mass.sum(-1), torch.ones(5, dtype=torch.double), atol=2e-13)
    assert torch.allclose(poisson_survival(rq), mass[:, -1], atol=2e-13)
    # gammainc's numerical accuracy near rate=cap is ~1e-9 on some torch builds.
    assert torch.allclose(actual, expected, atol=2e-8, rtol=1e-9)
    assert torch.allclose(capped_poisson_log_q(torch.full_like(rq, 64.), rq), mass[:, -1].log())
    assert torch.autograd.gradcheck(capped_poisson_kl, (rq, rp), eps=1e-5, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize('mode', ['capped64', 'exact64'])
def test_cts_distribution_high_rate_tail_and_st_gradients(mode):
    torch.manual_seed(4)
    rate = torch.full((30000,), 80., requires_grad=True)
    result = cts_st(rate, tau=0.1, mode=mode)
    z = result['z_st']
    assert torch.equal(z, z.round()) and (z >= 0).all()
    if mode == 'capped64':
        assert z.max() == 64
        k = torch.arange(64.)
        expected = (poisson_log_pmf(k, torch.tensor(80.)).exp() * k).sum()
        expected += 64 * poisson_survival(torch.tensor(80.)).float()
        assert abs(z.mean() - expected) < 0.03
    else:
        assert z.max() > 64
        assert abs(z.mean().item() - 80) < 0.2
        assert abs(z.var().item() - 80) < 1.6
    z.mean().backward()
    assert torch.isfinite(rate.grad).all() and rate.grad.abs().sum() > 0
    torch.manual_seed(5)
    training = cts_st(rate.detach(), tau=0.1, mode=mode)['z_hard']
    torch.manual_seed(5)
    evaluation = cts_st(rate.detach(), tau=None, mode=mode)['z_hard']
    assert torch.equal(training, evaluation)


@pytest.mark.parametrize('mode', ['capped64', 'exact64'])
@pytest.mark.parametrize('dataset,vanilla', [('mnist', True), ('mnist', False),
                                           ('cifar10', False), ('celeba_64', False)])
def test_nvae_topology_loss_gradient_prior_generation_and_checkpoint(mode, dataset, vanilla, tmp_path):
    torch.manual_seed(7)
    args = model_args(dataset, mode, vanilla)
    model = new_model(args).train()
    assert len(model.nf_cells) == 0
    assert [model._latent_kind(i) for i in range(len(model.enc_sampler))] == (
        ['ar'] if vanilla else ['ar', 'nar', 'ar', 'nar', 'ar', 'nar'])
    # Scale boundary after 3 groups must continue alternating rather than restart.
    assert all(head[-1].out_channels == 1 for head in model.dec_sampler)
    assert 'top_prior.raw' in model.state_dict()
    size = utils.get_input_size(dataset)
    x = torch.rand(2, 1 if dataset == 'mnist' else 3, size, size)
    if dataset == 'mnist':
        x = torch.bernoulli(x)
    seen = []
    handles = [head.register_forward_hook(lambda module, inputs, output: seen.append(output))
               for head in model.enc_sampler]
    logits, log_q, log_p, kl_all, kl_diag = model(x)
    assert len(kl_all) == len(seen)
    assert torch.allclose(log_q, sum(out['log_q'] for out in seen))
    assert torch.allclose(log_p, sum(out['log_p'] for out in seen))
    for i, out in enumerate(seen):
        assert torch.equal(out['z_st'], out['z_hard'].float())
        assert torch.equal(out['kl'], kl_all[i])
        if out['kind'] == 'ar':
            assert torch.allclose(out['kl'], out['log_q'] - out['log_p'])
        else:
            expected = (capped_poisson_kl if mode == 'capped64' else poisson_kl)(
                out['rate_q'], out['rate_p']).flatten(1).sum(-1)
            assert torch.allclose(out['kl'], expected)
        assert kl_diag[i].shape == (1,)
    if not vanilla:
        # Later-group analytic KL must retain a gradient path into an earlier AR group.
        gradient = torch.autograd.grad(seen[1]['kl'].sum(), model.enc_sampler[0].rec.head.weight,
                                      retain_graph=True)[0]
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    recon = utils.reconstruction_loss(model.decoder_output(logits), x, crop=model.crop_output)
    loss, metrics = mixed_nelbo(-recon, seen)
    assert torch.allclose(loss, (recon + sum(kl_all)).mean())
    assert torch.isfinite(loss) and torch.equal(metrics['nelbo'], loss.detach())
    loss.backward()
    assert_grad([model.top_prior.raw])
    assert_grad(model.image_conditional.parameters())
    for head in model.enc_sampler:
        assert_grad(head.parameters())
    for head in model.dec_sampler:
        assert_grad(head.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    old = model.top_prior.raw.detach().clone()
    optimizer.step()
    assert not torch.equal(old, model.top_prior.raw)
    for handle in handles:
        handle.remove()
    model.eval()
    checkpoint = tmp_path / 'checkpoint.pt'
    torch.save({'args': args, 'state_dict': model.state_dict(), 'arch_instance': model.arch_instance}, checkpoint)
    restored_checkpoint = utils.load_checkpoint(checkpoint)
    restored = new_model(restored_checkpoint['args']).eval()
    restored.load_state_dict(restored_checkpoint['state_dict'], strict=True)
    with torch.no_grad():
        torch.manual_seed(8)
        first = model(x)
        torch.manual_seed(8)
        second = restored(x)
        assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])
        iw = utils.log_iw(model.decoder_output(first[0]), x, first[1], first[2], crop=model.crop_output)
        assert torch.isfinite(iw).all()
        def forbid(module, inputs):
            raise AssertionError('Recognition head entered prior generation.')
        handles = [head.register_forward_pre_hook(forbid) for head in model.enc_sampler]
        # Force a large prior; generation must accept counts beyond 63/64.
        model.top_prior.raw.fill_(20.)
        latent_maps = []
        consumer = model.stem_decoder if vanilla else next(
            cell for cell in model.dec_tower if cell.cell_type == 'combiner_dec')
        handle = consumer.register_forward_pre_hook(
            lambda module, inputs: latent_maps.append(inputs[0 if vanilla else 1].detach()))
        assert torch.isfinite(model.sample(2, 1.)).all()
        assert latent_maps[0].max() > 64
        assert torch.equal(latent_maps[0], latent_maps[0].round())
        handle.remove()
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize('mode', ['capped64', 'exact64'])
def test_group_amp_keeps_exact_counts_and_finite_probabilities(mode):
    ar = ARPoissonGroup((4, 2, 2), (1, 2, 2), embed_dim=8, num_heads=2, num_layers=1)
    nar = IndependentPoissonGroup(4, (1, 2, 2))
    feature = torch.randn(2, 4, 2, 2)
    raw = torch.randn(2, 1, 2, 2, requires_grad=True)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        outs = [ar(feature.bfloat16(), raw), nar(feature.bfloat16(), raw, mode=mode)]
    for out in outs:
        assert out['z_st'].dtype == torch.float32
        assert torch.equal(out['z_st'], out['z_hard'].float())
        assert torch.isfinite(out['log_p']).all() and torch.isfinite(out['kl']).all()
    sum(out['kl'].mean() for out in outs).backward()
    assert_grad(ar.parameters())
    assert_grad(nar.parameters())


def test_adaptive_three_scales_multichannel_order_and_max_size():
    args = model_args()
    args.num_latent_scales = 3
    args.num_groups_per_scale = 4
    args.ada_groups = True
    args.num_latent_per_group = 2
    args.num_channels_dec = 8
    model = new_model(args).train()
    assert model.groups_per_scale == [4, 2, 1]
    assert [head.latent_shape for head in model.enc_sampler] == [
        (2, 2, 2), (2, 4, 4), (2, 4, 4), *[(2, 8, 8)] * 4]
    x = torch.bernoulli(torch.rand(2, 1, 32, 32))
    logits, log_q, log_p, kl_all, _ = model(x)
    loss = (utils.reconstruction_loss(model.decoder_output(logits), x, crop=True) + sum(kl_all)).mean()
    loss.backward()
    assert torch.isfinite(log_q).all() and torch.isfinite(log_p).all()
    assert_grad(model.enc_sampler[0].parameters())
    assert_grad(model.enc_sampler[-1].parameters())
    assert_grad(model.dec_sampler[-1].parameters())
    # Boundary dimension is accepted; larger dimensions are rejected separately.
    head = ARPoissonGroup((4, 2, 2), (16, 8, 8), embed_dim=8, num_heads=2, num_layers=1)
    assert head.rec.d_lat == 1024


@pytest.mark.parametrize('key,value,match', [
    ('num_nf', 1, 'num_nf 0'), ('poisson_max_count', 32, 'max_count 64'),
    ('poisson_gradient_estimator', 'relaxed', 'straight_through'),
    ('ar_poisson_mc_samples', 0, 'positive'), ('ar_gumbel_temperature', 0, 'positive'),
    ('ar_embed_dim', 7, 'even'), ('ar_num_heads', 3, 'divisible'),
    ('ar_num_layers', 0, 'positive'), ('num_latent_per_group', 17, '1024')])
def test_invalid_configuration_rejected(key, value, match):
    args = model_args()
    setattr(args, key, value)
    with pytest.raises(ValueError, match=match):
        new_model(args)
