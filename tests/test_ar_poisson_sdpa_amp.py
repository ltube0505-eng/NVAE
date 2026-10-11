"""Regression checks for SDPA masks, ST gradients and AMP precision boundaries."""
import copy
import math

import pytest
import torch
from torch.nn import functional as F

from recognition_ar_poisson import MHA, ARPoissonGroup, IndependentPoissonGroup


@pytest.mark.parametrize('layout', ['encoder', 'causal', 'cached'])
def test_sdpa_matches_original_attention_outputs_and_gradients(layout):
    torch.manual_seed(21)
    actual = MHA(16, 4).train()
    reference = copy.deepcopy(actual)
    query = torch.randn(2, 1 if layout == 'cached' else 5, 16, requires_grad=True)
    memory = torch.randn(2, 5, 16, requires_grad=True)
    qr = query.detach().clone().requires_grad_()
    mr = memory.detach().clone().requires_grad_()
    allowed = (torch.ones(5, 5, dtype=torch.bool).tril()[None, None]
               if layout == 'causal' else None)
    output = actual(query, memory, allowed)
    q = reference.split(reference.q(qr))
    k, v = reference.project_kv(mr)
    score = q @ k.transpose(-2, -1) / math.sqrt(reference.head_dim)
    if allowed is not None:
        score = score.masked_fill(~allowed, float('-inf'))
    weights = F.softmax(score.float(), -1).to(v.dtype)
    h = (weights @ v).transpose(1, 2).contiguous().reshape(qr.shape)
    expected = reference.out(h)
    torch.testing.assert_close(output, expected, atol=2e-6, rtol=2e-5)
    upstream = torch.randn_like(output)
    (output * upstream).sum().backward()
    (expected * upstream).sum().backward()
    for a, b in [(query.grad, qr.grad), (memory.grad, mr.grad)]:
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    for a, b in zip(actual.parameters(), reference.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=3e-6, rtol=3e-5)


def test_sdpa_dropout_disabled_in_eval_and_cached_mask(monkeypatch):
    original = F.scaled_dot_product_attention
    calls = []
    def capture(q, k, v, **kwargs):
        calls.append(kwargs)
        return original(q, k, v, **kwargs)
    monkeypatch.setattr(F, 'scaled_dot_product_attention', capture)
    layer = MHA(8, 2, attention_dropout=0.25).eval()
    x = torch.randn(2, 1, 8)
    _, cache = layer.step(x, None)
    a, _ = layer.step(x, cache)
    b, _ = layer.step(x, cache)
    assert torch.equal(a, b)
    assert all(c['dropout_p'] == 0.0 and c['attn_mask'] is None
               and c['is_causal'] is False for c in calls)
    layer.train()
    layer.step(x, cache)
    assert calls[-1]['dropout_p'] == 0.25


@pytest.mark.parametrize('device,dtype', [('cpu', torch.bfloat16), ('cuda', torch.float16)])
@pytest.mark.parametrize('mode', ['capped64', 'exact64'])
def test_recognition_amp_compute_with_fp32_sampling_and_prefix_gradients(device, dtype, mode, monkeypatch):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    torch.manual_seed(22)
    ar = ARPoissonGroup((4, 2, 2), (1, 2, 2), embed_dim=16,
                        num_heads=4, num_layers=2).to(device).train()
    nar = IndependentPoissonGroup(4, (1, 2, 2)).to(device).train()
    feature = torch.randn(2, 4, 2, 2, device=device, requires_grad=True)
    raw = torch.randn(2, 1, 2, 2, device=device, requires_grad=True)
    calls, head_dtypes = [], []
    original = F.scaled_dot_product_attention
    def capture(q, k, v, **kwargs):
        calls.append((q.dtype, k.dtype, v.dtype))
        return original(q, k, v, **kwargs)
    monkeypatch.setattr(F, 'scaled_dot_product_attention', capture)
    hooks = [ar.rec.head.register_forward_hook(
        lambda module, inputs, output: head_dtypes.append(output.dtype)),
        nar.head[-1].register_forward_hook(
        lambda module, inputs, output: head_dtypes.append(output.dtype))]
    with torch.autocast(device_type=device, dtype=dtype):
        a = ar(feature, raw)
        n = nar(feature, raw, mode=mode)
        # Cached/full causal equality within mixed-precision tolerance.
        memory = ar.rec.encode_image(feature)
        out = ar.rec.sample_memory(memory, tau=0.7)
        full = ar.rec.teacher_logits_memory(memory, out['y_st'])
    for hook in hooks:
        hook.remove()
    assert calls and all(c == (dtype, dtype, dtype) for c in calls)
    assert head_dtypes and all(d == dtype for d in head_dtypes)
    torch.testing.assert_close(out['logits'], full, atol=0.04, rtol=0.04)
    for result in (a, n):
        for key in ('z_st', 'kl', 'log_q', 'log_p'):
            assert result[key].dtype == torch.float32
            assert torch.isfinite(result[key]).all()
        assert torch.equal(result['z_st'], result['z_hard'].float())
    for key in ('logits', 'y_st', 'log_probs', 'log_q_i'):
        assert a['categorical'][key].dtype == torch.float32
    prefix_grad = torch.autograd.grad(out['logits'][:, -1, 1].sum(),
                                     out['self_kv'][0][0], retain_graph=True)[0]
    assert torch.isfinite(prefix_grad).all() and prefix_grad[:, :, 1:-1].abs().sum() > 0
    (a['kl'].mean() + n['kl'].mean() + out['logits'][:, -1, 1].sum()).backward()
    for parameters in (ar.parameters(), nar.parameters()):
        grads = [p.grad for p in parameters if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)
    assert torch.isfinite(feature.grad).all() and feature.grad.abs().sum() > 0
    assert torch.isfinite(raw.grad).all() and raw.grad.abs().sum() > 0


@pytest.mark.parametrize('mode', ['capped64', 'exact64'])
def test_full_hierarchy_cpu_amp_forward_backward(mode):
    from test_ar_poisson import model_args, new_model
    import utils
    torch.manual_seed(23)
    model = new_model(model_args(mode=mode)).train()
    x = torch.bernoulli(torch.rand(2, 1, 32, 32))
    with torch.autocast('cpu', dtype=torch.bfloat16):
        logits, log_q, log_p, kl_all, _ = model(x)
        recon = utils.reconstruction_loss(model.decoder_output(logits), x, crop=True)
        loss = (recon + sum(kl_all)).mean()
    assert log_q.dtype == log_p.dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()
    for head in model.enc_sampler:
        grads = [p.grad for p in head.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
