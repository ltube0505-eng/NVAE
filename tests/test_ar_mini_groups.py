"""Independent mini-group streams and chunk-grown KV cache regression tests."""
import math

import pytest
import torch
from torch.nn import functional as F

from recognition_ar_poisson import RecognitionScale1, ARPoissonGroup, MHA, GrowingKV


@pytest.mark.parametrize('groups', [1, 3, 12])
@pytest.mark.parametrize('grow', [0, 2, 8])
def test_parallel_cached_scoring_matches_teacher_and_drops_padding(groups, grow):
    torch.manual_seed(31)
    rec = RecognitionScale1((1, 2, 2), (1, 1), 10, 8, 2, 2,
                            mini_groups=groups, grow_target_every=grow).train()
    x = torch.randn(2, 1, 2, 2)
    memory = rec.encode_image(x)
    out = rec.sample_memory(memory, tau=0.7)
    teacher = rec.teacher_logits_memory(memory, out['y_st'])
    torch.testing.assert_close(out['logits'], teacher, atol=3e-6, rtol=3e-5)
    assert out['z'].shape == (2, 10)
    assert out['logits'].shape == (2, 10, 64)
    expected = F.log_softmax(teacher, -1).gather(-1, out['z'][..., None]).squeeze(-1)
    torch.testing.assert_close(out['log_q'], expected.sum(-1), atol=3e-6, rtol=3e-5)
    for k, v in out['self_kv']:
        assert k.shape == v.shape == (2 * min(groups, 10), 2, math.ceil(10 / min(groups, 10)), 4)
        assert k.requires_grad and v.requires_grad
    out['log_q'].sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in rec.parameters())
    assert rec.embedding.weight.grad.abs().sum() > 0


def test_balanced_chw_layout_and_cross_memory_batch_order():
    rec = RecognitionScale1((1, 2, 2), (1, 1), 10, 8, 2, 1, mini_groups=3)
    # Flat CHW positions: [0..3], [4..6], [7..9]; each image stays separate.
    flat = torch.arange(20).reshape(2, 10, 1)
    packed = rec._pack_mini(flat).reshape(2, 3, 4)
    assert packed[0].tolist() == [[0, 1, 2, 3], [4, 5, 6, 7], [7, 8, 9, 9]]
    assert packed[1].tolist() == [[10, 11, 12, 13], [14, 15, 16, 17], [17, 18, 19, 19]]
    assert torch.equal(rec._unpack_mini(packed.reshape(6, 4, 1), 2), flat)
    memory = torch.randn(2, 5, 8)
    base_k, base_v = rec.prepare_cross_kv(memory)[0]
    k, v = rec._mini_cross_kv(memory)[0]
    for b in range(2):
        for g in range(3):
            torch.testing.assert_close(k[b * 3 + g], base_k[b])
            torch.testing.assert_close(v[b * 3 + g], base_v[b])


def test_mini_groups_have_no_cross_prefix_value_or_gradient_dependency():
    torch.manual_seed(32)
    rec = RecognitionScale1((1, 2, 2), (1, 1), 10, 8, 2, 2, mini_groups=3).eval()
    memory = rec.encode_image(torch.randn(2, 1, 2, 2))
    y = F.one_hot(torch.randint(64, (2, 10)), 64).float().requires_grad_()
    logits = rec.teacher_logits_memory(memory, y)
    changed = y.detach().clone()
    changed[:, 4:7] = changed[:, 4:7].roll(1, -1)
    altered = rec.teacher_logits_memory(memory, changed)
    assert torch.equal(logits[:, :4], altered[:, :4])
    assert torch.equal(logits[:, 7:], altered[:, 7:])
    assert torch.equal(logits[:, 4], altered[:, 4])  # Independent BOS for stream 1.
    assert not torch.allclose(logits[:, 5:7], altered[:, 5:7])
    gradient = torch.autograd.grad(logits[:, 3, 1].sum(), y)[0]
    assert gradient[:, :3].abs().sum() > 0
    assert torch.count_nonzero(gradient[:, 3:]) == 0


def test_grown_cache_matches_prefix_cache_gradients_and_state_dict():
    torch.manual_seed(33)
    prefix = RecognitionScale1((1, 2, 2), (1, 1), 14, 8, 2, 2, mini_groups=3)
    grown = RecognitionScale1((1, 2, 2), (1, 1), 14, 8, 2, 2,
                              mini_groups=3, grow_target_every=3)
    grown.load_state_dict(prefix.state_dict(), strict=True)
    assert set(grown.state_dict()) == set(prefix.state_dict())
    x = torch.randn(2, 1, 2, 2, requires_grad=True)
    x2 = x.detach().clone().requires_grad_()
    torch.manual_seed(34)
    a = prefix(x)
    torch.manual_seed(34)
    b = grown(x2)
    assert torch.equal(a['z'], b['z'])
    torch.testing.assert_close(a['logits'], b['logits'], atol=3e-6, rtol=3e-5)
    (a['log_q'].sum() + a['logits'][:, -1, 1].sum()).backward()
    (b['log_q'].sum() + b['logits'][:, -1, 1].sum()).backward()
    torch.testing.assert_close(x.grad, x2.grad, atol=1e-5, rtol=1e-4)
    for p, q in zip(prefix.parameters(), grown.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=3e-5, rtol=2e-4)


def test_grow_capacity_schedule_and_no_grad_storage_reuse():
    layer = MHA(8, 2)
    layer.grow_target_every, layer.max_cache_length = 3, 7
    capacities, pointers = [], []
    with torch.no_grad():
        cache = None
        for t in range(7):
            _, cache = layer.step(torch.randn(2, 1, 8), cache)
            assert isinstance(cache, GrowingKV) and cache.length == t + 1
            capacities.append(cache.k.shape[2])
            pointers.append(cache.k.data_ptr())
    assert capacities == [3, 3, 3, 6, 6, 6, 7]
    assert pointers[0] == pointers[1] == pointers[2]
    assert pointers[3] == pointers[4] == pointers[5]


@pytest.mark.parametrize('device,dtype', [('cpu', torch.bfloat16), ('cuda', torch.float16)])
def test_mini_group_device_amp_and_count_probability_dimensions(device, dtype):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    ar = ARPoissonGroup((4, 2, 5), (1, 2, 5), embed_dim=8, num_heads=2,
                        num_layers=2, mini_groups=3, grow_target_every=3).to(device)
    feature = torch.randn(2, 4, 2, 5, device=device, requires_grad=True)
    raw = torch.randn(2, 1, 2, 5, device=device, requires_grad=True)
    with torch.autocast(device_type=device, dtype=dtype):
        out = ar(feature, raw)
    assert out['z_st'].shape == out['z_hard'].shape == raw.shape
    assert torch.equal(out['z_st'], out['z_hard'].float())
    assert out['kl_per_var'].shape == (2, 10)
    assert out['log_q'].shape == out['log_p'].shape == (2,)
    for key in ('z_st', 'kl', 'log_q', 'log_p'):
        assert out[key].dtype == torch.float32 and out[key].device.type == device
    expected = torch.distributions.Poisson(
        torch.exp(5 * torch.tanh(raw.float().flatten(1) / 5))).log_prob(
        out['z_hard'].flatten(1)).sum(-1)
    torch.testing.assert_close(out['log_p'], expected, atol=2e-4, rtol=2e-5)
    out['kl'].mean().backward()
    assert torch.isfinite(feature.grad).all() and feature.grad.abs().sum() > 0
    assert torch.isfinite(raw.grad).all() and raw.grad.abs().sum() > 0


@pytest.mark.parametrize('mode', ['capped64', 'exact64'])
def test_mini_groups_full_hierarchy_backward_and_checkpoint(mode, tmp_path):
    from test_ar_poisson import model_args, new_model
    import utils
    args = model_args(mode=mode)
    args.ar_mini_groups, args.ar_grow_target_every = 5, 3
    model = new_model(args).train()
    x = torch.bernoulli(torch.rand(2, 1, 32, 32))
    logits, log_q, log_p, kl, _ = model(x)
    loss = (utils.reconstruction_loss(model.decoder_output(logits), x, crop=True) + sum(kl)).mean()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert torch.isfinite(log_q).all() and torch.isfinite(log_p).all()
    for head in model.enc_sampler:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
    checkpoint = tmp_path / 'mini.pt'
    torch.save({'args': args, 'state_dict': model.state_dict()}, checkpoint)
    loaded = utils.load_checkpoint(checkpoint)
    restored = new_model(loaded['args']).eval()
    restored.load_state_dict(loaded['state_dict'], strict=True)
    with torch.no_grad():
        model.eval()
        torch.manual_seed(35)
        a = model(x)
        torch.manual_seed(35)
        b = restored(x)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


@pytest.mark.parametrize('kwargs', [{'mini_groups': 0}, {'mini_groups': -1},
                                    {'grow_target_every': -1}])
def test_invalid_mini_group_cache_configuration(kwargs):
    with pytest.raises(ValueError, match='integer'):
        RecognitionScale1(**kwargs)
