"""CNN cross-memory and group-local cache/gradient contracts."""

import pytest
import torch

from recognition_ar_poisson import ARFeatureRecognition, EncoderBlock
from test_ar_poisson import model_args, new_model, assert_grad


@pytest.mark.parametrize('shape', [(4, 2, 2), (8, 4, 4), (4, 8, 8)])
@pytest.mark.parametrize('memory_tokens', [1, 6, 64])
def test_cnn_memory_length_is_independent_of_scale(shape, memory_tokens):
    rec = ARFeatureRecognition(shape, d_lat=5, memory_tokens=memory_tokens,
                               embed_dim=8, num_heads=2, num_layers=2)
    feature = torch.randn(2, *shape, requires_grad=True)
    memory = rec.encode_image(feature)
    assert memory.shape == (2, memory_tokens, 8)
    assert not any(isinstance(module, EncoderBlock) for module in rec.modules())
    assert len(rec.decoder) == 2
    for key, value in rec.prepare_cross_kv(memory):
        assert key.shape == value.shape == (2, 2, memory_tokens, 4)
    memory.square().mean().backward()
    assert torch.isfinite(feature.grad).all() and feature.grad.abs().sum() > 0
    assert_grad(rec.feature_cnn.parameters())


def test_group_caches_are_local_and_cross_memory_is_projected_once(monkeypatch):
    torch.manual_seed(41)
    rec = ARFeatureRecognition((4, 3, 3), d_lat=5, memory_tokens=6,
                               embed_dim=8, num_heads=2, num_layers=2).train()
    feature = torch.randn(2, 4, 3, 3, requires_grad=True)
    step_records, prefix_inputs, cross_projections = [], [], []
    first_block = rec.decoder[0]
    original_step = first_block.step

    def capture_step(x, self_kv, cross_kv):
        step_records.append((0 if self_kv is None else self_kv[0].shape[2],
                             id(cross_kv), cross_kv[0].shape[2]))
        prefix_inputs.append(x)
        return original_step(x, self_kv, cross_kv)

    monkeypatch.setattr(first_block, 'step', capture_step)
    handles = [projection.register_forward_hook(
        lambda module, inputs, output: cross_projections.append(inputs[0].shape))
        for block in rec.decoder
        for projection in (block.cross_attn.k, block.cross_attn.v)]
    out = rec(feature, tau=0.7)
    assert 'self_kv' not in out and 'cross_kv' not in out
    assert [row[0] for row in step_records] == list(range(5))
    assert len({row[1] for row in step_records}) == 1
    assert all(row[2] == 6 for row in step_records)
    assert cross_projections == [torch.Size((2, 6, 8))] * 4

    # The input at step 1 embeds the first sampled ST latent. Later logits
    # must still backpropagate to it after explicit cache references are cleared.
    late_logit = out['logits'][:, -1, 1].sum()
    prefix_gradient = torch.autograd.grad(late_logit, prefix_inputs[1],
                                          retain_graph=True)[0]
    assert torch.isfinite(prefix_gradient).all() and prefix_gradient.abs().sum() > 0
    late_logit.backward()
    assert_grad(rec.feature_cnn.parameters())
    assert_grad(rec.embedding.parameters())
    assert torch.isfinite(feature.grad).all() and feature.grad.abs().sum() > 0

    step_records.clear()
    cross_projections.clear()
    prefix_inputs.clear()
    with torch.no_grad():
        torch.manual_seed(42)
        second = rec(feature.detach(), tau=None)
        torch.manual_seed(42)
        repeated = rec(feature.detach(), tau=None)
    assert [row[0] for row in step_records] == list(range(5)) * 2
    assert torch.equal(second['logits'], repeated['logits'])
    assert torch.equal(second['z'], repeated['z'])
    assert len(cross_projections) == 8
    assert not any('cache' in name or '_kv' in name for name in vars(rec))
    for handle in handles:
        handle.remove()


def test_later_ar_memory_backpropagates_to_both_combiner_inputs():
    torch.manual_seed(43)
    args = model_args()
    args.ar_memory_tokens = 6
    model = new_model(args).train()
    combiner_inputs, ar_outputs = [], []
    handles = [cell.register_forward_pre_hook(
        lambda module, inputs: combiner_inputs.append(inputs))
        for cell in model.enc_tower if cell.cell_type == 'combiner_enc']
    handles.append(model.enc_sampler[2].register_forward_hook(
        lambda module, inputs, output: ar_outputs.append(output)))
    model(torch.bernoulli(torch.rand(2, 1, 32, 32)))
    assert all(head.rec.memory_tokens == 6 for head in model.enc_sampler
               if hasattr(head, 'rec'))
    # Calls follow top-down order: combiner 0 feeds NAR group 1,
    # combiner 1 feeds AR group 2.
    bottom_up, top_down = combiner_inputs[1]
    signal = ar_outputs[0]['categorical']['logits'][:, -1, 1].sum()
    for gradient in torch.autograd.grad(signal, (bottom_up, top_down), retain_graph=True):
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    signal.backward()
    assert_grad(model.enc_sampler[2].rec.feature_cnn.parameters())
    assert_grad(model.enc_sampler[0].rec.head.parameters())
    for handle in handles:
        handle.remove()


def test_invalid_memory_length_rejected():
    args = model_args()
    args.ar_memory_tokens = 0
    with pytest.raises(ValueError, match='positive'):
        new_model(args)


def test_top_ar_fuses_initial_decoder_state_without_changing_top_prior():
    torch.manual_seed(44)
    args = model_args()
    args.num_channels_dec = 8  # Combiner must project different channel widths.
    args.ar_memory_tokens = 6
    model = new_model(args).train()
    captured = []
    handle = model.enc_sampler[0].register_forward_hook(
        lambda module, inputs, output: captured.append(output))
    prior = model.top_prior(2).detach().clone()
    model(torch.bernoulli(torch.rand(2, 1, 32, 32)))
    signal = captured[0]['categorical']['logits'][:, -1, 1].sum()
    signal.backward()
    assert_grad(model.top_enc_combiner.parameters())
    assert_grad([model.prior_ftr0])
    assert_grad(model.enc0.parameters())
    assert torch.equal(model.top_prior(2), prior)
    assert model.top_prior.raw.grad is None  # q logits never read prior rates.
    handle.remove()
