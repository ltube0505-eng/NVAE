"""Logical scales can share spatial sizes while retaining sample-only bridges."""
from unittest import mock

import pytest
import torch

from distributions import Poisson
from neural_operations import ScaleBoundaryCell
from test_scale_last_sample import make_model


def scheduled_model(**kwargs):
    config = dict(latent_distribution='poisson', num_latent_scales=5,
                  latent_scale_groups='3,2,4,4,2', latent_scale_sizes='4,4,8,8,8',
                  num_latent_per_group=20, num_preprocess_blocks=2,
                  num_postprocess_blocks=2, ada_groups=True)
    config.update(kwargs)
    return make_model(**config)


def test_requested_shapes_training_gradients_generation_and_restore():
    torch.manual_seed(89)
    model = scheduled_model().train()
    assert model.groups_per_scale == [2, 4, 4, 2, 3]
    assert model.latent_scale_sizes == [4, 4, 8, 8, 8]
    assert model.z0_size == [20, 4, 4]
    boundaries = [m for m in model.dec_tower if isinstance(m, ScaleBoundaryCell)]
    assert [tuple(m.initial_feature.shape[1:]) for m in boundaries] == [
        (4, 4), (8, 8), (8, 8), (8, 8)]
    assert sum(m.cell_type == 'up_dec' for m in model.image_dec_tower) == 1
    samples = []
    original = model._sample_latent

    def record(dist, training_sample=False):
        assert isinstance(dist, Poisson)
        z, aux = original(dist, training_sample)
        if training_sample:
            z.retain_grad()
        samples.append(z)
        return z, aux

    x = torch.rand(2, 1, 32, 32)
    with mock.patch.object(model, '_sample_latent', side_effect=record):
        logits, log_q, log_p, kl_all, kl_diag = model(x)
    expected = [(2, 20, 4, 4)] * 5 + [(2, 20, 8, 8)] * 10
    assert [tuple(z.shape) for z in samples] == expected
    assert logits.shape == x.shape
    assert len(kl_all) == len(kl_diag) == 15
    assert torch.isfinite(log_q).all() and torch.isfinite(log_p).all()
    logits.square().mean().backward()  # reconstruction only: exercise STE in every group
    for z in samples:
        assert z.grad is not None and torch.isfinite(z.grad).all()
        assert z.grad.abs().sum() > 0
    for boundary in boundaries:
        for p in boundary.parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all()
    samples.clear()
    with mock.patch.object(model, '_sample_latent', side_effect=record), torch.no_grad():
        generated = model.eval().sample(2, 1.)
    assert [tuple(z.shape) for z in samples] == expected
    assert generated.shape == x.shape and torch.isfinite(generated).all()
    restored = scheduled_model().eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.manual_seed(97)
    with torch.no_grad():
        a = model(x)[0]
        torch.manual_seed(97)
        b = restored(x)[0]
    assert torch.equal(a, b)


@pytest.mark.parametrize('prior', [False, True])
def test_same_resolution_bridge_conditions_only_next_first_group(prior):
    torch.manual_seed(101)
    model = scheduled_model().eval()

    def trajectory(change_index=None):
        count = [0]
        priors = []
        handles = [h.register_forward_hook(lambda m, i, o: priors.append(o.detach().clone()))
                   for h in model.dec_sampler]

        def fixed(dist, training_sample=False):
            index = count[0]
            count[0] += 1
            return torch.full_like(dist.rate, float(index + 1) / 10 +
                                   (3. if index == change_index else 0.)), None

        try:
            with mock.patch.object(model, '_sample_latent', side_effect=fixed), torch.no_grad():
                output = model.sample(2, 1.) if prior else model(torch.zeros(2, 1, 32, 32))[0]
            return output, priors
        finally:
            for h in handles:
                h.remove()

    output, priors = trajectory()
    for terminal in (2, 4, 8, 12):
        changed, changed_priors = trajectory(terminal)
        # p(z_(terminal+1)) changes, p(z_(terminal+2)) resets after its fixed first sample.
        assert not torch.allclose(priors[terminal], changed_priors[terminal])
        assert torch.equal(priors[terminal + 1], changed_priors[terminal + 1])


@pytest.mark.parametrize('kwargs,match', [
    ({'latent_scale_groups': '3,2'}, 'length'),
    ({'latent_scale_groups': '3,2,0,4,2'}, 'positive'),
    ({'latent_scale_sizes': '4,4,8'}, 'length'),
    ({'latent_scale_sizes': '4,4,8,8,16'}, 'preprocessed'),
    ({'latent_scale_sizes': '2,4,4,2,8'}, 'equal or double'),
])
def test_invalid_schedules_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        scheduled_model(**kwargs)
