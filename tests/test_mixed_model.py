from types import SimpleNamespace

import torch

from model import AutoEncoder
import utils


def _args():
    return SimpleNamespace(
        dataset='mnist',
        use_se=False,
        res_dist=True,
        num_x_bits=8,
        latent_distribution='mixed_poisson_gamma',
        reconstruction_distribution='gaussian',
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


def test_group_mapping_and_gaussian_head():
    model = _model()
    assert model.groups_per_scale == [4, 2]
    assert [model._latent_kind(i) for i in range(6)] == [
        'poisson', 'poisson', 'gamma', 'gamma', 'gamma', 'gamma']

    model.train()
    logits, log_q, log_p, kl_all, kl_diag = model(torch.rand(2, 1, 32, 32))
    assert tuple(logits.shape) == (2, 2, 32, 32)
    assert tuple(log_q.shape) == (2,)
    assert tuple(log_p.shape) == (2,)
    assert len(kl_all) == 6
    assert len(kl_diag) == 6
    assert all(torch.isfinite(kl).all() for kl in kl_all)


def test_exact_prior_sampling_path():
    model = _model().eval()
    with torch.no_grad():
        logits = model.sample(num_samples=2, t=1.)
    assert tuple(logits.shape) == (2, 2, 32, 32)
    assert torch.isfinite(logits).all()
