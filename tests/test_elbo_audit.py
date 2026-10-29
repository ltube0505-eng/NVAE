from unittest import mock

import pytest
import torch

from audit_poisson_elbo import PoissonAudit, evaluate_batch
from distributions import Poisson
from test_explicit_scales import scheduled_model


@pytest.mark.parametrize('sampler', ['native', 'cpu', 'fresh'])
def test_audit_reconciles_every_group_and_jensen(sampler):
    torch.manual_seed(113)
    model = scheduled_model().eval()
    with PoissonAudit(sampler) as audit:
        values, groups, iw_nll = evaluate_batch(model, torch.zeros(2, 1, 32, 32), 3, audit)
    assert groups.shape == (2, 3, 15, 8)
    assert torch.allclose(groups[:, :, :, 1] - groups[:, :, :, 0],
                          groups[:, :, :, 2], atol=1e-10)
    assert (groups[:, :, :, 3] >= 0).all()
    assert groups[:, :, :, 4].abs().max() < 1e-3
    assert groups[:, :, :, 5].abs().max() < 1e-3
    mc_nelbo = (values[:, :, 0] + values[:, :, 2]).mean(1)
    assert torch.all(iw_nll <= mc_nelbo + 1e-8)


def test_audit_detects_inconsistent_kl_implementation():
    model = scheduled_model().eval()
    original = Poisson.kl
    with mock.patch.object(Poisson, 'kl', lambda q, p: original(q, p) + 0.1):
        with PoissonAudit() as audit:
            _, groups, _ = evaluate_batch(model, torch.zeros(2, 1, 32, 32), 1, audit)
    # Deliberately corrupt each scalar KL. The audit must report it, rather
    # than silently replacing the implementation's value with a new estimate.
    assert torch.all(groups[:, :, :, 4] > 30.)


def test_audit_restores_original_distribution_methods():
    original = Poisson.log_p
    with pytest.raises(RuntimeError, match='sentinel'):
        with PoissonAudit():
            raise RuntimeError('sentinel')
    assert Poisson.log_p is original


def test_checkpoint_cli_workflow(tmp_path):
    from types import SimpleNamespace
    from audit_poisson_elbo import run
    from model import AutoEncoder
    from test_poisson_ste import training_defaults
    import utils

    args = training_defaults()
    for key, value in dict(arch_instance='res_elu', use_se=False,
                           num_latent_scales=5, latent_scale_groups='3,2,4,4,2',
                           latent_scale_sizes='4,4,8,8,8', num_channels_enc=4,
                           num_channels_dec=4, num_cell_per_cond_enc=1,
                           num_cell_per_cond_dec=1, num_preprocess_cells=1,
                           num_postprocess_cells=1).items():
        setattr(args, key, value)
    arch = utils.get_arch_cells(args.arch_instance)
    model = AutoEncoder(args, None, arch)
    path = tmp_path / 'checkpoint.pt'
    torch.save({'args': args, 'arch_instance': arch, 'epoch': 120,
                'state_dict': model.state_dict()}, str(path))
    fake_data = SimpleNamespace(get_loaders=lambda args: (
        None, [(torch.zeros(2, 1, 32, 32), torch.zeros(2))], None))
    options = SimpleNamespace(checkpoint=str(path), seed=1, data=None,
                              batch_size=2, device='cpu', sampler='native',
                              max_batches=1, samples=2)
    with mock.patch.dict('sys.modules', {'datasets': fake_data}):
        report = run(options)
    assert report['images'] == 2 and len(report['groups']) == 15
    assert report['mc_nelbo'] >= report['iw_nll']
    assert report['poisson_residual_se'] > 0
    decomposition = (report['poisson_residual64'] +
                     report['total_log_ratio_numeric_error'] -
                     report['total_kl_numeric_error'])
    assert abs(report['mc_minus_analytic'] - decomposition) < 1e-8


def test_fresh_audit_seed_is_reproducible_and_not_reset_per_forward():
    def trajectory():
        with PoissonAudit('fresh', seed=7) as audit:
            q = Poisson(torch.full((2, 20, 4, 4), 1.))
            first = q.sample(estimator='exact')[0]
            audit.reset()
            second = q.sample(estimator='exact')[0]
        return first, second
    a, b = trajectory()
    c, d = trajectory()
    assert torch.equal(a, c) and torch.equal(b, d)
    assert not torch.equal(a, b)
