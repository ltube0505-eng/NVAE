from types import SimpleNamespace

import numpy as np
import torch

from rate_monitor import gather_group1_rates, write_rate_snapshot


def test_capture_is_actual_first_group_rate_and_does_not_change_forward():
    from test_mixed_model import _poisson_model
    for estimator in ('straight_through', 'relaxed', 'reinforce', 'obbvi'):
        model = _poisson_model(estimator).train()
        x = torch.rand(2, 1, 32, 32)
        expected = []
        handle = model.enc_sampler[0].register_forward_hook(
            lambda module, args, output: expected.append(
                model._latent_from_param(0, output).rate.detach().float()))
        torch.manual_seed(81)
        baseline = model(x)[0].detach()
        assert model.group1_posterior_rates is None
        model.collect_group1_rates = True
        torch.manual_seed(81)
        actual = model(x)[0].detach()
        handle.remove()
        assert torch.equal(baseline, actual)
        assert torch.equal(model.group1_posterior_rates, expected[-1])
        assert not model.group1_posterior_rates.requires_grad
        values = gather_group1_rates(model.group1_posterior_rates)
        assert values.shape == tuple(expected[-1].shape)
        model.collect_group1_rates = False
        model(x)
        assert model.group1_posterior_rates is None


def test_snapshot_preserves_all_values_and_histogram_mass(tmp_path):
    import csv
    values = np.array([.007, .02, .1, .5, 1., 3., 10., 30.], dtype=np.float32).reshape(2, 1, 2, 2)
    logged = {}
    histograms = []
    writer = SimpleNamespace(add_scalar=lambda name, value, step: logged.update({name: value}),
                             add_histogram=lambda *a, **kw: histograms.append((a, kw)))
    logger = SimpleNamespace(info=lambda *a: None)
    stats = write_rate_snapshot(values, str(tmp_path), 500, writer, logger)
    folder = tmp_path / 'posterior_rate_monitor' / 'group_001'
    with np.load(str(folder / 'step_000000500.npz')) as data:
        assert np.array_equal(data['rates'], values)
        assert data['counts'].sum() == values.size
    with (folder / 'step_000000500_histogram.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert sum(int(r['count']) for r in rows) == values.size
    assert abs(sum(float(r['fraction']) for r in rows) - 1.) < 1e-12
    assert stats['mean'] == values.mean(dtype=np.float64)
    assert stats['num_images'] == 2 and stats['num_rates'] == 8
    assert len(histograms) == 1
    from PIL import Image
    with Image.open(str(folder / 'step_000000500.png')) as img:
        assert img.size[0] > 500 and img.size[1] > 300
    write_rate_snapshot(values, str(tmp_path), 1000, writer, logger)
    with (folder / 'summary.csv').open() as stream:
        assert len(list(csv.DictReader(stream))) == 2


def test_distributed_gather_keeps_every_position_with_unequal_batches(monkeypatch):
    import rate_monitor as monitor
    local = torch.tensor([[[[.1, .2]]]])
    remote = torch.tensor([.3, .4, .5, .6])
    monkeypatch.setattr(monitor.dist, 'is_available', lambda: True)
    monkeypatch.setattr(monitor.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(monitor.dist, 'get_world_size', lambda: 2)
    monkeypatch.setattr(monitor.dist, 'get_rank', lambda: 0)

    def all_gather(outputs, tensor):
        if tensor.dtype == torch.long:
            outputs[0].fill_(2)
            outputs[1].fill_(4)
        else:
            outputs[0].copy_(tensor)
            outputs[1].copy_(remote)
    monkeypatch.setattr(monitor.dist, 'all_gather', all_gather)
    result = gather_group1_rates(local, distributed=True)
    assert result.shape == (3, 1, 1, 2)
    np.testing.assert_array_equal(result.reshape(-1), np.array([.1,.2,.3,.4,.5,.6], dtype=np.float32))
    monkeypatch.setattr(monitor.dist, 'get_rank', lambda: 1)
    assert gather_group1_rates(local, distributed=True) is None
