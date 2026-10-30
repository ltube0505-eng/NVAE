from types import SimpleNamespace

import numpy as np
import torch

from rate_monitor import gather_group1_rates, log_rate_distribution


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


def test_logs_all_summary_and_bins_without_exporting_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    values = np.array([.007, .02, .1, .5, 1., 3., 10., 30.], dtype=np.float32).reshape(2, 1, 2, 2)
    logged = {}
    histograms = []
    messages = []
    writer = SimpleNamespace(add_scalar=lambda name, value, step: logged.update({name: value}),
                             add_histogram=lambda *a, **kw: histograms.append((a, kw)))
    logger = SimpleNamespace(info=lambda fmt, *args: messages.append(fmt % args))
    stats = log_rate_distribution(values, 500, writer, logger)
    assert list(tmp_path.iterdir()) == []
    assert stats['mean'] == values.mean(dtype=np.float64)
    assert stats['std'] == values.std(dtype=np.float64)
    assert stats['min'] == values.min() and stats['max'] == values.max()
    for name, expected in zip(('p01', 'p05', 'p25', 'p50', 'p75', 'p95', 'p99'),
                              np.percentile(values, [1, 5, 25, 50, 75, 95, 99])):
        assert stats[name] == expected
    assert stats['num_images'] == 2 and stats['num_rates'] == 8
    assert len(histograms) == 1
    args, kwargs = histograms[0]
    assert args[0] == 'posterior_rate/group_001' and args[2] == 500
    np.testing.assert_array_equal(args[1], values.reshape(-1))
    expected_counts, _ = np.histogram(values, bins=kwargs['bins'])
    count_tags = sorted(k for k in logged if k.endswith('/count'))
    fraction_tags = sorted(k for k in logged if k.endswith('/fraction'))
    assert len(count_tags) == len(fraction_tags) == 60
    np.testing.assert_array_equal([logged[k] for k in count_tags], expected_counts)
    np.testing.assert_allclose([logged[k] for k in fraction_tags], expected_counts / values.size)
    assert sum(logged[k] for k in count_tags) == values.size
    assert abs(sum(logged[k] for k in fraction_tags) - 1.) < 1e-12
    for name, stat in stats.items():
        if name != 'step':
            assert logged['posterior_rate/group_001/' + name] == stat
    for name in ('mean', 'std', 'min', 'max', 'p01', 'p05', 'p25', 'p50', 'p75', 'p95', 'p99'):
        assert name + '=' in messages[0]
    assert len(messages) == 61
    for index, message in enumerate(messages[1:]):
        assert 'index=%03d' % index in message
        assert 'count=%d' % expected_counts[index] in message
        assert 'fraction=' in message and 'rate=[' in message
    log_rate_distribution(values, 1000, writer, logger)
    assert len(histograms) == 2 and len(messages) == 122
    assert list(tmp_path.iterdir()) == []


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

