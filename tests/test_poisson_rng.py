import math
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

import check_poisson_rng as probe


def test_pair_stats_identical_and_reversed_arrays():
    x = torch.arange(10, dtype=torch.float64)
    stats = probe.PairStats()
    stats.add(x, x)
    row = stats.result()
    assert row['pearson'] == pytest.approx(1.)
    assert row['equal_fraction'] == 1.
    reverse = probe.PairStats()
    reverse.add(x, x.flip(0))
    assert reverse.result()['pearson'] == pytest.approx(-1.)


def test_poisson_collision_probability_includes_expected_duplicates():
    assert probe.equality_probability(1.) == pytest.approx(0.308508322553671, abs=1e-12)
    assert probe.equality_probability(0.01) > 0.98


@pytest.mark.parametrize('backend', ['native', 'cpu', 'fresh'])
def test_fixed_rate_statistics_and_conditional_mean_on_cpu(backend):
    row = probe.marginal_probe(backend, 'cpu', (8, 20, 4, 4), 3., 16, 11, torch.float32)
    assert abs(row['mean'] - 3.) < 6 * row['mean_iid_se']
    assert abs(row['fano'] - 1.) < 0.1
    assert abs(row['across_calls']['pearson']) < 0.1
    conditional = probe.conditional_probe(backend, 'cpu', (8, 20, 4, 4),
                                          16, 13, torch.float32)
    assert abs(conditional['weighted_residual_mean']) < 6 * conditional['conditional_iid_se']


def test_correlation_probe_detects_repeated_calls():
    class Repeated:
        def __init__(self, *args):
            self.saved = None

        def __call__(self, rate):
            if self.saved is None:
                self.saved = torch.poisson(rate)
            return self.saved.clone()
    with mock.patch.object(probe, 'Sampler', Repeated):
        row = probe.marginal_probe('native', 'cpu', (8, 20, 4, 4), 3., 4, 19, torch.float32)
    assert row['across_calls']['pearson'] == pytest.approx(1.)
    assert row['across_calls']['equal_fraction'] == 1.


def test_report_preserves_unsupported_backend_error(tmp_path):
    options = SimpleNamespace(device='cpu', backends='fresh,cpu', seeds=[1], sizes=[1],
                              batch_size=2, rates=[1.], calls=2, dtype=torch.float32,
                              out=str(tmp_path / 'rng.json'))
    original = probe.Sampler

    def make_sampler(backend, seed):
        if backend == 'fresh':
            raise TypeError('unsupported CUDA generator')
        return original(backend, seed)

    with mock.patch.object(probe, 'Sampler', side_effect=make_sampler):
        report = probe.execute(options)
    assert report['errors'] == [{'backend': 'fresh', 'error': 'unsupported CUDA generator'}]
    assert len(report['results']) == 2
    assert (tmp_path / 'rng.json').is_file()
