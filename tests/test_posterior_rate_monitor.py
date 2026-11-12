import argparse
import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from test_mixed_model import _vanilla_model, _poisson_model
from train_poisson_posterior_rate import (
    PosteriorRateMonitor, RateOnlyWriter, configure_args,
)


class MemoryWriter:
    def __init__(self):
        self.values = {}

    def add_rate(self, tag, value, step):
        self.values[(tag, step)] = value

    def flush(self):
        pass


@pytest.mark.parametrize('hierarchical', [False, True])
def test_actual_posteriors_all_positions_without_changing_rng_or_gradients(hierarchical):
    model = (_poisson_model() if hierarchical else _vanilla_model()).train()
    reference = (_poisson_model() if hierarchical else _vanilla_model()).train()
    reference.load_state_dict(model.state_dict())
    x = torch.rand(2, 1, 32, 32)
    writer = MemoryWriter()
    monitor = PosteriorRateMonitor(writer, every=100, position_images=2)
    model.posterior_rate_monitor = monitor
    rates = []
    original = model._sample_latent

    def capture(dist, training_sample=False):
        if training_sample:
            rates.append(dist.rate.detach().clone())
        return original(dist, training_sample=training_sample)

    torch.manual_seed(42)
    expected = reference(x)
    (expected[0].square().mean() + sum(expected[3]).mean()).backward()
    rng = torch.get_rng_state().clone()
    torch.manual_seed(42)
    monitor.begin(77771)
    with patch.object(model, '_sample_latent', side_effect=capture):
        actual = model(x)
    (actual[0].square().mean() + sum(actual[3]).mean()).backward()
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(expected[0], actual[0])
    for p, q in zip(model.parameters(), reference.parameters()):
        assert (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            assert torch.equal(p.grad, q.grad)
    assert len(rates) == sum(model.groups_per_scale)
    assert len(writer.values) == 3 * sum(rate[0].numel() for rate in rates)
    for group, rate in enumerate(rates):
        for view, grid in [('batch_mean', rate.mean(0)), ('sample_0', rate[0]), ('sample_1', rate[1])]:
            for c in range(grid.size(0)):
                for h in range(grid.size(1)):
                    for w in range(grid.size(2)):
                        tag = ('grad_positions/current_batch/%s/posterior/q%d/rate/c%02d/h%d_w%d'
                               % (view, group, c, h, w))
                        assert writer.values[(tag, 77771)] == grid[c, h, w].item()
    assert not monitor.active
    monitor.begin(77772)
    assert not monitor.active
    monitor.begin(77800)
    assert monitor.active
    model.eval()
    count = len(writer.values)
    with torch.no_grad():
        model(x)
        model.sample(2, t=1.)
    assert len(writer.values) == count


def test_rate_only_event_stream_and_resume_purge(tmp_path):
    writer = RateOnlyWriter(tmp_path)
    writer.open()
    monitor = PosteriorRateMonitor(writer, position_images=0)
    for step in (0, 100, 200):
        monitor.begin(step)
        monitor.record([SimpleNamespace(rate=torch.full((2, 1, 1, 1), 42.))])
    writer.add_scalar('train/loss', 1., 0)
    writer.add_image('input', torch.zeros(1, 2, 2), 0)
    writer.add_histogram('gradient', torch.ones(2), 0)
    writer.close()
    writer.open(purge_step=100)
    monitor.begin(100)
    monitor.record([SimpleNamespace(rate=torch.full((2, 1, 1, 1), 55.))])
    writer.close()
    events = EventAccumulator(writer.log_dir).Reload()
    tags = events.Tags()
    assert tags['images'] == tags['histograms'] == []
    assert len(tags['scalars']) == 1
    tag = tags['scalars'][0]
    assert tag == 'grad_positions/current_batch/batch_mean/posterior/q0/rate/c00/h0_w0'
    assert [(event.step, event.value) for event in events.Scalars(tag)] == [(0, 42.), (100, 55.)]


def test_resume_restores_training_settings_but_keeps_monitor_options(tmp_path):
    path = tmp_path / 'eval-test' / 'checkpoint.pt'
    path.parent.mkdir()
    args = SimpleNamespace(rate_every=100, rate_position_images=1, rate_resume=None,
                           num_process_per_node=1, num_proc_node=1, num_nf=0,
                           latent_distribution='poisson', poisson_gradient_estimator='straight_through',
                           batch_size=64, epochs=200)
    torch.save(dict(args=args, epoch=83, global_step=77771, state_dict={},
                    optimizer={}, scheduler={}, grad_scalar={}), path)
    current = copy.deepcopy(args)
    current.rate_resume = str(path)
    current.rate_every = 50
    current.rate_position_images = 2
    current.batch_size = 32
    restored = configure_args(current, argparse.ArgumentParser())
    assert restored.batch_size == 64
    assert restored.rate_every == 50 and restored.rate_position_images == 2
    assert restored.cont_training and restored.posterior_rate_only
    assert restored.root == str(tmp_path) and restored.save == 'test'


def test_resume_rejects_checkpoint_from_gaussian_branch(tmp_path):
    path = tmp_path / 'eval-test' / 'checkpoint.pt'
    path.parent.mkdir()
    current = SimpleNamespace(rate_every=100, rate_position_images=1, rate_resume=str(path),
                              num_process_per_node=1, num_proc_node=1)
    saved = SimpleNamespace(image_likelihood='gaussian_fixed')
    torch.save(dict(args=saved, epoch=1, global_step=10, state_dict={},
                    optimizer={}, scheduler={}, grad_scalar={}), path)
    with pytest.raises(SystemExit):
        configure_args(current, argparse.ArgumentParser())


def test_training_loop_emits_only_rates(tmp_path):
    import train
    import utils

    model = _vanilla_model()
    writer = RateOnlyWriter(tmp_path)
    writer.open()
    model.posterior_rate_monitor = PosteriorRateMonitor(writer)
    args = SimpleNamespace(learning_rate=0.001, num_x_bits=8, distributed=False,
                           kl_anneal_portion=0.3, num_total_iter=1000,
                           kl_const_portion=0.0001, kl_const_coeff=0.0001,
                           weight_decay_norm_anneal=False, weight_decay_norm=0.,
                           posterior_rate_only=True)
    optimizer = torch.optim.Adamax(model.parameters(), lr=0.001)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    logger = SimpleNamespace(info=lambda *args: None)
    batches = [(torch.rand(2, 1, 32, 32).round(), torch.zeros(2)) for _ in range(2)]
    with patch.object(train, 'args', args, create=True), \
            patch.object(torch.Tensor, 'cuda', lambda self, *a, **kw: self), \
            patch.object(model, 'spectral_norm_parallel', return_value=torch.tensor(0.)), \
            patch.object(model, 'batchnorm_loss', return_value=torch.tensor(0.)), \
            patch.object(utils, 'average_gradients', side_effect=AssertionError('single-rank reduction')):
        _, step = train.train(batches, model, optimizer, scaler, 99, 0, writer, logger)
    writer.close()
    assert step == 101
    events = EventAccumulator(writer.log_dir).Reload()
    assert len(events.Tags()['scalars']) == 2 * model.z0_size[0] * model.z0_size[1] * model.z0_size[2]
    for tag in events.Tags()['scalars']:
        assert '/posterior/q0/rate/' in tag
        assert [event.step for event in events.Scalars(tag)] == [99, 100]

