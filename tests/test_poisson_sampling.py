from unittest import mock

import pytest
import torch

from distributions import Poisson
from poisson_sampling import PoissonSampler, collect_sampler_states, restore_sampler_state
from test_explicit_scales import scheduled_model


@pytest.mark.parametrize('backend', ['fresh', 'cpu'])
def test_sampler_state_replays_next_draw_without_repeating_every_call(backend):
    rate = torch.full((1024,), 28.)
    sampler = PoissonSampler(backend, seed=7)
    first = sampler(rate)
    states = collect_sampler_states(sampler)
    second = sampler(rate)
    restored = PoissonSampler(backend, seed=999)
    assert restore_sampler_state(restored, states)
    assert torch.equal(restored(rate), second)
    assert not torch.equal(first, second)
    assert not restore_sampler_state(restored, None)
    assert not restore_sampler_state(restored, states, rank=2)


def test_fresh_uses_different_device_generator_seeds_and_preserves_global_cpu_rng():
    sampler = PoissonSampler('fresh', seed=13)
    global_state = torch.get_rng_state().clone()
    seeds = []
    original = torch.poisson

    def record(rate, generator=None):
        assert generator is not None
        seeds.append(generator.initial_seed())
        return original(rate, generator=generator)

    with mock.patch.object(torch, 'poisson', side_effect=record):
        for _ in range(4):
            sampler(torch.ones(128))
    assert len(set(seeds)) == 4
    assert torch.equal(global_state, torch.get_rng_state())


@pytest.mark.parametrize('backend', ['fresh', 'cpu'])
def test_half_counts_and_ste_gradient_are_supported(backend):
    rate = torch.full((1024,), 28., dtype=torch.float16, requires_grad=True)
    hard = PoissonSampler(backend, seed=17)(rate)
    assert hard.dtype == rate.dtype and hard.device == rate.device
    assert not hard.requires_grad
    assert torch.equal(hard, hard.round())
    z = hard + (rate - rate.detach())
    z.sum().backward()
    assert torch.equal(rate.grad, torch.ones_like(rate))


@pytest.mark.parametrize('estimator', ['exact', 'reinforce', 'straight_through', 'obbvi'])
def test_all_integer_estimators_use_shared_sampler(estimator):
    sampler = mock.Mock(return_value=torch.full((3,), 4.))
    q = Poisson(torch.zeros(3, requires_grad=True), sampler=sampler)
    z, _ = q.sample(estimator=estimator, proposal_tau=3.)
    sampler.assert_called_once()
    expected = q.proposal_rate(3.) if estimator == 'obbvi' else q.rate
    assert torch.equal(sampler.call_args[0][0], expected)
    assert torch.equal(z.detach(), torch.full((3,), 4.))
    assert z.requires_grad == (estimator == 'straight_through')


def test_fixed_and_conditional_groups_share_model_sampler_in_train_and_generation():
    model = scheduled_model(poisson_sampling_backend='fresh').train()
    q = model._latent_from_param(0, torch.zeros(2, 40, 4, 4))
    p = model._fixed_prior(0, torch.zeros(2, 20, 4, 4))
    assert q.sampler is model.poisson_sampler and p.sampler is model.poisson_sampler
    with mock.patch.object(PoissonSampler, '__call__', wraps=None,
                           side_effect=lambda rate: torch.poisson(rate.detach())) as sample:
        output = model(torch.zeros(2, 1, 32, 32))[0]
        assert sample.call_count == 15
        output.square().mean().backward()
        model.eval().sample(2, 1.)
        assert sample.call_count == 30


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match='backend'):
        PoissonSampler('typo')


def test_distributed_sampler_states_are_collected_and_restored_by_rank():
    first = PoissonSampler('fresh', seed=1)
    second = PoissonSampler('fresh', seed=1000004)
    rate = torch.ones(64)
    first(rate)
    second(rate)
    expected_states = [first.get_state(), second.get_state()]
    expected_next = second(rate)

    def gather(outputs, payload):
        assert payload.device.type == 'cpu'  # gloo test; NCCL uses CUDA payloads
        for output, state in zip(outputs, expected_states):
            output.copy_(state)

    with mock.patch('torch.distributed.is_initialized', return_value=True), \
            mock.patch('torch.distributed.get_world_size', return_value=2), \
            mock.patch('torch.distributed.get_backend', return_value='gloo'), \
            mock.patch('torch.distributed.all_gather', side_effect=gather):
        states = collect_sampler_states(first, distributed=True)
    assert len(states) == 2
    resumed = PoissonSampler('fresh', seed=0)
    assert restore_sampler_state(resumed, states, rank=1)
    assert torch.equal(resumed(rate), expected_next)
