"""Poisson integer sampling with per-call isolated CUDA random streams."""
import torch


class PoissonSampler:
    """Share one instance across a model's groups and forward passes.

    ``fresh`` draws a new seed on CPU for every call and uses a new generator
    on the rate's device. Never reseed every group with the same seed.
    Explicit ``native`` is retained only for legacy reproduction/diagnostics.
    """
    def __init__(self, backend='fresh', seed=None):
        if backend not in {'fresh', 'cpu', 'native'}:
            raise ValueError('Unknown Poisson sampling backend: %s' % backend)
        self.backend = backend
        self.generator = None
        if seed is not None:
            self.generator = torch.Generator().manual_seed(int(seed) + 100003)

    def __call__(self, rate):
        with torch.no_grad():
            # Old CPU builds do not implement half-precision Poisson. Counts
            # are cast back after sampling; no arrival/count cutoff is used.
            value = rate.detach()
            if value.dtype not in (torch.float32, torch.float64):
                value = value.float()
            if self.backend == 'native':
                result = torch.poisson(value)
            elif self.backend == 'cpu':
                result = torch.poisson(value.cpu(), generator=self.generator)
            else:
                seed = int(torch.randint(0, 2**63 - 1, (1,), device='cpu',
                                         generator=self.generator).item())
                generator = torch.Generator(device=value.device).manual_seed(seed)
                result = torch.poisson(value, generator=generator)
            return result.to(device=rate.device, dtype=rate.dtype)

    def get_state(self):
        state = self.generator.get_state() if self.generator is not None else torch.get_rng_state()
        return state.clone()

    def set_state(self, state):
        state = state.detach().cpu().to(dtype=torch.uint8)
        if self.generator is None:
            torch.set_rng_state(state)
        else:
            self.generator.set_state(state)


def collect_sampler_states(sampler, distributed=False):
    """Collect each rank's small CPU RNG state before rank zero saves a checkpoint."""
    import torch.distributed as dist
    state = sampler.get_state()
    if distributed and dist.is_initialized() and dist.get_world_size() > 1:
        device = torch.device('cuda', torch.cuda.current_device()) \
            if dist.get_backend() == 'nccl' else torch.device('cpu')
        payload = state.to(device)
        states = [torch.empty_like(payload) for _ in range(dist.get_world_size())]
        dist.all_gather(states, payload)
        return [value.cpu() for value in states]
    return [state]


def restore_sampler_state(sampler, states, rank=0):
    """Legacy checkpoints have no sampler state and start a newly seeded stream."""
    if states is None or rank >= len(states):
        return False
    sampler.set_state(states[rank])
    return True
