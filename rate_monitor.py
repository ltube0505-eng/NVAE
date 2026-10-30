"""Periodic TensorBoard and training-log summaries of group-1 posterior rates."""
import numpy as np
import torch
import torch.distributed as dist


def gather_group1_rates(rate, distributed=False):
    """Gather all ranks' [B,C,H,W] rates; return numpy only on rank zero.

    The first group is decoder-order index zero. No averaging over images,
    channels, or positions is performed. Unequal local batch sizes work.
    """
    if rate is None:
        raise RuntimeError('No group-1 posterior rates captured by forward().')
    values = rate.detach().float().contiguous()
    if distributed and dist.is_available() and dist.is_initialized():
        flat = values.view(-1)
        local_size = torch.tensor([flat.numel()], dtype=torch.long, device=flat.device)
        sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
        dist.all_gather(sizes, local_size)
        lengths = [int(size.item()) for size in sizes]
        padded = flat.new_zeros(max(lengths))
        padded[:flat.numel()] = flat
        gathered = [torch.empty_like(padded) for _ in lengths]
        dist.all_gather(gathered, padded)
        if dist.get_rank() != 0:
            return None
        values = torch.cat([v[:n] for v, n in zip(gathered, lengths)])
        values = values.view(-1, *rate.shape[1:])
    return values.cpu().numpy()


def log_rate_distribution(rates, step, writer, logging):
    """Log every summary statistic and fixed rate bin without exporting files."""
    rates = np.asarray(rates, dtype=np.float32)
    flat = rates.reshape(-1)
    if flat.size == 0 or not np.isfinite(flat).all() or (flat <= 0.).any():
        raise ValueError('Posterior rates must be nonempty, finite and positive.')
    # Fixed edges cover exp([-5,5]) from Poisson.soft_clamp5, including
    # half precision roundoff. These are RATE bins, not Poisson count bins.
    edges = np.logspace(-3., 3., 61)
    counts, _ = np.histogram(flat, bins=edges)
    if counts.sum() != flat.size:
        raise ValueError('Rate histogram edges do not cover every value.')
    q = np.percentile(flat, [1, 5, 25, 50, 75, 95, 99])
    stats = dict(step=int(step), num_images=int(rates.shape[0]), num_rates=int(flat.size),
                 min=float(flat.min()), max=float(flat.max()),
                 mean=float(flat.mean(dtype=np.float64)), std=float(flat.std(dtype=np.float64)),
                 **dict(zip(['p01', 'p05', 'p25', 'p50', 'p75', 'p95', 'p99'], map(float, q))))
    writer.add_histogram('posterior_rate/group_001', flat, step, bins=edges)
    for name, value in stats.items():
        if name != 'step':
            writer.add_scalar('posterior_rate/group_001/' + name, value, step)
    logging.info('group1 posterior rate summary step %d images=%d N=%d '
                 'mean=%.9g std=%.9g min=%.9g max=%.9g '
                 'p01=%.9g p05=%.9g p25=%.9g p50=%.9g p75=%.9g p95=%.9g p99=%.9g',
                 step, stats['num_images'], flat.size, stats['mean'], stats['std'],
                 stats['min'], stats['max'], stats['p01'], stats['p05'], stats['p25'],
                 stats['p50'], stats['p75'], stats['p95'], stats['p99'])
    for index, (left, right, count) in enumerate(zip(edges[:-1], edges[1:], counts)):
        fraction = float(count) / flat.size
        tag = 'posterior_rate/group_001/bins/%03d_%.9g_to_%.9g' % (index, left, right)
        writer.add_scalar(tag + '/count', int(count), step)
        writer.add_scalar(tag + '/fraction', fraction, step)
        # Match numpy.histogram: half-open bins except the closed final bin.
        closing = ']' if index == len(counts) - 1 else ')'
        logging.info('group1 posterior rate bin step %d index=%03d '
                     'rate=[%.9g,%.9g%s count=%d fraction=%.9g',
                     step, index, left, right, closing, int(count), fraction)
    return stats

