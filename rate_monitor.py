"""Periodic distribution snapshots of every group-1 posterior rate."""
import csv
import os

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


def write_rate_snapshot(rates, save_dir, step, writer, logging):
    """Persist raw rates, fixed log-bin histogram, summary CSV and PNG."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    rates = np.asarray(rates, dtype=np.float32)
    flat = rates.reshape(-1)
    if flat.size == 0 or not np.isfinite(flat).all() or (flat <= 0.).any():
        raise ValueError('Posterior rates must be nonempty, finite and positive.')
    folder = os.path.join(save_dir, 'posterior_rate_monitor', 'group_001')
    os.makedirs(folder, exist_ok=True)
    prefix = os.path.join(folder, 'step_%09d' % step)
    # Fixed edges cover exp([-5,5]) from Poisson.soft_clamp5, including
    # half precision roundoff. These are RATE bins, not Poisson count bins.
    edges = np.logspace(-3., 3., 61)
    counts, _ = np.histogram(flat, bins=edges)
    if counts.sum() != flat.size:
        raise ValueError('Rate histogram edges do not cover every value.')
    np.savez_compressed(prefix + '.npz', rates=rates, bin_edges=edges, counts=counts)
    with open(prefix + '_histogram.csv', 'w', newline='') as stream:
        rows = csv.writer(stream)
        rows.writerow(['rate_left', 'rate_right', 'count', 'fraction'])
        rows.writerows((left, right, int(n), float(n) / flat.size)
                       for left, right, n in zip(edges[:-1], edges[1:], counts))
    q = np.percentile(flat, [1, 5, 25, 50, 75, 95, 99])
    stats = dict(step=int(step), num_images=int(rates.shape[0]), num_rates=int(flat.size),
                 min=float(flat.min()), max=float(flat.max()),
                 mean=float(flat.mean(dtype=np.float64)), std=float(flat.std(dtype=np.float64)),
                 **dict(zip(['p01', 'p05', 'p25', 'p50', 'p75', 'p95', 'p99'], map(float, q))))
    summary = os.path.join(folder, 'summary.csv')
    new_file = not os.path.exists(summary) or os.path.getsize(summary) == 0
    with open(summary, 'a', newline='') as stream:
        rows = csv.DictWriter(stream, fieldnames=list(stats))
        if new_file:
            rows.writeheader()
        rows.writerow(stats)
    figure = Figure(figsize=(7, 4), tight_layout=True)
    FigureCanvasAgg(figure)
    ax = figure.add_subplot(111)
    ax.bar(edges[:-1], counts / float(flat.size), width=np.diff(edges), align='edge')
    ax.set_xscale('log')
    ax.set_xlabel('Posterior Poisson rate (log scale)')
    ax.set_ylabel('Fraction of all rates')
    ax.set_title('Group 1, step %d, %d images, %d rates' %
                 (step, rates.shape[0], flat.size))
    figure.savefig(prefix + '.png', dpi=130)
    writer.add_histogram('posterior_rate/group_001', flat, step, bins=edges)
    for name, value in stats.items():
        if name != 'step':
            writer.add_scalar('posterior_rate/group_001/' + name, value, step)
    logging.info('group1 posterior rate step %d N=%d mean=%.6g std=%.6g '
                 'min=%.6g p05=%.6g p50=%.6g p95=%.6g max=%.6g',
                 step, flat.size, stats['mean'], stats['std'], stats['min'],
                 stats['p05'], stats['p50'], stats['p95'], stats['max'])
    return stats
