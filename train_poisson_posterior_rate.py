#!/usr/bin/env python3
"""Single-GPU Poisson ST training with only per-position posterior rates in TensorBoard.

Use this entry point instead of train.py with the same training arguments.
Logs batch_mean and sample_0 by default, every 100 steps and on the first step
after launch/resume. sample_N is a current minibatch slot, not a fixed image.
No extra forwards, sampling, gradient probes, or retained graphs are needed.
Events go to RUN/posterior_rates; ordinary progress remains in RUN/log.txt.
"""

import copy
import inspect
from pathlib import Path


class RateOnlyWriter:
    """Keep train.py's ordinary metrics/images out of the rate-only event stream."""

    def __init__(self, save):
        self.log_dir = str(Path(save) / 'posterior_rates')
        self.writer = None

    def open(self, purge_step=None):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=20,
                                    purge_step=purge_step)

    def add_rate(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def add_scalar(self, *args, **kwargs):
        pass

    add_image = add_scalar
    add_figure = add_scalar
    add_histogram = add_scalar
    add_histogram_if = add_scalar

    def flush(self):
        if self.writer is not None:
            self.writer.flush()

    def close(self):
        if self.writer is not None:
            self.writer.close()


class PosteriorRateMonitor:
    def __init__(self, writer, every=100, position_images=1):
        if every < 1 or position_images < 0:
            raise ValueError('every must be positive and position_images must be nonnegative.')
        self.writer = writer
        self.every = every
        self.position_images = position_images
        self.first_step = True
        self.active = False

    def begin(self, step):
        self.step = int(step)
        self.active = self.first_step or self.step % self.every == 0
        self.first_step = False

    def record(self, posteriors):
        if not self.active:
            return
        for group, posterior in enumerate(posteriors):
            # Copy only when logging. No live tensors are stored on the monitor.
            rate = posterior.rate.detach().float().cpu()
            if rate.dim() != 4:
                raise ValueError('Posterior rate must have shape [B,C,H,W].')
            views = [('batch_mean', rate.mean(dim=0))]
            views.extend(('sample_%d' % index, rate[index])
                         for index in range(min(self.position_images, rate.size(0))))
            for view, grid in views:
                for channel, plane in enumerate(grid.tolist()):
                    for row, entries in enumerate(plane):
                        for column, value in enumerate(entries):
                            self.writer.add_rate(
                                'grad_positions/current_batch/%s/posterior/q%d/rate/c%02d/h%d_w%d'
                                % (view, group, channel, row, column), value, self.step)
        self.writer.flush()
        self.active = False


def load_checkpoint(path):
    """Load a trusted NVAE checkpoint, including its saved argparse.Namespace."""
    import torch
    kwargs = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = False
    return torch.load(str(path), **kwargs)


def add_monitor_arguments(parser):
    parser.add_argument('--rate-every', '--grad-every', dest='rate_every', type=int, default=100,
                        help='Log every N global steps, plus the first step after launch/resume.')
    parser.add_argument('--rate-position-images', '--grad-position-images', dest='rate_position_images',
                        type=int, default=1,
                        help='Current batch slots to log; 0 records only batch_mean.')
    parser.add_argument('--rate-resume', '--grad-resume', dest='rate_resume',
                        help='Trusted eval-*/checkpoint.pt; saved training settings take precedence.')


def configure_args(current, parser):
    if current.rate_every < 1 or current.rate_position_images < 0:
        parser.error('--rate-every must be positive; --rate-position-images must be >= 0.')
    if current.num_process_per_node != 1 or current.num_proc_node != 1:
        parser.error('Posterior rate monitoring supports exactly one GPU/process.')
    if current.rate_resume:
        path = Path(current.rate_resume).resolve()
        if path.name != 'checkpoint.pt' or not path.parent.name.startswith('eval-'):
            parser.error('--rate-resume requires eval-*/checkpoint.pt.')
        checkpoint = load_checkpoint(path)
        required = {'args', 'state_dict', 'optimizer', 'scheduler', 'grad_scalar', 'epoch', 'global_step'}
        missing = required.difference(checkpoint)
        if missing:
            parser.error('Incomplete checkpoint; missing ' + ', '.join(sorted(missing)))
        if getattr(checkpoint['args'], 'image_likelihood', 'dataset_default') != 'dataset_default':
            parser.error('This branch requires a checkpoint trained with the dataset-default likelihood.')
        options = {key: getattr(current, key)
                   for key in ('rate_every', 'rate_position_images', 'rate_resume')}
        # Keep new parser defaults if the saved checkpoint predates a flag.
        vars(current).update(vars(copy.deepcopy(checkpoint['args'])))
        vars(current).update(options)
        current.root = str(path.parent.parent)
        current.save = path.parent.name[len('eval-'):]
        current.cont_training = True
        if checkpoint['epoch'] >= current.epochs:
            parser.error('Checkpoint has already completed its configured training epochs.')
        print('Resume checkpoint: %s; completed epochs=%d; global_step=%d' %
              (path, checkpoint['epoch'], checkpoint['global_step']), flush=True)
    if current.num_process_per_node != 1 or current.num_proc_node != 1:
        parser.error('Posterior rate monitoring requires a single-GPU checkpoint.')
    if (current.latent_distribution != 'poisson' or
            current.poisson_gradient_estimator != 'straight_through' or current.num_nf != 0):
        parser.error('Use --latent_distribution poisson --poisson_gradient_estimator straight_through --num_nf 0.')
    current.global_rank = current.local_rank = current.node_rank = 0
    current.posterior_rate_only = True
    return current


if __name__ == '__main__':
    from train import main_cli
    main_cli(posterior_rate_only=True)

