# Posterior-rate-only TensorBoard monitoring

Run `train_poisson_posterior_rate.py` instead of `train.py`, keeping your training
arguments. This entry point supports one GPU/process, Poisson straight-through
latents, and `num_nf=0`. Both vanilla and hierarchical all-Poisson models are
supported; all groups and all channel/height/width positions are recorded.

The monitor reads the actual posterior distributions from the existing training
forward pass. It performs no extra sampling, forward passes, gradient probes,
or backward passes and stores no autograd graphs. Prior rates, raw parameters,
latent samples, gradients, sampling derivatives, losses, images, and histograms
are not written to its TensorBoard event stream. Training progress and NELBO
remain available in the ordinary `log.txt`.

## Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--rate-every` | `100` | Record at global steps divisible by this value, plus the first step after launch/resume. |
| `--rate-position-images` | `1` | Record every latent position for the first N current minibatch slots; use 0 for batch means only. |
| `--rate-resume` | unset | Resume a trusted `eval-*/checkpoint.pt`, restoring its saved training settings. |

The equivalent `--grad-every`, `--grad-position-images`, and `--grad-resume`
spellings from the reference script are accepted as aliases. No other gradient
monitoring flags are supported.

Example fresh run (one group, 8 channels, 4x4 latent map on padded MNIST):

```bash
python train_poisson_posterior_rate.py \
  --root /root/autodl-tmp/nvae-checkpoints \
  --save mnist_poisson_bernoulli_rate_only \
  --data /root/autodl-tmp/datasets/mnist --dataset mnist \
  --latent_distribution poisson --poisson_gradient_estimator straight_through \
  --num_latent_scales 1 --num_groups_per_scale 1 --num_latent_per_group 8 \
  --num_channels_enc 16 --num_channels_dec 16 --num_nf 0 \
  --num_preprocess_blocks 3 --num_preprocess_cells 2 \
  --num_postprocess_blocks 3 --num_postprocess_cells 2 \
  --num_cell_per_cond_enc 1 --num_cell_per_cond_dec 1 \
  --arch_instance res_mbconv --batch_size 64 --epochs 200 \
  --learning_rate 0.001 --learning_rate_min 0.0001 --fast_adamax \
  --kl_anneal_portion 0.3 --kl_const_coeff 0.0001 --kl_const_portion 0.0001 \
  --rate-every 100 --rate-position-images 1
```

Resume without re-entering the architecture or optimizer options:

```bash
python train_poisson_posterior_rate.py \
  --rate-resume /root/autodl-tmp/nvae-checkpoints/eval-mnist_poisson_bernoulli_rate_only/checkpoint.pt \
  --rate-every 100 --rate-position-images 1
```

Saved training arguments take precedence over command-line training settings
with `--rate-resume`; only the monitor options above are overridden. Model,
optimizer, scheduler, AMP scaler, epoch, and global step are restored by the
training loop. An already completed run is rejected. The checkpoint does not
save RNG or data-loader state, so resumption is not a bitwise replay.

## TensorBoard tags

Events are written to `RUN/posterior_rates`, separating them from any existing
gradient/loss event files in `RUN`. Point TensorBoard directly at that subfolder
to see only posterior rates, including when resuming a previously monitored run:

```bash
tensorboard \
  --logdir /root/autodl-tmp/nvae-checkpoints/eval-mnist_poisson_bernoulli_rate_only/posterior_rates \
  --host 127.0.0.1 --port 16084 --load_fast=false
```

Example tags (the reference script's prefix is retained for existing filters):

```text
grad_positions/current_batch/sample_0/posterior/q0/rate/c00/h0_w0
grad_positions/current_batch/batch_mean/posterior/q0/rate/c00/h0_w0
```

`q0`, `q1`, ... follow the model's latent-group order. `sample_0` means the first
image in the **current** minibatch, not a fixed image across steps. `batch_mean`
averages over images at each latent position; it does not average over latent
positions. The redundant `batch_mean_abs` view is omitted because rates are
positive. With 8x4x4 latents and the default one image, there are 256 rate curves.
To record every image in each batch, set `--rate-position-images` to the batch
size (this multiplies the event volume).

On resume, the first recorded step is the checkpoint's `global_step`, followed
by the next multiple of `--rate-every`. TensorBoard purges stale events at or
after the restored global step within this rate-only stream.

## Running alongside another NVAE job

Training uses TCP port `6020` by default for NCCL initialization. Set
`MASTER_PORT` to a different free port for this job if another NVAE process
uses the default. For example, prefix the training command with
`CUDA_VISIBLE_DEVICES=0 MASTER_PORT=16091`. This port is separate from
the TensorBoard HTTP port. Both processes share GPU memory and compute if
they use the same device.
