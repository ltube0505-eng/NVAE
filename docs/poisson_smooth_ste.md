# Smooth low-rate STE with adaptive CTS and diagonal Fisher

At small rates, the CTS derivative can be almost zero on most arrival paths,
with rare large values. This experiment blends its rate derivative with the
constant STE derivative 1. It retains the adaptive arrival sampler and the
existing rate parameterization `lambda = exp(5 * tanh(a / 5))`.

## Algorithm

For each latent coordinate, let `G = d soft_count / d lambda`. Set

```text
w(lambda) = lambda**p / (lambda**p + lambda0**p)
G_blend   = (1 - w) + w * G
```

The default transition scale is `lambda0 = 0.35`, with power `p = 4` and the
existing default sigmoid temperature `tau = 0.1`. These are experimental
starting values, not a trained or optimal cutoff. The scale is the rate at which
both derivatives have weight 1/2. There is no discontinuous low-rate branch.

| Rate | Weight on STE, 1-w | Weight on CTS, w |
| --- | --- | --- |
| 0.10 | 0.99338 | 0.00662 |
| 0.20 | 0.90365 | 0.09635 |
| 0.35 | 0.50000 | 0.50000 |
| 0.50 | 0.19361 | 0.80639 |
| 1.00 | 0.01478 | 0.98522 |

More generally, at least a fraction `rho` of STE weight is used when
`lambda <= lambda0 * ((1-rho)/rho)**(1/p)`. With `rho = 0.9`, this means
`lambda <= 0.20207`. This describes the transition, not a threshold used by the
code. Retune the scale when changing temperature or task.

Implementation uses the numerically stable equivalent

```python
w = sigmoid(p * (log_rate.detach() - log(lambda0)))
surrogate = (1 - w) * rate + w * soft_count
z = forward_count.detach() + (surrogate - surrogate.detach())
```

`forward_count` is the hard integer count for `straight_through`, or the original
soft count for `relaxed`. The forward value and RNG stream are unchanged.
Detaching the weight is essential: differentiating it would add the undesired
`w'(lambda) * (soft_count - lambda)` term. Half/bfloat16 weights are calculated
in FP32 and then cast back, avoiding overflow from explicitly taking powers.
The custom backward is a first-order surrogate, not the true derivative of the
stored hard forward value; ordinary finite-difference gradcheck is inappropriate.

The derivative with respect to the raw encoder output remains

```text
d z / d a = G_blend * lambda * sech(a/5)**2
```

Thus `G_blend -> 1` at small rates does not make the raw-output derivative 1,
remove the tanh Jacobian, or guarantee recovery of saturated/collapsed units.

## Statistical interpretation and Fisher interaction

For fixed rate and temperature, the weight is deterministic, so

```text
E[G_blend]   = 1 + w * (E[G] - 1)
Var[G_blend] = w**2 * Var[G]
```

These statements concern the coordinate derivative G. A reconstruction gradient
also contains `d loss / d z`, which is random and generally correlated with G.
The blend can therefore change the mean objective gradient and need not reduce
its variance. It is a biased surrogate estimator, not an unbiased control
variate. For adaptive finite CTS, do not assume `E[G]` is exactly 1.

Analytical KL and log-probability formulas, including their direct derivatives
at fixed parent inputs, are unchanged. In the hierarchy, downstream KL terms
and log rates can depend on sampled ancestors; those paths use the blend too.

The diagonal Fisher optimizer continues to estimate the same form
`diag(J.T @ diag(lambda) @ J) / batch_size`. Its Jacobian J now includes the
blended derivatives through ancestor samples. Its detached Poisson rate weights,
probe distribution, damping and update budget are unchanged. This remains the
surrogate output metric described in [the Fisher notes](poisson_diagonal_fisher.md),
not the exact joint or marginal model Fisher.

## Enable and compare

Append these options to your existing all-Poisson training command:

```bash
--latent_distribution poisson \
--poisson_gradient_estimator straight_through \
--poisson_relaxation_temperature 0.1 \
--poisson_smooth_ste \
--poisson_ste_rate_scale 0.35 \
--poisson_ste_power 4 \
--optimizer poisson_fisher
```

Keep the architecture, dataset, seed, KL schedule, learning-rate schedule and
Fisher parameters matched to the baseline. Omit `--poisson_smooth_ste` for the
original CTS backward. The option also supports relaxed sampling and Adamax;
in a mixed Poisson/Gamma model it only changes the Poisson groups. The model
rejects it for normal-only latents and score-function estimators.

All new configuration values are saved in checkpoint args by the existing
training code. No model parameters or buffers are added. Old checkpoints without
these args default to the original backward; evaluation's exact sampler is
unchanged. When resuming training, pass the desired CLI blend settings again,
as the existing training entry builds its model from current CLI args. For a
new Fisher experiment initialized from an old checkpoint, use a separate save
directory and consider `--reset_optimizer` so the curvature EMA is re-estimated
for the new ancestor Jacobian. To continue the same experiment, keep settings
and optimizer state. See the Fisher notes for scheduler/resume semantics.

## Validation

Run `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests`.
Tests cover the analytical blended rate derivative and tanh chain rule on fixed
arrival paths, unchanged forward samples and RNG, empirical variance contraction,
direct KL gradients, exact/score sampling, low-precision weight stability,
CLI-to-model wiring, old state_dict loading, invalid configurations, the
hierarchical Fisher Jacobian, and CPU model/training/Fisher update integration.
These checks do not establish improved ELBO or CUDA/distributed performance.

## TensorBoard gradient monitoring

Use `train_poisson_grad_monitor.py` as the training entry point for the
single-GPU MNIST experiment with one scale, two 4×4 Poisson ST groups and no
normalizing flows. The monitor works with Fisher or Adamax and supports a CTS
baseline run with the blend disabled.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -u train_poisson_grad_monitor.py \
  --grad-every 100 --grad-summary --grad-hist-every 1000 --grad-position-images 1 \
  --root /root/autodl-tmp/nvae-checkpoints \
  --save mnist_poisson_fisher_smooth_ste_s1_g2_4x4 \
  --dataset mnist --data /root/autodl-tmp/datasets/mnist \
  --arch_instance res_mbconv --batch_size 64 --epochs 200 --warmup_epochs 5 \
  --learning_rate 0.001 --learning_rate_min 0.0001 \
  --weight_decay 0.0003 --weight_decay_norm 0 \
  --kl_anneal_portion 0.3 --kl_const_portion 0.0001 --kl_const_coeff 0.0001 \
  --latent_distribution poisson --poisson_gradient_estimator straight_through \
  --poisson_relaxation_temperature 0.1 --poisson_cts_tail_tolerance 0.0001 \
  --poisson_smooth_ste --poisson_ste_rate_scale 0.35 --poisson_ste_power 4 \
  --optimizer poisson_fisher --fisher_damping 0.01 --fisher_decay 0.95 \
  --fisher_probes 1 --fisher_step_budget 0.01 \
  --num_latent_scales 1 --num_groups_per_scale 2 --num_latent_per_group 8 \
  --num_nf 0 --num_channels_enc 16 --num_channels_dec 16 \
  --num_preprocess_blocks 3 --num_preprocess_cells 2 \
  --num_postprocess_blocks 3 --num_postprocess_cells 2 \
  --num_cell_per_cond_enc 1 --num_cell_per_cond_dec 1 --num_mixture_dec 10 \
  --num_process_per_node 1 --seed 1
```

```bash
tensorboard --logdir /root/autodl-tmp/nvae-checkpoints/eval-mnist_poisson_fisher_smooth_ste_s1_g2_4x4 \
  --host 0.0.0.0 --port 6016
```

Position tags start with `grad_positions/current_batch/{view}/` and end with a
coordinate such as `/c02/h1_w1`. The views `batch_mean` and `batch_mean_abs`
average over images and retain the channel, row and column. Each `sample_N`
refers to a slot in the current batch. Its input image is logged at the same
step. The first batch and subsequent steps divisible by `--grad-every` are
recorded.

| Tag after the view | Recorded quantity |
| --- | --- |
| `posterior/q{0,1}/{raw,rate,z}` | Raw posterior input, rate, and integer ST forward count |
| `prior/p1/{raw,rate}` | Conditional prior input and rate |
| `sampling/q{0,1}/{cts_weight,ste_weight}` | Weights on the original CTS and unit derivatives |
| `sampling/q{0,1}/cts_dz_drate` | Original CTS derivative \(G\) on the training arrivals |
| `sampling/q{0,1}/dz_drate` | Rate derivative used in backward, \(G_{\mathrm{blend}}\) when enabled |
| `sampling/q{0,1}/dz_draw` | Backward derivative with respect to the raw posterior input |
| `gradient/{recon,weighted_kl,raw_kl0,raw_kl1,total}/q{0,1}/{raw,rate,z}` | Signed batch-mean loss gradients before the optimizer update |

The same objective gradients are recorded for connected `p1/{raw,rate}` nodes.
The fixed prior for group 0 has rate 1. A disconnected gradient is marked with
`grad_monitor/.../connected=0`. Nonfinite coordinate points are omitted and
counted in `grad_monitor/positions/{view}/.../nonfinite_fraction`. Under
`grad_monitor/`, the scalar tags also include the actual CTS horizon and term
count for each posterior group, the near-zero tanh slope fraction, the loss
components, and the current Fisher quadratic step and step scale. With
`--grad-summary`, tensor summaries are added and finite-value histograms follow
`--grad-hist-every` (0 disables histograms).

The probes use the training forward pass and its exact arrivals. They retain
the graph for the regular training backward and add work on the monitored
steps. The monitored derivatives describe the loss before Fisher
preconditioning. To resume with the saved model, optimizer, scaler, scheduler
and configuration, stop the previous writer and run the same entry point
with only the monitor options and `--grad-resume /path/to/eval-RUN/checkpoint.pt`.
The restored step is recorded and used as TensorBoard's purge step. Checkpoints
restore epoch and global step. The loader position and RNG states are outside
the checkpoint format.

Run the CPU monitor self-test with
`python train_poisson_grad_monitor.py --grad-self-test`.
