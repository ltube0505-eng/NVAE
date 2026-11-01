# Poisson NVAE: semi-amortized inference for the first group only

Base: `feature/all-poisson-gradient-estimators`, commit
`777d4d7608294a6e87af7df409a31e949203bd66`.
Algorithm reference: [SA-VAE OptimN2N](https://github.com/harvardnlp/sa-vae/blob/master/optim_n2n.py)
and [image training](https://github.com/harvardnlp/sa-vae/blob/master/train_img.py).
The reverse-mode implementation follows `forward_mom` / `backward_mom`, with
no local gradient clipping (the OptimN2N class default). It recomputes base
gradients during backward instead of caching a full model-gradient copy per step.

## Forward and objectives

Number groups in **decoder order**, starting with the coarsest scale. Only group
1 (code index 0) has a free local variational parameter. All later groups retain
the original NVAE encoder/decoder-conditioned posterior computation.

The local coordinate is NVAE's unconstrained raw log-rate `a`, not the positive
rate itself. The actual posterior parameter is

```
lambda(a) = exp(min(5*tanh(a/5), log(poisson_max_rate))).
```

This retains the base branch's positivity, soft clamp, hard rate cap, sampling,
and analytic Poisson KL. It is momentum descent **in a**, not projected descent
in lambda. The hard cap's existing zero gradient above the cap is unchanged.

Start from `a_0 = encoder_group1(x)` and `v_0 = 0`. For k = 0,...,n-1:

1. Construct q1 using a_k; draw z1 with the configured Poisson ST/relaxed sampler.
2. Recompute every downstream conditional prior/posterior and resample all other
   latent groups. Keep the normal within-scale and cross-scale feature paths.
3. Decode the full hierarchy and compute the minibatch mean
   `f_k = NLL(x; z_1,...,z_G) + KL(q1(a_k) || p1)`.
4. `v_{k+1} = momentum*v_k - grad_a f_k`;
   `a_{k+1} = a_k + lr*v_{k+1}`.

The local KL coefficient is exactly **1**. No downstream KL, KL balancing,
annealing, or weight/spectral regularization enters the **inner** objective.
There are n updates, each with its own random draws. Encoder bottom-up features
are computed once; downstream conditionals are recomputed at every inner step.

After update n, make a **fresh full hierarchical pass** at a_n, so no stale
lower-group parameters/samples enter the outer loss. The unweighted NELBO is
`mean(NLL + sum_g KL_g)`. The normal `train.py` outer loss retains the base NVAE
KL annealing, detached group balancing coefficients, spectral/BN regularization,
weight decay, and Adamax update. `train/refined_nelbo` separately records the
unweighted refined NELBO; `train/nelbo_iter` follows the base annealed/balanced
reporting convention. Only after outer backward does the model optimizer step.

## Backward

For the outer loss initialize `bar_a_n = dL/da_n`, `bar_v_n = 0`. In reverse order:

```
u = bar_v_{k+1} + lr * bar_a_{k+1}
bar_a_k = bar_a_{k+1} - H_aa(f_k) u
bar_theta += -H_theta,a(f_k) u
bar_v_k = momentum * u
```

Each Hessian-vector product uses the SA-VAE one-sided difference
`(grad f_k(a_k + eps*u) - grad f_k(a_k))/eps`. CPU and CUDA RNG states are replayed
for **all groups**, without advancing the outer RNG stream during backward.
The ordinary outer NELBO gradients are automatically added to these indirect
ones. The initial-posterior adjoint reaches the encoder; mixed derivatives also
reach downstream posterior heads, combiners, encoder feature inputs, and decoder
parameters. No model parameter is updated by the inner loop. This is not a
stop-gradient fine-tuning scheme or a first-order meta-gradient approximation.

Finite differences are approximate. With ST Poisson samples these are finite
differences of the base branch's **surrogate gradient field**, not unbiased
second derivatives of an exact discrete expectation. Count-threshold crossings,
small fp32 epsilon, or large directions can make the estimate noisy. The default
epsilon matches OptimN2N (`1e-5`) and is configurable. Neither inner loss nor
NELBO is guaranteed to decrease on every stochastic step. This implementation
does not claim numerical identity with a Gaussian SA-VAE run.

Full precision and unscaled outer backward are used when SVI is enabled. Inner
passes and backward replays do not accumulate BatchNorm running-stat updates;
the bottom-up encoder and final decoder pass update their buffers normally.
Data-dependent decoder initialization runs once before the first trajectory.
Unconditional prior sampling is unchanged. Validation under `torch.no_grad()`
still enables local gradients, then uses the original **exact Poisson** final
sampling path; no model `.grad` is accumulated in validation.

## Options and training

- `--group1_svi_steps 0`: default; preserve ordinary NVAE behavior and AMP.
- `--group1_svi_steps n`: enable only group1, with n positive integer updates.
- `--group1_svi_lr 0.1`, `--group1_svi_momentum 0.5`, `--group1_svi_eps 1e-5`.
- Enabled SVI requires `--latent_distribution poisson`, ST or relaxed gradients,
  and no normalizing flows. REINFORCE/O-BBVI remain available with n=0; combining
  their score-function objectives with this pathwise inner optimizer is rejected.

```bash
conda activate nvae
cd /root/autodl-tmp/NVAE-poisson-group1-svi
git fetch origin
git switch feature/poisson-group1-semi-amortized
git pull --ff-only origin feature/poisson-group1-semi-amortized
mkdir -p logs
set -o pipefail
CUDA_VISIBLE_DEVICES=0 MASTER_PORT=6031 bash scripts/train_mnist_poisson_group1_svi.sh \
  --group1_svi_steps 5 \
  2>&1 | tee logs/mnist_poisson_group1_svi_s2_g2-4_n5.log
```

Run inside `screen -S nvae_group1_svi` for a persistent remote terminal. The
launcher defaults to MNIST, 200 epochs, batch 64, 16 channels, 8 latent channels
per group, Poisson ST temperature 0.1, and the base NVAE KL schedule. The base
training hyperparameters match the supplied 2026-09-23 run:

| Option | Value |
|---|---|
| `use_se`, `res_dist` | `True`, `True` |
| `min_groups_per_scale` | 1 (encoder groups still `[4,2]`) |
| `weight_decay`, `weight_decay_norm` | 0.0003, 0.0 |
| `weight_decay_norm_anneal`, `weight_decay_norm_init` | `False`, 10.0 |
| `fast_adamax`, `cont_training` | `False`, `False` |
| `num_mixture_dec`, `num_x_bits`, `seed` | 10, 8, 1 |
| KL anneal / constant portion / constant coefficient | 0.3 / 0.0001 / 0.0001 |

The model has 2,104,790 trainable parameters (2.104790M); local refinement adds
no learned parameters. With the same 60,000-image MNIST train loader and one
GPU, batch 64 gives 937 batches per epoch and 187,400 total optimizer steps.
`num_total_iter` is computed from the loader; it is not a CLI argument.
The new save name prevents reuse of the baseline run's output directory. The
SVI-specific options and full-precision execution are additional to that baseline;
all pre-existing training options above follow the supplied configuration.

| Decoder scale | Spatial size | Groups | Locally optimized groups |
|---|---|---|---|
| 1 (top) | 4 x 4 | 2 | first only |
| 2 | 8 x 8 | 4 | none |

NVAE prints encoder-order `groups_per_scale=[4,2]`; this represents decoder-order
`[2,4]`, not a reversed configuration. The default checkpoint is
`/root/autodl-tmp/nvae-checkpoints/eval-mnist_poisson_group1_svi_s2_g2-4_n5/checkpoint.pt`.
All SVI settings are saved in checkpoint args and restored by `evaluate.py`.

For CPU smoke training use `--device cpu --arch_instance res_elu`. The original
CUDA SyncBatchNorm architecture is retained for GPU runs; it is not emulated by
the CPU path. `--max_train_batches` and `--max_valid_batches` default to 0 (full
loader), and evaluation sample defaults remain 10 / 1000. Bounded runs still
save normal checkpoints and TensorBoard events. They are not full-data training.

## Verification

`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests -q`

Tests cover smooth stochastic meta-gradients against exact differentiable
unrolling (including encoder features and direct/indirect parameter paths), RNG
preservation, local objective membership, all six groups being recomputed at
each iteration, finite gradients and actual encoder/decoder parameter updates,
validation under no-grad, no repeated BN buffer updates, n=0, prior sampling,
and the existing Normal/Gamma/Poisson estimator regression suite.
