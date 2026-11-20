# Vanilla Poisson VAE with a learnable prior

This branch ports the residual/uniform-prior and BatchNorm-free changes onto
`feature/poisson-vanilla-fixed-gaussian-likelihood`. Select the vanilla path with:

```bash
--latent_distribution poisson \
--num_latent_scales 1 \
--num_groups_per_scale 1 \
--num_nf 0 \
--poisson_gradient_estimator straight_through \
--arch_instance res_elu \
--image_likelihood gaussian_fixed --gaussian_variance 0.01
```

No additional prior flag is needed. Only the single-scale, single-group Poisson
model gets the learnable prior. Hierarchical models and Normal latents retain
their previous prior structure. All Poisson priors and posteriors use the
non-soft-cap parameterization `rate = exp(raw_log_rate)`, without tanh or clamps.
Identity STE (`dz/drate = 1`) is unchanged; no rate=30 hard cap is restored.
Its forward pass samples an exact integer Poisson count, and its backward pass
has `dz/d(raw_log_rate) = rate`. Use `--poisson_gradient_estimator straight_through`
as shown above; this does not use the CTS surrogate.

## Prior and initialization

`prior_log_rate` is an `nn.Parameter` of shape `[1, C, H, W]`, where
`C = num_latent_per_group` and `H = W = input_size / 2**num_preprocess_blocks`.
Each latent position has its own unconditional prior rate; these parameters are
shared across images, not across spatial positions. The prior is

```text
p(z) = product_{c,h,w} Poisson(z[c,h,w]; exp(prior_log_rate[0,c,h,w]))
```

Every element of `prior_log_rate` is initialized independently from
`Uniform(-6, -2)`. The actual prior rate is `exp(prior_log_rate)`, initially
in `[exp(-6), exp(-2)]`, approximately `[0.002479, 0.135335]`.
The existing `prior_ftr0` random initialization is unchanged; that feature tensor
is not the Poisson rate and is not used to decode in the vanilla path.

The vanilla Poisson encoder predicts a residual `delta(x)`, using the same
posterior head and its existing initialization:

```text
q_raw_log_rate(x) = prior_log_rate + delta(x)
q_rate(x) = exp(q_raw_log_rate(x)) = p_rate * exp(delta(x))
```

This residual parameterization is always active for the vanilla Poisson model,
including when `--res_dist` is omitted. That flag retains its existing meaning
for hierarchical groups. A zero residual makes the posterior equal to the
prior. Gradients flow through both `delta(x)` and `prior_log_rate`; the prior
is not detached. This residual change is limited to vanilla Poisson models;
the non-soft-cap rate mapping applies to all Poisson distributions. Normal and
Gamma parameterizations are unchanged.

Both `forward()` (analytic KL and sampled log-p) and `sample()` use `_prior0()`.
The training optimizer already receives `model.parameters()`, so it updates the
parameter without a separate optimizer. Holding the posterior rate fixed,
for each raw prior log-rate `a`,
`p_rate = exp(a)` and
`d KL(Poisson(q_rate) || Poisson(p_rate)) / da = p_rate - q_rate`.
Poisson sampling temperature behavior is unchanged from the base branch.
With the residual posterior, the total prior gradient additionally includes
the posterior's dependence on the shared `prior_log_rate`.

No parameters or command-line flags are added by residual parameterization.
Existing learnable-prior checkpoint shapes still load, and loading overwrites
the random prior initialization. However, old encoder outputs are now interpreted
as residuals and Poisson rates use exp directly, so old distributions change.
Start a fresh run for a controlled comparison.

The default architecture is now `res_elu`; it replaces the former MBConv/Swish
blocks with the existing ELU convolution blocks and contains no BatchNorm.
Weight normalization, spectral regularization, and one-time data-dependent
initialization remain. Loading saved convolution parameters skips reinitialization.
An old `res_mbconv` checkpoint is not architecture-compatible with `res_elu`;
`--rate-resume` restores its saved architecture rather than converting it.

The target branch's Gaussian behavior is retained: grayscale MNIST in `[0, 1]`,
`mu = sigmoid(logits)`, fixed variance (default `0.01` when selected), Gaussian
log-density including its normalizer, and mean images for rendering/export.
The likelihood still requires `--image_likelihood gaussian_fixed`; the CLI
likelihood default remains `dataset_default`. Existing posterior-rate monitoring
and Gaussian checkpoint likelihood/variance checks are unchanged.

New checkpoints include `prior_log_rate` and reload with the same architecture.
An old fixed-prior vanilla checkpoint lacks that parameter, so it is not a
drop-in strict model/optimizer resume; use a fresh run for this experiment.

## Cell counts

The vanilla condition empties `enc_tower` and `dec_tower`, but does not remove
preprocessing or postprocessing. Their total Cell counts are:

| Part | Total Cells | Resizing Cells | Other Cells |
| --- | --- | --- | --- |
| Preprocessing | `num_preprocess_blocks * num_preprocess_cells` | One `down_pre` per block | Remaining `normal_pre` |
| Postprocessing | `num_postprocess_blocks * num_postprocess_cells` | One `up_post` per block | Remaining `normal_post` |

With `train.py` defaults (2 blocks, 3 cells per block), each side has **6 Cells**:
4 normal + 2 down on the encoder side, and 2 up + 4 normal on the decoder side.
For padded 32x32 MNIST this gives an 8x8 latent map.

For a 4x4 MNIST latent map, explicitly use:

```bash
--num_preprocess_blocks 3 --num_preprocess_cells 2 \
--num_postprocess_blocks 3 --num_postprocess_cells 2
```

This also gives **6 Cells on each side**, but now 3 normal + 3 down and
3 up + 3 normal, respectively. With 8 latent channels, the prior has
`8 * 4 * 4 = 128` learnable log-rates. Stem convolutions, the posterior head,
`enc0`, `stem_decoder`, and the output head are not counted as `Cell` objects.

## Validation

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests
```

The vanilla tests cover elementwise prior initialization bounds, Cell counts,
residual posterior rates and gradients through both the encoder and shared prior,
the analytic fixed-posterior prior gradient and an optimizer update, learned log-p during forward evaluation,
generation with learned rates, strict state-dict roundtrip, and unchanged priors
for other model configurations. Model CPU tests use `res_elu` and disable
distributed data-dependent convolution initialization, as the existing tests do.
