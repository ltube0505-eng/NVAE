# Vanilla Poisson VAE with a learnable prior

This branch extends `feature/poisson-identity-ste-no-soft-cap`. Select the
existing vanilla path with:

```bash
--latent_distribution poisson \
--num_latent_scales 1 \
--num_groups_per_scale 1 \
--num_nf 0 \
--poisson_gradient_estimator straight_through
```

No additional prior flag is needed. Only the single-scale, single-group Poisson
model gets the learnable prior. Hierarchical models and Normal latents retain
their previous prior structure. All Poisson priors and posteriors now use the
reference `soft_clamp5`: `rate = exp(5 * tanh(raw_log_rate / 5))`.
Identity STE (`dz/drate = 1`) is unchanged; no rate=30 hard cap is restored.

## Prior and initialization

`prior_log_rate` is an `nn.Parameter` of shape `[1, C, H, W]`, where
`C = num_latent_per_group` and `H = W = input_size / 2**num_preprocess_blocks`.
Each latent position has its own unconditional prior rate; these parameters are
shared across images, not across spatial positions. The prior is

```text
p(z) = product_{c,h,w} Poisson(z[c,h,w]; exp(5 * tanh(prior_log_rate[0,c,h,w] / 5)))
```

Every log-rate starts at exactly zero, so every initial rate is exactly one.
**The initial prior distribution is exactly the base branch's Poisson(1).**
The existing `prior_ftr0` random initialization is unchanged; that feature tensor
is not the Poisson rate and is not used to decode in the vanilla path.

Both `forward()` (analytic KL and sampled log-p) and `sample()` use `_prior0()`.
The training optimizer already receives `model.parameters()`, so it updates the
new parameter without a separate optimizer. For each log-rate `a`,
`p_rate = exp(5*tanh(a/5))` and
`d KL(Poisson(q_rate) || Poisson(p_rate)) / da = (p_rate - q_rate) * (1 - tanh(a/5)^2)`.
Poisson sampling temperature behavior is unchanged from the base branch.

The soft-cap adds no parameters or command-line flags. Existing learnable-prior
checkpoint shapes still load, but nonzero raw values now define different rates;
resuming or evaluating an uncapped checkpoint uses the new parameterization.
Start a fresh run for a controlled comparison.

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

The vanilla tests cover initialization equality, Cell counts, the analytic prior
gradient and an optimizer update, learned log-p during forward evaluation,
generation with learned rates, strict state-dict roundtrip, and unchanged priors
for other model configurations. Model CPU tests use `res_elu` and disable
distributed data-dependent convolution initialization, as the existing tests do.
