# Annealed posterior entropy: L_alpha

This branch starts from `feature/all-poisson-gradient-estimators` at
`777d4d7608294a6e87af7df409a31e949203bd66`.

## Objective

For all-Poisson models, `train.py` now defaults to
`--poisson_objective nvae_alpha`. The minimized training loss is

```
loss(x, step) = reconstruction_nll + beta(step) * sum_g w_g * (KL_g + a(step) * H_g).
```

For ST/relaxed training, the parent probabilistic loss is reconstruction NLL
plus the original annealed, adaptively balanced group KL. The exact
`utils.kl_coeff` and `utils.kl_balancer(..., kl_balance=True, alpha_i=alpha_i)`
calls are preserved. Balancing is active only while beta < 1, just as in the
parent. REINFORCE/O-BBVI preserve their parent's sampled joint KL and beta
schedule; the parent did not use adaptive group balancing for those estimators.
The entropy penalty uses exactly the same beta and group weights as the KL.
Group weights are detached in BOTH terms. With score estimators, w_g=1,
matching their original KL policy. The alpha CLI flags now schedule the
relative coefficient a; the effective entropy coefficient of group g is
beta * w_g * a. This replaces the previous separately weighted entropy loss.
Optimizer weight decay and spectral/BN regularizers retain their parent settings.
At a=0 this mode recovers the parent training objective at that step.

The earlier `--poisson_objective elbo_alpha` mode remains available. Its
minimized probabilistic loss is

```
negative_L_alpha(x) = reconstruction_nll(x) + KL(q(z|x) || p(z))
                     + a(step) * H(q(z|x)).
```

All terms are **summed over latent coordinates/groups, then averaged over
images**, in nats per image. Entropy is not averaged per coordinate.
For hierarchical q, the joint entropy is

```
H(q(z|x)) = sum_g E_{q(z_<g|x)} H(q(z_g|x,z_<g)).
```

Only `elbo_alpha` uses **beta=1 and no adaptive group KL balancing from the first
step**. The parent's KL warm-up is deliberately disabled for the ELBO term:
keeping it would optimize a different beta-weighted objective, not L_alpha.
The existing optimizer weight decay and spectral/BN regularization are
separate, unchanged regularizers. If norm-coefficient annealing is enabled,
it still follows the original KL schedule, independently of beta=1 here.
Set their coefficients to zero if the entire update must contain only the
probabilistic loss.

`--poisson_objective nvae` restores the parent branch's probabilistic
objective, KL warm-up and balancing, without entropy regularization.
`--poisson_entropy_alpha_init 0` recovers the parent objective in `nvae_alpha`,
and the ordinary full-KL ELBO in `elbo_alpha`.
Normal and mixed Poisson/Gamma models keep their original objectives; this
new joint discrete-entropy mode applies only to `--latent_distribution poisson`.

## Default schedule and rationale

```
--poisson_entropy_alpha_init 0.1
--poisson_entropy_anneal_portion 0.5
```

Let s=0,...,T-1 be the global minibatch/optimizer-step index and
u=s/(T-1). Then

```
a(u) = 0.05 * (1 + cos(pi*u/0.5))   if 0 <= u < 0.5
           0                            if u >= 0.5.
```

The last step is exactly zero, including with anneal_portion=1. A one-step
run also uses zero. For a 200-epoch run, the approximate milestones are:

| Training progress | Approximate epoch | relative a |
| --- | --- | --- |
| 0% | start | 0.1 |
| 12.5% | 25 | 0.085355 |
| 25% | 50 | 0.05 |
| 37.5% | 75 | 0.014645 |
| 50% and later | 100-200 | 0 |

This is a conservative experimental starting point, not a tuned optimum.
In both modes, each group term can be written as
`beta*w_g*(cross_entropy_g - (1-a)*H_g)`. With a=0.1, the entropy
coefficient inside the brackets is -0.9. The added penalty is scaled down
with the KL during warm-up, rather than imposed at full strength from the
first step. This does not guarantee reconstruction improvement; a=0.1
remains an experimental setting, and rate monitoring remains useful.
Cosine decay has zero endpoint slopes, and the final half of training uses
the chosen base objective with no added entropy penalty. No claim is
made that this continuation reaches the same local optimum as training
with alpha=0 from the beginning, or that it improves FID/NLL.

The schedule is recomputed from the checkpoint's `global_step`. New
checkpoints record entropy_weighting=same_as_kl_v1 as well as its settings and total step count; resume rejects changed
settings or changed total steps rather than silently restarting/re-timing
annealing. Resume with the same arguments, including epochs and batch size.
For `nvae_alpha`, checkpoint metadata also records all three KL schedule
parameters and rejects changes on resume. Checkpoints from `elbo_alpha` cannot
be resumed as `nvae_alpha`; start a fresh experiment to change objective.
Checkpoints with the old independent-entropy schedule metadata are rejected
on resume because the loss semantics have changed. Start a fresh experiment;
the launch script now uses the new save name
`mnist_poisson_st_official_s2_t01_e200_nvae_sync_a01_cos50`.
Older checkpoints without any schedule metadata can still use their saved
global_step with the current arguments, as in the parent branch.
As in the parent, the step counter advances on a GradScaler overflow even
if that individual optimizer update is skipped.

## Gradient estimators

* **straight_through / relaxed:** compute each conditional discrete Poisson
  entropy from its rate, summing `-p(k)*log p(k)`. A conservative numerical
  support extends past the configured rate bound; it is independent of
  `poisson_max_count`, and probabilities are not renormalized. The backward
  pass sums the derivative of this series and saves only one tensor per
  rate array. Float16 entropy arithmetic is promoted to float32. The helper
  implements first derivatives, not higher-order differentiation.
  Group entropies are retained separately and weighted by the same detached
  w_g returned by the KL balancer; that balancer sees KL only, not KL+a*H.
  Ancestor gradients use the existing ST/relaxed path, so their usual bias
  and sampler truncation remain. With relaxed prefixes this is a surrogate
  for the discrete joint entropy, just as the original relaxed ELBO is a
  surrogate for the discrete objective.
* **reinforce / obbvi:** use the full sampled joint entropy `-log q(z|x)`.
  The score signal is `recon + beta*(1-a)*log_q - beta*log_p`, with the
  original scheduled beta in `nvae_alpha` and beta=1 in `elbo_alpha`. The direct
  generative gradient is unchanged. The expectation of the omitted direct
  `beta*(1-a)*grad(log_q)` is zero. O-BBVI applies the existing detached DMIS
  importance weights to the entire signal, including entropy. This handles
  entropy dependence on all ancestor groups. The EMA baseline tracks the
  regularized signal and is updated after all current samples.

Neither the model's state_dict layout nor forward()'s five-tensor return
signature changes. No extra entropy computation is done in evaluation, or
in the pathwise training branch once alpha reaches zero.

## Run the October 16 MNIST configuration

From this repository, inside the existing `nvae` environment:

```bash
bash scripts/train_mnist_poisson_entropy.sh
```

The script matches the October 16 baseline: batch 200, 200 epochs, learning
rate 0.01 to 0.0001, 5 warm-up epochs, fast Adamax, channels 32, latent channels
20, two scales with adaptive groups [10,5], two conditional cells, and three
cells per pre/postprocessing block. ST temperature is fixed at 0.1, count
limit 64, configured rate limit 30, weight decay 0.0003 and constant norm
regularization 0.01. KL parameters remain 0.3/0.0001/0.0001. The run starts
fresh and uses a new save name. Override arguments at the end, for example:

```bash
bash scripts/train_mnist_poisson_entropy.sh \
  --poisson_gradient_estimator reinforce \
  --save mnist_poisson_entropy_a01_reinforce_s2
```

To compare the effect of entropy with the **same base objective**, use
`--poisson_entropy_alpha_init 0` and a different `--save` name. Comparing
`nvae_alpha` against `nvae` preserves the same KL warm-up/balancing; comparing
`elbo_alpha` against `nvae` also changes KL warm-up/balancing.

## Diagnostics and validation

### Group 1 posterior rate distribution during training

`--poisson_rate_monitor_every 500` (default) captures the first training
batch and every 500th batch thereafter. Set `0` to disable monitoring.
Group 1 means the **first group in decoder/generative order**, i.e.
`enc_sampler[0]` / internal group index 0, not the second group in zero-based
indexing. The captured tensor is the posterior Poisson rate used for that
forward pass, after rate parameterization/clamping, before sampling and the
optimizer update. It is not a latent count or a prior parameter.

Every image, channel and spatial coordinate in the current batch is included,
with no averaging or subsampling. With distributed training, all ranks'
current batches are gathered and rank zero logs one combined distribution.
These are current training-batch distributions, not full-dataset estimates;
different monitoring batches generally contain different shuffled images.
Repeated score-estimator samples do not duplicate the rates: group 1 has
no stochastic ancestor, so its posterior rates are the same across these
draws for the same batch.

Monitoring writes **only to TensorBoard event files and the training log**;
it does not export NPZ, CSV, PNG or a separate monitoring directory.

* TensorBoard: `posterior_rate/group_001` shows the histogram/distribution.
  Summary scalars under that prefix include image/rate counts, mean,
  population standard deviation, min/max, and p01/p05/p25/p50/p75/p95/p99.
* Bins: 60 fixed logarithmic intervals from 0.001 to 1000. TensorBoard scalars
  `posterior_rate/group_001/bins/<index>_<left>_to_<right>/count` and `/fraction`
  record the count and fraction of all rates in every bin, including empty
  bins. Fractions sum to one; they are not densities per unit rate.
* Training log: one `group1 posterior rate summary` line contains all summary
  statistics, followed by 60 `group1 posterior rate bin` lines with interval
  boundaries, counts and fractions. Intervals are [left,right), except the
  last interval which includes the right boundary.

Capture uses detached tensors and does not add a forward pass or consume RNG
state. Training gradients are unchanged. Monitoring performs a CPU transfer
and, with DDP, a gather.

For example, monitor every 200 steps:

```bash
bash scripts/train_mnist_poisson_entropy.sh --poisson_rate_monitor_every 200
```

### ELBO and entropy diagnostics

* `train/nelbo_iter`, `train/nelbo`: base probabilistic loss without the added
  entropy penalty. In `nvae_alpha` these retain the parent's historical names
  and can include annealed/balanced KL; they are not necessarily true NELBO.
* `train/negative_nvae_alpha`: regularized base loss in `nvae_alpha`.
* `train/negative_elbo_alpha`: backward-compatible regularized loss tag in
  both entropy modes, excluding norm
  and optimizer weight penalties; do not interpret the score surrogate's
  raw scalar value as the loss.
* `train/entropy_alpha` and `train/entropy_relative_a`: the relative a schedule.
* `entropy_coeff/layer_g`: actual entropy coefficient beta*w_g*a for each group.
* `train/posterior_entropy_nats`: unweighted joint entropy, for comparison.
* `train/entropy_penalty_nats`: actual beta*a*sum_g w_g*H_g contribution. Pathwise
  entropy logging stops at alpha=0 to avoid extra computation; its penalty
  continues to log zero. Score paths can keep logging their sampled entropy.
* Validation NELBO, importance-sampling NLL and generation retain the
  original unregularized/exact-sampling evaluation behavior.

Run focused and existing regression tests with:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests
```

Tests check numerical entropy values and derivatives, support-cap
independence, schedule endpoints, full hierarchical REINFORCE/O-BBVI
gradients against enumerated expectations, model backward passes, and the
actual training loop before/after the entropy penalty reaches zero.

Implementation validation: 87 CPU tests passed on Python 3.12 / PyTorch 2.15,
including zero-alpha loss/gradient/parameter-update equivalence with the
parent-objective mode and beta-scaled hierarchical score gradients. Tests
also check actual per-group entropy coefficients in the training loop,
nonuniform weighted entropy values/gradients, and stopped weight gradients.
The example script's 52 baseline settings were checked against the October
16 log; CPU model construction confirms 33.363134M parameters and [10,5]
groups. Changed Python files also parse with the Python 3.7 grammar. The
repository's original Python 3.7 / PyTorch 1.6 CUDA environment, distributed
training, GPU mixed precision and full MNIST training have not been run here.
Rate-monitor checks cover all four estimators, exact captured rates, no
change in forward outputs, histogram mass and per-bin counts/fractions,
first-batch and periodic training-loop logging without exports, and
simulated unequal-batch distributed gathering.
An actual TensorBoard SummaryWriter smoke check creates only event files.


