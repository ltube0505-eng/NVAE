# Annealed posterior entropy: L_alpha

This branch starts from `feature/all-poisson-gradient-estimators` at
`777d4d7608294a6e87af7df409a31e949203bd66`.

## Objective

For all-Poisson models, `train.py` now defaults to
`--poisson_objective nvae_alpha`. The minimized training loss is

```
loss(x, step) = parent_NVAE_loss(x, step) + alpha(step) * H(q(z|x)).
```

For ST/relaxed training, the parent probabilistic loss is reconstruction NLL
plus the original annealed, adaptively balanced group KL. The exact
`utils.kl_coeff` and `utils.kl_balancer(..., kl_balance=True, alpha_i=alpha_i)`
calls are preserved. Balancing is active only while beta < 1, just as in the
parent. REINFORCE/O-BBVI preserve their parent's sampled joint KL and beta
schedule; the parent did not use adaptive group balancing for those estimators.
The entropy penalty is added separately, without beta or group-balancing weights.
Optimizer weight decay and spectral/BN regularizers retain their parent settings.
At alpha=0 this mode recovers the parent training objective at that step.

The earlier `--poisson_objective elbo_alpha` mode remains available. Its
minimized probabilistic loss is

```
negative_L_alpha(x) = reconstruction_nll(x) + KL(q(z|x) || p(z))
                     + alpha(step) * H(q(z|x)).
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
alpha(u) = 0.05 * (1 + cos(pi*u/0.5))   if 0 <= u < 0.5
           0                            if u >= 0.5.
```

The last step is exactly zero, including with anneal_portion=1. A one-step
run also uses zero. For a 200-epoch run, the approximate milestones are:

| Training progress | Approximate epoch | alpha |
| --- | --- | --- |
| 0% | start | 0.1 |
| 12.5% | 25 | 0.085355 |
| 25% | 50 | 0.05 |
| 37.5% | 75 | 0.014645 |
| 50% and later | 100-200 | 0 |

This is a conservative experimental starting point, not a tuned optimum.
In `elbo_alpha`, the explicit entropy coefficient in the maximized objective
is 1-alpha=0.9 at initialization. This interpretation does not apply during
the annealed/balanced phase of `nvae_alpha`. The separate entropy penalty can
dominate a small initial KL weight, so monitor rates for excessive suppression;
0.1 is an experimental setting, not a guarantee against rate collapse.
Cosine decay has zero endpoint slopes, and the final half of training uses
the chosen base objective with no added entropy penalty. No claim is
made that this continuation reaches the same local optimum as training
with alpha=0 from the beginning, or that it improves FID/NLL.

The schedule is recomputed from the checkpoint's `global_step`. New
checkpoints record its settings and total step count; resume rejects changed
settings or changed total steps rather than silently restarting/re-timing
annealing. Resume with the same arguments, including epochs and batch size.
For `nvae_alpha`, checkpoint metadata also records all three KL schedule
parameters and rejects changes on resume. Checkpoints from `elbo_alpha` cannot
be resumed as `nvae_alpha`; start a fresh experiment to change objective.
Older checkpoints without schedule metadata can be resumed using the current
arguments and their saved global_step (the user is explicitly starting the
new objective at that progress). A fresh run is recommended for comparison.
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
  Ancestor gradients use the existing ST/relaxed path, so their usual bias
  and sampler truncation remain. With relaxed prefixes this is a surrogate
  for the discrete joint entropy, just as the original relaxed ELBO is a
  surrogate for the discrete objective.
* **reinforce / obbvi:** use the full sampled joint entropy `-log q(z|x)`.
  The score signal is `recon + (beta-alpha)*log_q - beta*log_p`, with the
  original scheduled beta in `nvae_alpha` and beta=1 in `elbo_alpha`. The direct
  generative gradient is unchanged. The expectation of the omitted direct
  `(beta-alpha)*grad(log_q)` is zero. O-BBVI applies the existing detached DMIS
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
current batches are gathered and rank zero writes one combined snapshot.
These are current training-batch distributions, not full-dataset estimates;
different snapshots generally contain different shuffled images.
Repeated score-estimator samples do not duplicate the rates: group 1 has
no stochastic ancestor, so its posterior rates are the same across these
draws for the same batch.

Outputs are under `<experiment>/posterior_rate_monitor/group_001/`:

* `summary.csv`: step, number of images/rates, min/max, mean, population std,
  and p01/p05/p25/p50/p75/p95/p99.
* `step_XXXXXXXXX.png`: logarithmic rate bins versus the fraction of all
  rates in each bin. Fixed bin edges allow comparisons over training.
* `step_XXXXXXXXX_histogram.csv`: bin boundaries, counts and fractions.
* `step_XXXXXXXXX.npz`: `rates` with shape `[global_B,C,H,W]`, `bin_edges`
  and `counts`. Images are concatenated in rank order.

TensorBoard contains the distribution under `posterior_rate/group_001` and
scalar summaries under that prefix. The training log prints mean, standard
deviation, min, p05, median, p95 and max at each monitoring step. Capture
uses detached tensors and does not add a model forward pass. Gathering,
CPU file writes and plotting add overhead only at the monitoring interval.

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
* `train/entropy_alpha`, `train/posterior_entropy_nats`,
  `train/entropy_penalty_nats`: schedule and entropy diagnostics. Pathwise
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

Implementation validation: 78 CPU tests passed on Python 3.12 / PyTorch 2.15,
including zero-alpha loss/gradient/parameter-update equivalence with the
parent-objective mode and annealed-beta hierarchical score gradients.
The example script's 52 baseline settings were checked against the October
16 log; CPU model construction confirms 33.363134M parameters and [10,5]
groups. Changed Python files also parse with the Python 3.7 grammar. The
repository's original Python 3.7 / PyTorch 1.6 CUDA environment, distributed
training, GPU mixed precision and full MNIST training have not been run here.
Rate-monitor checks cover all four estimators, exact captured rates, no
change in forward outputs, saved histogram mass/raw arrays, training-loop
snapshot creation, and simulated unequal-batch distributed gathering.

