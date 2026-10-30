# Annealed posterior entropy: L_alpha

This branch starts from `feature/all-poisson-gradient-estimators` at
`777d4d7608294a6e87af7df409a31e949203bd66`.

## Objective

For all-Poisson models, `train.py` now defaults to
`--poisson_objective elbo_alpha`. The minimized probabilistic loss is

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

This mode uses **beta=1 and no adaptive group KL balancing from the first
step**. The parent's KL warm-up is deliberately disabled for the ELBO term:
keeping it would optimize a different beta-weighted objective, not L_alpha.
The existing optimizer weight decay and spectral/BN regularization are
separate, unchanged regularizers. If norm-coefficient annealing is enabled,
it still follows the original KL schedule, independently of beta=1 here.
Set their coefficients to zero if the entire update must contain only the
probabilistic loss.

`--poisson_objective nvae` restores the parent branch's probabilistic
objective, KL warm-up and balancing, without entropy regularization.
`--poisson_entropy_alpha_init 0` instead uses ordinary full-KL ELBO throughout.
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
At initialization, the explicit entropy coefficient in the maximized
objective is 1-alpha=0.9. A large alpha could push Poisson rates toward zero;
0.1 keeps that pressure modest. Cosine decay has zero endpoint slopes, and
the final half of training is reserved for the ordinary ELBO. No claim is
made that this continuation reaches the same local optimum as training
with alpha=0 from the beginning, or that it improves FID/NLL.

The schedule is recomputed from the checkpoint's `global_step`. New
checkpoints record its settings and total step count; resume rejects changed
settings or changed total steps rather than silently restarting/re-timing
annealing. Resume with the same arguments, including epochs and batch size.
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
  The score signal is `recon + (1-alpha)*log_q - log_p`. The direct
  generative gradient is unchanged. The expectation of the omitted direct
  `(1-alpha)*grad(log_q)` is zero. O-BBVI applies the existing detached DMIS
  importance weights to the entire signal, including entropy. This handles
  entropy dependence on all ancestor groups. The EMA baseline tracks the
  regularized signal and is updated after all current samples.

Neither the model's state_dict layout nor forward()'s five-tensor return
signature changes. No extra entropy computation is done in evaluation, or
in the pathwise training branch once alpha reaches zero.

## Run the compact MNIST experiment

From this repository, inside the existing `nvae` environment:

```bash
bash scripts/train_mnist_poisson_entropy.sh
```

The example uses the compact two-scale architecture, batch size 64, 200
epochs and ST gradients. Override arguments at the end, for example:

```bash
bash scripts/train_mnist_poisson_entropy.sh \
  --poisson_gradient_estimator reinforce \
  --save mnist_poisson_entropy_a01_reinforce_s2
```

To compare the effect of entropy with the **same full-KL objective**, use
`--poisson_entropy_alpha_init 0` and a different `--save` name. Comparing
against `--poisson_objective nvae` also changes KL warm-up/balancing.

## Diagnostics and validation

* `train/nelbo_iter`, `train/nelbo`: unpenalized negative ELBO in the new mode.
* `train/negative_elbo_alpha`: regularized probabilistic loss, excluding norm
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

Implementation validation: 39 tests passed on Python 3.12 / PyTorch 2.15
CPU. Changed Python files also parse with the Python 3.7 grammar. The
repository's original Python 3.7 / PyTorch 1.6 CUDA environment, distributed
training, GPU mixed precision and full MNIST training have not been run here.
