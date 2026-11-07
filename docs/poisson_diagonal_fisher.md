# Poisson diagonal Fisher training with the original tanh rate

This experiment addresses slow low-rate updates by changing the weight-space
optimizer. The forward mapping remains exactly

    eta = 5 * tanh(a / 5)
    lambda = exp(eta)

Adaptive CTS, exact hard counts, analytic KL, and the model state_dict layout
remain compatible with the base branch. There is no added hard rate cap.
The smooth mapping still has asymptotic bounds exp(-5) and exp(5).

## Metric and algorithm

Martens (2020), *New Insights and Perspectives on the Natural Gradient Method*,
https://www.jmlr.org/papers/v21/17-678.html motivates Fisher preconditioning,
Tikhonov damping and controlled updates (especially Sections 10–12). This PR is
an explicitly approximate Poisson-output implementation, not a reproduction of
full natural-gradient optimization of NVAE's marginal likelihood.

For a scalar raw output, h = sech(a/5)^2 and the Poisson Fisher is
F_a = lambda * h^2. Dividing only by lambda would miss the tanh Jacobian.
For network weights theta, collect all trainable posterior and conditional-prior
log-rate tensors from the existing forward (the fixed top prior contributes zero).
Let eta concatenate these tensors and J = d eta / d theta. Define

    F = (1/B) J^T diag(exp(eta)) J

where B is the local image batch size. Latent coordinates and groups are summed,
images are averaged. This is the pullback of an auxiliary product-Poisson output
metric. In a hierarchical model the collected outputs depend on sampled ancestor
latents. Their ST/relaxed derivatives remain in J. Consequently this is a
pathwise surrogate metric, not the exact Fisher of the joint hierarchical q or
p, nor the Fisher of p(x). Batch-normalized examples are also coupled.

For independent Rademacher signs epsilon, compute

    v = J^T [sqrt(lambda) * epsilon] / sqrt(B)
    d_hat = average_over_probes(v * v)

Then E[d_hat] = diag(F) for the chosen differentiable/surrogate Jacobian.
Rate weights and signs are detached. This uses analytical Poisson Fisher weights,
not squared minibatch loss gradients (the empirical-Fisher issue in Section 11).
Each probe costs an additional retained-graph reverse pass and needs no
second-order autograd. Probe randomness is isolated from the CTS RNG stream.
Parameters outside the latent-rate graph, such as the final image head, use an
identity metric. Coordinates inside a connected tensor can still have zero
curvature and are stabilized by damping.

Use the first estimate to initialize d, and subsequently

    d = beta * d_previous + (1-beta) * d_hat
    g = gradient(training_loss) + weight_decay * theta
    delta_0 = -learning_rate * g / (d + damping)
    Q = 0.5 * sum((d + damping) * delta_0^2)
    scale = min(1, sqrt(step_budget / max(Q, 1e-30)))
    theta_next = theta + scale * delta_0

The quadratic budget applies to the actual parameter update, including L2 weight
decay. It bounds the damped diagonal surrogate cost, not the actual nonlinear KL
or a guaranteed ELBO decrease. Damping is fixed/configurable, not an adaptive
Levenberg–Marquardt acceptance/rejection rule. Diagonalization, damping, finite
steps and the surrogate Jacobian all limit reparameterization invariance.
This optimizer does not guarantee recovery of saturated units or eliminate
posterior collapse. Forward tanh can still have zero derivative at finite precision.

## Training

Keep your existing architecture, data, CTS, KL schedule and training arguments,
and append

```bash
--optimizer poisson_fisher \
--fisher_damping 0.01 \
--fisher_decay 0.95 \
--fisher_probes 1 \
--fisher_step_budget 0.01
```

This mode supports `--latent_distribution poisson` with `straight_through` or
`relaxed` sampling. Other combinations fail explicitly. Default training remains
Adamax. `--fast_adamax` has no effect in Fisher mode. Fisher mode replaces Adamax
with the update above; it is not a preconditioner fed into Adamax. The existing
learning-rate/warmup/cosine schedule controls the nominal step. Start with the
existing learning rate (e.g. 0.001); these defaults are experimental, not tuned
training results. A fair comparison must examine ELBO and actual update size,
not only the coordinate Jacobian.

Fisher mode uses full FP32 forward/backward and disables AMP/GradScaler scaling.
This avoids underflow in small curvature estimates and prevents scaled probe
values or scaled gradients from entering the metric/budget. Expect additional
compute and memory for retained backward graphs, a curvature EMA and step tensors.
NaN/Inf curvature, gradients or directions raise before any parameter update.
Monitor `fisher/step_scale` and `fisher/quadratic_step` in TensorBoard.

Distributed training averages local squared probe results across ranks and
averages objective gradients before preconditioning. All ranks need equal local
batch sizes and matching graph topology. With cross-rank SyncBatchNorm, local
backwards exchange derivatives; this introduces a further approximation relative
to a full global batch Jacobian Fisher. The diagonal budget is identical on ranks
with synchronized parameters and optimizer state. GPU/distributed numerical
validation is still required on the production PyTorch 1.6/CUDA environment.

## Checkpoints

Same-optimizer continuation restores the Fisher EMA from the ordinary optimizer
state_dict. Evaluation uses the unchanged forward/model state and collects no
curvature tensors. Old checkpoint args without `optimizer` mean Adamax.

To initialize a Fisher experiment from an Adamax checkpoint, copy that checkpoint
into a **new experiment save directory**, keep the normal `--cont_training`, and
add `--reset_optimizer`. This preserves model, epoch, global step, current learning
rate and scheduler progress while resetting optimizer/scaler state. Switching
optimizers without this flag fails explicitly. It does not restart completed
training or extend the saved scheduler horizon automatically.

## Validation

`python -m pytest -q tests/test_poisson_fisher.py` checks the analytical scalar
Fisher including tanh, shared-weight/batch normalization of the estimator, actual
step budget, identity fallback, checkpoint EMA continuity, nonfinite rejection,
and a small CPU Poisson AutoEncoder forward/backward/update. The tests do not
establish final NELBO/FID improvement or CUDA multi-GPU performance.
