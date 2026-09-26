# Diagnosing the Poisson NVAE inference gap

`diagnose_inference_gap.py` compares trained straight-through and relaxed
all-Poisson checkpoints without changing either training method. It separates
three effects that otherwise appear together in `NELBO - IW-NLL`.

## Quantities reported

For each validation minibatch, the script freezes the complete checkpoint and
adds one local, per-example log-rate correction to every Poisson posterior
group. Only these corrections are optimized. Batch-normalization statistics
remain frozen, and trained convolution weights are not data-initialized again.

1. **Amortized exact NELBO**: the checkpoint encoder evaluated with exact
   Poisson samples, `beta=1`, and no KL balancing.
2. **Refined exact NELBO**: the same quantity after local log-rate refinement.
   Their difference is a conservative estimate of the amortization gap.
3. **IW-NLL sweep and ESS/K**: NLL at selected sample counts, normalized
   effective sample size, and maximum normalized weight. These determine
   whether the residual gap can be interpreted reliably.

The decomposition used for the final summary is

```text
amortization_gap = NELBO(amortized q) - NELBO(refined q)
residual_gap     = NELBO(refined q)   - IW-NLL_K(refined q)
```

The residual is labelled a **posterior-family proxy**, not an exact
approximation gap. The local correction is deliberately small and may not
reach the best hierarchical Poisson posterior. IW-NLL is also still a bound at
finite `K`.

## Recommended ST/relaxed comparison

```bash
cd /root/autodl-tmp/NVAE

/root/miniconda3/envs/nvae/bin/python diagnose_inference_gap.py \
  --checkpoints \
    st=/root/autodl-tmp/nvae-checkpoints/eval-mnist_poisson_st_s2/checkpoint.pt \
    relaxed=/root/autodl-tmp/nvae-checkpoints/eval-mnist_poisson_relaxed_s2/checkpoint.pt \
  --data /root/autodl-tmp/datasets/mnist \
  --batch_size 8 \
  --max_batches 5 \
  --local_steps 100 \
  --local_lr 0.05 \
  --iw_samples 1000 \
  --iw_checkpoints 1,10,100,1000 \
  --output /root/autodl-tmp/nvae-checkpoints/inference_gap_st_vs_relaxed.json
```

The same dynamically binarized MNIST minibatches are cached once and reused by
both checkpoints. Exact evaluation samples also use matching random seeds.

## Interpretation

- `amortization_dominated`: local refinement explains at least 60% of the
  estimated total inference gap. Improve iterative/semi-amortized inference or
  the top-down encoder.
- `posterior_family_dominated_proxy`: the stable residual explains at least
  60%. Test a richer count posterior, such as autoregressive Poisson,
  Poisson-Gamma/negative-binomial, or structured within-group dependence.
- `surrogate_mismatch`: the relaxed/ST local objective improves but the exact
  NELBO does not. The gradient surrogate, rather than encoder capacity alone,
  is the immediate problem.
- `inconclusive_importance_sampling`: `ESS/K < 0.01` or the last IW-NLL step
  still improves by more than `0.5` nat/example. Increase `K` or improve the
  proposal before attributing the residual to the posterior family.
- `mixed`: neither component reaches the 60% dominance threshold.

The thresholds are command-line options. They are decision aids rather than
formal hypothesis tests. For the final experiment, increase `--max_batches`
after the five-batch pilot is stable.
