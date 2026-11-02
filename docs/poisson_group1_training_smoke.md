# Recorded MNIST CPU training smoke run

This is a bounded integration run, **not** a full 200-epoch training result.
It was recorded at commit `9db9f4efc25dbc1df340c8173b9ce13f9e1dd34d`, using
that commit's launcher defaults (`use_se=False`, `res_dist=False`,
`min_groups_per_scale=2`, `weight_decay_norm=0.01`). The current launcher has
since been aligned to the supplied baseline. The command below documents the
historical run and should be interpreted with the launcher at that commit.
The host has no CUDA device. Python 3.12.14, PyTorch 2.15.0+cu132 (CPU execution).
The original Python 3.7 / PyTorch 1.6 CUDA environment and multi-GPU SyncBN path
have not been executed here; the implementation retains their APIs.

Command (from the repository root, with writable paths substituted):

```bash
bash scripts/train_mnist_poisson_group1_svi.sh \
  --device cpu --arch_instance res_elu \
  --root /workspace/scratch/901f51767d0c/training \
  --data /workspace/scratch/901f51767d0c/mnist \
  --save group1_svi_cpu_smoke \
  --num_channels_enc 4 --num_channels_dec 4 --num_latent_per_group 2 \
  --batch_size 8 --epochs 1 --warmup_epochs 0 \
  --max_train_batches 8 --max_valid_batches 2 \
  --num_eval_samples 2 --final_eval_samples 2
```

The architecture has two scales, decoder groups `[2,4]`, spatial sizes 4x4 and
8x8, 137,654 parameters. All groups use Poisson ST. Only the top first group
performs 5 local momentum steps per batch, with lr=0.1, momentum=0.5, eps=1e-5.
The run used real MNIST, 8 optimizer steps (64 training examples), and 16
validation examples per validation pass. KL scheduling uses the original
full-loader iteration horizon even when a smoke batch limit is supplied.

| Measurement | First training batch | Eighth training batch |
|---|---:|---:|
| Unweighted refined NELBO (nat/image) | 585.397705 | 542.083191 |
| NLL (nat/image) | 562.857727 | 460.232422 |
| Sum of all six analytic KLs (nat/image) | 22.539970 | 81.850784 |
| Mean absolute group1 local raw-parameter change | 0.00639021 | 0.00652559 |

These are different minibatches, not a controlled inference-improvement test.
The tiny validation sample does not support model-quality conclusions.

- Average annealed/balanced training objective: 513.248047 nat/image.
- Final validation NELBO: 579.872620 nat/image.
- Final validation negative IW log probability, K=2: 566.712158 nat/image.
- Checkpoint: epoch=1, global_step=8, all state tensors finite.
- TensorBoard includes unweighted refined NELBO and the group1 update size.
- 19 tests passed, including comparison of the reverse recursion to exact
  stochastic differentiable unrolling on a smooth reference problem.

The checkpoint uses PyTorch 2.15 serialization and is a CPU smoke artifact;
loading it in the older CUDA environment is not a claimed compatibility test.
