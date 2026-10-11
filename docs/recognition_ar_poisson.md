# Alternating AR categorical / NAR Poisson recognition

Select `--latent_distribution ar_poisson` to implement
`Recognition_Model_NVAE_AR_Poisson.md` on the NVAE hierarchy. The existing
Gaussian, all-Poisson (including discrete flows), and Poisson/Gamma modes
retain their previous heads and sampling paths.

## Model and architecture

Groups are numbered in top-down generation order, across all scales. Even
groups use an autoregressive 64-class categorical posterior, odd groups use
conditionally independent Poisson counts. The sequence starts with AR and
does not restart at scale boundaries. Each group must have
positive latent dimensions, with sequence length `d_lat = C_z * H_g * W_g`.
There is no fixed upper bound on `d_lat`; positional encodings are generated
for the configured length. `num_latent_per_group` remains the number of latent
**channels**, not the flattened sequence length. AR sampling is still sequential,
so available memory and runtime determine the practical maximum.

The bottom-up encoder, feature/state combiners, top-down decoder, and
dataset-specific image likelihood remain NVAE modules. Before each non-top
group, the encoder combiner forms `feature = h_i(x) + Conv1x1(s_i)`, where
`s_i` is the decoder state before injecting the current group. An AR head
encodes this fused map with `Conv3x3(C_enc,D) -> ELU -> Conv3x3(D,D) -> ELU`,
then adaptive average pooling produces a fixed spatial grid. Flattening,
sinusoidal positions, and LayerNorm give memory `[B,M,D]`. The grid uses the
closest factor pair of M (8x8 for the default M=64); its output length is
independent of the input scale and the number of scalar latent coordinates.
For input maps smaller than that grid, adaptive pooling can produce repeated
or overlapping pooled features; M=64 does not imply 64 independent patches.
The top AR group fuses NVAE's `enc0` feature with the learnable initial
decoder state `prior_ftr0` using an additional original `EncCombinerCell`,
then uses the same CNN. Its learned Poisson prior remains image-independent.

Each latent Transformer decoder block projects this memory to its own fixed
cross-attention K/V once per group. Its Pre-LN causal self-attention reads
only BOS/right-shifted latents within that group, followed by cross-attention
and a ReLU FFN. There is no feature Transformer encoder in the NVAE AR head.
Defaults: `--ar_memory_tokens 64`, width 128, 4 heads,
`--ar_num_layers 2` latent decoder blocks, FFN ratio 4, dropout 0. Attention
projections have no bias; CNN, FFN, and output projections have bias.
`RecognitionScale1` retains the standalone padded patch-image interface and
its original feature Transformer encoder.

`DecCombinerCell` is unchanged from NVlabs/NVAE: channel-concatenate numerical
`s_i` and `Z_i`, then apply the original `Conv2D` 1x1 projection. CNN memory
and cross K/V enter recognition only, not the generative decoder combiner.

The standalone reference and latent decoder are based on jadex/DAPS:
[recognition model](https://github.com/mdrolet01/jadex/blob/fd0a606d9e718a7b62c4ac7733a2b7e02e71761a/jadex/networks/variational/recognition_models/transformer_rec.py)
and [Transformer](https://github.com/mdrolet01/jadex/blob/fd0a606d9e718a7b62c4ac7733a2b7e02e71761a/jadex/networks/transformer.py).
There is no policy-search objective or Flax checkpoint loader.

The shared recognition embedding table has 64 rows. Inputs are right-shifted
`(BOS, z_0, ..., z_{T-2})`, using row 0 as BOS while keeping class 0 valid.
Full-prefix within-mini-group self-attention KV caches are separate per decoder block;
cross-attention KV is computed once per group from CNN memory. Both caches
are local variables rebuilt on every group invocation; the NVAE AR result
does not return either cache, and explicit cache lists are cleared at the
group boundary. All ST embedding and attention gradient paths are retained:
autograd still saves the tensors needed until backward. Clearing a cache
does not detach it or force all its GPU storage to be freed before backward.
The standalone reference can still return caches for scoring/debugging.
Sampling is sequential; teacher-forced logits are only for scoring known
sequences. There is no sliding cache window.

AR hard sampling is Gumbel-max from untempered `softmax(logits)`. Gumbel-ST
temperature affects only the backward path. The next input is
`y_st @ embedding.weight`, and the NVAE decoder receives
`(y_st @ arange(64)).reshape(B,C_z,H_g,W_g)` in channel/row/column order.
Ordinary counts enter the existing numerical decoder combiners, so prior
counts above 63 require no embedding lookup or clipping.

Every group's prior is an ordinary, unbounded Poisson. The top group has a
trainable raw rate per coordinate; later prior heads receive only the
top-down state before injecting the current group. Heads emit `C_z` raw
rate channels, mapped by `exp(5*tanh(raw/5))`. NAR uses ELU + 3x3 Conv and
optionally adds the prior raw parameter via `--res_dist`. The legacy
`--poisson_max_rate` does not clip rates in this new mode.


## Parallel mini-groups and chunk-grown KV cache

`--ar_mini_groups G` (alias `--mini_group`) partitions each AR group's flat
CHW sequence of length `T = C_z * H_g * W_g` into `min(G,T)` balanced,
contiguous mini-groups. Defaults to 1 for the previous full-group posterior.
Mini-groups share the recognition weights and image/fused-state memory but
have independent BOS inputs, Gumbel draws and self-attention prefixes.
Conditioned on that memory (and the preceding NVAE groups), the posterior is

\[
q(z_i\mid x,z_{<i})=
\prod_{g=1}^{G_i}\prod_{t=1}^{L_g}
q(z_{i,g,t}\mid x,z_{<i},z_{i,g,<t}).
\]

This changes the posterior family when G > 1; it is not an exact acceleration
of the original full-group autoregressive posterior. Independence is
conditional, not marginal across images or preceding NVAE groups. NAR groups
already factorize over all coordinates and remain unchanged.

There is one time loop of length `L = ceil(T/G_i)`, with mini-group streams
merged into the batch axis. No Python loop over mini-groups is used.

| Tensor | Shape |
| --- | --- |
| Original group | `[B,C_z,H_g,W_g]` |
| Flat group | `[B,T]` |
| Padded mini-group counts | `[B,G_i,L]` |
| Current ST one-hot | `[B*G_i,1,64]` |
| Current Transformer input | `[B*G_i,1,embed_dim]` |
| Self-attention cache | `[B*G_i,num_heads,capacity,head_dim]` |
| Cross-attention K/V | `[B*G_i,num_heads,memory_tokens,head_dim]` |
| Returned counts | `[B,C_z,H_g,W_g]` |

Lengths differing by one coordinate are padded only at stream ends. A
device-resident index map restores the original CHW order and removes padding
before `log_q`, Poisson `log_p`, KL and decoder inputs are computed. The same
partition is used for teacher-forced scoring, including a separate right-shift
and BOS per stream. Global positional encodings preserve coordinate identity.
Image memory is encoded once per original batch; cross K/V is projected once
per decoder layer then broadcast to mini-group streams. Metadata buffers move
with the module to CUDA and are reconstructed from args on checkpoint load.

`--ar_grow_target_every N` (alias `--grow_target_every`) follows jadex's
[cache-capacity growth mechanism](https://github.com/mdrolet01/jadex/blob/fd0a606d9e718a7b62c4ac7733a2b7e02e71761a/jadex/networks/autoregressive_sampling.py):
capacity grows in N-token increments, bounded by each stream's padded length.
It does not grow latent dimensionality, discard prefixes, or sample N dependent
tokens simultaneously. Unwritten cache slots are masked. The default 0 keeps
the previous exact-prefix `torch.cat` cache.

No-grad inference reuses storage within a capacity chunk. Training performs
functional/cloned cache updates to preserve gradients through all prior ST
tokens and avoid modifying tensors needed by earlier backwards. Consequently
training still copies cache tensors; chunk growth alone is not a claim of
GPU speedup. Parallel mini-groups reduce sequential sampling steps from T to L,
but actual throughput and extra memory from broadcasting must be measured.

Example flags to add to an existing AR/NAR training command:

```bash
--ar_mini_groups 8 --ar_grow_target_every 8
```

For `[B,4,8,8]`, T=256 becomes eight independent 32-step streams, executed
with Transformer batch `B*8`. New options are saved in the normal checkpoint
args. No parameter shapes or persistent state-dict keys are added; checkpoints
without the options retain G=1 and growth=0. Use matching mini-group arguments
when resuming training so the posterior factorization is preserved.

## Probabilities, KL, and CTS modes

AR records each sampled categorical log probability and the ordinary
Poisson PMF at the same count. KL is the trajectory MC sum of `log_q-log_p`.
Training uses ST one-hot selection of the 64 discrete log PMF values. Those
Poisson values are **never** passed through `log_softmax`; the prior is not
renormalized over 0..63. AR MC KL and its channel statistics may be negative
for an individual trajectory; they are not clamped.

NAR expands exactly 64 exponential arrivals and uses sigmoid CTS-ST gradients.

| `--ar_poisson_cts_mode` | Actual hard posterior | KL | Log q at 64 |
| --- | --- | --- | --- |
| `capped64` (default) | `min(Pois(rate_q),64)` | Exact finite sum versus ordinary Poisson prior | Full `P(N>=64)` tail atom |
| `exact64` | Unbounded `Pois(rate_q)` | Standard analytic Poisson/Poisson KL | Ordinary single-point PMF |

`exact64` completes the hard tail with a Poisson draw of rate
`rate_q * max(1-S_64,0)`; its backward surrogate still uses only the first
64 arrivals. Finite-sum KL and tail probability use float64 internally;
`capped64` is a censored distribution, not a renormalized truncated Poisson.
Evaluation removes soft gradients and retains the same selected hard law.
All group sampling/probability operations run in float32 with autocast
disabled, so AMP does not distort integer counts or PMF selection.

The five-item NVAE return contract is unchanged:
`pixel_params, log_q, log_p, kl_all, kl_diag`. Group KLs sum over scalar
coordinates; `kl_diag` sums over space and averages over batch, giving one
statistic per channel. NAR KL is analytic, while its sampled log probabilities
are recorded separately for importance weights. Therefore total KL need not
equal the sampled `log_q-log_p` for a single mixed trajectory.

The existing IW code uses the actual sampled joint probabilities and image
likelihood. AR posterior support is finite, whereas its prior is unbounded;
`capped64` additionally excludes NAR values above 64. Consequently,
importance averaging approaches `p(x,z in posterior support)`, not necessarily
the entire `p(x)`, even as K grows. The reported IW-NLL is an upper-bound
estimate and cannot by itself establish the true likelihood or inference gap.
`exact64` removes only the NAR support restriction. ESS remains a diagnostic
of the sampled importance weights, not a test of omitted support mass.

## Training options and checkpoint evaluation

`--ar_poisson_objective standard` (default) optimizes reconstruction NLL plus
mixed group KLs with beta=1, no KL balancing, no spectral/BN penalties, and no
optimizer weight decay. Legacy annealing/regularization flags have no effect
in this objective. LR warmup and scheduling remain unchanged.
`--ar_poisson_objective nvae` explicitly enables the existing annealing,
balancing, norm/BN penalties, and configured optimizer weight decay.
`train/nelbo*` always reports the unweighted NELBO in the new mode;
`train/objective_iter` reports the actual loss including optional penalties.
`train/nar_saturation_*` reports the last trajectory's 64th-arrival fraction.
`--ar_poisson_mc_samples M` averages M fresh complete training trajectories;
the existing evaluation sample-count options control validation/IW sampling.

Required fixed choices: `--num_nf 0`, `--poisson_max_count 64`,
`--poisson_gradient_estimator straight_through`. Incompatible configurations
fail at model construction. ST gradients are biased in both posterior types.

Initialize/train a new model: old Poisson-flow, Gaussian, or AR/NAR checkpoints
with the former feature Transformer encoder have different head parameters
and cannot be strictly resumed with this CNN architecture. All new
parameters and configuration are included in the normal checkpoint. On
evaluation, AR/NAR checkpoints load strictly. The checkpoint loader explicitly
loads the trusted local Namespace metadata on modern PyTorch, while retaining
compatibility with the older checkpoint-loading API.

Example MNIST command (each group has 64 or 256 scalars):

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python train.py \
  --dataset mnist --data /root/autodl-tmp/datasets/mnist \
  --root /root/autodl-tmp/nvae-checkpoints --save mnist_ar_nar_poisson \
  --latent_distribution ar_poisson --num_nf 0 \
  --ar_poisson_cts_mode capped64 --ar_poisson_objective standard \
  --ar_poisson_mc_samples 1 --ar_gumbel_temperature 1.0 --ar_memory_tokens 64 \
  --poisson_gradient_estimator straight_through \
  --poisson_max_count 64 --poisson_relaxation_temperature 0.1 \
  --num_latent_scales 2 --num_groups_per_scale 4 --ada_groups \
  --min_groups_per_scale 1 --num_latent_per_group 4 \
  --num_channels_enc 16 --num_channels_dec 16 \
  --num_preprocess_blocks 2 --num_preprocess_cells 2 \
  --num_postprocess_blocks 2 --num_postprocess_cells 2 \
  --num_cell_per_cond_enc 1 --num_cell_per_cond_dec 1 \
  --use_se --res_dist --arch_instance res_mbconv \
  --batch_size 32 --epochs 200 --learning_rate 1e-3 \
  --num_process_per_node 1
```

Switch only `--ar_poisson_cts_mode exact64` for the uncapped NAR experiment.
Prior generation skips recognition entirely and samples unbounded Poisson
counts. At generation temperature 1 it uses the declared model; other positive
temperatures multiply prior rates by that factor.

The reference initializes raw top rates at zero (initial rate 1) and the AR
output head with Xavier weights and zero bias. A nearly uniform initial
categorical posterior can therefore have large initial KL against rate 1.
No extra prior-like initialization or rate constraint was silently added.
Long AR sequences remain costly despite KV caching; no training speed or
quality result is claimed.

## Runtime and validation

Use Python >=3.8 and PyTorch >=2.0 with a matching torchvision build for this
new mode; the original torch 1.6 environment cannot compute its differentiable
`gammainc` tail. Existing modes do not acquire that runtime requirement.
Do not install the repository's original pinned torch/torchvision pair for an
AR/NAR experiment. Existing single/multi-GPU NVAE training infrastructure
is reused; GPU/SyncBN training must still be validated on the target server.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:tests python -m pytest tests -q
```

CPU tests cover fixed CNN memory across scales, per-group cache isolation,
cross K/V projection reuse, gradients through both combiner inputs and
earlier ST latents after cache release, cached/full causal logits, patch layout, ordinary AR Poisson
PMFs, CTS cap/tail-completion statistics, finite-sum KL versus independently
enumerated mass and numerical gradients, MNIST/CIFAR-10/CelebA-64 multiscale
forwards and backward paths, single-group mode, AMP precision, strict checkpoint
restore, full-prior generation, actual train/validation loop losses and MC
averaging, and existing distribution/flow regressions. Synthetic tests use
the `res_elu` backbone and small Transformer widths. They are not GPU training,
dataset convergence, FID/NLL benchmarks, or gradient-unbiasedness evidence.

Validation runtime: Python 3.12, PyTorch 2.14.1+cpu, torchvision 0.29.1+cpu.
