#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -u train.py \
  --root /root/autodl-tmp/nvae-checkpoints \
  --save mnist_poisson_entropy_a01_cos50_s2 \
  --dataset mnist \
  --data /root/autodl-tmp/datasets/mnist \
  --arch_instance res_mbconv \
  --latent_distribution poisson \
  --poisson_gradient_estimator straight_through \
  --poisson_relaxation_temperature 0.1 \
  --poisson_max_count 64 \
  --poisson_max_rate 30 \
  --poisson_objective elbo_alpha \
  --poisson_entropy_alpha_init 0.1 \
  --poisson_entropy_anneal_portion 0.5 \
  --num_latent_scales 2 \
  --num_groups_per_scale 4 \
  --ada_groups \
  --min_groups_per_scale 1 \
  --num_latent_per_group 8 \
  --num_channels_enc 16 \
  --num_channels_dec 16 \
  --num_preprocess_blocks 2 \
  --num_preprocess_cells 2 \
  --num_postprocess_blocks 2 \
  --num_postprocess_cells 2 \
  --num_cell_per_cond_enc 1 \
  --num_cell_per_cond_dec 1 \
  --num_nf 0 \
  --use_se \
  --res_dist \
  --batch_size 64 \
  --epochs 200 \
  --learning_rate 0.001 \
  --learning_rate_min 0.0001 \
  --warmup_epochs 5 \
  --fast_adamax \
  --weight_decay 0 \
  --weight_decay_norm 0 \
  --num_process_per_node 1 \
  --seed 1 \
  "$@"
