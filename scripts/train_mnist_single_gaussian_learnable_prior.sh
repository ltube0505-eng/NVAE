#!/usr/bin/env bash
# Single-layer Gaussian baseline with a learned diagonal prior.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

"${PYTHON:-python}" -u train.py \
  --dataset mnist \
  --data "${DATA_DIR:-/root/autodl-tmp/datasets/mnist}" \
  --root "${CHECKPOINT_ROOT:-/root/autodl-tmp/nvae-checkpoints}" \
  --save mnist_single_gaussian_learnable_prior \
  --latent_distribution normal --learnable_prior \
  --num_latent_scales 1 --num_groups_per_scale 1 --num_latent_per_group 20 \
  --num_channels_enc 32 --num_channels_dec 32 \
  --num_preprocess_blocks 2 --num_preprocess_cells 3 \
  --num_postprocess_blocks 2 --num_postprocess_cells 3 \
  --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 \
  --num_nf 0 --use_se --res_dist --arch_instance res_mbconv --num_x_bits 8 \
  --batch_size 200 --epochs 400 --learning_rate 1e-2 --learning_rate_min 1e-4 \
  --warmup_epochs 5 --weight_decay 3e-4 --weight_decay_norm 1e-2 \
  --kl_anneal_portion 0.3 --kl_const_portion 1e-4 --kl_const_coeff 1e-4 \
  --num_process_per_node 1 --seed 1 \
  "$@"
