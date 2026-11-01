#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MASTER_PORT="${MASTER_PORT:-6031}"
# Decoder order: top scale = 2 groups (4x4), next scale = 4 groups (8x8).
# Extra command-line arguments override defaults below (argparse last value wins).
python -u train.py \
  --root /root/autodl-tmp/nvae-checkpoints \
  --save mnist_poisson_group1_svi_s2_g2-4_n5 \
  --dataset mnist --data /root/autodl-tmp/datasets/mnist \
  --arch_instance res_mbconv \
  --num_latent_scales 2 --num_groups_per_scale 4 --ada_groups --min_groups_per_scale 1 \
  --num_latent_per_group 8 --num_channels_enc 16 --num_channels_dec 16 \
  --num_preprocess_blocks 2 --num_preprocess_cells 2 \
  --num_postprocess_blocks 2 --num_postprocess_cells 2 \
  --num_cell_per_cond_enc 1 --num_cell_per_cond_dec 1 --num_nf 0 \
  --num_mixture_dec 10 --num_x_bits 8 --use_se --res_dist \
  --latent_distribution poisson --poisson_gradient_estimator straight_through \
  --poisson_relaxation_temperature 0.1 --poisson_max_count 64 --poisson_max_rate 30 \
  --group1_svi_steps 5 --group1_svi_lr 0.1 --group1_svi_momentum 0.5 --group1_svi_eps 1e-5 \
  --batch_size 64 --epochs 200 --warmup_epochs 5 \
  --learning_rate 0.001 --learning_rate_min 0.0001 \
  --weight_decay 0.0003 --weight_decay_norm 0.0 \
  --kl_anneal_portion 0.3 --kl_const_portion 0.0001 --kl_const_coeff 0.0001 \
  --seed 1 \
  "$@"
