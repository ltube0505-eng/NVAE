#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -u train.py \
  --root /root/autodl-tmp/nvae-checkpoints \
  --save mnist_poisson_st_official_s2_t01_e200_nvae_alpha01_cos50 \
  --dataset mnist \
  --data /root/autodl-tmp/datasets/mnist \
  --arch_instance res_mbconv \
  --latent_distribution poisson \
  --poisson_gradient_estimator straight_through \
  --poisson_relaxation_temperature 0.1 \
  --poisson_max_count 64 \
  --poisson_max_rate 30 \
  --poisson_objective nvae_alpha \
  --poisson_entropy_alpha_init 0.1 \
  --poisson_entropy_anneal_portion 0.5 \
  --poisson_rate_monitor_every 500 \
  --kl_anneal_portion 0.3 \
  --kl_const_portion 0.0001 \
  --kl_const_coeff 0.0001 \
  --num_latent_scales 2 \
  --num_groups_per_scale 10 \
  --ada_groups \
  --min_groups_per_scale 1 \
  --num_latent_per_group 20 \
  --num_channels_enc 32 \
  --num_channels_dec 32 \
  --num_preprocess_blocks 2 \
  --num_preprocess_cells 3 \
  --num_postprocess_blocks 2 \
  --num_postprocess_cells 3 \
  --num_cell_per_cond_enc 2 \
  --num_cell_per_cond_dec 2 \
  --num_mixture_dec 10 \
  --num_nf 0 \
  --num_x_bits 8 \
  --use_se \
  --res_dist \
  --batch_size 200 \
  --epochs 200 \
  --learning_rate 0.01 \
  --learning_rate_min 0.0001 \
  --warmup_epochs 5 \
  --fast_adamax \
  --weight_decay 0.0003 \
  --weight_decay_norm 0.01 \
  --weight_decay_norm_init 10.0 \
  --reinforce_num_samples 1 \
  --score_baseline_decay 0.9 \
  --obbvi_num_samples 8 \
  --obbvi_taus 1.0,3.0 \
  --num_proc_node 1 \
  --num_process_per_node 1 \
  --node_rank 0 \
  --local_rank 0 \
  --global_rank 0 \
  --master_address 127.0.0.1 \
  --seed 1 \
  "$@"

