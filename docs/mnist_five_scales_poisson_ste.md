# MNIST：五个逻辑 scale，两个空间分辨率

以下编号均按生成顺序。

| scale | groups | 每组隐层形状 |
| --- | --- | --- |
| 1 | 3 | 20 × 4 × 4 |
| 2 | 2 | 20 × 4 × 4 |
| 3 | 4 | 20 × 8 × 8 |
| 4 | 4 | 20 × 8 × 8 |
| 5 | 2 | 20 × 8 × 8 |

共 15 组。新增的 `latent_scale_groups`、`latent_scale_sizes` 都按生成顺序指定，
长度必须等于 `num_latent_scales`。显式 groups 优先于 `ada_groups` 和
`num_groups_per_scale`；后两项保留以对应参考实验 args，不再决定本次分组。
内部 `groups_per_scale` 仍按 encoder 顺序保存，日志应为 `[2,4,4,2,3]`。

相同分辨率的 scale 仍具有独立可训练初始特征，其首组先验/后验只接受上一 scale
末组样本的桥接条件，采样后从本 scale 初始特征开始组内传播。仅 2→3 边界改变
空间分辨率（4→8）；对应的隐变量条件特征通道由 256 变为 128。
图像解码器融合全部 15 组，只有 2→3 边界上采样一次，之后通过两个 postprocess
上采样块从 8×8 恢复 32×32。MNIST 的有效重建区域仍为 28×28。

KL balancing、KL warm-up、Adamax、学习率调度和正则规则沿用参考实现。
KL balancing 仍按逻辑 scale 索引和组数计算，而非合并相同空间大小的 scale；
每一组 KL 均保留。未指定这两个新参数的历史配置保持原架构。

## 训练命令

在 `feature/scale-last-sample-poisson-ste` 分支的最新代码目录运行：

```bash
python train.py \
  --root /root/autodl-tmp/nvae-checkpoints \
  --save mnist_s5_g3-2-4-4-2_hw4-4-8-8-8_poisson_ste \
  --dataset mnist \
  --data /root/autodl-tmp/datasets/mnist \
  --arch_instance res_mbconv \
  --num_latent_scales 5 \
  --latent_scale_groups 3,2,4,4,2 \
  --latent_scale_sizes 4,4,8,8,8 \
  --num_groups_per_scale 10 --ada_groups --min_groups_per_scale 1 \
  --num_latent_per_group 20 \
  --num_channels_enc 32 --num_channels_dec 32 \
  --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 \
  --num_preprocess_blocks 2 --num_preprocess_cells 3 \
  --num_postprocess_blocks 2 --num_postprocess_cells 3 \
  --num_mixture_dec 10 --num_x_bits 8 \
  --latent_distribution poisson \
  --poisson_gradient_estimator straight_through \
  --poisson_max_rate 30 --poisson_max_count 64 \
  --poisson_relaxation_temperature 0.1 \
  --scale_connection last_sample --num_nf 0 \
  --res_dist --use_se --fast_adamax \
  --batch_size 200 --epochs 200 --warmup_epochs 5 \
  --learning_rate 0.01 --learning_rate_min 0.0001 \
  --weight_decay 0.0003 \
  --weight_decay_norm 0.01 --weight_decay_norm_init 10 \
  --kl_anneal_portion 0.3 --kl_const_coeff 0.0001 --kl_const_portion 0.0001 \
  --reinforce_num_samples 1 --score_baseline_decay 0.9 \
  --seed 1 --num_proc_node 1 --num_process_per_node 1 \
  --node_rank 0 --local_rank 0 --global_rank 0 --master_address 127.0.0.1
```

不传 `--cont_training`，从新实验目录开始训练；不传 `--weight_decay_norm_anneal`，
保持 `False`。`distributed` 由入口自动设置，`num_total_iter` 由数据加载器长度乘
训练轮数自动计算，不是可传入的命令行选项。STE 不使用 max_count 和松弛温度。

checkpoint 路径：
`/root/autodl-tmp/nvae-checkpoints/eval-mnist_s5_g3-2-4-4-2_hw4-4-8-8-8_poisson_ste/checkpoint.pt`。
