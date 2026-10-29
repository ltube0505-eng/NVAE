# 全 Poisson 隐变量 + STE

基于 `feature/scale-last-sample-bridge` 的提交 `aefc807266bc5eb9f12f8a00c014bc65c2c0495b`。
新训练默认使用 `latent_distribution=poisson`、`poisson_gradient_estimator=straight_through`、
`scale_connection=last_sample`、`num_nf=0`。所有 scale、所有 group 的后验和先验都是
Poisson；第一组先验为 Poisson(1)，后续组先验的 rate 由条件网络产生。

## 前向和反向

```python
hard = sampler(rate)  # 默认 fresh：每次调用独立的新设备 generator
z = hard + (rate - rate.detach())
```

前向为精确 Poisson 非负整数。反向使用 `dz/d(rate) = 1`，再通过已有 rate 参数化
`exp(clamp(soft_clamp5(raw_log_rate), max=log(max_rate)))` 链式传播到网络参数。
这不意味着对 raw log-rate 的导数也等于 1；rate 达到硬上限时，该 clamp 的导数为零。
STE 是有偏代理梯度。它替换了父分支使用截断 hard-arrival count 和 soft-count 梯度的 ST。

`poisson_max_count` 和 `poisson_relaxation_temperature` 不参与 STE；它们只用于
显式选择的 `relaxed` 模式。`poisson_max_rate` 仍限制分布 rate，但样本可以超过 64。
验证、先验生成和生成式 BN 重校准均经过共用 sampler 精确抽样，默认 fresh 后端。
Poisson 的采样温度 `t` 沿用原实现，不改变 latent rate；比较此模型时使用 `t=1`。

每组保留解析 Poisson KL：`rate_q * (log_rate_q - log_rate_p) + rate_p - rate_q`。
重建项经 STE 反传，KL 使用解析梯度；训练保留 KL annealing、KL balancing 和已有正则。

## 保留的结构

上一尺度的最后一组真实样本只为下一尺度首组的先验/后验提供跨尺度条件。
桥接不 detach；独立图像解码器继续接收所有尺度的全部样本，重建梯度可以到达每组后验。
MNIST/Omniglot 的观测似然仍为 Bernoulli；“全 Poisson”指隐变量。
模型头仍保留原先双参数的输出形状，其中 Poisson 只使用第一半作为 log-rate。

## 两尺度 MNIST 示例

```bash
python train.py \
  --root /root/autodl-tmp/checkpoints \
  --save mnist_s2_last_sample_poisson_ste \
  --dataset mnist \
  --data /root/autodl-tmp/datasets/mnist \
  --num_latent_scales 2 \
  --num_groups_per_scale 8 \
  --num_latent_per_group 20 \
  --num_channels_enc 32 --num_channels_dec 32 \
  --latent_distribution poisson \
  --poisson_gradient_estimator straight_through \
  --scale_connection last_sample --num_nf 0 \
  --poisson_max_rate 30 \
  --res_dist --weight_decay_norm 1e-2 \
  --batch_size 200 --epochs 200
```

示例参数未进行完整训练验证，batch size 需按显存调整。默认尺度数仍为 1，
因此需要如上显式设置至少两个尺度才能启用跨尺度桥接和独立图像解码器。
请使用新的实验目录从头训练；Gaussian/Gamma checkpoint 不能当作训练好的 Poisson 模型。
继续训练时需保持完整模型配置一致。`evaluate.py` 从 checkpoint args 恢复模型配置。
为兼容历史 checkpoint，缺少分布/连接字段的 args 在模型构造器中仍按 Normal/feature 解释；
新训练 parser 会保存本分支的显式默认值。历史 Poisson checkpoint 缺少后端字段时也改用 fresh，评估随机行为随之变化；
显式 native 仅用于旧行为诊断。在本分支以 `straight_through` 续训使用 rate-STE。

显式设置旧分布或梯度估计选项仍可用于对照实验；本分支的新训练默认配置为全 Poisson + STE。

## 检查

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests
```

测试覆盖精确整数前向、无 count 截断、STE 的加权向量梯度、所有后验/先验分布、
仅重建损失到所有后验头的梯度、跨尺度边界梯度、评估/生成路径及原有架构回归测试。

五个逻辑 scale、分组 `[3,2,4,4,2]`、尺寸 `[4,4,8,8,8]` 的完整参考实验配置见
[五尺度 MNIST 训练命令](mnist_five_scales_poisson_ste.md)。

抽样默认使用 fresh 隔离每次调用的随机流；详见[修正说明](poisson_sampling_fix.md)。
