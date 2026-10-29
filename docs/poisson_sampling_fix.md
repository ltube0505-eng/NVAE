# Poisson 随机流隔离修正

## 修正内容

`poisson_sampling.py` 提供共用的 `PoissonSampler`，默认 `fresh`：每次调用从 CPU
随机源取一个新 seed，为当前设备创建新的 generator，再执行 torch.poisson。
同一个模型的所有组、所有 forward 共用一个持续推进的 CPU seed 源，不会每组都重置成同一个 seed。
该实现与用户已测试的 fresh 隔离方法一致；用户 checkpoint 的修正后收敛/性能仍需实测。

- distributions.py：exact、REINFORCE、STE 的整数前向和 O-BBVI proposal 均走同一入口。
- model.py：固定先验及所有条件先验/后验使用模型共用 sampler。
- train.py：默认 fresh；checkpoint 保存每个 rank 的 CPU sampler RNG 状态，续训按 rank 恢复。
- evaluate.py：支持后端覆盖；无后端字段的旧 checkpoint 默认 fresh。采样 seed 来自评估 --seed。
  后端同时用于验证、无条件先验生成、FID 生成及生成式 BN 重校准。
- audit_poisson_elbo.py：native/cpu/fresh 仍明确覆盖模型后端，native 诊断不会被新默认值替换。
- check_poisson_rng.py：复用正式 sampler，探针与模型使用相同实现。

默认参数为 `--poisson_sampling_backend fresh`。`cpu` 为独立 CPU 抽样对照；`native`
仅保留作历史行为复现或诊断，不推荐在已出现相关性异常的环境训练。
松弛 estimator=relaxed 的指数到达采样未更改。

STE 为：

```python
hard = sampler(rate)
z = hard + (rate - rate.detach())
```

硬计数抽样不进入 autograd；rate-identity STE 导数不变。没有 count 截断。
旧 PyTorch 的半精度率会转为 float32 抽样，再转回原设备、原 dtype。
单卡 CPU sampler 状态保存/恢复和多 rank 收集逻辑已经有 CPU 测试；实际多卡 NCCL
通信与 PyTorch 1.6 CUDA 修正后模型运行仍需用户环境验证。

## 现有 checkpoint 先验生成

在仓库目录运行，GPU 0 和端口 6021 按实际可用资源调整：

```bash
MASTER_PORT=6021 CUDA_VISIBLE_DEVICES=0 python -u evaluate.py \
  --checkpoint /root/autodl-tmp/nvae-checkpoints/eval-mnist_s5_g3-2-4-4-2_hw4-4-8-8-8_poisson_ste/checkpoint.pt \
  --save /root/autodl-tmp/nvae-evaluation/mnist_s5_poisson_ste_prior_fresh \
  --eval_mode sample --temp 1.0 --readjust_bn \
  --poisson_sampling_backend fresh \
  --world_size 1 --local_rank 0 --master_address 127.0.0.1 --seed 1
```

生成脚本仍固定每批 100 张，共 50000 张。readjust_bn 使用 fresh 先验样本，在评估
进程内更新 BN，不写回原 checkpoint。仅换抽样路径不能撤销历史训练可能受到的影响。

## 训练与续训

原训练命令加上 `--poisson_sampling_backend fresh`，建议使用带 fresh 后缀的新实验目录
从头建立干净基线。旧 checkpoint 可加载并以 fresh 继续训练，但不保证恢复已有质量，
也不能把修正前后的日志作为同一采样协议直接比较。

更新磁盘文件不会替换已经运行的 Python 进程中的旧实现；使用修正代码需要重新启动
训练/评估进程。本次修改不自动停止服务器上已有任务。

新 checkpoint 中的 poisson_rng_states 专门保存 Poisson sampler 的状态；不代表已经
实现全训练流程的逐位可重复续训，原项目其他全局 RNG/DataLoader 状态仍沿用原行为。
历史 checkpoint 没有该字段，续训时从 args.seed 和 global_rank 派生的新 seed 源启动。
