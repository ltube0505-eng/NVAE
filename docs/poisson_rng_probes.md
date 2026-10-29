# 下一步：独立检查 Poisson 随机抽样

用户在 torch 1.6.0+cu92 上的 checkpoint 183 对照显示：CPU 抽样的
MC-NELBO - analytic-NELBO 约 -0.023，native CUDA 约 +25.094。
这定位了需要调查的抽样路径，但尚未确认具体 CUDA/cuRAND 缺陷或相关性机制。
下面的测试不加载模型，不初始化分布式进程，不修改训练/生成代码。

## 命令

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0 python -u check_poisson_rng.py \
  --device cuda:0 --batch-size 32 --calls 32 --seeds 1,2,3 \
  --sizes 4,8 --rates 0.01,1,3,10,28,30 \
  --backends native,cpu,fresh \
  --out /root/autodl-tmp/nvae-evaluation/poisson_ste_elbo_audit/rng_probes.json
```

先用 batch=32 对应此前 audit；需要排查训练 batch 几何时，再用 `--batch-size 200`
运行，并使用不同输出文件名。`--dtype float64` 可检查计数类型相关性，不会更换 GPU
Poisson 底层算法。

## 三种抽样路径

- native：直接 torch.poisson(rate)，使用调用进程的默认生成器。
- cpu：将 rate 复制到 CPU，抽样后送回 GPU。
- fresh：每次调用创建对应设备的生成器，使用 CPU 随机生成的新 seed；这是隔离随机流的
  诊断实验，未通过目标设备的统计检查前不能宣称修复成功。老构建不支持时会明确记录错误，
  不会悄悄退回 native。

## 两类检查

固定 rate 测试对张量 [B,20,4,4]、[B,20,8,8] 连续抽样：

- mean 应接近 rate；variance/mean（Fano）应接近 1；零计数比例应接近 exp(-rate)。
- within_flat_lags 检查扁平数组在若干间隔的相关性；这些配对可能跨图像边界。
- across_images 检查相邻图像相同坐标的相关性。
- across_calls 检查相邻调用相同坐标的相关性。
- equal_fraction 与独立 Poisson 样本的理论相等概率比较；离散样本自然可能重复，
  不能把正常重复当作 RNG 故障。

条件抽样测试先采 prefix ~ Pois(10)，根据 prefix>10 将后续 rate 设为 28 或 0.1，
再抽 z。用固定先验 rate=1 计算 sum (z-rate)*log(rate)。相同分辨率和 4→8 上采样
条件分别测试。如果条件分布正确，其期望应为零。

输出 conditional_iid_se 假设坐标和调用正确独立，不是相关样本下有效的标准误。
call_block_se 允许同一调用内部相关，但仍假设不同调用独立。
因此还要查看各次调用的均值和不同 seed 的结果，不能只把 IID z-score 当作最终证据。
大量相关性检查涉及多重比较，单个偏离不能自动认定缺陷。

## 判读

- 固定 rate 的均值、Fano 就异常：优先检查边缘分布。
- 边缘统计正常，但坐标/跨调用相关显著且多 seed 重复：优先检查联合独立性。
- 固定 rate 正常，条件抽样残差持续异常：进一步检查前后调用随机流与条件 rate 的依赖。
- fresh 在 native 失败的项目上恢复、CPU 也通过：支持随机流复用/相关性方向，
  但仍应回到训练 checkpoint 验证。
- 三种路径都通过：不能据此排除模型特定 rate 模式或更长历史相关性，需要对 checkpoint
  做多 seed 检查、捕获真实 rate，或更换独立 PyTorch/CUDA 环境复核。

本地验证使用现代 CPU torch，不能替代用户的 CUDA 9.2 运行结果。
