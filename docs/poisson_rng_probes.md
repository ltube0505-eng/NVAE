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

条件抽样测试先采 prefix ~ Pois(10) 或 Pois(28)，根据 prefix 是否超过对应 rate 将后续 rate 设为 28 或 0.1，
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

新增 `--prefix-rates 10,28`，使条件测试同时覆盖低/高前序发放率。此前 prefix=10 的简单
条件实验未显著复现 checkpoint 残差，不能把它误报为已经复现；高前序 rate 是重要补充。

## 已收到的 V100 / torch 1.6.0+cu92 结果

用户提供的报告包括 3 个 seed、两种尺寸，且 errors 为空。下面对 6 组实验的
相邻调用同坐标 Pearson 相关系数取算术平均：

| rate | native | cpu | fresh |
| --- | ---: | ---: | ---: |
| 28 | 0.291013 | 0.000356 | 0.001049 |
| 30 | 0.335782 | -0.000037 | 0.000773 |

native 的 rate=28 相关范围为 [0.290184,0.292080]，rate=30 为
[0.334530,0.336698]，全部 seed 和尺寸复现。均值、Fano 同时接近正确值，
说明固定 rate 的边缘分布统计没有明显失败，但连续调用未表现出独立性。
单次坐标间的已测 lag 相关性也没有出现类似 0.3 的强异常。

原条件探针使用 prefix~Pois(10)，其残差没有强烈、稳定地异常。因此不能把该探针
当作已复现模型中的 +25 nats 残差。新增 prefix~Pois(28)，针对高 rate 的前序调用。

fresh 已消除本轮固定 rate 测试中的强跨调用相关，但尚不能据此宣布完整模型问题已修复。
现在 audit_poisson_elbo.py 支持 --sampler fresh，对同一份 checkpoint 快照比较
cpu/fresh 的 MC-minus-analytic、残差标准误及重建项。每次 GPU 调用的新 seed
来自独立的 CPU 生成器；不会在每个 forward 重置成同一个 seed。

已核对 PyTorch v1.6.0 的 Distributions.cu：_s_poisson_cuda 为一次调用申请固定
philox_engine_inputs(20)。cuRAND 的公开 Knuth 实现对较小 rate 逐次消费随机数，
随机消耗量可超过 20；状态重叠是与现象高度吻合的机制解释，但这里未直接取得用户
运行时 cuRAND 9.2 对应头文件或做内核级跟踪，不把机制解释当成额外实测结论。

源码参考：
https://github.com/pytorch/pytorch/blob/v1.6.0/aten/src/ATen/native/cuda/Distributions.cu
https://docs.nvidia.com/cuda/curand/curand/curand__poisson_8h_source.html
