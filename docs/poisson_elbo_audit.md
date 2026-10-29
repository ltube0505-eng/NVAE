# NELBO / importance-sampling NLL 一致性检查

检查基线为 `38a1955`。源码中的 Poisson 解析 KL、log-PMF、15 组汇总和 IW log-sum-exp
公式没有在 CPU 测试中出现代数不一致。这里的 CPU 测试使用构造的模型，不是用户训练的
checkpoint，也不是用户的 PyTorch 1.6 / CUDA 9.2 运行环境，不能据此宣称根因已经确定。

已有验证的 NELBO 使用解析条件 KL，而 IW-NLL 使用实际样本的 log q - log p。
同一份样本必然满足 IW-NLL <= MC-NELBO；解析 KL 估计没有逐次排序保证。
大幅、持续反转应当检查差值的标准误及数值计算，而不是直接判为正常采样噪声。

`audit_poisson_elbo.py` 在原模型上拦截 Poisson log_p/kl/sample 调用：不更新模型，
不重校准 BN，不重新训练；用 float64 对原有样本与分布参数独立核算每组概率和 KL。
支持 `--sampler cpu`：网络仍在 GPU 上运行，只把每次 Poisson 抽样放到 CPU，
用于排查设备/版本相关采样问题。CPU 与 GPU 的抽样序列不同，应比较标准化差异而非逐样本相等。

在仓库目录运行：

```bash
python audit_poisson_elbo.py \
  --checkpoint /root/autodl-tmp/nvae-checkpoints/eval-mnist_s5_g3-2-4-4-2_hw4-4-8-8-8_poisson_ste/checkpoint.pt \
  --data /root/autodl-tmp/datasets/mnist \
  --device cuda:0 --batch-size 32 --samples 20 --max-batches 10 \
  --sampler native
```

再将 `--sampler native` 改为 `--sampler cpu` 运行一次。`--max-batches 0` 表示完整验证集。
脚本输出进度和 JSON，组编号为生成顺序，从 0 开始。

主要字段：

- `analytic_nelbo`：使用原代码解析 KL 的 NELBO。
- `mc_nelbo`：同样本的重建项 + 原代码 log q - log p。
- `iw_nll`：同样本的重要性采样 NLL 估计。
- `mc_minus_analytic`：两个 NELBO 的差值。
- `poisson_residual64`：逐组求和的 (z - rate_q) log(rate_q / rate_p)，使用 float64。
- `poisson_residual_se`：在抽样正确且条件独立的假设下，由可预测条件方差
  sum rate_q * log(rate_q/rate_p)^2 估计的均值标准误；可用于检查偏离零是否显著，
  不是对错误采样器仍成立的保证。
- `total_kl_numeric_error`：原代码 KL 总和减去 float64 核算结果，包含求和误差。
- `total_log_ratio_numeric_error`：原代码总 log q - log p 与 float64 闭式直接核算的差。
- 每组 `kl_numeric_error`：原代码解析 KL 减去 float64 核算结果。
- 每组 `density_numeric_error`：原代码逐组 log q - log p 减去 float64 核算结果。
- 每组 `count_residual` / `count_residual_se`：样本总计数减条件 rate 之和的均值及理论标准误。

判读：若数值误差接近零而 residual 明显偏离零，应继续检查样本是否来自给定后验，
以及运行设备/版本；若仅 GPU native 模式持续偏离而 CPU 抽样模式不偏离，可进一步隔离 GPU
随机抽样路径，但一次 CPU 对照仍不足以认定框架存在缺陷。若数值误差解释了差异，再定位
对应组的 KL、log-PMF 或总和精度。不要通过调整日志名字或用 MC-NELBO 替换解析值来掩盖异常。

## fresh 随机流对照

新增 `--sampler fresh`：网络仍在 GPU，每次 Poisson 调用使用独立新 seed 的设备生成器，
seed 从单独的 CPU 随机源按 `--seed` 可重复地产生。此路径与 check_poisson_rng.py 的
fresh 模式一致。输出包含 seed。应先固定一份 checkpoint 快照，再在同一快照上对比
cpu/fresh；检查 MC-minus-analytic 是否接近零，而不是要求两次随机抽样的指标逐位相同。
