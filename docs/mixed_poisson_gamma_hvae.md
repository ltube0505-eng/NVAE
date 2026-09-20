# NVAE 中的 Poisson–Gamma 分层隐变量

## 结论

该模型在概率建模上可行，已有三条直接相关的研究线索：

1. [Deep Exponential Families](https://proceedings.mlr.press/v38/ranganath15.html) 已建立多层 Poisson / Gamma 指数族隐变量模型，并报告深层结构优于单层结构；其推断依赖 black-box variational inference。
2. [Poisson Variational Autoencoder](https://arxiv.org/abs/2405.14473) 与 [ESVAE](https://arxiv.org/abs/2310.14839) 都把 Poisson 分布用于 VAE 隐空间；ESVAE 明确令先验与后验均为 Poisson，并设计了可重参数化的脉冲采样方法。
3. [Generalized Gamma Belief Network](https://arxiv.org/abs/2408.03388) 证明非线性、非负的深层 Gamma 隐变量可以在 VAE/变分推断框架中训练。Gamma 的低方差梯度可依据[隐式重参数化梯度](https://arxiv.org/abs/1805.08498)获得。

不过，这不是“把 Normal 类名替换掉”即可得到的严格等价 NVAE。Poisson 是离散分布，没有普通 Gaussian VAE 使用的精确路径重参数化；已有的 [Overdispersed BBVI](https://arxiv.org/abs/1603.01140) 工作也专门指出并缓解了 Poisson 深层指数族模型的随机梯度高方差问题。本实现因此应视作保留 NVAE 网络骨架的实验性混合指数族 HVAE。

## 模型定义

保持 NVAE 原有的单一 top-down 路径、encoder/decoder cells、combiner、KL balancing、谱正则和训练流程不变。使用两个 latent scales，共六个 group；encoder 中的 scale 顺序为高到低 `[4, 2]`，decoder / 生成顺序为低到高 `[2, 4]`。

| Decoder group | 空间尺度 | 先验与后验 | 参数 |
|---:|---|---|---|
| 0–1 | 低分辨率 | Poisson | `log_rate` |
| 2–5 | 高分辨率 | Gamma | `log_shape`, `log_rate` |

联合分布为

$$
p_\theta(x,z_{1:6})
=p_\theta(x\mid z_{1:6})p(z_1)
\prod_{i=2}^{6}p_\theta(z_i\mid z_{<i}),
$$

其中 $p(z_1)=\operatorname{Poisson}(1)$，group 1 的条件先验仍为 Poisson，groups 2–5 的条件先验为 Gamma。后验保持 NVAE 的 top-down 条件形式 $q_\phi(z_i\mid x,z_{<i})$，并在同一祖先样本路径上计算条件 KL。观测模型统一为

$$
p_\theta(x\mid z_{1:6})=\mathcal N(\mu_\theta(z),\sigma_\theta^2(z)).
$$

训练目标仍是负 ELBO（外加原 NVAE 正则项）：

$$
\mathcal L
=-\mathbb E_q[\log p_\theta(x\mid z)]
+\sum_i\mathbb E_{q(z_{<i}\mid x)}
\operatorname{KL}\left[q(z_i\mid x,z_{<i})\|p(z_i\mid z_{<i})\right].
$$

## 顺序采样与梯度

### Poisson groups

Poisson 条件 KL 有解析式，因此 KL 本身不需要对离散样本使用 score-function：

$$
\operatorname{KL}(\operatorname{Pois}(\lambda_q)\|\operatorname{Pois}(\lambda_p))
=\lambda_q\log\frac{\lambda_q}{\lambda_p}+\lambda_p-\lambda_q.
$$

重建项必须经过 sampled count 回传。本实现把 Poisson process 写成指数到达时间 $T_k=\sum_{j\le k}E_j$，$E_j\sim\operatorname{Exp}(\lambda)$，并将指示函数 $\mathbf 1[T_k\le1]$ 的反向传播替换为 sigmoid 松弛。前向值用 hard integer count，反向值用 soft surrogate，即 straight-through estimator。

- 训练：hard-forward / soft-backward 的到达时间松弛。
- 验证：精确 `torch.poisson` 后验采样并计算精确 Poisson log-probability。
- 生成：所有 Poisson 先验均使用精确 `torch.poisson` 采样。
- 条件 KL：始终使用解析式。

为控制有限到达序列的截断误差，默认 `max_rate=30`、`max_count=64`。训练前向 count 仍可能在极小概率下被截断；生成没有该截断。

### Gamma groups

Gamma 使用 shape/rate 参数化，采样调用 PyTorch `Gamma.rsample()` 的隐式重参数化梯度；Gamma–Gamma KL 使用解析式。生成温度保持均值不变，只按 `temperature` 缩放方差。

## 已知阻碍与风险

1. **Poisson 梯度有偏。** Straight-through 松弛可降低方差，但不是离散 ELBO 重建项的无偏梯度估计。应与 REINFORCE/VIMCO/RELAX 或 generalized Gumbel-Softmax 基线对照。
2. **误差会沿层级传播。** 第二个 Poisson group 及其后的 Gamma groups 都依赖前序随机样本；一次 Monte Carlo 路径的噪声会影响所有后续条件分布。可增加每个样本的路径数，但显存与计算量近似线性增长。
3. **Poisson 的均值等于方差。** 数据若明显过度离散，Negative Binomial（Gamma–Poisson compound）通常比纯 Poisson 更合适。
4. **计数可能爆炸或塌缩为零。** 本实现限制 rate，并应持续监控每组 rate、零比例、最大 count 与 KL；rate clipping 本身也会造成边界梯度饱和。
5. **非负 latent 改变 decoder 输入统计。** 原 NVAE 针对近似零均值 Gaussian latent 调优。网络结构虽然未变，学习率、KL warm-up、batch norm 统计和初始化可能需要重新搜索。
6. **Normalizing flow 不兼容。** 原代码的 flow 是为 Normal base distribution 编写的；混合模式会拒绝 `num_nf > 0`，否则先验/后验将不再是用户指定的 Poisson/Gamma。
7. **似然口径改变。** Gaussian reconstruction 与原版彩色图像常用的 discretized mixture of logistics 不可直接比较 bits-per-dimension；对整数像素，Gaussian 还是连续密度而非离散质量。

## 使用方法

```bash
python train.py \
  --dataset cifar10 \
  --latent_distribution mixed_poisson_gamma \
  --reconstruction_distribution gaussian \
  --num_latent_scales 2 \
  --num_groups_per_scale 4 \
  --ada_groups \
  --min_groups_per_scale 2 \
  --num_nf 0 \
  --res_dist \
  --poisson_relaxation_temperature 0.1 \
  --poisson_max_rate 30 \
  --poisson_max_count 64
```

混合模式会校验 group 配置、Gaussian reconstruction 以及 `num_nf=0`，避免静默训练成与设计不同的模型。原版 NVAE 的默认参数仍使用 Normal latents 和 dataset-default reconstruction，因此已有命令与 checkpoint 保持兼容。

## 建议实验

- 先在 MNIST/CIFAR-10 上做小规模 smoke run，检查六个 group 的 KL 是否均为有限值。
- 记录两个 Poisson group 的 mean rate、零比例、达到 `max_count` 的比例，以及四个 Gamma group 的 shape/rate 分位数。
- 对比至少三个估计器：当前 straight-through、score-function + control variate、连续松弛。
- 做消融：全 Gamma、低分辨率 Negative Binomial、高分辨率 Normal，以及每个输入 1/4/8 条随机路径。
- 只有在截断比例接近零、梯度无 NaN、各 group 非塌缩后，再进行完整 NVAE 规模训练。
