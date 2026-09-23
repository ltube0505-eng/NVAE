# 全 Poisson NVAE 与四种梯度估计接口

本分支新增 `--latent_distribution poisson`：NVAE 的每一个 latent group（包括顶层无条件后验、所有条件后验和对应条件先验）都使用 Poisson 分布。网络骨架、group 顺序、decoder combiner、观测似然、KL warm-up 和正则项保持不变；Poisson 模式要求 `--num_nf 0`。

## 1. 统一命令行接口

```text
--poisson_gradient_estimator relaxed|reinforce|obbvi|straight_through
```

| 取值 | 前向 latent | 后向估计 | 是否无偏估计离散目标 |
| --- | --- | --- | --- |
| `relaxed` | 连续 soft count | 到达时间的路径梯度 | 否，优化连续松弛目标 |
| `straight_through` | 整数 hard count | soft count 的路径梯度 | 否，hard-forward/soft-backward |
| `reinforce` | 从后验精确采样的整数 | score function + 上一批次 EMA baseline | 是 |
| `obbvi` | 从过度离散 proposal 精确采样的整数 | DMIS importance-weighted score function + EMA baseline | 是（支撑覆盖且矩存在时） |

`relaxed` 与 `straight_through` 使用同一组指数到达时间

\[
E_k\sim\operatorname{Exp}(\lambda),\qquad
T_k=\sum_{j\le k}E_j,
\]

并令

\[
\widetilde z=\sum_{k=1}^{K}
\sigma\!\left(\frac{1-T_k}{t}\right).
\]

- `relaxed` 直接把 \(\widetilde z\) 送入 decoder；
- `straight_through` 的实际张量为
  \(z_{\rm hard}+\widetilde z-\operatorname{stopgrad}(\widetilde z)\)，所以前向是整数、反向使用松弛梯度。

这一区分与 NegBio-VAE 中 Poisson/Gamma-arrival 松弛的语义一致，避免把“连续松弛”和“straight-through”误称为同一种估计器。

## 2. REINFORCE 目标

对一条精确后验路径 \(z\sim q_\phi(z\mid x)\)，定义带 KL warm-up 的逐样本负目标

\[
F_\beta(z)=
-\log p_\theta(x\mid z)
+\beta\bigl(\log q_\phi(z\mid x)-\log p_\theta(z)\bigr).
\]

实现分别构造：

\[
L_{\rm gen}=
-\log p_\theta(x\mid z)-\beta\log p_\theta(z),
\]

\[
L_{\rm score}=
\operatorname{stopgrad}(F_\beta-b)
\log q_\phi(z\mid x).
\]

其中 \(b\) 是仅由此前 minibatch 更新的指数移动平均。当前批次先使用旧 baseline，再更新 baseline，因此 baseline 与当前抽样独立，不改变 score estimator 的期望。

score-function 模式不再把解析 Poisson KL 直接加入反向目标；否则会把路径梯度目标和 likelihood-ratio 目标重复计算。解析 KL 仍用于监控各 group 的规模与活跃度。KL balancing 只保留在 `relaxed`/`straight_through` 模式；`reinforce`/`obbvi` 使用统一标量 KL warm-up 系数 \(\beta\)。

## 3. O-BBVI/DMIS 实现

论文的 Poisson proposal 为

\[
q_i(z_i\mid z_{<i},x)=\operatorname{Pois}(\lambda_i),
\qquad
r_{ij}(z_i\mid z_{<i},x)=
\operatorname{Pois}\!\left(\lambda_i^{1/\tau_j}\right),
\quad \tau_j\ge 1.
\]

层级模型中每条轨迹的第 \(j\) 个 proposal 是条件分布的乘积：

\[
r_j(z\mid x)=\prod_i r_{ij}(z_i\mid z_{<i},x).
\]

实现使用 deterministic multiple importance sampling（DMIS）：从每个 \(r_j\) 抽取相同数量的完整层级轨迹，并使用 balance-heuristic mixture

\[
m(z\mid x)=\frac1J\sum_{j=1}^{J}r_j(z\mid x),
\qquad
w(z)=\frac{q_\phi(z\mid x)}{m(z\mid x)}.
\]

O-BBVI 的 surrogate 为

\[
w(z)L_{\rm gen}
+\operatorname{stopgrad}\!\left(w(z)(F_\beta-b)\right)
\log q_\phi(z\mid x).
\]

proposal sample 和 importance weight 均从 autograd 图中分离；只有 `log q` 承担 score-function 梯度。默认 `--obbvi_taus 1.0,3.0`、`--obbvi_num_samples 8`。样本数必须能被 proposal 数整除。

配置强制 mixture 含有 \(\tau=1\)。此时一个分量正好是 \(q\)，从而逐点有

\[
m(z)\ge \frac1Jq(z),\qquad 0\le w(z)\le J,
\]

避免联合 importance weight 上溢。TensorBoard 另外记录 `train/importance_ess`。

当前实现采用可复现、固定的 \(\tau_j\)，没有实现论文中可选的每变量在线 dispersion 更新；它不影响固定 proposal 下估计器的无偏性。当前版本使用完整轨迹权重，数学上正确但可能比论文的逐变量 Rao–Blackwellized 权重方差更高。

## 4. GitHub 参考实现核查

GitHub code search 未找到论文作者公开的独立、可运行 O-BBVI 代码库。可核验的作者侧 GitHub artifact 是 [blei-lab/publications](https://github.com/blei-lab/publications/tree/master/2016_RuizTitsiasBlei_a)，其中保存论文 LaTeX 源文件及 Poisson proposal、DMIS、control-variate 公式。另一个公开仓库 [jamesvuc/BBVI](https://github.com/jamesvuc/BBVI) 提供通用 BBVI 和 control-variate 示例，但没有实现 O-BBVI proposal。这里的 O-BBVI 代码因此直接按论文及其官方 GitHub 论文源实现，并由恒等式测试验证，而不是复制一个来源不明的第三方实现。

松弛采样参考：

- [ltube0505-eng/NegBio-VAE](https://github.com/ltube0505-eng/NegBio-VAE) 的 `distribution.py`；
- [hadivafaii/PoissonVAE](https://github.com/hadivafaii/PoissonVAE) 的指数到达时间 Poisson 松弛。

## 5. 训练示例

以下四条命令共享同一个全 Poisson NVAE，只改变梯度估计器：

```bash
# 连续松弛前向
python train.py ... --latent_distribution poisson --num_nf 0 \
  --poisson_gradient_estimator relaxed \
  --poisson_relaxation_temperature 0.2 --poisson_max_count 64

# REINFORCE
python train.py ... --latent_distribution poisson --num_nf 0 \
  --poisson_gradient_estimator reinforce \
  --reinforce_num_samples 1 --score_baseline_decay 0.9

# O-BBVI，2 个 proposal、每个 proposal 4 条轨迹
python train.py ... --latent_distribution poisson --num_nf 0 \
  --poisson_gradient_estimator obbvi \
  --obbvi_taus 1.0,3.0 --obbvi_num_samples 8 \
  --score_baseline_decay 0.9

# 原有 hard-forward / soft-backward 方法
python train.py ... --latent_distribution poisson --num_nf 0 \
  --poisson_gradient_estimator straight_through \
  --poisson_relaxation_temperature 0.2 --poisson_max_count 64
```

验证与生成始终使用目标后验/先验的精确 `torch.poisson` 整数采样，不使用松弛值或 O-BBVI proposal。

## 6. 正确性测试

```bash
pytest -q
```

新增测试覆盖：

1. `poisson` 模式把所有 group 映射为 Poisson；
2. continuous relaxation 前向非整数，而 straight-through 前向严格为整数；
3. 两种路径估计器均有有限梯度；
4. Poisson–Poisson 解析 KL 与 PyTorch 精确结果一致；
5. O-BBVI proposal 的 \(\lambda^{1/\tau}\) 参数化、\(w\le J\) 上界及
   \(\mathbb E_m[(q/m)z]=\mathbb E_q[z]\) 恒等式；
6. 全模型 REINFORCE surrogate 可反向传播；
7. 全模型 O-BBVI 产生有限且有界的 DMIS 权重；
8. 生成路径仍使用精确 Poisson 先验采样。

