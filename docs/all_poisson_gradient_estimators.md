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
| `obbvi` | 仅当前组从过度离散 proposal 精确采样，后续组从条件后验重采样 | 逐组条件 DMIS score；逐样本或解析 KL 目标 | 是（支撑覆盖且矩存在时） |

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

`reinforce` 和 `obbvi --obbvi_objective sampled` 的直接导数不加入解析 KL；解析 KL 用于监控。`obbvi --obbvi_objective analytic_kl` 对解析 KL 求直接导数，同时将后续 KL 值纳入各组 score 系数。默认 `--kl_balance_mode original` 保留原行为：路径梯度模式使用当前批次的 KL balancing，score 模式仅使用统一标量 KL warm-up 系数 \(\beta\)。若要比较相同 KL 权重下的各估计器，可统一启用下文的 `shared_lagged`。

## 3. 沿 NVAE 顺序的逐组条件 proposal

论文的 Poisson proposal 为

\[
q_i(z_i\mid z_{<i},x)=\operatorname{Pois}(\lambda_i),
\qquad
r_{ij}(z_i\mid z_{<i},x)=
\operatorname{Pois}\!\left(\lambda_i^{1/\tau_j}\right),
\quad \tau_j\ge 1.
\]

给定基准轨迹 \(z^{(0)}\sim q_\phi\)，逐组固定其前缀 \(z_{<i}^{(0)}\)。对于组 \(i\)，从 \(r_{it}\) 重新采样 \(z_i\)，把它注入 decoder combiner；随后重算先验头、融合后验头及 decoder 状态，按 \(q_j(z_j\mid x,z_{<j})\) 逐组重采样整个后缀。对应完整轨迹的抽样分布是

\[
\widetilde q_{it}(z\mid x)=q_\phi(z_{<i}\mid x)\,r_{it}(z_i\mid x,z_{<i})\,
q_\phi(z_{>i}\mid x,z_{\le i}).
\]

对固定的组 \(i\)，实现 deterministic multiple importance sampling（DMIS）：每个 \(r_{it}\) 抽相同数量的条件轨迹，权重只含被替换的这一组：

\[
m_i(z_i\mid x,z_{<i})=\frac1J\sum_{t=1}^{J}r_{it}(z_i\mid x,z_{<i}),
\qquad w_i=\frac{q_i(z_i\mid x,z_{<i})}{m_i(z_i\mid x,z_{<i})}.
\]

`--obbvi_objective sampled` 使用逐样本 \(\log q-\log p\)。对最小化 NELBO，第 \(i\) 组的 score 系数是

\[
\widehat g_i^{\rm score}=\frac1S\sum_s w_i^{(s)}
\operatorname{stopgrad}\left[R_{\rm loss}(z^{(s)})+
\beta\sum_{j=i}^G(\log q_j-\log p_j)-b\right]
\nabla_\phi\log q_i(z_i^{(s)}\mid x,z_{<i}^{(0)}).
\]

其直接导数另用一条基准后验轨迹上的 \(R_{\rm loss}-\beta\sum_j\log p_j\) 估计。`--obbvi_objective analytic_kl`（默认）使用同一组条件 proposal，但把全部组的 KL 积分为解析式 \(K_j(z_{<j})\)。这时组 \(i\) 的 score 系数变成 \(R_{\rm loss}+\beta\sum_{j>i}K_j-b\)；基准轨迹的直接导数是 \(R_{\rm loss}+\beta\sum_j K_j\)，包含每个 \(K_i\) 对参数的直接导数。对于最小化符号，以上所有符号与最大化 ELBO 的公式相反。Poisson 本组 KL 为 \(\sum_d[\lambda^q_d\log(\lambda^q_d/\lambda^p_d)+\lambda^p_d-\lambda^q_d]\)。

基准轨迹的离散样本与复用的前缀均从 autograd 图分离；每个组的 score、proposal 权重、KL 值和 baseline 在 score 系数内固定。基准轨迹的解析 KL 本身正常反传。直接导数**只计算一次**，逐组 score 求和。默认每组 `--obbvi_taus 1.0,3.0`、`--obbvi_num_samples 8`，样本数必须能被 tau 数整除。训练代价约为 \(1+G\times S\) 次前向传播，早期组需重算最长的后缀。

### 跨估计器使用相同的 KL balancing

启用 `--kl_balance_mode shared_lagged` 后，所有估计器都使用相同的 NVAE `kl_balancer` 公式（同一 `alpha_i`、KL warm-up 系数、归一化方式）；只用**上一 minibatch** 的解析 group KL 计算本批次系数 \(c_j\)，然后将它们固定。第一批没有历史值时取 \(c_j=1\)；当 \(\beta=1\) 时，与原 NVAE 一样取 \(c_j=1\)。该模式会改变原有路径梯度模式在 warm-up 期间使用当前批次系数的细节，因此比较实验需要四种估计器**全部**使用此选项。

固定 \(c_j\) 后，训练中的 KL 部分是 \(\beta\sum_j c_j K_j\)。`sampled`/REINFORCE 的逐样本 KL 和 prior 的直接梯度均逐组乘 \(c_j\)；解析 KL O-BBVI 的第 \(i\) 组 score 使用 \(\beta\sum_{j>i}c_jK_j\)，基准轨迹的直接导数使用 \(\beta\sum_j c_j\nabla K_j\)。若从**当前**轨迹计算 \(c_j\) 再用于 score，它可能依赖所抽出的离散 latent，简单地 `detach()` 不能保证上述 score 推导成立。上一批系数解决这一依赖；它和常见的 stop-gradient KL balancing 一样仍是动态训练启发式，而不是对固定原始 ELBO 的无偏梯度。用于报告模型质量的评估 NLL 保持不变；warm-up 期间的训练损失不再直接是标准 NELBO。

配置强制 mixture 含有 \(\tau=1\)。此时这一组的一个分量正好是 \(q_i\)，从而逐点有

\[
m_i(z_i)\ge \frac1Jq_i(z_i),\qquad 0\le w_i(z_i)\le J,
\]

避免当前组 importance weight 上溢。TensorBoard 记录各组 ESS 的平均 `train/importance_ess`。

当前实现采用固定的 \(\tau_t\)，没有实现论文中可选的每变量在线 dispersion 更新。以上条件重采样是针对 NVAE 条件后验的额外构造；论文的 mean-field 公式不能直接移用。单个基准轨迹的前缀引入额外方差，组数多时计算开销很大。

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

# 逐组条件 O-BBVI，每组 2 个 proposal、每个 proposal 4 条轨迹
python train.py ... --latent_distribution poisson --num_nf 0 \
  --poisson_gradient_estimator obbvi \
  --obbvi_taus 1.0,3.0 --obbvi_num_samples 8 \
  --obbvi_objective analytic_kl \
  --score_baseline_decay 0.9

# 若要使用逐样本 log q/p 版本，将上一条命令的目标切换为：
# --obbvi_objective sampled

# 若要按相同 KL balancing 方案比较四种估计器，在每条训练命令均追加：
# --kl_balance_mode shared_lagged

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
7. 全模型条件 O-BBVI 固定前缀、只对当前组加权、两种目标产生有限梯度；
8. 改变目标组计数时重新计算后续组的条件参数；小型双层例子中两种 score 公式在均匀及非均匀固定 KL 权重下均与精确目标的数值导数一致；
9. 生成路径仍使用精确 Poisson 先验采样。
