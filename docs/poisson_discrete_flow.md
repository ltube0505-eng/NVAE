# Poisson NVAE：保持非负整数的条件相邻交换 flow

本实现支持 `--latent_distribution poisson --poisson_gradient_estimator straight_through --num_nf 1`。
它是新增的实验性离散后验 flow，不是 NVAE 原论文的 Gaussian 加性 flow。
保留 Poisson 先验和 Poisson **基础**后验；经过 flow 后的后验通常不再是 Poisson。

## 1. 标量双射：两种相邻交换

令计数 $n\in\mathbb N_0$、固定二值门 $b\in\{0,1\}$ 和配对偏移 $\delta\in\{0,1\}$。
定义

$$
T_{\delta,b}(n)=
\begin{cases}
n,&n<\delta,\\
n+b\,[1-2((n-\delta)\bmod 2)],&n\ge\delta.
\end{cases}
$$

- $b=0$：恒等变换。
- $b=1,\delta=0$：交换 $(0,1),(2,3),(4,5),\ldots$。
- $b=1,\delta=1$：固定 $0$，交换 $(1,2),(3,4),(5,6),\ldots$。

写 $n=\delta+2k+r$，其中 $r\in\{0,1\}$，则
$T_{\delta,b}(n)=\delta+2k+(r\oplus b)$。因此配对区块不变、奇偶位按门翻转。
每个区块内是一个排列，边界固定，故输出非负且为整数、无有限计数上限，并且

$$T_{\delta,b}^{-1}=T_{\delta,b},\qquad T_{\delta,b}(T_{\delta,b}(n))=n.$$

这里没有对最终计数使用 `clamp`、`round` 或全局取模压缩。

## 2. 自回归神经网络与多维可逆性

固定当前 group 的条件 $c_i=(x,z_{<i})$。编码器特征 $h_i=h_\phi(c_i)$ 在整个
group 的正逆变换中保持不变。将 group 内的空间/通道坐标按顺序编号。
对一个 flow cell 定义

$$
\ell_j=g_{\psi,j}(u_{<j},h_i),\qquad
b_j=\mathbf 1\{\ell_j\ge0\},\qquad
z_j=T_{\delta,b_j}(u_j).
$$

**门不能读取当前 $u_j$ 或未来坐标。** 这保证同一前缀下两个配对计数使用同一门。
反例：若对 $n=0$ 用 $b=1$，对 $n=1$ 用 $b=0$，两个输入都变成 $1$，双射就失效。

网络使用原仓库的 masked AR convolutions：`log1p(u)` 经一个严格零对角
3×3 masked convolution（6 倍隐藏通道）、5×5 masked depthwise convolution、
ELU 激活和 1×1 masked 输出头；另加固定 $h_i$ 的 1×1 特征投影。
门网络没有 BatchNorm、dropout 或重新抽样的随机门。输出头偏置初始化为 -2，
小权重尺度使初始变换倾向恒等，但不对所有可能输入承诺恒等初始化。

顺序为先 raster 空间位置、每个位置内递增通道。mirror cell 使用反向 raster，
但同一位置的通道顺序仍为递增。mask 的严格性有输入 Jacobian 测试。

逆变换从第一个坐标开始。若 $u_{<j}$ 已恢复，用相同网络计算 $b_j$，然后

$$u_j=T_{\delta,b_j}(z_j).$$

由归纳法每个坐标都有唯一前像；任何非负整数目标都能恢复一个非负整数源。
所以整个 cell 是 $\mathbb N_0^d\to\mathbb N_0^d$ 的双射。
一个 `num_nf` block 依次运行 $\delta=0$ 正向 cell 和 $\delta=1$ mirror cell。
不同配对避免计数永远困在同一对中。多个 block 是双射的复合，逆变换逆序执行。
每个 cell 的每个计数最多移动 1，因此 $L$ 个 block 的坐标位移最多 $2L$。

参考 `inverse()` 是逐坐标实现，用于验证/外部密度计算；图像大小 group 上很慢。
实际采样、NELBO 和 IW-NLL 在前向时已知基础样本，不需要运行 inverse。

## 3. 概率质量、归一化、依赖与熵

基础后验为

$$q_{0,i}(u\mid c_i)=\prod_j\operatorname{Pois}(u_j;\lambda_{ij}(c_i)).$$

令 $F_i$ 为所有 flow block 的复合，则

$$
q_{F,i}(z_i\mid c_i)=q_{0,i}(F_i^{-1}(z_i;c_i)\mid c_i).
$$

双射使求和可以重编号：

$$
\sum_{z_i\in\mathbb N_0^d}q_{F,i}(z_i\mid c_i)
=\sum_{u\in\mathbb N_0^d}q_{0,i}(u\mid c_i)=1.
$$

连续 Jacobian 不适用于这里。API 返回零 correction 表示“不修正离散概率质量”，
不是声称 ST sigmoid 代理具有单位 Jacobian。
前向计算 $u\sim q_0$、$z=F(u)$ 时，**log_q 在基础样本 $u$ 上计算**，
**log_p 在变换后的 $z$ 上计算**。不能在 $z$ 上套基础 Poisson 后验的公式。

后验可以出现组内相关性，但双射只重排联合概率质量。在给定 $c_i$ 下
$H(q_{F,i})=H(q_{0,i})$；单个坐标的边缘熵可以改变。
固定基础分布的双射不是任意目标 PMF 的通用逼近器；基础 rates 可学习，但也
不能据此宣称整个模型能表示任意后验。

## 4. 分层 ELBO、KL 与重要性权重

Poisson 先验维持原生成模型：$p_i(z_i\mid z_{<i})$。
整个后验的精确 log mass 是

$$\log q_F(z\mid x)=\sum_i\log q_{0,i}(u_i\mid x,z_{<i}),\quad
u_i=F_i^{-1}(z_i;x,z_{<i}).$$

group 的条件 KL 为

$$
K_i(c_i)=\mathbb E_{u_i\sim q_{0,i}}[
\log q_{0,i}(u_i\mid c_i)-\log p_i(F_i(u_i;c_i)\mid z_{<i})].
$$

使用当前前向样本计算 $\widehat K_i$。没有 flow 时的 Poisson--Poisson
解析 KL 对变换后后验不再成立；单个 MC KL 值可以为负，**不要截到零**。
group 内逐坐标差仅是联合 log mass 的一种加和分配，不是输出坐标的边缘 KL。

$$
\widehat{\mathrm{NELBO}}=-\log p_\theta(x\mid z)
 +\sum_i[\log q_{0,i}(u_i\mid c_i)-\log p_i(z_i\mid z_{<i})].
$$

固定参数下这是有效 ELBO 的无偏数值 MC 估计（期望存在时）。warm-up/balancing
仍按原训练代码；$\beta\ne1$ 或组间重加权时，训练目标不能称为标准 NELBO。
验证使用 $\beta=1$、不启用 balancing。

$$\log w=\log p_\theta(x\mid z)+\log p_\theta(z)-\log q_F(z\mid x).$$

已有 IW-NLL 路径继续使用这些正确 log mass。
所有 rates 严格为正，基础 Poisson 在整个 $\mathbb N_0^d$ 上有质量，双射保留
全支撑；因而先验不会因为负数/非整数而赋零概率。

## 5. ST 梯度与精确前向计数

门确定性地阈值化；**不是**从 Bernoulli 门分布随机抽样，不额外添加门概率。
仅反向使用 $s_j=\sigma(\ell_j/\tau_F)$。
令 $d_j=\mathbf1\{u_j\ge\delta\}[1-2((u_j-\delta)\bmod2)]$，反向冻结其值，
构造代理 $v_j=u_j+s_j\operatorname{stopgrad}(d_j)$，实际张量为

$$z_j^{\rm ST}=z_j^{\rm hard}+v_j-\operatorname{stopgrad}(v_j).$$

代码用 `hard + (surrogate - surrogate.detach())` 保持前向位级整数。
这是有偏梯度，不能宣称是离散 ELBO 的无偏梯度或声称代理本身可逆。
`poisson_flow_temperature` 只控制门的 backward sigmoid，与基础到达时间的
`poisson_relaxation_temperature` 是不同参数。

flow 路径还修正基础 ST 的计数截断。先生成前 $M$ 个到达时间；若第 $M$ 个
到达时间 $T_M<1$，条件剩余事件数为

$$N_{\rm tail}\mid T_M\sim\operatorname{Pois}(\lambda(1-T_M)).$$

训练前向计数为 $\sum_{k=1}^M\mathbf1\{T_k\le1\}+N_{\rm tail}$，其中
$T_M\ge1$ 时 tail 为 0。由 Poisson 过程在到达时刻的独立增量/无记忆性质，
这是完整 $\operatorname{Pois}(\lambda)$ 样本，无计数上限。反向仍用前 $M$ 个
到达时间的松弛值，tail detach；精确前向不等于无偏反向。
`num_nf=0` 的旧采样行为保留，避免改变已有实验。

flow 的 log mass 使用 `Poisson.log_p(..., strict=True)`：负数、非整数、NaN、
Inf 返回 $-\infty$，不会静默把负数截成 0。
旧 relaxed 路径保留默认的连续扩展表达式，不能把它解读为精确 Poisson PMF。

## 6. 使用与验证范围

```bash
python train.py ... \
  --latent_distribution poisson \
  --poisson_gradient_estimator straight_through \
  --num_nf 1 \
  --poisson_flow_temperature 1.0
```

生成先验样本不运行后验 flow。精确验证从基础 `torch.poisson` 抽样再运行硬门。
REINFORCE/O-BBVI 目前没有为确定性排列参数实现训练估计器，明确禁止搭配 flow；
relaxed 非整数前向和 mixed Poisson/Gamma 也明确禁止。Gaussian flow 路径保持。

测试覆盖：两个配对的标量 involution 和大计数、神经 mask 严格性、正逆两个方向、
枚举 joint PMF 归一化/相关性/采样 KL/IW identity、非法输入的 strict PMF、
小到达预算下的 Poisson tail 补采样、MNIST/CIFAR10/CelebA64 完整小宽度模型的
forward/backward、多个 flow block、生成路径不调用 flow、state dict round trip。
彩色图像测试保留 10 分量输出似然；CIFAR10 与 CelebA64 测试保留原论文尺度和
group 数，缩小网络通道以在 CPU 上执行。这些不是完整 GPU 训练或画质验证。

```bash
OMP_NUM_THREADS=1 python -m pytest -q
```

本次验证（Python 3.12、PyTorch 2.14.1 CPU）：**31 tests passed**，并检查了
修改文件的 Python 3.7 语法兼容性。未在原 PyTorch 1.6/V100 环境执行 GPU 训练。
双变量枚举采用基础 rates $(0.7,1.4)$、先验 rates $(1.1,1.8)$、每坐标 0 到 24；
表中归一化是忽略极小尾部后的数值验证，无限支撑上的归一化由双射证明保证。

| 验证项 | 数值 |
| --- | ---: |
| 枚举输入 / 唯一输出数 | 625 / 625 |
| 逆变换最大误差 | 0 |
| 后验概率质量和 | 1.0000000000000002 |
| 输出双变量协方差 | 0.16228363647916821 |
| 枚举 KL | 0.3356090293746657 |
| $\sum q_F(z)\,p(z)/q_F(z)$ | 1.0 |

## 7. 与已有工作的关系

- Tran et al., [Discrete Flows: Invertible Generative Models of Discrete Data](https://arxiv.org/abs/1905.10347), NeurIPS 2019：离散双射的质量变换、XOR/有限类别模运算、自回归结构。
- Hoogeboom et al., [Integer Discrete Flows and Lossless Compression](https://arxiv.org/abs/1905.07376), NeurIPS 2019：整数 coupling 和 ST 训练；通常工作在允许负值的 $\mathbb Z^d$。

这里的 staggered adjacent swaps 是为本项目构造的 $\mathbb N_0^d$ 适配设计，
不是声称上述论文原样给出了本实现或证明了 Poisson NVAE 的实验优势。
