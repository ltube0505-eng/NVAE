# 单层 Gaussian VAE：参数上采样与 2×2 块选择

## 开关与范围

在原 NVAE 的单层路径上增加可选模块：

```text
--latent_distribution normal --gaussian_block_upscale
--num_latent_scales 1 --num_groups_per_scale 1 --num_nf 0
```

这是一个随机潜变量层；四个位置是同一层的混合分量，不是四个层级。
未打开 block 或 learnable_prior 开关时，不创建任何新参数，旧参数配置与 state_dict 仍兼容。
打开后必须重新训练；原 Gaussian checkpoint 没有本模块的参数。

## 普通 Gaussian 基线：可学习先验

不使用 block 上采样时，通过 `--learnable_prior` 开启可学习对角高斯先验：

\[
p_\theta(z)=\prod_{c,h,w}\mathcal N(z_{c,h,w};\mu_{c,h,w},\sigma_{c,h,w}^2).
\]

`model.prior_param` 的形状是 `[1,2C,H,W]`，前后各 C 个通道分别为
原始均值和 log_sigma。参数在所有输入图像之间共享，每个空间位置和通道
各有自己的参数；沿用 `Normal` 的 soft clamp 和尺度公式。
零初始化与旧固定先验完全一致：有效均值为 0，标准差为 1.01（仓库有 +0.01）。
训练时原有解析 Gaussian KL 对这些参数求梯度，由同一个 Adamax 优化器更新。
有 normalizing flow 时仍使用原有采样 KL。训练 log_p、IW-NLL 与先验生成
统一读取这套参数；生成温度 t 只缩放标准差。

本开关限定为 `normal`、一层、一组且关闭 `gaussian_block_upscale`；
block 模式自身已有可学习先验。单层模型不使用的 `prior_ftr0` 被冻结，
避免原手动分布式梯度归约访问不存在的梯度。

与 baseline 日志相同的网络和优化超参数（400 epochs、batch 200、lr 0.01）：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHON=/root/miniconda3/envs/nvae/bin/python \
  bash scripts/train_mnist_single_gaussian_learnable_prior.sh
```

MNIST 的 C=20、H=W=8，因此新增 2560 个先验参数。默认保存至
`/root/autodl-tmp/nvae-checkpoints/eval-mnist_single_gaussian_learnable_prior/checkpoint.pt`。
新开关会保存到 checkpoint args，先验参数保存到 state_dict；原评估入口会自动恢复。
旧 checkpoint 中没有此开关时仍构造固定先验，可严格加载；
不支持将旧固定先验 checkpoint 直接以 `--cont_training --learnable_prior` 续训，
因为模型与 optimizer 的参数集合不同。本脚本应作为新实验启动。

## 两个必须明确的数学细节

### 1. stride=1 的转置卷积本身不能放大两倍

转置卷积的单轴输出大小为

\[
H_{out}=(H_{in}-1)s-2p+d(k-1)+o+1.
\]

在 `kernel_size=3, stride=1, padding=1, dilation=1, output_padding=0`
下，输出仍为输入尺寸。为同时保留这些参数以及要求的两倍尺寸，
本实现明确加入**零插入**：

\[
U(A)_{2h,2w}=A_{h,w},\qquad
U(A)_{i,j}=0\quad\text{当 }i\text{ 或 }j\text{ 为奇数}.
\]

`U(A)` 的大小直接定义为 `2H × 2W`，包括最后一行/列的零。
随后执行所指定的 stride=1 转置卷积，输出维持 `2H × 2W`。
这是一项显式补充的尺寸约定；没有把 stride 偷换为 2，也没有使用双线性插值。

### 2. 四项 KL 求和是采用的正则，通常不是混合分布 KL 的等式

本文档保留用户要求的四项**不加权**求和作为训练目标，记为
`R_sum`。实际采样得到的边缘分布是高斯混合分布，不能将它的 KL
与这个求和写成恒等式。下面给出它们的准确关系。

## Step 1：后验参数上采样

保留 NVAE 编码器，输出原始参数
`[B, 2C, H, W] = concat(mean, log_sigma)`。
均值和 log_sigma 各自经过零插入及独立的
`ConvTranspose2d(C,C,3,stride=1,padding=1)`，得到两个
`[B,C,2H,2W]` 参数张量。上采样发生在参数上，而不是高斯样本上。

高斯分布采用仓库 `distributions.Normal` 的相同定义：

\[
\mu=5\tanh(\mu_{raw}/5),\qquad
\sigma=\exp\{5\tanh(\ell_{raw}/5)\}+0.01.
\]

因此变换的是原始均值与 log_sigma 参数，不是方差本身。

## Step 2：先验参数与额外的 logit channel

先验有一个与输入图像无关的可学习粗参数张量
`[1,2C,H,W]`，初始化为零。它通过独立于后验的同构上采样模块。
另外增加一个
`ConvTranspose2d(2C,1,3,stride=1,padding=1)`，对零插入后的
粗先验参数输出 `[1,1,2H,2W]` 的位置 logits。

高斯需要均值和尺度两个参数，因此实际保存的是 **C 个均值 channel、
C 个 log_sigma channel、1 个位置 logit channel**，总计 `2C+1`。
从每个高斯参数场的角度看，它是 `C` 个参数 channel 加一个共享的选择场；
并不把 `C+1` 个标量误当成 C 个一般高斯所需的全部参数。
额外的选择场只有一个，不给均值和尺度分别抽取不同的位置。

从左上角开始，把每个非重叠 2×2 块排成
`[00,01,10,11]`，在最后的四位置轴上 softmax：

\[
\pi_{b,k}=\frac{\exp(a_{b,k})}{\sum_{j=0}^{3}\exp(a_{b,j})}.
\]

这里 `b=(h,w)` 是空间块的索引，`k` 是四个位置之一；
`π` 就是需求中的四个归一化选择概率。它只来自先验模块，不看 x。
单个 logit channel 对应**每个空间块选择一次，C 个 channel 共享位置**。
不同空间块独立选择。后验和先验共享同一组 `π`。

零初始化时粗先验的输出均值为零、尺度遵循上述 Normal 约定，
选择概率是均匀的。粗先验参数及其上采样网络随后一起学习。
这一点与原单层 NVAE 固定标准先验不同，是新增先验模块的定义。

## Step 3：选择分布参数，然后采样

对每一个空间块，先采样

\[
K_b\sim\operatorname{Categorical}(\pi_b).
\]

再取出该位置 C 个 channel 的均值、尺度，采样

\[
z_{b,c}=\mu^q_{b,c,K_b}+\sigma^q_{b,c,K_b}\epsilon_{b,c},
\qquad \epsilon_{b,c}\sim\mathcal N(0,1).
\]

输出恢复为 `[B,C,H,W]`，交给原 `stem_decoder`、后处理以及原图像条件分布。
选择的是参数，随后才执行高斯采样；不会对四个高斯样本做软加权平均。
训练重建项就是用这个 z 算出的原 NVAE reconstruction loss，MNIST 仍裁剪
到 28×28 并使用 Bernoulli 图像条件分布。

无条件生成执行同样的“按先验 π 选择位置 → 从该位置的先验高斯采样”，
将得到的 `[B,C,H,W]` 样本交给相同解码器。因此生成使用的是新先验，
不会退回原 `_fixed_prior`。本实现也在先验生成时做块选择，以保持
“后验使用的 z 与生成使用的 z 属于同一空间”以及原解码器输入尺寸。
`sample(n,t)` 的 t 仅缩放选中高斯的标准差，保持原 NVAE 高斯温度约定；
不改变位置 softmax。

## 分布、KL 和评估的准确含义

记一个位置上的 C 维对角高斯为

\[
q_{b,k}(z_b\mid x)=\prod_c\mathcal N(z_{b,c};\mu^q_{b,c,k},(\sigma^q_{b,c,k})^2),
\]

并类似定义 `p_{b,k}`。传入解码器的变量分布为

\[
\widehat q(z\mid x)=\prod_b\sum_{k=0}^3\pi_{b,k}q_{b,k}(z_b\mid x),
\qquad
\widehat p(z)=\prod_b\sum_{k=0}^3\pi_{b,k}p_{b,k}(z_b).
\]

需求中的四项 KL 求和在代码中实现为

\[
R_{sum}=\sum_b\sum_{k=0}^3D_{KL}(q_{b,k}\Vert p_{b,k}).
\]

如果将所有块的同一偏移位置的高斯组合成 `q_00`、`q_01`、`q_10`、`q_11`
四个乘积分布，上式正好等于这四个乘积分布 KL 的求和。
单维高斯 KL 继续使用仓库的解析公式，先对四位置求和，再按原接口汇总 channel、空间。
此处没有乘选择概率，也没有除以 4。

由于两侧选择概率相同，对扩充变量 `(K,z)` 有

\[
D_{KL}(\widehat q(z\mid x)\Vert\widehat p(z))
\le D_{KL}(q(K,z\mid x)\Vert p(K,z))
=\sum_b\sum_k\pi_{b,k}D_{KL}(q_{b,k}\Vert p_{b,k})
\le R_{sum}.
\]

第一个不等式来自 KL 对边缘化的数据处理不等式，最后一个不等式来自
`0 ≤ π ≤ 1` 以及分量 KL 非负。
所以 β=1 时，期望重建损失加 `R_sum` 仍给出负对数似然的一个上界，
但通常比实际混合分布 ELBO 更松。训练仍沿用原 KL 退火；β<1 阶段
不能声称这一退火目标也满足同一上界。日志沿用 `NELBO` 名称，
本模式下应理解为**使用四项 KL 正则的较松上界目标**。

验证时，`log_q` 与 `log_p` 使用真正的边缘混合密度：

\[
\log\widehat q(z\mid x)=\sum_b\operatorname{logsumexp}_k
\left(\log\pi_{b,k}+\sum_c\log q_{b,c,k}(z_{b,c}\mid x)\right).
\]

因为一个块的 C 个 channel 共享位置，channel 求和必须在 `logsumexp`
里面。原 NVAE 的 IW 权重仍为
`log p(x|z) + log p_hat(z) - log q_hat(z|x)`，不是选中单个分量的密度，
也不是将 `R_sum` 塞入重要性权重。

## 离散选择的梯度

位置使用真实 categorical 样本；高斯使用重参数化。
原 NVAE 并无这次 categorical 抽样，因此仅依赖 gather 的反传会漏掉
位置概率的重建梯度。本模式在 `train.py` 中补充无偏 REINFORCE 项：

\[
(\operatorname{stopgrad}(L_{rec})-b)
\sum_b\nabla\log\pi_{b,K_b}.
\]

baseline 是过去 minibatch 重建损失的 EMA（默认 decay=0.9），
初值为 0，在计算本批梯度项之后更新，避免用本批均值引入偏差。
EMA 状态保存进 checkpoint。不同 GPU 可各自维护过去数据的 baseline，
仍沿用仓库手工平均参数梯度的方式。
代码使用 `score-score.detach()`，前向数值为零，因此不会污染目标日志。
`R_sum` 不依赖抽中的 K，解析梯度直接反传，不加入这个随机 reward。
没有使用 Gumbel-Softmax、ST 或把 π detach。该估计量可能有较高方差。

## 保留的 NVAE 流程

数据处理、图像分布、重建损失、KL 退火/平衡、优化器、学习率调度、
AMP、梯度同步、原卷积与 BN 正则、checkpoint、IW 评估和 BN readjustment
都继续走原入口。新增转置卷积使用普通可学习参数和已有优化器的
weight decay；原 spectral-norm collector 只处理原 Conv2D/ARConv2d，
不对新增转置卷积另加 spectral 正则。

原单层路径不使用层级 `prior_ftr0`，新模式将其冻结，避免原梯度同步器
对这个未参与损失的张量访问空梯度。关闭开关时不改变原行为。

## MNIST 运行示例

激活原 NVAE 环境，在仓库根目录运行：

```bash
bash scripts/train_mnist_single_gaussian_block.sh
```

脚本使用 1 scale、1 group、20 latent channels、32 encoder/decoder channels、
2 个预/后处理 blocks、每 block 3 cells、400 epochs。
MNIST 在 NVAE 内补边为 32×32，因此 `H=W=8`，参数上采样至 16×16，
选取后回到 `20×8×8`，每张图传入解码器的连续标量总数为 1280。
这是训练配置示例，不是已训练 checkpoint，也不承诺具体 NLL/FID。

可覆盖环境路径和设备，末尾参数覆盖脚本中的同名 argparse 参数：

```bash
CUDA_VISIBLE_DEVICES=0 DATA_DIR=/root/autodl-tmp/datasets/mnist \
CHECKPOINT_ROOT=/root/autodl-tmp/nvae-checkpoints \
bash scripts/train_mnist_single_gaussian_block.sh --batch_size 64 --epochs 200
```

默认 checkpoint 路径：

```text
/root/autodl-tmp/nvae-checkpoints/eval-mnist_single_gaussian_block/checkpoint.pt
```

评估时模型开关从 checkpoint args 恢复，不需要另加新参数：

```bash
python evaluate.py \
  --checkpoint /root/autodl-tmp/nvae-checkpoints/eval-mnist_single_gaussian_block/checkpoint.pt \
  --save /root/autodl-tmp/nvae-evaluation/single_gaussian_block \
  --data /root/autodl-tmp/datasets/mnist \
  --eval_mode evaluate --num_iw_samples 1000 --batch_size 64
```

无条件生成（包含 BN 重新校准）：

```bash
python evaluate.py \
  --checkpoint /root/autodl-tmp/nvae-checkpoints/eval-mnist_single_gaussian_block/checkpoint.pt \
  --save /root/autodl-tmp/nvae-evaluation/single_gaussian_block_samples \
  --eval_mode sample --temp 1.0 --readjust_bn
```

## 验证与文件

```bash
python -m pytest -q tests/test_learnable_prior.py tests/test_gaussian_block.py tests/test_mixed_distributions.py tests/test_mixed_model.py
bash -n scripts/train_mnist_single_gaussian_block.sh
bash -n scripts/train_mnist_single_gaussian_learnable_prior.sh
```

新测试覆盖块顺序、上采样形状、共享位置及非均匀抽样频率、真实混合密度、
四项解析 KL、精确枚举验证 REINFORCE 梯度、训练/生成实际传入解码器的样本、
所有可学习参数的有限梯度、checkpoint 和旧开关兼容性。
CPU 测试使用仓库 `res_elu` 小模型并关闭依赖分布式初始化的 data init；
不能替代 GPU AMP、多 GPU 或完整数据集训练验证。

- `gaussian_block.py`：上采样、块分布及位置梯度。
- `model.py`：接入单层 forward/sample 路径。
- `train.py`：新开关、位置梯度与模式日志。
- `scripts/train_mnist_single_gaussian_block.sh`：训练入口。
- `tests/test_gaussian_block.py`：数学与模型接入验证。
