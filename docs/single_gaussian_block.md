# 单层 Gaussian VAE：输入相关的位置后验

## 开关与范围

```text
--latent_distribution normal --gaussian_block_upscale
--num_latent_scales 1 --num_groups_per_scale 1 --num_nf 0
```

这是一个随机潜变量层；每个 2×2 块有四个混合分量。
关闭开关时不创建新参数，原 Gaussian 模型的配置与 state_dict 仍兼容。
本版本增加后验 selector，并将转置卷积改为普通卷积；原先仅用先验位置
概率的 block checkpoint 不能直接继续训练，须使用新版本重新训练。

## 参数上采样：补零，再接普通卷积

编码器输出 `[B,2C,H,W] = concat(mean_raw, log_sigma_raw)`。
对两个参数场分别插入零：

\[
U(A)_{2h,2w}=A_{h,w},\qquad
U(A)_{i,j}=0\quad\text{当 }i\text{ 或 }j\text{ 为奇数}.
\]

补零后的大小为 `2H × 2W`，包括最后一行、最后一列的零。
随后通过随机初始化的普通 `nn.Conv2d`，所有头均使用
`kernel_size=3, stride=1, padding=1`，因此卷积保持补零后的空间尺寸。
两倍尺寸由零插入实现，卷积不负责扩大尺寸。

后验有三个输出头：

- `mean`：`Conv2d(C,C,3,stride=1,padding=1)`，处理均值参数场。
- `log_sigma`：同样的独立卷积，处理 log_sigma 参数场。
- `selector`：`Conv2d(2C,1,3,stride=1,padding=1)`，处理两个补零参数场的拼接。

输出为两个 `[B,C,2H,2W]` 高斯参数场和一个 `[B,1,2H,2W]`
未归一化位置 logit 场。从每个高斯参数场看是 C 个参数 channel 加
一个共享的位置 channel，即需求中的 C+1。高斯需要均值和尺度两个
参数场，因此总计保存 `2C+1` 个 channel，只有一个共享的 selector。
上采样操作发生在参数上，而不是在已经抽取的样本上。

高斯使用原 NVAE 的参数约定：

\[
\mu=5\tanh(\mu_{raw}/5),\qquad
\sigma=\exp\{5\tanh(\ell_{raw}/5)\}+0.01.
\]

## 每个块的位置概率依赖输入

从左上角开始，把位置 logit 场分成不重叠的 2×2 块，按
`[00,01,10,11]` 排序，在每个块的四位置轴上 softmax：

\[
q_{b,k}(x)=\frac{\exp a^q_{b,k}(x)}{\sum_{j=0}^3\exp a^q_{b,j}(x)},
\qquad k\in\{00,01,10,11\}.
\]

`b=(h,w)` 是空间块索引。不同输入的编码器参数不同，因此可以得到
不同的位置概率。一个块只抽取一个 categorical 位置，C 个 channel
共享该位置；不同块独立抽取。概率不会在整个 `2H × 2W` 图上统一归一化。

先验保留独立可学习的粗参数 `[1,2C,H,W]`（初始化为零），以及独立
的同构上采样网络。先验位置概率记作 `p_{b,k}`，只由先验参数决定，
不读取输入图像，也不复用后验 logits。所有新卷积保留 PyTorch 默认随机
初始化；零粗先验下 selector 初始输出为常数，块内概率仍为 1/4。

## 重建与无条件先验生成

记第 b 块、第 k 位置的 C 维对角高斯分量为

\[
g^q_{b,k}(z_b\mid x)=\prod_c\mathcal N(z_{b,c};\mu^q_{b,c,k}(x),
(\sigma^q_{b,c,k}(x))^2),
\]

并类似定义 `g^p`。实际交给解码器的分布为

\[
\widehat q(z\mid x)=\prod_b\sum_k q_{b,k}(x)g^q_{b,k}(z_b\mid x),
\qquad
\widehat p(z)=\prod_b\sum_k p_{b,k}g^p_{b,k}(z_b).
\]

重建时先抽取 `K_b ~ Categorical(q_b(x))`，取出该位置所有 C 个
channel 的均值和尺度，再用高斯重参数化抽样。得到 `[B,C,H,W]` 的 z，
送入原 `stem_decoder`、后处理和图像条件分布。没有把四个样本软加权平均。
MNIST 的 reconstruction loss 仍为裁剪到 28×28 的 Bernoulli 负对数似然。

无条件生成时，`AutoEncoder.sample(n,t)` 从上述 `p_hat(z)` 抽样：
按先验概率选择位置，再从选中先验高斯抽样，得到 `[n,C,H,W]`。
这个样本直接替代原单层 Gaussian 样本进入相同解码器。
该路径不访问编码器、输入图像或后验上采样网络。
`t` 仅缩放高斯标准差，不改变位置概率；`t=1` 对应原先验。

## KL 正则与评估

保留原分支指定的四项**不加权** Gaussian KL 求和：

\[
R_{sum}(x)=\sum_b\sum_k D_{KL}(g^q_{b,k}(\cdot\mid x)\Vert g^p_{b,k}).
\]

后验、先验不再共用位置概率，因此补充解析 categorical KL：

\[
R_{pos}(x)=\sum_b\sum_k q_{b,k}(x)\log\frac{q_{b,k}(x)}{p_{b,k}},
\qquad R(x)=R_{sum}(x)+R_{pos}(x).
\]

位置变量由 C 个 channel 共享，`R_pos` 在每个块只计算一次。
代码为保持 `[B,C,H,W]` 接口，将该项平均分配到 C 个 channel，
后续 channel 求和恢复恰好一份 categorical KL。

训练目标为

\[
L=L_{rec}+\beta(t)(R_{sum}+R_{pos})+\text{原 NVAE 正则项}.
\]

两个 KL 项均解析求导。由于分量 KL 非负，

\[
D_{KL}(\widehat q\Vert\widehat p)
\le D_{KL}(q(K,z\mid x)\Vert p(K,z))
=R_{pos}+\sum_b\sum_k q_{b,k}D_{KL}(g^q_{b,k}\Vert g^p_{b,k})
\le R_{pos}+R_{sum}.
\]

因此这个正则仍为边缘混合 KL 的上界，并非其闭式等式。
日志中的 `NELBO` 应理解为使用该较松正则的目标；不把 `R_sum` 单独
当作不同位置概率下的完整 KL。β=1 时期望重建项加 R 给出 NLL 上界；
β<1 的退火阶段不作同一上界保证。

IW 评估仍使用真实边缘混合密度，各自使用自己的位置概率：

\[
\log\widehat q(z\mid x)=\sum_b\operatorname{logsumexp}_k
\left(\log q_{b,k}(x)+\sum_c\log g^q_{b,c,k}(z_{b,c}\mid x)\right).
\]

先验对应使用 `log p_{b,k}` 与先验高斯分量。channel 求和在
`logsumexp` 内，重要性权重为
`log p(x|z) + log p_hat(z) - log q_hat(z|x)`。
不把正则上界 R 代入重要性权重。

## 位置选择的梯度

使用真实 categorical 抽样和 Gaussian 重参数化。重建项的位置梯度为

\[
(\operatorname{stopgrad}(L_{rec})-b)
\sum_b\nabla\log q_{b,K_b}(x).
\]

REINFORCE 训练后验 selector 并通过参数场反传到编码器。
先验 selector 由解析 categorical KL 训练，不接收后验选择的重建 reward。
两个 KL 项不依赖抽中的 K，解析梯度直接反传，不加入随机 reward。

baseline 为过去 minibatch 重建损失的 EMA，默认 decay=0.9，初值为零，
本批使用旧值后再更新。`score-score.detach()` 前向为零，不污染目标日志。
EMA 保存进 checkpoint。没有使用 ST 或 Gumbel-Softmax。

## 原 NVAE 流程和运行命令

数据、图像条件分布、KL 退火/平衡、优化器、AMP、梯度同步、原卷积和
BN 正则、checkpoint、IW 评估及 BN readjustment 保持原入口。
新普通卷积参与优化器 weight decay；原 spectral-norm collector
只处理原 `Conv2D/ARConv2d`。单层路径不使用的 `prior_ftr0` 继续冻结。

```bash
CUDA_VISIBLE_DEVICES=0 DATA_DIR=/root/autodl-tmp/datasets/mnist \
CHECKPOINT_ROOT=/root/autodl-tmp/nvae-checkpoints \
bash scripts/train_mnist_single_gaussian_block.sh --save mnist_single_gaussian_block_input_positions
```

脚本默认 1 scale、1 group、20 latent channels、32 encoder/decoder channels、
2 个预/后处理 blocks、每 block 3 cells、400 epochs。
MNIST 在 NVAE 内补边为 32×32，因此 H=W=8，参数上采样至 16×16，
选择后回到 `20×8×8`。这是训练配置示例，不是已经验证的训练结果。

```bash
python evaluate.py \
  --checkpoint /root/autodl-tmp/nvae-checkpoints/eval-mnist_single_gaussian_block_input_positions/checkpoint.pt \
  --save /root/autodl-tmp/nvae-evaluation/single_gaussian_block_input_positions \
  --data /root/autodl-tmp/datasets/mnist \
  --eval_mode evaluate --num_iw_samples 1000 --batch_size 64

python evaluate.py \
  --checkpoint /root/autodl-tmp/nvae-checkpoints/eval-mnist_single_gaussian_block_input_positions/checkpoint.pt \
  --save /root/autodl-tmp/nvae-evaluation/single_gaussian_block_input_positions_samples \
  --eval_mode sample --temp 1.0 --readjust_bn
```

## 验证与文件

```bash
python -m pytest -q tests/test_gaussian_block.py tests/test_mixed_distributions.py tests/test_mixed_model.py
bash -n scripts/train_mnist_single_gaussian_block.sh
```

测试覆盖零插入与普通卷积、块顺序、输入相关的后验位置概率、输入无关的先验、
共享位置采样频率、边缘混合密度、独立概率下的解析正则及 categorical KL 计数、
两个 selector 的梯度、精确枚举的 REINFORCE 梯度、真实解码器输入、
训练/生成、checkpoint 和关闭开关的兼容性。CPU 小模型测试使用 `res_elu`，
关闭依赖分布式初始化的 data init，不能替代 GPU AMP、多 GPU 或完整训练。

- `gaussian_block.py`：上采样、分布、解析正则与位置梯度。
- `model.py`：既有单层 forward/sample 接口，无需额外修改。
- `train.py`：训练入口、帮助信息与模式日志。
- `scripts/train_mnist_single_gaussian_block.sh`：既有训练脚本。
- `tests/test_gaussian_block.py`：数学与接入验证。
