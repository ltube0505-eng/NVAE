# Scale 之间仅通过末组样本连接的 NVAE

## 开关和约定

训练时增加 `--scale_connection last_sample` 启用本架构。默认值
`feature` 保留原来的 NVAE 特征图连接；旧 checkpoint 的 args 缺少该字段时也使用
`feature`。本实现基于本仓库 master，保留已有 Normal、Poisson 和混合
Poisson/Gamma 分布及其采样、KL 和梯度估计实现。

本文按**生成顺序**编号 scale：s=1 是分辨率最小的 scale，s=S 是最接近图像输出的
scale。第 s 个 scale 有 G_s 组样本，记为 z_(s,1), ..., z_(s,G_s)。代码中的
`groups_per_scale` 按 encoder 顺序保存，所以生成顺序使用它的逆序。

明确采用以下两条约定：

1. 每个 scale 的初始特征图是独立的、创建模型时随机初始化一次的可训练参数，
   不是每次 forward 重新采样的噪声。它们不依赖 x，也不产生额外的随机变量或 KL。
2. 上一 scale 的末组样本只参与下一 scale **首组的先验和后验计算**。
   采出该首组后，从下一 scale 自己的初始特征图开始组内传播，不携带桥接条件特征。
   因此图像似然只直接依赖最后一个 scale 的全部隐变量组。

这里切断的是生成/自顶向下推断路径上的跨 scale 特征连接。自底向上的 encoder
仍按原仓库提取并缓存各组的 x 特征，供 `EncCombinerCell` 使用；x 特征不会进入先验
或图像输出的确定性路径。

## 文字算法

### 1. 创建初始特征图及跨尺度模块

第一个 scale 沿用 `prior_ftr0`，其余每个 scale 在 `ScaleBoundaryCell` 内创建
`initial_feature`。各自从 U(0,1) 初始化，作为 `nn.Parameter` 训练并保存到 checkpoint。
前向时只在 batch 维度广播，不重新随机初始化。

若第一尺度的空间尺寸、通道数为 H_1、C_1，则第 s 个尺度为
H_s = 2^(s-1) H_1，C_s = C_1 / 2^(s-1)。初始参数形状是
[C_s, H_s, H_s]。这与原来的每次上采样扩大空间、减少通道的配置对应。

在原 `up_dec` 的位置放置 `ScaleBoundaryCell`。这个模块**只接收上一 scale
的最后一个采样张量**，不接收上一 scale 的 decoder 特征图。将该样本最近邻上采样到
[B, num_latent_per_group, H_s, H_s]，与本 scale 的初始特征图按通道拼接，再通过
`DecCombinerCell` 的 1x1 卷积得到桥接条件特征。这里只做确定性尺寸适配，没有再次采样，
也没有使用样本均值或分布参数代替真实样本。

### 2. 自底向上提取 x 特征

沿用原有 `stem`、`pre_process`、`enc_tower`。缓存每个 encoder combiner 对应的
x 特征，随后反转缓存顺序，与生成方向的隐变量组对齐。

### 3. 第一个 scale

第一组后验由最深的 encoder 特征经 `enc0` 和 `enc_sampler[0]` 得到；第一组先验
沿用现有固定先验。采样以后，将样本与 `prior_ftr0` 通过原 `DecCombinerCell` 合并。

该 scale 内的其余组保持原有操作顺序：对当前 decoder 特征执行 `normal_dec`；
由 `dec_sampler` 得到先验参数；由 `EncCombinerCell(x_feature, decoder_feature)`
和 `enc_sampler` 得到后验参数；按既有 `res_dist` 选项构造后验；采样并计算 log q、
log p；最后把样本注入当前 decoder 特征。

### 4. 到达下一个 scale

保存刚刚采到的上一 scale 末组样本 b_s = z_(s-1,G_(s-1))。
如果启用了 Normal posterior flow，这里传递的是 **flow 变换后的样本**。
舍弃旧 decoder 特征图，调用本 scale 的边界模块，得到两个张量：

- 本 scale 独立的初始特征 a_s；
- 由 a_s 和上采样后的 b_s 合并得到的桥接条件特征。

桥接条件特征经过原有首组前的 `normal_dec` 层，得到 c_s。
`dec_sampler` 仅以 c_s 为输入构造首组先验。
`EncCombinerCell` 将 c_s 与该组缓存的 x 特征合并，再由 `enc_sampler` 构造首组后验。
因此先验不看 x，后验同时利用 x 与上一 scale 的末组样本。`res_dist` 打开时，
后验继续使用仓库已有的先验参数加残差参数方式。

从首组后验采出 z_(s,1)，完成可能的 flow 和密度计算以后，**丢弃 c_s**。
令当前组内状态为 `DecCombinerCell(a_s, z_(s,1))`，而不是
`DecCombinerCell(c_s, z_(s,1))`。这是保证最终似然条件限制的关键步骤。

### 5. 当前 scale 的后续 group

与步骤 3 的组内传播相同，保留原来的特征图递推。
第 g>1 组的先验由该 scale 前面组的状态产生；后验再额外结合该组的 x 特征。
上一 scale 的 b_s 不再作为额外特征拼接或残差送入这些组。

到该 scale 的最后一组时，继续把它的样本传给下一 scale。
跨边界不使用 `detach()`：Gaussian 的重参数化梯度及已有 Poisson 路径梯度仍可经过
末组样本传播。REINFORCE/O-BBVI 继续采用原有 score-function 目标。

### 6. 输出图像和无条件生成

最后一个 scale 完成后，沿用 `post_process` 和 `image_conditional` 输出似然参数。
它们只接收最后一个 scale 的组内状态。对于固定的最后一尺度样本，改变更早 scale
的样本不会改变图像似然参数。

无条件生成采用同样的边界和重置步骤，但每组都从先验采样，不使用 encoder 或 x 特征。
`forward()` 和 `sample()` 的 prior 与 image-conditional 路径保持一致。

## 概率分解与训练目标

记 Z_s 为第 s 个 scale 的全部隐变量组。对 s>1，

```text
p(Z_s | b_s) = p(z_(s,1) | b_s) product_(g=2..G_s) p(z_(s,g) | z_(s,<g))
q(Z_s | x,b_s) = q(z_(s,1) | x,b_s) product_(g=2..G_s) q(z_(s,g) | x,z_(s,<g))
```

这里后续组仍会通过本 scale 已采出的隐变量间接受到 b_s 的影响，但不存在额外的
b_s 确定性特征旁路。第一 scale 使用相同的组内分解，只是首组先验不带 b_1。
联合分布为：

```text
p(x,Z_1,...,Z_S) = p(x | Z_S) p(Z_1) product_(s=2..S) p(Z_s | b_s)
```

最小化的负 ELBO 为：

```text
J(x) = E_q[-log p(x | Z_S)] + sum_(s=1..S) sum_(g=1..G_s) K_(s,g)(x)
```

每个 K 都是相应条件后验与先验之间的 KL，并对其随机父变量的后验取期望。
首组的父变量是上一尺度末组样本（第一尺度首组没有父变量）；其余组的父变量是
本尺度前面的组。**所有 scale 的 KL 都保留**，`log_q`、`log_p` 也累加所有组，
以保持 ELBO、重要性采样评估及 score-function 训练的一致性。

Normal 未使用 flow 时、Poisson、Gamma 继续使用仓库已有的解析条件 KL；
使用 flow 时继续使用 log q - log p 估计。训练中的 KL annealing、KL balancing、
谱正则和 BN 正则沿用原实现，它们应与单位 KL 权重的理论负 ELBO 区分。
Poisson 的 relaxed/ST 仍是已有近似训练估计器，本架构不改变其估计偏差或截断设置。

条件独立并不意味着前面 scale 不参与生成：它们仍通过末组样本影响下一 scale
首组的分布。也不能据此断言它们一定会发生后验坍缩。

## 使用示例

从本分支启动新的两尺度 Gaussian MNIST 实验，例如：

```bash
python train.py \
  --root /root/autodl-tmp/nvae-checkpoints \
  --save mnist_s2_last_sample_bridge \
  --dataset mnist \
  --data /root/autodl-tmp/datasets/mnist \
  --num_latent_scales 2 \
  --num_groups_per_scale 8 \
  --num_latent_per_group 20 \
  --num_channels_enc 32 \
  --num_channels_dec 32 \
  --latent_distribution normal \
  --num_nf 0 \
  --scale_connection last_sample \
  --res_dist \
  --weight_decay_norm 1e-2 \
  --batch_size 200 \
  --epochs 200
```

这里的 batch size 和训练超参数是示例，需要按显存和实验设计选择，不代表已验证的
收敛配置。比较架构时建议保持其他超参数相同，以 `--scale_connection feature`
作为原架构对照。

`train.py` 会把开关写入 checkpoint 的 args，`evaluate.py` 读取 args 后自动恢复架构，
无需新增评估参数。新架构有不同的边界模块和参数，不能直接将旧 feature 架构 checkpoint
当成新架构完整恢复；应从新的实验目录开始训练。继续训练新架构 checkpoint 时，要保留
`--scale_connection last_sample` 和相同的模型配置。

只有一个 scale 时没有边界，该开关不改变模型。多个 scale 但每个仅一个 group
也是受支持的配置。由于旧尺度末组之后的 decoder 特征被丢弃，部分仅服务于该特征的
参数可能没有梯度；`average_gradients` 会跳过 `grad is None` 的参数，避免同步时报错。

## 验证与限制

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

本次验证：Python 3.12、PyTorch 2.14.1+cpu，35 项测试通过。测试覆盖：

- 改变非末组样本、固定末组样本，不改变下一尺度的首组先验；
- 改变末组样本会改变下一尺度首组先验；固定下一尺度样本以后，后续先验和输出不变；
- 原 feature 架构的干预对照，确认测试能够检测特征旁路；
- forward 与 prior sampling 在相同隐变量轨迹下给出相同先验参数和输出；
- 两/三尺度、不同分组数、自适应分组、单组尺度、pre/postprocess、残差后验开关；
- Gaussian、Poisson、混合分布、Gaussian flow、REINFORCE/O-BBVI；
- 重建项对前面 scale 末组样本的非零梯度，以及每个边界参数的有限梯度；
- 全部组的 KL、独立初始参数、state_dict 恢复和旧 args 的兼容性；
- 谱正则、一次 Adamax 更新和无梯度参数的梯度同步逻辑。

CPU 测试只对旧代码中 AR mask 和谱正则的硬编码 `.cuda()` 分配临时替换设备位置，
实际模型计算和反向传播均执行；同步测试使用模拟 all_reduce。
未运行真实 GPU、多 GPU、原 PyTorch 1.6 环境或完整数据集训练，未报告收敛、FID 或 NLL 改善。
本改动没有更新原有依赖版本，也没有改造旧 CUDA/BN 实现。
