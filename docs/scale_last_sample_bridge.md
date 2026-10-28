# 隐变量尺度间以末组样本连接、图像 decoder 使用全部隐变量的 NVAE

## 开关和约定

本分支新训练默认使用 `--scale_connection last_sample`、全 Poisson 隐变量和 rate-STE。
显式指定 `--scale_connection feature` 可选择原来的 NVAE 特征图连接；旧 checkpoint 的
args 缺少该字段时仍使用 `feature`。Normal 和混合 Poisson/Gamma 仍可显式选择。
Poisson STE 的前向已改为精确整数采样，反向使用 rate 的恒等代理梯度，详见
[全 Poisson + STE 分支说明](scale_last_sample_poisson_ste.md)。

本文按**生成顺序**编号 scale：s=1 是分辨率最小的 scale，s=S 是最接近图像输出的
scale。第 s 个 scale 有 G_s 组样本，记为 z_(s,1), ..., z_(s,G_s)。代码中的
`groups_per_scale` 按 encoder 顺序保存，所以生成顺序使用它的逆序。

明确采用以下三条约定：

1. 每个 scale 的初始特征图是独立的、创建模型时随机初始化一次的可训练参数，
   不是每次 forward 重新采样的噪声。它们不依赖 x，也不产生额外的随机变量或 KL。
2. 上一 scale 的末组样本只参与下一 scale **首组的先验和后验计算**。
   采出该首组后，从下一 scale 自己的初始特征图开始组内传播，不携带桥接条件特征。
   这条限制只针对隐变量条件网络。
3. **图像 decoder 使用所有 scale 的所有 group 样本**。它拥有独立参数和持续传播的
   特征图，逐组融合样本、跨 scale 上采样，最后输出似然参数。它的特征不会反馈到
   先验或后验网络。

这里切断的是隐变量先验/自顶向下推断路径上的跨 scale 特征连接。图像 decoder 的
跨 scale 特征连接保留。自底向上的 encoder
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

在隐变量条件网络原 `up_dec` 的位置放置 `ScaleBoundaryCell`。这个模块**只接收上一 scale
的最后一个采样张量**，不接收上一 scale 的条件特征图。将该样本最近邻上采样到
[B, num_latent_per_group, H_s, H_s]，与本 scale 的初始特征图按通道拼接，再通过
`DecCombinerCell` 的 1x1 卷积得到桥接条件特征。这里只做确定性尺寸适配，没有再次采样，
也没有使用样本均值或分布参数代替真实样本。

### 2. 自底向上提取 x 特征

沿用原有 `stem`、`pre_process`、`enc_tower`。缓存每个 encoder combiner 对应的
x 特征，随后反转缓存顺序，与生成方向的隐变量组对齐。

### 3. 第一个 scale

第一组后验由最深的 encoder 特征经 `enc0` 和 `enc_sampler[0]` 得到；第一组先验
沿用现有固定先验。采样以后，将样本与 `prior_ftr0` 通过原 `DecCombinerCell` 合并。

该 scale 内的其余组保持原有操作顺序：对当前条件特征执行 `normal_dec`；
由 `dec_sampler` 得到先验参数；由 `EncCombinerCell(x_feature, condition_feature)`
和 `enc_sampler` 得到后验参数；按既有 `res_dist` 选项构造后验；采样并计算 log q、
log p；最后把样本注入当前条件特征。同时按生成顺序收集每一组的实际样本，供图像 decoder 使用。

### 4. 到达下一个 scale

保存刚刚采到的上一 scale 末组样本 b_s = z_(s-1,G_(s-1))。
如果启用了 Normal posterior flow，这里传递的是 **flow 变换后的样本**。
舍弃旧隐变量条件特征图，调用本 scale 的边界模块，得到两个张量：

- 本 scale 独立的初始特征 a_s；
- 由 a_s 和上采样后的 b_s 合并得到的桥接条件特征。

桥接条件特征经过原有首组前的 `normal_dec` 层，得到 c_s。
`dec_sampler` 仅以 c_s 为输入构造首组先验。
`EncCombinerCell` 将 c_s 与该组缓存的 x 特征合并，再由 `enc_sampler` 构造首组后验。
因此先验不看 x，后验同时利用 x 与上一 scale 的末组样本。`res_dist` 打开时，
后验继续使用仓库已有的先验参数加残差参数方式。

从首组后验采出 z_(s,1)，完成可能的 flow 和密度计算以后，**丢弃 c_s**。
令当前组内状态为 `DecCombinerCell(a_s, z_(s,1))`，而不是
`DecCombinerCell(c_s, z_(s,1))`。这保证上一 scale 的条件特征不会直接传到当前 scale
后续组的先验/后验。该重置不作用于独立的图像 decoder。

### 5. 当前 scale 的后续 group

与步骤 3 的组内传播相同，保留原来的特征图递推。
第 g>1 组的先验由该 scale 前面组的状态产生；后验再额外结合该组的 x 特征。
上一 scale 的 b_s 不再作为额外特征拼接或残差送入这些组。

到该 scale 的最后一组时，继续把它的样本传给下一 scale。
跨边界不使用 `detach()`：Gaussian 的重参数化梯度及已有 Poisson 路径梯度仍可经过
末组样本传播。REINFORCE/O-BBVI 继续采用原有 score-function 目标。

### 6. 用所有 scale 的所有隐变量输出图像

另建独立的 `image_dec_tower`，采用原 NVAE 的 feature-connected decoder 架构。
它从自己的可训练参数 `image_ftr0` 开始，按低分辨率到高分辨率的生成顺序，依次将
**每个 scale 的每一个 group** 的实际采样值送入对应的 `DecCombinerCell`。
组间继续执行原 `normal_dec`，scale 间继续执行原 `up_dec`，特征图不会在尺度边界重置。

所有组融合完成后，沿用 `post_process` 和 `image_conditional` 输出似然参数。
图像路径接收的样本与计算 log q/log p 的样本相同；有 flow 时使用 flow 后的值。
不重新采样，不替换为均值，也不 detach。图像路径不接收 x 特征、后验参数或隐变量
条件网络的特征图，其输出不用于计算任何 group 的先验/后验参数。

因此固定最后一尺度样本、改变更早尺度的任一组样本，图像输出仍可以变化。
每组样本都有直接通往图像似然的计算路径，而不必只通过后续尺度样本间接影响图像。
这不保证训练结束后模型一定有效利用每一组，仍需另行检查后验坍缩。

### 7. 无条件生成

无条件生成采用同样的边界和重置步骤，但每组都从先验采样，不使用 encoder 或 x 特征。
收集所有先验样本后，送入同一个 `image_dec_tower`。`forward()` 和 `sample()` 的
prior 与 image-conditional 路径保持一致。

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
p(x,Z_1,...,Z_S) = p(x | Z_1,...,Z_S) p(Z_1) product_(s=2..S) p(Z_s | b_s)
```

最小化的负 ELBO 为：

```text
J(x) = E_q[-log p(x | Z_1,...,Z_S)] + sum_(s=1..S) sum_(g=1..G_s) K_(s,g)(x)
```

每个 K 都是相应条件后验与先验之间的 KL，并对其随机父变量的后验取期望。
首组的父变量是上一尺度末组样本（第一尺度首组没有父变量）；其余组的父变量是
本尺度前面的组。**所有 scale 的 KL 都保留**，`log_q`、`log_p` 也累加所有组，
以保持 ELBO、重要性采样评估及 score-function 训练的一致性。

Normal 未使用 flow 时、Poisson、Gamma 继续使用仓库已有的解析条件 KL；
使用 flow 时继续使用 log q - log p 估计。训练中的 KL annealing、KL balancing、
谱正则和 BN 正则沿用原实现，它们应与单位 KL 权重的理论负 ELBO 区分。
Poisson 的 relaxed/STE 都是有偏训练估计器；本分支 STE 使用精确 Poisson 前向，
只有 relaxed 模式受有限到达序列截断影响。

前面 scale 的样本既参与图像 decoder，也通过末组样本影响下一 scale 首组的分布。
仅改变跨尺度条件结构，不删除任何 scale 的重建输入或 KL。

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

本版本增加独立的图像解码网络，因此参数量、计算量和训练显存需求会增加。它与此前
仅从最后一个 scale 解码的 PR 初版 checkpoint 也不兼容，不能用宽松加载忽略缺少的
`image_ftr0` / `image_dec_tower` 参数后直接评估。

只有一个 scale 时没有边界，该开关不改变模型。多个 scale 但每个仅一个 group
也是受支持的配置。由于隐变量条件路径的终端特征被丢弃，部分仅服务于该特征的
参数可能没有梯度；`average_gradients` 会跳过 `grad is None` 的参数，避免同步时报错。

## 验证与限制

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

本次验证：Python 3.12、PyTorch 2.14.1+cpu，39 项测试通过。测试覆盖：

- 改变非末组样本、固定末组样本，不改变下一尺度的首组先验；
- 改变末组样本会改变下一尺度首组先验；固定下一尺度样本以后，后续先验不变；
- 固定其他样本，逐个改变任一尺度的任一 group，图像输出均发生变化；
- 独立叶子样本测试确认每个 group 都有直接重建梯度；
- 图像网络与条件网络参数独立，图像网络变化不改变给定轨迹的先验参数；
- 固定全部隐变量后改变 x，不改变图像输出，排除 x 特征旁路；
- forward 与 prior sampling 在相同隐变量轨迹下给出相同先验参数和输出；
- 两/三尺度、不同分组数、自适应分组、单组尺度、pre/postprocess、残差后验开关；
- Gaussian、Poisson、混合分布、Gaussian flow、REINFORCE/O-BBVI，图像 decoder 接收全部实际 flow 后样本；
- 重建项对前面 scale 末组样本的非零梯度，以及每个边界参数的有限梯度；
- 全部组的 KL、独立初始参数、state_dict 恢复和旧 args 的兼容性；
- 谱正则、一次 Adamax 更新和无梯度参数的梯度同步逻辑。

CPU 测试只对旧代码中 AR mask 和谱正则的硬编码 `.cuda()` 分配临时替换设备位置，
实际模型计算和反向传播均执行；同步测试使用模拟 all_reduce。
未运行真实 GPU、多 GPU、原 PyTorch 1.6 环境或完整数据集训练，未报告收敛、FID 或 NLL 改善。
本改动没有更新原有依赖版本，也没有改造旧 CUDA/BN 实现。

