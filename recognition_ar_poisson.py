"""AR/NAR Poisson recognition from Recognition_Model_NVAE_AR_Poisson.md.

Transformer topology follows jadex recognition; training uses NELBO with ST
rather than the DAPS policy-search objective. Requires PyTorch >= 2.0.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F


def sinusoidal_positions(length, width):
    # 与仓库一致：前半维 sin，后半维 cos。
    if width < 4 or width % 2:
        raise ValueError("embed_dim must be even and >= 4")
    half = width // 2
    pos = torch.arange(length, dtype=torch.float32)[:, None]
    freq = torch.exp(-math.log(10000.0) *
                     torch.arange(half, dtype=torch.float32) / (half - 1))
    return torch.cat([torch.sin(pos * freq), torch.cos(pos * freq)], -1)[None]


class MHA(nn.Module):
    def __init__(self, width, heads, attention_dropout=0.0):
        super().__init__()
        if width % heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.heads, self.head_dim = heads, width // heads
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        self.attention_dropout = attention_dropout

    def split(self, x):
        b, t, _ = x.shape
        return x.reshape(b, t, self.heads, self.head_dim).transpose(1, 2)

    def project_kv(self, memory):
        return self.split(self.k(memory)), self.split(self.v(memory))

    def attend(self, query, kv, allowed=None):
        q = self.split(self.q(query))
        k, v = kv
        score = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if allowed is not None:  # True = 可以看见
            score = score.masked_fill(~allowed, float("-inf"))
        weights = F.softmax(score.float(), dim=-1).to(v.dtype)
        weights = F.dropout(weights, self.attention_dropout, self.training)
        h = weights @ v
        h = h.transpose(1, 2).contiguous().reshape(query.shape)
        return self.out(h)

    def forward(self, query, memory, allowed=None):
        return self.attend(query, self.project_kv(memory), allowed)

    def step(self, x, old_kv):
        k, v = self.project_kv(x)
        if old_kv is not None:
            k = torch.cat([old_kv[0], k], dim=2)
            v = torch.cat([old_kv[1], v], dim=2)
        # 每次只传入一个新位置；cache 中只有过去及当前输入，无未来位置。
        return self.attend(x, (k, v)), (k, v)


class FFN(nn.Module):
    def __init__(self, width, ratio, dropout):
        super().__init__()
        self.fc1 = nn.Linear(width, width * ratio)
        self.fc2 = nn.Linear(width * ratio, width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(F.relu(self.fc1(x)))))


class EncoderBlock(nn.Module):
    def __init__(self, width, heads, ratio, dropout, attention_dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(width, eps=1e-6)
        self.ln2 = nn.LayerNorm(width, eps=1e-6)
        self.attn = MHA(width, heads, attention_dropout)
        self.ffn = FFN(width, ratio, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.drop(self.attn(h, h))
        return x + self.ffn(self.ln2(x))


class DecoderBlock(nn.Module):
    def __init__(self, width, heads, ratio, dropout, attention_dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(width, eps=1e-6)
        self.ln2 = nn.LayerNorm(width, eps=1e-6)
        self.ln3 = nn.LayerNorm(width, eps=1e-6)
        self.self_attn = MHA(width, heads, attention_dropout)
        self.cross_attn = MHA(width, heads, attention_dropout)
        self.ffn = FFN(width, ratio, dropout)
        self.drop = nn.Dropout(dropout)

    def step(self, x, self_kv, cross_kv):
        h, new_self_kv = self.self_attn.step(self.ln1(x), self_kv)
        x = x + self.drop(h)
        x = x + self.drop(self.cross_attn.attend(self.ln2(x), cross_kv))
        x = x + self.ffn(self.ln3(x))
        return x, new_self_kv

    def full(self, x, cross_kv, causal_mask):
        h = self.ln1(x)
        x = x + self.drop(self.self_attn(h, h, causal_mask))
        x = x + self.drop(self.cross_attn.attend(self.ln2(x), cross_kv))
        return x + self.ffn(self.ln3(x))


class RecognitionScale1(nn.Module):
    """单尺度；q(z|x)=prod_i q(z_i|x,z_<i)，V 固定为 64。"""
    def __init__(self, image_shape=(3, 32, 32), patch_grid=(8, 8),
                 d_lat=64, embed_dim=128, num_heads=4, num_layers=2,
                 mlp_ratio=4, dropout=0.0, attention_dropout=0.0):
        super().__init__()
        c, ih, iw = image_shape
        gh, gw = patch_grid
        if not 1 <= d_lat <= 1024:
            raise ValueError("d_lat must be in [1, 1024]")
        if min(c, ih, iw, gh, gw, d_lat, num_layers) <= 0:
            raise ValueError("shape, d_lat and num_layers must be positive")
        if embed_dim < 4 or embed_dim % 2:
            raise ValueError("embed_dim must be even and >= 4")
        # 仓库 Patcher：ceil 后把 patch 边长向上取成偶数，再居中补零。
        ph = 2 * math.ceil(math.ceil(ih / gh) / 2)
        pw = 2 * math.ceil(math.ceil(iw / gw) / 2)
        self.image_shape = tuple(image_shape)
        self.grid, self.patch = (gh, gw), (ph, pw)
        self.d_lat, self.vocab_size = d_lat, 64
        self.patch_proj = nn.Linear(ph * pw * c, embed_dim)
        self.embedding = nn.Embedding(64, embed_dim)
        args = (embed_dim, num_heads, mlp_ratio, dropout, attention_dropout)
        self.encoder = nn.ModuleList([EncoderBlock(*args) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([DecoderBlock(*args) for _ in range(num_layers)])
        self.enc_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dec_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, 64)
        self.input_drop = nn.Dropout(dropout)
        self.register_buffer("image_pos", sinusoidal_positions(gh * gw, embed_dim))
        self.register_buffer("latent_pos", sinusoidal_positions(d_lat, embed_dim))
        # PyTorch 参考初始化；层拓扑对齐，未声称逐位复现 Flax 初始化。
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=1.0)

    def patchify(self, x):
        if tuple(x.shape[1:]) != self.image_shape:
            raise ValueError("x must be [batch, channels, height, width]")
        b, c, ih, iw = x.shape
        gh, gw = self.grid
        ph, pw = self.patch
        dh, dw = gh * ph - ih, gw * pw - iw
        x = F.pad(x, (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2))
        # patch 内按 (patch_height, patch_width, channel) 展平，与 NHWC 源码一致。
        return (x.reshape(b, c, gh, ph, gw, pw)
                .permute(0, 2, 4, 3, 5, 1).contiguous()
                .reshape(b, gh * gw, ph * pw * c))

    def encode_image(self, x):
        h = self.input_drop(self.patch_proj(self.patchify(x)) + self.image_pos)
        for block in self.encoder:
            h = block(h)
        return self.enc_norm(h)

    def prepare_cross_kv(self, memory):
        # 只在本次 forward 内复用；训练时不 detach。
        return [block.cross_attn.project_kv(memory) for block in self.decoder]

    def sample(self, x, tau=1.0):
        return self.sample_memory(self.encode_image(x), tau)

    def sample_memory(self, memory, tau=1.0):
        if tau is not None and tau <= 0:
            raise ValueError("tau must be positive or None")
        b = memory.shape[0]
        cross_kv = self.prepare_cross_kv(memory)
        self_kv = [None] * len(self.decoder)
        # 与仓库 PAD_ID=0 起始方式一致。0 仍是有效类别，不作为 padding mask。
        prev = memory.new_zeros(b, 1, 64)
        prev[..., 0] = 1.0
        all_logits, all_y, all_z = [], [], []
        for t in range(self.d_lat):
            h = self.input_drop(prev @ self.embedding.weight +
                                self.latent_pos[:, t:t + 1])
            new_cache = []
            for block, old, cross in zip(self.decoder, self_kv, cross_kv):
                h, kv = block.step(h, old, cross)
                new_cache.append(kv)
            self_kv = new_cache
            logits = self.head(self.dec_norm(h)).squeeze(1)
            # 独立 Gumbel；硬采样分布为 softmax(logits)，与 tau 无关。
            g = -torch.empty_like(logits).exponential_().log()
            noisy = logits + g
            z = noisy.argmax(-1)
            hard = F.one_hot(z, 64).to(logits.dtype)
            if tau is None:
                y = hard  # 评估：仍是随机 categorical 采样，非 argmax(logits)。
            else:
                soft = F.softmax(noisy / tau, -1)
                y = hard + (soft - soft.detach())
            all_logits.append(logits)
            all_y.append(y)
            all_z.append(z)
            prev = y[:, None, :]  # 不转换成整数索引，不 detach。
        logits = torch.stack(all_logits, dim=1)
        y_st = torch.stack(all_y, dim=1)
        z = torch.stack(all_z, dim=1)
        log_probs = F.log_softmax(logits, dim=-1)
        # 训练：前向为精确离散 log q，反向保留 y_st 路径。
        log_q_i = (y_st * log_probs).sum(-1)
        # 记录：同一个硬样本的 log q；仅用于指标/一致性检查。
        log_q_i_hard = log_probs.gather(-1, z[..., None]).squeeze(-1)
        return dict(z=z, y_st=y_st, logits=logits, probs=log_probs.exp(),
                    log_probs=log_probs, log_q_i=log_q_i,
                    log_q=log_q_i.sum(-1), log_q_i_hard=log_q_i_hard,
                    self_kv=self_kv, cross_kv=cross_kv)

    def forward(self, x, tau=1.0):
        return self.sample(x, tau)

    def teacher_logits(self, x, y):
        """给定序列评估 q；不能代替从 q 自回归采样。y: [B,d_lat,64]。"""
        if tuple(y.shape) != (x.shape[0], self.d_lat, 64):
            raise ValueError("y must be [batch, d_lat, 64]")
        return self.teacher_logits_memory(self.encode_image(x), y)

    def teacher_logits_memory(self, memory, y):
        if tuple(y.shape) != (memory.shape[0], self.d_lat, 64):
            raise ValueError("y must be [batch, d_lat, 64]")
        cross_kv = self.prepare_cross_kv(memory)
        bos = y.new_zeros(y.shape[0], 1, 64)
        bos[..., 0] = 1.0
        shifted = torch.cat([bos, y[:, :-1]], dim=1)
        h = self.input_drop(shifted @ self.embedding.weight + self.latent_pos)
        allowed = torch.ones(self.d_lat, self.d_lat, device=memory.device,
                             dtype=torch.bool).tril()[None, None]
        for block, cross in zip(self.decoder, cross_kv):
            h = block.full(h, cross, allowed)
        return self.head(self.dec_norm(h))



# 以下代码接在 RecognitionScale1 定义之后，合并为一个 Python 模块。

def positive_rate(raw):
    # 与 NVAE 的 soft_clamp5 形式一致；本参考实现不额外裁剪 rate 上限。
    return torch.exp(5.0 * torch.tanh(raw / 5.0))


def poisson_log_pmf(count, rate):
    # count 的前向必须是非负整数；ST 反向可以使用该连续扩展。
    return count * torch.log(rate) - rate - torch.lgamma(count + 1.0)


def poisson_kl(rate_q, rate_p):
    return rate_q * (torch.log(rate_q) - torch.log(rate_p)) + rate_p - rate_q


def poisson_survival(rate, cap=64):
    # P(Pois(rate) >= cap) = 正则化下不完全 gamma(cap, rate)。
    r = rate.double()
    return torch.special.gammainc(torch.full_like(r, float(cap)), r)


def capped_poisson_kl(rate_q, rate_p, cap=64):
    """KL(Law(min(Pois(rate_q),cap)) || Pois(rate_p))，逐坐标解析有限和。"""
    rq, rp = rate_q.double(), rate_p.double()
    k = torch.arange(cap, device=rq.device, dtype=rq.dtype)
    log_q_low = poisson_log_pmf(k, rq[..., None])
    log_p_low = poisson_log_pmf(k, rp[..., None])
    low = (log_q_low.exp() * (log_q_low - log_p_low)).sum(-1)
    tail = poisson_survival(rq, cap)
    log_tail = tail.clamp_min(torch.finfo(tail.dtype).tiny).log()
    log_p_cap = poisson_log_pmf(rp.new_tensor(float(cap)), rp)
    kl = low + tail * (log_tail - log_p_cap)
    return kl.to(rate_q.dtype)


def capped_poisson_log_q(hard_count, rate_q, cap=64):
    # cap 处是完整尾质量，不能使用普通 Poisson 在 cap 的单点质量。
    tail = poisson_survival(rate_q, cap)
    log_tail = tail.clamp_min(torch.finfo(tail.dtype).tiny).log().to(rate_q.dtype)
    return torch.where(hard_count == cap, log_tail,
                       poisson_log_pmf(hard_count, rate_q))


def cts_st(rate, tau=0.1, max_count=64, mode="capped64"):
    """
    capped64: 与原 NVAE 的 64 次到达展开一致，硬计数范围 0..64。
    exact64: 软展开仍为 64 项；在硬分支补尾，硬样本严格为普通 Poisson。
    tau=None: 关闭软反向，但保持所选 mode 的硬样本分布。
    """
    if max_count != 64:
        raise ValueError("this specification fixes max_count=64")
    if mode not in ("capped64", "exact64"):
        raise ValueError("mode must be capped64 or exact64")
    if tau is not None and tau <= 0:
        raise ValueError("tau must be positive or None")
    # base ~ Exp(1)，arrival ~ Exp(rate)；可微重参数化。
    base = torch.empty((64,) + tuple(rate.shape), device=rate.device,
                       dtype=rate.dtype).exponential_()
    arrivals = (base / rate.unsqueeze(0)).cumsum(dim=0)
    hard = (arrivals <= 1.0).to(rate.dtype).sum(0)
    if mode == "exact64":
        # 条件于第 64 次到达 S_64<1，剩余计数服从 Pois(rate*(1-S_64))。
        # 仅硬分支补尾；梯度按用户要求使用前 64 项软近似。
        remaining_rate = (rate * (1.0 - arrivals[-1]).clamp_min(0)).detach()
        hard = hard + torch.poisson(remaining_rate)
    if tau is None:
        z_st = hard
    else:
        soft = torch.sigmoid((1.0 - arrivals) / tau).sum(0)
        z_st = hard + (soft - soft.detach())
    return dict(z_st=z_st, z_hard=hard.detach(),
                saturated=(arrivals[-1] <= 1.0).detach())


class LearnablePoissonPrior(nn.Module):
    """单层/顶层的可学习先验；每个标量坐标一个 raw rate 参数。"""
    def __init__(self, latent_shape):
        super().__init__()
        self.raw = nn.Parameter(torch.zeros(1, *latent_shape))

    def forward(self, batch_size):
        return self.raw.expand(batch_size, *self.raw.shape[1:])


class ARFeatureRecognition(RecognitionScale1):
    """在 NVAE posterior feature map 上使用附录 I 的 encoder/decoder 结构。"""
    def __init__(self, feature_shape, d_lat, **kwargs):
        c, h, w = feature_shape
        super().__init__(image_shape=feature_shape, patch_grid=(h, w),
                         d_lat=d_lat, **kwargs)
        # NVAE 已提供卷积特征；把每个空间位置当一个 patch，而非再次补零。
        del self.patch_proj
        self.feature_proj = nn.Linear(c, self.embedding.embedding_dim)
        self._init(self.feature_proj)

    def encode_image(self, feature):
        if tuple(feature.shape[1:]) != self.image_shape:
            raise ValueError("unexpected NVAE posterior feature shape")
        tokens = feature.flatten(2).transpose(1, 2)
        h = self.input_drop(self.feature_proj(tokens) + self.image_pos)
        for block in self.encoder:
            h = block(h)
        return self.enc_norm(h)


def ar_group_result(out, rate_p, latent_shape):
    b, t, v = out["y_st"].shape
    if v != 64 or math.prod(latent_shape) != t:
        raise ValueError("latent_shape must contain exactly d_lat scalars")
    rp = rate_p.flatten(1)
    if rp.shape != (b, t):
        raise ValueError("prior rate shape must match the latent group")
    k = torch.arange(64, device=rp.device, dtype=rp.dtype)
    # 这是普通 Poisson 的 64 个 PMF 值；不要 log_softmax 或除以它们的和。
    log_p_all = poisson_log_pmf(k, rp[..., None])
    log_p_i = (out["y_st"] * log_p_all).sum(-1)
    kl_i = out["log_q_i"] - log_p_i
    count_st = out["y_st"] @ k
    return dict(kind="ar", z_st=count_st.reshape(b, *latent_shape),
                z_hard=out["z"].reshape(b, *latent_shape),
                kl_per_var=kl_i, kl=kl_i.sum(-1),
                log_q=out["log_q"], log_p=log_p_i.sum(-1),
                categorical=out)


class ARPoissonGroup(nn.Module):
    def __init__(self, feature_shape, latent_shape, **kwargs):
        super().__init__()
        self.latent_shape = tuple(latent_shape)
        self.rec = ARFeatureRecognition(feature_shape,
                                       d_lat=math.prod(latent_shape), **kwargs)

    def forward(self, feature, prior_raw, tau=1.0):
        # Sampling, one-hot/count matmuls and PMFs must remain float32 under AMP.
        with torch.autocast(device_type=feature.device.type, enabled=False):
            out = self.rec(feature.float(), tau=tau)
            return ar_group_result(out, positive_rate(prior_raw.float()), self.latent_shape)


class IndependentPoissonGroup(nn.Module):
    """NVAE 风格的 ELU+3x3 Conv rate head，组内条件独立。"""
    def __init__(self, feature_channels, latent_shape, res_dist=True):
        super().__init__()
        c, h, w = latent_shape
        if not 1 <= math.prod(latent_shape) <= 1024:
            raise ValueError("each group must contain <= 1024 scalar latents")
        self.latent_shape, self.res_dist = tuple(latent_shape), res_dist
        self.head = nn.Sequential(nn.ELU(), nn.Conv2d(feature_channels, c, 3, padding=1))

    def forward(self, feature, prior_raw, tau=0.1, mode="capped64"):
        with torch.autocast(device_type=feature.device.type, enabled=False):
            return self._forward_float(feature.float(), prior_raw.float(), tau, mode)

    def _forward_float(self, feature, prior_raw, tau, mode):
        raw_q = self.head(feature)
        if self.res_dist:
            raw_q = raw_q + prior_raw
        if tuple(raw_q.shape[1:]) != self.latent_shape or raw_q.shape != prior_raw.shape:
            raise ValueError("feature and prior spatial dimensions must match latent_shape")
        rq, rp = positive_rate(raw_q), positive_rate(prior_raw)
        sample = cts_st(rq, tau=tau, mode=mode)
        if mode == "capped64":
            # 保留硬截断，使用该截尾后验与普通 Poisson 先验的精确解析有限和。
            kl_map = capped_poisson_kl(rq, rp)
            log_q_map = capped_poisson_log_q(sample["z_hard"], rq)
        else:
            # 真正的独立 Poisson 后验，可使用标准解析 KL。
            kl_map = poisson_kl(rq, rp)
            log_q_map = poisson_log_pmf(sample["z_hard"], rq)
        log_p_map = poisson_log_pmf(sample["z_hard"], rp)
        return dict(kind="nar", mode=mode, z_st=sample["z_st"],
                    z_hard=sample["z_hard"], kl_per_var=kl_map.flatten(1),
                    kl=kl_map.flatten(1).sum(-1),
                    log_q=log_q_map.flatten(1).sum(-1),
                    log_p=log_p_map.flatten(1).sum(-1),
                    rate_q=rq, rate_p=rp, saturated=sample["saturated"])


def mixed_nelbo(log_px, group_outputs):
    """一次完整分层轨迹：log_px:[B]；按位置求和，最后按 batch 求平均。"""
    kl = torch.stack([out["kl"] for out in group_outputs], dim=0).sum(0)
    per_image = -log_px + kl
    return per_image.mean(), dict(nelbo=per_image.detach().mean(),
                                  recon_nll=(-log_px).detach().mean(),
                                  kl=kl.detach().mean())
