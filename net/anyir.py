"""
AnyIR: 
Author: Bin Ren
"""

import numbers
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from fvcore.nn import FlopCountAnalysis, flop_count_table
from timm.models.layers import trunc_normal_, DropPath


# Constants
BIAS_FREE = 'BiasFree'
WITH_BIAS = 'WithBias'


# Utility Functions
def to_3d(x):
    """Reshape tensor from 4D to 3D."""
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    """Reshape tensor from 3D back to 4D."""
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


# Layer Normalization Components
class BiasFree_LayerNorm(nn.Module):
    """Layer normalization without bias."""
    
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        
        assert len(normalized_shape) == 1
        
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape
    
    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    """Layer normalization with bias."""
    
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        
        assert len(normalized_shape) == 1
        
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape
    
    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    """Layer normalization factory that creates either biased or unbiased norm."""
    
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == BIAS_FREE:
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)
    
    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# Neural Network Building Blocks
class FeedForward(nn.Module):
    """Feed Forward Network with GELU activation and depthwise convolution."""
    
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        
        hidden_features = int(dim * ffn_expansion_factor)
        
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2, 
            hidden_features * 2, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            groups=hidden_features * 2, 
            bias=bias
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
    
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class GatedCNNBlock(nn.Module):
    """Gated CNN Block implementation based on https://arxiv.org/pdf/1612.08083"""
    
    def __init__(
        self, 
        dim, 
        expansion_ratio=8/3, 
        kernel_size=7, 
        conv_ratio=1.0,
        norm_layer=nn.BatchNorm2d, 
        act_layer=nn.GELU,
        drop_path=0.,
        **kwargs
    ):
        super().__init__()
        self.norm = norm_layer(dim)
        hidden = int(expansion_ratio * dim)
        self.gic1 = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=False)
        self.act = act_layer()
        
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        self.cconv = nn.Conv2d(
            conv_channels, 
            conv_channels, 
            kernel_size=kernel_size, 
            padding=kernel_size//2, 
            groups=conv_channels
        )
        self.gic2 = nn.Conv2d(hidden, dim, kernel_size=1, bias=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
    
    def forward(self, x):
        shortcut = x
        x = self.norm(x)

        gic = self.gic1(x)  # [B, 2 * hidden, H, W]
        g, i, c = torch.split(gic, self.split_indices, dim=1)  # [B, x, H, W] for each
        c = self.cconv(c)  # [B, conv_channels, H, W]
        x = self.gic2(self.act(g) * torch.cat((i, c), dim=1))  # [B, C, H, W]
        x = self.drop_path(x)

        x = x + shortcut
        x = self.proj_out(x)

        return x

class GatedDegradationAdaption(nn.Module):
    """
    Algorithm 1: Gated Degradation Adaption

    Input:  F_in^gate  ∈ R^{C×H×W}
    Output:  F_out^gate ∈ R^{C×H×W}

    """
    def __init__(
        self,
        dim: int,
        expansion_ratio: float = 8/3,   # 与你原实现保持一致（≈2.667）
        kernel_size: int = 7,
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.GELU,
        bias: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert expansion_ratio > 2.0, "expansion_ratio 必须 > 2 才能切出 γ、β、α 三段"

        self.eps = eps
        self.norm = norm_layer(dim)
        self.act = act_layer()

        # ---- W_exp：通道扩展后再 split γ、β、α
        hidden = int(round(expansion_ratio * dim))
        gamma_ch = dim
        beta_ch  = dim
        alpha_ch = hidden - gamma_ch - beta_ch
        assert alpha_ch > 0, "alpha 通道数必须 > 0，请增大 expansion_ratio"

        self.W_exp = nn.Conv2d(dim, hidden, kernel_size=1, bias=bias)

        # ---- W_depth：对 α 做 depthwise conv
        self.W_depth = nn.Conv2d(
            alpha_ch, alpha_ch,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=alpha_ch,
            bias=bias
        )

        # ---- W_gate：concat(β, α') 之后回到 C
        self.W_gate = nn.Conv2d(beta_ch + alpha_ch, dim, kernel_size=1, bias=bias)

        # ---- W_proj：残差后 1×1 投影
        self.W_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # 初始温度 τ（可学习），按通道缩放更灵活
        self.tau = nn.Parameter(torch.ones(1, dim, 1, 1))

        # 保存切分索引
        self.split_indices = (gamma_ch, beta_ch, alpha_ch)

    def forward(self, x):
        F_in = x

        # 1) Normalize
        F_hat = self.norm(F_in)

        # 2) 每通道的空间均值/方差
        mu = F_hat.mean(dim=(2, 3), keepdim=True)                        # [B,C,1,1]
        var = ((F_hat - mu) ** 2).mean(dim=(2, 3), keepdim=True)
        sigma = torch.sqrt(var + self.eps)

        # 3) Δ = Sigmoid(mu + sigma)
        Delta = torch.sigmoid(mu + sigma)

        # 4) τ_adj = τ * Δ
        tau_adj = self.tau * Delta

        # 5) F' = F_hat W_exp
        F_prime = self.W_exp(F_hat)

        # 6) γ, β, α = split(F')
        gamma, beta, alpha = torch.split(F_prime, self.split_indices, dim=1)

        # 7) α' = (α ⊗ W_depth) * (1 + τ_adj)
        alpha_dw = self.W_depth(alpha)
        # 广播到 α 的通道数（tau_adj 是 [B,C,1,1]，只影响前 C 个通道；采用通道切片对齐 α' 规模）
        # 简洁起见，用逐元素缩放时把 tau_adj 限制到 dim，再按需要重复/插值到 alpha 的通道数
        if alpha_dw.shape[1] != tau_adj.shape[1]:
            # 重复/裁剪以匹配 α' 的通道数
            repeat_factor = (alpha_dw.shape[1] + tau_adj.shape[1] - 1) // tau_adj.shape[1]
            tau_alpha = tau_adj.repeat(1, repeat_factor, 1, 1)[:, :alpha_dw.shape[1]]
        else:
            tau_alpha = tau_adj
        alpha_prime = alpha_dw * (1.0 + tau_alpha)

        # 8) F_gate = σ(γ) * concat(β, α') W_gate
        gate = torch.sigmoid(gamma)
        merged = torch.cat([beta, alpha_prime], dim=1)
        F_gate = gate * self.W_gate(merged)

        # 9) F_out = (F_gate + F_in) W_proj
        F_out = self.W_proj(F_gate + F_in)

        # 10) return
        return F_out


class Attention(nn.Module):
    """Multi-head self-attention with Spatial-Frequency Fusion (per screenshots).
       - SpatialFusion: cross-enhance + channel-wise concat
       - FrequencyFusion: rFFT2 combine + iFFT2, channel repeat to match
    """
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        assert dim % 2 == 0, "dim must be even (split att/gate)"
        self.num_heads = num_heads

        # 可学习的注意力温度（每个头）
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        half_dim = dim // 2

        # 注意力分支
        self.qkv = nn.Conv2d(half_dim, half_dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            half_dim * 3, half_dim * 3,
            kernel_size=3, stride=1, padding=1,
            groups=half_dim * 3, bias=bias
        )

        # 门控卷积分支（使用你给的实现）
        # self.gatedcnn = GatedCNNBlock(dim=half_dim, drop_path=0.1)

        self.gatedcnn = GatedDegradationAdaption(dim=half_dim, expansion_ratio=8/3, kernel_size=7)


        # 融合的可学习权重 λ（Algorithm 2 第3步）
        self.lam = nn.Parameter(torch.tensor(0.5))

        # 最终输出投影
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    # -------------------- Algorithm 2: helpers --------------------
    @staticmethod
    def _spatial_fusion(F_att, F_gate):
        # F^α_g = F_att + Sigmoid(F_gate)
        F_ag = F_att + torch.sigmoid(F_gate)
        # F^α_a = F_gate + Sigmoid(F_att)
        F_ga = F_gate + torch.sigmoid(F_att)
        # channel-wise merge: Concat(F^α_g, F^α_a)
        F_s = torch.cat([F_ag, F_ga], dim=1)
        return F_s

    @staticmethod
    def _frequency_fusion(F_att, F_gate, size_hw):
        # real FFT
        Fhat_att = torch.fft.rfft2(F_att)
        Fhat_gate = torch.fft.rfft2(F_gate)
        Fhat = Fhat_att + Fhat_gate
        # inverse FFT
        F_freq = torch.fft.irfft2(Fhat, s=size_hw)
        # Repeat to match channels of SpatialFusion (which is 2 * half_dim)
        F_freq_rep = F_freq.repeat_interleave(2, dim=1)
        return F_freq_rep
    # -------------------------------------------------------------

    def forward(self, x):
        # 1) 拆分通道：注意力 / 门控
        x_att = x[:, 0::2, :, :]  # even channels
        x_gate = x[:, 1::2, :, :]  # odd channels

        # --- Attention path ---
        b, c, h, w = x_att.shape
        qkv = self.qkv_dwconv(self.qkv(x_att))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out_att = attn @ v
        out_att = rearrange(out_att, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # --- Gated (degradation) path ---
        out_gate = self.gatedcnn(x_gate)

        # --- Algorithm 2: Spatial-Frequency Fusion ---
        Fs = self._spatial_fusion(out_att, out_gate)                 # spatial cross-enhance + concat
        Ff = self._frequency_fusion(out_att, out_gate, (h, w))       # rFFT combine + iFFT + repeat

        F_fuse = self.lam * Fs + (1.0 - self.lam) * Ff               # weighted fusion
        out = self.project_out(F_fuse)                               # residual/project in outer block if needed
        return out


class ResBlock(nn.Module):
    """Residual block with two convolutions."""
    
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


# Resizing modules
class Downsample(nn.Module):
    """Downsample spatial resolution by 2x and double channels."""
    
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    """Upsample spatial resolution by 2x and halve channels."""
    
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward network."""
    
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    """Overlapped image patch embedding with 3x3 Conv."""
    
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class AnyIR(nn.Module):
    """
    AnyIR: A neural network for image restoration.
    
    This model uses a UNet-like architecture with transformer blocks for feature extraction
    and processing at multiple resolutions.
    
    Args:
        inp_channels (int): Number of input channels (default: 3)
        out_channels (int): Number of output channels (default: 3)
        dim (int): Base dimension for feature maps (default: 24)
        num_blocks (list): Number of transformer blocks at each resolution (default: [3,5,5,7])
        num_refinement_blocks (int): Number of refinement blocks (default: 2)
        heads (list): Number of attention heads at each resolution (default: [1,2,4,8])
        ffn_expansion_factor (float): Expansion factor for FFN (default: 2)
        bias (bool): Whether to use bias in convolutions (default: False)
        LayerNorm_type (str): Type of layer normalization (default: 'WithBias')
    """
    
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=32,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2,
        bias=False,
        LayerNorm_type=WITH_BIAS,
    ):
        super(AnyIR, self).__init__()

        # Initial feature extraction
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # Encoder path
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(
                dim=dim,
                num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[0])
        ])
        
        # Downsampling and level 2 processing
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim*2**1),
                num_heads=heads[1],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[1])
        ])
        
        # Downsampling and level 3 processing
        self.down2_3 = Downsample(int(dim*2**1))
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim*2**2),
                num_heads=heads[2],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[2])
        ])
        
        # Downsampling and bottleneck (latent) processing
        self.down3_4 = Downsample(int(dim*2**2))
        self.latent = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim*2**3),
                num_heads=heads[3],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[3])
        ])
        
        # Decoder path
        # Level 4 to 3
        self.up4_3 = Upsample(int(dim*2**2))
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**1)+int(dim*2**2), int(dim*2**2), kernel_size=1, bias=bias)
        self.reduce_dim_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim*2**2),
                num_heads=heads[2],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[2])
        ])
        
        # Level 3 to 2
        self.up3_2 = Upsample(int(dim*2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim*2**1),
                num_heads=heads[1],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[1])
        ])
        
        # Level 2 to 1
        self.up2_1 = Upsample(int(dim*2**1))
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim*2**1),
                num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[0])
        ])
        
        # Final refinement and output
        self.refinement = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim*2**1),
                num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_refinement_blocks)
        ])
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img, noise_emb=None):
        """
        Forward pass of the AnyIR network.
        
        Args:
            inp_img (Tensor): Input image tensor [B, inp_channels, H, W]
            noise_emb (Tensor, optional): Noise embedding tensor (not used in this implementation)
            
        Returns:
            Tensor: Restored image tensor [B, out_channels, H, W]
        """
        # --- Initial feature extraction
        inp_enc_level1 = self.patch_embed(inp_img)  # [B, dim, H, W]
        
        # --- Encoder path (downsampling)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)  # [B, dim, H, W]
        
        inp_enc_level2 = self.down1_2(out_enc_level1)  # [B, dim*2, H/2, W/2]
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        
        inp_enc_level3 = self.down2_3(out_enc_level2)  # [B, dim*4, H/4, W/4]
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        
        inp_enc_level4 = self.down3_4(out_enc_level3)  # [B, dim*8, H/8, W/8]
        latent = self.latent(inp_enc_level4)
        
        # --- Prepare bottleneck features for decoder
        latent = self.reduce_dim_level3(latent)  # [B, dim*4, H/8, W/8]
        
        # --- Decoder path (upsampling with skip connections)
        # Level 4 to 3
        inp_dec_level3 = self.up4_3(latent)  # [B, dim*4, H/4, W/4]
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        
        # Level 3 to 2
        inp_dec_level2 = self.up3_2(out_dec_level3)  # [B, dim*2, H/2, W/2]
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        
        # Level 2 to 1
        inp_dec_level1 = self.up2_1(out_dec_level2)  # [B, dim*2, H, W]
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        # --- Final refinement
        out_dec_level1 = self.refinement(out_dec_level1)
        
        # Output with residual connection to input
        output = self.output(out_dec_level1) + inp_img
        
        return output


# Testing code with performance metrics
if __name__ == "__main__":
    # Create model instance
    model = AnyIR(
        num_blocks=[3, 5, 5, 7],
        dim=28,
        ffn_expansion_factor=2,
        num_refinement_blocks=4
    ).cuda()
    
    # Create test input
    x = torch.randn(1, 3, 224, 224).cuda()
    
    # Print model architecture
    print(model)
    
    # Run forward pass
    out = model(x)
    
    # Print memory usage
    print('{:>16s} : {:<.3f} [M]'.format(
        'Max Memory', 
        torch.cuda.max_memory_allocated(torch.cuda.current_device())/1024**2
    ))
    print(f'Output shape: {out.shape}')
    
    # Calculate and print FLOPS and parameters
    flops = FlopCountAnalysis(model, (x))
    print(flop_count_table(flops))