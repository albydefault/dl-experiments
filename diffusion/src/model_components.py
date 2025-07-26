import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class FiLMGroupNorm(nn.Module):
    def __init__(self,
                 num_groups: int,
                 num_channels: int,
                 conditioning_dim: int):
        super(FiLMGroupNorm, self).__init__()
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
        self.gamma = nn.Linear(conditioning_dim, num_channels)
        self.beta = nn.Linear(conditioning_dim, num_channels)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        normed = self.group_norm(x)
        gamma = self.gamma(conditioning).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(conditioning).unsqueeze(-1).unsqueeze(-1)
        return normed * (1 + gamma) + beta

class DoubleConv(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 mid_channels: int,
                 conditioning_dim: int,
                 dropout: float,
                 activation: nn.Module = nn.SiLU()):
        super(DoubleConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.conditioning_dim = conditioning_dim
        self.dropout = dropout
        self.activation = activation

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.norm1 = FiLMGroupNorm(num_groups=32, num_channels=mid_channels, conditioning_dim=conditioning_dim)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = FiLMGroupNorm(num_groups=32, num_channels=out_channels, conditioning_dim=conditioning_dim)
        self.dropout_layer = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.norm1(x, conditioning)
        x = self.activation(x)
        x = self.dropout_layer(x)
        x = self.conv2(x)
        x = self.norm2(x, conditioning)
        x = self.activation(x)
        x = self.dropout_layer(x)
        return x
    
class Down(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 mid_channels: int,
                 conditioning_dim: int,
                 dropout: float = 0.0):
        super(Down, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels, mid_channels, conditioning_dim, dropout)
        self.pool = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        x = self.conv(x, conditioning)
        pooled = self.pool(x)
        return pooled
    
class Up(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 mid_channels: int,
                 conditioning_dim: int,
                 dropout: float = 0.0):
        super(Up, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels, mid_channels, conditioning_dim, dropout)
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor, skip_connection: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip_connection], dim=1)
        x = self.conv(x, conditioning)
        return x
    

class AttentionBlock(nn.Module):
    def __init__(self,
                 channels: int,
                 num_heads: int = 1):
        super(AttentionBlock, self).__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = self.norm_layer(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    @staticmethod
    def norm_layer(channels: int) -> nn.Module:
        num_groups = min(32, channels)
        while channels % num_groups != 0:
            num_groups -= 1

        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(B * self.num_heads, -1, H * W).chunk(3, dim=1)
        scale = 1.0 / math.sqrt(math.sqrt(C // self.num_heads))
        
        attn = torch.einsum('bci,bcj->bij', q * scale, k * scale)
        attn = attn.softmax(dim=-1)
        h = torch.einsum("bij,bcj->bci", attn, v)
        h = h.reshape(B, -1, H, W)
        h = self.proj(h)
        return x + h