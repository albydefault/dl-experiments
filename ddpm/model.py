# Standard library imports

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import math

MODEL_REGISTRY = {}

def register_model(name):
    """
    A decorator to register a model class in the MODEL_REGISTRY.
    """
    def decorator(model_class):
        MODEL_REGISTRY[name] = model_class
        return model_class
    return decorator

def sinusoidal_positional_encoding(timesteps, dim):
    """
    Generates sinusoidal positional encoding.
    Args:
        timesteps: torch.Tensor of shape (batch_size, 1)
        dim: int, the dimension of the encoding.
    Returns:
        torch.Tensor of shape (batch_size, dim)
    """
    half_dim = dim // 2
    freq_embeddings = math.log(10000) / (half_dim - 1)
    freq_embeddings = torch.exp(torch.arange(half_dim, device=timesteps.device) * -freq_embeddings)

    time_embeddings = timesteps.unsqueeze(1) * freq_embeddings.unsqueeze(0)

    final_embeddings = torch.cat((time_embeddings.sin(), time_embeddings.cos()), dim=-1)
    
    if dim % 2 == 1:
        if final_embeddings.shape[1] < dim:
            final_embeddings = F.pad(final_embeddings, (0, 1))
    
    return final_embeddings

@register_model("unet32")
class UNet32(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, latent_dim=256, time_emb_dim=256):
        super(UNet32, self).__init__()
        self.dim3 = latent_dim
        self.dim2 = latent_dim // 2
        self.dim1 = latent_dim // 4

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        self.time_emb_dim = time_emb_dim

        # We expect the input to be of shape (batch_size, 3, 32, 32)

        # Encoding blocks

        self.t_proj1 = nn.Linear(time_emb_dim, self.dim1)
        self.block1 = self._make_block(in_channels, self.dim1)

        self.t_proj2 = nn.Linear(time_emb_dim, self.dim2)
        self.block2 = self._make_block(self.dim1, self.dim2)

        self.t_proj3 = nn.Linear(time_emb_dim, self.dim3)
        self.block3 = self._make_block(self.dim2, self.dim3)

        self.maxpool = nn.MaxPool2d(2)

        # Decoding blocks
        self.upconv1 = nn.ConvTranspose2d(self.dim3, self.dim2, kernel_size=2, stride=2)
        self.t_proj4 = nn.Linear(time_emb_dim, self.dim2)
        self.block4 = self._make_block(self.dim2 * 2, self.dim2)

        self.upconv2 = nn.ConvTranspose2d(self.dim2, self.dim1, kernel_size=2, stride=2)
        self.t_proj5 = nn.Linear(time_emb_dim, self.dim1)
        self.block5 = self._make_block(self.dim1 * 2, self.dim1)
        
        self.t_proj6 = nn.Linear(time_emb_dim, self.dim1)
        self.block6 = nn.Sequential(
            nn.Conv2d(self.dim1, self.dim1, kernel_size=3, padding=1),
            nn.GroupNorm(8, self.dim1),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.dim1, out_channels, kernel_size=1)
        )

        self.init_weights()




    def _make_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels), # GroupNorm is common in DDPMs
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True)
        )


    def forward(self, x, t):
        t_input = t.float().squeeze(-1)
        if t_input.ndim == 0:
            t_input = t_input.unsqueeze(0)
        sin_emb = sinusoidal_positional_encoding(t_input, self.time_emb_dim)
        temb = self.time_mlp(sin_emb)

        # Encoding
        enc1 = self.block1(x)
        enc1 = enc1 + self.t_proj1(temb)[:, :, None, None]

        enc2 = self.block2(self.maxpool(enc1))
        enc2 = enc2 + self.t_proj2(temb)[:, :, None, None]

        enc3 = self.block3(self.maxpool(enc2))
        enc3 = enc3 + self.t_proj3(temb)[:, :, None, None]

        # Decoding
        dec1 = self.upconv1(enc3)
        dec1 = torch.cat((dec1, enc2), dim=1)
        dec1 = self.block4(dec1)
        dec1 = dec1 + self.t_proj4(temb)[:, :, None, None]

        dec2 = self.upconv2(dec1)
        dec2 = torch.cat((dec2, enc1), dim=1)
        dec2 = self.block5(dec2)
        dec2 = dec2 + self.t_proj5(temb)[:, :, None, None]

        out = dec2 + self.t_proj6(temb)[:, :, None, None]
        out = self.block6(out)

        return out
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

@register_model("wnet32")
class WNet32(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, latent_dim=256, time_emb_dim=256):
        super(WNet32, self).__init__()
        self.first = UNet32(in_channels, out_channels, latent_dim, time_emb_dim)
        self.second = UNet32(in_channels, out_channels, latent_dim, time_emb_dim)

    def forward(self, x, t):
        x = self.first(x, t)
        x = self.second(x, t)
        return x

@register_model("recurrent_wnet32")
class RecurrentWNet32(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, latent_dim=256, time_emb_dim=256):
        super(RecurrentWNet32, self).__init__()
        self.first = UNet32(in_channels, out_channels, latent_dim, time_emb_dim)
        self.second = UNet32(in_channels, out_channels, latent_dim, time_emb_dim)

    def forward(self, x, t):
        first_x = self.first(x, t)
        second_x = self.second(x, t)
        third_x = self.second(first_x, t)
        return (first_x + second_x + third_x) / 3

@register_model("unet_attn")
class UNetAttn(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, latent_dim=256, time_emb_dim=256):
        super(UNetAttn, self).__init__()
        self.unet = UNet32(in_channels, 16, latent_dim, time_emb_dim)
        self.attn = nn.MultiheadAttention(embed_dim=16, num_heads=8, batch_first=True)
        self.pos_emb = nn.Parameter(torch.randn(32 * 32, 16))
        self.output_layer = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x, t):
        x = self.unet(x, t)
        b, c, h, w = x.size()

        x_flat = x.view(b, c, -1).permute(0, 2, 1)  # (batch_size, h*w, channels)
        x_flat += self.pos_emb.unsqueeze(0)  # Add positional encoding
        attn_output, _ = self.attn(x_flat, x_flat, x_flat)
        out = attn_output.permute(0, 2, 1).view(b, c, h, w)  # (batch_size, channels, h*w)
        out = self.output_layer(out)
        return out