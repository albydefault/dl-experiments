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

class UNet32(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, latent_dim=256):
        super(UNet32, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.dim3 = latent_dim
        self.dim2 = latent_dim // 2
        self.dim1 = latent_dim // 4

        # We expect the input to be of shape (batch_size, 3, 32, 32)

        # Encoding blocks
        # Input image: (batch_size, 3, 32, 32)
        # Output image: (batch_size, 64, 32, 32)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, self.dim1, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=self.dim1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.dim1, self.dim1, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=self.dim1),
            nn.LeakyReLU(inplace=True),
        )

        # Previous block output: (batch_size, 64, 32, 32)
        # Max pooling: (batch_size, 64, 16, 16)
        # Output image: (batch_size, 128, 16, 16)
        self.block2 = nn.Sequential(
            nn.Conv2d(self.dim1, self.dim2, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=self.dim2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.dim2, self.dim2, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=self.dim2),
            nn.LeakyReLU(inplace=True),
            # nn.Dropout2d(0.3),
        )

        # Previous block output: (batch_size, 128, 16, 16)
        # Max pooling: (batch_size, 128, 8, 8)
        # Output image: (batch_size, 256, 8, 8)
        self.block3 = nn.Sequential(
            nn.Conv2d(self.dim2, self.dim3, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=self.dim3),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.dim3, self.dim3, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=self.dim3),
            nn.LeakyReLU(inplace=True),
            # nn.Dropout2d(0.3),
        )
        
        # Decoding blocks
        # Previous block output: (batch_size, 256, 8, 8)
        # Upconv: (batch_size, 128, 16, 16)
        # Concatenate with block2 output: (batch_size, 256, 16, 16)
        # Output image: (batch_size, 128, 16, 16)
        self.block4 = nn.Sequential(
            nn.Conv2d(self.dim3, self.dim2, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=self.dim2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.dim2, self.dim2, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=self.dim2),
            nn.LeakyReLU(inplace=True),
        )

        # Previous block output: (batch_size, 128, 16, 16)
        # Upconv: (batch_size, 64, 32, 32)
        # Concatenate with block1 output: (batch_size, 128, 32, 32)
        # Output image: (batch_size, 64, 32, 32)
        self.block5 = nn.Sequential(
            nn.Conv2d(self.dim2, self.dim1, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=self.dim1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.dim1, self.dim1, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=self.dim1),
            nn.LeakyReLU(inplace=True),
        )

        # Input image: (batch_size, 64, 32, 32)
        # Output image: (batch_size, 3, 32, 32)
        self.block6 = nn.Sequential(
            nn.Conv2d(self.dim1, self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=self.out_channels, num_channels=self.out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
            # nn.Tanh(), 
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Input image: (batch_size, 256, 8, 8)
        # Output image: (batch_size, 128, 16, 16)
        self.upconv1 = nn.ConvTranspose2d(self.dim3, self.dim2, kernel_size=2, stride=2)

        # Input image: (batch_size, 128, 16, 16)
        # Output image: (batch_size, 64, 32, 32)
        self.upconv2 = nn.ConvTranspose2d(self.dim2, self.dim1, kernel_size=2, stride=2)
        
        self.upconv3 = nn.Sequential(
            nn.Conv2d(self.dim1, self.dim1, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=self.dim1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.dim1, self.dim1, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=self.dim1),
            nn.LeakyReLU(inplace=True),
        )

        self.upconv4 = nn.Sequential(
            nn.Conv2d(self.dim1, self.dim1, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=self.dim1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.dim1, self.dim1, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=self.dim1),
            nn.LeakyReLU(inplace=True),
        )

        self.mlp1 = nn.Sequential(
            nn.Linear(1, self.dim3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.dim3, 1024)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(1, self.dim3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.dim3, 256),
        )

        self.mlp3 = nn.Sequential(
            nn.Linear(1, self.dim3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.dim3, 64)
        )

        def positional_encoding(x, dim, max_len=1000):
            # x: (batch_size, 1)
            
            x = x.repeat(1, dim) # (batch_size, dim)
            # apply sinusoidal encoding
            div_term = torch.exp(torch.arange(0, dim, 2).float() * -(np.log(10000.0) / dim))
            pos = torch.arange(0, max_len).unsqueeze(1).float() # (max_len, 1)
            pe = torch.zeros(max_len, dim).float()
            pe[:, 0::2] = torch.sin(pos * div_term)
            pe[:, 1::2] = torch.cos(pos * div_term)
            pe = pe.unsqueeze(0)
            return nn.Parameter(pe)
        
        self.pe1 = positional_encoding(torch.arange(0, 1000).unsqueeze(0), 1024) # (1, 1000, 1024)
        self.pe2 = positional_encoding(torch.arange(0, 1000).unsqueeze(0), 256) # (1, 1000, 256)
        self.pe3 = positional_encoding(torch.arange(0, 1000).unsqueeze(0), 64) # (1, 1000, 64)
        # self.init_weights()


    def forward(self, x, t):
        # Encoding
        t = t.float() # (batch_size, 1)

        enc1 = self.block1(x) # (batch_size, 64, 32, 32)
        tenc1 = self.mlp1(t) # (batch_size, 1024)
        tenc1 += self.pe1[:, t.long().squeeze(), :].squeeze(0) # (batch_size, 1024)
        tenc1 = tenc1.view(-1, 1, 32, 32) # (batch_size, 1, 32, 32)
        enc1 = enc1 + tenc1

        enc2 = self.block2(self.maxpool(enc1)) # (batch_size, 128, 16, 16)
        tenc2 = self.mlp2(t)
        tenc2 += self.pe2[:, t.long().squeeze(), :].squeeze(0) # (batch_size, 256)
        tenc2 = tenc2.view(-1, 1, 16, 16)
        enc2 = enc2 + tenc2

        enc3 = self.block3(self.maxpool(enc2)) # (batch_size, 256, 8, 8)
        tenc3 = self.mlp3(t)
        tenc3 += self.pe3[:, t.long().squeeze(), :].squeeze(0) # (batch_size, 64)
        tenc3 = tenc3.view(-1, 1, 8, 8)
        enc3 = enc3 + tenc3

        # Decoding
        dec1 = self.upconv1(enc3) # (batch_size, 128, 16, 16)
        dec1 = torch.cat((dec1, enc2), dim=1) # (batch_size, 256, 16, 16)
        dec1 = self.block4(dec1) # (batch_size, 128, 16, 16)

        dec2 = self.upconv2(dec1) # (batch_size, 64, 32, 32)
        dec2 = torch.cat((dec2, enc1), dim=1) # (batch_size, 128, 32, 32)
        dec2 = self.block5(dec2) # (batch_size, 64, 32, 32)

        dec2 = self.upconv3(dec2) # (batch_size, 64, 32, 32)
        dec2 = self.upconv4(dec2) # (batch_size, 64, 32, 32)

        out = self.block6(dec2) # (batch_size, 3, 32, 32)
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


