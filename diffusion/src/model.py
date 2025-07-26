import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_components import FiLMGroupNorm, DoubleConv, Down, Up, AttentionBlock
import math

class UNet(nn.Module):
    """
    A UNet model for image processing tasks.

    Args:
        layers (tuple[int,...]): Number of channels in each layer.
        attention_layers (tuple[int,...]): Channel dimensions where attention blocks are applied.
        dropout (float): Dropout rate for the model.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            layers: tuple[int,...],
            attention_layers: tuple[int,...],
            conditioning_dim: int,
            dropout: float = 0.0,
            num_heads: int = 8
    ):
        """
        Initialize the UNet model.
        """
        super(UNet, self).__init__()
        self.layers = layers
        self.attention_layers = attention_layers
        self.dropout = dropout
        self.num_heads = num_heads
        self.conditioning_dim = conditioning_dim

        assert len(layers) > 1, "UNet must have at least two layers."
        assert all(layer in layers for layer in attention_layers), "Dimensions for attention layers must be in the layers tuple."


        # Input and output convolutions
        self.input_conv = DoubleConv(in_channels, layers[0], layers[0], conditioning_dim, dropout)
        self.output_conv = nn.Conv2d(layers[0], out_channels, kernel_size=1)

        self.down_blocks = nn.ModuleList(
            Down(
                in_channels=layers[i],
                out_channels=layers[i + 1],
                mid_channels=layers[i + 1],
                conditioning_dim=conditioning_dim,
                dropout=dropout
            ) for i in range(len(layers) - 1)
        )
        self.up_blocks = nn.ModuleList(
            Up(
                in_channels=layers[i],
                out_channels=layers[i - 1],
                mid_channels=layers[i - 1],
                conditioning_dim=conditioning_dim,
                dropout=dropout
            ) for i in range(len(layers) - 1, 0, -1)
        )
        
        self.attention_blocks = nn.ModuleList()

        down_attention_blocks = [i for i in attention_layers if i in layers]
        up_attention_blocks = [i for i in attention_layers if i in layers[:-1]][::-1] # Reverse for up blocks
        
        for channels in down_attention_blocks:
            self.attention_blocks.append(
                AttentionBlock(
                    channels=channels,
                    num_heads=num_heads
                )
            )
        
        for channels in up_attention_blocks:
            self.attention_blocks.append(
                AttentionBlock(
                    channels=channels,
                    num_heads=num_heads
                )
            )
        
        # Store counts for indexing
        self.num_down_attention = len(down_attention_blocks)
        self.down_attention_channels = down_attention_blocks
        self.up_attention_channels = up_attention_blocks



    def time_embedding(self, time: torch.Tensor, conditioning_dim: int) -> torch.Tensor:
        """
        Embeds time steps into a higher dimensional space.

        Args:
            time (torch.Tensor): Time steps tensor of shape (B,).
            conditioning_dim (int): Dimension of the conditioning vector.
        
        Returns:
            torch.Tensor: Embedded time tensor of shape (B, conditioning_dim).
        """
        half_dim = conditioning_dim // 2
        emb = torch.exp(
            -torch.arange(half_dim, dtype=torch.float32, device=time.device) * (math.log(10000.0) / half_dim)
        )
        emb = time[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the UNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            time (torch.Tensor): Time conditioning tensor of shape (B,)
        
        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W).
        """

        time_embedding = self.time_embedding(time, self.conditioning_dim)

        x = self.input_conv(x, time_embedding)
        current_channel_index = 0
        down_attention_index = 0

        # Apply attention after input conv if needed (down path)
        current_channels = self.layers[current_channel_index]
        if current_channels in self.down_attention_channels:
            x = self.attention_blocks[down_attention_index](x)
            down_attention_index += 1

        skip_connections = [x]
        for down_block in self.down_blocks:
            x = down_block(x, time_embedding)
            current_channel_index += 1
            current_channels = self.layers[current_channel_index]
            if current_channels in self.down_attention_channels:
                x = self.attention_blocks[down_attention_index](x)
                down_attention_index += 1
            skip_connections.append(x)

        up_attention_index = self.num_down_attention
        
        for i, up_block in enumerate(self.up_blocks):
            skip_connection = skip_connections[-(i + 2)]
            x = up_block(x, skip_connection, time_embedding)
            current_channel_index -= 1
            current_channels = self.layers[current_channel_index]
            if current_channels in self.up_attention_channels:
                x = self.attention_blocks[up_attention_index](x)
                up_attention_index += 1

        return self.output_conv(x)