Diffusion
=========

A minimal diffusion (DDPM) training package with a UNet backbone, dataset loaders, and a simple CLI.

Features
- UNet with FiLM conditioning and optional attention.
- Config-driven training via YAML.
- Optional Weights & Biases logging (install extra and enable with a flag).
- Dataset loaders for MNIST, CIFAR-10, CelebA, Flowers102.

Installation
- From source (inside this folder):
  - python -m pip install .
  - To enable W&B logging: python -m pip install .[wandb]

Quick Start
- Prepare a config yaml, e.g. `config/cifar10.yaml`.
- Run training:
  - diffusion-train --config config/cifar10.yaml --dataset_root ./data --use_wandb

Config Schema
- dataset: One of mnist, cifar10, celeba, flowers102
- in_channels/out_channels: Usually 3 for RGB, 1 for grayscale
- layers: Channel sizes per UNet stage (e.g. [32,64,128,256])
- attention_layers: Which channel sizes to apply attention at (subset of layers)
- conditioning_dim: Time embedding dim (e.g. 128)
- dropout: Dropout rate (e.g. 0.1)
- num_heads: Attention heads
- diffusion_steps: Number of timesteps (e.g. 1000)
- diffusion_schedule: linear or cosine
- ema_decay: EMA decay (e.g. 0.9999)
- log_interval/save_interval: Logging and checkpoint cadence
- model_save_path: Directory to store checkpoints
- use_wandb: Optional; set true to enable logging (or pass --use_wandb)

Notes
- Set DATASET_DIR env var or pass --dataset_root to control dataset location.
- Requires PyTorch with CUDA for GPU training if available.
