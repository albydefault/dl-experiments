import argparse
from pathlib import Path
import datetime
import os

import torch
import torch.nn as nn
import numpy as np

from diffusion.utils import load_yaml_config, create_ddpm_schedule
from diffusion.train import train
from diffusion.data import make_data_loader
from diffusion.model import UNet




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('--exp_type', type=int, default=0, help='Type of experiment to run')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run the model on (e.g., cpu, cuda, or GPU index)')
    parser.add_argument('--dataset_root', type=str, default=os.environ.get('DATASET_DIR', './data'), help='Root directory for datasets')
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging if installed')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()
    config = load_yaml_config(args.config)

    config["exp_type"] = args.exp_type
    config["epochs"] = args.epochs
    config["batch_size"] = args.batch_size
    config["learning_rate"] = args.learning_rate
    config["device"] = args.device

    if args.device.isdigit():
        torch.cuda.set_device(int(args.device))
        device = torch.device("cuda")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = UNet(
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        layers=tuple(config['layers']),
        attention_layers=tuple(config['attention_layers']),
        conditioning_dim=config['conditioning_dim'],
        dropout=config['dropout'],
        num_heads=config['num_heads']
    ).to(device)

    train_loader, val_loader = make_data_loader(
        dataset_name=config['dataset'],
        batch_size=config['batch_size'],
        resize=config.get('resize', (32, 32)),
        root=args.dataset_root,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs * len(train_loader), 
        eta_min=1e-6
    ) if config.get('use_scheduler', True) else None

    diffusion_schedule = create_ddpm_schedule(
        timesteps=config['diffusion_steps'],
        schedule_type=config['diffusion_schedule']
    )

    model_save_path = Path(config['model_save_path']) / datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_save_path.mkdir(parents=True, exist_ok=True)
    config['model_save_path'] = str(model_save_path)

    train(
        model=model,
        train_loader=train_loader,
        test_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        epochs=args.epochs,
        diffusion_steps=config['diffusion_steps'],
        diffusion_schedule=diffusion_schedule,
        ema_decay=config['ema_decay'],
        device=device,
        config={
            **config,
            'use_wandb': bool(args.use_wandb or config.get('use_wandb', False)),
        }
    )

    # list files in the model save directory to delete the folder if empty
    if not any(model_save_path.iterdir()):
        model_save_path.rmdir()
    
if __name__ == "__main__":
    main()
