import yaml, argparse
from pathlib import Path
import datetime
import torch
import torch.nn as nn
import numpy as np
from ddpm import make_diffusion_schedule, train

def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    

# Argparse for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="Path to the config file", default="config/cifar10.yaml")
parser.add_argument("--exp_name", type=str, help="Experiment name", default=f"_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
parser.add_argument("--epochs", type=int, help="Number of epochs", default=200)
parser.add_argument("--batch_size", type=int, help="Batch size", default=128)
parser.add_argument("--lr", type=float, help="Learning rate", default=5e-5)
parser.add_argument("--latent_dim", type=int, help="Latent dimension", default=512)
parser.add_argument("--step_count", type=int, help="Number of steps", default=1000)
parser.add_argument("--model", type=str, help="Model name", default="unet32")
parser.add_argument("--device", type=str, help="Device to use", default="cuda" if torch.cuda.is_available() else "cpu")

args = parser.parse_args()
config = load_yaml(args.config)
config["exp_name"] = f"{config['dataset']}_{args.model}_{args.exp_name}"
config["latent_dim"] = args.latent_dim
config["step_count"] = args.step_count
config["epochs"] = args.epochs
config["batch_size"] = args.batch_size
config["lr"] = args.lr

# Model and optimizer
from model import *
model_cls = MODEL_REGISTRY[args.model]
model = model_cls(
    in_channels=config["channels"],
    out_channels=config["channels"],
    latent_dim=config["latent_dim"],
    time_emb_dim=config["time_emb_dim"]
).to(args.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=1e-6)
criterion = nn.MSELoss()

# Diffusion schedule
schedule = make_diffusion_schedule(config["step_count"], device=args.device)


# Dataset
from data import *
dataset_cls = DATALOADER_REGISTRY[config["dataset"]]
train_loader = dataset_cls(
    batch_size=config["batch_size"]
)

# Training loop
import wandb
wandb.init(
    project=config["project"],
    name=config["exp_name"],
    config={
        "learning_rate": config["lr"],
        "batch_size": config["batch_size"],
        "epochs": config["epochs"],
        "step_count": config["step_count"],
        "beta_schedule": config["beta_schedule"],
        "optimizer": "AdamW",
        "criterion": "MSELoss",
        "weight_decay": 1e-2,
        "scheduler": "CosineAnnealingLR",
        "T_max": config["epochs"],
        "eta_min": 1e-6,
        "clip_grad_norm": 1.0,
        "dataset": config["dataset"],
        "model": args.model,
        "latent_dim": config["latent_dim"],
    }
)

if __name__ == "__main__":
    model = train(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        num_epochs=config["epochs"],
        step_count=config["step_count"],
        schedule=schedule
    )
    