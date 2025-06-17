import yaml, argparse
from pathlib import Path
import datetime
import torch
import torch.nn as nn
import numpy as np
from diffusion import make_diffusion_schedule, train

def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    

# Argparse for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="Path to the config file", default="config/cifar10.yaml")
parser.add_argument("--exp_name", type=str, help="Experiment name (will be auto-generated if not provided)", default=None)
parser.add_argument("--exp_type", type=str, help="Experiment type (e.g., baseline, ablation, tuning)", default="baseline")
parser.add_argument("--epochs", type=int, help="Number of epochs", default=200)
parser.add_argument("--batch_size", type=int, help="Batch size", default=128)
parser.add_argument("--lr", type=float, help="Learning rate", default=5e-5)
parser.add_argument("--latent_dim", type=int, help="Latent dimension", default=512)
parser.add_argument("--step_count", type=int, help="Number of steps", default=1000)
parser.add_argument("--model", type=str, help="Model name", default="unet32")
parser.add_argument("--device", type=str, help="Device to use", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--seed", type=int, help="Random seed", default=42)
parser.add_argument("--size", type=int, help="Image size", default=32)
parser.add_argument("--notes", type=str, help="Additional notes about this experiment", default="")

args = parser.parse_args()
config = load_yaml(args.config)

# Generate experiment name if not provided
if args.exp_name is None:
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    lr_str = f"lr{args.lr:.0e}".replace("-", "")  # Convert 5e-5 to lr5e5
    args.exp_name = f"{args.model}_{config['dataset']}_bs{args.batch_size}_{lr_str}_{args.exp_type}_{timestamp}"

config["exp_name"] = args.exp_name
config["latent_dim"] = args.latent_dim
config["step_count"] = args.step_count
config["epochs"] = args.epochs
config["batch_size"] = args.batch_size
config["lr"] = args.lr

if args.device.isdigit():
    torch.cuda.set_device(int(args.device))
    device = torch.device("cuda")
elif args.device == "cuda" and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Initialize random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Model and optimizer
from model import *
model_cls = MODEL_REGISTRY[args.model]
model = model_cls(
    in_channels=config["channels"],
    out_channels=config["channels"],
    latent_dim=config["latent_dim"],
    time_emb_dim=config["time_emb_dim"]
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=1e-6)
criterion = nn.MSELoss()

# Diffusion schedule
schedule = make_diffusion_schedule(config["step_count"], device=device)


# Dataset
from data import *
dataset_cls = DATALOADER_REGISTRY[config["dataset"]]
train_loader = dataset_cls(
    batch_size=config["batch_size"],
    resize=(args.size, args.size),
)

# Training loop
import wandb
wandb.init(
    project=config["project"],
    name=config["exp_name"],
    notes=args.notes,
    tags=[config["dataset"], args.model, args.exp_type],
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
        "image_size": args.size,
        "seed": args.seed,
        "exp_type": args.exp_type,
        "device": device,
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
    