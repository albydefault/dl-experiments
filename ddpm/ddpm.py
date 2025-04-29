# Standard library imports

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import CosineAnnealingLR


import numpy as np
import matplotlib.pyplot as plt
import datetime

# Local application imports
from model import UNet32 as UNet
### Data preparation
DATASET_DIR = "/home/alazar/desktop/datasets"

cifar10_dataset = CIFAR10(
    root=DATASET_DIR,
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
)

mnist_dataset = MNIST(
    root=DATASET_DIR,
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.Normalize((0.5,), (0.5,)),
    ])
)

cifar10_dataloader = DataLoader(
    cifar10_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

mnist_dataloader = DataLoader(
    mnist_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

### Model definition
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(latent_dim=512, in_channels=1, out_channels=1).to(device)
model.init_weights()
# optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2)
scheduler = CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
criterion = nn.MSELoss()
# criterion = nn.L1Loss()
# criterion = nn.SmoothL1Loss()


# Trainer
def cosine_beta_schedule(timesteps, s=0.008):
    steps = torch.arange(timesteps + 1, dtype=torch.float32)
    alphas_cumprod = torch.cos(((steps / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod /= alphas_cumprod[0].clone()
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-5, 0.999)

STEP_COUNT = 1000
betas = cosine_beta_schedule(STEP_COUNT, s=0.008)
# betas = (betas - betas.min()) / (betas.max() - betas.min())
# betas = betas * torch.pi / 2
# betas = torch.sin(betas)
# betas = betas * (0.02 - 1e-4) + 1e-4

alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

betas = betas.to(device)
alphas = alphas.to(device)
alphas_cumprod = alphas_cumprod.to(device)

sqrt_acum = alphas_cumprod.sqrt()
sqrt_1macum = (1-alphas_cumprod).sqrt()


# exp_name = f"ddpm_cifar10_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
exp_name = f"ddpm_mnist_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
writer = SummaryWriter(log_dir=f"runs/{exp_name}")

hparams = {"lr": 1e-4, "batch_size": 64}



def train(model, dataloader, optimizer, criterion, num_epochs=20):
    global_step = 0
    
    for epoch in range(num_epochs):
        running_loss = 0.0

        for i, (images, _) in enumerate(tqdm(dataloader)):
            model.train()
            images = images.to(device)

            # Sample random time steps
            t = torch.randint(0, STEP_COUNT, (images.size(0),1), device=device)
            noise = torch.randn_like(images)

            # Compute the noisy images
            noisy_images = sqrt_acum[t].view(-1, 1, 1, 1) * images + sqrt_1macum[t].view(-1, 1, 1, 1) * noise
            noisy_images = noisy_images.to(device)

            # Compute the predicted noise
            predicted_noise = model(noisy_images, t)

            # Compute the loss
            loss = criterion(predicted_noise, noise) / sqrt_1macum[t].view(-1, 1, 1, 1).mean()
            optimizer.zero_grad()
            loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()

            writer.add_scalar("loss/train_step", loss.item(), global_step)
            if global_step % 500 == 0:
                writer.add_images("noisy_images", (1 + noisy_images[:8].clamp(-1, 1)) / 2, global_step, dataformats='NCHW')
                one_minus_acum = sqrt_1macum[t].view(-1, 1, 1, 1)
                acum = sqrt_acum[t].view(-1, 1, 1, 1)
                predicted_images = (noisy_images - one_minus_acum * predicted_noise) / acum
                writer.add_images("predicted_images", (1 + predicted_images[:8].clamp(-1, 1)) / 2, global_step, dataformats='NCHW')

                with torch.no_grad():
                    samples = sample(model, img_size=images.size()[1:], writer=writer, global_step=global_step)
                    samples = (samples.clamp(-1, 1) + 1) / 2
                    grid = make_grid(samples, nrow=4, normalize=True)
                    writer.add_image("generated_images", grid, global_step)

            global_step += 1
        
        writer.add_scalar("lr/train_step", scheduler.get_last_lr()[0], global_step)
        scheduler.step()

        epoch_loss = running_loss / len(dataloader)
        writer.add_scalar("loss/train_epoch", epoch_loss, epoch)
        # writer.add_hparams(hparams, {"hparam/loss": epoch_loss})
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # Save the model every 5 epochs 
        # if (epoch + 1) % 5 == 0:
        #     torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")
        #     print(f"Model saved at epoch {epoch + 1}")

    # Save the final model  
    # torch.save(model.state_dict(), "model_final.pth")
    # writer.add_hparams(hparams, {"hparam/loss": epoch_loss})

    writer.close()
    print("Training complete. Model saved.")
    return model
### Sampling
def sample(model, img_size=(3, 32, 32), writer=None, num_images=8, global_step=0):
    model.eval()
    with torch.no_grad():
        # Start with random noise
        x = torch.randn(num_images, *img_size).to(device)

        for t in range(STEP_COUNT-1, -1, -1):
            # Compute the predicted noise
            t_tensor = torch.full((num_images,1), t, device=device, dtype=torch.long)
            predicted_noise = model(x, t_tensor)
            x_mean = (x - (1 - alphas[t]) * predicted_noise / sqrt_1macum[t]) / torch.sqrt(alphas[t])

            if t > 0:
                posterior_var = betas[t] * (1 - alphas_cumprod[t-1])/(1 - alphas_cumprod[t])
                sigma = torch.sqrt(posterior_var + 1e-20)

                x = x_mean + sigma * torch.randn_like(x)
            else:
                x = x_mean
                        
        return x

# Sample and visualize the generated image
# Training the model
sample_input = torch.randn(1,1,32,32,device=device)
# sample_input = torch.randn(1, 3, 32, 32, device=device)
writer.add_graph(model, (sample_input, torch.tensor([0],device=device)))

model = train(model, mnist_dataloader, optimizer, criterion, num_epochs=200)
