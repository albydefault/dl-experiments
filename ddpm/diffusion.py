import torch
from tqdm import tqdm
import wandb
from typing import Dict, Tuple
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Schedule for betas
def cosine_beta_schedule(timesteps, s=0.008)-> torch.Tensor:
    steps = torch.arange(timesteps + 1, dtype=torch.float32)
    alphas_cumprod = torch.cos(((steps / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod /= alphas_cumprod[0].clone()
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-5, 0.999)

def make_diffusion_schedule(step_count, device) -> Dict[str, torch.Tensor]:
    betas = cosine_beta_schedule(step_count)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    return {
        'betas': betas.to(device),
        'alphas': alphas.to(device),
        'alphas_cumprod': alphas_cumprod.to(device),
        'sqrt_acum': alphas_cumprod.sqrt().to(device),
        'sqrt_1macum': (1 - alphas_cumprod).sqrt().to(device)
    }


def train(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, scheduler: torch.optim.lr_scheduler._LRScheduler, num_epochs: int, step_count: int, schedule: Dict[str, torch.Tensor]) -> torch.nn.Module:
    global_step = 0

    # Extract schedule values more safely
    betas = schedule['betas']
    alphas = schedule['alphas']
    alphas_cumprod = schedule['alphas_cumprod']
    sqrt_acum = schedule['sqrt_acum']
    sqrt_1macum = schedule['sqrt_1macum']

    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(num_epochs):
        running_loss = 0.0

        for i, (images, _) in enumerate(tqdm(dataloader)):
            model.train()
            images = images.to(device)

            # Sample random time steps
            t = torch.randint(0, step_count, (images.size(0),), device=device)
            noise = torch.randn_like(images)

            # Add noise to images according to the diffusion schedule
            noisy_images = sqrt_acum[t].reshape(-1, 1, 1, 1) * images + sqrt_1macum[t].reshape(-1, 1, 1, 1) * noise
            predicted_noise = model(noisy_images, t.unsqueeze(1))

            loss = criterion(predicted_noise, noise)
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Update EMA model
            decay = 0.9999
            with torch.no_grad():
                for param, ema_param in zip(model.parameters(), ema_model.parameters()):
                    ema_param.data = 0.9999 * ema_param.data + 0.0001 * param.data

            running_loss += loss.item()

            wandb.log({"loss/training_step": loss.item()}, step=global_step)
            if global_step % 500 == 0:
                wandb.log({
                    "noisy_images": wandb.Image((1 + noisy_images[:8].clamp(-1, 1)) / 2),
                }, step=global_step)

                # Calculate predicted original images for visualization
                reshaped_sqrt_1macum = sqrt_1macum[t].reshape(-1, 1, 1, 1)
                reshaped_sqrt_acum = sqrt_acum[t].reshape(-1, 1, 1, 1)
                predicted_images = (noisy_images - reshaped_sqrt_1macum * predicted_noise) / reshaped_sqrt_acum

                wandb.log({
                    "predicted_images": wandb.Image((1 + predicted_images[:8].clamp(-1, 1)) / 2),
                }, step=global_step)

                with torch.no_grad():
                    samples = sample(model=ema_model, schedule=schedule, step_count=step_count, img_size=images.size()[1:])
                    samples = (samples.clamp(-1, 1) + 1) / 2
                    
                    wandb.log({
                        "generated_images": wandb.Image(samples),
                    }, step=global_step)

            global_step += 1
        
        wandb.log({
            "lr/train_step": scheduler.get_last_lr()[0],
        }, step=global_step)
        scheduler.step()

        epoch_loss = running_loss / len(dataloader)
        wandb.log({"loss/train_epoch": epoch_loss}, step=global_step)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # Save the model every 5 epochs 
        # if (epoch + 1) % 5 == 0:
        #     torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")
        #     print(f"Model saved at epoch {epoch + 1}")

    # Save the final model  
    wandb.finish()
    print("Training complete.")
    return model

### Sampling
def sample(model, schedule, step_count, img_size=(3, 32, 32), num_images=8,):
    model.eval()
    betas = schedule['betas']
    alphas = schedule['alphas']
    alphas_cumprod = schedule['alphas_cumprod']
    sqrt_acum = schedule['sqrt_acum']
    sqrt_1macum = schedule['sqrt_1macum']

    sqrt_alphas = torch.sqrt(alphas)
    with torch.no_grad():
        # Start with random noise
        x = torch.randn(num_images, *img_size).to(device)

        for t in range(step_count-1, -1, -1):
            # Compute the predicted noise
            t_tensor = torch.full((num_images,), t, device=device, dtype=torch.long)
            predicted_noise = model(x, t_tensor.unsqueeze(1))
            
            # Calculate mean of the previous step
            x_mean = (x - (1 - alphas[t]) * predicted_noise / sqrt_1macum[t]) / sqrt_alphas[t]

            if t > 1:
                # Calculate posterior variance for adding noise
                posterior_var = betas[t] * (1 - alphas_cumprod[t-1])/(1 - alphas_cumprod[t])
                sigma = torch.sqrt(posterior_var + 1e-20)  # Add small epsilon to prevent numerical issues

                x = x_mean + sigma * torch.randn_like(x)
                # Ensure x is within the range [-1, 1]
                x = torch.clamp(x, -1, 1)
            else:
                x = x_mean
                        
        return x
