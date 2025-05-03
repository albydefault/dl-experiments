import torch
from tqdm import tqdm
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Schedule for betas
def cosine_beta_schedule(timesteps, s=0.008):
    steps = torch.arange(timesteps + 1, dtype=torch.float32)
    alphas_cumprod = torch.cos(((steps / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod /= alphas_cumprod[0].clone()
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-5, 0.999)

def make_diffusion_schedule(step_count, device):
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


def train(model, dataloader, optimizer, criterion, scheduler, num_epochs, step_count, schedule):
    global_step = 0
    
    _, _, _, sqrt_acum, sqrt_1macum = schedule.values()

    
    for epoch in range(num_epochs):
        running_loss = 0.0

        for i, (images, _) in enumerate(tqdm(dataloader)):
            model.train()
            images = images.to(device)

            # Sample random time steps
            t = torch.randint(0, step_count, (images.size(0),1), device=device)
            noise = torch.randn_like(images)


            noisy_images = sqrt_acum[t].view(-1, 1, 1, 1) * images + sqrt_1macum[t].view(-1, 1, 1, 1) * noise
            predicted_noise = model(noisy_images, t)

            loss = criterion(predicted_noise, noise)
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()

            wandb.log({"loss/training_step": loss.item()}, step=global_step)
            if global_step % 500 == 0:
                wandb.log({
                    "noisy_images": wandb.Image((1 + noisy_images[:8].clamp(-1, 1)) / 2),
                }, step=global_step)

                
                reshaped_sqrt_1macum = sqrt_1macum[t].view(-1, 1, 1, 1)
                reshaped_sqrt_acum = sqrt_acum[t].view(-1, 1, 1, 1)
                predicted_images = (noisy_images - reshaped_sqrt_1macum * predicted_noise) / reshaped_sqrt_acum

                wandb.log({
                    "predicted_images": wandb.Image((1 + predicted_images[:8].clamp(-1, 1)) / 2),
                }, step=global_step)

                with torch.no_grad():
                    samples = sample(model=model, schedule=schedule, step_count=step_count, img_size=images.size()[1:])
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
            t_tensor = torch.full((num_images,1), t, device=device, dtype=torch.long)
            predicted_noise = model(x, t_tensor)
            x_mean = (x - (1 - alphas[t]) * predicted_noise / sqrt_1macum[t]) / sqrt_alphas[t]

            if t > 0:
                posterior_var = betas[t] * (1 - alphas_cumprod[t-1])/(1 - alphas_cumprod[t])
                sigma = torch.sqrt(posterior_var + 1e-20)

                x = x_mean + sigma * torch.randn_like(x)
            else:
                x = x_mean
                        
        return x
