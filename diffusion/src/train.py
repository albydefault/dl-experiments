import torch
import copy
from tqdm import tqdm
import wandb

from .utils import denormalize_image_tensor

def train(model: torch.nn.Module,
          train_loader: torch.utils.data.DataLoader,
          test_loader: torch.utils.data.DataLoader | None,
          optimizer: torch.optim.Optimizer,
          criterion: torch.nn.Module,
          scheduler: torch.optim.lr_scheduler.LRScheduler | None,
          epochs: int,
          diffusion_steps: int,
          diffusion_schedule: dict[str, torch.Tensor],
          ema_decay: float,
          device: torch.device,
          config: dict) -> torch.nn.Module:

    """
    Trains the provided diffusion model.

    Args:
        model (torch.nn.Module): The diffusion model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        test_loader (torch.utils.data.DataLoader | None): DataLoader for the test dataset,
            or None if no test dataset is provided.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion (torch.nn.Module): Loss function to use for training.
        scheduler (torch.optim.lr_scheduler.LRScheduler | None): Learning rate scheduler,
            or None if no scheduler is used.
        epochs (int): Number of training epochs.
        diffusion_steps (int): Number of diffusion steps.
        diffusion_schedule (dict[str, torch.Tensor]): Dictionary containing diffusion schedule tensors.
        ema_decay (float): Exponential moving average decay factor.
        device (torch.device): Device to run the training on (CPU or GPU).
        config (dict): Configuration dictionary containing various training parameters.

    Returns:
        torch.nn.Module: The trained diffusion model.

    Notes:
        - The function initializes Weights & Biases (wandb) for logging.
        - Expects the `diffusion_schedule` dictionary to be structured with keys:
            diffusion_schedule = {
                'betas': torch.Tensor,                              # β values for each timestep
                'cumulative_alphas': torch.Tensor,                  # ∏(1-β) cumulative product
                'sqrt_cumulative_alphas': torch.Tensor,             # √(cumulative_alphas)
                'sqrt_one_minus_cumulative_alphas': torch.Tensor,   # √(1 - cumulative_alphas)
                'sqrt_alphas': torch.Tensor,                        # √α = √(1-β)
            }
    """

    with wandb.init(project=config['project_name'], config=config) as run:
        global_step = 0
        
        sqrt_cumulative_alphas = diffusion_schedule['sqrt_cumulative_alphas'].to(device)
        sqrt_one_minus_cumulative_alphas = diffusion_schedule['sqrt_one_minus_cumulative_alphas'].to(device)

        sqrt_cumulative_alphas = sqrt_cumulative_alphas.reshape(-1, 1, 1, 1)
        sqrt_one_minus_cumulative_alphas = sqrt_one_minus_cumulative_alphas.reshape(-1, 1, 1, 1)


        ema_model = copy.deepcopy(model)
        ema_model.eval()
        ema_model.requires_grad_(False)

        parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Training model with {parameter_count} trainable parameters.')

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

            for i, (images, _) in enumerate(tqdm(train_loader)):
                images = images.to(device)

                t = torch.randint(0, diffusion_steps, (images.size(0),), device=device)
                
                noise = torch.randn_like(images)
                noisy_images = (
                    sqrt_cumulative_alphas[t] * images +
                    sqrt_one_minus_cumulative_alphas[t] * noise
                )
                predicted_noise = model(noisy_images, t)

                loss = criterion(predicted_noise, noise)
                optimizer.zero_grad()
                loss.backward()

                per_sample_loss = torch.nn.functional.mse_loss(predicted_noise, noise, reduction='none').mean(dim=[1, 2, 3])

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
                

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                with torch.no_grad():
                    for param, ema_param in zip(model.parameters(), ema_model.parameters()):
                        ema_param.data = ema_decay * ema_param.data + (1 - ema_decay) * param.data

                running_loss += loss.item()

                # LOGS
                run.log({
                    'train_loss': loss.item(),
                    'global_step': global_step,
                    'lr': scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr'],
                    'grad_norm': grad_norm.item(),
                    'avg_timestep': t.float().mean().item(),  # Average timestep being trained
                })

                run.log({
                    'loss_timestep_early': per_sample_loss[t < diffusion_steps // 4].mean().item() if (t<diffusion_steps // 4).any() else None,
                    'loss_timestep_mid_early': per_sample_loss[(t >= diffusion_steps // 4) & (t < diffusion_steps // 2)].mean().item() if ((t >= diffusion_steps // 4) & (t < diffusion_steps // 2)).any() else None,
                    'loss_timestep_mid_late': per_sample_loss[(t >= diffusion_steps // 2) & (t < diffusion_steps * 3 // 4)].mean().item() if ((t >= diffusion_steps // 2) & (t < diffusion_steps * 3 // 4)).any() else None,
                    'loss_timestep_late': per_sample_loss[t >= diffusion_steps * 3 // 4].mean().item() if (t >= diffusion_steps * 3 // 4).any() else None,
                })
                
                if global_step % config['log_interval'] == 0:
                    # One step denoising for visualization
                    with torch.no_grad():
                        predicted_images = (noisy_images - sqrt_one_minus_cumulative_alphas[t] * predicted_noise) / sqrt_cumulative_alphas[t]
                    
                        ema_predicted_noise = ema_model(noisy_images, t)
                        ema_predicted_images = (noisy_images - sqrt_one_minus_cumulative_alphas[t] * ema_predicted_noise) / sqrt_cumulative_alphas[t]
                        ema_loss = criterion(ema_predicted_noise, noise)

                    # Convert images to proper format for wandb logging
                    def format_images_for_wandb(img_tensor):
                        """Convert batch of images to format suitable for wandb.Image"""
                        # Denormalize and convert to CPU
                        img_tensor = denormalize_image_tensor(img_tensor).cpu()
                        # Convert from [B, C, H, W] to [B, H, W, C]
                        img_tensor = img_tensor.permute(0, 2, 3, 1)
                        # Convert to numpy and ensure proper range [0, 1]
                        img_array = img_tensor.numpy().clip(0, 1)
                        return img_array

                    run.log({
                        'original_images': [wandb.Image(img) for img in format_images_for_wandb(images[:8])],
                        'noisy_images': [wandb.Image(img) for img in format_images_for_wandb(noisy_images[:8])],
                        'denoised_images': [wandb.Image(img) for img in format_images_for_wandb(predicted_images[:8])],
                        'ema_denoised_images': [wandb.Image(img) for img in format_images_for_wandb(ema_predicted_images[:8])],
                        'ema_loss': ema_loss.item(),
                    })

                    if test_loader is not None:
                        model_test_loss = test(model, test_loader, criterion, diffusion_steps, diffusion_schedule, device)
                        model.train()
                        
                        ema_model_test_loss = test(ema_model, test_loader, criterion, diffusion_steps, diffusion_schedule, device)

                        run.log({
                            'model_test_loss': model_test_loss,
                            'ema_model_test_loss': ema_model_test_loss,
                        })


                if global_step % (config['log_interval'] * 10) == 0:
                    param_norm = sum(p.norm().item() for p in model.parameters())
                    run.log({'param_norm': param_norm})

                global_step += 1

            epoch_loss = running_loss / len(train_loader)
            run.log({'epoch_loss': epoch_loss, 'epoch': epoch})
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}')

            if epoch % config.get('save_interval', 100) == 0:

                torch.save(model.state_dict(), f'{config.get("model_save_path", ".")}/model_epoch_{epoch + 1}.pth')
                torch.save(ema_model.state_dict(), f'{config.get("model_save_path", ".")}/ema_model_epoch_{epoch + 1}.pth')
                print(f'Models saved for epoch {epoch + 1}')
    
        # Save the final model
        torch.save(model.state_dict(), f'{config.get("model_save_path", ".")}/final_model.pth')
        torch.save(ema_model.state_dict(), f'{config.get("model_save_path", ".")}/final_ema_model.pth')
        print('Final models saved.')
    return model

def test(model: torch.nn.Module,
         test_loader: torch.utils.data.DataLoader,
         criterion: torch.nn.Module,
         diffusion_steps: int,
         diffusion_schedule: dict[str, torch.Tensor],
         device: torch.device) -> float:
    """
    Evaluates the model on the test dataset.
    """
    model.eval()
    
    sqrt_cumulative_alphas = diffusion_schedule['sqrt_cumulative_alphas'].to(device).reshape(-1, 1, 1, 1)
    sqrt_one_minus_cumulative_alphas = diffusion_schedule['sqrt_one_minus_cumulative_alphas'].to(device).reshape(-1, 1, 1, 1)

    with torch.no_grad():
        running_loss = 0.0
        for images, _ in test_loader:
            images = images.to(device)

            t = torch.randint(0, diffusion_steps, (images.size(0),), device=device)

            noise = torch.randn_like(images)
            noisy_images = (
                sqrt_cumulative_alphas[t] * images +
                sqrt_one_minus_cumulative_alphas[t] * noise
            )
            predicted_noise = model(noisy_images, t)

            loss = criterion(predicted_noise, noise)
            running_loss += loss.item()

    return running_loss / len(test_loader)


def sample_images(
        model: torch.nn.Module,
        diffusion_steps: int,
        diffusion_schedule: dict[str, torch.Tensor],
        device: torch.device,
        num_samples: int,
        image_size: tuple[int, int, int] = (3, 32, 32)
) -> torch.Tensor:
    B, C, H, W = num_samples, *image_size
    
    # CHANGE 1: Extract all needed schedule components upfront
    betas = diffusion_schedule['betas'].to(device)
    cumulative_alphas = diffusion_schedule['cumulative_alphas'].to(device)
    sqrt_cumulative_alphas = diffusion_schedule['sqrt_cumulative_alphas'].to(device)
    sqrt_one_minus_cumulative_alphas = diffusion_schedule['sqrt_one_minus_cumulative_alphas'].to(device)
    
    # CHANGE 2: Precompute coefficients for numerical stability (like reference implementations)
    alphas = 1.0 - betas
    sqrt_alphas = torch.sqrt(alphas.clamp(min=1e-8))
    
    # CHANGE 3: Precompute mean coefficients (two-coefficient method)
    alpha_bar_prev = torch.cat([torch.ones(1, device=device), cumulative_alphas[:-1]])
    mean_x0_coef = betas * torch.sqrt(alpha_bar_prev.clamp(min=1e-8)) / (1.0 - cumulative_alphas).clamp(min=1e-8)
    mean_xt_coef = (1.0 - alpha_bar_prev) * sqrt_alphas / (1.0 - cumulative_alphas).clamp(min=1e-8)
    
    # CHANGE 4: Precompute variance properly
    posterior_variance = betas * (1.0 - alpha_bar_prev) / (1.0 - cumulative_alphas).clamp(min=1e-8)
    posterior_variance = torch.clamp(posterior_variance, min=1e-20, max=0.9999)
    log_posterior_variance = torch.log(posterior_variance)

    model.eval()
    with torch.no_grad():
        sample = torch.randn(num_samples, *image_size, device=device)

        for i in reversed(range(diffusion_steps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            
            # CHANGE 5: Cleaner indexing - use scalar indexing since i is a single timestep
            alpha_bar_t = cumulative_alphas[i]
            sqrt_alpha_bar_t = sqrt_cumulative_alphas[i]
            sqrt_one_minus_alpha_bar_t = sqrt_one_minus_cumulative_alphas[i]
            
            predicted_noise = model(sample, t)
            
            # CHANGE 6: Use the reference implementation approach - predict x0 first
            sqrt_recip_alpha_bar = 1.0 / sqrt_alpha_bar_t.clamp(min=1e-8)
            sqrt_recip_m1_alpha_bar = torch.sqrt((1.0 / alpha_bar_t.clamp(min=1e-8)) - 1.0)
            x0_pred = sqrt_recip_alpha_bar * sample - sqrt_recip_m1_alpha_bar * predicted_noise
            
            # CHANGE 7: Clamp x0 prediction to prevent explosion
            x0_pred = torch.clamp(x0_pred, -10.0, 10.0)
            
            # CHANGE 8: Use two-coefficient method for mean (more stable)
            mean = mean_x0_coef[i] * x0_pred + mean_xt_coef[i] * sample
            
            # CHANGE 9: Simplified noise addition with precomputed log variance
            if i > 0:
                noise = torch.randn_like(sample)
                std = torch.exp(0.5 * log_posterior_variance[i])
                sample = mean + std * noise
            else:
                sample = mean
            
            # CHANGE 10: Aggressive clamping to prevent explosion
            sample = torch.clamp(sample, -50.0, 50.0)

    return sample.clamp(-1.0, 1.0)