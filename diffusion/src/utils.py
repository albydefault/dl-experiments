import torch
import yaml


def load_yaml_config(config_path: str) -> dict:
    """
    Load a YAML configuration file.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config



def denormalize_image_tensor(image_tensor: torch.Tensor):
    """
    Denormalizes an image tensor by scaling it to the range [0, 1].
    """

    return image_tensor.clamp(-1, 1) * 0.5 + 0.5

def create_ddpm_schedule(timesteps: int = 1000, 
                         beta_start: float = 0.0001, 
                         beta_end: float = 0.02, 
                         schedule_type: str = "linear"):
    """
    Create a proper DDPM beta schedule.
    
    Args:
        timesteps: Number of diffusion timesteps (usually 1000)
        beta_start: Starting beta value (usually 0.0001)
        beta_end: Ending beta value (usually 0.02)
        schedule_type: "linear" or "cosine"
    
    Returns:
        Dictionary with all schedule tensors
    """
    
    if schedule_type == "linear":
        # Linear schedule from original DDPM paper
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
    
    elif schedule_type == "cosine":
        # Cosine schedule (often works better)
        def cosine_beta_schedule(timesteps, s=0.008):
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)
        
        betas = cosine_beta_schedule(timesteps)
    
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    # Ensure betas are in valid range
    betas = torch.clamp(betas, min=1e-6, max=0.9999)
    
    # Compute derived quantities
    alphas = 1.0 - betas
    cumulative_alphas = torch.cumprod(alphas, dim=0)
    
    # All the schedule components your code expects
    schedule = {
        'betas': betas,
        'alphas': alphas,
        'cumulative_alphas': cumulative_alphas,
        'sqrt_cumulative_alphas': torch.sqrt(cumulative_alphas),
        'sqrt_one_minus_cumulative_alphas': torch.sqrt(1.0 - cumulative_alphas),
        'sqrt_alphas': torch.sqrt(alphas),
    }
    
    return schedule
