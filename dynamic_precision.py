import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import numpy as np
from typing import Optional, Tuple
import logging

from src.utils.util import set_seeds

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PrecisionMetrics:
    """Tracks and analyzes precision changes"""
    def __init__(self):
        self.precision_history = []
        self.memory_usage = []
        
    def log_precision(self, step: int, precision: int, memory_used: float):
        self.precision_history.append((step, precision))
        self.memory_usage.append(memory_used)
        
    def get_statistics(self) -> dict:
        return {
            "avg_precision": np.mean([p[1] for p in self.precision_history]),
            "memory_savings": 1 - (np.mean(self.memory_usage) / self.memory_usage[0])
        }

class DynamicPrecisionLayer(nn.Module):
    """Layer with dynamic precision support"""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.current_precision = 32
        
    def quantize(self, precision: int) -> None:
        if precision == 32:
            return
        
        scale = (2 ** (precision - 1)) - 1
        self.weight.data = torch.clamp(
            torch.round(self.weight.data * scale) / scale,
            -1, 1
        )
        self.current_precision = precision
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

class SimpleDiffusion(nn.Module):
    """Simplified diffusion model with dynamic precision"""
    def __init__(self, input_size: int = 16):
        super().__init__()
        self.input_size = input_size
        
        # Simple U-Net like structure
        self.encoder = nn.Sequential(
            DynamicPrecisionLayer(input_size * input_size, 256),
            nn.ReLU(),
            DynamicPrecisionLayer(256, 128),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            DynamicPrecisionLayer(128, 256),
            nn.ReLU(),
            DynamicPrecisionLayer(256, input_size * input_size),
            nn.Sigmoid()
        )
        
        self.metrics = PrecisionMetrics()
        
    def update_precision(self, step: int, total_steps: int) -> int:
        """Calculate required precision based on denoising step"""
        progress = step / total_steps
        if progress < 0.2:
            precision = 8
        elif progress < 0.6:
            precision = 16
        else:
            precision = 32
            
        # Update all dynamic layers
        for layer in self.encoder:
            if isinstance(layer, DynamicPrecisionLayer):
                layer.quantize(precision)
        for layer in self.decoder:
            if isinstance(layer, DynamicPrecisionLayer):
                layer.quantize(precision)
                
        return precision
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.input_size * self.input_size)
        x = self.encoder(x)
        x = self.decoder(x)
        return x.view(-1, 1, self.input_size, self.input_size)

class DiffusionTrainer:
    """Handles training and inference with precision tracking"""
    def __init__(self, model: SimpleDiffusion, total_steps: int = 50):
        self.model = model
        self.total_steps = total_steps
        self.betas = torch.linspace(1e-4, 0.02, total_steps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def add_noise(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x)
        return (
            torch.sqrt(self.alphas_cumprod[t]) * x + 
            torch.sqrt(1 - self.alphas_cumprod[t]) * noise
        ), noise
        
    @torch.no_grad()
    def sample(self, shape: Tuple[int, ...]) -> torch.Tensor:
        x = torch.randn(shape).to(device)
        
        for t in range(self.total_steps - 1, -1, -1):
            logger.info(f"Sampling step {t}")
            
            # Update precision based on current step
            precision = self.model.update_precision(t, self.total_steps)
            
            # Log memory usage and precision
            memory_used = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            self.model.metrics.log_precision(t, precision, memory_used)
            
            t_tensor = torch.tensor([t])
            noise_pred = self.model(x, t_tensor)
            
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]
            
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
                
            x = (1 / torch.sqrt(alpha)) * (
                x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * noise_pred
            ) + torch.sqrt(beta) * noise
            
        return x

def test_diffusion():
    """Run tests on the implementation"""
    # Test setup
    model = SimpleDiffusion(input_size=16).to(device)
    trainer = DiffusionTrainer(model, total_steps=50)
    
    # Test 1: Precision Changes
    logger.info("Testing precision changes...")
    sample_shape = (1, 1, 16, 16)
    generated = trainer.sample(sample_shape)
    
    stats = model.metrics.get_statistics()
    assert 8 <= stats["avg_precision"] <= 32, "Precision out of expected range"
    assert len(model.metrics.precision_history) > 0, "No precision changes recorded"
    
    # Test 2: Memory Savings
    logger.info("Testing memory savings...")
    assert stats["memory_savings"] > 0.3, "Insufficient memory savings"
    
    # Test 3: Output Quality
    logger.info("Testing output quality...")
    assert generated.shape == sample_shape, "Invalid output shape"
    assert torch.max(generated) <= 1 and torch.min(generated) >= 0, "Output values out of range"
    
    return {
        "tests_passed": True,
        "statistics": stats,
        "sample_output": generated
    }

# Run the implementation
if __name__ == "__main__":
    # Example usage
    set_seeds(42)
    logger.info("Starting diffusion PoC with dynamic precision...")
    
    try:
        results = test_diffusion()
        logger.info("All tests passed!")
        logger.info(f"Statistics: {results['statistics']}")
        
        # Save sample output
        save_image(results['sample_output'], "sample_output.png")
        
    except AssertionError as e:
        logger.error(f"Test failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")