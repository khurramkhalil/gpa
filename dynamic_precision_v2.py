import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
from tqdm import tqdm
import logging
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DynamicPrecisionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights with proper scaling
        weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        # Xavier/Glorot initialization
        nn.init.xavier_normal_(weight)
        self.register_buffer('original_weight', weight)
        self.register_buffer('original_bias', torch.zeros(out_channels))
        self.register_buffer('quantized_weight', None)
        self.register_buffer('quantized_bias', None)
        
        self.current_precision = 32
        
    def quantize_tensor(self, tensor, bits):
        if bits == 32:
            return tensor
            
        max_val = torch.max(torch.abs(tensor))
        # Add small epsilon to prevent division by zero
        step = (max_val + 1e-8) / (2 ** (bits - 1))
        quantized = torch.round(tensor / step) * step
        return quantized
        
    def quantize(self, bits):
        if bits == self.current_precision:
            return
            
        self.quantized_weight = self.quantize_tensor(self.original_weight, bits)
        self.quantized_bias = self.quantize_tensor(self.original_bias, bits)
        self.current_precision = bits
        
        # Force cleanup
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    def forward(self, x):
        weight = self.quantized_weight if self.quantized_weight is not None else self.original_weight
        bias = self.quantized_bias if self.quantized_bias is not None else self.original_bias
        return F.conv2d(x, weight, bias, self.stride, padding=self.padding)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.conv1 = DynamicPrecisionConv(in_channels, out_channels)
        self.conv2 = DynamicPrecisionConv(out_channels, out_channels)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.time_mlp = nn.Linear(time_dim, out_channels)
        
        if in_channels != out_channels:
            self.shortcut = DynamicPrecisionConv(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x, t):
        identity = self.shortcut(x)
        
        h = self.conv1(x)
        h = self.norm1(h)
        
        time_emb = self.time_mlp(t)[:, :, None, None]
        h = h + time_emb
        
        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h + identity)
        return h
        
    def update_precision(self, bits):
        self.conv1.quantize(bits)
        self.conv2.quantize(bits)
        if isinstance(self.shortcut, DynamicPrecisionConv):
            self.shortcut.quantize(bits)

class DynamicPrecisionDiffusion(nn.Module):
    def __init__(self, image_size=28, device='cuda'):
        super().__init__()
        self.image_size = image_size
        self.device = device
        
        # Diffusion parameters (unchanged)
        self.beta_start = 0.0001
        self.beta_end = 0.02
        self.timesteps = 1000
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.timesteps).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Time embedding dimension
        self.time_dim = 256
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim)
        ).to(device)
        
        # Modified channel configuration
        self.channels = [32, 64, 128, 256]  # Reduced channel dimensions
        
        # Initial conv
        self.init_conv = DynamicPrecisionConv(1, self.channels[0])
        
        # Encoder
        self.down_blocks = nn.ModuleList([
            ResidualBlock(self.channels[i], self.channels[i+1], self.time_dim)
            for i in range(len(self.channels)-1)
        ])
        
        # Middle
        self.middle_block = ResidualBlock(self.channels[-1], self.channels[-1], self.time_dim)
        
        # Decoder - Corrected channel dimensions
        self.up_blocks = nn.ModuleList([
            ResidualBlock(self.channels[i+1] * 2, self.channels[i], self.time_dim)  # *2 because of skip connections
            for i in range(len(self.channels)-1)[::-1]
        ])
        
        # Final conv
        self.final_conv = nn.Sequential(
            DynamicPrecisionConv(self.channels[0], self.channels[0]),
            nn.GroupNorm(8, self.channels[0]),
            nn.SiLU(),
            DynamicPrecisionConv(self.channels[0], 1)
        )
        
        self.precision_history = []
        self.to(device)
        
    def update_precision(self, bits):
        """Update precision for all dynamic layers"""
        self.init_conv.quantize(bits)
        
        for block in self.down_blocks:
            block.update_precision(bits)
            
        self.middle_block.update_precision(bits)
        
        for block in self.up_blocks:
            block.update_precision(bits)
            
        for layer in self.final_conv:
            if isinstance(layer, DynamicPrecisionConv):
                layer.quantize(bits)
                
        self.precision_history.append(bits)
        logger.debug(f"Updated model precision to {bits} bits")
        
    def get_memory_usage(self):
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
        return 0
        
    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_mlp(t.float().view(-1, 1))
        
        # Initial conv
        h = self.init_conv(x)
        logger.debug(f"After init_conv: {h.shape}")
        
        # Encoder
        skip_connections = []
        h_sizes = []
        for idx, block in enumerate(self.down_blocks):
            h = block(h, t_emb)
            skip_connections.append(h)
            h_sizes.append(h.shape[-2:])
            h = F.avg_pool2d(h, 2)
            logger.debug(f"After down_block {idx}: {h.shape}")
            
        # Middle
        h = self.middle_block(h, t_emb)
        logger.debug(f"After middle: {h.shape}")
        
        # Decoder - Proper channel handling
        for idx, block in enumerate(self.up_blocks):
            h = F.interpolate(h, size=h_sizes.pop(), mode='nearest')
            skip = skip_connections.pop()
            logger.debug(f"Up step {idx} - Before concat - h: {h.shape}, skip: {skip.shape}")
            h = torch.cat([h, skip], dim=1)
            logger.debug(f"After concat: {h.shape}")
            h = block(h, t_emb)
            logger.debug(f"After up_block {idx}: {h.shape}")
            
        # Final conv
        output = self.final_conv(h)
        logger.debug(f"Final output: {output.shape}")
        
        return output
        
    @torch.no_grad()
    def sample(self, batch_size=16):
        self.eval()
        x = torch.randn(batch_size, 1, self.image_size, self.image_size).to(self.device)
        
        for t in tqdm(range(self.timesteps-1, -1, -1)):
            if t > self.timesteps * 0.8:
                self.update_precision(8)
            elif t > self.timesteps * 0.4:
                self.update_precision(16)
            else:
                self.update_precision(32)
                
            t_tensor = torch.tensor([t], device=self.device).repeat(batch_size)
            predicted_noise = self.forward(x, t_tensor)
            
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]
            
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
                
            x = (1 / torch.sqrt(alpha)) * (
                x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
            ) + torch.sqrt(beta) * noise
            
        return x

# Modify the training loop:
def train_diffusion():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Modified normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = datasets.MNIST('./data', train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
    
    model = DynamicPrecisionDiffusion(image_size=28, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Reduced learning rate
    
    # Add gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    try:
        epochs = 50
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for batch_idx, (images, _) in enumerate(tqdm(dataloader)):
                images = images.to(device)
                batch_size = images.shape[0]
                
                t = torch.randint(0, model.timesteps, (batch_size,), device=device)
                
                # Scale noise to prevent extreme values
                noise = torch.randn_like(images) * 0.1
                
                # Ensure alphas_cumprod is not too close to 0 or 1
                alphas_cumprod = torch.clamp(model.alphas_cumprod, 1e-5, 1.0 - 1e-5)
                
                noisy_images = torch.sqrt(alphas_cumprod[t].view(-1, 1, 1, 1)) * images + \
                              torch.sqrt(1 - alphas_cumprod[t].view(-1, 1, 1, 1)) * noise
                
                # Update precision as before
                if batch_idx % 3 == 0:
                    model.update_precision(8)
                elif batch_idx % 3 == 1:
                    model.update_precision(16)
                else:
                    model.update_precision(32)
                
                optimizer.zero_grad()
                
                # Use mixed precision training
                with torch.cuda.amp.autocast():
                    predicted_noise = model(noisy_images, t)
                    loss = F.mse_loss(predicted_noise, noise)
                
                # Scale loss and backpropagate
                scaler.scale(loss).backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update weights
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                
                # Add gradient norm logging
                grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) 
                                                  for p in model.parameters() if p.grad is not None]))
                
                if batch_idx % 100 == 0:
                    memory = model.get_memory_usage()
                    logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, "
                              f"Grad norm: {grad_norm:.4f}, Memory: {memory:.2f}MB")
                    
                    # Check for NaN values
                    if torch.isnan(predicted_noise).any():
                        logger.error(f"NaN detected in predicted_noise at epoch {epoch}, batch {batch_idx}")
                        nan_positions = torch.where(torch.isnan(predicted_noise))
                        logger.error(f"NaN positions: {nan_positions}")
                        raise ValueError("NaN detected in predicted noise")
                    
            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
            
            if (epoch + 1) % 5 == 0:
                model.eval()
                samples = model.sample(batch_size=2)
                torchvision.utils.save_image(samples, f"samples_epoch_{epoch}.png", normalize=True)
                
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise e

if __name__ == "__main__":
    try:
        train_diffusion()
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")