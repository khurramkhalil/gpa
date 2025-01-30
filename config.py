from dataclasses import dataclass

@dataclass
class Config:
    # Dataset parameters
    dataset: str = 'amazonreviews'

    # Model parameters
    dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    max_seq_len: int = 256  # Reduced from 512
    
    # Training parameters
    batch_size: int = 512
    num_epochs: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 0.001
    warmup_steps: int = 1000
    dropout: float = 0.2
    gradient_clip: float = 1.0
    
    # Tokenizer parameters
    min_freq: int = 2
    max_tokens: int = 30000
    
    # Attention mechanism
    attention: str = 'gpa'
    num_landmarks: int = 4
