from dataclasses import dataclass

@dataclass
class Config:
    # Model parameters
    dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    max_seq_len: int = 128
    
    # Training parameters
    batch_size: int = 128
    num_epochs: int = 50
    learning_rate: float = 3e-4
    
    # Tokenizer parameters
    min_freq: int = 2
    max_tokens: int = 20000

    # Attention mechanis,
    attention: str = 'gpa'

    # Efficient GPA parameters
    num_clusters: int = 32
    min_points_per_centroid: int = 8

    # Can be tuned based on sequence length
    num_landmarks: int = 16