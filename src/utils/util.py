import random
import numpy as np
import torch

def set_seeds(seed=42):
    """
    Sets the seed for Python, NumPy, PyTorch, and CUDA to ensure reproducibility.

    Parameters:
    seed (int): The seed value to be used for all libraries.
    """
    # Set the seed for Python's built-in random module
    random.seed(seed)
    
    # Set the seed for NumPy
    np.random.seed(seed)
    
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    
    # If you are using CUDA, set the seed for CUDA as well
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
    
    # Ensure that CUDA operations are deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

