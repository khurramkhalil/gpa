import torch
from torch.utils.data import Dataset
from sklearn.datasets import fetch_20newsgroups
from typing import Tuple, List
from .tokenizer import SimpleTokenizer

class NewsDataset(Dataset):
    def __init__(self, split: str, tokenizer: SimpleTokenizer):
        assert split in ['train', 'test']
        self.tokenizer = tokenizer
        
        # Load 20 Newsgroups dataset
        dataset = fetch_20newsgroups(
            subset=split,
            shuffle=True,
            random_state=42,
            remove=('headers', 'footers', 'quotes')  # Remove metadata for clean text
        )
        
        self.texts = dataset.data
        self.labels = dataset.target
            
        if split == 'train':
            self.tokenizer.fit(self.texts)
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        tokens = self.tokenizer.encode(text)
        return torch.tensor(tokens), torch.tensor(label)