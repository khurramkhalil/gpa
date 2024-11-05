import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import numpy as np
from collections import Counter
import pandas as pd

# 1. First, let's create a simple vocabulary and tokenizer
class SimpleTokenizer:
    def __init__(self, min_freq: int = 1):
        self.word2idx = {}
        self.idx2word = {}
        self.min_freq = min_freq
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        
    def fit(self, texts: List[str]):
        # Count words
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
        
        # Create vocabulary
        vocab = [self.pad_token, self.unk_token]
        vocab.extend([word for word, count in word_counts.items() 
                     if count >= self.min_freq])
        
        # Create mappings
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for idx, word in enumerate(vocab)}
        
    def encode(self, text: str, max_len: int = 32) -> List[int]:
        words = text.lower().split()
        tokens = [self.word2idx.get(word, self.word2idx[self.unk_token]) 
                 for word in words]
        
        # Padding/truncating
        if len(tokens) < max_len:
            tokens = tokens + [self.word2idx[self.pad_token]] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
        
        return tokens
    
    def vocab_size(self) -> int:
        return len(self.word2idx)

# 2. Create a custom dataset
class TextClassificationDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: SimpleTokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        tokens = self.tokenizer.encode(text)
        return torch.tensor(tokens), torch.tensor(label)

# 3. Implementation of Geometric Progressive Attention
class GPAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, and values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention calculation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Geometric attention computation
        # Instead of Q @ K.transpose(-2, -1), we use geometric operations
        # This is where we'll implement our novel approach
        
        # For now, using simple dot product as placeholder
        # We'll replace this with geometric operations in next iteration
        attn = (q @ k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        
        out = attn @ v
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.dim)
        
        return self.out_proj(out)

# 4. Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.attention = GPAttention(dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x

# 5. Complete Model
class GPTransformer(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 num_classes: int,
                 dim: int = 256,
                 num_layers: int = 4,
                 num_heads: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 32, dim))
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads) for _ in range(num_layers)
        ])
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len)
        x = self.embedding(x)
        x = x + self.pos_embedding
        
        for block in self.transformer_blocks:
            x = block(x)
            
        # Global average pooling
        x = x.mean(dim=1)
        
        return self.classifier(x)

# 6. Training setup
def train_model():
    # Create dummy dataset
    texts = [
        "this is a positive review",
        "negative review here",
        "great movie loved it",
        "terrible waste of time",
    ]
    labels = [1, 0, 1, 0]
    
    # Initialize tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.fit(texts)
    
    # Create dataset
    dataset = TextClassificationDataset(texts, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Initialize model
    model = GPTransformer(
        vocab_size=tokenizer.vocab_size(),
        num_classes=2,
        dim=256,
        num_layers=4,
        num_heads=8
    )
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    for epoch in range(5):
        total_loss = 0
        for batch_idx, (tokens, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(tokens)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

if __name__ == "__main__":
    train_model()
