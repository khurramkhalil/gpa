import time
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict
from tqdm import tqdm

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    epoch_start_time = time.time()
    
    for batch_idx, (tokens, labels) in enumerate(tqdm(dataloader)):
        tokens, labels = tokens.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(tokens)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Log attention metrics if using GPA
        if hasattr(model, 'transformer_blocks'):
            for idx, block in enumerate(model.transformer_blocks):
                if hasattr(block.attention, 'attention_metrics'):
                    metrics = block.attention.attention_metrics
                    if metrics:
                        wandb.log({
                            f'block_{idx}/attention_scores_distribution': 
                                wandb.Histogram(metrics['attention_scores_distribution'].cpu().numpy()),
                            f'block_{idx}/precision_levels': 
                                metrics['precision_levels']
                        })
    
    epoch_time = time.time() - epoch_start_time
    wandb.log({'time_per_epoch': epoch_time})
    
    return total_loss / len(dataloader)

def evaluate(model: nn.Module, 
            dataloader: DataLoader,
            criterion: nn.Module,
            device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for tokens, labels in dataloader:
            tokens, labels = tokens.to(device), labels.to(device)
            outputs = model(tokens)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': 100. * correct / total
    }