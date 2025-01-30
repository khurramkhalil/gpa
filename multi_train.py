import os
import glob
import time
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data.tokenizer import SimpleTokenizer
from src.data.dataset import NewsDataset
from src.models.transformer import GPTransformer
from src.utils.training import train_epoch, evaluate
from config import Config
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import torchtext
torchtext.disable_torchtext_deprecation_warning()

def setup_distributed_training():
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = str(torch.cuda.device_count())
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
#    dist.init_process_group(backend='nccl')


def find_latest_checkpoint(config):
    """Find the latest checkpoint with matching configuration."""
    if not os.path.exists('models'):
        return None
    
    # Look for models with matching attention mechanism
    pattern = f'models/baseline_model_{config.attention}_*.pt'
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        return None
    
    # Sort by timestamp and accuracy
    latest_checkpoint = max(checkpoints, key=lambda x: os.path.getmtime(x))
    return latest_checkpoint

def load_checkpoint(checkpoint_path, model, optimizer, device):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Add debug prints
    print(f"Checkpoint contains:")
    print(f"- Epoch: {checkpoint['epoch']}")
    print(f"- Best accuracy: {checkpoint['accuracy']:.2f}%")
    print(f"- Attention type: {checkpoint['config']['attention']}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['accuracy'], checkpoint['epoch'] + 1

def get_scheduler(optimizer, config):
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        epochs=config.num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )

def main(resume_training=True):
    config = Config()
    # setup_distributed_training()
    # Initialize distributed processing
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    # Initialize tokenizer and datasets
    tokenizer = SimpleTokenizer(min_freq=config.min_freq, max_tokens=config.max_tokens)
    train_dataset = NewsDataset('train', tokenizer)
    test_dataset = NewsDataset('test', tokenizer)

    train_sampler = DistributedSampler(train_dataset)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
#        shuffle=True,
        num_workers=4,
        sampler=train_sampler
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size,
        num_workers=4,
        sampler=test_sampler
    )
    
    # Initialize model
    model = GPTransformer(
        vocab_size=tokenizer.vocab_size(),
        num_classes=20,
        dim=config.dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        max_seq_len=config.max_seq_len,
        attention=config.attention,
        dropout=config.dropout
    ).to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Initialize criterion with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Initialize optimizer with weight decay
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Initialize scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        epochs=config.num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )

    # Synchronize across processes
    dist.barrier()

    # Initialize starting epoch and best accuracy
    start_epoch = 0
    best_accuracy = 0
    
    # Load checkpoint if resuming and checkpoint exists
    if resume_training:
        checkpoint_path = find_latest_checkpoint(config)
        if checkpoint_path:
            best_accuracy, start_epoch = load_checkpoint(
                checkpoint_path, model, optimizer, device
            )
            print(f"Resuming training from epoch {start_epoch} with best accuracy {best_accuracy:.2f}%")
    
    # Initialize wandb
    wandb.init(
        project="gp-transformer",
        config=config,
        resume=True if (resume_training and start_epoch > 0) else False
    )

    # Define custom wandb plots
    wandb.define_metric("epoch")
    wandb.define_metric("train_loss", step_metric="epoch")
    wandb.define_metric("test_loss", step_metric="epoch")
    wandb.define_metric("test_accuracy", step_metric="epoch")
    wandb.define_metric("time_per_epoch", step_metric="epoch")
    
    for block_idx in range(config.num_layers):
        wandb.define_metric(f"block_{block_idx}/precision_levels", step_metric="epoch")

    # Add class names for interpretability
    # class_names = fetch_20newsgroups()['target_names']
    
    # Training loop
    for epoch in range(start_epoch, config.num_epochs):
        train_sampler.set_epoch(epoch)  # Set epoch to shuffle data properly
        epoch_start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        eval_metrics = evaluate(model, test_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start_time


        if local_rank == 0:  # Only print and save from rank 0
            print(f"Epoch {epoch+1}/{config.num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Test Loss: {eval_metrics['loss']:.4f}")
            print(f"Test Accuracy: {eval_metrics['accuracy']:.2f}%")
            print(f"Time per epoch: {epoch_time:.2f}s")

            # Save model if it's the best so far
            if eval_metrics['accuracy'] > best_accuracy + 2:
                best_accuracy = eval_metrics['accuracy']
                os.makedirs('models', exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = f'models/baseline_model_{config.attention}_{timestamp}_acc{best_accuracy:.2f}.pt'
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': best_accuracy,
                    'config': vars(config),
                }, model_path)
                print(f"Saved best model to {model_path}")

            # Log metrics with wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": eval_metrics['loss'],
                "test_accuracy": eval_metrics['accuracy'],
                "time_per_epoch": epoch_time
            })

    dist.destroy_process_group()

"""        
#        print(f"Epoch {epoch+1}/{config.num_epochs}")
#        print(f"Train Loss: {train_loss:.4f}")
#        print(f"Test Loss: {eval_metrics['loss']:.4f}")
#        print(f"Test Accuracy: {eval_metrics['accuracy']:.2f}%")
#        print(f"Time per epoch: {epoch_time:.2f}s")

        # Save model if it's the best so far
#        if eval_metrics['accuracy'] > best_accuracy:
#            best_accuracy = eval_metrics['accuracy']
            
#            os.makedirs('models', exist_ok=True)
            
#            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#            model_path = f'models/baseline_model_{config.attention}_{timestamp}_acc{best_accuracy:.2f}.pt'
            
#            torch.save({
#                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_accuracy,
                'config': vars(config),
            }, model_path)
            print(f"Saved best model to {model_path}")

        # Log metrics
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": eval_metrics['loss'],
            "test_accuracy": eval_metrics['accuracy'],
            "time_per_epoch": epoch_time
        })
"""

if __name__ == "__main__":
    # Set resume_training=False if you want to start fresh
    # setup_distributed_training()
    main(resume_training=False)
