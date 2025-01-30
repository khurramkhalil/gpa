import os
import glob
import time
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data.tokenizer import SimpleTokenizer
from src.data.dataset import get_dataset
from src.models.transformer import GPTransformer
from src.utils.training import train_epoch, evaluate
from config import Config
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

import os
import csv
from datetime import datetime
import pandas as pd
from dataclasses import asdict

class TrainingLogger:
    """A comprehensive logger for ML training metrics and configurations."""
    
    def __init__(self, config, log_dir='logs'):
        self.config = config
        self.log_dir = log_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"experiment_{config.attention}_{self.timestamp}"
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize CSV files
        self.metrics_file = os.path.join(log_dir, f"{self.experiment_name}_metrics.csv")
        self.config_file = os.path.join(log_dir, f"{self.experiment_name}_config.csv")
        
        # Save configuration
        self._save_config()
        
        # Initialize metrics file with headers
        self._initialize_metrics_file()
        
        # Keep track of best metrics
        self.best_metrics = {
            'accuracy': 0,
            'epoch': -1,
            'loss': float('inf')
        }
    
    def _save_config(self):
        """Save configuration parameters to CSV."""
        config_dict = asdict(self.config)
        df = pd.DataFrame([config_dict])
        df.to_csv(self.config_file, index=False)
        print(f"Configuration saved to {self.config_file}")
    
    def _initialize_metrics_file(self):
        """Initialize the metrics CSV file with headers."""
        headers = [
            'timestamp',
            'epoch',
            'train_loss',
            'test_loss',
            'test_accuracy',
            'learning_rate',
            'epoch_time',
            'memory_used_gb',
            'is_best_model'
        ]
        
        # Add per-layer metrics headers
        for layer in range(self.config.num_layers):
            headers.extend([
                f'block_{layer}_precision_level',
                f'block_{layer}_attention_entropy'
            ])
        
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def log_epoch(self, epoch_metrics, model_state=None):
        """
        Log metrics for a single epoch.
        
        Args:
            epoch_metrics (dict): Dictionary containing metrics for the epoch
            model_state (optional): Model state for extracting additional metrics
        """
        # Get current timestamp
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Update best metrics
        is_best = False
        if epoch_metrics['test_accuracy'] > self.best_metrics['accuracy']:
            self.best_metrics['accuracy'] = epoch_metrics['test_accuracy']
            self.best_metrics['epoch'] = epoch_metrics['epoch']
            self.best_metrics['loss'] = epoch_metrics['test_loss']
            is_best = True
        
        # Get GPU memory usage if available
        memory_used = 0
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB
        
        # Prepare row data
        row_data = [
            current_time,
            epoch_metrics['epoch'],
            epoch_metrics['train_loss'],
            epoch_metrics['test_loss'],
            epoch_metrics['test_accuracy'],
            epoch_metrics.get('learning_rate', 0),
            epoch_metrics['time_per_epoch'],
            memory_used,
            is_best
        ]
        
        # Add per-layer metrics if model state is provided
        if model_state is not None:
            for layer in range(self.config.num_layers):
                if hasattr(model_state, f'block_{layer}'):
                    block = getattr(model_state, f'block_{layer}')
                    row_data.extend([
                        getattr(block, 'precision_level', 0),
                        getattr(block, 'attention_entropy', 0)
                    ])
                else:
                    row_data.extend([0, 0])  # Default values if metrics not available
        
        # Write to CSV
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)
    
    def get_summary(self):
        """Return a summary of the training run."""
        try:
            df = pd.read_csv(self.metrics_file)
            summary = {
                'best_accuracy': self.best_metrics['accuracy'],
                'best_epoch': self.best_metrics['epoch'],
                'best_loss': self.best_metrics['loss'],
                'total_epochs': len(df),
                'avg_epoch_time': df['epoch_time'].mean(),
                'experiment_name': self.experiment_name,
                'config_file': self.config_file,
                'metrics_file': self.metrics_file
            }
            return summary
        except Exception as e:
            print(f"Error generating summary: {e}")
            return None



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

    # Initialize logger
    logger = TrainingLogger(config)

    NewsDataset = get_dataset()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize tokenizer and datasets
    tokenizer = SimpleTokenizer(min_freq=config.min_freq, max_tokens=config.max_tokens)
    train_dataset = NewsDataset('train', tokenizer)
    test_dataset = NewsDataset('test', tokenizer)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size,
        num_workers=4
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
        num_landmarks=config.num_landmarks,
        dropout=config.dropout
    ).to(device)
    
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
        epoch_start_time = time.time()

        # Get current learning rate
        current_lr = scheduler.get_last_lr()[0]        
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        eval_metrics = evaluate(model, test_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start_time

        # Prepare metrics dictionary
        epoch_metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'test_loss': eval_metrics['loss'],
            'test_accuracy': eval_metrics['accuracy'],
            'learning_rate': current_lr,
            'time_per_epoch': epoch_time
        }
        
        # Log metrics
        logger.log_epoch(epoch_metrics, model)

        print(f"Epoch {epoch+1}/{config.num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test Loss: {eval_metrics['loss']:.4f}")
        print(f"Test Accuracy: {eval_metrics['accuracy']:.2f}%")
        print(f"Time per epoch: {epoch_time:.2f}s")
        print(f"Learning rate: {current_lr:.6f}")

        # Save model if it's the best so far
        if (eval_metrics['accuracy'] > best_accuracy + 2) and (eval_metrics['accuracy'] > 40):
            best_accuracy = eval_metrics['accuracy']
            
            os.makedirs('models', exist_ok=True)

            # Include experiment name in model path
            model_path = f'models/{logger.experiment_name}_acc{best_accuracy:.2f}.pt'
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_accuracy,
                'config': vars(config),
                'experiment_name': logger.experiment_name
            }, model_path)
            print(f"Saved best model to {model_path}")

        # Log to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": eval_metrics['loss'],
            "test_accuracy": eval_metrics['accuracy'],
            "time_per_epoch": epoch_time,
            "learning_rate": current_lr
        })
    
    # Print training summary
    summary = logger.get_summary()
    print("\nTraining Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    # Set resume_training=False if you want to start fresh
    main(resume_training=False)
