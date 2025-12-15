import torch
from pathlib import Path

def save_checkpoint(model, optimizer, scheduler, epoch, best_metric, save_path):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: optimizer
        scheduler: learning rate scheduler
        epoch: current epoch
        best_metric: best validation metric so far
        save_path: path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_metric': best_metric,
    }
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(model, checkpoint_path, optimizer=None, scheduler=None):
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        checkpoint_path: path to checkpoint
        optimizer: optimizer (optional)
        scheduler: learning rate scheduler (optional)
    
    Returns:
        epoch, best_metric
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_metric = checkpoint.get('best_metric', 0.0)
    
    print(f"Loaded checkpoint from epoch {epoch}")
    return epoch, best_metric


def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params