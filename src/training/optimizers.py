import torch
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR

def get_optimizer(model, optimizer_type='adamw', lr=1e-4, weight_decay=1e-4):
    """
    Get optimizer.
    
    Args:
        model: PyTorch model
        optimizer_type: 'adamw' or 'sgd'
        lr: learning rate
        weight_decay: weight decay
    
    Returns:
        optimizer
    """
    if optimizer_type == 'adamw':
        optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
    elif optimizer_type == 'sgd':
        optimizer = SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9,
            nesterov=True
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer


def get_scheduler(optimizer, scheduler_type='cosine', num_epochs=30, 
                 steps_per_epoch=None, warmup_epochs=5):
    """
    Get learning rate scheduler.
    
    Args:
        optimizer: optimizer
        scheduler_type: 'cosine', 'plateau', or 'onecycle'
        num_epochs: total number of epochs
        steps_per_epoch: steps per epoch (needed for onecycle)
        warmup_epochs: number of warmup epochs
    
    Returns:
        scheduler
    """
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=1e-6
        )
    
    elif scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            verbose=True
        )
    
    elif scheduler_type == 'onecycle':
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch required for onecycle scheduler")
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]['lr'],
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=warmup_epochs/num_epochs
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler