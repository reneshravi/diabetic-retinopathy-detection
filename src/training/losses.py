import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedCrossEntropyLoss(nn.Module):
    """
    Cross entropy loss with class weights for handling imbalance.
    """
    
    def __init__(self, weights):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weights = weights
    
    def forward(self, outputs, targets):
        return F.cross_entropy(outputs, targets, weight=self.weights)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Reference: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_loss_function(loss_type='weighted_ce', class_weights=None, device='cuda'):
    """
    Factory function to get loss function.
    
    Args:
        loss_type: 'ce', 'weighted_ce', or 'focal'
        class_weights: weights for each class
        device: device to put weights on
    
    Returns:
        loss function
    """
    if loss_type == 'ce':
        return nn.CrossEntropyLoss()
    
    elif loss_type == 'weighted_ce':
        if class_weights is None:
            raise ValueError("class_weights required for weighted_ce")
        weights = class_weights.to(device)
        return WeightedCrossEntropyLoss(weights)
    
    elif loss_type == 'focal':
        if class_weights is not None:
            alpha = class_weights.to(device)
        else:
            alpha = None
        return FocalLoss(alpha=alpha, gamma=2.0)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")