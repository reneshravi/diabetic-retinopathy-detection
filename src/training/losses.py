import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedCrossEntropyLoss(nn.Module):
    """
    Cross entropy loss with class weights for handling imbalance.
    """
    
    def __init__(self, weights, label_smoothing=0.0):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weights = weights
        self.label_smoothing = label_smoothing
    
    def forward(self, outputs, targets):
        return F.cross_entropy(outputs, targets, weight=self.weights, label_smoothing=self.label_smoothing)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Reference: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha, label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_loss_function(loss_type='weighted_ce', class_weights=None, label_smoothing=0.0, device='cuda'):
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
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    elif loss_type == 'weighted_ce':
        if class_weights is None:
            raise ValueError("class_weights required for weighted_ce")
        weights = class_weights.to(device)
        return WeightedCrossEntropyLoss(weights, label_smoothing=label_smoothing)
    
    elif loss_type == 'focal':
        if class_weights is not None:
            alpha = class_weights.to(device)
        else:
            alpha = None
        return FocalLoss(alpha=alpha, gamma=2.0, label_smoothing=label_smoothing)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")