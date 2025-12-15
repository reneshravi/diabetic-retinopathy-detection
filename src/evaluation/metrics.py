import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report,
    cohen_kappa_score
)

def quadratic_weighted_kappa(y_true, y_pred, num_classes=5):
    """
    Calculate quadratic weighted kappa score.
    This is the primary metric for DR detection.
    
    Args:
        y_true: ground truth labels
        y_pred: predicted labels
        num_classes: number of classes
    
    Returns:
        quadratic weighted kappa score
    """
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def compute_metrics(y_true, y_pred, num_classes=5):
    """
    Compute all relevant metrics.
    
    Args:
        y_true: ground truth labels (numpy array or list)
        y_pred: predicted labels (numpy array or list)
        num_classes: number of classes
    
    Returns:
        dict of metrics
    """
    # Convert to numpy if needed
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    qwk = quadratic_weighted_kappa(y_true, y_pred, num_classes)
    cm = confusion_matrix(y_true, y_pred)
    
    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    metrics = {
        'accuracy': acc,
        'quadratic_kappa': qwk,
        'confusion_matrix': cm,
        'per_class_accuracy': per_class_acc
    }
    
    return metrics


def print_metrics(metrics, class_names=None):
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: dict of metrics from compute_metrics
        class_names: optional list of class names
    """
    if class_names is None:
        class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    
    print("="*60)
    print("METRICS")
    print("="*60)
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Quadratic Weighted Kappa: {metrics['quadratic_kappa']:.4f}")
    
    print("\nPer-Class Accuracy:")
    for i, (name, acc) in enumerate(zip(class_names, metrics['per_class_accuracy'])):
        print(f"  Class {i} ({name:15s}): {acc:.4f}")
    
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print("="*60)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking metrics during training.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count