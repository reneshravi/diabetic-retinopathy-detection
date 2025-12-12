# augmentation.py
# Description: This module contains functions for data augmentation techniques used in training machine learning models.
# Created by: Renesh Ravi


import torch
from torchvision import transforms
import numpy as np

def get_train_transforms(image_size=224, advanced=False):
    """
    Get training data augmentation transforms.
    
    Args:
        image_size: size to resize images to
        advanced: whether to use advanced augmentations
    
    Returns:
        torchvision transforms composition
    """
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    if not advanced:
        # Basic augmentations
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        # Advanced augmentations
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    return transform


def get_val_transforms(image_size=224):
    """
    Get validation/test data transforms (no augmentation).
    
    Args:
        image_size: size to resize images to
    
    Returns:
        torchvision transforms composition
    """
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return transform


class Mixup:
    """
    Mixup data augmentation.
    Reference: https://arxiv.org/abs/1710.09412
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, batch):
        """
        Apply mixup to a batch.
        
        Args:
            batch: tuple of (images, labels)
        
        Returns:
            Mixed images and labels
        """
        images, labels = batch
        
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        
        mixed_images = lam * images + (1 - lam) * images[index]
        labels_a, labels_b = labels, labels[index]
        
        return mixed_images, labels_a, labels_b, lam