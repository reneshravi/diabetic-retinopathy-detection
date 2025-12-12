# dataset.py
# Description: This module contains the PyTorch Dataset class for Diabetic Retinopathy images along with functions to create dataloaders.
# Created by: Renesh Ravi


import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from .preprocessing import preprocess_image


class DiabeticRetinopathyDataset(Dataset):
    """
    PyTorch Dataset for Diabetic Retinopathy images.
    """
    
    def __init__(self, csv_file, img_dir, indices=None, transform=None, 
                 preprocess=True, apply_clahe=False):
        """
        Args:
            csv_file: path to CSV file with annotations
            img_dir: directory with all the images
            indices: optional array of indices to use (for train/val split)
            transform: optional transform to be applied on images
            preprocess: whether to apply preprocessing (circle crop, etc.)
            apply_clahe: whether to apply CLAHE in preprocessing
        """
        self.df = pd.read_csv(csv_file)
        
        # Filter by indices if provided
        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)
        
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.preprocess = preprocess
        self.apply_clahe = apply_clahe
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image path and label
        img_name = self.df.iloc[idx]['id_code']
        img_path = self.img_dir / f"{img_name}.png"
        label = self.df.iloc[idx]['diagnosis']
        
        # Load and preprocess image
        if self.preprocess:
            image = preprocess_image(img_path, target_size=(224, 224), 
                                    apply_clahe_flag=self.apply_clahe)
        else:
            image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_weights(self):
        """
        Calculate class weights for handling imbalance.
        
        Returns:
            torch tensor of class weights
        """
        class_counts = self.df['diagnosis'].value_counts().sort_index().values
        total = len(self.df)
        
        # Inverse frequency weighting
        weights = total / (len(class_counts) * class_counts)
        
        return torch.FloatTensor(weights)
    
    def get_class_distribution(self):
        """
        Get the distribution of classes in the dataset.
        
        Returns:
            dict with class counts
        """
        return self.df['diagnosis'].value_counts().sort_index().to_dict()


def get_dataloaders(csv_file, img_dir, train_indices, val_indices, 
                   batch_size=32, num_workers=4, image_size=224,
                   preprocess=True, apply_clahe=False, advanced_aug=False):
    """
    Create train and validation dataloaders.
    
    Args:
        csv_file: path to CSV file with annotations
        img_dir: directory with all the images
        train_indices: array of training indices
        val_indices: array of validation indices
        batch_size: batch size for dataloaders
        num_workers: number of workers for data loading
        image_size: size to resize images to
        preprocess: whether to apply preprocessing
        apply_clahe: whether to apply CLAHE
        advanced_aug: whether to use advanced augmentations
    
    Returns:
        train_loader, val_loader, train_dataset, val_dataset
    """
    from .augmentation import get_train_transforms, get_val_transforms
    
    # Get transforms
    train_transform = get_train_transforms(image_size, advanced=advanced_aug)
    val_transform = get_val_transforms(image_size)
    
    # Create datasets
    train_dataset = DiabeticRetinopathyDataset(
        csv_file=csv_file,
        img_dir=img_dir,
        indices=train_indices,
        transform=train_transform,
        preprocess=preprocess,
        apply_clahe=apply_clahe
    )
    
    val_dataset = DiabeticRetinopathyDataset(
        csv_file=csv_file,
        img_dir=img_dir,
        indices=val_indices,
        transform=val_transform,
        preprocess=preprocess,
        apply_clahe=apply_clahe
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset, val_dataset

