# create_splits.py
# Description: This script creates training, validation, and test splits from a
# dataset of images and their corresponding annotations.
# Created by: Renesh Ravi

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import os

# Set random seed for reproducibility
np.random.seed(42)

# Get the project root directory (parent of scripts/)
project_root = Path(__file__).parent.parent

# Load data
data_dir = project_root / 'data' / 'raw'
train_csv = pd.read_csv(data_dir / 'train.csv')

# Stratified split to maintain class distribution
train_indices, val_indices = train_test_split(
    np.arange(len(train_csv)),
    test_size=0.2,
    stratify=train_csv['diagnosis'],
    random_state=42
)

# Create splits directory
splits_dir = project_root / 'data' / 'splits'
splits_dir.mkdir(parents=True, exist_ok=True)

# Save splits
np.save(splits_dir / 'train_indices.npy', train_indices)
np.save(splits_dir / 'val_indices.npy', val_indices)

print(f"Train set size: {len(train_indices)}")
print(f"Validation set size: {len(val_indices)}")
print(f"\nTrain set class distribution:")
print(train_csv.iloc[train_indices]['diagnosis'].value_counts().sort_index())
print(f"\nValidation set class distribution:")
print(train_csv.iloc[val_indices]['diagnosis'].value_counts().sort_index())
print("\n✓ Splits saved to data/splits/")