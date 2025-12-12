# preprocessing.py
# Description: This module contains functions for preprocessing data before analysis or modeling.
# Created by: Renesh Ravi

import numpy as np
import cv2
from PIL import Image
import torch

def crop_image_from_gray(img, tol=7):
    """
    Crop out black borders from image.
    
    Args:
        img: numpy array of image
        tol: tolerance for what's considered "black"
    
    Returns:
        Cropped image
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def circle_crop(img, sigmaX=10):
    """
    Create circular crop of retinal image.
    
    Args:
        img: numpy array of image (RGB)
        sigmaX: Gaussian blur sigma
    
    Returns:
        Circularly cropped image
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = crop_image_from_gray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply Gaussian blur
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigmaX), -4, 128)
    
    return img


def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve contrast.
    
    Args:
        img: numpy array of image (RGB)
        clip_limit: threshold for contrast limiting
        tile_grid_size: size of grid for histogram equalization
    
    Returns:
        Image with enhanced contrast
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Convert back to RGB
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return img


def preprocess_image(image_path, target_size=(224, 224), apply_clahe_flag=False):
    """
    Complete preprocessing pipeline for a single image.
    
    Args:
        image_path: path to image file
        target_size: (height, width) to resize to
        apply_clahe_flag: whether to apply CLAHE
    
    Returns:
        Preprocessed PIL Image
    """
    # Read image
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Circle crop to remove black borders
    img = circle_crop(img, sigmaX=30)
    
    # Apply CLAHE if requested
    if apply_clahe_flag:
        img = apply_clahe(img)
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Convert to PIL Image
    img = Image.fromarray(img)
    
    return img


def get_normalization_params():
    """
    Get ImageNet normalization parameters (for pre-trained models).
    
    Returns:
        mean and std for normalization
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return mean, std