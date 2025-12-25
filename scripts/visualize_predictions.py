import sys
sys.path.append('.')

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from PIL import Image

from src.data.dataset import DiabeticRetinopathyDataset
from src.data.augmentation import get_val_transforms
from src.models import get_model
from src.models.model_utils import load_checkpoint


class GradCAM:
    """Gradient-weighted Class Activation Mapping"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        """Generate CAM for the input image"""
        # Forward pass
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()
        
        # Generate CAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2), keepdim=True)  # [C, 1, 1]
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=0)  # [H, W]
        
        # ReLU and normalize
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy(), output


def visualize_predictions(model, dataset, indices, save_dir, device, num_samples=12):
    """
    Visualize model predictions with Grad-CAM.
    
    Args:
        model: trained model
        dataset: dataset
        indices: indices to visualize
        save_dir: directory to save visualizations
        device: device
        num_samples: number of samples to visualize
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    
    # Get target layer for Grad-CAM (last conv layer before classifier)
    # For ConvNeXt V2, we need to access the last stage
    target_layer = model.model.stages[-1].blocks[-1].conv_dw
    
    # Create Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    # Select random samples
    selected_indices = np.random.choice(indices, size=min(num_samples, len(indices)), replace=False)
    
    # Create figure
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('Model Predictions with Attention Maps', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    model.eval()
    
    for idx, sample_idx in enumerate(selected_indices):
        # Get image and label
        image, true_label = dataset[sample_idx]
        
        # Get original image for visualization
        img_id = dataset.df.iloc[sample_idx]['id_code']
        img_path = dataset.img_dir / f"{img_id}.png"
        original_img = cv2.imread(str(img_path))
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        original_img = cv2.resize(original_img, (224, 224))
        
        # Add batch dimension
        image_input = image.unsqueeze(0).to(device)
        
        # Generate prediction and CAM
        with torch.set_grad_enabled(True):
            cam, output = grad_cam.generate_cam(image_input)
        
        probs = torch.softmax(output, dim=1)[0]
        pred_label = output.argmax(dim=1).item()
        pred_prob = probs[pred_label].item()
        
        # Resize CAM to match image size
        cam_resized = cv2.resize(cam, (224, 224))
        
        # Create heatmap overlay
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Blend with original image
        overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
        
        # Plot
        ax = axes[idx]
        ax.imshow(overlay)
        ax.axis('off')
        
        # Title with prediction info
        color = 'green' if pred_label == true_label else 'red'
        title = f'True: {class_names[true_label]}\n'
        title += f'Pred: {class_names[pred_label]} ({pred_prob:.2%})'
        ax.set_title(title, fontsize=10, fontweight='bold', color=color)
    
    # Hide unused subplots
    for idx in range(len(selected_indices), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'predictions_with_attention.png', dpi=300, bbox_inches='tight')
    print(f"Saved predictions visualization to {save_dir / 'predictions_with_attention.png'}")
    plt.show()


def visualize_correct_vs_incorrect(model, dataset, indices, save_dir, device):
    """
    Visualize correct vs incorrect predictions separately.
    """
    save_dir = Path(save_dir)
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    
    model.eval()
    
    # Collect correct and incorrect predictions
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for idx in indices:
            image, true_label = dataset[idx]
            image_input = image.unsqueeze(0).to(device)
            
            output = model(image_input)
            pred_label = output.argmax(dim=1).item()
            
            if pred_label == true_label:
                correct_samples.append((idx, true_label, pred_label))
            else:
                incorrect_samples.append((idx, true_label, pred_label))
    
    # Visualize incorrect predictions
    if len(incorrect_samples) > 0:
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle('Incorrect Predictions (Failure Cases)', fontsize=16, fontweight='bold', color='red')
        axes = axes.flatten()
        
        num_display = min(12, len(incorrect_samples))
        selected = np.random.choice(len(incorrect_samples), size=num_display, replace=False)
        
        for plot_idx, sample_idx in enumerate(selected):
            idx, true_label, pred_label = incorrect_samples[sample_idx]
            
            img_id = dataset.df.iloc[idx]['id_code']
            img_path = dataset.img_dir / f"{img_id}.png"
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            
            ax = axes[plot_idx]
            ax.imshow(img)
            ax.axis('off')
            
            title = f'True: {class_names[true_label]}\n'
            title += f'Pred: {class_names[pred_label]}'
            ax.set_title(title, fontsize=10, fontweight='bold', color='red')
        
        for idx in range(num_display, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'incorrect_predictions.png', dpi=300, bbox_inches='tight')
        print(f"Saved incorrect predictions to {save_dir / 'incorrect_predictions.png'}")
        plt.show()
        
        print(f"\nTotal incorrect predictions: {len(incorrect_samples)} / {len(indices)} ({100*len(incorrect_samples)/len(indices):.2f}%)")
    
    # Visualize correct predictions from each class
    fig, axes = plt.subplots(5, 4, figsize=(16, 20))
    fig.suptitle('Correct Predictions by Class', fontsize=16, fontweight='bold', color='green')
    
    for class_id in range(5):
        class_correct = [s for s in correct_samples if s[1] == class_id]
        
        for col in range(4):
            ax = axes[class_id, col]
            
            if col < len(class_correct):
                idx, true_label, pred_label = class_correct[col]
                
                img_id = dataset.df.iloc[idx]['id_code']
                img_path = dataset.img_dir / f"{img_id}.png"
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                
                ax.imshow(img)
                ax.set_title(f'{class_names[class_id]} ✓', fontsize=10, fontweight='bold', color='green')
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12)
            
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'correct_predictions_by_class.png', dpi=300, bbox_inches='tight')
    print(f"Saved correct predictions by class to {save_dir / 'correct_predictions_by_class.png'}")
    plt.show()


def main():
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description='Visualize model predictions')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--save_dir', type=str, required=True,
                       help='Directory to save visualizations')
    parser.add_argument('--num_samples', type=int, default=12,
                       help='Number of samples to visualize')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("PREDICTION VISUALIZATION")
    print("="*60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load validation dataset
    val_indices = np.load(config['data']['val_indices'])
    transform = get_val_transforms(config['data']['image_size'])
    
    dataset = DiabeticRetinopathyDataset(
        csv_file=config['data']['csv_file'],
        img_dir=config['data']['img_dir'],
        indices=val_indices,
        transform=transform,
        preprocess=config['data']['preprocess'],
        apply_clahe=config['data']['apply_clahe']
    )
    
    print(f"Loaded {len(dataset)} validation samples")
    
    # Load model
    dropout = config['model'].get('dropout', 0.0)  
    model = get_model(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        pretrained=False,
        dropout=dropout  
    )
    
    load_checkpoint(model, args.checkpoint)
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully")
    
    # Generate visualizations
    save_dir = Path(args.save_dir)
    
    print("\nGenerating attention map visualizations...")
    visualize_predictions(model, dataset, val_indices, save_dir, device, args.num_samples)
    
    print("\nGenerating correct vs incorrect predictions...")
    visualize_correct_vs_incorrect(model, dataset, val_indices, save_dir, device)
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    main()