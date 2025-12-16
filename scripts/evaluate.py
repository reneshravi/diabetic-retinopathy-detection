import sys
sys.path.append('.')

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from tqdm import tqdm

from src.data.dataset import get_dataloaders
from src.models import get_model
from src.models.model_utils import load_checkpoint
from src.evaluation.metrics import compute_metrics, print_metrics


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion'})
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    
    # Add counts as text
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            count = cm[i, j]
            if count > 0:
                plt.text(j + 0.5, i + 0.7, f'({count})', 
                        ha='center', va='center', fontsize=9, color='gray')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix to {save_path}")
    plt.show()


def plot_per_class_metrics(metrics, class_names, save_path):
    """Plot per-class accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Per-class accuracy
    per_class_acc = metrics['per_class_accuracy']
    colors = ['green' if acc > 0.7 else 'orange' if acc > 0.5 else 'red' 
              for acc in per_class_acc]
    
    ax1.bar(range(len(class_names)), per_class_acc, color=colors, alpha=0.7)
    ax1.set_xlabel('Class', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(class_names)))
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.axhline(y=0.7, color='green', linestyle='--', alpha=0.3, label='Good (>0.7)')
    ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.3, label='Fair (>0.5)')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(per_class_acc):
        ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Class distribution in predictions
    cm = metrics['confusion_matrix']
    pred_distribution = cm.sum(axis=0)
    true_distribution = cm.sum(axis=1)
    
    x = np.arange(len(class_names))
    width = 0.35
    
    ax2.bar(x - width/2, true_distribution, width, label='True', alpha=0.7)
    ax2.bar(x + width/2, pred_distribution, width, label='Predicted', alpha=0.7)
    ax2.set_xlabel('Class', fontsize=12)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title('True vs Predicted Class Distribution', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved per-class metrics to {save_path}")
    plt.show()


def analyze_errors(y_true, y_pred, save_path):
    """Analyze prediction errors."""
    errors = np.abs(y_true - y_pred)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Error distribution
    error_counts = np.bincount(errors, minlength=5)
    ax1.bar(range(len(error_counts)), error_counts, alpha=0.7, 
            color=['green', 'yellow', 'orange', 'red', 'darkred'][:len(error_counts)])
    ax1.set_xlabel('Prediction Error (|True - Pred|)', fontsize=12)
    ax1.set_ylabel('Number of Samples', fontsize=12)
    ax1.set_title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(error_counts)))
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels
    total = len(y_true)
    for i, count in enumerate(error_counts):
        pct = 100 * count / total
        ax1.text(i, count + total*0.01, f'{count}\n({pct:.1f}%)', 
                ha='center', va='bottom', fontweight='bold')
    
    # Error by true class
    error_by_class = []
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    for true_class in range(5):
        mask = y_true == true_class
        if mask.sum() > 0:
            avg_error = errors[mask].mean()
            error_by_class.append(avg_error)
        else:
            error_by_class.append(0)
    
    colors = ['green' if e < 0.5 else 'orange' if e < 1.0 else 'red' 
              for e in error_by_class]
    ax2.bar(range(5), error_by_class, color=colors, alpha=0.7)
    ax2.set_xlabel('True Class', fontsize=12)
    ax2.set_ylabel('Average Prediction Error', fontsize=12)
    ax2.set_title('Average Error by True Class', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(5))
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, v in enumerate(error_by_class):
        ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved error analysis to {save_path}")
    plt.show()


def evaluate_model(checkpoint_path, config_path, save_dir):
    """
    Evaluate model and generate all metrics.
    
    Args:
        checkpoint_path: path to model checkpoint
        config_path: path to config file
        save_dir: directory to save results
    """
    import yaml
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("MODEL EVALUATION")
    print("="*60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config: {config_path}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    train_indices = np.load(config['data']['train_indices'])
    val_indices = np.load(config['data']['val_indices'])
    
    _, val_loader, _, _ = get_dataloaders(
        csv_file=config['data']['csv_file'],
        img_dir=config['data']['img_dir'],
        train_indices=train_indices,
        val_indices=val_indices,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        image_size=config['data']['image_size'],
        preprocess=config['data']['preprocess'],
        apply_clahe=config['data']['apply_clahe'],
        advanced_aug=False  # No augmentation for evaluation
    )
    
    print(f"Validation samples: {len(val_indices)}")
    
    # Load model
    model = get_model(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        pretrained=False  # We'll load weights from checkpoint
    )
    
    load_checkpoint(model, checkpoint_path)
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully")
    
    # Collect predictions
    print("\nRunning inference on validation set...")
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Evaluating'):
            images = images.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(all_labels, all_preds)
    
    # Print metrics
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    print_metrics(metrics, class_names)
    
    # Save metrics to JSON
    metrics_dict = {
        'accuracy': float(metrics['accuracy']),
        'quadratic_kappa': float(metrics['quadratic_kappa']),
        'per_class_accuracy': metrics['per_class_accuracy'].tolist(),
        'confusion_matrix': metrics['confusion_matrix'].tolist()
    }
    
    with open(save_dir / 'evaluation_metrics.json', 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"\nSaved metrics to {save_dir / 'evaluation_metrics.json'}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Confusion matrix
    plot_confusion_matrix(metrics['confusion_matrix'], class_names, 
                         save_dir / 'confusion_matrix.png')
    
    # Per-class metrics
    plot_per_class_metrics(metrics, class_names, 
                          save_dir / 'per_class_metrics.png')
    
    # Error analysis
    analyze_errors(all_labels, all_preds, 
                  save_dir / 'error_analysis.png')
    
    # Save predictions
    predictions_df = {
        'true_label': all_labels.tolist(),
        'predicted_label': all_preds.tolist(),
        'probabilities': all_probs.tolist()
    }
    
    with open(save_dir / 'predictions.json', 'w') as f:
        json.dump(predictions_df, f, indent=4)
    print(f"Saved predictions to {save_dir / 'predictions.json'}")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--save_dir', type=str, required=True,
                       help='Directory to save evaluation results')
    args = parser.parse_args()
    
    evaluate_model(args.checkpoint, args.config, args.save_dir)


if __name__ == '__main__':
    main()