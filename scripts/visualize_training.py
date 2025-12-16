import sys
sys.path.append('.')

import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def plot_training_history(history_path, save_dir):
    """
    Plot training history curves.
    
    Args:
        history_path: path to training_history.json
        save_dir: directory to save plots
    """
    # Load history
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    # Plot 1: Loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Quadratic Kappa
    ax3 = axes[1, 0]
    ax3.plot(epochs, history['val_qwk'], 'g-', linewidth=2)
    ax3.axhline(y=max(history['val_qwk']), color='r', linestyle='--', 
                label=f'Best: {max(history["val_qwk"]):.4f}')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Quadratic Weighted Kappa', fontsize=12)
    ax3.set_title('Validation Quadratic Weighted Kappa', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Learning Rate
    ax4 = axes[1, 1]
    ax4.plot(epochs, history['learning_rate'], 'purple', linewidth=2)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Learning Rate', fontsize=12)
    ax4.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_history.png', dpi=300, bbox_inches='tight')
    print(f"Saved training history plot to {save_dir / 'training_history.png'}")
    plt.show()
    
    # Create detailed metrics plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Detailed Training Metrics', fontsize=16, fontweight='bold')
    
    # Plot generalization gap
    ax1 = axes[0]
    generalization_gap = np.array(history['train_acc']) - np.array(history['val_acc'])
    ax1.plot(epochs, generalization_gap, 'orange', linewidth=2)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax1.fill_between(epochs, 0, generalization_gap, alpha=0.3, color='orange')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Train Acc - Val Acc', fontsize=12)
    ax1.set_title('Generalization Gap (Overfitting Indicator)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot improvement rate
    ax2 = axes[1]
    qwk_improvement = np.diff([0] + history['val_qwk'])
    ax2.bar(epochs, qwk_improvement, color=['green' if x > 0 else 'red' for x in qwk_improvement])
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('QWK Change', fontsize=12)
    ax2.set_title('Validation QWK Improvement per Epoch', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_metrics_detailed.png', dpi=300, bbox_inches='tight')
    print(f"Saved detailed metrics plot to {save_dir / 'training_metrics_detailed.png'}")
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total Epochs: {len(epochs)}")
    print(f"\nBest Validation Metrics:")
    print(f"  Best QWK: {max(history['val_qwk']):.4f} (Epoch {np.argmax(history['val_qwk']) + 1})")
    print(f"  Best Val Acc: {max(history['val_acc']):.4f} (Epoch {np.argmax(history['val_acc']) + 1})")
    print(f"  Best Val Loss: {min(history['val_loss']):.4f} (Epoch {np.argmin(history['val_loss']) + 1})")
    
    print(f"\nFinal Epoch Metrics:")
    print(f"  Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"  Train Acc: {history['train_acc'][-1]:.4f}")
    print(f"  Val Loss: {history['val_loss'][-1]:.4f}")
    print(f"  Val Acc: {history['val_acc'][-1]:.4f}")
    print(f"  Val QWK: {history['val_qwk'][-1]:.4f}")
    
    print(f"\nGeneralization Analysis:")
    final_gap = history['train_acc'][-1] - history['val_acc'][-1]
    print(f"  Final generalization gap: {final_gap:.4f}")
    if final_gap < 0.05:
        print("  Status: Excellent generalization ✓")
    elif final_gap < 0.10:
        print("  Status: Good generalization ✓")
    elif final_gap < 0.15:
        print("  Status: Moderate overfitting ⚠")
    else:
        print("  Status: Significant overfitting ⚠")
    print("="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize training history')
    parser.add_argument('--experiment', type=str, required=True,
                       help='Path to experiment directory')
    args = parser.parse_args()
    
    experiment_dir = Path(args.experiment)
    history_path = experiment_dir / 'training_history.json'
    save_dir = experiment_dir / 'visualizations'
    
    if not history_path.exists():
        print(f"Error: {history_path} not found!")
        return
    
    plot_training_history(history_path, save_dir)


if __name__ == '__main__':
    main()