import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json

from ..evaluation.metrics import AverageMeter, compute_metrics, print_metrics
from ..models.model_utils import save_checkpoint


class Trainer:
    """
    Trainer class for training and evaluating the model.
    """
    
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, 
                 scheduler, device, config, logger):
        """
        Args:
            model: PyTorch model
            train_loader: training data loader
            val_loader: validation data loader
            criterion: loss function
            optimizer: optimizer
            scheduler: learning rate scheduler
            device: device to train on
            config: configuration dict
            logger: logger instance
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.logger = logger
        
        self.best_qwk = 0.0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_qwk': [],
            'learning_rate': []
        }
        
        # Create save directory
        self.save_dir = Path(config['save_dir']) / config['experiment_name']
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.save_dir / 'logs').mkdir(exist_ok=True)
        
        self.logger.info(f"Save directory: {self.save_dir}")
    
    def train_epoch(self, epoch):
        """
        Train for one epoch.
        
        Args:
            epoch: current epoch number
        
        Returns:
            average loss and accuracy
        """
        self.model.train()
        
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')
        
        all_preds = []
        all_labels = []
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            _, preds = torch.max(outputs, 1)
            acc = (preds == labels).float().mean()
            
            # Update meters
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(acc.item(), images.size(0))
            
            # Store for metrics
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'acc': f'{acc_meter.avg:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Log at intervals
            if batch_idx % self.config.get('log_interval', 10) == 0:
                self.logger.info(
                    f'Epoch {epoch+1} [{batch_idx}/{len(self.train_loader)}] '
                    f'Loss: {loss_meter.avg:.4f} Acc: {acc_meter.avg:.4f}'
                )
        
        return loss_meter.avg, acc_meter.avg
    
    def validate_epoch(self, epoch):
        """
        Validate for one epoch.
        
        Args:
            epoch: current epoch number
        
        Returns:
            average loss, accuracy, and quadratic kappa
        """
        self.model.eval()
        
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]')
        
        with torch.no_grad():
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Calculate accuracy
                _, preds = torch.max(outputs, 1)
                acc = (preds == labels).float().mean()
                
                # Update meters
                loss_meter.update(loss.item(), images.size(0))
                acc_meter.update(acc.item(), images.size(0))
                
                # Store predictions and labels
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss_meter.avg:.4f}',
                    'acc': f'{acc_meter.avg:.4f}'
                })
        
        # Compute metrics
        metrics = compute_metrics(all_labels, all_preds)
        
        self.logger.info(f'\nValidation Metrics - Epoch {epoch+1}:')
        self.logger.info(f'Loss: {loss_meter.avg:.4f}')
        self.logger.info(f'Accuracy: {metrics["accuracy"]:.4f}')
        self.logger.info(f'Quadratic Kappa: {metrics["quadratic_kappa"]:.4f}')
        
        return loss_meter.avg, metrics['accuracy'], metrics['quadratic_kappa'], metrics
    
    def train(self, num_epochs, freeze_epochs=0):
        """
        Full training loop.
        
        Args:
            num_epochs: total number of epochs
            freeze_epochs: number of epochs to freeze backbone (for pretrained)
        """
        self.logger.info("="*60)
        self.logger.info("Starting Training")
        self.logger.info("="*60)
        self.logger.info(f"Total epochs: {num_epochs}")
        self.logger.info(f"Freeze epochs: {freeze_epochs}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self.logger.info("="*60)
        
        # Freeze backbone if specified
        if freeze_epochs > 0 and self.config['model']['pretrained']:
            self.logger.info(f"Freezing backbone for first {freeze_epochs} epochs...")
            self.model.freeze_backbone()
        
        for epoch in range(num_epochs):
            # Unfreeze after freeze_epochs
            if epoch == freeze_epochs and freeze_epochs > 0:
                self.logger.info(f"\nUnfreezing all layers at epoch {epoch+1}")
                self.model.unfreeze_all()
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, val_qwk, val_metrics = self.validate_epoch(epoch)
            
            # Update learning rate
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_qwk)
            else:
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_qwk'].append(val_qwk)
            self.history['learning_rate'].append(current_lr)
            
            # Log epoch summary
            self.logger.info("="*60)
            self.logger.info(f"Epoch {epoch+1}/{num_epochs} Summary:")
            self.logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            self.logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val QWK: {val_qwk:.4f}")
            self.logger.info(f"Learning Rate: {current_lr:.2e}")
            self.logger.info("="*60)
            
            # Save best model
            if val_qwk > self.best_qwk:
                self.best_qwk = val_qwk
                self.logger.info(f"New best QWK: {self.best_qwk:.4f}! Saving model...")
                
                if self.config.get('save_best', True):
                    save_path = self.save_dir / 'checkpoints' / 'best_model.pth'
                    save_checkpoint(
                        self.model, self.optimizer, self.scheduler,
                        epoch, self.best_qwk, save_path
                    )
                    
                    # Save metrics
                    metrics_path = self.save_dir / 'best_metrics.json'
                    with open(metrics_path, 'w') as f:
                        json.dump({
                            'epoch': epoch + 1,
                            'qwk': float(val_qwk),
                            'accuracy': float(val_acc),
                            'loss': float(val_loss),
                            'confusion_matrix': val_metrics['confusion_matrix'].tolist(),
                            'per_class_accuracy': val_metrics['per_class_accuracy'].tolist()
                        }, f, indent=4)
            
            # Save last model
            if self.config.get('save_last', True):
                save_path = self.save_dir / 'checkpoints' / 'last_model.pth'
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    epoch, val_qwk, save_path
                )
            
            # Save history
            history_path = self.save_dir / 'training_history.json'
            with open(history_path, 'w') as f:
                json.dump(self.history, f, indent=4)
        
        self.logger.info("="*60)
        self.logger.info("Training Complete!")
        self.logger.info(f"Best Quadratic Kappa: {self.best_qwk:.4f}")
        self.logger.info("="*60)
        
        return self.history