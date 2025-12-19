import sys
sys.path.append('.')

import torch
import yaml
import numpy as np
from pathlib import Path

from src.data.dataset import get_dataloaders
from src.models import get_model
from src.training.losses import get_loss_function
from src.training.optimizers import get_optimizer, get_scheduler
from src.training.trainer import Trainer
from src.utils.logger import setup_logger
from src.utils.seed import set_seed


def main():
    # Load config
    config_path = 'configs/config_scratch.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seed
    set_seed(config['seed'])
    
    # Setup logger
    save_dir = Path(config['save_dir']) / config['experiment_name']
    logger = setup_logger(save_dir / 'logs', name='scratch_training')
    
    logger.info("="*60)
    logger.info("From-Scratch ConvNeXt V2 Training")
    logger.info("="*60)
    logger.info(f"Configuration: {config_path}")
    
    # Setup device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data splits
    train_indices = np.load(config['data']['train_indices'])
    val_indices = np.load(config['data']['val_indices'])
    logger.info(f"Train samples: {len(train_indices)}")
    logger.info(f"Val samples: {len(val_indices)}")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, train_dataset, val_dataset = get_dataloaders(
        csv_file=config['data']['csv_file'],
        img_dir=config['data']['img_dir'],
        train_indices=train_indices,
        val_indices=val_indices,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        image_size=config['data']['image_size'],
        preprocess=config['data']['preprocess'],
        apply_clahe=config['data']['apply_clahe'],
        advanced_aug=config['data']['advanced_aug']
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # Get class weights
    class_weights = train_dataset.get_class_weights()
    logger.info(f"Class weights: {class_weights.numpy()}")
    
    # Create model
    logger.info("Creating model...")
    model = get_model(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained']
    )
    model = model.to(device)
    
    # Create loss function
    logger.info(f"Using loss: {config['training']['loss_type']}")
    label_smoothing = config['training'].get('label_smoothing', 0.0)  # NEW LINE
    logger.info(f"Label smoothing: {label_smoothing}")  # NEW LINE
    criterion = get_loss_function(
    loss_type=config['training']['loss_type'],
    class_weights=class_weights,
    label_smoothing=label_smoothing,  # NEW LINE
    device=device
    )
    
    # Create optimizer
    logger.info(f"Using optimizer: {config['training']['optimizer']}")
    optimizer = get_optimizer(
        model=model,
        optimizer_type=config['training']['optimizer'],
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create scheduler
    logger.info(f"Using scheduler: {config['training']['scheduler']}")
    scheduler = get_scheduler(
        optimizer=optimizer,
        scheduler_type=config['training']['scheduler'],
        num_epochs=config['training']['num_epochs'],
        steps_per_epoch=len(train_loader)
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        logger=logger
    )
    
    # Train
    history = trainer.train(
        num_epochs=config['training']['num_epochs'],
        freeze_epochs=config['training']['freeze_epochs']
    )
    
    logger.info("Training completed successfully!")


if __name__ == '__main__':
    main()