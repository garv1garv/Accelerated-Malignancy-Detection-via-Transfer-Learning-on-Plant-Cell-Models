"""
Training module for the Plant2Skin transfer learning pipeline.

⚠️  RESEARCH PROTOTYPE - NOT FOR CLINICAL USE ⚠️
"""

import os
import time
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from .config import config
from .models import create_model
from .data_utils import create_dataloaders
from .utils import (
    compute_metrics, save_checkpoint, save_training_history,
    plot_training_history, print_device_info
)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Tuple[float, Dict[str, float]]:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
    
    Returns:
        Tuple of (average_loss, metrics_dict)
    """
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits, probabilities = model(images)
        
        # Compute loss
        if logits.shape[1] == 2:
            # Binary classification with 2 outputs
            loss = criterion(logits, labels.float())
        else:
            # Binary classification with 1 output
            loss = criterion(logits.squeeze(), labels.float())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Collect predictions and labels
        if logits.shape[1] == 2:
            preds = (logits[:, 1] > 0).long()
            probs = probabilities[:, 1] if probabilities.shape[1] == 2 else probabilities.squeeze()
        else:
            preds = (logits.squeeze() > 0).long()
            probs = probabilities.squeeze()
        
        all_predictions.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probabilities.extend(probs.cpu().numpy())
        
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
        })
    
    # Compute metrics
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probabilities)
    )
    
    return avg_loss, metrics


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> Tuple[float, Dict[str, float]]:
    """
    Validate for one epoch.
    
    Args:
        model: Model to validate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
    
    Returns:
        Tuple of (average_loss, metrics_dict)
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits, probabilities = model(images)
            
            # Compute loss
            if logits.shape[1] == 2:
                loss = criterion(logits, labels.float())
            else:
                loss = criterion(logits.squeeze(), labels.float())
            
            # Collect predictions and labels
            if logits.shape[1] == 2:
                preds = (logits[:, 1] > 0).long()
                probs = probabilities[:, 1] if probabilities.shape[1] == 2 else probabilities.squeeze()
            else:
                preds = (logits.squeeze() > 0).long()
                probs = probabilities.squeeze()
            
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
            })
    
    # Compute metrics
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probabilities)
    )
    
    return avg_loss, metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    device: str = "auto",
    save_dir: str = "outputs",
    class_weights: Optional[torch.Tensor] = None
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train the model.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        device: Device to train on
        save_dir: Directory to save checkpoints
        class_weights: Optional class weights for imbalanced datasets
    
    Returns:
        Tuple of (trained_model, training_history)
    """
    # Set device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    print(f"Training on device: {device}")
    print_device_info()
    
    # Move model to device
    model = model.to(device)
    
    # Setup loss function
    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Setup learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'train_precision': [],
        'val_precision': [],
        'train_recall': [],
        'val_recall': [],
        'train_f1_score': [],
        'val_f1_score': [],
        'train_roc_auc': [],
        'val_roc_auc': []
    }
    
    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['train_precision'].append(train_metrics['precision'])
        history['val_precision'].append(val_metrics['precision'])
        history['train_recall'].append(train_metrics['recall'])
        history['val_recall'].append(val_metrics['recall'])
        history['train_f1_score'].append(train_metrics['f1_score'])
        history['val_f1_score'].append(val_metrics['f1_score'])
        history['train_roc_auc'].append(train_metrics['roc_auc'])
        history['val_roc_auc'].append(val_metrics['roc_auc'])
        
        # Print epoch summary
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.2f}s)")
        print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"  Train Acc: {train_metrics['accuracy']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"  Train F1: {train_metrics['f1_score']:.4f}, Val F1: {val_metrics['f1_score']:.4f}")
        print(f"  Train AUC: {train_metrics['roc_auc']:.4f}, Val AUC: {val_metrics['roc_auc']:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            
            # Save checkpoint
            if config.save_best_model:
                checkpoint_path = os.path.join(save_dir, "best_model.pt")
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    loss=val_loss,
                    metrics=val_metrics,
                    filepath=checkpoint_path,
                    config={
                        'backbone_name': model.backbone_name,
                        'num_classes': model.num_classes,
                        'dropout_rate': model.dropout_rate,
                        'freeze_backbone': model.freeze_backbone
                    }
                )
        
        # Save regular checkpoint
        if config.save_checkpoints and (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt")
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=val_loss,
                metrics=val_metrics,
                filepath=checkpoint_path
            )
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best model (val_loss: {best_val_loss:.4f})")
    
    # Save training history
    history_path = os.path.join(save_dir, "training_history.json")
    save_training_history(history, history_path)
    
    # Plot training history
    plot_path = os.path.join(save_dir, "training_history.png")
    plot_training_history(history, plot_path)
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Training history saved to: {history_path}")
    print(f"Training plots saved to: {plot_path}")
    
    return model, history


def main():
    """Main training function."""
    print("⚠️  RESEARCH PROTOTYPE - NOT FOR CLINICAL USE ⚠️")
    print("Starting Plant2Skin transfer learning training...")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=config.data_root,
        batch_size=config.batch_size,
        image_size=config.image_size,
        num_workers=config.num_workers,
        use_augmentation=config.use_augmentation,
        augmentation_prob=config.augmentation_prob
    )
    
    # Create model
    model = create_model(
        backbone_name=config.backbone_name,
        num_classes=config.num_classes,
        dropout_rate=config.dropout_rate,
        freeze_backbone=config.freeze_backbone,
        pretrained_checkpoint=config.pretrained_checkpoint,
        unfreeze_layers=config.unfreeze_layers
    )
    
    # Compute class weights if needed
    class_weights = None
    if hasattr(train_loader.dataset, 'get_class_weights'):
        class_weights = train_loader.dataset.get_class_weights()
        print(f"Using class weights: {class_weights}")
    
    # Train model
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=config.device,
        save_dir=config.output_dir,
        class_weights=class_weights
    )
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
