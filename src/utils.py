"""
Utility functions for the Plant2Skin transfer learning pipeline.

⚠️  RESEARCH PROTOTYPE - NOT FOR CLINICAL USE ⚠️
"""

import os
import json
import pickle
from typing import Dict, Any, Optional, Tuple
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metrics: Dict[str, float],
    filepath: str,
    config: Optional[Dict[str, Any]] = None
):
    """Save model checkpoint with metadata."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
        'config': config
    }
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    filepath: str
) -> Tuple[int, float, Dict[str, float], Optional[Dict[str, Any]]]:
    """Load model checkpoint and return metadata."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    metrics = checkpoint['metrics']
    config = checkpoint.get('config')
    
    print(f"Checkpoint loaded from {filepath} (epoch {epoch}, loss {loss:.4f})")
    return epoch, loss, metrics, config


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics."""
    # Convert to numpy arrays if needed
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    
    # Binary classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # ROC AUC (use probability for positive class)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.5  # Default for edge cases
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }


def save_metrics(metrics: Dict[str, float], filepath: str):
    """Save metrics to CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame([metrics])
    df.to_csv(filepath, index=False)
    print(f"Metrics saved to {filepath}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    save_path: str
):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str
):
    """Plot and save ROC curve."""
    from sklearn.metrics import roc_curve
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to {save_path}")


def save_training_history(history: Dict[str, list], filepath: str):
    """Save training history to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    history_serializable = {}
    for key, values in history.items():
        if isinstance(values, list):
            history_serializable[key] = [float(v) if isinstance(v, (np.floating, float)) else v for v in values]
        else:
            history_serializable[key] = float(values) if isinstance(values, (np.floating, float)) else values
    
    with open(filepath, 'w') as f:
        json.dump(history_serializable, f, indent=2)
    print(f"Training history saved to {filepath}")


def plot_training_history(history: Dict[str, list], save_path: str):
    """Plot and save training history."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history['train_accuracy'], label='Train Accuracy')
    axes[0, 1].plot(history['val_accuracy'], label='Val Accuracy')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[1, 0].plot(history['train_f1_score'], label='Train F1')
    axes[1, 0].plot(history['val_f1_score'], label='Val F1')
    axes[1, 0].set_title('F1 Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # ROC AUC
    axes[1, 1].plot(history['train_roc_auc'], label='Train ROC AUC')
    axes[1, 1].plot(history['val_roc_auc'], label='Val ROC AUC')
    axes[1, 1].set_title('ROC AUC')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('ROC AUC')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {save_path}")


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device_info() -> Dict[str, Any]:
    """Get information about available devices."""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None
    }
    
    if torch.cuda.is_available():
        info['device_name'] = torch.cuda.get_device_name()
        info['memory_allocated'] = torch.cuda.memory_allocated()
        info['memory_reserved'] = torch.cuda.memory_reserved()
    
    return info


def print_device_info():
    """Print device information."""
    info = get_device_info()
    print("Device Information:")
    print(f"  CUDA Available: {info['cuda_available']}")
    if info['cuda_available']:
        print(f"  Device Count: {info['device_count']}")
        print(f"  Current Device: {info['current_device']}")
        print(f"  Device Name: {info['device_name']}")
        print(f"  Memory Allocated: {info['memory_allocated'] / 1024**2:.2f} MB")
        print(f"  Memory Reserved: {info['memory_reserved'] / 1024**2:.2f} MB")
    else:
        print("  Using CPU")
