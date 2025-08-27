"""
Evaluation module for the Plant2Skin transfer learning pipeline.

⚠️  RESEARCH PROTOTYPE - NOT FOR CLINICAL USE ⚠️
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from .config import config
from .models import load_model_from_checkpoint, create_model
from .data_utils import create_dataloaders
from .utils import (
    compute_metrics, save_metrics, plot_confusion_matrix,
    plot_roc_curve, print_device_info
)


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "auto",
    save_dir: str = "outputs"
) -> Dict[str, float]:
    """
    Evaluate the model on the test set.
    
    Args:
        model: Trained model to evaluate
        test_loader: Test data loader
        device: Device to evaluate on
        save_dir: Directory to save results
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Set device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    print(f"Evaluating on device: {device}")
    print_device_info()
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Setup loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Evaluation
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_image_paths = []
    
    print("Evaluating model on test set...")
    progress_bar = tqdm(test_loader, desc="Evaluation")
    
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
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Compute metrics
    avg_loss = total_loss / len(test_loader)
    metrics = compute_metrics(all_labels, all_predictions, all_probabilities)
    metrics['loss'] = avg_loss
    
    # Print results
    print(f"\nTest Set Evaluation Results:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
    
    # Save results
    os.makedirs(save_dir, exist_ok=True)
    
    # Save metrics to CSV
    metrics_path = os.path.join(save_dir, "test_metrics.csv")
    save_metrics(metrics, metrics_path)
    
    # Create detailed results DataFrame
    results_df = pd.DataFrame({
        'true_label': all_labels,
        'predicted_label': all_predictions,
        'probability': all_probabilities,
        'correct': all_labels == all_predictions
    })
    
    # Save detailed results
    results_path = os.path.join(save_dir, "test_results.csv")
    results_df.to_csv(results_path, index=False)
    
    # Create visualizations
    create_evaluation_plots(
        all_labels, all_predictions, all_probabilities,
        config.class_names, save_dir
    )
    
    # Print per-class metrics
    print_per_class_metrics(all_labels, all_predictions, all_probabilities)
    
    return metrics


def create_evaluation_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    save_dir: str
):
    """Create and save evaluation plots."""
    print("Creating evaluation plots...")
    
    # Confusion matrix
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path)
    
    # ROC curve
    roc_path = os.path.join(save_dir, "roc_curve.png")
    plot_roc_curve(y_true, y_prob, roc_path)
    
    # Probability distribution
    prob_path = os.path.join(save_dir, "probability_distribution.png")
    plot_probability_distribution(y_true, y_prob, class_names, prob_path)
    
    # Metrics summary
    summary_path = os.path.join(save_dir, "metrics_summary.png")
    plot_metrics_summary(y_true, y_pred, y_prob, summary_path)
    
    print(f"Evaluation plots saved to: {save_dir}")


def plot_probability_distribution(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    save_path: str
):
    """Plot probability distribution for each class."""
    plt.figure(figsize=(12, 6))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot for each class
    for i, class_name in enumerate(class_names):
        mask = y_true == i
        if np.any(mask):
            ax1.hist(y_prob[mask], bins=20, alpha=0.7, label=f'True {class_name}', density=True)
    
    ax1.set_xlabel('Predicted Probability')
    ax1.set_ylabel('Density')
    ax1.set_title('Probability Distribution by True Class')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot for each predicted class
    for i, class_name in enumerate(class_names):
        mask = (y_prob > 0.5) == i
        if np.any(mask):
            ax2.hist(y_prob[mask], bins=20, alpha=0.7, label=f'Predicted {class_name}', density=True)
    
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Density')
    ax2.set_title('Probability Distribution by Predicted Class')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    save_path: str
):
    """Create a comprehensive metrics summary plot."""
    from sklearn.metrics import classification_report
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_prob)
    
    # Get classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Overall metrics bar plot
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
    metric_values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1_score'],
        metrics['roc_auc']
    ]
    
    bars = ax1.bar(metric_names, metric_values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'])
    ax1.set_ylabel('Score')
    ax1.set_title('Overall Model Performance')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # 2. Per-class precision and recall
    if len(report) > 3:  # Has per-class metrics
        classes = list(report.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
        precisions = [report[cls]['precision'] for cls in classes]
        recalls = [report[cls]['recall'] for cls in classes]
        
        x = np.arange(len(classes))
        width = 0.35
        
        ax2.bar(x - width/2, precisions, width, label='Precision', alpha=0.8)
        ax2.bar(x + width/2, recalls, width, label='Recall', alpha=0.8)
        
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Score')
        ax2.set_title('Per-Class Precision and Recall')
        ax2.set_xticks(x)
        ax2.set_xticklabels(classes)
        ax2.legend()
        ax2.set_ylim(0, 1)
    
    # 3. Confusion matrix heatmap
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
    ax3.set_title('Confusion Matrix')
    ax3.set_ylabel('True Label')
    ax3.set_xlabel('Predicted Label')
    
    # 4. ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = metrics['roc_auc']
    
    ax4.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax4.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax4.set_xlim([0.0, 1.0])
    ax4.set_ylim([0.0, 1.05])
    ax4.set_xlabel('False Positive Rate')
    ax4.set_ylabel('True Positive Rate')
    ax4.set_title('ROC Curve')
    ax4.legend(loc="lower right")
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def print_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
):
    """Print detailed per-class metrics."""
    from sklearn.metrics import classification_report
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=config.class_names))
    
    # Per-class statistics
    for i, class_name in enumerate(config.class_names):
        mask = y_true == i
        if np.any(mask):
            class_probs = y_prob[mask]
            class_preds = y_pred[mask]
            
            print(f"\n{class_name.capitalize()} Class Statistics:")
            print(f"  Count: {np.sum(mask)}")
            print(f"  Mean probability: {np.mean(class_probs):.4f}")
            print(f"  Std probability: {np.std(class_probs):.4f}")
            print(f"  Min probability: {np.min(class_probs):.4f}")
            print(f"  Max probability: {np.max(class_probs):.4f}")
            print(f"  Correct predictions: {np.sum(class_preds == i)}")
            print(f"  Accuracy: {np.mean(class_preds == i):.4f}")


def evaluate_from_checkpoint(
    checkpoint_path: str,
    test_loader: Optional[DataLoader] = None,
    device: str = "auto",
    save_dir: str = "outputs"
) -> Dict[str, float]:
    """
    Evaluate a model from a checkpoint file.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        test_loader: Test data loader (if None, will create one)
        device: Device to evaluate on
        save_dir: Directory to save results
    
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Load model
    model = load_model_from_checkpoint(checkpoint_path)
    
    # Create test loader if not provided
    if test_loader is None:
        _, _, test_loader = create_dataloaders(
            data_root=config.data_root,
            batch_size=config.batch_size,
            image_size=config.image_size,
            num_workers=config.num_workers,
            use_augmentation=False  # No augmentation for evaluation
        )
    
    # Evaluate model
    metrics = evaluate_model(model, test_loader, device, save_dir)
    
    return metrics


def main():
    """Main evaluation function."""
    print("⚠️  RESEARCH PROTOTYPE - NOT FOR CLINICAL USE ⚠️")
    print("Starting Plant2Skin transfer learning evaluation...")
    
    # Check if best model exists
    best_model_path = os.path.join(config.output_dir, "best_model.pt")
    
    if not os.path.exists(best_model_path):
        print(f"Best model not found at {best_model_path}")
        print("Please run training first or specify a checkpoint path.")
        return
    
    # Evaluate model
    metrics = evaluate_from_checkpoint(
        checkpoint_path=best_model_path,
        device=config.device,
        save_dir=config.output_dir
    )
    
    print(f"\nEvaluation completed!")
    print(f"Results saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
