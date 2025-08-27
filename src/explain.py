"""
Explanation module for the Plant2Skin transfer learning pipeline.

⚠️  RESEARCH PROTOTYPE - NOT FOR CLINICAL USE ⚠️
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm

from .config import config
from .models import load_model_from_checkpoint
from .data_utils import create_dataloaders


class GradCAM:
    """
    Grad-CAM implementation for model interpretability.
    
    This class implements the Grad-CAM algorithm to generate class activation maps
    that highlight the regions of the input image that are most important for
    the model's prediction.
    """
    
    def __init__(self, model: nn.Module, target_layer: Optional[str] = None):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Trained model
            target_layer: Name of the target layer for Grad-CAM
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks to capture gradients and activations."""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Find target layer
        target_module = self._find_target_layer()
        if target_module is not None:
            target_module.register_forward_hook(forward_hook)
            target_module.register_backward_hook(backward_hook)
            print(f"Registered hooks for layer: {self.target_layer}")
        else:
            print("Warning: Could not find target layer for Grad-CAM")
    
    def _find_target_layer(self) -> Optional[nn.Module]:
        """Find the target layer for Grad-CAM."""
        if self.target_layer is None:
            # Try to find a suitable layer automatically
            if hasattr(self.model.backbone, 'layer4'):
                # ResNet-style
                return self.model.backbone.layer4
            elif hasattr(self.model.backbone, 'blocks'):
                # EfficientNet-style
                return self.model.backbone.blocks[-1]
            elif hasattr(self.model.backbone, 'features'):
                # VGG-style
                return self.model.backbone.features[-1]
            else:
                return None
        else:
            # Find layer by name
            for name, module in self.model.named_modules():
                if name == self.target_layer:
                    return module
            return None
    
    def generate_cam(
        self,
        input_image: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate class activation map for the input image.
        
        Args:
            input_image: Input image tensor
            target_class: Target class index (if None, use predicted class)
        
        Returns:
            Class activation map as numpy array
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Forward pass
        input_image.requires_grad_(True)
        logits, probabilities = self.model(input_image)
        
        # Get target class
        if target_class is None:
            if logits.shape[1] == 2:
                target_class = (logits[:, 1] > 0).long()
            else:
                target_class = (logits.squeeze() > 0).long()
        
        # Backward pass
        self.model.zero_grad()
        if logits.shape[1] == 2:
            logits[:, target_class].backward(torch.ones_like(logits[:, target_class]))
        else:
            logits.squeeze().backward(torch.ones_like(logits.squeeze()))
        
        # Check if we have gradients and activations
        if self.gradients is None or self.activations is None:
            raise ValueError("Gradients or activations not captured. Check target layer.")
        
        # Compute weights
        weights = torch.mean(self.gradients, dim=[2, 3])
        
        # Generate CAM
        cam = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * self.activations, dim=1)
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.detach().cpu().numpy()
    
    def overlay_cam(
        self,
        image: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Overlay CAM on the original image.
        
        Args:
            image: Original image (H, W, C)
            cam: Class activation map (H, W)
            alpha: Transparency factor
        
        Returns:
            Overlaid image
        """
        # Resize CAM to match image size
        cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
        
        # Convert to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        
        # Overlay
        overlaid = heatmap * alpha + image * (1 - alpha)
        overlaid = np.clip(overlaid, 0, 1)
        
        return overlaid


def generate_gradcam_explanations(
    model_path: str,
    test_loader: Optional[DataLoader] = None,
    num_samples: int = 10,
    save_dir: str = "outputs/explain",
    device: str = "auto"
):
    """
    Generate Grad-CAM explanations for test samples.
    
    Args:
        model_path: Path to the trained model
        test_loader: Test data loader
        num_samples: Number of samples to explain
        save_dir: Directory to save explanations
        device: Device to run on
    """
    print("⚠️  RESEARCH PROTOTYPE - NOT FOR CLINICAL USE ⚠️")
    print("Generating Grad-CAM explanations...")
    
    # Set device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    # Load model
    model = load_model_from_checkpoint(model_path)
    model = model.to(device)
    model.eval()
    
    # Create test loader if not provided
    if test_loader is None:
        _, _, test_loader = create_dataloaders(
            data_root=config.data_root,
            batch_size=1,  # Process one image at a time
            image_size=config.image_size,
            num_workers=config.num_workers,
            use_augmentation=False
        )
    
    # Initialize Grad-CAM
    grad_cam = GradCAM(model)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate explanations
    sample_count = 0
    progress_bar = tqdm(test_loader, desc="Generating explanations")
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        if sample_count >= num_samples:
            break
        
        images = images.to(device)
        labels = labels.to(device)
        
        # Get predictions
        with torch.no_grad():
            logits, probabilities = model(images)
            if logits.shape[1] == 2:
                predictions = (logits[:, 1] > 0).long()
                probs = probabilities[:, 1] if probabilities.shape[1] == 2 else probabilities.squeeze()
            else:
                predictions = (logits.squeeze() > 0).long()
                probs = probabilities.squeeze()
        
        # Generate CAM for each image in batch
        for i in range(images.shape[0]):
            if sample_count >= num_samples:
                break
            
            # Get single image
            image = images[i:i+1]
            label = labels[i].item()
            pred = predictions[i].item()
            prob = probs[i].item()
            
            try:
                # Generate CAM
                cam = grad_cam.generate_cam(image, target_class=pred)
                
                # Convert image to numpy for visualization
                img_np = image.squeeze().cpu().numpy().transpose(1, 2, 0)
                
                # Denormalize image
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_np = img_np * std + mean
                img_np = np.clip(img_np, 0, 1)
                
                # Overlay CAM
                overlaid = grad_cam.overlay_cam(img_np, cam[0])
                
                # Create visualization
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Original image
                axes[0].imshow(img_np)
                axes[0].set_title(f'Original Image\nTrue: {config.class_names[label]}\nPred: {config.class_names[pred]} ({prob:.3f})')
                axes[0].axis('off')
                
                # CAM
                axes[1].imshow(cam[0], cmap='jet')
                axes[1].set_title('Grad-CAM Heatmap')
                axes[1].axis('off')
                
                # Overlaid
                axes[2].imshow(overlaid)
                axes[2].set_title('Grad-CAM Overlay')
                axes[2].axis('off')
                
                plt.tight_layout()
                
                # Save explanation
                save_path = os.path.join(save_dir, f"explanation_{sample_count:03d}.png")
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                sample_count += 1
                progress_bar.set_postfix({'saved': sample_count})
                
            except Exception as e:
                print(f"Error generating explanation for sample {sample_count}: {e}")
                continue
    
    print(f"\nGenerated {sample_count} Grad-CAM explanations")
    print(f"Explanations saved to: {save_dir}")


def generate_class_comparison_explanations(
    model_path: str,
    test_loader: Optional[DataLoader] = None,
    num_samples: int = 5,
    save_dir: str = "outputs/explain",
    device: str = "auto"
):
    """
    Generate Grad-CAM explanations comparing both classes.
    
    Args:
        model_path: Path to the trained model
        test_loader: Test data loader
        num_samples: Number of samples per class
        save_dir: Directory to save explanations
        device: Device to run on
    """
    print("Generating class comparison explanations...")
    
    # Set device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    # Load model
    model = load_model_from_checkpoint(model_path)
    model = model.to(device)
    model.eval()
    
    # Create test loader if not provided
    if test_loader is None:
        _, _, test_loader = create_dataloaders(
            data_root=config.data_root,
            batch_size=1,
            image_size=config.image_size,
            num_workers=config.num_workers,
            use_augmentation=False
        )
    
    # Initialize Grad-CAM
    grad_cam = GradCAM(model)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Collect samples by class
    class_samples = {0: [], 1: []}  # benign, malignant
    
    for images, labels in test_loader:
        for i in range(images.shape[0]):
            label = labels[i].item()
            if len(class_samples[label]) < num_samples:
                class_samples[label].append((images[i:i+1], label))
    
    # Generate explanations for each class
    for class_idx, samples in class_samples.items():
        class_name = config.class_names[class_idx]
        print(f"Generating explanations for {class_name} class...")
        
        for sample_idx, (image, label) in enumerate(samples):
            image = image.to(device)
            
            try:
                # Generate CAM for both classes
                cams = {}
                for target_class in [0, 1]:
                    cam = grad_cam.generate_cam(image, target_class=target_class)
                    cams[target_class] = cam[0]
                
                # Convert image to numpy
                img_np = image.squeeze().cpu().numpy().transpose(1, 2, 0)
                
                # Denormalize
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_np = img_np * std + mean
                img_np = np.clip(img_np, 0, 1)
                
                # Create visualization
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                
                # Original image
                axes[0].imshow(img_np)
                axes[0].set_title(f'Original Image\nTrue: {class_name}')
                axes[0].axis('off')
                
                # CAM for benign
                axes[1].imshow(cams[0], cmap='jet')
                axes[1].set_title(f'Grad-CAM for Benign')
                axes[1].axis('off')
                
                # CAM for malignant
                axes[2].imshow(cams[1], cmap='jet')
                axes[2].set_title(f'Grad-CAM for Malignant')
                axes[2].axis('off')
                
                # Difference
                diff = cams[1] - cams[0]
                axes[3].imshow(diff, cmap='RdBu_r', vmin=-1, vmax=1)
                axes[3].set_title('Difference (Malignant - Benign)')
                axes[3].axis('off')
                
                plt.tight_layout()
                
                # Save explanation
                save_path = os.path.join(save_dir, f"class_comparison_{class_name}_{sample_idx:02d}.png")
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"Error generating explanation for {class_name} sample {sample_idx}: {e}")
                continue
    
    print(f"Class comparison explanations saved to: {save_dir}")


def main():
    """Main explanation function."""
    print("⚠️  RESEARCH PROTOTYPE - NOT FOR CLINICAL USE ⚠️")
    print("Starting Grad-CAM explanation generation...")
    
    # Check if best model exists
    best_model_path = os.path.join(config.output_dir, "best_model.pt")
    
    if not os.path.exists(best_model_path):
        print(f"Best model not found at {best_model_path}")
        print("Please run training first or specify a checkpoint path.")
        return
    
    # Generate explanations
    generate_gradcam_explanations(
        model_path=best_model_path,
        num_samples=10,
        save_dir=config.get_explanation_save_path(""),
        device=config.device
    )
    
    # Generate class comparison explanations
    generate_class_comparison_explanations(
        model_path=best_model_path,
        num_samples=3,
        save_dir=config.get_explanation_save_path(""),
        device=config.device
    )
    
    print("Explanation generation completed!")


if __name__ == "__main__":
    main()
