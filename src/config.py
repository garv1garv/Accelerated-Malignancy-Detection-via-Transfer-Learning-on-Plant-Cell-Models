"""
Configuration settings for the Plant2Skin transfer learning pipeline.

⚠️  RESEARCH PROTOTYPE - NOT FOR CLINICAL USE ⚠️
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple
import torch


@dataclass
class Config:
    """Configuration class for the transfer learning pipeline."""
    
    # Data settings
    data_root: str = "data/synthetic"
    image_size: Tuple[int, int] = (128, 128)
    num_classes: int = 2  # Binary classification: benign vs malignant
    num_channels: int = 3  # RGB images
    
    # Training settings
    batch_size: int = 16
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    
    # Model settings
    backbone_name: str = "resnet18"
    pretrained_checkpoint: str = "checkpoints/plant_cell_pretrained.pt"
    freeze_backbone: bool = False
    dropout_rate: float = 0.5
    
    # Transfer learning settings
    transfer_strategy: str = "fine_tune"  # Options: "fine_tune", "feature_extraction"
    unfreeze_layers: Optional[int] = None  # Number of layers to unfreeze from end
    
    # Augmentation settings
    use_augmentation: bool = True
    augmentation_prob: float = 0.5
    
    # Device settings
    device: str = "auto"  # "auto", "cuda", "cpu"
    
    # Output settings
    output_dir: str = "outputs"
    save_best_model: bool = True
    save_checkpoints: bool = True
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    # Validation settings
    val_split: float = 0.2
    test_split: float = 0.2
    
    def __post_init__(self):
        """Set up device and create output directories."""
        # Set device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "explain"), exist_ok=True)
        
        # Set random seed
        self._set_random_seed()
    
    def _set_random_seed(self):
        """Set random seed for reproducibility."""
        import random
        import numpy as np
        
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)
    
    @property
    def class_names(self) -> list:
        """Get class names for binary classification."""
        return ["benign", "malignant"]
    
    @property
    def num_workers(self) -> int:
        """Get number of workers for data loading."""
        return min(4, os.cpu_count() or 1)
    
    def get_model_save_path(self, filename: str = "best_model.pt") -> str:
        """Get path for saving model checkpoints."""
        return os.path.join(self.output_dir, filename)
    
    def get_explanation_save_path(self, filename: str) -> str:
        """Get path for saving explanation visualizations."""
        return os.path.join(self.output_dir, "explain", filename)


# Default configuration instance
config = Config()


def update_config(**kwargs):
    """Update configuration with new values."""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown config key: {key}")
    
    # Re-run post-init
    config.__post_init__()
    return config
