"""
Data utilities for the Plant2Skin transfer learning pipeline.

⚠️  RESEARCH PROTOTYPE - NOT FOR CLINICAL USE ⚠️
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .config import config


class SkinLesionDataset(Dataset):
    """Dataset class for skin lesion images with synthetic data support."""
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        transform: Optional[A.Compose] = None,
        class_names: Optional[list] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            data_root: Root directory containing the dataset
            split: Dataset split ('train', 'val', 'test')
            transform: Albumentations transform pipeline
            class_names: List of class names
        """
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.class_names = class_names or ["benign", "malignant"]
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        print(f"Loaded {len(self.metadata)} samples for {split} split")
    
    def _load_metadata(self) -> pd.DataFrame:
        """Load dataset metadata from CSV file."""
        csv_path = os.path.join(self.data_root, f"{self.split}_metadata.csv")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata file not found: {csv_path}")
        
        metadata = pd.read_csv(csv_path)
        
        # Validate required columns
        required_cols = ['image_path', 'label', 'class_name']
        missing_cols = [col for col in required_cols if col not in metadata.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return metadata
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample."""
        row = self.metadata.iloc[idx]
        
        # Load image
        image_path = os.path.join(self.data_root, row['image_path'])
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        # Get label
        label = int(row['label'])
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, label
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for handling imbalanced datasets."""
        class_counts = self.metadata['label'].value_counts().sort_index()
        total_samples = len(self.metadata)
        
        # Compute inverse frequency weights
        weights = total_samples / (len(class_counts) * class_counts)
        return torch.FloatTensor(weights.values)


def get_transforms(
    image_size: Tuple[int, int] = (128, 128),
    is_training: bool = True,
    augmentation_prob: float = 0.5
) -> A.Compose:
    """
    Get data augmentation transforms.
    
    Args:
        image_size: Target image size (height, width)
        is_training: Whether to apply training augmentations
        augmentation_prob: Probability of applying augmentations
    
    Returns:
        Albumentations transform pipeline
    """
    if is_training:
        # Training transforms with augmentations
        transform = A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.HorizontalFlip(p=augmentation_prob),
            A.VerticalFlip(p=augmentation_prob * 0.5),
            A.RandomRotate90(p=augmentation_prob * 0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=augmentation_prob
            ),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=1.0
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            ], p=augmentation_prob),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=augmentation_prob * 0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        # Validation/test transforms (no augmentation)
        transform = A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    return transform


def create_dataloaders(
    data_root: str = "data/synthetic",
    batch_size: int = 16,
    image_size: Tuple[int, int] = (128, 128),
    num_workers: int = 4,
    use_augmentation: bool = True,
    augmentation_prob: float = 0.5,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        data_root: Root directory containing the dataset
        batch_size: Batch size for data loaders
        image_size: Target image size (height, width)
        num_workers: Number of workers for data loading
        use_augmentation: Whether to use data augmentation
        augmentation_prob: Probability of applying augmentations
        shuffle_train: Whether to shuffle training data
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Get transforms
    train_transform = get_transforms(
        image_size=image_size,
        is_training=use_augmentation,
        augmentation_prob=augmentation_prob
    )
    
    val_transform = get_transforms(
        image_size=image_size,
        is_training=False
    )
    
    # Create datasets
    train_dataset = SkinLesionDataset(
        data_root=data_root,
        split="train",
        transform=train_transform
    )
    
    val_dataset = SkinLesionDataset(
        data_root=data_root,
        split="val",
        transform=val_transform
    )
    
    test_dataset = SkinLesionDataset(
        data_root=data_root,
        split="test",
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Created data loaders:")
    print(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"  Val: {len(val_loader)} batches ({len(val_dataset)} samples)")
    print(f"  Test: {len(test_loader)} batches ({len(test_dataset)} samples)")
    
    return train_loader, val_loader, test_loader


def get_dataset_info(data_root: str) -> Dict[str, Any]:
    """Get information about the dataset."""
    info = {}
    
    for split in ["train", "val", "test"]:
        csv_path = os.path.join(data_root, f"{split}_metadata.csv")
        if os.path.exists(csv_path):
            metadata = pd.read_csv(csv_path)
            info[split] = {
                'total_samples': len(metadata),
                'class_distribution': metadata['label'].value_counts().to_dict(),
                'class_names': metadata['class_name'].unique().tolist()
            }
    
    return info


def print_dataset_info(data_root: str):
    """Print dataset information."""
    info = get_dataset_info(data_root)
    
    print("Dataset Information:")
    for split, split_info in info.items():
        print(f"  {split.capitalize()} Split:")
        print(f"    Total samples: {split_info['total_samples']}")
        print(f"    Class distribution: {split_info['class_distribution']}")
        print(f"    Classes: {split_info['class_names']}")
        print()


def validate_dataset(data_root: str) -> bool:
    """Validate that the dataset is properly structured."""
    try:
        # Check if metadata files exist
        for split in ["train", "val", "test"]:
            csv_path = os.path.join(data_root, f"{split}_metadata.csv")
            if not os.path.exists(csv_path):
                print(f"Missing metadata file: {csv_path}")
                return False
        
        # Check if images exist
        for split in ["train", "val", "test"]:
            csv_path = os.path.join(data_root, f"{split}_metadata.csv")
            metadata = pd.read_csv(csv_path)
            
            for _, row in metadata.iterrows():
                image_path = os.path.join(data_root, row['image_path'])
                if not os.path.exists(image_path):
                    print(f"Missing image: {image_path}")
                    return False
        
        print("Dataset validation passed!")
        return True
        
    except Exception as e:
        print(f"Dataset validation failed: {e}")
        return False
