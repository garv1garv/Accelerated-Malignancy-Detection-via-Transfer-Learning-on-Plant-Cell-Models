"""
Unit tests for data utilities and synthetic dataset generation.

⚠️  RESEARCH PROTOTYPE - NOT FOR CLINICAL USE ⚠️
"""

import pytest
import os
import tempfile
import shutil
import numpy as np
import pandas as pd
import torch
from PIL import Image

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data_utils import SkinLesionDataset, create_dataloaders, get_transforms
from src.scripts.generate_synthetic_data import generate_synthetic_dataset


class TestSyntheticDataGeneration:
    """Test synthetic data generation functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_root = os.path.join(self.temp_dir, "test_data")
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_generate_synthetic_dataset(self):
        """Test synthetic dataset generation."""
        # Generate small test dataset
        generate_synthetic_dataset(
            data_root=self.data_root,
            num_samples=20,  # Small number for testing
            image_size=64,   # Small size for testing
            seed=42
        )
        
        # Check directory structure
        assert os.path.exists(self.data_root)
        
        splits = ['train', 'val', 'test']
        classes = ['benign', 'malignant']
        
        for split in splits:
            for class_name in classes:
                split_class_dir = os.path.join(self.data_root, split, class_name)
                assert os.path.exists(split_class_dir)
        
        # Check metadata files
        for split in splits:
            metadata_path = os.path.join(self.data_root, f"{split}_metadata.csv")
            assert os.path.exists(metadata_path)
            
            # Load and verify metadata
            df = pd.read_csv(metadata_path)
            assert len(df) > 0
            assert all(col in df.columns for col in ['image_path', 'label', 'class_name'])
            
            # Check that images exist
            for _, row in df.iterrows():
                image_path = os.path.join(self.data_root, row['image_path'])
                assert os.path.exists(image_path)
                
                # Check image can be loaded
                img = Image.open(image_path)
                assert img.size == (64, 64)
                assert img.mode == 'RGB'
    
    def test_dataset_class_distribution(self):
        """Test that dataset has balanced class distribution."""
        generate_synthetic_dataset(
            data_root=self.data_root,
            num_samples=20,
            image_size=64,
            seed=42
        )
        
        for split in ['train', 'val', 'test']:
            metadata_path = os.path.join(self.data_root, f"{split}_metadata.csv")
            df = pd.read_csv(metadata_path)
            
            # Check class distribution
            class_counts = df['label'].value_counts()
            assert len(class_counts) == 2  # Two classes
            
            # Check approximate balance (allowing for rounding)
            total_samples = len(df)
            expected_per_class = total_samples // 2
            for count in class_counts.values:
                assert abs(count - expected_per_class) <= 1


class TestSkinLesionDataset:
    """Test SkinLesionDataset class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_root = os.path.join(self.temp_dir, "test_data")
        
        # Generate test dataset
        generate_synthetic_dataset(
            data_root=self.data_root,
            num_samples=20,
            image_size=64,
            seed=42
        )
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_dataset_initialization(self):
        """Test dataset initialization."""
        dataset = SkinLesionDataset(
            data_root=self.data_root,
            split="train"
        )
        
        assert len(dataset) > 0
        assert dataset.class_names == ["benign", "malignant"]
    
    def test_dataset_getitem(self):
        """Test dataset item retrieval."""
        dataset = SkinLesionDataset(
            data_root=self.data_root,
            split="train"
        )
        
        # Get first item
        image, label = dataset[0]
        
        # Check image tensor
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 64, 64)  # C, H, W
        assert image.dtype == torch.float32
        
        # Check label
        assert isinstance(label, int)
        assert label in [0, 1]
    
    def test_dataset_with_transforms(self):
        """Test dataset with transforms."""
        transform = get_transforms(image_size=(64, 64), is_training=True)
        
        dataset = SkinLesionDataset(
            data_root=self.data_root,
            split="train",
            transform=transform
        )
        
        image, label = dataset[0]
        
        # Check transformed image
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 64, 64)
        assert image.dtype == torch.float32
        
        # Check normalization (should be roughly in [-1, 1] range)
        assert image.min() >= -3
        assert image.max() <= 3
    
    def test_class_weights(self):
        """Test class weights computation."""
        dataset = SkinLesionDataset(
            data_root=self.data_root,
            split="train"
        )
        
        weights = dataset.get_class_weights()
        
        assert isinstance(weights, torch.Tensor)
        assert weights.shape == (2,)  # Two classes
        assert torch.all(weights > 0)  # All weights should be positive


class TestDataLoaders:
    """Test data loader creation."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_root = os.path.join(self.temp_dir, "test_data")
        
        # Generate test dataset
        generate_synthetic_dataset(
            data_root=self.data_root,
            num_samples=20,
            image_size=64,
            seed=42
        )
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_create_dataloaders(self):
        """Test dataloader creation."""
        train_loader, val_loader, test_loader = create_dataloaders(
            data_root=self.data_root,
            batch_size=4,
            image_size=(64, 64),
            num_workers=0,  # Use 0 for testing
            use_augmentation=False  # Disable augmentation for testing
        )
        
        # Check loaders exist
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None
        
        # Check batch iteration
        for batch_idx, (images, labels) in enumerate(train_loader):
            assert images.shape[0] <= 4  # Batch size
            assert images.shape[1:] == (3, 64, 64)  # C, H, W
            assert labels.shape[0] == images.shape[0]
            assert torch.all((labels == 0) | (labels == 1))
            
            if batch_idx >= 2:  # Test first few batches
                break
    
    def test_dataloader_with_augmentation(self):
        """Test dataloader with augmentation."""
        train_loader, val_loader, test_loader = create_dataloaders(
            data_root=self.data_root,
            batch_size=4,
            image_size=(64, 64),
            num_workers=0,
            use_augmentation=True,
            augmentation_prob=0.5
        )
        
        # Check that training loader has augmentation
        for batch_idx, (images, labels) in enumerate(train_loader):
            assert images.shape == (images.shape[0], 3, 64, 64)
            if batch_idx >= 1:
                break


class TestTransforms:
    """Test data transformation functions."""
    
    def test_get_transforms_training(self):
        """Test training transforms."""
        transform = get_transforms(
            image_size=(64, 64),
            is_training=True,
            augmentation_prob=0.5
        )
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        # Apply transform
        transformed = transform(image=dummy_image)
        image_tensor = transformed['image']
        
        assert isinstance(image_tensor, torch.Tensor)
        assert image_tensor.shape == (3, 64, 64)
        assert image_tensor.dtype == torch.float32
    
    def test_get_transforms_validation(self):
        """Test validation transforms."""
        transform = get_transforms(
            image_size=(64, 64),
            is_training=False
        )
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        # Apply transform
        transformed = transform(image=dummy_image)
        image_tensor = transformed['image']
        
        assert isinstance(image_tensor, torch.Tensor)
        assert image_tensor.shape == (3, 64, 64)
        assert image_tensor.dtype == torch.float32


if __name__ == "__main__":
    pytest.main([__file__])
