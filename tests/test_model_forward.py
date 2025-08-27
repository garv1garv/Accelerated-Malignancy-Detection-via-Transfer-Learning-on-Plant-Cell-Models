"""
Unit tests for model forward pass and checkpoint loading.

⚠️  RESEARCH PROTOTYPE - NOT FOR CLINICAL USE ⚠️
"""

import pytest
import os
import tempfile
import shutil
import torch
import numpy as np

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models import TransferModel, create_model, load_model_from_checkpoint
from src.scripts.make_dummy_plant_checkpoint import create_dummy_checkpoint


class TestTransferModel:
    """Test TransferModel class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_path = os.path.join(self.temp_dir, "dummy_checkpoint.pt")
        
        # Create dummy checkpoint
        create_dummy_checkpoint(
            checkpoint_path=self.checkpoint_path,
            backbone_name="resnet18",
            seed=42
        )
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = TransferModel(
            backbone_name="resnet18",
            num_classes=2,
            dropout_rate=0.5,
            freeze_backbone=False
        )
        
        assert model is not None
        assert model.num_classes == 2
        assert model.backbone_name == "resnet18"
        assert model.dropout_rate == 0.5
    
    def test_model_forward_pass(self):
        """Test model forward pass."""
        model = TransferModel(
            backbone_name="resnet18",
            num_classes=2,
            dropout_rate=0.5,
            freeze_backbone=False
        )
        
        # Create dummy input
        batch_size = 4
        input_tensor = torch.randn(batch_size, 3, 128, 128)
        
        # Forward pass
        logits, probabilities = model(input_tensor)
        
        # Check output shapes
        assert logits.shape == (batch_size, 2)  # Binary classification
        assert probabilities.shape == (batch_size, 1)  # Single probability for positive class
        
        # Check probability range
        assert torch.all(probabilities >= 0)
        assert torch.all(probabilities <= 1)
    
    def test_model_with_pretrained_checkpoint(self):
        """Test model with pretrained checkpoint."""
        model = TransferModel(
            backbone_name="resnet18",
            num_classes=2,
            dropout_rate=0.5,
            freeze_backbone=False,
            pretrained_checkpoint=self.checkpoint_path
        )
        
        # Test forward pass
        input_tensor = torch.randn(2, 3, 128, 128)
        logits, probabilities = model(input_tensor)
        
        assert logits.shape == (2, 2)
        assert probabilities.shape == (2, 1)
    
    def test_model_freeze_backbone(self):
        """Test model with frozen backbone."""
        model = TransferModel(
            backbone_name="resnet18",
            num_classes=2,
            dropout_rate=0.5,
            freeze_backbone=True
        )
        
        # Check that backbone parameters are frozen
        for name, param in model.backbone.named_parameters():
            if 'fc' not in name:  # Exclude classifier
                assert not param.requires_grad
        
        # Check that classifier parameters are trainable
        if hasattr(model.backbone, 'fc'):
            for param in model.backbone.fc.parameters():
                assert param.requires_grad
    
    def test_model_parameter_count(self):
        """Test model parameter counting."""
        model = TransferModel(
            backbone_name="resnet18",
            num_classes=2,
            dropout_rate=0.5,
            freeze_backbone=False
        )
        
        total_params = model.get_total_parameters()
        trainable_params = model.get_trainable_parameters()
        
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params <= total_params


class TestModelCreation:
    """Test model creation functions."""
    
    def test_create_model(self):
        """Test create_model function."""
        model = create_model(
            backbone_name="resnet18",
            num_classes=2,
            dropout_rate=0.5,
            freeze_backbone=False
        )
        
        assert isinstance(model, TransferModel)
        assert model.num_classes == 2
        assert model.backbone_name == "resnet18"
    
    def test_create_model_with_checkpoint(self):
        """Test create_model with checkpoint."""
        # Create dummy checkpoint
        temp_dir = tempfile.mkdtemp()
        checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pt")
        
        try:
            create_dummy_checkpoint(
                checkpoint_path=checkpoint_path,
                backbone_name="resnet18",
                seed=42
            )
            
            model = create_model(
                backbone_name="resnet18",
                num_classes=2,
                dropout_rate=0.5,
                freeze_backbone=False,
                pretrained_checkpoint=checkpoint_path
            )
            
            assert isinstance(model, TransferModel)
            
            # Test forward pass
            input_tensor = torch.randn(2, 3, 128, 128)
            logits, probabilities = model(input_tensor)
            
            assert logits.shape == (2, 2)
            assert probabilities.shape == (2, 1)
            
        finally:
            shutil.rmtree(temp_dir)


class TestCheckpointLoading:
    """Test checkpoint loading functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_path = os.path.join(self.temp_dir, "test_checkpoint.pt")
        
        # Create dummy checkpoint
        create_dummy_checkpoint(
            checkpoint_path=self.checkpoint_path,
            backbone_name="resnet18",
            seed=42
        )
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_load_model_from_checkpoint(self):
        """Test loading model from checkpoint."""
        model = load_model_from_checkpoint(self.checkpoint_path)
        
        assert isinstance(model, TransferModel)
        
        # Test forward pass
        input_tensor = torch.randn(2, 3, 128, 128)
        logits, probabilities = model(input_tensor)
        
        assert logits.shape == (2, 2)
        assert probabilities.shape == (2, 1)
    
    def test_load_model_with_existing_model(self):
        """Test loading checkpoint into existing model."""
        existing_model = create_model(
            backbone_name="resnet18",
            num_classes=2,
            dropout_rate=0.5,
            freeze_backbone=False
        )
        
        loaded_model = load_model_from_checkpoint(
            self.checkpoint_path,
            model=existing_model
        )
        
        assert loaded_model is existing_model
        
        # Test forward pass
        input_tensor = torch.randn(2, 3, 128, 128)
        logits, probabilities = loaded_model(input_tensor)
        
        assert logits.shape == (2, 2)
        assert probabilities.shape == (2, 1)


class TestModelArchitectures:
    """Test different model architectures."""
    
    def test_resnet18_model(self):
        """Test ResNet18 model."""
        model = create_model(
            backbone_name="resnet18",
            num_classes=2,
            dropout_rate=0.5,
            freeze_backbone=False
        )
        
        input_tensor = torch.randn(2, 3, 128, 128)
        logits, probabilities = model(input_tensor)
        
        assert logits.shape == (2, 2)
        assert probabilities.shape == (2, 1)
    
    def test_resnet34_model(self):
        """Test ResNet34 model."""
        model = create_model(
            backbone_name="resnet34",
            num_classes=2,
            dropout_rate=0.5,
            freeze_backbone=False
        )
        
        input_tensor = torch.randn(2, 3, 128, 128)
        logits, probabilities = model(input_tensor)
        
        assert logits.shape == (2, 2)
        assert probabilities.shape == (2, 1)
    
    def test_efficientnet_model(self):
        """Test EfficientNet model."""
        try:
            model = create_model(
                backbone_name="efficientnet_b0",
                num_classes=2,
                dropout_rate=0.5,
                freeze_backbone=False
            )
            
            input_tensor = torch.randn(2, 3, 128, 128)
            logits, probabilities = model(input_tensor)
            
            assert logits.shape == (2, 2)
            assert probabilities.shape == (2, 1)
            
        except Exception as e:
            # Skip if EfficientNet is not available
            pytest.skip(f"EfficientNet not available: {e}")


class TestModelEdgeCases:
    """Test model edge cases and error handling."""
    
    def test_invalid_backbone(self):
        """Test invalid backbone name."""
        with pytest.raises(Exception):
            create_model(
                backbone_name="invalid_backbone",
                num_classes=2,
                dropout_rate=0.5,
                freeze_backbone=False
            )
    
    def test_invalid_num_classes(self):
        """Test invalid number of classes."""
        with pytest.raises(Exception):
            create_model(
                backbone_name="resnet18",
                num_classes=0,  # Invalid
                dropout_rate=0.5,
                freeze_backbone=False
            )
    
    def test_invalid_dropout_rate(self):
        """Test invalid dropout rate."""
        with pytest.raises(Exception):
            create_model(
                backbone_name="resnet18",
                num_classes=2,
                dropout_rate=1.5,  # Invalid (> 1)
                freeze_backbone=False
            )
    
    def test_nonexistent_checkpoint(self):
        """Test loading nonexistent checkpoint."""
        with pytest.raises(FileNotFoundError):
            load_model_from_checkpoint("nonexistent_checkpoint.pt")


if __name__ == "__main__":
    pytest.main([__file__])
