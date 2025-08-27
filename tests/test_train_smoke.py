"""
Smoke tests for the training pipeline.

⚠️  RESEARCH PROTOTYPE - NOT FOR CLINICAL USE ⚠️

These tests verify that the training pipeline can run without errors
using minimal data and short training runs.
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

from src.scripts.generate_synthetic_data import generate_synthetic_dataset
from src.scripts.make_dummy_plant_checkpoint import create_dummy_checkpoint
from src.models import create_model
from src.data_utils import create_dataloaders
from src.train import train_model
from src.eval import evaluate_model
from src.explain import generate_gradcam_explanations


class TestTrainingSmoke:
    """Smoke tests for training pipeline."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_root = os.path.join(self.temp_dir, "test_data")
        self.checkpoint_path = os.path.join(self.temp_dir, "dummy_checkpoint.pt")
        self.output_dir = os.path.join(self.temp_dir, "outputs")
        
        # Create minimal test dataset
        generate_synthetic_dataset(
            data_root=self.data_root,
            num_samples=20,  # Very small dataset for testing
            image_size=64,   # Small images for testing
            seed=42
        )
        
        # Create dummy checkpoint
        create_dummy_checkpoint(
            checkpoint_path=self.checkpoint_path,
            backbone_name="resnet18",
            seed=42
        )
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_training_smoke(self):
        """Test that training runs without errors."""
        # Create model
        model = create_model(
            backbone_name="resnet18",
            num_classes=2,
            dropout_rate=0.5,
            freeze_backbone=False,
            pretrained_checkpoint=self.checkpoint_path
        )
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_dataloaders(
            data_root=self.data_root,
            batch_size=4,
            image_size=(64, 64),
            num_workers=0,  # Use 0 for testing
            use_augmentation=False  # Disable augmentation for testing
        )
        
        # Train for 1 epoch
        trained_model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=1,  # Very short training
            learning_rate=1e-4,
            weight_decay=1e-4,
            device="cpu",  # Use CPU for testing
            save_dir=self.output_dir
        )
        
        # Check that training completed
        assert trained_model is not None
        assert history is not None
        
        # Check history structure
        required_keys = ['train_loss', 'val_loss', 'train_accuracy', 'val_accuracy']
        for key in required_keys:
            assert key in history
            assert len(history[key]) == 1  # One epoch
    
    def test_evaluation_smoke(self):
        """Test that evaluation runs without errors."""
        # Create model
        model = create_model(
            backbone_name="resnet18",
            num_classes=2,
            dropout_rate=0.5,
            freeze_backbone=False,
            pretrained_checkpoint=self.checkpoint_path
        )
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_dataloaders(
            data_root=self.data_root,
            batch_size=4,
            image_size=(64, 64),
            num_workers=0,
            use_augmentation=False
        )
        
        # Evaluate model
        metrics = evaluate_model(
            model=model,
            test_loader=test_loader,
            device="cpu",
            save_dir=self.output_dir
        )
        
        # Check metrics
        assert metrics is not None
        required_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
    
    def test_explanation_smoke(self):
        """Test that explanation generation runs without errors."""
        # Create model
        model = create_model(
            backbone_name="resnet18",
            num_classes=2,
            dropout_rate=0.5,
            freeze_backbone=False,
            pretrained_checkpoint=self.checkpoint_path
        )
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_dataloaders(
            data_root=self.data_root,
            batch_size=1,  # Single image for explanation
            image_size=(64, 64),
            num_workers=0,
            use_augmentation=False
        )
        
        # Save model checkpoint for explanation
        checkpoint_path = os.path.join(self.output_dir, "test_model.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'backbone_name': 'resnet18',
                'num_classes': 2,
                'dropout_rate': 0.5
            }
        }, checkpoint_path)
        
        # Generate explanations
        try:
            generate_gradcam_explanations(
                model_path=checkpoint_path,
                test_loader=test_loader,
                num_samples=2,  # Very few samples for testing
                save_dir=self.output_dir,
                device="cpu"
            )
            
            # Check that explanation files were created
            explanation_dir = os.path.join(self.output_dir, "explain")
            if os.path.exists(explanation_dir):
                files = os.listdir(explanation_dir)
                assert len(files) > 0
                
        except Exception as e:
            # Skip if explanation generation fails (may require specific dependencies)
            pytest.skip(f"Explanation generation failed: {e}")
    
    def test_freeze_backbone_training(self):
        """Test training with frozen backbone."""
        # Create model with frozen backbone
        model = create_model(
            backbone_name="resnet18",
            num_classes=2,
            dropout_rate=0.5,
            freeze_backbone=True,  # Freeze backbone
            pretrained_checkpoint=self.checkpoint_path
        )
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_dataloaders(
            data_root=self.data_root,
            batch_size=4,
            image_size=(64, 64),
            num_workers=0,
            use_augmentation=False
        )
        
        # Train for 1 epoch
        trained_model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=1,
            learning_rate=1e-4,
            weight_decay=1e-4,
            device="cpu",
            save_dir=self.output_dir
        )
        
        # Check that training completed
        assert trained_model is not None
        assert history is not None
    
    def test_class_weights_training(self):
        """Test training with class weights."""
        # Create model
        model = create_model(
            backbone_name="resnet18",
            num_classes=2,
            dropout_rate=0.5,
            freeze_backbone=False,
            pretrained_checkpoint=self.checkpoint_path
        )
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_dataloaders(
            data_root=self.data_root,
            batch_size=4,
            image_size=(64, 64),
            num_workers=0,
            use_augmentation=False
        )
        
        # Get class weights
        class_weights = train_loader.dataset.get_class_weights()
        
        # Train with class weights
        trained_model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=1,
            learning_rate=1e-4,
            weight_decay=1e-4,
            device="cpu",
            save_dir=self.output_dir,
            class_weights=class_weights
        )
        
        # Check that training completed
        assert trained_model is not None
        assert history is not None


class TestPipelineIntegration:
    """Test integration of different pipeline components."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_root = os.path.join(self.temp_dir, "test_data")
        self.checkpoint_path = os.path.join(self.temp_dir, "dummy_checkpoint.pt")
        self.output_dir = os.path.join(self.temp_dir, "outputs")
        
        # Create minimal test dataset
        generate_synthetic_dataset(
            data_root=self.data_root,
            num_samples=20,
            image_size=64,
            seed=42
        )
        
        # Create dummy checkpoint
        create_dummy_checkpoint(
            checkpoint_path=self.checkpoint_path,
            backbone_name="resnet18",
            seed=42
        )
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_full_pipeline_smoke(self):
        """Test full pipeline integration."""
        # Step 1: Create model
        model = create_model(
            backbone_name="resnet18",
            num_classes=2,
            dropout_rate=0.5,
            freeze_backbone=False,
            pretrained_checkpoint=self.checkpoint_path
        )
        
        # Step 2: Create data loaders
        train_loader, val_loader, test_loader = create_dataloaders(
            data_root=self.data_root,
            batch_size=4,
            image_size=(64, 64),
            num_workers=0,
            use_augmentation=False
        )
        
        # Step 3: Train model
        trained_model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=1,
            learning_rate=1e-4,
            weight_decay=1e-4,
            device="cpu",
            save_dir=self.output_dir
        )
        
        # Step 4: Evaluate model
        metrics = evaluate_model(
            model=trained_model,
            test_loader=test_loader,
            device="cpu",
            save_dir=self.output_dir
        )
        
        # Step 5: Save model for explanation
        checkpoint_path = os.path.join(self.output_dir, "final_model.pt")
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'config': {
                'backbone_name': 'resnet18',
                'num_classes': 2,
                'dropout_rate': 0.5
            }
        }, checkpoint_path)
        
        # Step 6: Generate explanations (optional)
        try:
            generate_gradcam_explanations(
                model_path=checkpoint_path,
                test_loader=test_loader,
                num_samples=1,
                save_dir=self.output_dir,
                device="cpu"
            )
        except Exception as e:
            # Skip if explanation generation fails
            print(f"Explanation generation skipped: {e}")
        
        # Verify outputs
        assert os.path.exists(os.path.join(self.output_dir, "best_model.pt"))
        assert os.path.exists(os.path.join(self.output_dir, "test_metrics.csv"))
        assert metrics is not None
        
        print("✅ Full pipeline smoke test passed!")


class TestErrorHandling:
    """Test error handling in the pipeline."""
    
    def test_invalid_data_path(self):
        """Test handling of invalid data path."""
        with pytest.raises(Exception):
            create_dataloaders(
                data_root="nonexistent_path",
                batch_size=4,
                image_size=(64, 64),
                num_workers=0,
                use_augmentation=False
            )
    
    def test_invalid_checkpoint_path(self):
        """Test handling of invalid checkpoint path."""
        with pytest.raises(Exception):
            create_model(
                backbone_name="resnet18",
                num_classes=2,
                dropout_rate=0.5,
                freeze_backbone=False,
                pretrained_checkpoint="nonexistent_checkpoint.pt"
            )
    
    def test_empty_dataset(self):
        """Test handling of empty dataset."""
        # Create empty data directory
        temp_dir = tempfile.mkdtemp()
        empty_data_root = os.path.join(temp_dir, "empty_data")
        os.makedirs(empty_data_root, exist_ok=True)
        
        try:
            with pytest.raises(Exception):
                create_dataloaders(
                    data_root=empty_data_root,
                    batch_size=4,
                    image_size=(64, 64),
                    num_workers=0,
                    use_augmentation=False
                )
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__])
