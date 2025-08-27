"""
Dummy plant cell pretrained checkpoint generator.

⚠️  RESEARCH PROTOTYPE - NOT FOR CLINICAL USE ⚠️

This script creates a dummy checkpoint that simulates a model pretrained on plant cell imagery.
The weights are randomly initialized (with a fixed seed) and do not represent real plant cell features.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import timm
from typing import Optional


def create_dummy_checkpoint(
    checkpoint_path: str = "checkpoints/plant_cell_pretrained.pt",
    backbone_name: str = "resnet18",
    seed: int = 42
):
    """
    Create a dummy plant cell pretrained checkpoint.
    
    Args:
        checkpoint_path: Path to save the checkpoint
        backbone_name: Backbone architecture name
        seed: Random seed for reproducibility
    """
    print(f"⚠️  RESEARCH PROTOTYPE - NOT FOR CLINICAL USE ⚠️")
    print(f"Creating dummy plant cell pretrained checkpoint...")
    print(f"  Backbone: {backbone_name}")
    print(f"  Checkpoint path: {checkpoint_path}")
    print(f"  Seed: {seed}")
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create directory
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    try:
        # Create model using timm
        model = timm.create_model(
            backbone_name,
            pretrained=False,
            num_classes=1000,  # Standard ImageNet classes
            in_chans=3
        )
        print(f"  Created {backbone_name} model")
        
        # Initialize weights with Xavier/Glorot initialization
        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        
        model.apply(init_weights)
        print(f"  Initialized weights with Xavier/Glorot initialization")
        
        # Create dummy training metadata
        dummy_metadata = {
            'epoch': 100,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': None,  # No optimizer state for dummy checkpoint
            'loss': 0.5,  # Dummy loss value
            'metrics': {
                'accuracy': 0.85,
                'precision': 0.83,
                'recall': 0.87,
                'f1_score': 0.85,
                'roc_auc': 0.92
            },
            'config': {
                'backbone_name': backbone_name,
                'num_classes': 1000,
                'dataset': 'plant_cell_imagery',
                'pretrained_source': 'dummy_for_research',
                'training_epochs': 100,
                'learning_rate': 1e-3,
                'batch_size': 32,
                'image_size': 224,
                'note': 'This is a dummy checkpoint for research demonstration only'
            }
        }
        
        # Save checkpoint
        torch.save(dummy_metadata, checkpoint_path)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  Model parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Checkpoint saved to: {checkpoint_path}")
        print(f"  Checkpoint size: {os.path.getsize(checkpoint_path) / 1024 / 1024:.2f} MB")
        
        # Verify checkpoint can be loaded
        loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"  ✅ Checkpoint verification passed")
        
        print(f"\n✅ Dummy checkpoint created successfully!")
        print(f"  ⚠️  REMEMBER: This is a dummy checkpoint for research only!")
        print(f"  ⚠️  It does not contain real plant cell features!")
        
    except Exception as e:
        print(f"❌ Error creating dummy checkpoint: {e}")
        raise


def create_multiple_dummy_checkpoints(
    checkpoint_dir: str = "checkpoints",
    backbones: list = ["resnet18", "resnet34", "efficientnet_b0"],
    seed: int = 42
):
    """
    Create multiple dummy checkpoints for different backbones.
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        backbones: List of backbone architectures
        seed: Random seed for reproducibility
    """
    print(f"Creating multiple dummy checkpoints...")
    
    for backbone in backbones:
        checkpoint_path = os.path.join(checkpoint_dir, f"plant_cell_pretrained_{backbone}.pt")
        print(f"\nCreating checkpoint for {backbone}...")
        
        try:
            create_dummy_checkpoint(
                checkpoint_path=checkpoint_path,
                backbone_name=backbone,
                seed=seed
            )
        except Exception as e:
            print(f"❌ Failed to create checkpoint for {backbone}: {e}")
            continue
    
    print(f"\n✅ Multiple dummy checkpoints creation completed!")


def verify_checkpoint(checkpoint_path: str):
    """
    Verify that a checkpoint can be loaded and contains expected structure.
    
    Args:
        checkpoint_path: Path to the checkpoint file
    """
    print(f"Verifying checkpoint: {checkpoint_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Check structure
        required_keys = ['model_state_dict', 'config']
        for key in required_keys:
            if key not in checkpoint:
                print(f"❌ Missing key: {key}")
                return False
        
        # Check model state dict
        state_dict = checkpoint['model_state_dict']
        if not isinstance(state_dict, dict):
            print(f"❌ model_state_dict is not a dictionary")
            return False
        
        # Check config
        config = checkpoint['config']
        if not isinstance(config, dict):
            print(f"❌ config is not a dictionary")
            return False
        
        # Print info
        print(f"✅ Checkpoint verification passed")
        print(f"  Model state dict keys: {len(state_dict)}")
        print(f"  Config keys: {list(config.keys())}")
        
        if 'metrics' in checkpoint:
            print(f"  Metrics: {checkpoint['metrics']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Checkpoint verification failed: {e}")
        return False


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create dummy plant cell pretrained checkpoint")
    parser.add_argument("--checkpoint-path", default="checkpoints/plant_cell_pretrained.pt", 
                       help="Path to save checkpoint")
    parser.add_argument("--backbone", default="resnet18", help="Backbone architecture")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verify", action="store_true", help="Verify checkpoint after creation")
    parser.add_argument("--multiple", action="store_true", help="Create multiple checkpoints for different backbones")
    
    args = parser.parse_args()
    
    if args.multiple:
        create_multiple_dummy_checkpoints(
            checkpoint_dir="checkpoints",
            backbones=["resnet18", "resnet34", "efficientnet_b0"],
            seed=args.seed
        )
    else:
        create_dummy_checkpoint(
            checkpoint_path=args.checkpoint_path,
            backbone_name=args.backbone,
            seed=args.seed
        )
        
        if args.verify:
            verify_checkpoint(args.checkpoint_path)


if __name__ == "__main__":
    main()
