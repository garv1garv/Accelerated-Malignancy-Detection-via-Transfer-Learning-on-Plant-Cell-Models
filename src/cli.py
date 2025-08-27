"""
Command-line interface for the Plant2Skin transfer learning pipeline.

⚠️  RESEARCH PROTOTYPE - NOT FOR CLINICAL USE ⚠️
"""

import typer
from typing import Optional
import os
import sys

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import config, update_config
from src.scripts.generate_synthetic_data import generate_synthetic_dataset
from src.scripts.make_dummy_plant_checkpoint import create_dummy_checkpoint
from src.train import train_model
from src.eval import evaluate_from_checkpoint
from src.explain import generate_gradcam_explanations, generate_class_comparison_explanations
from src.data_utils import create_dataloaders, print_dataset_info, validate_dataset
from src.models import create_model
from src.utils import print_device_info

app = typer.Typer(
    name="plant2skin-transfer",
    help="Plant2Skin Transfer Learning Pipeline - Research Prototype",
    add_completion=False
)


@app.command()
def prepare(
    data_root: str = typer.Option("data/synthetic", "--data-root", "-d", help="Data root directory"),
    num_samples: int = typer.Option(120, "--num-samples", "-n", help="Number of synthetic samples to generate"),
    image_size: int = typer.Option(128, "--image-size", "-s", help="Image size (square)"),
    seed: int = typer.Option(42, "--seed", help="Random seed")
):
    """Generate synthetic skin lesion dataset."""
    print("⚠️  RESEARCH PROTOTYPE - NOT FOR CLINICAL USE ⚠️")
    print("Generating synthetic skin lesion dataset...")
    
    try:
        generate_synthetic_dataset(
            data_root=data_root,
            num_samples=num_samples,
            image_size=image_size,
            seed=seed
        )
        print("✅ Synthetic dataset generated successfully!")
        
        # Print dataset info
        print_dataset_info(data_root)
        
        # Validate dataset
        if validate_dataset(data_root):
            print("✅ Dataset validation passed!")
        else:
            print("❌ Dataset validation failed!")
            
    except Exception as e:
        print(f"❌ Error generating dataset: {e}")
        raise typer.Exit(1)


@app.command()
def make_checkpoint(
    checkpoint_path: str = typer.Option("checkpoints/plant_cell_pretrained.pt", "--checkpoint-path", "-c", help="Path to save checkpoint"),
    backbone_name: str = typer.Option("resnet18", "--backbone", "-b", help="Backbone architecture"),
    seed: int = typer.Option(42, "--seed", help="Random seed")
):
    """Create dummy plant cell pretrained checkpoint."""
    print("⚠️  RESEARCH PROTOTYPE - NOT FOR CLINICAL USE ⚠️")
    print("Creating dummy plant cell pretrained checkpoint...")
    
    try:
        create_dummy_checkpoint(
            checkpoint_path=checkpoint_path,
            backbone_name=backbone_name,
            seed=seed
        )
        print("✅ Dummy checkpoint created successfully!")
        
    except Exception as e:
        print(f"❌ Error creating checkpoint: {e}")
        raise typer.Exit(1)


@app.command()
def train(
    epochs: int = typer.Option(10, "--epochs", "-e", help="Number of training epochs"),
    batch_size: int = typer.Option(16, "--batch-size", "-b", help="Batch size"),
    learning_rate: float = typer.Option(1e-4, "--lr", "-l", help="Learning rate"),
    freeze_backbone: bool = typer.Option(False, "--freeze-backbone", "-f", help="Freeze backbone parameters"),
    data_root: str = typer.Option("data/synthetic", "--data-root", "-d", help="Data root directory"),
    checkpoint_path: str = typer.Option("checkpoints/plant_cell_pretrained.pt", "--checkpoint", "-c", help="Pretrained checkpoint path"),
    output_dir: str = typer.Option("outputs", "--output-dir", "-o", help="Output directory"),
    device: str = typer.Option("auto", "--device", help="Device to use (auto/cuda/cpu)")
):
    """Train the transfer learning model."""
    print("⚠️  RESEARCH PROTOTYPE - NOT FOR CLINICAL USE ⚠️")
    print("Starting transfer learning training...")
    
    try:
        # Update config
        update_config(
            num_epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            freeze_backbone=freeze_backbone,
            data_root=data_root,
            pretrained_checkpoint=checkpoint_path,
            output_dir=output_dir,
            device=device
        )
        
        # Print device info
        print_device_info()
        
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
        
        print("✅ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        raise typer.Exit(1)


@app.command()
def eval(
    checkpoint_path: str = typer.Option("outputs/best_model.pt", "--checkpoint", "-c", help="Model checkpoint path"),
    data_root: str = typer.Option("data/synthetic", "--data-root", "-d", help="Data root directory"),
    output_dir: str = typer.Option("outputs", "--output-dir", "-o", help="Output directory"),
    device: str = typer.Option("auto", "--device", help="Device to use (auto/cuda/cpu)")
):
    """Evaluate the trained model."""
    print("⚠️  RESEARCH PROTOTYPE - NOT FOR CLINICAL USE ⚠️")
    print("Evaluating trained model...")
    
    try:
        # Update config
        update_config(
            data_root=data_root,
            output_dir=output_dir,
            device=device
        )
        
        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            print("Please run training first or specify a valid checkpoint path.")
            raise typer.Exit(1)
        
        # Evaluate model
        metrics = evaluate_from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=config.device,
            save_dir=config.output_dir
        )
        
        print("✅ Evaluation completed successfully!")
        print(f"Results saved to: {config.output_dir}")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        raise typer.Exit(1)


@app.command()
def explain(
    checkpoint_path: str = typer.Option("outputs/best_model.pt", "--checkpoint", "-c", help="Model checkpoint path"),
    num_samples: int = typer.Option(10, "--num-samples", "-n", help="Number of samples to explain"),
    data_root: str = typer.Option("data/synthetic", "--data-root", "-d", help="Data root directory"),
    output_dir: str = typer.Option("outputs/explain", "--output-dir", "-o", help="Output directory"),
    device: str = typer.Option("auto", "--device", help="Device to use (auto/cuda/cpu)")
):
    """Generate Grad-CAM explanations."""
    print("⚠️  RESEARCH PROTOTYPE - NOT FOR CLINICAL USE ⚠️")
    print("Generating Grad-CAM explanations...")
    
    try:
        # Update config
        update_config(
            data_root=data_root,
            output_dir=output_dir,
            device=device
        )
        
        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            print("Please run training first or specify a valid checkpoint path.")
            raise typer.Exit(1)
        
        # Generate explanations
        generate_gradcam_explanations(
            model_path=checkpoint_path,
            num_samples=num_samples,
            save_dir=output_dir,
            device=config.device
        )
        
        # Generate class comparison explanations
        generate_class_comparison_explanations(
            model_path=checkpoint_path,
            num_samples=3,
            save_dir=output_dir,
            device=config.device
        )
        
        print("✅ Explanation generation completed!")
        print(f"Explanations saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error during explanation generation: {e}")
        raise typer.Exit(1)


@app.command()
def all(
    epochs: int = typer.Option(10, "--epochs", "-e", help="Number of training epochs"),
    batch_size: int = typer.Option(16, "--batch-size", "-b", help="Batch size"),
    learning_rate: float = typer.Option(1e-4, "--lr", "-l", help="Learning rate"),
    freeze_backbone: bool = typer.Option(False, "--freeze-backbone", "-f", help="Freeze backbone parameters"),
    num_samples: int = typer.Option(120, "--num-samples", "-n", help="Number of synthetic samples"),
    device: str = typer.Option("auto", "--device", help="Device to use (auto/cuda/cpu)")
):
    """Run the complete pipeline: prepare -> make_checkpoint -> train -> eval -> explain."""
    print("⚠️  RESEARCH PROTOTYPE - NOT FOR CLINICAL USE ⚠️")
    print("Running complete Plant2Skin transfer learning pipeline...")
    
    try:
        # Step 1: Prepare data
        print("\n" + "="*50)
        print("STEP 1: Generating synthetic dataset")
        print("="*50)
        prepare(
            data_root="data/synthetic",
            num_samples=num_samples,
            image_size=128,
            seed=42
        )
        
        # Step 2: Create dummy checkpoint
        print("\n" + "="*50)
        print("STEP 2: Creating dummy plant checkpoint")
        print("="*50)
        make_checkpoint(
            checkpoint_path="checkpoints/plant_cell_pretrained.pt",
            backbone_name="resnet18",
            seed=42
        )
        
        # Step 3: Train model
        print("\n" + "="*50)
        print("STEP 3: Training transfer learning model")
        print("="*50)
        train(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            freeze_backbone=freeze_backbone,
            data_root="data/synthetic",
            checkpoint_path="checkpoints/plant_cell_pretrained.pt",
            output_dir="outputs",
            device=device
        )
        
        # Step 4: Evaluate model
        print("\n" + "="*50)
        print("STEP 4: Evaluating model")
        print("="*50)
        eval(
            checkpoint_path="outputs/best_model.pt",
            data_root="data/synthetic",
            output_dir="outputs",
            device=device
        )
        
        # Step 5: Generate explanations
        print("\n" + "="*50)
        print("STEP 5: Generating explanations")
        print("="*50)
        explain(
            checkpoint_path="outputs/best_model.pt",
            num_samples=10,
            data_root="data/synthetic",
            output_dir="outputs/explain",
            device=device
        )
        
        print("\n" + "="*50)
        print("✅ COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
        print("="*50)
        print("Results saved to:")
        print("  - Model checkpoints: outputs/")
        print("  - Evaluation results: outputs/")
        print("  - Grad-CAM explanations: outputs/explain/")
        print("\n⚠️  REMEMBER: This is a research prototype, not for clinical use!")
        
    except Exception as e:
        print(f"❌ Error in pipeline: {e}")
        raise typer.Exit(1)


@app.command()
def info():
    """Display system and configuration information."""
    print("⚠️  RESEARCH PROTOTYPE - NOT FOR CLINICAL USE ⚠️")
    print("\nPlant2Skin Transfer Learning Pipeline Information")
    print("="*50)
    
    # System info
    print_device_info()
    
    # Config info
    print(f"\nConfiguration:")
    print(f"  Data root: {config.data_root}")
    print(f"  Image size: {config.image_size}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Backbone: {config.backbone_name}")
    print(f"  Freeze backbone: {config.freeze_backbone}")
    print(f"  Output directory: {config.output_dir}")
    
    # Check if data exists
    if os.path.exists(config.data_root):
        print_dataset_info(config.data_root)
    else:
        print(f"\nData directory not found: {config.data_root}")
    
    # Check if checkpoint exists
    if os.path.exists(config.pretrained_checkpoint):
        print(f"✅ Pretrained checkpoint found: {config.pretrained_checkpoint}")
    else:
        print(f"❌ Pretrained checkpoint not found: {config.pretrained_checkpoint}")


if __name__ == "__main__":
    app()
