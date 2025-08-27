"""
Plant2Skin Transfer Learning Pipeline

A research prototype demonstrating transfer learning from plant cell imagery
to skin lesion classification using synthetic data.

⚠️  RESEARCH PROTOTYPE - NOT FOR CLINICAL USE ⚠️
"""

__version__ = "1.0.0"
__author__ = "Research Team"
__description__ = "Transfer learning pipeline from plant cell to skin lesion classification"

# Import main components
from .config import Config
from .models import TransferModel, get_backbone
from .data_utils import SkinLesionDataset, create_dataloaders
from .train import train_model
from .eval import evaluate_model
from .explain import generate_gradcam_explanations

__all__ = [
    "Config",
    "TransferModel", 
    "get_backbone",
    "SkinLesionDataset",
    "create_dataloaders",
    "train_model",
    "evaluate_model",
    "generate_gradcam_explanations"
]
