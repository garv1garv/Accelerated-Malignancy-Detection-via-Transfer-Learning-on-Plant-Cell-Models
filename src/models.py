"""
Model definitions for the Plant2Skin transfer learning pipeline.

⚠️  RESEARCH PROTOTYPE - NOT FOR CLINICAL USE ⚠️
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import timm
from .config import config
from .utils import count_parameters


def get_backbone(
    name: str = "resnet18",
    pretrained: bool = False,
    num_classes: int = 1000
) -> nn.Module:
    """
    Get a backbone model using timm.
    
    Args:
        name: Model name (e.g., 'resnet18', 'efficientnet_b0')
        pretrained: Whether to use pretrained weights
        num_classes: Number of output classes
    
    Returns:
        Backbone model
    """
    try:
        # Try to load from timm
        model = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=3
        )
        print(f"Loaded {name} backbone from timm")
    except Exception as e:
        print(f"Failed to load {name} from timm: {e}")
        print("Falling back to torchvision ResNet18")
        
        # Fallback to torchvision
        import torchvision.models as models
        if name == "resnet18":
            model = models.resnet18(pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            raise ValueError(f"Unsupported model: {name}")
    
    return model


class TransferModel(nn.Module):
    """
    Transfer learning model that adapts a pretrained backbone for skin lesion classification.
    
    This model implements the transfer learning strategy by:
    1. Loading a backbone pretrained on plant cell imagery (simulated)
    2. Optionally freezing the backbone layers
    3. Replacing the classifier head with a custom one for binary classification
    4. Adding dropout for regularization
    """
    
    def __init__(
        self,
        backbone_name: str = "resnet18",
        num_classes: int = 2,
        dropout_rate: float = 0.5,
        freeze_backbone: bool = False,
        pretrained_checkpoint: Optional[str] = None,
        unfreeze_layers: Optional[int] = None
    ):
        """
        Initialize the transfer learning model.
        
        Args:
            backbone_name: Name of the backbone architecture
            num_classes: Number of output classes (2 for binary classification)
            dropout_rate: Dropout rate for regularization
            freeze_backbone: Whether to freeze backbone parameters
            pretrained_checkpoint: Path to pretrained checkpoint
            unfreeze_layers: Number of layers to unfreeze from the end
        """
        super().__init__()
        
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.freeze_backbone = freeze_backbone
        
        # Load backbone
        self.backbone = get_backbone(
            name=backbone_name,
            pretrained=False,  # We'll load our own checkpoint
            num_classes=num_classes
        )
        
        # Get feature dimension
        if hasattr(self.backbone, 'num_features'):
            feature_dim = self.backbone.num_features
        elif hasattr(self.backbone, 'fc'):
            feature_dim = self.backbone.fc.in_features
        else:
            # Estimate feature dimension based on backbone
            feature_dim = self._estimate_feature_dim()
        
        # Replace classifier head
        self._replace_classifier(feature_dim)
        
        # Load pretrained checkpoint if provided
        if pretrained_checkpoint:
            self._load_pretrained_checkpoint(pretrained_checkpoint)
        
        # Configure freezing strategy
        self._configure_freezing(unfreeze_layers)
        
        print(f"Initialized TransferModel:")
        print(f"  Backbone: {backbone_name}")
        print(f"  Feature dimension: {feature_dim}")
        print(f"  Total parameters: {count_parameters(self):,}")
        print(f"  Trainable parameters: {count_parameters(self):,}")
    
    def _estimate_feature_dim(self) -> int:
        """Estimate feature dimension based on backbone architecture."""
        # Common feature dimensions for different backbones
        feature_dims = {
            'resnet18': 512,
            'resnet34': 512,
            'resnet50': 2048,
            'resnet101': 2048,
            'efficientnet_b0': 1280,
            'efficientnet_b1': 1280,
            'densenet121': 1024,
            'densenet169': 1664,
        }
        
        for name, dim in feature_dims.items():
            if name in self.backbone_name.lower():
                return dim
        
        # Default fallback
        return 512
    
    def _replace_classifier(self, feature_dim: int):
        """Replace the classifier head with a custom one."""
        if hasattr(self.backbone, 'fc'):
            # ResNet-style models
            self.backbone.fc = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(feature_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
                nn.Linear(256, self.num_classes)
            )
        elif hasattr(self.backbone, 'classifier'):
            # EfficientNet-style models
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(feature_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
                nn.Linear(256, self.num_classes)
            )
        elif hasattr(self.backbone, 'head'):
            # Some timm models
            self.backbone.head = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(feature_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
                nn.Linear(256, self.num_classes)
            )
        else:
            raise ValueError(f"Unknown classifier structure for {self.backbone_name}")
    
    def _load_pretrained_checkpoint(self, checkpoint_path: str):
        """Load pretrained weights from checkpoint."""
        if not torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        else:
            checkpoint = torch.load(checkpoint_path)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Load state dict, ignoring mismatched keys
        model_dict = self.backbone.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        
        if len(pretrained_dict) > 0:
            model_dict.update(pretrained_dict)
            self.backbone.load_state_dict(model_dict)
            print(f"Loaded {len(pretrained_dict)} layers from pretrained checkpoint")
        else:
            print("Warning: No matching layers found in pretrained checkpoint")
    
    def _configure_freezing(self, unfreeze_layers: Optional[int]):
        """Configure which layers to freeze/unfreeze."""
        if self.freeze_backbone:
            # Freeze all backbone parameters
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Frozen all backbone parameters")
        
        elif unfreeze_layers is not None:
            # Unfreeze only the last N layers
            total_layers = len(list(self.backbone.parameters()))
            layers_to_unfreeze = min(unfreeze_layers, total_layers)
            
            # Freeze all layers first
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            # Unfreeze the last N layers
            params = list(self.backbone.parameters())
            for param in params[-layers_to_unfreeze:]:
                param.requires_grad = True
            
            print(f"Unfrozen last {layers_to_unfreeze} layers")
        
        # Always unfreeze classifier
        if hasattr(self.backbone, 'fc'):
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
        elif hasattr(self.backbone, 'classifier'):
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
        elif hasattr(self.backbone, 'head'):
            for param in self.backbone.head.parameters():
                param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        
        Returns:
            Tuple of (logits, probabilities)
        """
        # Forward pass through backbone
        logits = self.backbone(x)
        
        # Apply sigmoid for binary classification
        if self.num_classes == 2:
            probs = torch.sigmoid(logits)
            # For binary classification, we only need one output
            if logits.shape[1] == 2:
                probs = probs[:, 1:2]  # Take probability of positive class
        else:
            probs = F.softmax(logits, dim=1)
        
        return logits, probs
    
    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract feature maps from intermediate layers for visualization."""
        feature_maps = {}
        
        # Hook function to capture activations
        activations = {}
        
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output
            return hook
        
        # Register hooks for different layers
        if hasattr(self.backbone, 'layer4'):
            # ResNet-style
            self.backbone.layer4.register_forward_hook(get_activation('layer4'))
        elif hasattr(self.backbone, 'blocks'):
            # EfficientNet-style
            self.backbone.blocks[-1].register_forward_hook(get_activation('last_block'))
        
        # Forward pass
        with torch.no_grad():
            _ = self.forward(x)
        
        # Extract feature maps
        for name, activation in activations.items():
            feature_maps[name] = activation
        
        return feature_maps
    
    def get_trainable_parameters(self) -> int:
        """Get the number of trainable parameters."""
        return count_parameters(self)
    
    def get_total_parameters(self) -> int:
        """Get the total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def create_model(
    backbone_name: str = "resnet18",
    num_classes: int = 2,
    dropout_rate: float = 0.5,
    freeze_backbone: bool = False,
    pretrained_checkpoint: Optional[str] = None,
    unfreeze_layers: Optional[int] = None
) -> TransferModel:
    """
    Create a transfer learning model with the specified configuration.
    
    Args:
        backbone_name: Name of the backbone architecture
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        freeze_backbone: Whether to freeze backbone parameters
        pretrained_checkpoint: Path to pretrained checkpoint
        unfreeze_layers: Number of layers to unfreeze from the end
    
    Returns:
        Configured TransferModel
    """
    model = TransferModel(
        backbone_name=backbone_name,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone,
        pretrained_checkpoint=pretrained_checkpoint,
        unfreeze_layers=unfreeze_layers
    )
    
    return model


def load_model_from_checkpoint(
    checkpoint_path: str,
    model: Optional[TransferModel] = None
) -> TransferModel:
    """
    Load a model from a checkpoint file.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model: Optional model instance to load weights into
    
    Returns:
        Loaded model
    """
    if not torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    else:
        checkpoint = torch.load(checkpoint_path)
    
    # Extract model state dict
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Create model if not provided
    if model is None:
        # Try to infer configuration from checkpoint
        config = checkpoint.get('config', {})
        model = create_model(
            backbone_name=config.get('backbone_name', 'resnet18'),
            num_classes=config.get('num_classes', 2),
            dropout_rate=config.get('dropout_rate', 0.5),
            freeze_backbone=config.get('freeze_backbone', False)
        )
    
    # Load state dict
    model.load_state_dict(state_dict)
    print(f"Model loaded from checkpoint: {checkpoint_path}")
    
    return model
