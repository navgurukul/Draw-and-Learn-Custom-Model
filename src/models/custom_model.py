"""
Custom neural network models for Draw-and-Learn project.
Defines various model architectures for drawing/image classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50, efficientnet_b0
import logging

logger = logging.getLogger(__name__)


class SimpleCNN(nn.Module):
    """Simple CNN architecture for drawing classification."""
    
    def __init__(self, num_classes=10, input_channels=3, dropout_rate=0.5):
        """
        Initialize Simple CNN model.
        
        Args:
            num_classes (int): Number of output classes
            input_channels (int): Number of input channels (3 for RGB, 1 for grayscale)
            dropout_rate (float): Dropout rate for regularization
        """
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """Forward pass."""
        x = self.features(x)
        x = self.classifier(x)
        return x


class DrawingCNN(nn.Module):
    """More sophisticated CNN architecture optimized for drawing recognition."""
    
    def __init__(self, num_classes=10, input_channels=3, dropout_rate=0.5):
        """
        Initialize Drawing CNN model.
        
        Args:
            num_classes (int): Number of output classes
            input_channels (int): Number of input channels
            dropout_rate (float): Dropout rate for regularization
        """
        super(DrawingCNN, self).__init__()
        
        self.conv1 = self._make_conv_block(input_channels, 64)
        self.conv2 = self._make_conv_block(64, 128)
        self.conv3 = self._make_conv_block(128, 256)
        self.conv4 = self._make_conv_block(256, 512)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def _make_conv_block(self, in_channels, out_channels):
        """Create a convolutional block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
    
    def forward(self, x):
        """Forward pass."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x


class PretrainedModel(nn.Module):
    """Pre-trained model with custom classifier head."""
    
    def __init__(self, model_name='resnet18', num_classes=10, pretrained=True, freeze_features=False):
        """
        Initialize pre-trained model.
        
        Args:
            model_name (str): Name of the pre-trained model
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pre-trained weights
            freeze_features (bool): Whether to freeze feature extraction layers
        """
        super(PretrainedModel, self).__init__()
        
        self.model_name = model_name
        
        # Load pre-trained model
        if model_name == 'resnet18':
            self.backbone = resnet18(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif model_name == 'resnet50':
            self.backbone = resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif model_name == 'efficientnet_b0':
            self.backbone = efficientnet_b0(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Freeze feature extraction layers if requested
        if freeze_features:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """Forward pass."""
        features = self.backbone(x)
        output = self.classifier(features)
        return output


class MultiScaleCNN(nn.Module):
    """Multi-scale CNN for capturing features at different scales."""
    
    def __init__(self, num_classes=10, input_channels=3):
        """
        Initialize Multi-scale CNN.
        
        Args:
            num_classes (int): Number of output classes
            input_channels (int): Number of input channels
        """
        super(MultiScaleCNN, self).__init__()
        
        # Different scale branches
        self.scale1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.scale2 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.scale3 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """Forward pass."""
        s1 = self.scale1(x)
        s2 = self.scale2(x)
        s3 = self.scale3(x)
        
        # Concatenate multi-scale features
        fused = torch.cat([s1, s2, s3], dim=1)
        
        fused = self.fusion(fused)
        output = self.classifier(fused)
        
        return output


def get_model(model_type='simple_cnn', num_classes=10, **kwargs):
    """
    Factory function to get model by type.
    
    Args:
        model_type (str): Type of model to create
        num_classes (int): Number of output classes
        **kwargs: Additional model parameters
    
    Returns:
        torch.nn.Module: The requested model
    """
    models = {
        'simple_cnn': SimpleCNN,
        'drawing_cnn': DrawingCNN,
        'multiscale_cnn': MultiScaleCNN,
        'resnet18': lambda **kwargs: PretrainedModel('resnet18', **kwargs),
        'resnet50': lambda **kwargs: PretrainedModel('resnet50', **kwargs),
        'efficientnet_b0': lambda **kwargs: PretrainedModel('efficientnet_b0', **kwargs),
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")
    
    model = models[model_type](num_classes=num_classes, **kwargs)
    
    logger.info(f"Created {model_type} model with {num_classes} classes")
    return model


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Example usage
    model = get_model('drawing_cnn', num_classes=10)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"Output shape: {output.shape}")