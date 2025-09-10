"""
SageMaker training script for Draw-and-Learn Custom Model.
This script is designed to work with SageMaker training jobs.
"""

import argparse
import json
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def model_fn(model_dir):
    """Load model for SageMaker inference."""
    model_path = os.path.join(model_dir, 'model.pth')
    
    # Load your custom model here
    # This is a placeholder - replace with actual model architecture
    model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 10)
    )
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    
    model.eval()
    return model


def input_fn(request_body, request_content_type):
    """Parse input data for inference."""
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return torch.tensor(input_data['instances'])
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """Make prediction with the model."""
    with torch.no_grad():
        outputs = model(input_data)
        predictions = torch.argmax(outputs, dim=1)
    return predictions.numpy().tolist()


def output_fn(predictions, content_type):
    """Format output for response."""
    if content_type == 'application/json':
        return json.dumps({'predictions': predictions})
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def train_model(args):
    """Main training function."""
    logger.info("Starting model training...")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model
    model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, args.num_classes)
    ).to(device)
    
    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Placeholder for data loading - replace with actual data loading logic
    logger.info(f"Training data path: {args.train}")
    logger.info(f"Validation data path: {args.validation}")
    
    # Training loop placeholder
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        
        # Training logic would go here
        # This is a placeholder for the actual training loop
        
        # Log metrics (placeholder)
        train_loss = 0.5 - (epoch * 0.05)  # Simulated decreasing loss
        logger.info(f"Training loss: {train_loss:.4f}")
    
    # Save model
    model_path = os.path.join(args.model_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION', '/opt/ml/input/data/validation'))
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--num-classes', type=int, default=10)
    
    args = parser.parse_args()
    
    logger.info("Training arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    train_model(args)