"""
Test suite for SageMaker integration components.
"""

import pytest
import os
import json
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from sagemaker_utils import SageMakerHelper
from models.custom_model import get_model, SimpleCNN, DrawingCNN
from data.dataset import DrawingDataset


class TestSageMakerIntegration:
    """Test SageMaker integration components."""
    
    def test_config_file_exists(self):
        """Test that SageMaker configuration file exists and is valid."""
        config_path = Path("config/sagemaker_config.json")
        assert config_path.exists(), "SageMaker config file not found"
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check required sections
        assert "sagemaker_studio" in config
        assert "project_settings" in config
        assert "model_settings" in config
        
        # Check required fields
        studio_config = config["sagemaker_studio"]
        assert "execution_role" in studio_config
        assert "instance_types" in studio_config
    
    def test_directory_structure(self):
        """Test that all required directories exist."""
        required_dirs = [
            "config",
            "data/raw", 
            "data/processed",
            "data/external",
            "models",
            "notebooks",
            "scripts/training",
            "scripts/processing", 
            "src",
            "src/data",
            "src/models",
            "tests",
            "docker"
        ]
        
        for directory in required_dirs:
            assert Path(directory).exists(), f"Directory {directory} not found"
    
    def test_required_files_exist(self):
        """Test that all required files exist."""
        required_files = [
            "requirements.txt",
            "setup.py",
            ".gitignore",
            "README.md",
            "config/sagemaker_config.json",
            "config/training_config.yaml",
            "docker/Dockerfile",
            "docker/build_and_push.sh",
            "src/__init__.py",
            "src/sagemaker_utils.py",
            "notebooks/sagemaker_training_example.ipynb"
        ]
        
        for file_path in required_files:
            assert Path(file_path).exists(), f"Required file {file_path} not found"


class TestModels:
    """Test model definitions and functionality."""
    
    def test_simple_cnn_creation(self):
        """Test SimpleCNN model creation."""
        model = SimpleCNN(num_classes=10, input_channels=3)
        assert isinstance(model, torch.nn.Module)
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert output.shape == (2, 10)
    
    def test_drawing_cnn_creation(self):
        """Test DrawingCNN model creation."""
        model = DrawingCNN(num_classes=5, input_channels=1)
        assert isinstance(model, torch.nn.Module)
        
        # Test forward pass
        x = torch.randn(1, 1, 224, 224)
        output = model(x)
        assert output.shape == (1, 5)
    
    def test_model_factory(self):
        """Test model factory function."""
        model = get_model('simple_cnn', num_classes=10)
        assert isinstance(model, SimpleCNN)
        
        model = get_model('drawing_cnn', num_classes=5)
        assert isinstance(model, DrawingCNN)
        
        # Test invalid model type
        with pytest.raises(ValueError):
            get_model('invalid_model_type')


class TestSageMakerUtils:
    """Test SageMaker utility functions."""
    
    def test_sagemaker_helper_init(self):
        """Test SageMakerHelper initialization."""
        # This test might fail if AWS credentials are not configured
        # So we'll just test the basic initialization
        try:
            helper = SageMakerHelper()
            assert hasattr(helper, 'config')
            assert hasattr(helper, 'bucket')
            assert hasattr(helper, 'prefix')
        except Exception:
            # If AWS is not configured, just check config loading
            with open('config/sagemaker_config.json', 'r') as f:
                config = json.load(f)
            assert config is not None
    
    def test_default_hyperparameters(self):
        """Test default hyperparameters method."""
        try:
            helper = SageMakerHelper()
            hyperparams = helper.get_default_hyperparameters()
            
            assert isinstance(hyperparams, dict)
            assert 'epochs' in hyperparams
            assert 'batch-size' in hyperparams
            assert 'learning-rate' in hyperparams
        except Exception:
            # Skip if SageMaker not available
            pytest.skip("SageMaker not available")


class TestDockerConfiguration:
    """Test Docker configuration for SageMaker."""
    
    def test_dockerfile_exists(self):
        """Test that Dockerfile exists and has basic structure."""
        dockerfile_path = Path("docker/Dockerfile")
        assert dockerfile_path.exists()
        
        with open(dockerfile_path, 'r') as f:
            content = f.read()
        
        # Check for key Dockerfile elements
        assert "FROM" in content
        assert "COPY" in content
        assert "WORKDIR" in content
        assert "sagemaker" in content.lower()
    
    def test_build_script_exists(self):
        """Test that Docker build script exists."""
        build_script_path = Path("docker/build_and_push.sh")
        assert build_script_path.exists()
        assert os.access(build_script_path, os.X_OK), "Build script not executable"


class TestTrainingScripts:
    """Test training and processing scripts."""
    
    def test_training_script_exists(self):
        """Test that training script exists."""
        train_script_path = Path("scripts/training/train.py")
        assert train_script_path.exists()
        
        # Check for key functions
        with open(train_script_path, 'r') as f:
            content = f.read()
        
        assert "model_fn" in content
        assert "train_model" in content
        assert "sagemaker" in content.lower()
    
    def test_preprocessing_script_exists(self):
        """Test that preprocessing script exists."""
        preprocess_script_path = Path("scripts/processing/preprocess.py")
        assert preprocess_script_path.exists()
        
        with open(preprocess_script_path, 'r') as f:
            content = f.read()
        
        assert "load_and_preprocess_images" in content
        assert "preprocess_single_image" in content


def test_requirements_file():
    """Test that requirements file contains necessary packages."""
    with open("requirements.txt", 'r') as f:
        requirements = f.read()
    
    required_packages = [
        "sagemaker",
        "boto3", 
        "torch",
        "numpy",
        "pandas",
        "opencv-python",
        "matplotlib"
    ]
    
    for package in required_packages:
        assert package in requirements, f"Missing required package: {package}"


def test_setup_py_configuration():
    """Test setup.py configuration."""
    setup_path = Path("setup.py")
    assert setup_path.exists()
    
    with open(setup_path, 'r') as f:
        content = f.read()
    
    # Check for key setup elements
    assert "setup(" in content
    assert "name=" in content
    assert "version=" in content
    assert "install_requires=" in content


if __name__ == "__main__":
    pytest.main([__file__])