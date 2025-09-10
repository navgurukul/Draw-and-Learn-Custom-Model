"""
SageMaker utilities for Draw-and-Learn Custom Model project.
This module provides helper functions for working with AWS SageMaker.
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
import json
import os
from typing import Dict, Any, Optional


class SageMakerHelper:
    """Helper class for SageMaker operations."""
    
    def __init__(self, config_path: str = "config/sagemaker_config.json"):
        """Initialize SageMaker helper with configuration."""
        self.session = sagemaker.Session()
        self.role = sagemaker.get_execution_role()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.bucket = self.config["project_settings"]["bucket_name"]
        self.prefix = self.config["project_settings"]["prefix"]
        
    def upload_data_to_s3(self, local_path: str, s3_key: str = None) -> str:
        """Upload data to S3 bucket."""
        if s3_key is None:
            s3_key = f"{self.prefix}/data/{os.path.basename(local_path)}"
        
        s3_uri = f"s3://{self.bucket}/{s3_key}"
        self.session.upload_data(local_path, self.bucket, s3_key)
        print(f"Data uploaded to: {s3_uri}")
        return s3_uri
    
    def create_training_job(self, 
                          entry_point: str,
                          source_dir: str,
                          train_data_uri: str,
                          val_data_uri: str = None,
                          hyperparameters: Dict[str, Any] = None) -> PyTorch:
        """Create a SageMaker training job."""
        
        estimator = PyTorch(
            entry_point=entry_point,
            source_dir=source_dir,
            role=self.role,
            instance_type=self.config["sagemaker_studio"]["instance_types"]["training"],
            instance_count=1,
            framework_version=self.config["model_settings"]["framework_version"],
            py_version=self.config["model_settings"]["python_version"],
            hyperparameters=hyperparameters or {},
            output_path=f"s3://{self.bucket}/{self.prefix}/output/"
        )
        
        # Prepare training channels
        training_channels = {
            "training": train_data_uri
        }
        if val_data_uri:
            training_channels["validation"] = val_data_uri
        
        return estimator
    
    def create_processing_job(self, 
                            script_path: str,
                            input_data_uri: str,
                            output_data_uri: str = None) -> SKLearnProcessor:
        """Create a SageMaker processing job for data preprocessing."""
        
        if output_data_uri is None:
            output_data_uri = f"s3://{self.bucket}/{self.prefix}/processed/"
        
        processor = SKLearnProcessor(
            framework_version="0.23-1",
            instance_type=self.config["sagemaker_studio"]["instance_types"]["processing"],
            instance_count=1,
            role=self.role
        )
        
        return processor
    
    def deploy_model(self, model_data: str, 
                    initial_instance_count: int = 1,
                    instance_type: str = None) -> str:
        """Deploy model to SageMaker endpoint."""
        
        if instance_type is None:
            instance_type = self.config["sagemaker_studio"]["instance_types"]["inference"]
        
        # This is a placeholder - actual deployment depends on model specifics
        print(f"Deploying model from: {model_data}")
        print(f"Instance type: {instance_type}")
        print(f"Instance count: {initial_instance_count}")
        
        return "endpoint-name-placeholder"
    
    def get_default_hyperparameters(self) -> Dict[str, Any]:
        """Get default hyperparameters from config."""
        return {
            "epochs": 10,
            "batch-size": 32,
            "learning-rate": 0.001,
            "model-type": "cnn"
        }


def setup_sagemaker_studio_project():
    """Setup initial SageMaker Studio project structure."""
    
    # Create necessary S3 bucket structure
    session = sagemaker.Session()
    bucket = session.default_bucket()
    
    # Create folder structure in S3
    folders = [
        "data/raw",
        "data/processed", 
        "models",
        "output",
        "scripts"
    ]
    
    for folder in folders:
        # Create empty object to establish folder structure
        key = f"draw-learn-model/{folder}/.keep"
        session.boto_session.client('s3').put_object(
            Bucket=bucket, 
            Key=key, 
            Body=b''
        )
    
    print(f"S3 project structure created in bucket: {bucket}")
    return bucket


if __name__ == "__main__":
    # Example usage
    helper = SageMakerHelper()
    bucket = setup_sagemaker_studio_project()
    print(f"SageMaker project setup completed. Default bucket: {bucket}")