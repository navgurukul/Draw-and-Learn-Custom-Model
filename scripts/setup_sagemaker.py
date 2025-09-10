#!/usr/bin/env python3
"""
Setup script for AWS SageMaker Unified Studio integration.
This script helps configure the project for SageMaker usage.
"""

import os
import json
import boto3
import argparse
from pathlib import Path


def get_aws_info():
    """Get AWS account and region information."""
    try:
        session = boto3.Session()
        sts_client = session.client('sts')
        identity = sts_client.get_caller_identity()
        
        account_id = identity['Account']
        region = session.region_name or 'us-east-1'
        
        return account_id, region
    except Exception as e:
        print(f"Error getting AWS info: {e}")
        return None, None


def create_sagemaker_role(account_id, region):
    """Create SageMaker execution role ARN."""
    role_arn = f"arn:aws:iam::{account_id}:role/SageMakerExecutionRole"
    return role_arn


def update_sagemaker_config(account_id, region, bucket_name=None):
    """Update SageMaker configuration file."""
    config_path = Path("config/sagemaker_config.json")
    
    if not bucket_name:
        bucket_name = f"sagemaker-draw-learn-{account_id}-{region}"
    
    role_arn = create_sagemaker_role(account_id, region)
    
    config = {
        "sagemaker_studio": {
            "domain_name": "draw-learn-domain",
            "execution_role": role_arn,
            "instance_types": {
                "notebook": "ml.t3.medium",
                "processing": "ml.m5.large",
                "training": "ml.m5.xlarge",
                "inference": "ml.m5.large"
            }
        },
        "project_settings": {
            "project_name": "draw-and-learn-custom-model",
            "bucket_name": bucket_name,
            "prefix": "draw-learn-model",
            "region": region
        },
        "model_settings": {
            "framework": "pytorch",
            "framework_version": "1.12",
            "python_version": "py38",
            "model_artifacts_path": "models/",
            "training_image": f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:1.12.0-gpu-py38-cu113-ubuntu20.04-sagemaker"
        },
        "data_settings": {
            "input_data_path": "data/",
            "output_data_path": "output/",
            "train_channel": "training",
            "validation_channel": "validation", 
            "test_channel": "test"
        }
    }
    
    # Create config directory if it doesn't exist
    config_path.parent.mkdir(exist_ok=True)
    
    # Write configuration
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Updated SageMaker configuration: {config_path}")
    return config


def create_s3_bucket(bucket_name, region):
    """Create S3 bucket if it doesn't exist."""
    try:
        s3_client = boto3.client('s3', region_name=region)
        
        # Check if bucket exists
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            print(f"✅ S3 bucket '{bucket_name}' already exists")
            return True
        except:
            pass
        
        # Create bucket
        if region == 'us-east-1':
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': region}
            )
        
        # Set bucket policy for SageMaker access
        bucket_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "SageMakerAccess",
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "sagemaker.amazonaws.com"
                    },
                    "Action": [
                        "s3:GetObject",
                        "s3:PutObject",
                        "s3:DeleteObject",
                        "s3:ListBucket"
                    ],
                    "Resource": [
                        f"arn:aws:s3:::{bucket_name}/*",
                        f"arn:aws:s3:::{bucket_name}"
                    ]
                }
            ]
        }
        
        s3_client.put_bucket_policy(
            Bucket=bucket_name,
            Policy=json.dumps(bucket_policy)
        )
        
        print(f"✅ Created S3 bucket: {bucket_name}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating S3 bucket: {e}")
        return False


def setup_directory_structure():
    """Ensure all required directories exist."""
    directories = [
        "data/raw",
        "data/processed", 
        "data/external",
        "models",
        "notebooks",
        "scripts/training",
        "scripts/processing",
        "scripts/inference",
        "src/data",
        "src/models",
        "src/visualization",
        "tests",
        "config",
        "docker"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Created directory structure")


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import sagemaker
        import boto3
        import torch
        print("✅ Core dependencies available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False


def create_iam_role_instructions(account_id):
    """Provide instructions for creating IAM role."""
    role_name = "SageMakerExecutionRole"
    
    instructions = f"""
📋 IAM Role Setup Instructions:

1. Go to AWS IAM Console: https://console.aws.amazon.com/iam/
2. Click "Roles" → "Create role"
3. Select "AWS Service" → "SageMaker"
4. Attach the following policies:
   - AmazonSageMakerFullAccess
   - AmazonS3FullAccess
   - AmazonEC2ContainerRegistryFullAccess
   - CloudWatchLogsFullAccess

5. Name the role: {role_name}
6. Create the role

The role ARN will be: arn:aws:iam::{account_id}:role/{role_name}

Alternatively, you can create the role using AWS CLI:

aws iam create-role --role-name {role_name} --assume-role-policy-document file://trust-policy.json
aws iam attach-role-policy --role-name {role_name} --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
aws iam attach-role-policy --role-name {role_name} --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
"""
    
    print(instructions)


def main():
    parser = argparse.ArgumentParser(description="Setup AWS SageMaker integration")
    parser.add_argument("--bucket-name", help="S3 bucket name (optional)")
    parser.add_argument("--create-bucket", action="store_true", help="Create S3 bucket")
    parser.add_argument("--skip-aws-check", action="store_true", help="Skip AWS configuration check")
    
    args = parser.parse_args()
    
    print("🚀 Setting up AWS SageMaker integration...")
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Setup directory structure
    setup_directory_structure()
    
    # Get AWS information
    if not args.skip_aws_check:
        account_id, region = get_aws_info()
        
        if not account_id:
            print("❌ Could not get AWS account information")
            print("Please configure AWS credentials: aws configure")
            return 1
        
        print(f"📍 AWS Account: {account_id}")
        print(f"📍 AWS Region: {region}")
        
        # Update configuration
        config = update_sagemaker_config(account_id, region, args.bucket_name)
        
        # Create S3 bucket if requested
        if args.create_bucket:
            bucket_name = args.bucket_name or config["project_settings"]["bucket_name"]
            create_s3_bucket(bucket_name, region)
        
        # Show IAM role instructions
        create_iam_role_instructions(account_id)
    else:
        print("⏭️  Skipped AWS configuration check")
    
    print("""
🎉 Setup completed! Next steps:

1. 📝 Upload your training data to data/raw/
2. 🔧 Review and customize config/sagemaker_config.json
3. 📓 Open notebooks/sagemaker_training_example.ipynb in SageMaker Studio
4. 🚀 Start training your custom model!

For more information, see the README.md file.
""")
    
    return 0


if __name__ == "__main__":
    exit(main())