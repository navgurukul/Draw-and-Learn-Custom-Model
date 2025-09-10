#!/bin/bash

# Build and push Docker image to Amazon ECR for SageMaker

# Configuration
ACCOUNT_ID=${1:-$(aws sts get-caller-identity --query Account --output text)}
REGION=${2:-us-east-1}
IMAGE_NAME="draw-learn-custom-model"
TAG=${3:-latest}

# Full image URI
FULL_NAME="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${IMAGE_NAME}:${TAG}"

# Check if repository exists, create if not
aws ecr describe-repositories --repository-names ${IMAGE_NAME} --region ${REGION} || \
    aws ecr create-repository --repository-name ${IMAGE_NAME} --region ${REGION}

# Get login token
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

# Build image
echo "Building Docker image: ${FULL_NAME}"
docker build -t ${IMAGE_NAME} -f docker/Dockerfile .
docker tag ${IMAGE_NAME} ${FULL_NAME}

# Push image
echo "Pushing Docker image to ECR: ${FULL_NAME}"
docker push ${FULL_NAME}

echo "Docker image successfully built and pushed: ${FULL_NAME}"