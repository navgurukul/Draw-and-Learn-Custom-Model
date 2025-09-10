# SageMaker compatible Docker image for Draw-and-Learn Custom Model
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.12.0-gpu-py38-cu113-ubuntu20.04-sagemaker

# Install additional dependencies
COPY requirements.txt /opt/ml/code/requirements.txt
RUN pip install -r /opt/ml/code/requirements.txt

# Copy source code
COPY src/ /opt/ml/code/
COPY scripts/ /opt/ml/code/scripts/
COPY config/ /opt/ml/code/config/

# Set the working directory
WORKDIR /opt/ml/code

# Set environment variables for SageMaker
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml/code:${PATH}"

# SageMaker sets up the entry point
ENTRYPOINT ["python", "scripts/training/train.py"]