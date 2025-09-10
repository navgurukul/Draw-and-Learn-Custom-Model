# Draw-and-Learn Custom Model

A comprehensive machine learning project for training custom drawing recognition models with AWS SageMaker Unified Studio integration.

## 🚀 Overview

This repository provides a complete framework for developing, training, and deploying custom models for drawing/image recognition using AWS SageMaker. It's specifically designed to work seamlessly with SageMaker Unified Studio for streamlined ML workflows.

## 📁 Project Structure

```
Draw-and-Learn-Custom-Model/
├── config/                     # Configuration files
│   ├── sagemaker_config.json   # SageMaker settings
│   └── training_config.yaml    # Training hyperparameters
├── data/                       # Data directories
│   ├── raw/                    # Raw input data
│   ├── processed/              # Processed training data
│   └── external/               # External datasets
├── docker/                     # Docker configurations
│   ├── Dockerfile              # SageMaker container
│   └── build_and_push.sh      # Build script
├── models/                     # Trained model artifacts
├── notebooks/                  # Jupyter notebooks
│   └── sagemaker_training_example.ipynb
├── scripts/                    # Utility scripts
│   ├── training/               # Training scripts
│   ├── processing/             # Data processing scripts
│   └── inference/              # Inference scripts
├── src/                        # Source code
│   ├── data/                   # Data utilities
│   ├── models/                 # Model definitions
│   ├── visualization/          # Visualization tools
│   └── sagemaker_utils.py      # SageMaker helpers
├── tests/                      # Unit tests
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
└── README.md                   # This file
```

## 🛠️ AWS SageMaker Unified Studio Integration

This project is specifically designed for AWS SageMaker Unified Studio with the following features:

### Key Features
- **Pre-configured SageMaker Settings**: Ready-to-use configuration for SageMaker Studio
- **Custom Container Support**: Docker configurations for custom training environments
- **Automated Data Processing**: SageMaker Processing jobs for data preprocessing
- **Flexible Model Architecture**: Multiple model types for different use cases
- **End-to-End Pipeline**: From data ingestion to model deployment
- **Monitoring & Logging**: Built-in support for CloudWatch monitoring

### SageMaker Components Used
- **SageMaker Studio**: Integrated development environment
- **SageMaker Training Jobs**: Scalable model training
- **SageMaker Processing Jobs**: Data preprocessing at scale  
- **SageMaker Endpoints**: Real-time model inference
- **SageMaker Pipelines**: ML workflow orchestration
- **Model Registry**: Model versioning and management

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/navgurukul/Draw-and-Learn-Custom-Model.git
cd Draw-and-Learn-Custom-Model

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### 2. AWS Configuration

Ensure you have AWS credentials configured with appropriate SageMaker permissions:

```bash
aws configure
```

Required IAM permissions:
- SageMaker full access
- S3 read/write access
- ECR access (for custom containers)
- CloudWatch logs access

### 3. Update Configuration

Edit `config/sagemaker_config.json` with your AWS account details:

```json
{
  "sagemaker_studio": {
    "execution_role": "arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole",
    "region": "your-aws-region"
  },
  "project_settings": {
    "bucket_name": "your-sagemaker-bucket",
    "project_name": "draw-and-learn-custom-model"
  }
}
```

### 4. Data Preparation

Add your drawing/image data to the `data/raw/` directory:

```
data/raw/
├── class1/
│   ├── image1.jpg
│   └── image2.jpg
├── class2/
│   ├── image3.jpg
│   └── image4.jpg
└── ...
```

### 5. SageMaker Studio Setup

Launch SageMaker Studio and open the example notebook:

```bash
# In SageMaker Studio, open:
notebooks/sagemaker_training_example.ipynb
```

## 💡 Usage Examples

### Local Development

```python
from src.sagemaker_utils import SageMakerHelper
from src.models.custom_model import get_model

# Initialize SageMaker helper
helper = SageMakerHelper()

# Create a model
model = get_model('drawing_cnn', num_classes=10)

# Upload data to S3
data_uri = helper.upload_data_to_s3('data/raw', 'training-data')
```

### SageMaker Training Job

```python
# Create and run training job
estimator = helper.create_training_job(
    entry_point='train.py',
    source_dir='scripts/training',
    train_data_uri=train_data_uri,
    hyperparameters={'epochs': 10, 'batch-size': 32}
)

estimator.fit({'training': train_data_uri})
```

### Model Deployment

```python
# Deploy trained model
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large'
)

# Make predictions
result = predictor.predict(test_data)
```

## 🔧 Customization

### Adding New Models

Create new model architectures in `src/models/custom_model.py`:

```python
class YourCustomModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Your model architecture
    
    def forward(self, x):
        # Forward pass logic
        return x
```

### Custom Data Processing

Modify `scripts/processing/preprocess.py` for your specific data requirements:

```python
def custom_preprocessing(image):
    # Your custom preprocessing logic
    return processed_image
```

### Hyperparameter Tuning

Use SageMaker Hyperparameter Tuning:

```python
from sagemaker.tuner import HyperparameterTuner

tuner = HyperparameterTuner(
    estimator=estimator,
    objective_metric_name='validation:accuracy',
    hyperparameter_ranges={
        'learning-rate': ContinuousParameter(0.0001, 0.1),
        'batch-size': CategoricalParameter([16, 32, 64])
    }
)
```

## 🧪 Testing

Run tests to ensure everything works correctly:

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## 📊 Monitoring and Logging

### CloudWatch Integration

The project includes automatic logging to CloudWatch for:
- Training metrics
- Model performance
- Infrastructure utilization
- Error tracking

### Model Monitoring

Set up model monitoring for deployed endpoints:

```python
from sagemaker.model_monitor import DefaultModelMonitor

monitor = DefaultModelMonitor(
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge'
)

monitor.suggest_baseline(
    baseline_dataset=baseline_data_uri,
    dataset_format=DatasetFormat.csv(header=True)
)
```

## 🔄 CI/CD Pipeline

The project supports automated ML pipelines using SageMaker Pipelines:

```python
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

# Define pipeline steps
preprocessing_step = ProcessingStep(...)
training_step = TrainingStep(...)

# Create pipeline
pipeline = Pipeline(
    name="DrawLearnPipeline",
    steps=[preprocessing_step, training_step]
)

# Execute pipeline
pipeline.upsert(role_arn=role)
pipeline.start()
```

## 📚 Additional Resources

### Documentation Links
- [AWS SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/)
- [SageMaker Python SDK](https://sagemaker.readthedocs.io/)
- [SageMaker Studio User Guide](https://docs.aws.amazon.com/sagemaker/latest/ug/)

### Example Use Cases
- Handwritten digit recognition
- Sketch classification
- Drawing style analysis
- Educational drawing assessment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in this repository
- Contact NavGurukul team at info@navgurukul.org
- Check the AWS SageMaker documentation

## 🔮 Roadmap

- [ ] Add support for more model architectures
- [ ] Implement automated hyperparameter tuning
- [ ] Add real-time data streaming capabilities
- [ ] Integrate with SageMaker Feature Store
- [ ] Add multi-modal learning support
- [ ] Implement federated learning capabilities