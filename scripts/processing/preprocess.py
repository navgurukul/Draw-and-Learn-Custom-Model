"""
SageMaker data preprocessing script for Draw-and-Learn Custom Model.
This script handles data preprocessing for drawing/image data.
"""

import argparse
import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_preprocess_images(input_dir, output_dir, image_size=(224, 224)):
    """Load and preprocess drawing images."""
    logger.info(f"Processing images from: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'validation'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    
    processed_data = []
    
    # Walk through input directory
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                
                try:
                    # Load image
                    image = cv2.imread(file_path)
                    if image is None:
                        continue
                    
                    # Preprocess image
                    processed_image = preprocess_single_image(image, image_size)
                    
                    # Extract label from directory structure or filename
                    label = extract_label(file_path, root)
                    
                    processed_data.append({
                        'image_path': file_path,
                        'processed_image': processed_image,
                        'label': label,
                        'original_shape': image.shape
                    })
                    
                    logger.info(f"Processed: {file_path}")
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
    
    logger.info(f"Total images processed: {len(processed_data)}")
    
    # Split data
    train_data, temp_data = train_test_split(processed_data, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    # Save processed data
    save_processed_data(train_data, os.path.join(output_dir, 'train'))
    save_processed_data(val_data, os.path.join(output_dir, 'validation'))
    save_processed_data(test_data, os.path.join(output_dir, 'test'))
    
    # Create metadata
    metadata = {
        'total_samples': len(processed_data),
        'train_samples': len(train_data),
        'validation_samples': len(val_data),
        'test_samples': len(test_data),
        'image_size': image_size,
        'unique_labels': list(set([item['label'] for item in processed_data]))
    }
    
    # Save metadata
    import json
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Data preprocessing completed successfully")
    return metadata


def preprocess_single_image(image, target_size):
    """Preprocess a single image."""
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize image
    resized = cv2.resize(image_rgb, target_size)
    
    # Normalize pixel values
    normalized = resized.astype(np.float32) / 255.0
    
    return normalized


def extract_label(file_path, root_dir):
    """Extract label from file path or directory structure."""
    # This is a placeholder - implement based on your labeling scheme
    # For now, use parent directory name as label
    parent_dir = os.path.basename(os.path.dirname(file_path))
    
    # If parent directory is generic, try to extract from filename
    if parent_dir in ['images', 'data', 'raw']:
        # Extract from filename (assumes format like "class_001.jpg")
        filename = os.path.splitext(os.path.basename(file_path))[0]
        if '_' in filename:
            return filename.split('_')[0]
        else:
            return 'unknown'
    
    return parent_dir


def save_processed_data(data_list, output_dir):
    """Save processed data to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save images and create index file
    index_data = []
    
    for i, item in enumerate(data_list):
        # Save processed image
        image_filename = f"image_{i:06d}.npy"
        image_path = os.path.join(output_dir, image_filename)
        np.save(image_path, item['processed_image'])
        
        # Add to index
        index_data.append({
            'image_file': image_filename,
            'label': item['label'],
            'original_path': item['image_path'],
            'original_shape': item['original_shape']
        })
    
    # Save index file
    df = pd.DataFrame(index_data)
    df.to_csv(os.path.join(output_dir, 'index.csv'), index=False)
    
    logger.info(f"Saved {len(data_list)} processed images to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess drawing data for SageMaker')
    
    # SageMaker processing arguments
    parser.add_argument('--input-data', type=str, default='/opt/ml/processing/input')
    parser.add_argument('--output-data', type=str, default='/opt/ml/processing/output')
    
    # Processing parameters
    parser.add_argument('--image-size', type=int, nargs=2, default=[224, 224])
    parser.add_argument('--train-split', type=float, default=0.7)
    parser.add_argument('--val-split', type=float, default=0.2)
    
    args = parser.parse_args()
    
    logger.info("Preprocessing arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Run preprocessing
    metadata = load_and_preprocess_images(
        input_dir=args.input_data,
        output_dir=args.output_data,
        image_size=tuple(args.image_size)
    )
    
    logger.info("Preprocessing completed successfully")
    logger.info(f"Metadata: {metadata}")