"""
Custom dataset classes for Draw-and-Learn project.
Handles loading and preprocessing of drawing/image data.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DrawingDataset(Dataset):
    """Custom dataset for drawing/image data."""
    
    def __init__(self, data_dir, index_file='index.csv', transform=None, mode='train'):
        """
        Initialize the drawing dataset.
        
        Args:
            data_dir (str): Path to the data directory
            index_file (str): Name of the index CSV file
            transform: Data augmentation transforms
            mode (str): 'train', 'validation', or 'test'
        """
        self.data_dir = data_dir
        self.mode = mode
        
        # Load index file
        index_path = os.path.join(data_dir, index_file)
        if os.path.exists(index_path):
            self.index_df = pd.read_csv(index_path)
        else:
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        # Setup transforms
        self.transform = transform or self.get_default_transforms()
        
        # Create label mapping
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(self.index_df['label'].unique()))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
    def __len__(self):
        return len(self.index_df)
    
    def __getitem__(self, idx):
        """Get a single item from the dataset."""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image info
        row = self.index_df.iloc[idx]
        image_file = row['image_file']
        label = row['label']
        
        # Load image
        if image_file.endswith('.npy'):
            # Load preprocessed numpy array
            image_path = os.path.join(self.data_dir, image_file)
            image = np.load(image_path)
        else:
            # Load raw image file
            image_path = os.path.join(self.data_dir, image_file)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            if isinstance(self.transform, A.Compose):
                # Albumentations transforms
                augmented = self.transform(image=image)
                image = augmented['image']
            else:
                # PyTorch transforms
                image = self.transform(image)
        
        # Convert label to index
        label_idx = self.label_to_idx[label]
        
        return image, torch.tensor(label_idx, dtype=torch.long)
    
    def get_default_transforms(self):
        """Get default data augmentation transforms."""
        if self.mode == 'train':
            return A.Compose([
                A.Resize(224, 224),
                A.Rotate(limit=15, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def get_class_weights(self):
        """Calculate class weights for imbalanced datasets."""
        label_counts = self.index_df['label'].value_counts()
        total_samples = len(self.index_df)
        
        class_weights = {}
        for label, count in label_counts.items():
            class_weights[self.label_to_idx[label]] = total_samples / (len(label_counts) * count)
        
        # Convert to tensor
        weights = torch.zeros(len(self.label_to_idx))
        for idx, weight in class_weights.items():
            weights[idx] = weight
        
        return weights


class DrawingDataLoader:
    """Data loader utility for drawing datasets."""
    
    def __init__(self, data_root, batch_size=32, num_workers=4):
        """
        Initialize data loaders for train, validation, and test sets.
        
        Args:
            data_root (str): Root directory containing train/validation/test folders
            batch_size (int): Batch size for data loading
            num_workers (int): Number of worker processes for data loading
        """
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        self._setup_datasets()
    
    def _setup_datasets(self):
        """Setup datasets for each split."""
        train_dir = os.path.join(self.data_root, 'train')
        val_dir = os.path.join(self.data_root, 'validation')
        test_dir = os.path.join(self.data_root, 'test')
        
        if os.path.exists(train_dir):
            self.train_dataset = DrawingDataset(train_dir, mode='train')
        
        if os.path.exists(val_dir):
            self.val_dataset = DrawingDataset(val_dir, mode='validation')
        
        if os.path.exists(test_dir):
            self.test_dataset = DrawingDataset(test_dir, mode='test')
    
    def get_train_loader(self):
        """Get training data loader."""
        if self.train_dataset is None:
            raise ValueError("Training dataset not available")
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_val_loader(self):
        """Get validation data loader."""
        if self.val_dataset is None:
            raise ValueError("Validation dataset not available")
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_test_loader(self):
        """Get test data loader."""
        if self.test_dataset is None:
            raise ValueError("Test dataset not available")
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_dataset_info(self):
        """Get information about the datasets."""
        info = {}
        
        if self.train_dataset:
            info['train'] = {
                'num_samples': len(self.train_dataset),
                'num_classes': len(self.train_dataset.label_to_idx),
                'classes': list(self.train_dataset.label_to_idx.keys())
            }
        
        if self.val_dataset:
            info['validation'] = {
                'num_samples': len(self.val_dataset),
                'num_classes': len(self.val_dataset.label_to_idx),
                'classes': list(self.val_dataset.label_to_idx.keys())
            }
        
        if self.test_dataset:
            info['test'] = {
                'num_samples': len(self.test_dataset),
                'num_classes': len(self.test_dataset.label_to_idx),
                'classes': list(self.test_dataset.label_to_idx.keys())
            }
        
        return info