import os
import yaml
import numpy as np
import cv2
import logging
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import random
import json

from .augmentations import AugmentationManager


class ConeDataset(Dataset):
    """
    Dataset class for traffic cone detection with YOLOv8.
    
    This class handles loading images and labels, applying augmentations,
    and preprocessing for YOLOv8 training.
    """
    
    def __init__(
        self,
        img_dir: str,
        label_dir: str = None,
        transform: Any = None,
        augmentations_config: Dict = None,
        img_size: int = 640,
        is_train: bool = True
    ):
        """
        Initialize the cone detection dataset.
        
        Args:
            img_dir: Directory containing images
            label_dir: Directory containing labels (defaults to img_dir with 'labels' instead of 'images')
            transform: Optional additional transforms
            augmentations_config: Configuration for augmentations
            img_size: Image size for resizing
            is_train: Whether this is a training dataset (for augmentations)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.img_dir = img_dir
        self.img_size = img_size
        self.is_train = is_train
        
        # If label_dir is not provided, infer it from img_dir
        if label_dir is None:
            self.label_dir = img_dir.replace('images', 'labels')
        else:
            self.label_dir = label_dir
        
        # Get the list of image files
        self.img_files = self._get_image_files(img_dir)
        self.logger.info(f"Found {len(self.img_files)} images in {img_dir}")
        
        # Create augmentation manager if training
        if is_train:
            self.augmenter = AugmentationManager(augmentations_config)
        else:
            self.augmenter = None
        
        # External transform (for validation)
        self.transform = transform
    
    def _get_image_files(self, img_dir: str) -> List[str]:
        """
        Get the list of image files in the directory.
        
        Args:
            img_dir: Directory containing images
            
        Returns:
            List of image file paths
        """
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        img_files = []
        
        if not os.path.exists(img_dir):
            self.logger.warning(f"Image directory {img_dir} does not exist")
            return []
        
        for root, _, files in os.walk(img_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in img_extensions:
                    img_files.append(os.path.join(root, file))
        
        return sorted(img_files)
    
    def __len__(self) -> int:
        """
        Get the number of images in the dataset.
        
        Returns:
            Number of images
        """
        return len(self.img_files)
    
    def __getitem__(self, index: int) -> Dict:
        """
        Get a single item from the dataset.
        
        Args:
            index: Index of the item
            
        Returns:
            Dictionary containing image, labels, and metadata
        """
        # Get image file path
        img_path = self.img_files[index]
        
        # Load image
        img = self._load_image(img_path)
        
        # Get label file path
        label_path = self._get_label_path(img_path)
        
        # Load labels
        bboxes, class_ids = self._load_labels(label_path)
        
        # Apply augmentations if training
        if self.is_train and self.augmenter:
            img, bboxes = self.augmenter.apply(img, bboxes, class_ids)
        
        # Convert to tensors
        if len(img.shape) == 3:  # HWC
            img = img.transpose(2, 0, 1)  # to CHW
        
        # Create target for YOLOv8
        labels = []
        for i, bbox in enumerate(bboxes):
            if i < len(class_ids):
                class_id = class_ids[i]
                # YOLOv8 format: [class_id, x_center, y_center, width, height]
                label = np.array([class_id, *bbox])
                labels.append(label)
        
        # Apply additional transforms if provided
        if self.transform:
            img = self.transform(img)
        
        # Create return dictionary
        item = {
            'img': torch.from_numpy(img).float() if isinstance(img, np.ndarray) else img,
            'labels': torch.from_numpy(np.array(labels)).float() if labels else torch.zeros((0, 5)),
            'img_path': img_path,
            'ori_shape': img.shape[:2]  # H, W
        }
        
        return item
    
    def _load_image(self, img_path: str) -> np.ndarray:
        """
        Load an image from file.
        
        Args:
            img_path: Path to the image file
            
        Returns:
            Image as numpy array
        """
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size
            img = cv2.resize(img, (self.img_size, self.img_size))
            
            return img
            
        except Exception as e:
            self.logger.error(f"Error loading image {img_path}: {e}")
            # Return a blank image as fallback
            return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
    
    def _get_label_path(self, img_path: str) -> str:
        """
        Get the path to the label file corresponding to an image.
        
        Args:
            img_path: Path to the image file
            
        Returns:
            Path to the corresponding label file
        """
        # Get relative path of image from img_dir
        rel_path = os.path.relpath(img_path, self.img_dir)
        
        # Replace extension with .txt
        label_file = os.path.splitext(rel_path)[0] + '.txt'
        
        # Join with label_dir
        return os.path.join(self.label_dir, label_file)
    
    def _load_labels(self, label_path: str) -> Tuple[List[List[float]], List[int]]:
        """
        Load labels from a label file.
        
        Args:
            label_path: Path to the label file (YOLO format)
            
        Returns:
            Tuple of (bounding boxes, class IDs)
        """
        bboxes = []
        class_ids = []
        
        if not os.path.exists(label_path):
            self.logger.debug(f"Label file {label_path} does not exist")
            return bboxes, class_ids
        
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            # YOLO format: class_id, x_center, y_center, width, height
                            bbox = [float(x) for x in parts[1:5]]
                            bboxes.append(bbox)
                            class_ids.append(class_id)
            
            return bboxes, class_ids
            
        except Exception as e:
            self.logger.error(f"Error loading labels from {label_path}: {e}")
            return [], []
    
    def create_subset(self, indices: List[int]) -> 'ConeDataset':
        """
        Create a subset of this dataset with specified indices.
        
        Args:
            indices: List of indices to include in the subset
            
        Returns:
            New ConeDataset instance with only the specified items
        """
        subset = ConeDataset(
            img_dir=self.img_dir,
            label_dir=self.label_dir,
            transform=self.transform,
            augmentations_config=self.augmenter.config if self.augmenter else None,
            img_size=self.img_size,
            is_train=self.is_train
        )
        
        subset.img_files = [self.img_files[i] for i in indices]
        return subset


class DatasetManager:
    """
    Manager for creating and handling datasets for YOLOv8 training.
    
    This class handles loading dataset configurations, creating train/val splits,
    and setting up data loaders with hardware-aware optimizations.
    """
    
    def __init__(
        self,
        data_yaml_path: str,
        config: Dict = None,
        batch_size: int = 16,
        workers: int = 4,
        img_size: int = 640
    ):
        """
        Initialize the dataset manager.
        
        Args:
            data_yaml_path: Path to the data.yaml file
            config: Additional configuration parameters
            batch_size: Batch size for data loaders
            workers: Number of workers for data loaders
            img_size: Image size for input
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.data_yaml_path = data_yaml_path
        self.config = config or {}
        self.batch_size = batch_size
        self.workers = workers
        self.img_size = img_size
        
        # Load data configuration
        self.data_config = self._load_data_config(data_yaml_path)
        
        # Extract paths and augmentation settings
        self.train_path = self._resolve_path(self.data_config.get('train', ''))
        self.val_path = self._resolve_path(self.data_config.get('val', ''))
        self.augmentations_config = self.data_config.get('augmentations', {})
        
        self.logger.info(f"Dataset manager initialized with batch_size={batch_size}, workers={workers}")
        if self.train_path:
            self.logger.info(f"Training data: {self.train_path}")
        if self.val_path:
            self.logger.info(f"Validation data: {self.val_path}")
    
    def _load_data_config(self, data_yaml_path: str) -> Dict:
        """
        Load the data configuration from YAML.
        
        Args:
            data_yaml_path: Path to the data.yaml file
            
        Returns:
            Dictionary with data configuration
        """
        try:
            with open(data_yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)
                
                # Process any environment variables or placeholders
                data_config = self._process_variables(data_config)
                
                return data_config
                
        except Exception as e:
            self.logger.error(f"Error loading data config from {data_yaml_path}: {e}")
            return {}
    
    def _process_variables(self, config: Dict) -> Dict:
        """
        Process environment variables and placeholders in the configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Processed configuration dictionary
        """
        import os
        
        processed_config = {}
        
        for key, value in config.items():
            if isinstance(value, dict):
                processed_config[key] = self._process_variables(value)
            elif isinstance(value, str):
                # Replace environment variables like ${VAR}
                if "${" in value and "}" in value:
                    start = value.find("${") + 2
                    end = value.find("}", start)
                    if start > 1 and end > start:
                        var_name = value[start:end]
                        var_value = os.environ.get(var_name, "")
                        value = value.replace(f"${{{var_name}}}", var_value)
                processed_config[key] = value
            else:
                processed_config[key] = value
        
        return processed_config
    
    def _resolve_path(self, path: str) -> str:
        """
        Resolve a path relative to the data.yaml file.
        
        Args:
            path: Path to resolve
            
        Returns:
            Resolved absolute path
        """
        if not path:
            return ""
        
        if os.path.isabs(path):
            return path
        
        # Get directory of data.yaml
        base_dir = os.path.dirname(os.path.abspath(self.data_yaml_path))
        
        # Resolve path relative to data.yaml
        return os.path.normpath(os.path.join(base_dir, path))
    
    def create_datasets(self) -> Tuple[ConeDataset, Optional[ConeDataset]]:
        """
        Create training and validation datasets.
        
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        # Create training dataset
        train_dataset = None
        if self.train_path and os.path.exists(self.train_path):
            train_dataset = ConeDataset(
                img_dir=self.train_path,
                transform=None,  # No additional transforms needed
                augmentations_config=self.augmentations_config,
                img_size=self.img_size,
                is_train=True
            )
        
        # Create validation dataset
        val_dataset = None
        if self.val_path and os.path.exists(self.val_path):
            val_dataset = ConeDataset(
                img_dir=self.val_path,
                transform=None,  # No additional transforms needed
                augmentations_config=None,  # No augmentations for validation
                img_size=self.img_size,
                is_train=False
            )
        elif train_dataset and not self.val_path:
            # If no validation path specified, create a split from training data
            split_ratio = self.data_config.get('split_ratio', 0.2)
            train_dataset, val_dataset = self.create_train_val_split(train_dataset, split_ratio)
        
        return train_dataset, val_dataset
    
    def create_train_val_split(self, dataset: ConeDataset, val_ratio: float = 0.2) -> Tuple[ConeDataset, ConeDataset]:
        """
        Create a train/validation split from a dataset.
        
        Args:
            dataset: Dataset to split
            val_ratio: Ratio of validation samples (0-1)
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        # Calculate sizes
        val_size = int(len(dataset) * val_ratio)
        train_size = len(dataset) - val_size
        
        if val_size == 0:
            self.logger.warning(f"Validation size is 0 with val_ratio={val_ratio} and dataset size {len(dataset)}")
            val_size = 1
            train_size = len(dataset) - val_size
        
        # Generate indices
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create subsets
        train_dataset = dataset.create_subset(train_indices)
        train_dataset.is_train = True
        
        val_dataset = dataset.create_subset(val_indices)
        val_dataset.is_train = False
        val_dataset.augmenter = None  # Disable augmentations for validation
        
        self.logger.info(f"Created train/val split: {train_size}/{val_size} samples")
        
        return train_dataset, val_dataset
    
    def create_data_loaders(self, train_dataset: ConeDataset, val_dataset: Optional[ConeDataset] = None) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Create data loaders for training and validation.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.workers,
                pin_memory=True,
                collate_fn=self.collate_fn
            )
        
        return train_loader, val_loader
    
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        """
        Collate function for batching dataset items.
        
        Args:
            batch: List of items from dataset
            
        Returns:
            Batched data
        """
        imgs = []
        labels = []
        paths = []
        
        for item in batch:
            imgs.append(item['img'])
            labels.append(item['labels'])
            paths.append(item['img_path'])
        
        # Stack images
        imgs = torch.stack(imgs)
        
        return {
            'imgs': imgs,
            'labels': labels,  # Keep as list since they may have different sizes
            'paths': paths
        }
    
    def export_dataset_stats(self, output_path: str, train_dataset: ConeDataset, val_dataset: Optional[ConeDataset] = None) -> None:
        """
        Export statistics about the dataset.
        
        Args:
            output_path: Path to save statistics
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
        """
        stats = {
            'train': {
                'size': len(train_dataset),
                'img_dir': train_dataset.img_dir,
                'label_dir': train_dataset.label_dir
            }
        }
        
        if val_dataset:
            stats['val'] = {
                'size': len(val_dataset),
                'img_dir': val_dataset.img_dir,
                'label_dir': val_dataset.label_dir
            }
        
        # Calculate class distribution
        class_counts = self._calculate_class_distribution(train_dataset, val_dataset)
        stats['class_distribution'] = class_counts
        
        # Save statistics
        try:
            with open(output_path, 'w') as f:
                json.dump(stats, f, indent=2)
            self.logger.info(f"Exported dataset statistics to {output_path}")
        except Exception as e:
            self.logger.error(f"Error exporting dataset statistics: {e}")
    
    def _calculate_class_distribution(self, train_dataset: ConeDataset, val_dataset: Optional[ConeDataset] = None) -> Dict[str, Dict[str, int]]:
        """
        Calculate the distribution of classes in the datasets.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            
        Returns:
            Dictionary with class distribution statistics
        """
        class_names = self.data_config.get('names', {})
        train_counts = {i: 0 for i in class_names}
        val_counts = {i: 0 for i in class_names}
        
        # Count classes in training set
        for i in range(min(len(train_dataset), 1000)):  # Sample up to 1000 items
            item = train_dataset[i]
            labels = item['labels']
            if isinstance(labels, torch.Tensor):
                for label in labels:
                    class_id = int(label[0])
                    if class_id in train_counts:
                        train_counts[class_id] += 1
        
        # Count classes in validation set if available
        if val_dataset:
            for i in range(min(len(val_dataset), 1000)):  # Sample up to 1000 items
                item = val_dataset[i]
                labels = item['labels']
                if isinstance(labels, torch.Tensor):
                    for label in labels:
                        class_id = int(label[0])
                        if class_id in val_counts:
                            val_counts[class_id] += 1
        
        # Convert to human-readable format
        result = {
            'train': {class_names.get(str(i), f"class_{i}"): count for i, count in train_counts.items()},
            'val': {class_names.get(str(i), f"class_{i}"): count for i, count in val_counts.items()} if val_dataset else {}
        }
        
        return result


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Load configuration
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Dataset and augmentation example')
    parser.add_argument('--config', type=str, default='../../config.yaml', help='Path to main configuration')
    parser.add_argument('--data', type=str, default='../../data/data.yaml', help='Path to data configuration')
    args = parser.parse_args()
    
    # Check if paths exist
    if not os.path.exists(args.config):
        print(f"Config file {args.config} not found")
        args.config = None
    
    if not os.path.exists(args.data):
        print(f"Data config file {args.data} not found")
        exit(1)
    
    # Initialize dataset manager
    manager = DatasetManager(args.data, batch_size=8, workers=2)
    
    # Create datasets
    train_dataset, val_dataset = manager.create_datasets()
    
    if train_dataset:
        print(f"Training dataset: {len(train_dataset)} images")
        
        # Get a sample
        sample = train_dataset[0]
        print(f"Sample image shape: {sample['img'].shape}")
        print(f"Sample labels: {sample['labels']}")
    
    if val_dataset:
        print(f"Validation dataset: {len(val_dataset)} images")
    
    # Create data loaders
    if train_dataset:
        train_loader, val_loader = manager.create_data_loaders(train_dataset, val_dataset)
        
        # Test batch loading
        batch = next(iter(train_loader))
        print(f"Batch images shape: {batch['imgs'].shape}")
        print(f"Batch contains {len(batch['labels'])} label lists")
    
    print("Dataset testing completed successfully")
