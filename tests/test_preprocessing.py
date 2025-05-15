import os
import sys
import unittest
import tempfile
import shutil
import json
import yaml
import numpy as np
from PIL import Image
import cv2

# Add the parent directory to the path so we can import the modules under test
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.augmentations import AugmentationManager, LensDistortion
from src.preprocessing.dataset import ConeDataset, DatasetManager


class TestAugmentations(unittest.TestCase):
    """Test the image augmentation functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create a test image
        self.img_size = 640
        self.test_image = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * 128
        
        # Draw some shapes for visualization and testing
        cv2.rectangle(self.test_image, (100, 100), (300, 300), (255, 0, 0), 5)  # Blue rectangle
        cv2.circle(self.test_image, (400, 400), 100, (0, 255, 0), 5)  # Green circle
        
        # Save test image
        self.image_path = os.path.join(self.test_dir, "test_image.png")
        cv2.imwrite(self.image_path, self.test_image)
        
        # Create test bounding boxes [x_center, y_center, width, height] (normalized)
        self.bboxes = [
            [0.3125, 0.3125, 0.3125, 0.3125],  # Rectangle
            [0.625, 0.625, 0.3125, 0.3125]     # Circle
        ]
        
        self.class_labels = [0, 2]  # Blue, Orange
        
        # Create default augmentation config
        self.aug_config = {
            'noise': {
                'enabled': True,
                'gaussian_std': 0.03,
                'salt_pepper_prob': 0.01
            },
            'distortion': {
                'enabled': True,
                'barrel_factor': 0.2,
                'pincushion_factor': 0.1
            },
            'distance_simulation': {
                'enabled': True,
                'min_scale': 0.5,
                'max_scale': 1.5
            },
            'motion_blur': {
                'enabled': True,
                'kernel_size': 7,
                'angle_range': [-45, 45]
            },
            'occlusion': {
                'enabled': True,
                'max_area_percentage': 0.20
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures after each test."""
        shutil.rmtree(self.test_dir)
    
    def test_augmentation_manager_init(self):
        """Test initialization of AugmentationManager with different configs."""
        # Test with default config
        augmenter = AugmentationManager()
        self.assertIsNotNone(augmenter.transform)
        
        # Test with custom config
        custom_config = {
            'noise': {
                'enabled': False
            },
            'motion_blur': {
                'kernel_size': 9
            }
        }
        augmenter = AugmentationManager(custom_config)
        self.assertFalse(augmenter.config['noise']['enabled'])
        self.assertEqual(augmenter.config['motion_blur']['kernel_size'], 9)
    
    def test_apply_augmentations(self):
        """Test applying augmentations to an image and bounding boxes."""
        augmenter = AugmentationManager(self.aug_config)
        
        # Apply augmentations
        augmented_img, augmented_bboxes = augmenter.apply(self.test_image, self.bboxes, self.class_labels)
        
        # Check that image shape is preserved
        self.assertEqual(augmented_img.shape, self.test_image.shape)
        
        # Check that bounding boxes are still present
        self.assertEqual(len(augmented_bboxes), len(self.bboxes))
        
        # Check that bounding box format is correct
        for bbox in augmented_bboxes:
            self.assertEqual(len(bbox), 4)  # x_center, y_center, width, height
    
    def test_disable_augmentations(self):
        """Test disabling all augmentations."""
        disabled_config = {}
        for aug_type in self.aug_config:
            disabled_config[aug_type] = {'enabled': False}
        
        augmenter = AugmentationManager(disabled_config)
        
        # Apply augmentations
        augmented_img, augmented_bboxes = augmenter.apply(self.test_image, self.bboxes, self.class_labels)
        
        # Check that bounding boxes remain the same (within some small delta for floating point)
        for i, bbox in enumerate(augmented_bboxes):
            for j in range(4):
                self.assertAlmostEqual(bbox[j], self.bboxes[i][j], delta=0.1)
    
    def test_lens_distortion(self):
        """Test the custom LensDistortion transform."""
        # Create distortion transform
        distortion = LensDistortion(barrel_factor=0.2, pincushion_factor=0.1)
        
        # Apply distortion
        distorted = distortion.apply(self.test_image)
        
        # Check that image shape is preserved
        self.assertEqual(distorted.shape, self.test_image.shape)
        
        # Check that image has been modified
        self.assertFalse(np.array_equal(distorted, self.test_image))
    
    def test_single_augmentation(self):
        """Test applying individual augmentations."""
        augmenter = AugmentationManager(self.aug_config)
        
        # Test each augmentation type
        for aug_type in ['noise', 'distortion', 'motion_blur', 'occlusion']:
            augmented = augmenter.apply_single_augmentation(self.test_image, aug_type)
            
            # Check that image shape is preserved
            self.assertEqual(augmented.shape, self.test_image.shape)
            
            # Check that image has been modified
            self.assertFalse(np.array_equal(augmented, self.test_image))
    
    def test_augmentation_visualization(self):
        """Test visualization of augmentations."""
        augmenter = AugmentationManager(self.aug_config)
        
        # Create visualization output directory
        vis_dir = os.path.join(self.test_dir, "visualization")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Generate visualizations
        augmenter.visualize_augmentations(self.image_path, vis_dir)
        
        # Check that files were created
        expected_files = ['original.jpg', 'noise.jpg', 'distortion.jpg', 
                          'motion_blur.jpg', 'occlusion.jpg', 'combined.jpg']
        
        for file in expected_files:
            self.assertTrue(os.path.exists(os.path.join(vis_dir, file)))


class TestDataset(unittest.TestCase):
    """Test the dataset and data loading functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create directory structure
        self.data_dir = os.path.join(self.test_dir, "data")
        self.train_dir = os.path.join(self.data_dir, "train")
        self.train_images = os.path.join(self.train_dir, "images")
        self.train_labels = os.path.join(self.train_dir, "labels")
        
        os.makedirs(self.train_images, exist_ok=True)
        os.makedirs(self.train_labels, exist_ok=True)
        
        # Create test images and labels
        self.num_images = 5
        self.img_size = 640
        
        for i in range(self.num_images):
            # Create image
            img = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * 128
            cv2.rectangle(img, (100, 100), (300, 300), (255, 0, 0), 5)
            img_path = os.path.join(self.train_images, f"image_{i}.png")
            cv2.imwrite(img_path, img)
            
            # Create label file
            label_path = os.path.join(self.train_labels, f"image_{i}.txt")
            with open(label_path, 'w') as f:
                f.write("0 0.3125 0.3125 0.3125 0.3125\n")  # Blue cone
                f.write("2 0.625 0.625 0.3125 0.3125\n")    # Orange cone
        
        # Create data.yaml
        self.data_yaml = os.path.join(self.data_dir, "data.yaml")
        data_config = {
            'train': f"{self.train_images}",
            'val': '',
            'nc': 5,
            'names': {
                '0': 'blue',
                '1': 'big_orange',
                '2': 'orange',
                '3': 'unknown',
                '4': 'yellow'
            },
            'augmentations': {
                'noise': {'enabled': True},
                'distortion': {'enabled': False}
            },
            'split_ratio': 0.2
        }
        
        with open(self.data_yaml, 'w') as f:
            yaml.dump(data_config, f)
    
    def tearDown(self):
        """Clean up test fixtures after each test."""
        shutil.rmtree(self.test_dir)
    
    def test_cone_dataset_init(self):
        """Test initialization of ConeDataset."""
        dataset = ConeDataset(
            img_dir=self.train_images,
            label_dir=self.train_labels,
            img_size=self.img_size,
            is_train=True
        )
        
        # Check that the dataset contains the correct number of images
        self.assertEqual(len(dataset), self.num_images)
    
    def test_cone_dataset_getitem(self):
        """Test getting an item from the dataset."""
        dataset = ConeDataset(
            img_dir=self.train_images,
            label_dir=self.train_labels,
            img_size=self.img_size,
            is_train=False  # Disable augmentations for deterministic results
        )
        
        # Get an item
        item = dataset[0]
        
        # Check item structure
        self.assertIn('img', item)
        self.assertIn('labels', item)
        self.assertIn('img_path', item)
        
        # Check image shape
        img = item['img']
        self.assertEqual(img.shape[0], 3)  # C
        self.assertEqual(img.shape[1], self.img_size)  # H
        self.assertEqual(img.shape[2], self.img_size)  # W
        
        # Check labels
        labels = item['labels']
        self.assertEqual(labels.shape[0], 2)  # 2 labels
        self.assertEqual(labels.shape[1], 5)  # class_id, x, y, w, h
    
    def test_cone_dataset_augmentation(self):
        """Test that augmentations are applied when is_train=True."""
        # Create dataset with augmentations
        dataset_with_aug = ConeDataset(
            img_dir=self.train_images,
            label_dir=self.train_labels,
            img_size=self.img_size,
            is_train=True,
            augmentations_config={'noise': {'enabled': True}}
        )
        
        # Create dataset without augmentations
        dataset_no_aug = ConeDataset(
            img_dir=self.train_images,
            label_dir=self.train_labels,
            img_size=self.img_size,
            is_train=False
        )
        
        # Get same item from both
        item_with_aug = dataset_with_aug[0]
        item_no_aug = dataset_no_aug[0]
        
        # Check that images are different due to augmentation
        img_with_aug = item_with_aug['img'].numpy()
        img_no_aug = item_no_aug['img'].numpy()
        
        # Images should differ due to augmentation
        self.assertFalse(np.array_equal(img_with_aug, img_no_aug))
    
    def test_dataset_manager_init(self):
        """Test initialization of DatasetManager."""
        manager = DatasetManager(
            data_yaml_path=self.data_yaml,
            batch_size=8,
            workers=2
        )
        
        # Check that paths are loaded correctly
        self.assertEqual(manager.train_path, self.train_images)
        
        # Check that augmentation config is loaded
        self.assertTrue(manager.augmentations_config['noise']['enabled'])
        self.assertFalse(manager.augmentations_config['distortion']['enabled'])
    
    def test_dataset_manager_create_datasets(self):
        """Test creating datasets from DatasetManager."""
        manager = DatasetManager(
            data_yaml_path=self.data_yaml,
            batch_size=8,
            workers=2
        )
        
        # Create datasets
        train_dataset, val_dataset = manager.create_datasets()
        
        # Check that datasets were created
        self.assertIsNotNone(train_dataset)
        self.assertIsNotNone(val_dataset)
        
        # Check that train/val split was applied
        self.assertEqual(len(train_dataset) + len(val_dataset), self.num_images)
        
        # Check dataset types
        self.assertIsInstance(train_dataset, ConeDataset)
        self.assertIsInstance(val_dataset, ConeDataset)
    
    def test_dataset_manager_create_data_loaders(self):
        """Test creating data loaders from DatasetManager."""
        manager = DatasetManager(
            data_yaml_path=self.data_yaml,
            batch_size=2,  # Small batch size for test
            workers=0  # Use no workers for test
        )
        
        # Create datasets
        train_dataset, val_dataset = manager.create_datasets()
        
        # Create loaders
        train_loader, val_loader = manager.create_data_loaders(train_dataset, val_dataset)
        
        # Check loaders
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        
        # Test batch loading
        batch = next(iter(train_loader))
        
        # Check batch structure
        self.assertIn('imgs', batch)
        self.assertIn('labels', batch)
        self.assertIn('paths', batch)
        
        # Check batch size
        self.assertEqual(batch['imgs'].shape[0], 2)  # Batch size
    
    def test_dataset_manager_export_stats(self):
        """Test exporting dataset statistics."""
        manager = DatasetManager(
            data_yaml_path=self.data_yaml,
            batch_size=8,
            workers=2
        )
        
        # Create datasets
        train_dataset, val_dataset = manager.create_datasets()
        
        # Export stats
        stats_path = os.path.join(self.test_dir, "dataset_stats.json")
        manager.export_dataset_stats(stats_path, train_dataset, val_dataset)
        
        # Check that file was created
        self.assertTrue(os.path.exists(stats_path))
        
        # Load and check stats
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        # Check stats structure
        self.assertIn('train', stats)
        self.assertIn('val', stats)
        self.assertIn('class_distribution', stats)


if __name__ == "__main__":
    unittest.main()
