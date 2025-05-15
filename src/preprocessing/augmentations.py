import cv2
import numpy as np
import random
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
import torch
from PIL import Image, ImageFilter


class AugmentationManager:
    """
    Manager for applying various augmentations to images for training.
    
    This class provides a collection of augmentation techniques that can be
    applied to images during training to improve model robustness and generalization.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the augmentation manager.
        
        Args:
            config: Dictionary containing augmentation configuration
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Default configuration
        self.config = {
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
        
        # Update with provided config
        if config:
            self._update_config(config)
        
        # Initialize augmentation pipeline
        self.transform = self._create_transform_pipeline()
    
    def _update_config(self, config: Dict[str, Any]) -> None:
        """
        Update configuration with provided values.
        
        Args:
            config: Dictionary with configuration values to update
        """
        for category, settings in config.items():
            if category in self.config:
                for key, value in settings.items():
                    if key in self.config[category]:
                        self.config[category][key] = value
    
    def _create_transform_pipeline(self) -> A.Compose:
        """
        Create an augmentation pipeline using Albumentations.
        
        Returns:
            Albumentations Compose object with configured transformations
        """
        transforms = []
        
        # Add noise augmentations
        if self.config['noise']['enabled']:
            # Gaussian noise
            transforms.append(
                A.GaussNoise(
                    var_limit=(0, int(self.config['noise']['gaussian_std'] * 255)),
                    p=0.5
                )
            )
            # Salt and pepper noise
            transforms.append(
                A.MultiplicativeNoise(
                    multiplier=(1 - self.config['noise']['salt_pepper_prob'], 
                               1 + self.config['noise']['salt_pepper_prob']),
                    per_channel=True,
                    p=0.5
                )
            )
        
        # Add lens distortion (custom transform)
        if self.config['distortion']['enabled']:
            transforms.append(
                LensDistortion(
                    barrel_factor=self.config['distortion']['barrel_factor'],
                    pincushion_factor=self.config['distortion']['pincushion_factor'],
                    p=0.5
                )
            )
        
        # Add distance simulation via scaling
        if self.config['distance_simulation']['enabled']:
            transforms.append(
                A.RandomScale(
                    scale_limit=(
                        self.config['distance_simulation']['min_scale'] - 1.0,
                        self.config['distance_simulation']['max_scale'] - 1.0
                    ),
                    p=0.5
                )
            )
        
        # Add motion blur
        if self.config['motion_blur']['enabled']:
            transforms.append(
                A.MotionBlur(
                    blur_limit=self.config['motion_blur']['kernel_size'],
                    p=0.5
                )
            )
        
        # Add occlusion (random rectangles)
        if self.config['occlusion']['enabled']:
            transforms.append(
                A.CoarseDropout(
                    max_holes=3,
                    max_height=int(self.config['occlusion']['max_area_percentage'] * 100),
                    max_width=int(self.config['occlusion']['max_area_percentage'] * 100),
                    min_height=10,
                    min_width=10,
                    p=0.5
                )
            )
        
        # Color augmentations (common for object detection)
        transforms.extend([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5)
        ])
        
        # Standard YOLOv8 augmentations
        transforms.extend([
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255.0, p=1.0)
        ])
        
        # Create the final pipeline
        transform = A.Compose(
            transforms,
            bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
        )
        
        self.logger.info(f"Created augmentation pipeline with {len(transforms)} transforms")
        return transform
    
    def apply(self, image: np.ndarray, bboxes: List[List[float]] = None, 
              class_labels: List[int] = None) -> Tuple[np.ndarray, List[List[float]]]:
        """
        Apply augmentations to an image and its bounding boxes.
        
        Args:
            image: Input image as numpy array (H, W, C)
            bboxes: List of bounding boxes in YOLO format [x_center, y_center, width, height]
            class_labels: List of class labels corresponding to bounding boxes
            
        Returns:
            Tuple of (augmented image, augmented bboxes)
        """
        if bboxes is None:
            bboxes = []
        if class_labels is None:
            class_labels = [0] * len(bboxes)
        
        # Apply the transformations
        transformed = self.transform(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels
        )
        
        augmented_image = transformed['image']
        augmented_bboxes = transformed['bboxes']
        
        return augmented_image, augmented_bboxes
    
    def apply_single_augmentation(self, image: np.ndarray, augmentation_type: str, 
                                 strength: float = 1.0) -> np.ndarray:
        """
        Apply a single augmentation with adjustable strength.
        
        Args:
            image: Input image as numpy array (H, W, C)
            augmentation_type: Type of augmentation ('noise', 'distortion', 'motion_blur', etc.)
            strength: Strength factor for the augmentation (0.0-1.0)
            
        Returns:
            Augmented image
        """
        if augmentation_type == 'noise':
            # Apply Gaussian noise
            gaussian_std = self.config['noise']['gaussian_std'] * strength
            noise = np.random.normal(0, gaussian_std * 255, image.shape).astype(np.uint8)
            noisy_image = cv2.add(image, noise)
            return noisy_image
        
        elif augmentation_type == 'distortion':
            # Apply lens distortion
            factor = self.config['distortion']['barrel_factor'] * strength
            distortion = LensDistortion(barrel_factor=factor, pincushion_factor=factor/2)
            return distortion.apply(image)
        
        elif augmentation_type == 'motion_blur':
            # Apply motion blur
            kernel_size = int(self.config['motion_blur']['kernel_size'] * strength)
            kernel_size = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
            angle = random.uniform(*self.config['motion_blur']['angle_range'])
            motion_blur = A.MotionBlur(blur_limit=kernel_size)
            return motion_blur.apply(image)
        
        elif augmentation_type == 'occlusion':
            # Apply occlusion
            max_area = self.config['occlusion']['max_area_percentage'] * strength
            occlusion = A.CoarseDropout(
                max_holes=3,
                max_height=int(max_area * image.shape[0]),
                max_width=int(max_area * image.shape[1]),
                min_height=10,
                min_width=10,
                p=1.0
            )
            return occlusion.apply(image)
        
        else:
            self.logger.warning(f"Unknown augmentation type: {augmentation_type}")
            return image
    
    def visualize_augmentations(self, image_path: str, output_dir: str) -> None:
        """
        Visualize the effect of each augmentation on a sample image.
        
        Args:
            image_path: Path to sample image
            output_dir: Directory to save visualizations
        """
        import os
        import matplotlib.pyplot as plt
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load sample image
        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {e}")
            return
        
        # Original image
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, "original.jpg"))
        
        # Apply each augmentation separately
        augmentations = [
            ('noise', 'Gaussian & Salt-Pepper Noise'),
            ('distortion', 'Lens Distortion'),
            ('motion_blur', 'Motion Blur'),
            ('occlusion', 'Random Occlusion')
        ]
        
        for aug_type, aug_name in augmentations:
            plt.figure(figsize=(10, 10))
            augmented = self.apply_single_augmentation(image, aug_type, 1.0)
            plt.imshow(augmented)
            plt.title(aug_name)
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, f"{aug_type}.jpg"))
        
        # Apply combined augmentations
        plt.figure(figsize=(10, 10))
        augmented, _ = self.apply(image)
        plt.imshow(augmented)
        plt.title("Combined Augmentations")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, "combined.jpg"))
        
        self.logger.info(f"Saved augmentation visualizations to {output_dir}")


class LensDistortion(ImageOnlyTransform):
    """
    Custom augmentation for lens distortion effects (barrel and pincushion).
    """
    
    def __init__(
        self, 
        barrel_factor: float = 0.2,
        pincushion_factor: float = 0.1,
        always_apply: bool = False, 
        p: float = 0.5
    ):
        """
        Initialize the lens distortion transform.
        
        Args:
            barrel_factor: Strength of barrel distortion
            pincushion_factor: Strength of pincushion distortion
            always_apply: Whether to always apply the transform
            p: Probability of applying the transform
        """
        super().__init__(always_apply, p)
        self.barrel_factor = barrel_factor
        self.pincushion_factor = pincushion_factor
    
    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Apply the lens distortion to an image.
        
        Args:
            image: Input image
            
        Returns:
            Distorted image
        """
        # Choose between barrel (negative) and pincushion (positive) distortion
        if random.random() < 0.5:
            factor = -self.barrel_factor
        else:
            factor = self.pincushion_factor
        
        height, width = image.shape[:2]
        
        # Create distortion mesh
        distorted_image = np.zeros_like(image)
        
        # Get the center of the image
        center_x = width / 2
        center_y = height / 2
        
        # Apply distortion to each pixel
        for y in range(height):
            for x in range(width):
                # Normalize coordinates to [-1, 1]
                norm_x = (x - center_x) / center_x
                norm_y = (y - center_y) / center_y
                
                # Calculate radius from center
                r = np.sqrt(norm_x**2 + norm_y**2)
                
                # Apply distortion formula
                if r == 0:
                    distorted_r = 0
                else:
                    distorted_r = r * (1 + factor * r**2)
                
                # Convert back to pixel coordinates
                if r == 0:
                    src_x = x
                    src_y = y
                else:
                    src_x = int(center_x + norm_x / r * distorted_r * center_x)
                    src_y = int(center_y + norm_y / r * distorted_r * center_y)
                
                # Check if source coordinates are within image bounds
                if 0 <= src_x < width and 0 <= src_y < height:
                    distorted_image[y, x] = image[src_y, src_x]
        
        # For better performance, we can use cv2.remap instead of the loop above
        # This implementation is more readable but slower
        
        return distorted_image
    
    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """
        Get the names of the transform initialization arguments.
        
        Returns:
            Tuple of argument names
        """
        return ("barrel_factor", "pincushion_factor")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with a sample image
    import os
    import matplotlib.pyplot as plt
    
    # Create augmentation manager
    augmenter = AugmentationManager()
    
    # Create a simple test image if no image is provided
    test_image = np.ones((640, 640, 3), dtype=np.uint8) * 128
    
    # Draw some shapes for visualization
    cv2.rectangle(test_image, (100, 100), (300, 300), (255, 0, 0), 5)
    cv2.circle(test_image, (400, 400), 100, (0, 255, 0), 5)
    cv2.line(test_image, (0, 0), (640, 640), (0, 0, 255), 5)
    
    # Create test bounding boxes [x_center, y_center, width, height] (normalized)
    bboxes = [
        [0.3125, 0.3125, 0.3125, 0.3125],  # Rectangle
        [0.625, 0.625, 0.3125, 0.3125]     # Circle
    ]
    
    class_labels = [0, 1]  # Blue, Big Orange
    
    # Apply augmentations
    augmented_image, augmented_bboxes = augmenter.apply(test_image, bboxes, class_labels)
    
    # Visualize result
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(test_image)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(augmented_image)
    plt.title("Augmented Image")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("augmentation_example.jpg")
    plt.close()
    
    print("Saved augmentation example to augmentation_example.jpg")
