import pytest
import torch
import numpy as np
from src.preprocessing.augmentations import AugmentationManager

@pytest.mark.usefixtures("test_image")
def test_augmentation(test_image):
    """Test applying augmentations to a single image."""
    # Create augmentation manager
    augmentation_manager = AugmentationManager()
    
    # Apply augmentations
    result = augmentation_manager.apply_augmentations(test_image)
    
    # Verify results
    assert result.shape[:2] == test_image.shape[:2]


@pytest.mark.usefixtures("test_images")
def test_batch_augmentation(test_images):
    """Test applying augmentations to a batch of images."""
    # Create augmentation manager
    augmentation_manager = AugmentationManager()
    
    # Apply batch augmentations to padded images
    augmented = augmentation_manager.apply_batch_augmentations(test_images["padded"])
    
    # Convert to tensor and stack - this should work now that images are consistent size
    batch = torch.stack([torch.from_numpy(img).permute(2, 0, 1) for img in augmented])
    
    # Verify results
    assert batch.shape[0] == len(test_images["padded"])


def test_custom_size_image(pad_image_to_size):
    """Test padding images to a consistent size."""
    # Create a custom size image
    custom_img = np.zeros((359, 359, 3), dtype=np.uint8)
    
    # Pad it to the standard size
    padded = pad_image_to_size(custom_img, target_size=(640, 640))
    
    # Verify the padded dimensions
    assert padded.shape[:2] == (640, 640)
