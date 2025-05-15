"""
pytest configuration file with shared fixtures and logging setup.
"""
import os
import sys
import pytest
import logging
import tempfile
import shutil
import yaml
import numpy as np
import torch
import cv2
from datetime import datetime
from pathlib import Path

# Add the parent directory to the path so we can import the modules under test
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def test_config():
    """
    Create a test configuration directory and files.
    
    Returns:
        dict: Paths to created config files and directories
    """
    test_dir = tempfile.mkdtemp()
    
    # Create config files
    config_path = os.path.join(test_dir, "config.yaml")
    data_yaml_path = os.path.join(test_dir, "data.yaml")
    
    # Create a minimal config for testing
    config = {
        'hardware': {
            'device': 'cpu',  # Use CPU for tests
            'batch_size': 2,
            'workers': 0
        },
        'training': {
            'epochs': 1,
            'img_size': 320  # Small size for faster tests
        }
    }
    
    data_config = {
        'train': os.path.join(test_dir, "train"),
        'val': os.path.join(test_dir, "val"),
        'nc': 5,
        'names': {
            '0': 'blue',
            '1': 'big_orange',
            '2': 'orange',
            '3': 'unknown',
            '4': 'yellow'
        }
    }
    
    # Create the config files
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_config, f)
    
    # Create minimal dataset structure
    os.makedirs(os.path.join(test_dir, "train", "images"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "train", "labels"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "val", "images"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "val", "labels"), exist_ok=True)
    
    yield {
        "test_dir": test_dir,
        "config_path": config_path,
        "data_yaml_path": data_yaml_path,
        "train_dir": os.path.join(test_dir, "train"),
        "val_dir": os.path.join(test_dir, "val")
    }
    
    # Cleanup after test
    try:
        shutil.rmtree(test_dir)
    except (PermissionError, OSError):
        print(f"Warning: Could not remove {test_dir}")


def pytest_configure(config):
    """
    Configure pytest - set up logging before tests run.
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create test logs directory
    test_logs_dir = log_dir / "tests"
    test_logs_dir.mkdir(exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = test_logs_dir / f"test_run_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Log test run start
    logging.info(f"Test run started at {timestamp}")
    logging.info(f"Log file: {log_file}")


@pytest.fixture
def pad_image_to_size():
    """
    Fixture that provides the pad_image_to_size function.
    
    Returns:
        function: Function to pad images to a specified size
    """
    def _pad_image_to_size(image, target_size=(640, 640)):
        """
        Pad an image to the target size without changing aspect ratio.
        
        Works with both NumPy arrays and PyTorch tensors.
        
        Args:
            image: Image data as NumPy array or PyTorch tensor
            target_size: Tuple of (width, height)
            
        Returns:
            Padded image in same format as input
        """
        is_tensor = isinstance(image, torch.Tensor)
        
        # Convert torch tensor to numpy if needed
        if is_tensor:
            # If image is [C,H,W] format, convert to [H,W,C]
            if image.shape[0] == 3 and image.dim() == 3:
                image_np = image.permute(1, 2, 0).cpu().numpy()
            else:
                image_np = image.cpu().numpy()
        else:
            image_np = image.copy()
        
        # Get input image dimensions
        if len(image_np.shape) == 3:
            h, w, c = image_np.shape
        else:
            h, w = image_np.shape
            c = 1
            image_np = image_np.reshape((h, w, c))
        
        # Create empty target image
        if c == 3:
            padded = np.zeros((target_size[1], target_size[0], c), dtype=image_np.dtype)
        else:
            padded = np.zeros((target_size[1], target_size[0]), dtype=image_np.dtype)
        
        # Calculate padding to center the image
        h_offset = (target_size[1] - h) // 2
        w_offset = (target_size[0] - w) // 2
        
        # Ensure non-negative offsets (in case image is larger than target)
        h_offset = max(0, h_offset)
        w_offset = max(0, w_offset)
        
        # Calculate how much of the source image to copy
        h_copy = min(h, target_size[1])
        w_copy = min(w, target_size[0])
        
        # Copy the image to the padded array
        if c == 1 and len(padded.shape) == 2:
            padded[h_offset:h_offset+h_copy, w_offset:w_offset+w_copy] = image_np[:h_copy, :w_copy, 0]
        else:
            padded[h_offset:h_offset+h_copy, w_offset:w_offset+w_copy] = image_np[:h_copy, :w_copy]
        
        # Convert back to torch tensor if input was tensor
        if is_tensor:
            # If original was [C,H,W], reshape back to that format
            if image.shape[0] == 3 and image.dim() == 3:
                padded = torch.from_numpy(padded).permute(2, 0, 1)
            else:
                padded = torch.from_numpy(padded)
            
            # Move to same device as input
            padded = padded.to(image.device)
        
        return padded
    
    return _pad_image_to_size


@pytest.fixture
def test_image():
    """
    Create a test image for augmentation and dataset tests.
    
    Returns:
        numpy.ndarray: RGB image of size 640x480
    """
    # Create a blank RGB image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some shapes for testing
    # Red circle
    cv2.circle(img, (320, 240), 100, (0, 0, 255), -1)
    # Blue rectangle
    cv2.rectangle(img, (100, 100), (200, 200), (255, 0, 0), -1)
    # Green triangle
    pts = np.array([[500, 100], [600, 100], [550, 200]], np.int32)
    cv2.fillPoly(img, [pts], (0, 255, 0))
    
    return img


@pytest.fixture
def test_images(pad_image_to_size):
    """
    Create a list of test images with different sizes.
    
    Returns:
        list: List of numpy.ndarray images
    """
    # Create several images with different sizes
    img1 = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(img1, (320, 240), 100, (0, 0, 255), -1)
    
    img2 = np.zeros((380, 520, 3), dtype=np.uint8)
    cv2.rectangle(img2, (100, 100), (200, 200), (255, 0, 0), -1)
    
    img3 = np.zeros((520, 720, 3), dtype=np.uint8)
    pts = np.array([[500, 100], [600, 100], [550, 200]], np.int32)
    cv2.fillPoly(img3, [pts], (0, 255, 0))
    
    # Return both original images and padded versions
    return {
        "original": [img1, img2, img3],
        "padded": [pad_image_to_size(img) for img in [img1, img2, img3]]
    }
    

