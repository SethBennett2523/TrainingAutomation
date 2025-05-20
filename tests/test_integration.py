import os
import sys
import unittest
import tempfile
import shutil
import yaml
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the parent directory to the path so we can import the modules under test
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.label_conversion.darknet_txt_converter import DarknetTxtConverter
from src.preprocessing.dataset import DatasetManager
from src.training.yolo_trainer import YoloTrainer


class TestEndToEndWorkflow(unittest.TestCase):
    """Test the full training workflow from data preparation to training."""
    
    def setUp(self):
        """Set up test fixtures before each test."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create minimal test dataset and config files
        self._create_test_dataset()
        self._create_config_files()
    
    def tearDown(self):
        """Clean up test fixtures after each test."""
        # Ensure proper resource cleanup
        torch.cuda.empty_cache()  # Release GPU memory if used
        # Use a try-except to avoid issues if directory is already being deleted
        try:
            shutil.rmtree(self.test_dir)
        except PermissionError:
            print(f"Warning: Could not remove {self.test_dir} due to permission error.")
    
    def _create_test_dataset(self):
        """Create a minimal test dataset for integration testing."""
        # Create directory structure
        self.data_dir = Path(self.test_dir) / "data"
        self.train_dir = self.data_dir / "train"
        self.train_images = self.train_dir / "images"
        self.train_labels = self.train_dir / "labels"
        
        os.makedirs(self.train_images, exist_ok=True)
        os.makedirs(self.train_labels, exist_ok=True)
        
        # This is a minimal test - in a real integration test you would create
        # actual images and label files, but that's beyond the scope here
    
    def _create_config_files(self):
        """Create necessary configuration files."""
        self.config_path = Path(self.test_dir) / "config.yaml"
        self.data_yaml_path = Path(self.test_dir) / "data.yaml"
        
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
            },
            'paths': {
                'output': {
                    'models': str(Path(self.test_dir) / "output")
                }
            }
        }
        
        data_config = {
            'train': str(self.train_images),
            'val': '',  # No validation set for this test
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
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)
        
        with open(self.data_yaml_path, 'w') as f:
            yaml.dump(data_config, f)
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.get_device_name')
    @patch('torch.version.hip', None)
    @patch('ultralytics.YOLO')
    def test_dataset_to_trainer_integration(self, mock_yolo, mock_get_device_name, 
                                           mock_device_count, mock_cuda_available):
        """Test integration between dataset and trainer components."""
        # Mock NVIDIA environment
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 1
        mock_get_device_name.return_value = 'NVIDIA RTX 2060 Super'
        
        # 1. Configure DatasetManager
        dataset_manager = DatasetManager(
            data_yaml_path=self.data_yaml_path,
            batch_size=2,
            workers=0
        )
        
        # 2. Initialize trainer with mock model
        mock_model = mock_yolo.return_value
        mock_model.train.return_value = {'metrics': {'metrics/mAP50-95(B)': 0.5}}
        
        trainer = YoloTrainer(
            config_path=self.config_path,
            data_yaml_path=self.data_yaml_path,
            output_dir=os.path.join(self.test_dir, "output")
        )
        
        # 3. Test that trainer can initialize model (using implementation's method name)
        trainer.initialize_model()  # Note: Using the actual method name from implementation
        self.assertIsNotNone(trainer.model)
        
        # 4. Test a simulated training run
        with patch.object(trainer, '_setup_data_yaml'):
            results = trainer.train()
            self.assertIn('metrics', results)


if __name__ == '__main__':
    unittest.main()
