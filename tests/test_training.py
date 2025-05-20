import os
import sys
import unittest
import tempfile
import shutil
import yaml
import torch
from unittest.mock import MagicMock, patch
import numpy as np

# Add the parent directory to the path so we can import the modules under test
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.hardware_manager import HardwareManager
from src.training.yolo_trainer import YoloTrainer
from src.training.early_stopping import EarlyStopping


class TestHardwareManager(unittest.TestCase):
    """Test the hardware detection and optimisation functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create a test config file
        self.config_path = os.path.join(self.test_dir, "config.yaml")
        config = {
            'hardware': {
                'device': 'auto',
                'batch_size': 16,
                'workers': 4,
                'memory_fraction': 0.8
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)
    
    def tearDown(self):
        """Clean up test fixtures after each test."""
        shutil.rmtree(self.test_dir)
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.get_device_name')
    @patch('torch.version.hip', None)  # Block ROCm detection
    def test_device_detection_cuda(self, mock_get_device_name, mock_device_count, mock_cuda_available):
        """Test CUDA device detection for NVIDIA GPU."""
        # Mock NVIDIA environment
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 1
        mock_get_device_name.return_value = 'NVIDIA RTX 2060 Super'
        
        hw_manager = HardwareManager(self.config_path)
        self.assertEqual(hw_manager.device, 'cuda')
        self.assertEqual(hw_manager.device_name, 'NVIDIA RTX 2060 Super')
    
    @patch('torch.cuda.is_available')
    @patch('torch.version.hip', None)  # Block ROCm detection
    def test_device_detection_cpu(self, mock_cuda_available):
        """Test CPU fallback when no GPU is available."""
        mock_cuda_available.return_value = False
        
        hw_manager = HardwareManager(self.config_path)
        self.assertEqual(hw_manager.device, 'cpu')
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.get_device_properties')
    @patch('torch.version.hip', None)  # Block ROCm detection
    def test_vram_detection(self, mock_get_device_props, mock_device_count, mock_cuda_available):
        """Test VRAM detection for NVIDIA GPUs."""
        # Setup NVIDIA environment
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 1
        
        # Mock device properties with 8GB VRAM (RTX 2060 Super has 8GB)
        device_props = MagicMock()
        device_props.total_memory = 8 * 1024 * 1024 * 1024  # 8GB in bytes
        mock_get_device_props.return_value = device_props
        
        hw_manager = HardwareManager(self.config_path)
        # Should be approximately 8GB (might be slightly less due to reserved memory)
        self.assertGreaterEqual(hw_manager.vram_total, 7.5 * 1024 * 1024 * 1024)
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.get_device_name')
    @patch('torch.version.hip', None)  # Block ROCm detection
    def test_optimal_batch_size_calculation(self, mock_get_device_name, mock_device_count, mock_cuda_available):
        """Test batch size calculation based on NVIDIA hardware."""
        # Setup NVIDIA environment
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 1
        mock_get_device_name.return_value = 'NVIDIA RTX 2060 Super'
        
        with patch.object(HardwareManager, '__init__', return_value=None):
            hw_manager = HardwareManager()
            hw_manager.device = 'cuda'
            hw_manager.vram_total = 8 * 1024 * 1024 * 1024  # 8GB VRAM
            hw_manager.config = {'hardware': {'memory_fraction': 0.8}}
            
            # Test batch size calculation for YOLOv8
            batch_size = hw_manager.calculate_optimal_batch_size(image_size=640)
            self.assertIsInstance(batch_size, int)
            self.assertGreater(batch_size, 0)
    
    def test_optimal_worker_calculation(self):
        """Test worker count calculation."""
        with patch.object(HardwareManager, '__init__', return_value=None):
            hw_manager = HardwareManager()
            hw_manager.config = {'hardware': {'workers': 'auto'}}
            
            with patch('os.cpu_count', return_value=8):
                workers = hw_manager.calculate_optimal_workers()
                self.assertIsInstance(workers, int)
                self.assertGreaterEqual(workers, 1)
                self.assertLessEqual(workers, 8)  # Should not exceed CPU count


class TestEarlyStopping(unittest.TestCase):
    """Test the early stopping functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test."""
        self.patience = 5
        self.min_delta = 0.001
        self.early_stopping = EarlyStopping(patience=self.patience, min_delta=self.min_delta)
    
    def test_initialization(self):
        """Test initialization of EarlyStopping."""
        self.assertEqual(self.early_stopping.patience, self.patience)
        self.assertEqual(self.early_stopping.min_delta, self.min_delta)
        self.assertEqual(self.early_stopping.counter, 0)
        self.assertEqual(self.early_stopping.best_score, None)
        self.assertFalse(self.early_stopping.early_stop)
    
    def test_improvement_detection(self):
        """Test detection of model improvement."""
        # First call with a score should set best_score
        # Use the actual method name from implementation
        self.early_stopping.check_improvement(0.7)  # Changed from __call__ to check_improvement
        self.assertEqual(self.early_stopping.best_score, 0.7)
        self.assertEqual(self.early_stopping.counter, 0)
        
        # Improved score should reset counter
        self.early_stopping.check_improvement(0.8)  # Changed from __call__ to check_improvement
        self.assertEqual(self.early_stopping.best_score, 0.8)
        self.assertEqual(self.early_stopping.counter, 0)
    
    def test_no_improvement_increases_counter(self):
        """Test counter increment when no improvement."""
        # Set initial best score
        self.early_stopping.check_improvement(0.8)  # Changed from __call__ to check_improvement
        
        # No improvement should increase counter
        self.early_stopping.check_improvement(0.79)  # Changed from __call__ to check_improvement
        self.assertEqual(self.early_stopping.counter, 1)
        self.assertEqual(self.early_stopping.best_score, 0.8)
    
    def test_min_delta_threshold(self):
        """Test minimum delta threshold."""
        # Set initial best score
        self.early_stopping.check_improvement(0.8)  # Changed from __call__ to check_improvement
        
        # Small improvement within min_delta should still count as no improvement
        self.early_stopping.check_improvement(0.8005)  # Changed from __call__ to check_improvement
        self.assertEqual(self.early_stopping.counter, 1)
    
    def test_early_stopping_trigger(self):
        """Test early stopping is triggered after patience is exceeded."""
        # Set initial best score
        self.early_stopping.check_improvement(0.8)  # Changed from __call__ to check_improvement
        
        # No improvement for patience+1 epochs should trigger early stopping
        for _ in range(self.patience + 1):
            self.early_stopping.check_improvement(0.7)  # Changed from __call__ to check_improvement
        
        self.assertTrue(self.early_stopping.early_stop)
    
    def test_epoch_estimation(self):
        """Test optimal epoch estimation."""
        # Changed to use the actual method name and parameters
        # Mock training history
        history = {
            'metrics/mAP50-95(B)': [0.1, 0.2, 0.3, 0.4, 0.45, 0.48, 0.49, 0.495, 0.498, 0.499]
        }
        
        # Use the correct parameter name for the method
        optimal_epochs = self.early_stopping.estimate_optimal_epochs(
            history, metric='metrics/mAP50-95(B)'  # Changed from metric_name to metric
        )
        
        self.assertIsInstance(optimal_epochs, int)
        self.assertGreater(optimal_epochs, 0)
        self.assertLessEqual(optimal_epochs, len(history['metrics/mAP50-95(B)']))


class TestYoloTrainer(unittest.TestCase):
    """Test the YOLO trainer functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create test config files
        self.config_path = os.path.join(self.test_dir, "config.yaml")
        self.data_yaml_path = os.path.join(self.test_dir, "data.yaml")
        
        # Create a minimal config for testing
        config = {
            'hardware': {
                'device': 'cpu',  # Use CPU for tests
                'batch_size': 2,
                'workers': 0
            },
            'training': {
                'epochs': 1,
                'img_size': 320,  # Small size for faster tests
                'pretrained': False
            }
        }
        
        data_config = {
            'train': os.path.join(self.test_dir, "train"),
            'val': os.path.join(self.test_dir, "val"),
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
        
        # Create minimal dataset structure (empty dirs)
        os.makedirs(os.path.join(self.test_dir, "train", "images"), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, "train", "labels"), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, "val", "images"), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, "val", "labels"), exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures after each test."""
        shutil.rmtree(self.test_dir)
    
    @patch('ultralytics.YOLO')
    def test_model_initialization(self, mock_yolo):
        """Test YOLO model initialization."""
        # Mock the YOLO model
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        
        trainer = YoloTrainer(
            config_path=self.config_path,
            data_yaml_path=self.data_yaml_path,
            output_dir=os.path.join(self.test_dir, "output")
        )
        
        # Test initializing the model - use the actual method name from implementation
        trainer.initialize_model()  # Changed from initialise_model to match implementation
        mock_yolo.assert_called_once()
        self.assertIsNotNone(trainer.model)
    
    @patch('ultralytics.YOLO')
    def test_training_name_generation(self, mock_yolo):
        """Test training name generation."""
        mock_yolo.return_value = MagicMock()
        
        trainer = YoloTrainer(
            config_path=self.config_path,
            data_yaml_path=self.data_yaml_path,
            output_dir=os.path.join(self.test_dir, "output")
        )
        
        name = trainer._generate_run_name() 
        
        # Check that it follows the expected format (contains week number)
        self.assertRegex(name, r'W\d+')
    
    @patch('ultralytics.YOLO')
    def test_export_model(self, mock_yolo):
        """Test model export functionality."""
        # Mock the YOLO model and export method
        mock_model = MagicMock()
        mock_model.export.return_value = os.path.join(self.test_dir, "exported_model.onnx")
        mock_yolo.return_value = mock_model
        
        trainer = YoloTrainer(
            config_path=self.config_path,
            data_yaml_path=self.data_yaml_path,
            output_dir=os.path.join(self.test_dir, "output")
        )
        trainer.initialize_model()  # Changed from initialise_model
        
        # Test export
        export_path = trainer.export_model(format="onnx")
        mock_model.export.assert_called_once()
        self.assertIsNotNone(export_path)


if __name__ == '__main__':
    unittest.main()
