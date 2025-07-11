"""
Comprehensive integration tests for TrainingAutomation workflows.

This module tests end-to-end workflows including:
- Complete training pipeline
- Hyperparameter optimisation workflow  
- Threshold tuning workflow
- Label conversion workflow
- Dataset preparation workflow
"""

import os
import sys
import unittest
import tempfile
import shutil
import yaml
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add the parent directory to the path so we can import the modules under test
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    prepare_environment, train_model, optimise_hyperparameters,
    validate_model, export_model, convert_labels, tune_thresholds
)
from src.training.yolo_trainer import YoloTrainer
from src.hyperparameter_optimisation.optimiser import HyperparameterOptimiser
from src.inference.threshold_tuning import ThresholdTuner
from src.label_conversion.convert import convert_fsoco_dataset
from src.preprocessing.dataset import DatasetManager
from src.utils.file_io import FileIO


class TestCompleteWorkflows(unittest.TestCase):
    """Test complete end-to-end workflows."""
    
    def setUp(self):
        """Set up test fixtures before each test."""
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, "config.yaml")
        self.data_yaml_path = os.path.join(self.test_dir, "data.yaml")
        
        # Create test configuration
        test_config = {
            'model': {
                'version': 'yolov8n',
                'input_size': 640,
                'pretrained': False
            },
            'training': {
                'epochs': 5,
                'batch_size': 4,
                'patience': 3,
                'min_delta': 0.001
            },
            'hardware': {
                'device': 'cpu',
                'workers': 1
            },
            'paths': {
                'output_models': os.path.join(self.test_dir, 'models'),
                'output_results': os.path.join(self.test_dir, 'results')
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        # Create test data configuration
        test_data = {
            'path': self.test_dir,
            'train': 'train/images',
            'val': 'val/images',
            'names': {
                '0': 'blue_cone',
                '1': 'yellow_cone',
                '2': 'orange_cone'
            },
            'nc': 3
        }
        
        with open(self.data_yaml_path, 'w') as f:
            yaml.dump(test_data, f)
        
        # Create mock directories and files
        os.makedirs(os.path.join(self.test_dir, 'train', 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, 'train', 'labels'), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, 'val', 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, 'val', 'labels'), exist_ok=True)
        
        # Create dummy image and label files
        for i in range(3):
            # Create dummy images
            train_img = os.path.join(self.test_dir, 'train', 'images', f'img_{i}.jpg')
            val_img = os.path.join(self.test_dir, 'val', 'images', f'img_{i}.jpg')
            with open(train_img, 'wb') as f:
                f.write(b'fake_image_data')
            with open(val_img, 'wb') as f:
                f.write(b'fake_image_data')
            
            # Create dummy labels
            train_label = os.path.join(self.test_dir, 'train', 'labels', f'img_{i}.txt')
            val_label = os.path.join(self.test_dir, 'val', 'labels', f'img_{i}.txt')
            with open(train_label, 'w') as f:
                f.write('0 0.5 0.5 0.2 0.2\n')
            with open(val_label, 'w') as f:
                f.write('1 0.3 0.3 0.1 0.1\n')
    
    def tearDown(self):
        """Clean up test fixtures after each test."""
        shutil.rmtree(self.test_dir)
    
    @patch('main.YoloTrainer')
    @patch('main.HardwareManager')
    def test_complete_training_workflow(self, mock_hardware_manager, mock_yolo_trainer):
        """Test the complete training workflow from main.py."""
        # Set up mocks
        mock_hw_manager = Mock()
        mock_hardware_manager.return_value = mock_hw_manager
        
        mock_trainer = Mock()
        mock_yolo_trainer.return_value = mock_trainer
        
        # Mock training results
        mock_trainer.train.return_value = {
            'success': True,
            'best_epoch': 3,
            'best_fitness': 0.85,
            'model_path': os.path.join(self.test_dir, 'best.pt')
        }
        
        # Create mock arguments
        from argparse import Namespace
        args = Namespace(
            config=self.config_path,
            data=self.data_yaml_path,
            resume=False,
            verbose=True
        )
        
        # Prepare environment
        env = prepare_environment(args)
        
        # Test training
        train_model(args, env)
        
        # Verify trainer was called correctly
        mock_yolo_trainer.assert_called_once()
        mock_trainer.train.assert_called_once()
    
    @patch('src.hyperparameter_optimisation.optimiser.optuna')
    def test_hyperparameter_optimisation_workflow(self, mock_optuna):
        """Test the hyperparameter optimisation workflow."""
        # Set up mock study
        mock_study = Mock()
        mock_trial = Mock()
        
        mock_optuna.create_study.return_value = mock_study
        mock_study.best_trial = mock_trial
        mock_trial.value = 0.85
        mock_trial.params = {
            'learning_rate': 0.001,
            'batch_size': 16,
            'epochs': 50
        }
        
        # Create optimiser
        optimiser = HyperparameterOptimiser(
            config_path=self.config_path,
            data_yaml_path=self.data_yaml_path,
            output_dir=os.path.join(self.test_dir, 'optimisation'),
            n_trials=5
        )
        
        # Test optimisation
        with patch.object(optimiser, '_objective') as mock_objective:
            mock_objective.return_value = 0.8
            result = optimiser.optimize()
        
        # Verify results
        self.assertIsInstance(result, dict)
        mock_study.optimize.assert_called_once()
    
    def test_threshold_tuning_workflow(self):
        """Test the threshold tuning workflow."""
        # Create mock model path
        model_path = os.path.join(self.test_dir, 'model.pt')
        with open(model_path, 'wb') as f:
            f.write(b'fake_model_data')
        
        # Mock YOLO model
        with patch('src.inference.threshold_tuning.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model
            
            # Mock validation results
            mock_metrics = Mock()
            mock_metrics.results_dict = {
                'metrics/precision(B)': 0.8,
                'metrics/recall(B)': 0.75,
                'metrics/mAP50(B)': 0.82,
                'metrics/mAP50-95(B)': 0.65
            }
            mock_model.val.return_value = mock_metrics
            
            # Create tuner
            tuner = ThresholdTuner(
                model_path=model_path,
                data_yaml_path=self.data_yaml_path,
                output_dir=os.path.join(self.test_dir, 'threshold_tuning')
            )
            
            # Test confidence threshold tuning
            result = tuner.tune_confidence_threshold(
                conf_range=(0.1, 0.5),
                conf_steps=3,
                target_metric='recall'
            )
            
            # Verify results
            self.assertIn('optimal_confidence', result)
            self.assertIn('optimal_metrics', result)
            self.assertIn('all_results', result)
    
    def test_label_conversion_workflow(self):
        """Test the label conversion workflow."""
        # Create source directory structure
        source_dir = os.path.join(self.test_dir, 'source')
        target_train_images = os.path.join(self.test_dir, 'target', 'train', 'images')
        target_train_labels = os.path.join(self.test_dir, 'target', 'train', 'labels')
        target_val_images = os.path.join(self.test_dir, 'target', 'val', 'images')
        target_val_labels = os.path.join(self.test_dir, 'target', 'val', 'labels')
        
        os.makedirs(source_dir, exist_ok=True)
        os.makedirs(target_train_images, exist_ok=True)
        os.makedirs(target_train_labels, exist_ok=True)
        os.makedirs(target_val_images, exist_ok=True)
        os.makedirs(target_val_labels, exist_ok=True)
        
        # Mock converters
        with patch('src.label_conversion.convert._get_converter') as mock_get_converter:
            mock_source_converter = Mock()
            mock_target_converter = Mock()
            
            # Return appropriate converter based on format
            def converter_side_effect(format_name):
                if format_name == 'supervisely':
                    return mock_source_converter
                elif format_name == 'darknet':
                    return mock_target_converter
                return None
            
            mock_get_converter.side_effect = converter_side_effect
            
            # Mock processing results
            mock_source_converter.process_fsoco_dataset.return_value = {
                'total_images': 10,
                'train_images': 8,
                'val_images': 2,
                'total_labels': 15
            }
            
            # Test conversion
            result = convert_fsoco_dataset(
                source_format='supervisely',
                target_format='darknet',
                input_dir=source_dir,
                train_images_dir=target_train_images,
                train_labels_dir=target_train_labels,
                val_images_dir=target_val_images,
                val_labels_dir=target_val_labels,
                classes='blue_cone,yellow_cone,orange_cone',
                split_ratio=0.2
            )
            
            # Verify results
            self.assertIsInstance(result, dict)
            mock_source_converter.process_fsoco_dataset.assert_called_once()
    
    def test_dataset_manager_workflow(self):
        """Test the dataset manager workflow."""
        # Create dataset manager
        manager = DatasetManager(
            data_yaml_path=self.data_yaml_path,
            batch_size=2,
            workers=0
        )
        
        # Test dataset creation
        train_dataset, val_dataset = manager.create_datasets()
        
        # Verify datasets
        self.assertIsNotNone(train_dataset)
        self.assertIsNotNone(val_dataset)
        
        # Test data loader creation
        train_loader, val_loader = manager.create_data_loaders(train_dataset, val_dataset)
        
        # Verify loaders
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
    
    def test_file_io_workflow(self):
        """Test the file I/O utilities workflow."""
        # Create FileIO instance
        file_io = FileIO(base_dir=self.test_dir)
        
        # Test directory creation
        test_dir = file_io.create_directory('test_subdir')
        self.assertTrue(test_dir.exists())
        
        # Test YAML loading
        config = file_io.load_yaml(self.config_path)
        self.assertIsInstance(config, dict)
        self.assertIn('model', config)
        
        # Test file listing
        files = file_io.list_files('train/images', pattern="*.jpg")
        self.assertGreater(len(files), 0)
    
    @patch('main.logging')
    def test_error_handling_workflow(self, mock_logging):
        """Test error handling in workflows."""
        # Test with invalid configuration
        invalid_config = os.path.join(self.test_dir, 'invalid.yaml')
        with open(invalid_config, 'w') as f:
            f.write('invalid: yaml: content:')
        
        from argparse import Namespace
        args = Namespace(
            config=invalid_config,
            data=self.data_yaml_path,
            resume=False,
            verbose=True
        )
        
        # Test that errors are handled gracefully
        with self.assertRaises(SystemExit):
            prepare_environment(args)
    
    def test_integration_with_real_files(self):
        """Test integration with actual project files."""
        # Use the actual config files from the project
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        real_config = os.path.join(project_root, 'config.yaml')
        real_data = os.path.join(project_root, 'data', 'data.yaml')
        
        if os.path.exists(real_config):
            file_io = FileIO()
            config = file_io.load_yaml(real_config)
            self.assertIsInstance(config, dict)
        
        if os.path.exists(real_data):
            file_io = FileIO()
            data_config = file_io.load_yaml(real_data)
            self.assertIsInstance(data_config, dict)


class TestPerformanceAndScalability(unittest.TestCase):
    """Test performance characteristics and scalability."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_large_dataset_handling(self):
        """Test handling of large datasets."""
        # Create a larger mock dataset
        images_dir = os.path.join(self.test_dir, 'images')
        labels_dir = os.path.join(self.test_dir, 'labels')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        # Create 100 dummy files
        for i in range(100):
            img_path = os.path.join(images_dir, f'img_{i:03d}.jpg')
            label_path = os.path.join(labels_dir, f'img_{i:03d}.txt')
            
            with open(img_path, 'wb') as f:
                f.write(b'dummy_image_data' * 100)  # Larger dummy data
            
            with open(label_path, 'w') as f:
                f.write('0 0.5 0.5 0.2 0.2\n1 0.3 0.7 0.1 0.1\n')
        
        # Test file listing performance
        file_io = FileIO(base_dir=self.test_dir)
        import time
        
        start_time = time.time()
        files = file_io.list_files('images', pattern="*.jpg", recursive=True)
        end_time = time.time()
        
        # Should find all files quickly
        self.assertEqual(len(files), 100)
        self.assertLess(end_time - start_time, 1.0)  # Should complete in under 1 second


if __name__ == '__main__':
    unittest.main()
