import os
import sys
import unittest
import tempfile
import shutil
import yaml
import optuna
import json
from unittest.mock import MagicMock, patch

# Add the parent directory to the path so we can import the modules under test
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hyperparameter_optimisation.optimiser import HyperparameterOptimiser, TrialPruningCallback


class TestHyperparameterOptimiser(unittest.TestCase):
    """Test the hyperparameter optimisation functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test."""
        self.test_dir = tempfile.mkdtemp()
        
        # Track resources to clean up
        self._resources_to_cleanup = []
        
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
                'img_size': 320  # Small size for faster tests
            },
            'hyperparameter_optimisation': {
                'epochs_per_trial': 1,
                'direction': 'maximize',
                'param_spaces': {
                    'learning_rate': {
                        'type': 'float',
                        'low': 1e-4,
                        'high': 1e-2,
                        'log': True
                    },
                    'img_size': {
                        'type': 'categorical',
                        'choices': [320, 416]
                    }
                },
                'pruner': {
                    'type': 'median',
                    'n_startup_trials': 2
                },
                'sampler': {
                    'type': 'tpe',
                    'seed': 42
                }
            },
            'paths': {
                'output': {
                    'optimisation': os.path.join(self.test_dir, "optimisation")
                }
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
        
        # Set up output directory
        self.output_dir = os.path.join(self.test_dir, "optimisation")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures after each test."""
        # Clean up all tracked resources
        for resource in self._resources_to_cleanup:
            if hasattr(resource, 'close'):
                try:
                    resource.close()
                except:
                    pass
        
        # Try with a delay if needed
        try:
            shutil.rmtree(self.test_dir)
        except PermissionError:
            import time
            time.sleep(0.5)  # Wait a bit for resources to be released
            try:
                shutil.rmtree(self.test_dir)
            except:
                print(f"Warning: Could not remove {self.test_dir}")
    
    @patch('optuna.create_study')
    def test_initialisation(self, mock_create_study):
        """Test initialisation of HyperparameterOptimiser."""
        # Mock study creation
        mock_study = MagicMock()
        mock_create_study.return_value = mock_study
        
        # Test initialising with minimal arguments
        opt = HyperparameterOptimiser(
            config_path=self.config_path,
            data_yaml_path=self.data_yaml_path,
            n_trials=2,
            verbose=False
        )
        
        # Check that study was created
        mock_create_study.assert_called_once()
        
        # Check that param spaces were loaded
        self.assertIn('learning_rate', opt.param_spaces)
        self.assertIn('img_size', opt.param_spaces)
    
    @patch('optuna.create_study')
    def test_param_sampling(self, mock_create_study):
        """Test hyperparameter sampling."""
        # Mock study creation
        mock_study = MagicMock()
        mock_create_study.return_value = mock_study
        
        # Create optimiser
        opt = HyperparameterOptimiser(
            config_path=self.config_path,
            data_yaml_path=self.data_yaml_path,
            n_trials=2,
            verbose=False
        )
        
        # Mock trial
        trial = MagicMock()
        trial.suggest_float.return_value = 0.001
        trial.suggest_categorical.return_value = 416
        
        # Sample parameters
        params = opt._sample_params(trial)
        
        # Check that parameters were sampled correctly
        trial.suggest_float.assert_called_once()
        trial.suggest_categorical.assert_called_once()
        self.assertEqual(params['learning_rate'], 0.001)
        self.assertEqual(params['img_size'], 416)
    
    @patch('optuna.create_study')
    @patch('src.training.yolo_trainer.YoloTrainer')
    def test_objective_function(self, mock_trainer_class, mock_create_study):
        """Test objective function for optimisation."""
        # Mock study creation
        mock_study = MagicMock()
        mock_create_study.return_value = mock_study
        
        # Mock trainer
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {
            'metrics': {
                'metrics/mAP50-95(B)': 0.75
            }
        }
        mock_trainer_class.return_value = mock_trainer
        
        # Create optimiser
        opt = HyperparameterOptimiser(
            config_path=self.config_path,
            data_yaml_path=self.data_yaml_path,
            output_dir=self.output_dir,
            n_trials=2,
            verbose=False
        )
        
        # Mock trial
        trial = MagicMock()
        trial.suggest_float.return_value = 0.001
        trial.suggest_categorical.return_value = 416
        trial.number = 1
        
        # Test objective function
        result = opt.objective(trial)
        
        # Check result
        self.assertEqual(result, 0.75)
        mock_trainer.train.assert_called_once()
        
    @patch('optuna.create_study')
    def test_save_best_params(self, mock_create_study):
        """Test saving of best parameters."""
        # Mock study creation
        mock_study = MagicMock()
        mock_create_study.return_value = mock_study
        
        # Create optimiser
        opt = HyperparameterOptimiser(
            config_path=self.config_path,
            data_yaml_path=self.data_yaml_path,
            output_dir=self.output_dir,
            n_trials=2,
            verbose=False
        )
        
        # Set best params and save
        opt.best_value = 0.85
        opt.best_params = {'learning_rate': 0.001, 'img_size': 416}
        opt.best_trial_number = 5
        opt.best_model_path = os.path.join(self.test_dir, "best_model")
        
        opt._save_best_params()
        
        # Check that file was created
        best_params_path = os.path.join(self.output_dir, "best_params.json")
        self.assertTrue(os.path.exists(best_params_path))
        
        # Check content
        with open(best_params_path, 'r') as f:
            data = json.load(f)
        
        self.assertEqual(data['best_value'], 0.85)
        self.assertEqual(data['best_params']['learning_rate'], 0.001)


class TestTrialPruningCallback(unittest.TestCase):
    """Test the trial pruning callback functionality."""
    
    def test_callback_execution(self):
        """Test execution of the pruning callback."""
        # Mock trial
        trial = MagicMock()
        trial.report = MagicMock()
        trial.should_prune.return_value = False
        
        # Create callback
        callback = TrialPruningCallback(trial, metric_name='metrics/mAP50-95(B)')
        
        # Mock trainer and metrics
        mock_trainer = MagicMock()
        mock_trainer.current_epoch = 5
        metrics = {'metrics/mAP50-95(B)': 0.7}
        
        # Execute callback
        callback(mock_trainer, metrics)
        
        # Check that trial.report was called with correct values
        trial.report.assert_called_once_with(0.7, 5)
        trial.should_prune.assert_called_once()
    
    def test_pruning_decision(self):
        """Test pruning decision."""
        # Mock trial that should be pruned
        trial = MagicMock()
        trial.report = MagicMock()
        trial.should_prune.return_value = True
        
        # Create callback
        callback = TrialPruningCallback(trial, metric_name='metrics/mAP50-95(B)')
        
        # Mock trainer and metrics
        mock_trainer = MagicMock()
        mock_trainer.current_epoch = 5
        metrics = {'metrics/mAP50-95(B)': 0.3}  # Low performance
        
        # Execute callback should raise TrialPruned
        with self.assertRaises(optuna.exceptions.TrialPruned):
            callback(mock_trainer, metrics)


if __name__ == '__main__':
    unittest.main()
