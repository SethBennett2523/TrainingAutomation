import os
import yaml
import json
import logging
import optuna
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from datetime import datetime
import joblib
from pathlib import Path
import concurrent.futures

# Import local modules
from ..training.hardware_manager import HardwareManager
from ..training.yolo_trainer import YoloTrainer


class HyperparameterOptimiser:
    """
    Hyperparameter optimiser using Optuna framework for YOLOv8 models.
    
    This class handles the search for optimal hyperparameters for YOLOv8 model training,
    using efficient search strategies, pruning of poor performers, and hardware-aware
    parallel execution.
    """
    
    def __init__(
        self,
        config_path: str,
        data_yaml_path: str,
        output_dir: str = None,
        study_name: str = None,
        n_trials: int = 25,
        timeout: int = None,
        n_jobs: int = -1,
        verbose: bool = True,
        model_type: str = 'yolov8m'
    ):
        """
        Initialise the hyperparameter optimiser.
        
        Args:
            config_path: Path to the main configuration file
            data_yaml_path: Path to the data YAML file
            output_dir: Directory to save optimisation results
            study_name: Name for the Optuna study
            n_trials: Maximum number of trials to run
            timeout: Maximum seconds for optimisation (None for no timeout)
            n_jobs: Number of parallel jobs (-1 for hardware-based auto-detection)
            verbose: Whether to log detailed information            model_type: YOLO model type to use (e.g., 'yolov8m', 'yolov11s', 'yolov11m', 'yolov11l')
        """
        self.config_path = config_path
        self.data_yaml_path = data_yaml_path
        self.n_trials = n_trials
        self.timeout = timeout
        self.verbose = verbose
        self.model_type = model_type  # Store the model type
        
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        if verbose:
            self.logger.setLevel(logging.INFO)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Setup output directory
        if output_dir is None:
            output_dir = self.config.get('paths', {}).get('output', {}).get('optimisation', 'optimisation')
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
          # Generate study name if not provided
        if study_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            study_name = f"{self.model_type}_opt_{timestamp}"
        self.study_name = study_name
        
        # Setup storage
        self.storage_path = self.output_dir / f"{self.study_name}.db"
        self.storage = f"sqlite:///{self.storage_path}"
        
        # Initialize hardware manager
        self.hw_manager = HardwareManager(config_path)
        
        # Determine number of parallel jobs based on hardware
        if n_jobs == -1:
            self.n_jobs = self._determine_optimal_jobs()
        else:
            self.n_jobs = n_jobs
        
        # Load hyperparameter search spaces from config
        self.param_spaces = self.config.get('hyperparameter_optimisation', {}).get('param_spaces', {})
        if not self.param_spaces:
            self.logger.warning("No parameter spaces defined in config. Using default spaces.")
            self._set_default_param_spaces()
        
        self.logger.info(f"Initialised hyperparameter optimiser with study name '{study_name}'")
        self.logger.info(f"Using {self.n_jobs} parallel jobs for optimisation")
        
        # Track best parameters and results
        self.best_params = None
        self.best_value = None
        self.best_trial_number = None
        self.best_model_path = None
        
        # Create Optuna study
        self.study = self._create_or_load_study()
    
    def _load_config(self, config_path: str) -> Dict:
        """
        Load the configuration from a YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Dictionary containing the configuration
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config
        except Exception as e:
            self.logger.error(f"Failed to load configuration from {config_path}: {e}")
            return {}
    
    def _determine_optimal_jobs(self) -> int:
        """
        Determine the optimal number of parallel jobs based on hardware.
        
        Returns:
            Number of optimal parallel jobs
        """
        # Get hardware info
        cpu_count = os.cpu_count()
        
        # For CPU-only, use half the cores to avoid resource exhaustion
        if self.hw_manager.device == 'cpu':
            return max(1, cpu_count // 2)
        
        # For GPU, estimate based on VRAM
        vram_gb = self.hw_manager.vram_total / (1024**3)
        
        # Heuristic: each job needs ~4GB VRAM for YOLOv8m
        max_jobs_by_vram = max(1, int(vram_gb // 4))
        
        # Also consider CPU count for data loading
        max_jobs = min(max_jobs_by_vram, cpu_count // 2)
        
        self.logger.info(f"Determined optimal parallel jobs: {max_jobs}")
        return max_jobs
    
    def _set_default_param_spaces(self) -> None:
        """
        Set default hyperparameter search spaces if not specified in config.
        """
        self.param_spaces = {
            'lr0': {  # Changed from 'learning_rate' to 'lr0'
                'type': 'float',
                'low': 1e-5,
                'high': 1e-2,
                'log': True
            },
            'momentum': {
                'type': 'float',
                'low': 0.6,
                'high': 0.98
            },
            'weight_decay': {
                'type': 'float',
                'low': 1e-5,
                'high': 1e-2,
                'log': True
            },
            'warmup_epochs': {
                'type': 'int',
                'low': 0,
                'high': 5
            },
            'batch': {  # Changed from 'batch_size' to 'batch'
                'type': 'categorical',
                'choices': [8, 16, 24, 32]
            },
            'imgsz': {  # Changed from 'img_size' to 'imgsz'
                'type': 'categorical',
                'choices': [512, 640, 768]
            },
            'mosaic': {
                'type': 'categorical',
                'choices': [0, 0.5, 1.0]
            }
        }
    
    def _create_or_load_study(self) -> optuna.Study:
        """
        Create a new Optuna study or load an existing one.
        
        Returns:
            Optuna study object
        """
        # Load pruner settings from config
        pruner_config = self.config.get('hyperparameter_optimisation', {}).get('pruner', {})
        pruner_type = pruner_config.get('type', 'median')
        
        # Select pruner based on configuration
        if pruner_type == 'median':
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=pruner_config.get('n_startup_trials', 5),
                n_warmup_steps=pruner_config.get('n_warmup_steps', 5),
                interval_steps=pruner_config.get('interval_steps', 1)
            )
        elif pruner_type == 'percentile':
            pruner = optuna.pruners.PercentilePruner(
                percentile=pruner_config.get('percentile', 25.0),
                n_startup_trials=pruner_config.get('n_startup_trials', 5),
                n_warmup_steps=pruner_config.get('n_warmup_steps', 5),
                interval_steps=pruner_config.get('interval_steps', 1)
            )
        else:
            pruner = optuna.pruners.MedianPruner()  # Default
        
        # Load sampler settings from config
        sampler_config = self.config.get('hyperparameter_optimisation', {}).get('sampler', {})
        sampler_type = sampler_config.get('type', 'tpe')
        
        # Select sampler based on configuration
        if sampler_type == 'tpe':
            sampler = optuna.samplers.TPESampler(
                seed=sampler_config.get('seed', 42),
                n_startup_trials=sampler_config.get('n_startup_trials', 10)
            )
        elif sampler_type == 'random':
            sampler = optuna.samplers.RandomSampler(
                seed=sampler_config.get('seed', 42)
            )
        else:
            sampler = optuna.samplers.TPESampler()  # Default
        
        # Determine study direction
        direction = self.config.get('hyperparameter_optimisation', {}).get('direction', 'maximize')
        
        try:
            # Check if study already exists in storage
            study = optuna.load_study(
                study_name=self.study_name,
                storage=self.storage,
                sampler=sampler,
                pruner=pruner
            )
            self.logger.info(f"Loaded existing study '{self.study_name}' with {len(study.trials)} trials")
            
        except (optuna.exceptions.StorageInternalError, KeyError):
            # Create new study
            study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage,
                direction=direction,
                sampler=sampler,
                pruner=pruner,
                load_if_exists=True
            )
            self.logger.info(f"Created new study '{self.study_name}'")
        
        return study
    
    def _sample_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Sample hyperparameters from the defined search spaces.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of sampled parameters
        """
        params = {}
        
        for param_name, space in self.param_spaces.items():
            param_type = space.get('type')
            
            if param_type == 'float':
                if space.get('log', False):
                    params[param_name] = trial.suggest_float(
                        param_name, space['low'], space['high'], log=True
                    )
                else:
                    params[param_name] = trial.suggest_float(
                        param_name, space['low'], space['high']
                    )
            
            elif param_type == 'int':
                params[param_name] = trial.suggest_int(
                    param_name, space['low'], space['high'], step=space.get('step', 1)
                )
            
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name, space['choices']
                )
            
            else:
                self.logger.warning(f"Unknown parameter type '{param_type}' for {param_name}")
        
        return params
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for optimisation.
        """
        # Sample hyperparameters
        params = self._sample_params(trial)
        
        # Get hardware-optimised params
        hw_params = self.hw_manager.get_training_params(image_size=params.get('imgsz', 640))  # Updated from img_size
        
        # If batch is not sampled, use hardware-optimised value
        if 'batch' not in params:
            params['batch'] = hw_params['batch_size']
        
        # Use hardware-optimised workers
        params['workers'] = hw_params['workers']
        
        # Create unique run name for this trial
        trial_name = f"{self.study_name}_trial{trial.number}"
        
        # Create output directory for this trial
        trial_dir = self.output_dir / trial_name
        os.makedirs(trial_dir, exist_ok=True)
        
        self.logger.info(f"Starting trial {trial.number} with params: {params}")
        
        try:            # Initialize trainer
            trainer = YoloTrainer(
                config_path=self.config_path,
                data_yaml_path=self.data_yaml_path,
                output_dir=trial_dir,
                verbose=self.verbose,
                model_type=self.model_type
            )
            
            # Initialize model
            trainer.initialize_model()
            
            # Add callback for pruning
            pruning_callback = TrialPruningCallback(trial, metric_name='metrics/mAP50-95(B)')
            
            # Set up epochs for this trial
            epochs = self.config.get('hyperparameter_optimisation', {}).get('epochs_per_trial', 50)
            params['epochs'] = epochs
            
            # Add name parameter
            params['name'] = trial_name
            
            # Train with sampled hyperparameters
            results = trainer.train(params)
            
            # Extract validation metric
            val_metric = results.get('metrics', {}).get('metrics/mAP50-95(B)', 0.0)
            
            # Record metric
            self.logger.info(f"Trial {trial.number} finished with mAP: {val_metric:.4f}")
            
            # Track best result
            if self.best_value is None or val_metric > self.best_value:
                self.best_value = val_metric
                self.best_params = params
                self.best_trial_number = trial.number
                self.best_model_path = trainer.run_dir
                
                # Save best parameters
                self._save_best_params()
            
            return val_metric
            
        except optuna.exceptions.TrialPruned as e:
            self.logger.info(f"Trial {trial.number} pruned: {e}")
            raise e
            
        except Exception as e:
            self.logger.error(f"Error during trial {trial.number}: {e}")
            return float('-inf')  # Return worst possible score
    
    def _save_best_params(self) -> None:
        """
        Save the best parameters to a file.
        """
        best_params_path = self.output_dir / "best_params.json"
        
        best_info = {
            'best_value': self.best_value,
            'best_params': self.best_params,
            'best_trial_number': self.best_trial_number,
            'best_model_path': str(self.best_model_path) if self.best_model_path else None,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(best_params_path, 'w') as f:
                json.dump(best_info, f, indent=2)
            
            self.logger.info(f"Saved best parameters to {best_params_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving best parameters: {e}")
    
    def optimize(self) -> Dict:
        """
        Run the hyperparameter optimisation process.
        
        Returns:
            Dictionary containing optimisation results
        """
        self.logger.info(f"Starting hyperparameter optimisation with {self.n_trials} trials")
        
        try:
            # Run optimisation
            if self.n_jobs > 1:
                # Parallel execution
                self.study.optimize(
                    self.objective,
                    n_trials=self.n_trials,
                    timeout=self.timeout,
                    n_jobs=self.n_jobs,
                    gc_after_trial=True
                )
            else:
                # Sequential execution
                self.study.optimize(
                    self.objective,
                    n_trials=self.n_trials,
                    timeout=self.timeout,
                    gc_after_trial=True
                )
            
            # Get best parameters and score
            best_params = self.study.best_params
            best_value = self.study.best_value
            
            self.logger.info(f"Optimisation completed. Best mAP: {best_value:.4f}")
            self.logger.info(f"Best parameters: {best_params}")
            
            # Save results
            self._save_results()
            
            # Create visualisations
            self._create_visualisations()
            
            return {
                'best_value': best_value,
                'best_params': best_params,
                'study_name': self.study_name,
                'n_trials': len(self.study.trials),
                'completed_trials': len(self.study.get_trials(states=[optuna.trial.TrialState.COMPLETE])),
                'pruned_trials': len(self.study.get_trials(states=[optuna.trial.TrialState.PRUNED]))
            }
            
        except Exception as e:
            self.logger.error(f"Error during optimisation: {e}")
            return {
                'error': str(e)
            }
    
    def _save_results(self) -> None:
        """
        Save complete optimisation results.
        """
        results_path = self.output_dir / "optimisation_results.pkl"
        
        try:
            # Save study with pickle/joblib for complete reproducibility
            joblib.dump(self.study, results_path)
            self.logger.info(f"Saved optimisation results to {results_path}")
            
            # Save summary as JSON
            summary = {
                'study_name': self.study_name,
                'n_trials': len(self.study.trials),
                'best_value': self.study.best_value,
                'best_params': self.study.best_params,
                'best_trial': self.study.best_trial.number,
                'completed_trials': len(self.study.get_trials(states=[optuna.trial.TrialState.COMPLETE])),
                'pruned_trials': len(self.study.get_trials(states=[optuna.trial.TrialState.PRUNED])),
                'timestamp': datetime.now().isoformat()
            }
            
            summary_path = self.output_dir / "optimisation_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
    
    def _create_visualisations(self) -> None:
        """
        Create visualisations of the optimisation process.
        """
        try:
            # Create output directory for visualisations
            vis_dir = self.output_dir / "visualisations"
            os.makedirs(vis_dir, exist_ok=True)
            
            # 1. Parameter importance plot
            param_importance = optuna.importance.get_param_importances(self.study)
            
            plt.figure(figsize=(10, 6))
            params = list(param_importance.keys())
            importance = list(param_importance.values())
            
            # Sort by importance
            sorted_indices = np.argsort(importance)
            params = [params[i] for i in sorted_indices]
            importance = [importance[i] for i in sorted_indices]
            
            plt.barh(params, importance)
            plt.xlabel('Importance')
            plt.ylabel('Parameter')
            plt.title('Hyperparameter Importance')
            plt.tight_layout()
            plt.savefig(vis_dir / "param_importance.png")
            plt.close()
            
            # 2. Optimization history plot
            plt.figure(figsize=(10, 6))
            optuna.visualization.matplotlib.plot_optimization_history(self.study)
            plt.tight_layout()
            plt.savefig(vis_dir / "optimisation_history.png")
            plt.close()
            
            # 3. Parallel coordinate plot for parameters
            plt.figure(figsize=(12, 8))
            optuna.visualization.matplotlib.plot_parallel_coordinate(self.study)
            plt.tight_layout()
            plt.savefig(vis_dir / "parallel_coordinate.png")
            plt.close()
            
            # 4. Parameter contour plots (for pairs of important parameters)
            if len(param_importance) >= 2:
                important_params = list(param_importance.keys())[:2]  # Top 2 parameters
                plt.figure(figsize=(10, 8))
                optuna.visualization.matplotlib.plot_contour(self.study, params=important_params)
                plt.tight_layout()
                plt.savefig(vis_dir / "param_contour.png")
                plt.close()
            
            self.logger.info(f"Created visualisations in {vis_dir}")
            
        except Exception as e:
            self.logger.error(f"Error creating visualisations: {e}")
    
    def load_best_model(self) -> Dict:
        """
        Load the best model parameters from the optimisation process.
        
        Returns:
            Dictionary containing best model information
        """
        best_params_path = self.output_dir / "best_params.json"
        
        try:
            if os.path.exists(best_params_path):
                with open(best_params_path, 'r') as f:
                    best_info = json.load(f)
                
                self.best_value = best_info.get('best_value')
                self.best_params = best_info.get('best_params')
                self.best_trial_number = best_info.get('best_trial_number')
                self.best_model_path = best_info.get('best_model_path')
                
                self.logger.info(f"Loaded best parameters with mAP: {self.best_value:.4f}")
                
                return best_info
            else:
                self.logger.warning(f"Best parameters file not found: {best_params_path}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error loading best parameters: {e}")
            return {}
    
    def get_best_params(self) -> Dict:
        """
        Get the best hyperparameters from the optimisation.
        
        Returns:
            Dictionary of best hyperparameters
        """
        if self.best_params:
            return self.best_params
        
        if hasattr(self, 'study') and self.study.best_params:
            return self.study.best_params
        
        # Try to load best params from file
        best_info = self.load_best_model()
        if best_info and 'best_params' in best_info:
            return best_info['best_params']
        
        return {}


class TrialPruningCallback:
    """
    Callback for pruning unpromising trials during training.
    """
    
    def __init__(self, trial: optuna.Trial, metric_name: str = 'metrics/mAP50-95(B)'):
        """
        Initialise the pruning callback.
        
        Args:
            trial: Optuna trial object
            metric_name: Name of the metric to monitor for pruning
        """
        self.trial = trial
        self.metric_name = metric_name
    
    def __call__(self, trainer, metrics: Dict[str, float]) -> None:
        """
        Callback function called after each epoch.
        
        Args:
            trainer: YoloTrainer instance
            metrics: Dictionary of metrics from the current epoch
        """
        # Get current epoch and metric value
        epoch = trainer.current_epoch
        value = metrics.get(self.metric_name, 0.0)
        
        # Report the value to Optuna for potential pruning
        self.trial.report(value, epoch)
        
        # Prune if trial is unpromising
        if self.trial.should_prune():
            raise optuna.exceptions.TrialPruned()


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    import argparse
    parser = argparse.ArgumentParser(description="YOLOv8 Hyperparameter Optimisation")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--data', type=str, default='data/data.yaml', help='Path to data YAML file')
    parser.add_argument('--trials', type=int, default=25, help='Number of trials to run')
    parser.add_argument('--jobs', type=int, default=-1, help='Number of parallel jobs (-1 for auto)')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--study', type=str, default=None, help='Study name')
    parser.add_argument('--model', type=str, default='yolov8m', choices=['yolov8m', 'yolov11s', 'yolov11m', 'yolov11l'],
                        help='YOLO model to use')
    
    args = parser.parse_args()
      # Create optimiser
    optimiser = HyperparameterOptimiser(
        config_path=args.config,
        data_yaml_path=args.data,
        output_dir=args.output,
        study_name=args.study,
        n_trials=args.trials,
        n_jobs=args.jobs,
        verbose=True,
        model_type=args.model
    )
    
    # Run optimisation
    results = optimiser.optimize()
    
    print(f"\nOptimisation completed with best mAP: {results['best_value']:.4f}")
    print(f"Best parameters: {results['best_params']}")
