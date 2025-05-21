import os
import yaml
import logging
import torch
from typing import Dict, Optional, List, Any, Tuple, Union
import datetime
import getpass
import time
import shutil
from pathlib import Path
import numpy as np
import json
from ultralytics import YOLO

from .early_stopping import EarlyStopping
from .hardware_manager import HardwareManager

class YoloTrainer:
    """
    YOLO trainer with hardware optimization and early stopping.
    
    This class handles the training of YOLO models (YOLOv8 and YOLOv11) with hardware optimization,
    early stopping, and proper logging.
    """
    
    def __init__(
        self,
        config_path: str,
        data_yaml_path: str,
        output_dir: str = None,
        verbose: bool = True,
        model_type: str = 'yolov8m'
    ):
        """
        Initialize the YOLOv8 trainer.
          Args:
            config_path: Path to the main configuration file
            data_yaml_path: Path to the data YAML file for training
            output_dir: Directory to save outputs
            verbose: Whether to log detailed information
            model_type: YOLO model type to use (e.g., 'yolov8m', 'yolov11s', 'yolov11m', 'yolov11l')
        """
        self.config_path = config_path
        self.data_yaml_path = data_yaml_path
        self.verbose = verbose
        self.model_type = model_type
        
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        if self.verbose:
            self.logger.setLevel(logging.INFO)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Setup output directory
        if output_dir is None:
            output_dir = self.config.get('paths', {}).get('output', {}).get('models', 'models')
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize hardware manager
        self.hw_manager = HardwareManager(config_path)
        
        # Generate training run name
        self.run_name = self._generate_run_name()
        
        # Create run directory
        self.run_dir = os.path.join(self.output_dir, self.run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Copy configuration files to run directory for reproducibility
        shutil.copy2(config_path, os.path.join(self.run_dir, 'config.yaml'))
        shutil.copy2(data_yaml_path, os.path.join(self.run_dir, 'data.yaml'))
        
        # Initialize model
        self.model = None
        
        # Initialize early stopping
        early_stopping_config = self.config.get('training', {}).get('early_stopping', {})
        self.early_stopping = EarlyStopping(
            patience=early_stopping_config.get('patience', 20),
            min_delta=early_stopping_config.get('min_delta', 0.0001),
            metric_name='metrics/mAP50-95(B)',  # YOLOv8 metric name for mAP
            save_dir=self.run_dir,
            verbose=self.verbose
        )
        
        if self.verbose:
            self.logger.info(f"Initialized YoloTrainer with run name: {self.run_name}")
            self.hw_manager.print_hardware_summary()
    
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
    
    def _generate_run_name(self) -> str:
        """
        Generate a unique name for the training run using the specified format.
        
        Returns:
            Unique run name
        """
        name_format = self.config.get('training', {}).get('name_format', "${USERNAME}${YY}w${WW}${SUFFIX}")
        
        # Get current date information
        now = datetime.datetime.now()
        year = now.strftime("%y")  # Last two digits of year
        week = now.strftime("%W")   # Week number (00-53)
        
        # Get username
        username = getpass.getuser()
        
        # Replace variables in the format string
        name = name_format.replace("${USERNAME}", username)
        name = name.replace("${YY}", year)
        name = name.replace("${WW}", week)
        
        # Add a unique suffix if not specified
        if "${SUFFIX}" in name:
            timestamp = int(time.time()) % 10000  # Last 4 digits of timestamp
            suffix = f"_{timestamp}"
            name = name.replace("${SUFFIX}", suffix)
          # Ensure the name is filesystem-safe
        name = name.replace(" ", "_")
        name = ''.join(c for c in name if c.isalnum() or c in "_-.")
        
        return name
        
    def initialize_model(self) -> None:
        """Initialize the YOLO model."""
        try:
            # Check if we're dealing with a YOLOv8 built-in model
            if self.model_type == 'yolov8m':
                # For YOLOv8m, use the built-in model
                self.model = YOLO(self.model_type)
            else:
                # For YOLOv11 models, use the custom model config
                model_config = self.config.get('models', {}).get(self.model_type, {}).get('config_path')
                if model_config:
                    # Resolve environment variables and path variables
                    model_config = model_config.replace('${paths.output.model_configs}', 
                                                        os.path.join(os.path.dirname(self.config_path), 'data/model_config'))
                    self.model = YOLO(model_config)
                else:
                    self.logger.error(f"No model configuration found for {self.model_type}")
                    raise ValueError(f"No model configuration found for {self.model_type}")
                    
            if self.verbose:
                self.logger.info(f"Initialized {self.model_type} model with random weights (no pretrained)")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {e}")
            raise
    
    def prepare_training_params(self) -> Dict:
        """
        Prepare training parameters - apply same logic regardless of model type
        """
        # Get hardware-optimized parameters
        hw_params = self.hw_manager.get_training_params(
            image_size=self.config.get('training', {}).get('img_size', 640)
        )
        
        # Get training configuration
        training_config = self.config.get('training', {})
        
        # Same parameters for all models
        train_params = {
            'data': self.data_yaml_path,
            'epochs': training_config.get('epochs', 300),
            'imgsz': training_config.get('img_size', 640),
            'batch': hw_params.get('batch_size', 16),
            'workers': hw_params.get('workers', 8),
            'device': self.hw_manager.device,
            'name': self.run_name,
            'project': self.output_dir,
            'exist_ok': True,
            'pretrained': False,
            'verbose': self.verbose,
            'save': True,
            'save_period': self.config.get('logging', {}).get('checkpoint_interval', 10),
            'patience': training_config.get('early_stopping', {}).get('patience', 0)
        }
        
        return train_params
    
    def train(self, hyperparams: Dict = None) -> Dict:
        """
        Train the YOLOv8 model with the specified parameters and early stopping.
        
        Args:
            hyperparams: Optional dictionary of hyperparameters to override defaults
            
        Returns:
            Dictionary containing training results
        """
        if self.model is None:
            self.initialize_model()
        
        # Prepare training parameters
        train_params = self.prepare_training_params()
        
        # Override with custom hyperparameters if provided
        if hyperparams:
            train_params.update(hyperparams)
        
        # Record start time
        start_time = time.time()
        
        try:
            # Start training
            self.logger.info(f"Starting training with {train_params['epochs']} epochs")
            results = self.model.train(**train_params)
            
            # Gather training metrics
            training_metrics = self._extract_metrics(results)
            
            # Calculate training time
            train_time = time.time() - start_time
            
            # Get early stopping summary
            es_summary = self.early_stopping.get_training_summary()
            
            # Combine results
            final_results = {
                'run_name': self.run_name,
                'training_time': train_time,
                'metrics': training_metrics,
                'early_stopping': es_summary,
                'hardware': {
                    'device': self.hw_manager.device,
                    'gpu_name': self.hw_manager.gpu_name,
                    'batch_size': train_params['batch'],
                    'workers': train_params['workers']
                }
            }
            
            # Save results
            self._save_results(final_results)
            
            self.logger.info(f"Training completed in {train_time:.2f} seconds")
            self.logger.info(f"Best mAP: {es_summary['best_score']:.6f} at epoch {es_summary['best_epoch']}")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            raise
    
    def _extract_metrics(self, results) -> Dict:
        """
        Extract metrics from training results.
        
        Args:
            results: Training results object from YOLO
            
        Returns:
            Dictionary of metrics
        """
        # Extract relevant metrics from YOLOv8 results
        metrics = {}
        
        # Get the validation metrics
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
        
        return metrics
    
    def _save_results(self, results: Dict) -> None:
        """
        Save training results to a file.
        
        Args:
            results: Dictionary of training results
        """
        results_path = os.path.join(self.run_dir, 'training_results.json')
        
        try:
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.logger.info(f"Saved training results to {results_path}")
        
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
    
    def validate(self, data_path: str = None) -> Dict:
        """
        Validate the trained model on a dataset.
        
        Args:
            data_path: Path to validation data, defaults to validation set in data.yaml
            
        Returns:
            Dictionary of validation metrics
        """
        if self.model is None:
            self.logger.error("Model not initialized or trained")
            return {}
        
        # If no data path provided, use the one from data.yaml
        if data_path is None:
            data_path = self.data_yaml_path
        
        self.logger.info(f"Validating model on {data_path}")
        
        try:
            # Run validation
            results = self.model.val(data=data_path)
            
            # Extract validation metrics
            metrics = {}
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error during validation: {e}")
            return {}
    
    def export_model(self, format: str = 'onnx') -> str:
        """
        Export the trained model to the specified format.
        
        Args:
            format: Export format (e.g., 'onnx', 'torchscript', 'openvino')
            
        Returns:
            Path to the exported model
        """
        if self.model is None:
            self.logger.error("Model not initialized or trained")
            return ""
        
        export_path = os.path.join(self.run_dir, f'model_{format}')
        
        try:
            # Export the model
            result = self.model.export(format=format, imgsz=self.config.get('training', {}).get('img_size', 640))
              # Copy the exported model to our run directory
            if result and hasattr(result, 'export_dir') and result.export_dir:
                src_path = result.export_dir
                if os.path.exists(src_path):
                    shutil.copy2(src_path, export_path)
                    self.logger.info(f"Exported model to {export_path}")
                    return export_path
            
            self.logger.warning(f"Export succeeded but couldn't find the output file")
            return ""
            
        except Exception as e:
            self.logger.error(f"Error exporting model: {e}")
            return ""

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize trainer
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yaml")
    data_yaml_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "data.yaml")
    
    if os.path.exists(config_path) and os.path.exists(data_yaml_path):
        trainer = YoloTrainer(config_path, data_yaml_path)
        
        # Train model
        results = trainer.train()
        
        # Export model
        trainer.export_model()
    else:
        logging.error(f"Configuration files not found. Please check the paths:\nConfig: {config_path}\nData: {data_yaml_path}")
