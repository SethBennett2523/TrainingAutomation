#!/usr/bin/env python
"""
AutoTrainer: Automated YOLOv8 model training system.

This module provides a command-line interface for training, validating,
and optimising YOLOv8 models for cone detection with hardware-aware
optimisations for both NVIDIA and AMD GPUs.
"""

import os
import sys
import logging
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import local modules
from src.utils.file_io import FileIO
from src.training.hardware_manager import HardwareManager
from src.training.yolo_trainer import YoloTrainer
from src.training.early_stopping import EarlyStopping
from src.preprocessing.dataset import DatasetManager
from src.preprocessing.augmentations import AugmentationManager
from src.hyperparameter_optimisation.optimiser import HyperparameterOptimiser


def setup_logging(log_dir: str = 'logs', log_level: int = logging.INFO) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"autotrainer_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Log system information
    logger = logging.getLogger(__name__)
    logger.info(f"AutoTrainer started at {timestamp}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Operating system: {sys.platform}")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="AutoTrainer: Automated YOLOv8 Training System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Main operation mode
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'optimise', 'validate', 'export'],
                        help="Operation mode")
    
    # Configuration paths
    parser.add_argument('--config', type=str, default='config.yaml',
                        help="Path to main configuration file")
    parser.add_argument('--data', type=str, default='data/data.yaml',
                        help="Path to data configuration file")
    
    # Output directory
    parser.add_argument('--output', type=str, default=None,
                        help="Output directory (default from config)")
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=None,
                        help="Number of training epochs (overrides config)")
    parser.add_argument('--batch', type=int, default=None,
                        help="Batch size (overrides auto-detection)")
    parser.add_argument('--img-size', type=int, default=None,
                        help="Image size for training (overrides config)")
    
    # Hardware options
    parser.add_argument('--device', type=str, default=None,
                        choices=['cuda', 'rocm', 'cpu'],
                        help="Device to use (overrides auto-detection)")
    parser.add_argument('--workers', type=int, default=None,
                        help="Number of workers for data loading")
    
    # Hyperparameter optimisation
    parser.add_argument('--trials', type=int, default=25,
                        help="Number of optimisation trials")
    parser.add_argument('--study-name', type=str, default=None,
                        help="Name for the optimisation study")
    
    # Augmentation toggles
    parser.add_argument('--no-noise', action='store_true',
                        help="Disable noise augmentation")
    parser.add_argument('--no-distortion', action='store_true',
                        help="Disable lens distortion augmentation")
    parser.add_argument('--no-motion-blur', action='store_true',
                        help="Disable motion blur augmentation")
    parser.add_argument('--no-occlusion', action='store_true',
                        help="Disable occlusion augmentation")
    
    # Export options
    parser.add_argument('--export-format', type=str, default='onnx',
                        choices=['onnx', 'torchscript', 'openvino', 'tflite'],
                        help="Model export format")
    
    # Logging options
    parser.add_argument('--verbose', action='store_true',
                        help="Enable verbose output")
    parser.add_argument('--log-dir', type=str, default='logs',
                        help="Directory for log files")
    
    return parser.parse_args()


def prepare_environment(args: argparse.Namespace) -> Dict:
    """
    Prepare the environment for training.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Dictionary with environment setup information
    """
    # Set up file I/O
    file_io = FileIO()
    
    # Load configuration files
    config_path = file_io.resolve_path(args.config)
    data_path = file_io.resolve_path(args.data)
    
    config = file_io.load_yaml(config_path)
    data_config = file_io.load_yaml(data_path)
    
    # Apply command-line overrides to configuration
    if args.output:
        config.setdefault('paths', {}).setdefault('output', {})['models'] = args.output
    
    if args.device:
        config.setdefault('hardware', {})['device'] = args.device
    
    if args.workers is not None:
        config.setdefault('hardware', {})['workers'] = args.workers
    
    if args.epochs is not None:
        config.setdefault('training', {})['epochs'] = args.epochs
    
    if args.batch is not None:
        config.setdefault('hardware', {})['batch_size'] = args.batch
    
    if args.img_size is not None:
        config.setdefault('training', {})['img_size'] = args.img_size
    
    # Apply augmentation toggles
    if args.no_noise or args.no_distortion or args.no_motion_blur or args.no_occlusion:
        # Ensure the augmentations section exists
        data_config.setdefault('augmentations', {})
        
    if args.no_noise:
        data_config['augmentations']['noise'] = {'enabled': False}
    
    if args.no_distortion:
        data_config['augmentations']['distortion'] = {'enabled': False}
    
    if args.no_motion_blur:
        data_config['augmentations']['motion_blur'] = {'enabled': False}
    
    if args.no_occlusion:
        data_config['augmentations']['occlusion'] = {'enabled': False}
    
    # Initialize hardware manager
    hw_manager = HardwareManager(config_path)
    
    # Get output paths
    output_models_dir = file_io.resolve_path(
        config.get('paths', {}).get('output', {}).get('models', 'models'),
        allow_nonexistent=True
    )
    
    # Create output directories
    file_io.create_directory(output_models_dir)
    
    # Print hardware summary if verbose
    if args.verbose:
        hw_manager.print_hardware_summary()
    
    # Return environment information
    return {
        'config': config,
        'data_config': data_config,
        'hw_manager': hw_manager,
        'file_io': file_io,
        'output_models_dir': output_models_dir,
        'config_path': str(config_path),
        'data_path': str(data_path)
    }


def train_model(args: argparse.Namespace, env: Dict) -> None:
    """
    Train a YOLOv8 model.
    
    Args:
        args: Command-line arguments
        env: Environment information
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting model training")
    
    try:
        # Create dataset manager
        dataset_manager = DatasetManager(
            data_yaml_path=env['data_path'],
            config=env['data_config'],
            batch_size=env['hw_manager'].calculate_optimal_batch_size(),
            workers=env['hw_manager'].calculate_optimal_workers(),
            img_size=env['config'].get('training', {}).get('img_size', 640)
        )
        
        # Create datasets and data loaders
        logger.info("Preparing datasets")
        train_dataset, val_dataset = dataset_manager.create_datasets()
        
        # Create trainer
        logger.info("Initialising trainer")
        trainer = YoloTrainer(
            config_path=env['config_path'],
            data_yaml_path=env['data_path'],
            output_dir=env['output_models_dir'],
            verbose=args.verbose
        )
        
        # Initialize model
        trainer.initialise_model()
        
        # Train model
        logger.info("Starting training")
        start_time = time.time()
        results = trainer.train()
        training_time = time.time() - start_time
        
        # Log results
        logger.info(f"Training completed in {training_time:.2f} seconds")
        best_map = results.get('early_stopping', {}).get('best_score', 0)
        logger.info(f"Best mAP: {best_map:.4f}")
        
        # Export model
        if args.export_format:
            logger.info(f"Exporting model to {args.export_format}")
            export_path = trainer.export_model(format=args.export_format)
            logger.info(f"Model exported to {export_path}")
        
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        sys.exit(1)


def optimise_hyperparameters(args: argparse.Namespace, env: Dict) -> None:
    """
    Run hyperparameter optimisation.
    
    Args:
        args: Command-line arguments
        env: Environment information
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting hyperparameter optimisation")
    
    try:
        # Get output directory for optimisation
        output_dir = env['config'].get('paths', {}).get('output', {}).get('optimisation', 'optimisation')
        output_path = env['file_io'].resolve_path(output_dir, allow_nonexistent=True)
        env['file_io'].create_directory(output_path)
        
        # Create optimiser
        optimiser = HyperparameterOptimiser(
            config_path=env['config_path'],
            data_yaml_path=env['data_path'],
            output_dir=output_path,
            study_name=args.study_name,
            n_trials=args.trials,
            n_jobs=-1,  # Auto-detection
            verbose=args.verbose
        )
        
        # Run optimisation
        logger.info(f"Running optimisation with {args.trials} trials")
        start_time = time.time()
        results = optimiser.optimize()
        optimisation_time = time.time() - start_time
        
        # Log results
        logger.info(f"Optimisation completed in {optimisation_time:.2f} seconds")
        logger.info(f"Best mAP: {results.get('best_value', 0):.4f}")
        logger.info(f"Best parameters: {results.get('best_params', {})}")
        
    except Exception as e:
        logger.error(f"Error during hyperparameter optimisation: {e}", exc_info=True)
        sys.exit(1)


def validate_model(args: argparse.Namespace, env: Dict) -> None:
    """
    Validate a trained model.
    
    Args:
        args: Command-line arguments
        env: Environment information
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting model validation")
    
    try:
        # Create trainer
        trainer = YoloTrainer(
            config_path=env['config_path'],
            data_yaml_path=env['data_path'],
            output_dir=env['output_models_dir'],
            verbose=args.verbose
        )
        
        # Initialize model from the last checkpoint
        trainer.initialise_model()
        
        # Validate model
        logger.info("Running validation")
        metrics = trainer.validate()
        
        # Log results
        logger.info("Validation results:")
        for key, value in metrics.items():
            logger.info(f"{key}: {value:.4f}")
        
    except Exception as e:
        logger.error(f"Error during validation: {e}", exc_info=True)
        sys.exit(1)


def export_model(args: argparse.Namespace, env: Dict) -> None:
    """
    Export a trained model to the specified format.
    
    Args:
        args: Command-line arguments
        env: Environment information
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Exporting model to {args.export_format}")
    
    try:
        # Create trainer
        trainer = YoloTrainer(
            config_path=env['config_path'],
            data_yaml_path=env['data_path'],
            output_dir=env['output_models_dir'],
            verbose=args.verbose
        )
        
        # Initialize model from the last checkpoint
        trainer.initialise_model()
        
        # Export model
        export_path = trainer.export_model(format=args.export_format)
        logger.info(f"Model exported to {export_path}")
        
    except Exception as e:
        logger.error(f"Error during model export: {e}", exc_info=True)
        sys.exit(1)


def main():
    """
    Main entry point for the application.
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(args.log_dir, log_level)
    
    logger = logging.getLogger(__name__)
    
    try:
        # Prepare the environment
        logger.info("Preparing environment")
        env = prepare_environment(args)
        
        # Execute the requested operation
        if args.mode == 'train':
            train_model(args, env)
        elif args.mode == 'optimise':
            optimise_hyperparameters(args, env)
        elif args.mode == 'validate':
            validate_model(args, env)
        elif args.mode == 'export':
            export_model(args, env)
        else:
            logger.error(f"Unknown mode: {args.mode}")
            sys.exit(1)
        
        logger.info("Operation completed successfully")
        
    except Exception as e:
        logger.error(f"Unhandled error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
