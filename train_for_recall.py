#!/usr/bin/env python3
"""
Specialized training script for maximizing recall in cone detection.

This script implements advanced techniques specifically designed to reduce
false negatives in YOLO-based traffic cone detection for autonomous driving.
"""

import os
import logging
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch

def train_for_maximum_recall(config_path: str = "config_recall_optimized.yaml"):
    """
    Train YOLO model with optimizations for maximum recall.
    
    Args:
        config_path: Path to the recall-optimized configuration file
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting recall-optimized training for cone detection")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    model = YOLO('yolov8n.yaml')  # Start from scratch for optimal recall training
    
    # Custom training parameters for recall optimization
    train_params = {
        'data': config['paths']['dataset']['data_yaml'],
        'epochs': config['training']['epochs'],
        'imgsz': config['training']['img_size'],
        'batch': config['hardware']['batch_size'] if config['hardware']['batch_size'] > 0 else 16,
        'workers': config['hardware']['workers'],
        
        # Optimizer settings for recall
        'optimizer': 'AdamW',  # AdamW often better for recall optimization
        'lr0': 0.002,         # Conservative learning rate
        'lrf': 0.01,          # Final learning rate
        'momentum': 0.95,     # High momentum for stability
        'weight_decay': 0.0005,
        
        # Loss function weights optimized for recall
        'box': config['training']['loss_weights']['box'],
        'cls': config['training']['loss_weights']['cls'],
        'dfl': 1.5,  # Distribution focal loss for better localization
        
        # Augmentation parameters for recall
        'hsv_h': 0.015,    # Reduced hue augmentation
        'hsv_s': 0.7,      # Saturation augmentation
        'hsv_v': 0.4,      # Value augmentation
        'degrees': 5.0,    # Reduced rotation
        'translate': 0.1,  # Translation augmentation
        'scale': 0.5,      # Scale augmentation for distance simulation
        'shear': 2.0,      # Minimal shear
        'perspective': 0.0, # No perspective distortion
        'flipud': 0.0,     # No vertical flip for cones
        'fliplr': 0.5,     # Horizontal flip probability
        'mosaic': config['training']['optimisation']['mosaic_prob'],
        'mixup': config['training']['optimisation']['mixup_prob'],
        'copy_paste': config['training']['optimisation']['copy_paste_prob'],
        
        # Advanced settings for recall
        'patience': config['training']['early_stopping']['patience'],
        'save_period': 10,
        'seed': 42,
        'deterministic': True,
        'single_cls': False,
        'rect': False,     # No rectangular training for better augmentation
        'cos_lr': True,    # Cosine learning rate schedule
        'close_mosaic': 15, # Close mosaic early to focus on individual cones
        
        # Validation settings
        'val': True,
        'plots': True,
        'save_json': True,
        
        # Hardware settings
        'device': config['hardware']['device'],
        'amp': True,  # Mixed precision for memory efficiency
        
        # Custom callbacks for recall monitoring
        'project': config['paths']['output']['models'],
        'name': f"recall_optimized_run_{os.getenv('USER', 'user')}",
        'exist_ok': True,
        'verbose': True
    }
    
    # Custom callback for recall tracking
    def recall_callback(trainer):
        """Custom callback to monitor and save best recall model."""
        if hasattr(trainer, 'metrics'):
            current_recall = trainer.metrics.get('metrics/recall(B)', 0)
            if not hasattr(recall_callback, 'best_recall'):
                recall_callback.best_recall = 0
            
            if current_recall > recall_callback.best_recall:
                recall_callback.best_recall = current_recall
                # Save best recall model
                best_recall_path = Path(trainer.save_dir) / 'best_recall.pt'
                trainer.model.save(best_recall_path)
                logger.info(f"New best recall: {current_recall:.4f}, saved to {best_recall_path}")
    
    # Add custom callback
    model.add_callback('on_train_epoch_end', recall_callback)
    
    # Train the model
    logger.info("Starting training with recall optimization parameters")
    results = model.train(**train_params)
    
    # Post-training threshold optimization
    logger.info("Training completed. Starting threshold optimization for maximum recall.")
    
    # Find the best model
    best_model_path = Path(train_params['project']) / train_params['name'] / 'weights' / 'best.pt'
    if not best_model_path.exists():
        best_model_path = Path(train_params['project']) / train_params['name'] / 'weights' / 'last.pt'
    
    # Threshold tuning for maximum recall
    from src.inference.threshold_tuning import ThresholdTuner
    
    tuner = ThresholdTuner(
        model_path=str(best_model_path),
        data_yaml_path=config['paths']['dataset']['data_yaml'],
        output_dir=str(Path(train_params['project']) / train_params['name'] / 'threshold_tuning')
    )
    
    # Comprehensive threshold tuning optimized for recall
    optimal_config = tuner.comprehensive_threshold_tuning(
        conf_range=(0.01, 0.5),  # Very low confidence range
        iou_range=(0.1, 0.7),    # Lower IoU range for less aggressive NMS
        conf_steps=25,
        iou_steps=20
    )
    
    logger.info("Recall optimization complete!")
    logger.info(f"Best model saved to: {best_model_path}")
    logger.info(f"Optimal inference configuration:")
    logger.info(f"  Confidence threshold: {optimal_config['confidence_threshold']:.3f}")
    logger.info(f"  IoU threshold: {optimal_config['iou_threshold']:.3f}")
    logger.info(f"  Expected recall: {optimal_config['expected_metrics']['recall']:.3f}")
    logger.info(f"  Expected precision: {optimal_config['expected_metrics']['precision']:.3f}")
    
    return results, optimal_config

if __name__ == "__main__":
    results, config = train_for_maximum_recall()
