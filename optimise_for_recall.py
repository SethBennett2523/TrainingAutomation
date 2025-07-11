#!/usr/bin/env python3
"""
Enhanced configuration for reducing false negatives in cone detection models.

This script optimizes training parameters specifically to minimize false negatives
while maintaining acceptable precision for Formula Student Autonomous Driving.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_enhanced_config_for_recall() -> Dict[str, Any]:
    """
    Create an enhanced configuration specifically optimized for reducing false negatives.
    
    Returns:
        Enhanced configuration dictionary
    """
    config = {
        # Base paths
        'paths': {
            'dataset': {
                'root': "./data/FSOCO_dataset",
                'data_yaml': "./data/data.yaml"
            },
            'output': {
                'models': "./models",
                'logs': "./logs"
            }
        },
        
        # Hardware configuration
        'hardware': {
            'device': "auto",
            'batch_size': 0,  # Auto-optimized
            'workers': 0,
            'memory_threshold': 0.85
        },
        
        # Enhanced training parameters for recall optimization
        'training': {
            'epochs': 400,  # Increased for better convergence
            'img_size': 704,  # Higher resolution for small cone detection
            'early_stopping': {
                'enabled': True,
                'patience': 50,  # Increased patience for recall optimization
                'min_delta': 0.00005,
                'recall_threshold': 0.85,  # Minimum acceptable recall
                'recall_weight': 0.8  # High weight on recall
            },
            'pretrained': False,
            'name_format': "${USERNAME}${YY}w${WW}_recall_optimized",
            
            # Optimized loss weights for recall
            'loss_weights': {
                'box': 9.0,     # Increased for better localization
                'cls': 0.8,     # Increased for better classification
                'obj': 1.2      # Increased objectness weight
            },
            
            # Training optimisation for maximum recall
            'optimisation': {
                # Focal loss parameters - aggressive hard example focus
                'focal_loss_gamma': 2.0,  # Higher gamma for hard negatives
                'label_smoothing': 0.05,  # Reduced for stronger signals
                
                # Anchor optimisation - more positive anchors
                'anchor_threshold': 3.0,  # Lower threshold for more anchors
                
                # Conservative augmentation to preserve small cones
                'mosaic_prob': 0.6,      # Reduced to preserve cone visibility
                'mixup_prob': 0.05,      # Minimal mixup to avoid confusion
                'copy_paste_prob': 0.4,  # Increased copy-paste for more cone instances
                
                # Class imbalance handling
                'cls_pos_weight': 1.5,   # Boost positive class weight
                'objectness_smooth': 0.9 # High objectness smoothing
            }
        },
        
        # Hyperparameter optimization focused on recall
        'hyperparameter_optimisation': {
            'enabled': True,
            'method': "optuna",
            'trials': 50,  # Increased trials for better optimization
            
            # Primary objective: maximize recall
            'objective': {
                'primary_metric': "metrics/recall(B)",
                'secondary_metric': "metrics/mAP50(B)",
                'recall_weight': 0.85,  # Very high recall weight
                'map_weight': 0.15
            },
            
            'params': {
                # Learning rate - wider range for recall optimization
                'learning_rate': {
                    'min': 0.0001,
                    'max': 0.008,
                    'log': True
                },
                
                # Momentum for stable convergence
                'momentum': {
                    'min': 0.90,
                    'max': 0.99
                },
                
                # Weight decay
                'weight_decay': {
                    'min': 0.00005,
                    'max': 0.005,
                    'log': True
                },
                
                # Batch sizes optimized for small object detection
                'batch_size': [12, 16, 20, 24],
                
                # Higher image sizes for small cone detection
                'img_size': [640, 672, 704, 736],
                
                # Loss function weights optimized for recall
                'box_loss_gain': {
                    'min': 7.0,
                    'max': 12.0
                },
                
                'cls_loss_gain': {
                    'min': 0.6,
                    'max': 1.2
                },
                
                'obj_loss_gain': {
                    'min': 0.8,
                    'max': 1.5
                },
                
                # Focal loss for hard negative mining
                'focal_loss_gamma': {
                    'min': 1.0,
                    'max': 3.0
                },
                
                # Anchor threshold for more positive anchors
                'anchor_threshold': {
                    'min': 2.0,
                    'max': 4.0
                },
                
                # Conservative augmentation parameters
                'mosaic_prob': {
                    'min': 0.4,
                    'max': 0.8
                },
                
                'mixup_prob': {
                    'min': 0.0,
                    'max': 0.1
                },
                
                # Class positive weight for imbalance
                'cls_pos_weight': {
                    'min': 1.0,
                    'max': 2.5
                }
            }
        },
        
        # Inference configuration optimized for recall
        'inference': {
            # Very low confidence threshold for maximum recall
            'confidence_threshold': 0.05,  # Aggressive low threshold
            'iou_threshold': 0.4,          # Lower IoU for less aggressive NMS
            'max_detections': 500,         # Increased max detections
            
            # Multi-scale inference for better small cone detection
            'multi_scale': {
                'enabled': True,
                'scales': [0.75, 0.9, 1.0, 1.1, 1.25]  # More scales
            },
            
            # Test-time augmentation for critical deployment
            'tta': {
                'enabled': True,
                'horizontal_flip': True,
                'vertical_flip': False,  # Don't flip cones upside down
                'scales': [0.9, 1.0, 1.1],
                'confidence_aggregation': 'max'  # Take maximum confidence
            }
        },
        
        # Enhanced data augmentation for cone detection
        'data_augmentation': {
            # Small object augmentation - critical for distant cones
            'small_object_augmentation': {
                'enabled': True,
                'min_area_ratio': 0.0005,  # Very small cones
                'max_area_ratio': 0.08,
                'repeat_factor': 3,        # Triple small objects
                'scale_jitter': 0.1
            },
            
            # Copy-paste augmentation for more cone instances
            'copy_paste': {
                'enabled': True,
                'probability': 0.5,
                'max_instances': 5,        # More copied instances
                'scale_range': [0.8, 1.2],
                'aspect_ratio_range': [0.9, 1.1]
            },
            
            # Conservative photometric augmentations
            'photometric': {
                'brightness_limit': 0.1,   # Reduced to preserve cone visibility
                'contrast_limit': 0.1,
                'saturation_limit': 0.15,
                'hue_shift_limit': 5       # Minimal hue shift
            },
            
            # Distance simulation for various cone sizes
            'distance_simulation': {
                'enabled': True,
                'min_scale': 0.5,          # Simulate very distant cones
                'max_scale': 1.5,
                'blur_distant': True,
                'noise_distant': True
            },
            
            # Occlusion handling
            'occlusion': {
                'enabled': True,
                'max_occlusion': 0.3,      # Reduced max occlusion
                'preserve_keypoints': True
            }
        },
        
        # Validation strategy
        'validation': {
            'frequency': 5,                # Validate every 5 epochs
            'save_best_recall': True,      # Save model with best recall
            'recall_threshold': 0.90,     # Target recall threshold
            'precision_minimum': 0.75,    # Minimum acceptable precision
            'early_stop_on_recall': True  # Stop when recall target reached
        },
        
        # Logging configuration
        'logging': {
            'level': "INFO",
            'tensorboard': True,
            'save_best_only': True,
            'checkpoint_interval': 10,
            'log_recall_curves': True,
            'save_prediction_samples': True
        }
    }
    
    return config

def update_data_yaml_for_recall() -> Dict[str, Any]:
    """
    Create enhanced data.yaml configuration for recall optimization.
    
    Returns:
        Enhanced data configuration
    """
    data_config = {
        'train': 'FSOCO_dataset/train/',
        'val': 'FSOCO_dataset/valid/',
        'nc': 5,
        'names': {
            0: 'blue',
            1: 'big_orange', 
            2: 'orange',
            3: 'unknown',
            4: 'yellow'
        },
        'description': "FSOCO Dataset optimized for maximum recall",
        
        # Enhanced augmentations for recall
        'augmentations': {
            # Noise augmentation - moderate to preserve small cones
            'noise': {
                'enabled': True,
                'gaussian_std': 0.015,     # Reduced noise
                'salt_pepper_prob': 0.003
            },
            
            # Minimal distortion to preserve cone shapes
            'distortion': {
                'enabled': True,
                'barrel_factor': 0.08,     # Reduced distortion
                'pincushion_factor': 0.04
            },
            
            # Enhanced distance simulation
            'distance_simulation': {
                'enabled': True,
                'min_scale': 0.4,          # Simulate very distant cones
                'max_scale': 1.6,
                'progressive_blur': True,
                'atmospheric_effects': True
            },
            
            # Conservative motion blur
            'motion_blur': {
                'enabled': True,
                'kernel_size': 3,          # Smaller kernel
                'angle_range': [-15, 15],  # Reduced angle range
                'probability': 0.3
            },
            
            # Reduced occlusion
            'occlusion': {
                'enabled': True,
                'max_area_percentage': 0.10,  # Reduced max occlusion
                'probability': 0.4,
                'preserve_small_objects': True
            },
            
            # Enhanced color augmentations for different lighting
            'color_augmentations': {
                'enabled': True,
                'brightness_limit': 0.1,
                'contrast_limit': 0.1,
                'hue_shift_limit': 5,
                'saturation_limit': 0.15,
                'shadow_simulation': True,
                'highlight_simulation': True
            },
            
            # Enhanced copy-paste for more cone instances
            'copy_paste': {
                'enabled': True,
                'probability': 0.5,
                'max_instances': 4,
                'scale_jitter': 0.1,
                'position_jitter': 0.05,
                'rotation_range': [-5, 5]
            },
            
            # Critical: Small object augmentation
            'small_object_augmentation': {
                'enabled': True,
                'min_area_ratio': 0.0005,
                'max_area_ratio': 0.08,
                'repeat_factor': 3,
                'upsampling_probability': 0.7,
                'scale_variance': 0.1
            }
        },
        
        # Class weighting for imbalanced data
        'class_weights': {
            'auto_balance': True,
            'manual_weights': {
                0: 1.0,  # blue
                1: 1.2,  # big_orange (if less common)
                2: 1.0,  # orange
                3: 1.5,  # unknown (likely underrepresented)
                4: 1.0   # yellow
            }
        },
        
        # Validation strategy
        'validation': {
            'split_ratio': 0.15,  # Smaller validation set for more training data
            'stratified': True,   # Ensure balanced class distribution
            'min_samples_per_class': 100
        }
    }
    
    return data_config

def create_recall_optimization_training_script():
    """Create a specialized training script for recall optimization."""
    
    script_content = '''#!/usr/bin/env python3
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
'''
    
    return script_content

def main():
    """Create enhanced configuration files for recall optimization."""
    logger.info("Creating enhanced configuration for false negative reduction")
    
    # Create enhanced main config
    enhanced_config = create_enhanced_config_for_recall()
    config_path = Path("config_recall_optimized.yaml")
    
    with open(config_path, 'w') as f:
        yaml.dump(enhanced_config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Enhanced configuration saved to: {config_path}")
    
    # Create enhanced data config
    enhanced_data_config = update_data_yaml_for_recall()
    data_config_path = Path("data/data_recall_optimized.yaml")
    data_config_path.parent.mkdir(exist_ok=True)
    
    with open(data_config_path, 'w') as f:
        yaml.dump(enhanced_data_config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Enhanced data configuration saved to: {data_config_path}")
    
    # Create specialized training script
    training_script = create_recall_optimization_training_script()
    script_path = Path("train_for_recall.py")
    
    with open(script_path, 'w') as f:
        f.write(training_script)
    
    # Make script executable
    script_path.chmod(0o755)
    
    logger.info(f"Specialized training script saved to: {script_path}")
    
    # Create quick start guide
    quick_start = """
# Quick Start Guide: False Negative Reduction

## 1. Immediate Actions
```bash
# Run the recall-optimized training
python train_for_recall.py

# Or use the enhanced config with main script
python main.py train --config config_recall_optimized.yaml
```

## 2. Key Optimizations Implemented

### Training Optimizations:
- Higher image resolution (704px) for small cone detection
- Increased box loss weight (9.0) for better localization
- Focal loss gamma (2.0) for hard negative mining
- Lower anchor threshold (3.0) for more positive anchors
- Conservative augmentation to preserve cone visibility

### Inference Optimizations:
- Very low confidence threshold (0.05) for maximum recall
- Lower IoU threshold (0.4) for less aggressive NMS
- Multi-scale inference with 5 scales
- Test-time augmentation enabled
- Increased max detections (500)

### Data Optimizations:
- Small object augmentation with 3x repeat factor
- Enhanced copy-paste with up to 5 instances
- Distance simulation for very small cones
- Reduced occlusion and distortion

## 3. Expected Improvements
- Significant reduction in false negatives
- Better detection of distant/small cones
- Improved performance in challenging lighting
- More robust to occlusion and motion blur

## 4. Validation
After training, the system will automatically:
- Tune thresholds for optimal recall
- Generate performance visualizations
- Save optimized inference configuration
"""
    
    with open("FALSE_NEGATIVE_REDUCTION_GUIDE.md", 'w') as f:
        f.write(quick_start)
    
    logger.info("Configuration files created successfully!")
    logger.info("Next steps:")
    logger.info("1. Run: python train_for_recall.py")
    logger.info("2. Monitor recall metrics during training")
    logger.info("3. Use optimized thresholds for inference")

if __name__ == "__main__":
    main()
