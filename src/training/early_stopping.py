import numpy as np
import logging
from typing import Optional, Dict, Any, List
import os
import json
from datetime import datetime

class EarlyStopping:
    """
    Early stopping implementation for YOLOv8 training.
    
    This class provides a mechanism to stop training early when validation metrics
    stop improving for a specified number of epochs (patience). It also tracks the
    best model weights and can restore them.
    """
    
    def __init__(
        self, 
        patience: int = 20, 
        min_delta: float = 0.0001,
        metric_name: str = 'mAP50-95',
        maximize: bool = True,
        save_dir: Optional[str] = None,
        verbose: bool = True,
        recall_threshold: Optional[float] = None,
        recall_weight: float = 0.3
    ):
        """
        Initialise the early stopping mechanism.
        
        Args:
            patience: Number of epochs with no improvement before stopping
            min_delta: Minimum change in the monitored metric to qualify as improvement
            metric_name: Metric to monitor for improvement
            maximize: Whether the metric should be maximized (True) or minimized (False)
            save_dir: Directory to save best model checkpoints
            verbose: Whether to log early stopping events
            recall_threshold: Minimum recall threshold to consider (for recall-focused training)
            recall_weight: Weight for recall in composite scoring
        """
        self.patience = patience
        self.min_delta = min_delta
        self.metric_name = metric_name
        self.maximize = maximize
        self.save_dir = save_dir
        self.verbose = verbose
        self.recall_threshold = recall_threshold
        self.recall_weight = recall_weight
        
        # Initialise tracking variables
        self.best_score = None
        self.best_epoch = 0
        self.counter = 0
        self.early_stop = False
        self.history = []
        self.learning_rates = []
        self.validation_metrics = {}
        
        self.logger = logging.getLogger(self.__class__.__name__)
        if self.verbose:
            self.logger.info(f"Early stopping initialised with patience={patience}, "
                            f"min_delta={min_delta}, metric={metric_name}")
            if recall_threshold is not None:
                self.logger.info(f"Recall threshold set to {recall_threshold:.3f}")
    
    def check(self, epoch: int, metrics: Dict[str, float], model_path: Optional[str] = None) -> bool:
        """
        Check if training should be stopped based on validation metrics.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of validation metrics
            model_path: Path to the current model checkpoint
            
        Returns:
            True if training should stop, False otherwise
        """
        # Store metrics history
        self.history.append({'epoch': epoch, **metrics})
        
        # Save validation metrics for later analysis
        for key, value in metrics.items():
            if key not in self.validation_metrics:
                self.validation_metrics[key] = []
            self.validation_metrics[key].append(value)
        
        # Get the score to monitor
        current_score = metrics.get(self.metric_name)
        if current_score is None:
            self.logger.warning(f"Metric '{self.metric_name}' not found in metrics. "
                              f"Available metrics: {list(metrics.keys())}")
            return False
        
        # Apply recall threshold if specified
        if self.recall_threshold is not None:
            current_recall = metrics.get('metrics/recall(B)', 0.0)
            if current_recall < self.recall_threshold:
                self.logger.info(f"Recall {current_recall:.3f} below threshold {self.recall_threshold:.3f}, "
                               "not considering this epoch for best model")
                return False
        
        # Calculate composite score if recall weighting is enabled
        if self.recall_weight > 0 and 'metrics/recall(B)' in metrics:
            current_recall = metrics.get('metrics/recall(B)', 0.0)
            # Composite score: weighted combination of main metric and recall
            composite_score = (1 - self.recall_weight) * current_score + self.recall_weight * current_recall
            scoring_value = composite_score
            if self.verbose:
                self.logger.info(f"Using composite score: {composite_score:.6f} "
                               f"(metric: {current_score:.6f}, recall: {current_recall:.6f})")
        else:
            scoring_value = current_score
        
        # First score or score improved
        if self.best_score is None:
            self._update_best(scoring_value, epoch, model_path)
            return False
        
        # Check if score improved
        if self.maximize:
            improved = scoring_value > self.best_score + self.min_delta
        else:
            improved = scoring_value < self.best_score - self.min_delta
        
        if improved:
            self._update_best(scoring_value, epoch, model_path)
            return False
        else:
            self.counter += 1
            if self.verbose:
                self.logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}. "
                               f"Best {self.metric_name}: {self.best_score:.6f}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    self.logger.info(f"Early stopping triggered at epoch {epoch}. "
                                   f"Best {self.metric_name}: {self.best_score:.6f} at epoch {self.best_epoch}")
                return True
            
            return False
    
    def _update_best(self, score: float, epoch: int, model_path: Optional[str]):
        """
        Update the best score and related variables.
        
        Args:
            score: New best score
            epoch: Epoch where the best score was achieved
            model_path: Path to the model checkpoint
        """
        self.best_score = score
        self.best_epoch = epoch
        self.counter = 0
        
        if self.verbose:
            self.logger.info(f"New best {self.metric_name}: {self.best_score:.6f} at epoch {epoch}")
        
        # Save best model path
        if model_path and self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            
            try:
                # Copy or save reference to best model
                import shutil
                if os.path.exists(model_path):
                    shutil.copy2(model_path, best_path)
                    self.logger.info(f"Saved best model to {best_path}")
            except Exception as e:
                self.logger.warning(f"Error saving best model: {e}")
    
    def add_learning_rate(self, lr: float):
        """
        Add current learning rate to history for tracking.
        
        Args:
            lr: Current learning rate
        """
        self.learning_rates.append(lr)
    
    def estimate_optimal_epochs(self) -> int:
        """
        Estimate the optimal number of epochs based on validation history.
        This can be used to determine the ideal training duration for future runs.
        
        Returns:
            Optimal number of epochs estimated from validation history
        """
        if not self.validation_metrics or self.metric_name not in self.validation_metrics:
            return -1
        
        # Get the validation metric history
        scores = self.validation_metrics[self.metric_name]
        
        if not scores:
            return -1
        
        # Get the epoch with the best score
        if self.maximize:
            best_idx = np.argmax(scores)
        else:
            best_idx = np.argmin(scores)
        
        # Add a margin to the best epoch (e.g., 10% more epochs)
        optimal_epochs = int(best_idx * 1.1) + 1
        
        # Ensure we don't recommend fewer epochs than our best result
        optimal_epochs = max(optimal_epochs, best_idx + 1)
        
        if self.verbose:
            self.logger.info(f"Estimated optimal epochs: {optimal_epochs} "
                           f"based on best {self.metric_name} at epoch {best_idx + 1}")
        
        return optimal_epochs
    
    def save_history(self, filepath: str):
        """
        Save the training history to a file.
        
        Args:
            filepath: Path to save the history
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            history_data = {
                'best_score': self.best_score,
                'best_epoch': self.best_epoch,
                'patience': self.patience,
                'min_delta': self.min_delta,
                'metric_name': self.metric_name,
                'history': self.history,
                'learning_rates': self.learning_rates,
                'validation_metrics': self.validation_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(history_data, f, indent=2)
                
            if self.verbose:
                self.logger.info(f"Saved training history to {filepath}")
                
        except Exception as e:
            self.logger.error(f"Failed to save history: {e}")
    
    def get_best_checkpoint(self) -> str:
        """
        Get the path to the best model checkpoint.
        
        Returns:
            Path to the best model checkpoint or None if not available
        """
        if not self.save_dir:
            return None
        
        best_path = os.path.join(self.save_dir, 'best_model.pt')
        if os.path.exists(best_path):
            return best_path
        return None
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the training results.
        
        Returns:
            Dictionary with training summary information
        """
        return {
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'total_epochs': len(self.history),
            'stopping_reason': 'early_stopping' if self.early_stop else 'completed',
            'optimal_epochs': self.estimate_optimal_epochs()
        }
