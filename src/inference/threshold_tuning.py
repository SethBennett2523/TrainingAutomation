"""
Threshold tuning utilities for reducing false negatives in YOLO inference.

This module provides tools to automatically tune confidence and NMS thresholds
to optimize for recall (reduce false negatives) while maintaining reasonable precision.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import yaml

from ultralytics import YOLO


class ThresholdTuner:
    """
    Automated threshold tuning for YOLO models to optimize recall/precision balance.
    """
    
    def __init__(
        self,
        model_path: str,
        data_yaml_path: str,
        output_dir: str = "threshold_tuning",
        verbose: bool = True
    ):
        """
        Initialise the threshold tuner.
        
        Args:
            model_path: Path to the trained YOLO model
            data_yaml_path: Path to the data YAML file
            output_dir: Directory to save tuning results
            verbose: Whether to log detailed information
        """
        self.model_path = model_path
        self.data_yaml_path = data_yaml_path
        self.output_dir = output_dir
        self.verbose = verbose
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load model
        self.model = YOLO(model_path)
        
        if self.verbose:
            self.logger.info(f"Initialised ThresholdTuner with model: {model_path}")
    
    def tune_confidence_threshold(
        self,
        conf_range: Tuple[float, float] = (0.1, 0.9),
        conf_steps: int = 17,
        iou_threshold: float = 0.5,
        target_metric: str = "recall"
    ) -> Dict:
        """
        Tune confidence threshold to optimize for specified metric.
        
        Args:
            conf_range: Range of confidence thresholds to test (min, max)
            conf_steps: Number of confidence values to test
            iou_threshold: IoU threshold for NMS
            target_metric: Metric to optimize ('recall', 'precision', 'f1')
            
        Returns:
            Dictionary with optimal thresholds and results
        """
        self.logger.info(f"Tuning confidence threshold for {target_metric} optimisation")
        
        # Generate confidence values to test
        conf_values = np.linspace(conf_range[0], conf_range[1], conf_steps)
        
        results = []
        
        for conf in conf_values:
            self.logger.info(f"Testing confidence threshold: {conf:.3f}")
            
            # Run validation with current threshold
            metrics = self.model.val(
                data=self.data_yaml_path,
                conf=conf,
                iou=iou_threshold,
                verbose=False
            )
            
            # Extract metrics
            if hasattr(metrics, 'results_dict'):
                result = {
                    'confidence': conf,
                    'precision': metrics.results_dict.get('metrics/precision(B)', 0),
                    'recall': metrics.results_dict.get('metrics/recall(B)', 0),
                    'mAP50': metrics.results_dict.get('metrics/mAP50(B)', 0),
                    'mAP50-95': metrics.results_dict.get('metrics/mAP50-95(B)', 0),
                }
                
                # Calculate F1 score
                if result['precision'] + result['recall'] > 0:
                    result['f1'] = 2 * (result['precision'] * result['recall']) / (result['precision'] + result['recall'])
                else:
                    result['f1'] = 0
                
                results.append(result)
        
        # Find optimal threshold based on target metric
        if target_metric == "recall":
            optimal_idx = max(range(len(results)), key=lambda i: results[i]['recall'])
        elif target_metric == "precision":
            optimal_idx = max(range(len(results)), key=lambda i: results[i]['precision'])
        elif target_metric == "f1":
            optimal_idx = max(range(len(results)), key=lambda i: results[i]['f1'])
        else:
            raise ValueError(f"Unknown target metric: {target_metric}")
        
        optimal_result = results[optimal_idx]
        
        # Save results
        self._save_confidence_tuning_results(results, optimal_result, target_metric)
        
        # Create visualization
        self._plot_confidence_tuning(results, optimal_result, target_metric)
        
        self.logger.info(f"Optimal confidence threshold: {optimal_result['confidence']:.3f}")
        self.logger.info(f"Recall: {optimal_result['recall']:.3f}, Precision: {optimal_result['precision']:.3f}, F1: {optimal_result['f1']:.3f}")
        
        return {
            'optimal_confidence': optimal_result['confidence'],
            'optimal_metrics': optimal_result,
            'all_results': results
        }
    
    def tune_iou_threshold(
        self,
        confidence: float = 0.25,
        iou_range: Tuple[float, float] = (0.1, 0.9),
        iou_steps: int = 17
    ) -> Dict:
        """
        Tune IoU threshold for NMS to optimize detection performance.
        
        Args:
            confidence: Fixed confidence threshold to use
            iou_range: Range of IoU thresholds to test (min, max)
            iou_steps: Number of IoU values to test
            
        Returns:
            Dictionary with optimal IoU threshold and results
        """
        self.logger.info("Tuning IoU threshold for NMS optimisation")
        
        # Generate IoU values to test
        iou_values = np.linspace(iou_range[0], iou_range[1], iou_steps)
        
        results = []
        
        for iou in iou_values:
            self.logger.info(f"Testing IoU threshold: {iou:.3f}")
            
            # Run validation with current threshold
            metrics = self.model.val(
                data=self.data_yaml_path,
                conf=confidence,
                iou=iou,
                verbose=False
            )
            
            # Extract metrics
            if hasattr(metrics, 'results_dict'):
                result = {
                    'iou': iou,
                    'precision': metrics.results_dict.get('metrics/precision(B)', 0),
                    'recall': metrics.results_dict.get('metrics/recall(B)', 0),
                    'mAP50': metrics.results_dict.get('metrics/mAP50(B)', 0),
                    'mAP50-95': metrics.results_dict.get('metrics/mAP50-95(B)', 0),
                }
                
                # Calculate F1 score
                if result['precision'] + result['recall'] > 0:
                    result['f1'] = 2 * (result['precision'] * result['recall']) / (result['precision'] + result['recall'])
                else:
                    result['f1'] = 0
                
                results.append(result)
        
        # Find optimal IoU threshold (maximize F1 score)
        optimal_idx = max(range(len(results)), key=lambda i: results[i]['f1'])
        optimal_result = results[optimal_idx]
        
        # Save results
        self._save_iou_tuning_results(results, optimal_result, confidence)
        
        # Create visualization
        self._plot_iou_tuning(results, optimal_result)
        
        self.logger.info(f"Optimal IoU threshold: {optimal_result['iou']:.3f}")
        self.logger.info(f"Recall: {optimal_result['recall']:.3f}, Precision: {optimal_result['precision']:.3f}, F1: {optimal_result['f1']:.3f}")
        
        return {
            'optimal_iou': optimal_result['iou'],
            'optimal_metrics': optimal_result,
            'all_results': results
        }
    
    def comprehensive_threshold_tuning(
        self,
        conf_range: Tuple[float, float] = (0.05, 0.8),
        iou_range: Tuple[float, float] = (0.1, 0.9),
        conf_steps: int = 15,
        iou_steps: int = 15
    ) -> Dict:
        """
        Perform comprehensive threshold tuning for both confidence and IoU.
        
        Args:
            conf_range: Range of confidence thresholds to test
            iou_range: Range of IoU thresholds to test
            conf_steps: Number of confidence values to test
            iou_steps: Number of IoU values to test
            
        Returns:
            Dictionary with optimal thresholds optimized for recall
        """
        self.logger.info("Starting comprehensive threshold tuning")
        
        # First, tune confidence for maximum recall
        conf_results = self.tune_confidence_threshold(
            conf_range=conf_range,
            conf_steps=conf_steps,
            target_metric="recall"
        )
        
        # Then, tune IoU with the optimal confidence
        iou_results = self.tune_iou_threshold(
            confidence=conf_results['optimal_confidence'],
            iou_range=iou_range,
            iou_steps=iou_steps
        )
        
        # Create comprehensive results
        optimal_config = {
            'confidence_threshold': conf_results['optimal_confidence'],
            'iou_threshold': iou_results['optimal_iou'],
            'expected_metrics': {
                'recall': iou_results['optimal_metrics']['recall'],
                'precision': iou_results['optimal_metrics']['precision'],
                'f1': iou_results['optimal_metrics']['f1'],
                'mAP50': iou_results['optimal_metrics']['mAP50'],
                'mAP50-95': iou_results['optimal_metrics']['mAP50-95']
            }
        }
        
        # Save comprehensive results
        results_path = os.path.join(self.output_dir, "optimal_thresholds.json")
        with open(results_path, 'w') as f:
            json.dump(optimal_config, f, indent=2)
        
        self.logger.info(f"Comprehensive tuning complete. Results saved to: {results_path}")
        self.logger.info(f"Optimal thresholds - Confidence: {optimal_config['confidence_threshold']:.3f}, IoU: {optimal_config['iou_threshold']:.3f}")
        
        return optimal_config
    
    def _save_confidence_tuning_results(self, results: List[Dict], optimal: Dict, target_metric: str):
        """Save confidence tuning results to file."""
        output_data = {
            'target_metric': target_metric,
            'optimal_threshold': optimal['confidence'],
            'optimal_metrics': optimal,
            'all_results': results
        }
        
        results_path = os.path.join(self.output_dir, f"confidence_tuning_{target_metric}.json")
        with open(results_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        self.logger.info(f"Confidence tuning results saved to: {results_path}")
    
    def _save_iou_tuning_results(self, results: List[Dict], optimal: Dict, confidence: float):
        """Save IoU tuning results to file."""
        output_data = {
            'fixed_confidence': confidence,
            'optimal_iou': optimal['iou'],
            'optimal_metrics': optimal,
            'all_results': results
        }
        
        results_path = os.path.join(self.output_dir, "iou_tuning.json")
        with open(results_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        self.logger.info(f"IoU tuning results saved to: {results_path}")
    
    def _plot_confidence_tuning(self, results: List[Dict], optimal: Dict, target_metric: str):
        """Create visualization for confidence threshold tuning."""
        conf_values = [r['confidence'] for r in results]
        recall_values = [r['recall'] for r in results]
        precision_values = [r['precision'] for r in results]
        f1_values = [r['f1'] for r in results]
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(conf_values, recall_values, 'b-', label='Recall')
        plt.axvline(x=optimal['confidence'], color='r', linestyle='--', label=f'Optimal ({optimal["confidence"]:.3f})')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Recall')
        plt.title('Recall vs Confidence Threshold')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(conf_values, precision_values, 'g-', label='Precision')
        plt.axvline(x=optimal['confidence'], color='r', linestyle='--', label=f'Optimal ({optimal["confidence"]:.3f})')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Precision')
        plt.title('Precision vs Confidence Threshold')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(conf_values, f1_values, 'm-', label='F1 Score')
        plt.axvline(x=optimal['confidence'], color='r', linestyle='--', label=f'Optimal ({optimal["confidence"]:.3f})')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('F1 Score')
        plt.title('F1 Score vs Confidence Threshold')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(conf_values, recall_values, 'b-', label='Recall')
        plt.plot(conf_values, precision_values, 'g-', label='Precision')
        plt.plot(conf_values, f1_values, 'm-', label='F1 Score')
        plt.axvline(x=optimal['confidence'], color='r', linestyle='--', label=f'Optimal ({optimal["confidence"]:.3f})')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Score')
        plt.title('All Metrics vs Confidence Threshold')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, f"confidence_tuning_{target_metric}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Confidence tuning plot saved to: {plot_path}")
    
    def _plot_iou_tuning(self, results: List[Dict], optimal: Dict):
        """Create visualization for IoU threshold tuning."""
        iou_values = [r['iou'] for r in results]
        recall_values = [r['recall'] for r in results]
        precision_values = [r['precision'] for r in results]
        f1_values = [r['f1'] for r in results]
        
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(iou_values, recall_values, 'b-', label='Recall')
        plt.plot(iou_values, precision_values, 'g-', label='Precision')
        plt.plot(iou_values, f1_values, 'm-', label='F1 Score')
        plt.axvline(x=optimal['iou'], color='r', linestyle='--', label=f'Optimal ({optimal["iou"]:.3f})')
        plt.xlabel('IoU Threshold')
        plt.ylabel('Score')
        plt.title('Metrics vs IoU Threshold')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(precision_values, recall_values, 'b-', linewidth=2)
        plt.scatter([optimal['precision']], [optimal['recall']], color='r', s=100, label=f'Optimal (IoU={optimal["iou"]:.3f})')
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, "iou_tuning.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"IoU tuning plot saved to: {plot_path}")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    tuner = ThresholdTuner(
        model_path="path/to/your/best_model.pt",
        data_yaml_path="data/data.yaml",
        output_dir="threshold_optimisation"
    )
    
    # Perform comprehensive threshold tuning
    optimal_config = tuner.comprehensive_threshold_tuning()
    
    print(f"Optimal configuration for reduced false negatives:")
    print(f"Confidence threshold: {optimal_config['confidence_threshold']:.3f}")
    print(f"IoU threshold: {optimal_config['iou_threshold']:.3f}")
    print(f"Expected recall: {optimal_config['expected_metrics']['recall']:.3f}")
