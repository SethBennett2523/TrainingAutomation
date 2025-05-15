from abc import ABC, abstractmethod
import os
import yaml
import logging
from typing import Dict, List, Tuple, Any, Union

class BaseConverter(ABC):
    """
    Abstract base class for label format converters.
    
    Defines the interface that all format converters must implement to convert
    between their specific format and the standard internal format. This allows
    for modular conversion between any supported formats.
    """
    
    def __init__(self, standard_labels_path: str = None):
        """
        Initialise the base converter.
        
        Args:
            standard_labels_path: Path to the standard labels YAML file
        """
        # Set default path if not provided
        if standard_labels_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            standard_labels_path = os.path.join(current_dir, 'standard_labels.yaml')
        
        # Load standard label definitions
        self.standard_labels = self._load_standard_labels(standard_labels_path)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _load_standard_labels(self, path: str) -> Dict:
        """
        Load the standard label definitions from YAML file.
        
        Args:
            path: Path to the standard labels YAML file
            
        Returns:
            Dictionary containing standard label definitions
        """
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load standard labels from {path}: {e}")
    
    @abstractmethod
    def to_standard_format(self, input_path: str) -> List[Dict]:
        """
        Convert labels from the specific format to the standard internal format.
        
        Args:
            input_path: Path to the input label file or directory
            
        Returns:
            List of label dictionaries in the standard format
        """
        pass
    
    @abstractmethod
    def from_standard_format(self, labels: List[Dict], output_path: str) -> None:
        """
        Convert labels from the standard internal format to the specific format.
        
        Args:
            labels: List of label dictionaries in the standard format
            output_path: Path where to save the converted labels
        """
        pass
    
    def convert_directory(self, input_dir: str, output_dir: str, input_to_standard: bool = True) -> None:
        """
        Convert all label files in a directory.
        
        Args:
            input_dir: Directory containing input label files
            output_dir: Directory where to save converted label files
            input_to_standard: If True, convert from input format to standard format.
                              If False, convert from standard format to output format.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        for filename in os.listdir(input_dir):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, self._get_output_filename(filename))
            
            if input_to_standard:
                standard_labels = self.to_standard_format(input_path)
                self.from_standard_format(standard_labels, output_path)
            else:
                # Assumes input is already in standard format
                with open(input_path, 'r') as f:
                    standard_labels = yaml.safe_load(f)
                self.from_standard_format(standard_labels, output_path)
            
            self.logger.info(f"Converted {input_path} to {output_path}")
    
    def _get_output_filename(self, input_filename: str) -> str:
        """
        Generate appropriate output filename based on the input filename.
        Can be overridden by subclasses if needed.
        
        Args:
            input_filename: Original filename
            
        Returns:
            Output filename
        """
        # By default, keep the same name but subclasses may change extension
        return input_filename
    
    def validate_standard_format(self, labels: List[Dict]) -> bool:
        """
        Validate if the labels are in the correct standard format.
        
        Args:
            labels: List of label dictionaries to validate
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        required_keys = ['class_id', 'class_name', 'bbox']
        
        for label in labels:
            for key in required_keys:
                if key not in label:
                    raise ValueError(f"Missing required key '{key}' in label: {label}")
            
            # Validate class_id and class_name
            if label['class_id'] not in self.standard_labels['classes']:
                raise ValueError(f"Invalid class_id: {label['class_id']}")
            
            if label['class_name'] not in [c['name'] for c in self.standard_labels['classes'].values()]:
                raise ValueError(f"Invalid class_name: {label['class_name']}")
            
            # Validate bbox format (x_min, y_min, width, height)
            if len(label['bbox']) != 4:
                raise ValueError(f"Invalid bbox format, expected [x_min, y_min, width, height]: {label['bbox']}")
        
        return True
