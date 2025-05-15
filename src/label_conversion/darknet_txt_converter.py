import os
import logging
from typing import Dict, List
import cv2
from PIL import Image

from .base_converter import BaseConverter

class DarknetTxtConverter(BaseConverter):
    """
    Converter for Darknet TXT annotation format.
    
    Handles conversion between Darknet TXT format and the standard label format.
    Darknet format uses a single text file per image with one line per object:
    <class_id> <x_center> <y_center> <width> <height>
    where all values are normalized to [0-1].
    """
    
    def __init__(self, standard_labels_path: str = None):
        """
        Initialise the Darknet TXT converter.
        
        Args:
            standard_labels_path: Path to the standard labels YAML file
        """
        super().__init__(standard_labels_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create mapping from class ID to name for easier lookups
        self.class_id_to_name = {}
        for class_id, class_info in self.standard_labels['classes'].items():
            self.class_id_to_name[int(class_id)] = class_info['name']
    
    def to_standard_format(self, input_path: str) -> List[Dict]:
        """
        Convert Darknet TXT label to the standard internal format.
        
        Args:
            input_path: Path to the Darknet TXT annotation file
            
        Returns:
            List of label dictionaries in the standard format
        """
        try:
            with open(input_path, 'r') as f:
                lines = f.readlines()
            
            standard_labels = []
            
            for line in lines:
                # Skip empty lines
                line = line.strip()
                if not line:
                    continue
                
                # Parse Darknet format: <class_id> <x_center> <y_center> <width> <height>
                parts = line.split()
                if len(parts) != 5:
                    self.logger.warning(f"Skipping invalid line in {input_path}: {line}")
                    continue
                
                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                except ValueError as e:
                    self.logger.warning(f"Skipping line with invalid values in {input_path}: {line}, {e}")
                    continue
                
                # Check if class_id is valid
                if class_id not in self.class_id_to_name:
                    self.logger.warning(f"Skipping unknown class_id {class_id} in {input_path}")
                    continue
                
                # Convert from center coordinates to top-left coordinates
                x_min = x_center - (width / 2)
                y_min = y_center - (height / 2)
                
                # Create the standard format label
                standard_label = {
                    'class_id': class_id,
                    'class_name': self.class_id_to_name[class_id],
                    'bbox': [x_min, y_min, width, height],
                    'confidence': 1.0  # Annotations typically don't have confidence scores
                }
                
                standard_labels.append(standard_label)
            
            return standard_labels
            
        except Exception as e:
            self.logger.error(f"Error converting Darknet TXT annotation {input_path}: {e}")
            raise
    
    def from_standard_format(self, labels: List[Dict], output_path: str) -> None:
        """
        Convert labels from the standard format to Darknet TXT format.
        
        Args:
            labels: List of label dictionaries in the standard format
            output_path: Path where to save the Darknet TXT annotation
        """
        try:
            # Verify the labels are in the standard format
            self.validate_standard_format(labels)
            
            # Convert each label to Darknet TXT format
            darknet_lines = []
            
            for label in labels:
                class_id = label['class_id']
                
                # Get bbox coordinates (x_min, y_min, width, height)
                x_min = label['bbox'][0]
                y_min = label['bbox'][1]
                width = label['bbox'][2]
                height = label['bbox'][3]
                
                # Convert to center coordinates for Darknet format
                x_center = x_min + (width / 2)
                y_center = y_min + (height / 2)
                
                # Create line in Darknet format: <class_id> <x_center> <y_center> <width> <height>
                line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                darknet_lines.append(line)
            
            # Write to output file
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write('\n'.join(darknet_lines))
                
            self.logger.info(f"Saved Darknet TXT annotation to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error converting to Darknet TXT format: {e}")
            raise
    
    def _get_output_filename(self, input_filename: str) -> str:
        """
        Generate appropriate output filename for Darknet format.
        
        Args:
            input_filename: Original filename
            
        Returns:
            Output filename with .txt extension
        """
        # If input has an extension, replace it with .txt
        base_name = os.path.splitext(input_filename)[0]
        return f"{base_name}.txt"
    
    def get_class_mapping_file(self, output_dir: str) -> None:
        """
        Create a class mapping file (names file) for Darknet.
        
        Args:
            output_dir: Directory where to save the class mapping file
        """
        try:
            class_names = []
            for class_id in sorted(self.class_id_to_name.keys()):
                class_names.append(self.class_id_to_name[class_id])
            
            output_path = os.path.join(output_dir, 'classes.names')
            with open(output_path, 'w') as f:
                f.write('\n'.join(class_names))
                
            self.logger.info(f"Saved class mapping file to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating class mapping file: {e}")
            raise
