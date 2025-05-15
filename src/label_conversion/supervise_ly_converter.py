import os
import json
import logging
from typing import Dict, List, Any
import cv2
from PIL import Image

from .base_converter import BaseConverter

class SuperviseLyConverter(BaseConverter):
    """
    Converter for Supervise.ly annotation format.
    
    Handles conversion between Supervise.ly JSON format and the standard label format.
    The Supervise.ly format is structured as team/ann/imagename.png.json for annotations
    and team/img/imagename.png for images.
    """
    
    def __init__(self, standard_labels_path: str = None):
        """Initialize the Supervisely converter."""
        super().__init__(standard_labels_path)
        
        # Debug
        self.logger.info(f"SuperviseLyConverter initialized with standard_labels_path: {standard_labels_path}")
        self.logger.info(f"Standard labels keys: {list(self.standard_labels.keys())}")
        
        self.class_name_to_id = {}
        
        # Add name mapping for FSOCO dataset naming conventions
        self.class_name_mapping = {
            'blue_cone': 'blue',
            'yellow_cone': 'yellow',
            # ... existing mappings ...
        }
        
        # Create class name to ID mapping for easier lookups
        if 'classes' in self.standard_labels:
            self.logger.info(f"Classes in standard_labels: {list(self.standard_labels['classes'].keys())}")
            for class_id, class_info in self.standard_labels['classes'].items():
                if isinstance(class_info, dict) and 'name' in class_info:
                    self.class_name_to_id[class_info['name']] = int(class_id)
                else:
                    self.logger.error(f"Invalid class info format: {class_info}")
        else:
            self.logger.error("No 'classes' key in standard_labels!")
            
    def to_standard_format(self, input_path: str) -> List[Dict]:
        """
        Convert Supervise.ly JSON label to the standard internal format.
        
        Args:
            input_path: Path to the Supervise.ly JSON annotation file
            
        Returns:
            List of label dictionaries in the standard format
        """
        try:
            with open(input_path, 'r') as f:
                supervise_data = json.load(f)
                
            standard_labels = []
            
            # Get image dimensions - we'll need these to normalize coordinates
            img_width = supervise_data.get('size', {}).get('width')
            img_height = supervise_data.get('size', {}).get('height')
            
            # If sizes aren't in the JSON, try to get them from the actual image
            if not img_width or not img_height:
                # Find the corresponding image file
                img_path = self._get_image_path_from_annotation(input_path)
                if img_path and os.path.exists(img_path):
                    try:
                        img = Image.open(img_path)
                        img_width, img_height = img.size
                        img.close()
                    except Exception as e:
                        self.logger.warning(f"Failed to get image dimensions from {img_path}: {e}")
                        # Try with OpenCV as fallback
                        try:
                            img = cv2.imread(img_path)
                            img_height, img_width = img.shape[:2]
                        except Exception as e:
                            self.logger.error(f"Failed to get image dimensions with OpenCV: {e}")
                            raise ValueError(f"Cannot convert annotations without image dimensions")
            
            # Process each object in the Supervise.ly annotation
            for obj in supervise_data.get('objects', []):
                original_class_name = obj.get('classTitle')
                class_name = original_class_name

                # Apply mapping if available  
                if class_name in self.class_name_mapping:
                    mapped_name = self.class_name_mapping[class_name]
                    self.logger.debug(f"Mapping class name: {class_name} -> {mapped_name}")
                    class_name = mapped_name
                else:
                    self.logger.warning(f"No mapping found for class: {class_name}")
                
                # Skip if class is not in our standard labels
                if class_name not in self.class_name_to_id:
                    self.logger.warning(f"Skipping unknown class: {class_name}")
                    continue
                
                class_id = self.class_name_to_id[class_name]
                
                # Extract bbox information - Supervise.ly uses an array of points for the bbox
                points = obj.get('points', {}).get('exterior', [])
                if len(points) < 2:
                    self.logger.warning(f"Skipping object with insufficient points: {obj}")
                    continue
                    
                # Find the min/max coordinates to create a bounding box
                x_values = [p[0] for p in points]
                y_values = [p[1] for p in points]
                
                x_min = min(x_values)
                y_min = min(y_values)
                x_max = max(x_values)
                y_max = max(y_values)
                
                # Normalize coordinates to [0-1]
                x_min_norm = x_min / img_width
                y_min_norm = y_min / img_height
                width_norm = (x_max - x_min) / img_width
                height_norm = (y_max - y_min) / img_height
                
                # Create the standard format label
                standard_label = {
                    'class_id': class_id,
                    'class_name': class_name,
                    'bbox': [x_min_norm, y_min_norm, width_norm, height_norm],
                    'confidence': 1.0,  # Annotations typically don't have confidence scores
                    'attributes': obj.get('tags', {})  # Store any additional tags as attributes
                }
                
                standard_labels.append(standard_label)
                
            return standard_labels
            
        except Exception as e:
            self.logger.error(f"Error converting Supervise.ly annotation {input_path}: {e}")
            raise
            
    def from_standard_format(self, labels: List[Dict], output_path: str) -> None:
        """
        Convert labels from the standard format to Supervise.ly JSON format.
        
        Args:
            labels: List of label dictionaries in the standard format
            output_path: Path where to save the Supervise.ly JSON annotation
        """
        try:
            # Find the corresponding image file to get dimensions
            img_path = self._get_image_path_from_annotation(output_path)
            img_width, img_height = None, None
            
            if img_path and os.path.exists(img_path):
                try:
                    img = Image.open(img_path)
                    img_width, img_height = img.size
                    img.close()
                except Exception as e:
                    self.logger.warning(f"Failed to get image dimensions: {e}")
                    # Try OpenCV as fallback
                    try:
                        img = cv2.imread(img_path)
                        img_height, img_width = img.shape[:2]
                    except Exception as e:
                        self.logger.error(f"Failed to get image dimensions with OpenCV: {e}")
            
            if not img_width or not img_height:
                self.logger.warning("Image dimensions not available. Using defaults of 1920x1080")
                img_width, img_height = 1920, 1080
            
            # Create Supervise.ly JSON structure
            supervise_data = {
                'description': '',
                'tags': [],
                'size': {
                    'height': img_height,
                    'width': img_width
                },
                'objects': []
            }
            
            # Convert each standard label to Supervise.ly object
            for label in labels:
                # Get denormalized coordinates
                x_min = label['bbox'][0] * img_width
                y_min = label['bbox'][1] * img_height
                width = label['bbox'][2] * img_width
                height = label['bbox'][3] * img_height
                
                # Create a rectangular bounding box using exterior points
                exterior_points = [
                    [x_min, y_min],
                    [x_min + width, y_min],
                    [x_min + width, y_min + height],
                    [x_min, y_min + height]
                ]
                
                # Get the class name
                class_name = label['class_name']
                
                # Create the object in Supervise.ly format
                supervise_obj = {
                    'id': id(label),  # Generate a unique ID
                    'classId': label['class_id'],
                    'classTitle': class_name,
                    'tags': label.get('attributes', {}),
                    'description': '',
                    'geometryType': 'rectangle',
                    'points': {
                        'exterior': exterior_points,
                        'interior': []
                    }
                }
                
                supervise_data['objects'].append(supervise_obj)
            
            # Write to output file
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(supervise_data, f, indent=2)
                
            self.logger.info(f"Saved Supervise.ly annotation to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error converting to Supervise.ly format: {e}")
            raise
    
    def get_image_path_from_annotation(self, annotation_path: str) -> str:
        """
        Get the image file path from a Supervisely annotation path.
        
        For FSOCO dataset structure, annotations are stored in:
        - team/ann/image_file.jpg.json
        - or directly in team/image_file.jpg.json
        
        And images are in:
        - team/img/image_file.jpg
        - or directly in team/image_file.jpg
        
        Args:
            annotation_path: Path to Supervisely annotation file
            
        Returns:
            Path to the corresponding image file
        """
        try:
            # Get annotation directory and filename
            ann_dir = os.path.dirname(annotation_path)
            json_filename = os.path.basename(annotation_path)
            
            # Check if filename has .jpg.json or similar pattern
            if json_filename.endswith('.json'):
                # Remove .json extension
                img_filename = json_filename[:-5]
                
                # Check for standard FSOCO structure (team/ann/image.jpg.json)
                if os.path.basename(ann_dir).lower() == 'ann':
                    # FSOCO structure with annotations in 'ann' subfolder
                    team_dir = os.path.dirname(ann_dir)
                    
                    # Try standard structure first (team/img/image.jpg)
                    img_dir = os.path.join(team_dir, 'img')
                    if os.path.isdir(img_dir):
                        img_path = os.path.join(img_dir, img_filename)
                        if os.path.exists(img_path):
                            return img_path
                    
                    # Try alternative locations (directly in team dir)
                    img_path = os.path.join(team_dir, img_filename)
                    if os.path.exists(img_path):
                        return img_path
                else:
                    # Non-standard structure, try in the same directory as annotation
                    img_path = os.path.join(ann_dir, img_filename)
                    if os.path.exists(img_path):
                        return img_path
                    
                    # Try in a parallel 'img' directory
                    img_dir = os.path.join(os.path.dirname(ann_dir), 'img')
                    if os.path.isdir(img_dir):
                        img_path = os.path.join(img_dir, img_filename)
                        if os.path.exists(img_path):
                            return img_path
            
            # Could not determine image path
            self.logger.warning(f"Could not determine image path for annotation: {annotation_path}")
            return None
        except Exception as e:
            self.logger.error(f"Error getting image path: {e}")
            return None

    def _is_valid_file(self, filename: str) -> bool:
        """Check if this is a valid Supervisely annotation file."""
        return filename.endswith('.json')
