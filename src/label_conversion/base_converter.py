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
    
    @abstractmethod
    def get_image_path_from_annotation(self, annotation_path: str) -> str:
        """
        Get the corresponding image path from an annotation path.
        
        Args:
            annotation_path: Path to the annotation file
            
        Returns:
            Path to the corresponding image file
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
            
            # Skip directories and non-relevant files
            if os.path.isdir(input_path) or not self._is_valid_file(filename):
                continue
                
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
    
    def _is_valid_file(self, filename: str) -> bool:
        """
        Check if the file is a valid annotation file for this converter.
        Can be overridden by subclasses to filter files.
        
        Args:
            filename: Filename to check
            
        Returns:
            True if the file is valid for this converter, False otherwise
        """
        # By default, accept all files, subclasses should override
        return True
    
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
        if not labels:
            return True  # Empty labels are valid (means no objects)
            
        required_keys = ['class_id', 'class_name', 'bbox']
        
        for label in labels:
            for key in required_keys:
                if key not in label:
                    raise ValueError(f"Missing required key '{key}' in label: {label}")
            
            # Validate bbox format (x_min, y_min, width, height)
            if len(label['bbox']) != 4:
                raise ValueError(f"Invalid bbox format, expected [x_min, y_min, width, height]: {label['bbox']}")
        
        return True
        
    def process_fsoco_dataset(
        self,
        input_dir: str,
        train_images_dir: str,
        train_labels_dir: str,
        val_images_dir: str,
        val_labels_dir: str,
        target_converter: 'BaseConverter',
        split_ratio: float = 0.2,
        safe_mode: bool = False  # Add this parameter
    ) -> Dict[str, int]:
        """
        Process a complete FSOCO dataset, converting annotations and copying images.
        
        Args:
            input_dir: Root directory of FSOCO dataset with team folders
            train_images_dir: Output directory for training images
            train_labels_dir: Output directory for training labels
            val_images_dir: Output directory for validation images
            val_labels_dir: Output directory for validation labels
            target_converter: Converter for the output format
            split_ratio: Ratio of validation set (0.0 to 1.0)
            safe_mode: If True, process files sequentially instead of using threads
            
        Returns:
            Dictionary with statistics about processed files
        """
        import glob
        import random
        import shutil
        import concurrent.futures
        
        self.logger.info(f"Processing FSOCO dataset at {input_dir}")
        
        # Find all team directories
        team_dirs = []
        for item in os.listdir(input_dir):
            if os.path.isdir(os.path.join(input_dir, item)) and not item.startswith('.'):
                team_dirs.append(item)
        
        if not team_dirs:
            self.logger.error(f"No team directories found in {input_dir}")
            return {"error": "No team directories found"}
        
        self.logger.info(f"Found {len(team_dirs)} team directories")
        
        # Collect all annotation files across teams
        all_annotations = []
        
        for team_dir in team_dirs:
            team_path = os.path.join(input_dir, team_dir)
            
            # Check for annotations - either in an ann subdirectory or directly
            ann_dirs = [os.path.join(team_path, 'ann'), team_path]
            
            for ann_dir in ann_dirs:
                if os.path.isdir(ann_dir):
                    json_files = glob.glob(os.path.join(ann_dir, "*.json"))
                    if json_files:
                        all_annotations.extend(json_files)
                        self.logger.info(f"Found {len(json_files)} annotations in {ann_dir}")
                        break
        
        if not all_annotations:
            self.logger.error("No annotations found in the dataset")
            return {"error": "No annotations found"}
        
        self.logger.info(f"Found total of {len(all_annotations)} annotations")
        
        # Determine train/validation split
        random.shuffle(all_annotations)
        split_idx = int(len(all_annotations) * (1 - split_ratio))
        train_annotations = all_annotations[:split_idx]
        val_annotations = all_annotations[split_idx:]
        
        self.logger.info(f"Split: {len(train_annotations)} training, {len(val_annotations)} validation annotations")
        
        # Create output directories
        for directory in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Process training annotations
        self.logger.info("Converting training annotations...")
        train_stats = self._process_annotation_batch(
            train_annotations, 
            train_images_dir,
            train_labels_dir,
            target_converter,
            safe_mode  # Pass safe_mode parameter
        )
        
        # Process validation annotations
        self.logger.info("Converting validation annotations...")
        val_stats = self._process_annotation_batch(
            val_annotations, 
            val_images_dir,
            val_labels_dir,
            target_converter,
            safe_mode  # Pass safe_mode parameter
        )
        
        # Combine statistics
        stats = {
            "train_processed": train_stats.get("processed", 0),
            "train_errors": train_stats.get("errors", 0),
            "val_processed": val_stats.get("processed", 0),
            "val_errors": val_stats.get("errors", 0),
            "total_annotations": len(all_annotations),
            "total_teams": len(team_dirs),
            "train_images": len(os.listdir(train_images_dir)),
            "val_images": len(os.listdir(val_images_dir))
        }
        
        self.logger.info(f"Processing complete: {stats}")
        return stats
    
    def _process_annotation_batch(
        self, 
        annotation_files: List[str], 
        output_img_dir: str,
        output_label_dir: str,
        target_converter: 'BaseConverter',
        safe_mode: bool = False  # Add this parameter
    ) -> Dict[str, int]:
        """
        Process a batch of annotation files with limited concurrency.
        
        Args:
            annotation_files: List of annotation files to process
            output_img_dir: Directory to save output images
            output_label_dir: Directory to save output labels
            target_converter: Converter for the target format
            safe_mode: If True, process files sequentially instead of using threads
            
        Returns:
            Dictionary with statistics
        """
        import concurrent.futures
        
        processed = 0
        errors = 0
        
        # Process in smaller batches to prevent resource exhaustion
        batch_size = 100
        max_workers = min(8, os.cpu_count() or 4)  # Limit worker count
        
        for i in range(0, len(annotation_files), batch_size):
            batch = annotation_files[i:i+batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}/{len(annotation_files)//batch_size + 1} ({len(batch)} files)")
            
            if safe_mode:
                # Process files sequentially
                for ann_file in batch:
                    success = self._process_single_annotation(
                        ann_file,
                        output_img_dir, 
                        output_label_dir,
                        target_converter
                    )
                    if success:
                        processed += 1
                    else:
                        errors += 1
                        
                    if (processed + errors) % 20 == 0:
                        self.logger.info(f"Processed {processed+errors}/{len(annotation_files)} annotations")
            else:
                # Process files concurrently
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = []
                    
                    for ann_file in batch:
                        future = executor.submit(
                            self._process_single_annotation,
                            ann_file,
                            output_img_dir, 
                            output_label_dir,
                            target_converter
                        )
                        futures.append(future)
                    
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            success = future.result(timeout=60)  # Add timeout
                            if success:
                                processed += 1
                            else:
                                errors += 1
                                
                            if (processed + errors) % 20 == 0:
                                self.logger.info(f"Processed {processed+errors}/{len(annotation_files)} annotations")
                        except concurrent.futures.TimeoutError:
                            errors += 1
                            self.logger.warning("A task timed out after 60 seconds")
                        except Exception as e:
                            errors += 1
                            self.logger.error(f"Error processing annotation: {e}")
            
            # Log progress after each batch
            self.logger.info(f"Batch complete: {processed+errors}/{len(annotation_files)} files processed ({processed} success, {errors} errors)")
        
        return {
            "processed": processed,
            "errors": errors
        }
    
    def _process_single_annotation(
        self, 
        ann_file: str, 
        output_img_dir: str,
        output_label_dir: str,
        target_converter: 'BaseConverter'
    ) -> bool:
        """
        Process a single annotation file.
        
        Args:
            ann_file: Path to the annotation file
            output_img_dir: Directory to save the output image
            output_label_dir: Directory to save the output label
            target_converter: Converter for the target format
            
        Returns:
            True if successful, False otherwise
        """
        import shutil
        
        try:
            # Convert from source format to standard format
            standard_labels = self.to_standard_format(ann_file)
            
            if standard_labels is None:
                self.logger.debug(f"No valid labels found in {ann_file}")
                return False
            
            # Get the corresponding image path
            img_path = self.get_image_path_from_annotation(ann_file)
            if not img_path or not os.path.exists(img_path):
                self.logger.warning(f"Image not found for {ann_file}, tried {img_path}")
                return False
            
            # Determine output file paths
            img_filename = os.path.basename(img_path)
            label_filename = os.path.splitext(img_filename)[0] + '.txt'
            
            output_img_path = os.path.join(output_img_dir, img_filename)
            output_label_path = os.path.join(output_label_dir, label_filename)
            
            # Copy image to output directory
            shutil.copy2(img_path, output_img_path)
            
            # Convert standard format to target format and save
            target_converter.from_standard_format(standard_labels, output_label_path)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing {ann_file}: {e}")
            return False
