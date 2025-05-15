"""
Label conversion module for FSOCO dataset.

This module provides functionality for converting between different annotation formats
using a standard intermediate representation. It handles the specific structure of
FSOCO datasets where annotations are organized by team.
"""

import os
import logging
from typing import Dict, List, Any

from .supervise_ly_converter import SuperviseLyConverter
from .darknet_txt_converter import DarknetTxtConverter

def convert_fsoco_dataset(
    source_format: str,
    target_format: str,
    input_dir: str,
    train_images_dir: str,
    train_labels_dir: str,
    val_images_dir: str,
    val_labels_dir: str,
    classes: str,
    split_ratio: float = 0.2
) -> Dict[str, Any]:
    """
    Convert FSOCO dataset from source format to target format.
    
    Args:
        source_format: Source annotation format (e.g., 'supervisely')
        target_format: Target annotation format (e.g., 'darknet')
        input_dir: Root directory of FSOCO dataset with team folders
        train_images_dir: Output directory for training images
        train_labels_dir: Output directory for training labels
        val_images_dir: Output directory for validation images
        val_labels_dir: Output directory for validation labels
        classes: Comma-separated list of class names
        split_ratio: Ratio of validation images (0.0 to 1.0)
        
    Returns:
        Dictionary with statistics about the conversion process
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Converting FSOCO dataset from {source_format} to {target_format}")
    
    # Initialize converters
    source_converter = _get_converter(source_format)
    target_converter = _get_converter(target_format)
    
    if not source_converter or not target_converter:
        error_msg = f"Unsupported format combination: {source_format} to {target_format}"
        logger.error(error_msg)
        return {"error": error_msg}
    
    # Set up class names
    class_names = classes.split(',')
    logger.info(f"Using classes: {class_names}")
    
    if hasattr(source_converter, 'class_name_to_id'):
        for i, name in enumerate(class_names):
            source_converter.class_name_to_id[name] = i
            logger.debug(f"Mapped class {name} to ID {i} for source converter")
    
    if hasattr(target_converter, 'class_id_to_name'):
        for i, name in enumerate(class_names):
            target_converter.class_id_to_name[i] = name
            logger.debug(f"Mapped ID {i} to class {name} for target converter")
    
    # Write classes.txt file for Darknet format
    if target_format.lower() == 'darknet':
        for labels_dir in [train_labels_dir, val_labels_dir]:
            classes_file = os.path.join(labels_dir, 'classes.names')
            os.makedirs(labels_dir, exist_ok=True)
            with open(classes_file, 'w') as f:
                f.write('\n'.join(class_names))
            logger.info(f"Created classes file at {classes_file}")
    
    # Process the dataset
    stats = source_converter.process_fsoco_dataset(
        input_dir=input_dir,
        train_images_dir=train_images_dir,
        train_labels_dir=train_labels_dir,
        val_images_dir=val_images_dir,
        val_labels_dir=val_labels_dir,
        target_converter=target_converter,
        split_ratio=split_ratio
    )
    
    return stats


def _get_converter(format_name: str):
    """Get the appropriate converter instance for the specified format."""
    logger = logging.getLogger(__name__)
    
    format_name = format_name.lower()
    
    if format_name == 'supervisely':
        return SuperviseLyConverter()
    elif format_name == 'darknet':
        return DarknetTxtConverter()
    else:
        logger.error(f"Unsupported format: {format_name}")
        return None


def convert_annotations(
    source_format: str,
    target_format: str,
    input_dir: str,
    output_dir: str,
    classes: str,
    split: str = None
) -> None:
    """
    Legacy function for simple annotation conversion (without splitting).
    
    Args:
        source_format: Source annotation format
        target_format: Target annotation format
        input_dir: Input directory with annotations
        output_dir: Output directory for converted annotations
        classes: Comma-separated list of class names
        split: Optional split name ('train' or 'val')
    """
    logger = logging.getLogger(__name__)
    logger.warning("convert_annotations is deprecated, use convert_fsoco_dataset")
    
    # Initialize converters
    source_converter = _get_converter(source_format)
    target_converter = _get_converter(target_format)
    
    if not source_converter or not target_converter:
        logger.error(f"Unsupported format combination: {source_format} to {target_format}")
        return
    
    # Set up class names
    class_names = classes.split(',')
    for i, name in enumerate(class_names):
        if hasattr(source_converter, 'class_name_to_id'):
            source_converter.class_name_to_id[name] = i
        if hasattr(target_converter, 'class_id_to_name'):  
            target_converter.class_id_to_name[i] = name
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Write classes.txt file for Darknet format
    if target_format.lower() == 'darknet':
        with open(os.path.join(output_dir, 'classes.names'), 'w') as f:
            f.write('\n'.join(class_names))
    
    # Find annotation files
    if os.path.isdir(input_dir):
        source_converter.convert_directory(input_dir, output_dir, input_to_standard=True)
    else:
        logger.error(f"Input directory does not exist: {input_dir}")
