import os
import json
import shutil
import unittest
import tempfile
from PIL import Image
import numpy as np

import sys
# Add the parent directory to the path so we can import the module under test
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.label_conversion.supervise_ly_converter import SuperviseLyConverter
from src.label_conversion.darknet_txt_converter import DarknetTxtConverter


class TestLabelConversion(unittest.TestCase):
    """Test the label conversion functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create test data directories to match expected structure
        self.team_dir = os.path.join(self.test_dir, "test_team")
        self.ann_dir = os.path.join(self.team_dir, "ann")
        self.img_dir = os.path.join(self.team_dir, "img")
        self.output_dir = os.path.join(self.test_dir, "output")
        
        os.makedirs(self.ann_dir, exist_ok=True)
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create a test image with known dimensions
        self.image_width = 640
        self.image_height = 480
        self.image_name = "test_image.png"
        self.image_path = os.path.join(self.img_dir, self.image_name)
        
        # Create a blank test image
        test_image = Image.new('RGB', (self.image_width, self.image_height), color='black')
        test_image.save(self.image_path)
        
        # Create test annotation data
        self.create_test_data()
        
        # Initialize converters
        self.supervise_converter = SuperviseLyConverter()
        self.darknet_converter = DarknetTxtConverter()
    
    def tearDown(self):
        """Clean up test fixtures after each test."""
        shutil.rmtree(self.test_dir)
    
    def create_test_data(self):
        """Create test annotation data in both formats."""
        # Create Supervise.ly JSON annotation
        supervise_annotation = {
            "description": "Test annotation",
            "tags": [],
            "size": {
                "height": self.image_height,
                "width": self.image_width
            },
            "objects": [
                {
                    "id": 1,
                    "classId": 0,
                    "classTitle": "blue",
                    "tags": {},
                    "description": "",
                    "geometryType": "rectangle",
                    "points": {
                        "exterior": [
                            [64, 48],  # x_min, y_min
                            [128, 48],  # x_max, y_min
                            [128, 96],  # x_max, y_max
                            [64, 96]    # x_min, y_max
                        ],
                        "interior": []
                    }
                },
                {
                    "id": 2,
                    "classId": 2,
                    "classTitle": "orange",
                    "tags": {},
                    "description": "",
                    "geometryType": "rectangle",
                    "points": {
                        "exterior": [
                            [320, 240],  # x_min, y_min
                            [384, 240],  # x_max, y_min
                            [384, 288],  # x_max, y_max
                            [320, 288]   # x_min, y_max
                        ],
                        "interior": []
                    }
                }
            ]
        }
        
        # Save Supervise.ly JSON annotation
        self.supervise_path = os.path.join(self.ann_dir, f"{self.image_name}.json")
        with open(self.supervise_path, 'w') as f:
            json.dump(supervise_annotation, f, indent=2)
        
        # Create corresponding Darknet TXT annotation
        # The values should match the normalized coordinates of the objects above
        # <class_id> <x_center> <y_center> <width> <height>
        darknet_lines = [
            "0 0.15 0.15 0.1 0.1",  # blue cone
            "2 0.55 0.55 0.1 0.1"   # orange cone
        ]
        
        # Save Darknet TXT annotation
        self.darknet_path = os.path.join(self.output_dir, f"{os.path.splitext(self.image_name)[0]}.txt")
        with open(self.darknet_path, 'w') as f:
            f.write('\n'.join(darknet_lines))
        
        # Standard format for reference
        self.standard_format_blue = {
            'class_id': 0,
            'class_name': 'blue',
            'bbox': [0.1, 0.1, 0.1, 0.1],  # x_min, y_min, width, height
            'confidence': 1.0
        }
        
        self.standard_format_orange = {
            'class_id': 2,
            'class_name': 'orange',
            'bbox': [0.5, 0.5, 0.1, 0.1],  # x_min, y_min, width, height
            'confidence': 1.0
        }
    
    def test_supervise_ly_to_standard(self):
        """Test conversion from Supervise.ly format to standard format."""
        standard_labels = self.supervise_converter.to_standard_format(self.supervise_path)
        
        # Verify that we got the expected number of labels
        self.assertEqual(len(standard_labels), 2)
        
        # Verify that the converted labels are as expected (with some tolerance for floating point)
        for label in standard_labels:
            if label['class_name'] == 'blue':
                self.assertEqual(label['class_id'], 0)
                self.assertAlmostEqual(label['bbox'][0], 0.1, delta=0.05)  # x_min
                self.assertAlmostEqual(label['bbox'][1], 0.1, delta=0.05)  # y_min
                self.assertAlmostEqual(label['bbox'][2], 0.1, delta=0.05)  # width
                self.assertAlmostEqual(label['bbox'][3], 0.1, delta=0.05)  # height
            elif label['class_name'] == 'orange':
                self.assertEqual(label['class_id'], 2)
                self.assertAlmostEqual(label['bbox'][0], 0.5, delta=0.05)  # x_min
                self.assertAlmostEqual(label['bbox'][1], 0.5, delta=0.05)  # y_min
                self.assertAlmostEqual(label['bbox'][2], 0.1, delta=0.05)  # width
                self.assertAlmostEqual(label['bbox'][3], 0.1, delta=0.05)  # height
            else:
                self.fail(f"Unexpected class name: {label['class_name']}")
    
    def test_standard_to_darknet(self):
        """Test conversion from standard format to Darknet TXT format."""
        standard_labels = [self.standard_format_blue, self.standard_format_orange]
        output_path = os.path.join(self.output_dir, "test_converted.txt")
        
        self.darknet_converter.from_standard_format(standard_labels, output_path)
        
        # Verify that the file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Read the contents and verify
        with open(output_path, 'r') as f:
            lines = f.readlines()
        
        self.assertEqual(len(lines), 2)
        
        # Parse the lines and verify the format
        for line in lines:
            parts = line.strip().split()
            self.assertEqual(len(parts), 5)  # class_id, x_center, y_center, width, height
            
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            if class_id == 0:  # blue
                self.assertAlmostEqual(x_center, 0.15, delta=0.01)
                self.assertAlmostEqual(y_center, 0.15, delta=0.01)
            elif class_id == 2:  # orange
                self.assertAlmostEqual(x_center, 0.55, delta=0.01)
                self.assertAlmostEqual(y_center, 0.55, delta=0.01)
            else:
                self.fail(f"Unexpected class ID: {class_id}")
    
    def test_darknet_to_standard(self):
        """Test conversion from Darknet TXT format to standard format."""
        standard_labels = self.darknet_converter.to_standard_format(self.darknet_path)
        
        # Verify that we got the expected number of labels
        self.assertEqual(len(standard_labels), 2)
        
        # Verify that the converted labels are as expected
        for label in standard_labels:
            if label['class_id'] == 0:  # blue
                self.assertEqual(label['class_name'], 'blue')
                # Check bbox values (x_min, y_min, width, height)
                self.assertAlmostEqual(label['bbox'][0], 0.1, delta=0.01)
                self.assertAlmostEqual(label['bbox'][1], 0.1, delta=0.01)
            elif label['class_id'] == 2:  # orange
                self.assertEqual(label['class_name'], 'orange')
                self.assertAlmostEqual(label['bbox'][0], 0.5, delta=0.01)
                self.assertAlmostEqual(label['bbox'][1], 0.5, delta=0.01)
            else:
                self.fail(f"Unexpected class ID: {label['class_id']}")
    
    def test_standard_to_supervise_ly(self):
        """Test conversion from standard format to Supervise.ly format."""
        standard_labels = [self.standard_format_blue, self.standard_format_orange]
        output_path = os.path.join(self.output_dir, "test_converted.json")
        
        self.supervise_converter.from_standard_format(standard_labels, output_path)
        
        # Verify that the file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Load and verify the JSON content
        with open(output_path, 'r') as f:
            supervise_data = json.load(f)
        
        self.assertEqual(len(supervise_data['objects']), 2)
        
        # Check the objects
        for obj in supervise_data['objects']:
            if obj['classTitle'] == 'blue':
                self.assertEqual(obj['classId'], 0)
                exterior_points = obj['points']['exterior']
                # Check that it's a rectangle with 4 points
                self.assertEqual(len(exterior_points), 4)
            elif obj['classTitle'] == 'orange':
                self.assertEqual(obj['classId'], 2)
            else:
                self.fail(f"Unexpected class title: {obj['classTitle']}")
    
    def test_full_conversion_pipeline(self):
        """Test the full conversion pipeline: Supervise.ly -> standard -> Darknet TXT."""
        # Step 1: Convert from Supervise.ly to standard format
        standard_labels = self.supervise_converter.to_standard_format(self.supervise_path)
        
        # Step 2: Convert from standard format to Darknet TXT
        output_path = os.path.join(self.output_dir, "full_pipeline_test.txt")
        self.darknet_converter.from_standard_format(standard_labels, output_path)
        
        # Verify the final output
        self.assertTrue(os.path.exists(output_path))
        
        # Read the contents and verify
        with open(output_path, 'r') as f:
            lines = f.readlines()
        
        self.assertEqual(len(lines), 2)
    
    def test_class_mapping_file(self):
        """Test creation of class mapping file for Darknet."""
        self.darknet_converter.get_class_mapping_file(self.output_dir)
        
        # Check that the file was created
        mapping_file_path = os.path.join(self.output_dir, 'classes.names')
        self.assertTrue(os.path.exists(mapping_file_path))
        
        # Check the content of the mapping file
        with open(mapping_file_path, 'r') as f:
            lines = f.readlines()
        
        expected_classes = ['blue', 'big_orange', 'orange', 'unknown', 'yellow']
        self.assertEqual(len(lines), len(expected_classes))
        
        for i, expected_class in enumerate(expected_classes):
            self.assertEqual(lines[i].strip(), expected_class)
    
    def test_error_handling(self):
        """Test error handling for invalid input."""
        # Test with non-existent file
        with self.assertRaises(Exception):
            self.supervise_converter.to_standard_format("nonexistent_file.json")
        
        # Test with invalid format (provide a file that exists but is not in the correct format)
        invalid_format_path = os.path.join(self.output_dir, "invalid_format.json")
        with open(invalid_format_path, 'w') as f:
            f.write("This is not valid JSON")
        
        with self.assertRaises(Exception):
            self.supervise_converter.to_standard_format(invalid_format_path)


if __name__ == '__main__':
    unittest.main()
