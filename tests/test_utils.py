import os
import sys
import unittest
import tempfile
import shutil
import yaml
import json
import requests
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the parent directory to the path so we can import the modules under test
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.file_io import FileIO


class TestFileIO(unittest.TestCase):
    """Test the file I/O utility functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test."""
        self.test_dir = tempfile.mkdtemp()
        self.file_io = FileIO(base_dir=self.test_dir)
        
        # Create some test files
        self.yaml_path = os.path.join(self.test_dir, "test.yaml")
        self.json_path = os.path.join(self.test_dir, "test.json")
        
        # Create test YAML file
        test_yaml = {
            'section1': {
                'key1': 'value1',
                'key2': 42
            },
            'section2': ['item1', 'item2']
        }
        with open(self.yaml_path, 'w') as f:
            yaml.dump(test_yaml, f)
        
        # Create test JSON file
        test_json = {
            'name': 'test',
            'values': [1, 2, 3],
            'nested': {
                'inner': 'value'
            }
        }
        with open(self.json_path, 'w') as f:
            json.dump(test_json, f)
    
    def tearDown(self):
        """Clean up test fixtures after each test."""
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test initialization of FileIO."""
        self.assertEqual(self.file_io.base_dir, Path(self.test_dir))
    
    def test_resolve_path(self):
        """Test path resolution."""
        # Test with absolute path
        abs_path = os.path.abspath(self.yaml_path)
        resolved = self.file_io.resolve_path(abs_path)
        self.assertEqual(resolved, Path(abs_path))
        
        # Test with relative path
        rel_path = "test.yaml"
        resolved = self.file_io.resolve_path(rel_path)
        self.assertEqual(resolved, Path(self.test_dir) / rel_path)
        
        # Test with non-existent path
        non_existent = "non_existent.txt"
        with self.assertRaises(FileNotFoundError):
            self.file_io.resolve_path(non_existent)
        
        # Test with non-existent path but allow_nonexistent=True
        resolved = self.file_io.resolve_path(non_existent, allow_nonexistent=True)
        self.assertEqual(resolved, Path(self.test_dir) / non_existent)
    
    def test_create_directory(self):
        """Test directory creation."""
        # Test creating a new directory
        new_dir = os.path.join(self.test_dir, "new_dir")
        created_path = self.file_io.create_directory(new_dir)
        self.assertTrue(os.path.exists(new_dir))
        self.assertEqual(created_path, Path(new_dir))
        
        # Test creating an existing directory (should not error with exist_ok=True)
        created_path = self.file_io.create_directory(new_dir)
        self.assertEqual(created_path, Path(new_dir))
    
    def test_load_yaml(self):
        """Test loading YAML files."""
        # Test loading a valid YAML file
        data = self.file_io.load_yaml(self.yaml_path)
        self.assertIsInstance(data, dict)
        self.assertIn('section1', data)
        self.assertEqual(data['section1']['key1'], 'value1')
    
    def test_save_yaml(self):
        """Test saving YAML files."""
        # Test saving a new YAML file
        new_yaml_path = os.path.join(self.test_dir, "new.yaml")
        data = {'key': 'value', 'list': [1, 2, 3]}
        
        saved_path = self.file_io.save_yaml(data, new_yaml_path)
        self.assertTrue(os.path.exists(new_yaml_path))
        
        # Load and verify
        with open(new_yaml_path, 'r') as f:
            loaded = yaml.safe_load(f)
        
        self.assertEqual(loaded['key'], 'value')
    
    def test_load_json(self):
        """Test loading JSON files."""
        # Test loading a valid JSON file
        data = self.file_io.load_json(self.json_path)
        self.assertIsInstance(data, dict)
        self.assertEqual(data['name'], 'test')
        self.assertEqual(data['values'], [1, 2, 3])
    
    def test_save_json(self):
        """Test saving JSON files."""
        # Test saving a new JSON file
        new_json_path = os.path.join(self.test_dir, "new.json")
        data = {'key': 'value', 'list': [1, 2, 3]}
        
        saved_path = self.file_io.save_json(data, new_json_path)
        self.assertTrue(os.path.exists(new_json_path))
        
        # Load and verify
        with open(new_json_path, 'r') as f:
            loaded = json.load(f)
        
        self.assertEqual(loaded['key'], 'value')
    
    def test_substitute_env_vars(self):
        """Test environment variable substitution."""
        # Set a test environment variable
        os.environ['TEST_VAR'] = 'test_value'
        
        # Test ${VAR} syntax
        result = self.file_io._substitute_env_vars("Value: ${TEST_VAR}")
        self.assertEqual(result, "Value: test_value")
        
        # Test $VAR syntax
        result = self.file_io._substitute_env_vars("Value: $TEST_VAR")
        self.assertEqual(result, "Value: test_value")
        
        # Test missing variable
        result = self.file_io._substitute_env_vars("Value: ${MISSING_VAR}")
        self.assertEqual(result, "Value: ")
    
    def test_verify_directory_structure(self):
        """Test directory structure verification."""
        # Create test directories
        dir1 = os.path.join(self.test_dir, "dir1")
        dir2 = os.path.join(self.test_dir, "dir2")
        os.makedirs(dir1)
        
        # Test structure with one existing and one to be created
        required = [
            {'path': dir1, 'create': False},  # Exists
            {'path': dir2, 'create': True}    # Should be created
        ]
        
        result = self.file_io.verify_directory_structure(required)
        self.assertTrue(result)
        self.assertTrue(os.path.exists(dir2))  # Should be created
    
    @patch('requests.get')
    def test_download_file(self, mock_get):
        """Test file download functionality."""
        # Mock the requests.get response
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.headers.get.return_value = '1024'
        mock_response.iter_content.return_value = [b'test content']
        mock_get.return_value = mock_response
        
        # Test downloading a file
        url = 'https://example.com/file.txt'
        download_path = os.path.join(self.test_dir, "downloaded.txt")
        
        result = self.file_io.download_file(url, download_path)
        self.assertTrue(os.path.exists(download_path))
        
        # Check content
        with open(download_path, 'rb') as f:
            content = f.read()
        self.assertEqual(content, b'test content')
    
    def test_compute_file_hash(self):
        """Test file hash computation."""
        # Create a file with known content
        hash_file = os.path.join(self.test_dir, "hash_test.txt")
        with open(hash_file, 'w') as f:
            f.write("test content")
        
        # Compute MD5 hash
        md5_hash = self.file_io._compute_file_hash(hash_file, 'md5')
        self.assertIsInstance(md5_hash, str)
        self.assertEqual(len(md5_hash), 32)  # MD5 hash is 32 chars
        
        # Compute SHA1 hash
        sha1_hash = self.file_io._compute_file_hash(hash_file, 'sha1')
        self.assertIsInstance(sha1_hash, str)
        self.assertEqual(len(sha1_hash), 40)  # SHA1 hash is 40 chars
    
    def test_list_files(self):
        """Test file listing functionality."""
        # Create a directory structure with files
        test_subdir = os.path.join(self.test_dir, "subdir")
        os.makedirs(test_subdir)
        
        # Create files
        file1 = os.path.join(self.test_dir, "file1.txt")
        file2 = os.path.join(self.test_dir, "file2.jpg")
        file3 = os.path.join(test_subdir, "file3.txt")
        
        for f in [file1, file2, file3]:
            with open(f, 'w') as fh:
                fh.write("test")
        
        # Test listing all files
        files = self.file_io.list_files(self.test_dir)
        self.assertEqual(len(files), 5)  # 5 files total (including YAML and JSON from setUp)
        
        # Test with pattern
        txt_files = self.file_io.list_files(self.test_dir, pattern="*.txt")
        self.assertEqual(len(txt_files), 2)  # file1.txt and subdir/file3.txt
        
        # Test non-recursive
        root_files = self.file_io.list_files(self.test_dir, recursive=False)
        self.assertEqual(len(root_files), 4)  # 4 files in root dir


def create_test_image(width=640, height=640, channels=3):
    """
    Create a test image with consistent size.
    
    Args:
        width: Image width
        height: Image height
        channels: Number of channels (3 for RGB)
        
    Returns:
        numpy.ndarray: Test image
    """
    # Create a blank image
    img = np.zeros((height, width, channels), dtype=np.uint8)
    
    # Add some shapes to make it more realistic
    # Draw a red circle
    cv2.circle(img, (width // 2, height // 2), 50, (0, 0, 255), -1)
    # Draw a blue rectangle
    cv2.rectangle(img, (100, 100), (200, 200), (255, 0, 0), -1)
    # Draw a green triangle
    pts = np.array([[400, 100], [500, 100], [450, 200]], np.int32)
    cv2.fillPoly(img, [pts], (0, 255, 0))
    
    return img


def create_test_label(width=640, height=640, n_objects=2):
    """
    Create test YOLO format labels.
    
    Args:
        width: Image width
        height: Image height
        n_objects: Number of objects to generate
        
    Returns:
        list: YOLO format labels (class_id, x_center, y_center, w, h)
    """
    labels = []
    for i in range(n_objects):
        class_id = i % 5  # Use 5 classes (0-4)
        # Create normalized coordinates
        x_center = (i * 0.2) + 0.2  # Distribute across image
        y_center = 0.5
        w = 0.1
        h = 0.2
        labels.append([class_id, x_center, y_center, w, h])
    return labels


if __name__ == '__main__':
    unittest.main()
