import os
import re
import yaml
import json
import shutil
import logging
import zipfile
import hashlib
import requests
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

class FileIO:
    """
    Utility class for file I/O operations.
    
    This class provides utilities for path management, configuration parsing,
    directory validation/creation, and dataset handling.
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialise the FileIO utility.
        
        Args:
            base_dir: Base directory for relative path resolution, defaults to current working directory
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Set base directory for relative paths
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.logger.debug(f"Base directory: {self.base_dir}")
    
    def resolve_path(self, path: Union[str, Path], allow_nonexistent: bool = False) -> Path:
        """
        Resolve a path relative to the base directory.
        
        Args:
            path: Path to resolve (can be absolute or relative)
            allow_nonexistent: Whether to allow the path to not exist
            
        Returns:
            Resolved absolute path
            
        Raises:
            FileNotFoundError: If the path does not exist and allow_nonexistent is False
        """
        if path is None:
            return self.base_dir
        
        path_obj = Path(path)
        
        # If path is absolute, use it directly
        if path_obj.is_absolute():
            resolved_path = path_obj
        else:
            # Resolve path relative to base directory
            resolved_path = (self.base_dir / path_obj).resolve()
        
        # Check if path exists
        if not resolved_path.exists() and not allow_nonexistent:
            raise FileNotFoundError(f"Path does not exist: {resolved_path}")
        
        return resolved_path
    
    def create_directory(self, path: Union[str, Path], exist_ok: bool = True) -> Path:
        """
        Create a directory if it does not exist.
        
        Args:
            path: Path to the directory
            exist_ok: Whether to ignore if the directory already exists
            
        Returns:
            Path to the created directory
        """
        dir_path = self.resolve_path(path, allow_nonexistent=True)
        os.makedirs(dir_path, exist_ok=exist_ok)
        self.logger.debug(f"Created directory: {dir_path}")
        return dir_path
    
    def load_yaml(self, path: Union[str, Path]) -> Dict:
        """
        Load a YAML file with environment variable substitution.
        
        Args:
            path: Path to the YAML file
            
        Returns:
            Dictionary containing the parsed YAML content
        """
        yaml_path = self.resolve_path(path)
        
        try:
            with open(yaml_path, 'r') as file:
                # Read the file content
                content = file.read()
                
                # Substitute environment variables
                content = self._substitute_env_vars(content)
                
                # Parse the YAML content
                config = yaml.safe_load(content)
                
                return config
        except Exception as e:
            self.logger.error(f"Error loading YAML file {yaml_path}: {e}")
            return {}
    
    def save_yaml(self, data: Dict, path: Union[str, Path], backup: bool = True) -> Path:
        """
        Save data to a YAML file.
        
        Args:
            data: Data to save
            path: Path to the output file
            backup: Whether to create a backup of existing file
            
        Returns:
            Path to the saved file
        """
        out_path = self.resolve_path(path, allow_nonexistent=True)
        
        # Create parent directories if they don't exist
        os.makedirs(out_path.parent, exist_ok=True)
        
        # Backup existing file
        if backup and out_path.exists():
            backup_path = Path(str(out_path) + '.bak')
            shutil.copy2(out_path, backup_path)
            self.logger.debug(f"Created backup: {backup_path}")
        
        # Save the file
        try:
            with open(out_path, 'w') as file:
                yaml.dump(data, file, default_flow_style=False, sort_keys=False)
            
            self.logger.debug(f"Saved YAML file: {out_path}")
            return out_path
            
        except Exception as e:
            self.logger.error(f"Error saving YAML file {out_path}: {e}")
            raise
    
    def load_json(self, path: Union[str, Path]) -> Dict:
        """
        Load a JSON file.
        
        Args:
            path: Path to the JSON file
            
        Returns:
            Dictionary containing the parsed JSON content
        """
        json_path = self.resolve_path(path)
        
        try:
            with open(json_path, 'r') as file:
                data = json.load(file)
                return data
        except Exception as e:
            self.logger.error(f"Error loading JSON file {json_path}: {e}")
            return {}
    
    def save_json(self, data: Dict, path: Union[str, Path], backup: bool = True) -> Path:
        """
        Save data to a JSON file.
        
        Args:
            data: Data to save
            path: Path to the output file
            backup: Whether to create a backup of existing file
            
        Returns:
            Path to the saved file
        """
        out_path = self.resolve_path(path, allow_nonexistent=True)
        
        # Create parent directories if they don't exist
        os.makedirs(out_path.parent, exist_ok=True)
        
        # Backup existing file
        if backup and out_path.exists():
            backup_path = Path(str(out_path) + '.bak')
            shutil.copy2(out_path, backup_path)
            self.logger.debug(f"Created backup: {backup_path}")
        
        # Save the file
        try:
            with open(out_path, 'w') as file:
                json.dump(data, file, indent=2)
            
            self.logger.debug(f"Saved JSON file: {out_path}")
            return out_path
            
        except Exception as e:
            self.logger.error(f"Error saving JSON file {out_path}: {e}")
            raise
    
    def _substitute_env_vars(self, content: str) -> str:
        """
        Substitute environment variables in a string.
        
        Args:
            content: String with environment variables
            
        Returns:
            String with substituted environment variables
        """
        # Match ${VAR} or $VAR patterns
        pattern = r'\${([^}]+)}|\$([a-zA-Z0-9_]+)'
        
        def replace_var(match):
            # Get variable name (from either group)
            var_name = match.group(1) or match.group(2)
            # Get value from environment or empty string if not found
            return os.environ.get(var_name, '')
        
        # Replace all occurrences
        return re.sub(pattern, replace_var, content)
    
    def verify_directory_structure(self, required_dirs: List[Dict]) -> bool:
        """
        Verify that the required directory structure exists.
        
        Args:
            required_dirs: List of dictionaries with 'path' and 'create' keys
            
        Returns:
            True if all required directories exist or were created, False otherwise
        """
        all_ok = True
        
        for dir_info in required_dirs:
            path = dir_info['path']
            should_create = dir_info.get('create', False)
            
            try:
                if should_create:
                    self.create_directory(path)
                else:
                    self.resolve_path(path)
                
                self.logger.debug(f"Verified directory: {path}")
                
            except FileNotFoundError:
                self.logger.error(f"Required directory does not exist: {path}")
                all_ok = False
        
        return all_ok
    
    def download_file(self, url: str, output_path: Union[str, Path], 
                     expected_hash: str = None, hash_type: str = 'md5') -> Path:
        """
        Download a file from a URL.
        
        Args:
            url: URL of the file to download
            output_path: Path where the file should be saved
            expected_hash: Expected hash of the file (for validation)
            hash_type: Type of hash to use ('md5', 'sha1', 'sha256')
            
        Returns:
            Path to the downloaded file
            
        Raises:
            ValueError: If the hash of the downloaded file does not match the expected hash
        """
        out_path = self.resolve_path(output_path, allow_nonexistent=True)
        
        # Create parent directories if they don't exist
        os.makedirs(out_path.parent, exist_ok=True)
        
        try:
            self.logger.info(f"Downloading {url} to {out_path}")
            
            # Download the file
            with requests.get(url, stream=True) as response:
                response.raise_for_status()  # Raise an exception for HTTP errors
                
                # Get total file size
                total_size = int(response.headers.get('content-length', 0))
                
                # Calculate chunk size (1MB)
                chunk_size = 1024 * 1024
                
                # Open the output file
                with open(out_path, 'wb') as f:
                    # Download in chunks to handle large files
                    for i, chunk in enumerate(response.iter_content(chunk_size=chunk_size)):
                        if chunk:
                            f.write(chunk)
                            
                            # Log progress for large files
                            if total_size > 0 and i % 10 == 0:
                                progress = (i * chunk_size / total_size) * 100
                                self.logger.debug(f"Download progress: {min(progress, 100):.1f}%")
            
            # Verify hash if provided
            if expected_hash:
                file_hash = self._compute_file_hash(out_path, hash_type)
                if file_hash != expected_hash:
                    raise ValueError(
                        f"Hash mismatch for downloaded file. "
                        f"Expected: {expected_hash}, got: {file_hash}"
                    )
                self.logger.debug(f"Verified {hash_type} hash: {file_hash}")
            
            self.logger.info(f"Successfully downloaded file to {out_path}")
            return out_path
            
        except Exception as e:
            self.logger.error(f"Error downloading {url}: {e}")
            
            # Remove partially downloaded file
            if out_path.exists():
                out_path.unlink()
                
            raise
    
    def _compute_file_hash(self, path: Union[str, Path], hash_type: str) -> str:
        """
        Compute the hash of a file.
        
        Args:
            path: Path to the file
            hash_type: Type of hash to use ('md5', 'sha1', 'sha256')
            
        Returns:
            Hex string representation of the hash
        """
        file_path = self.resolve_path(path)
        
        # Select hash algorithm
        if hash_type == 'md5':
            hash_func = hashlib.md5()
        elif hash_type == 'sha1':
            hash_func = hashlib.sha1()
        elif hash_type == 'sha256':
            hash_func = hashlib.sha256()
        else:
            raise ValueError(f"Unsupported hash type: {hash_type}")
        
        # Compute hash in chunks
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    def extract_archive(self, archive_path: Union[str, Path], output_dir: Union[str, Path]) -> Path:
        """
        Extract an archive file.
        
        Args:
            archive_path: Path to the archive file
            output_dir: Path to the output directory
            
        Returns:
            Path to the output directory
        """
        archive = self.resolve_path(archive_path)
        out_dir = self.resolve_path(output_dir, allow_nonexistent=True)
        
        # Create output directory if it doesn't exist
        os.makedirs(out_dir, exist_ok=True)
        
        try:
            self.logger.info(f"Extracting {archive} to {out_dir}")
            
            # Extract the archive
            with zipfile.ZipFile(archive, 'r') as zip_ref:
                zip_ref.extractall(out_dir)
            
            self.logger.info(f"Successfully extracted archive to {out_dir}")
            return out_dir
            
        except Exception as e:
            self.logger.error(f"Error extracting {archive}: {e}")
            raise
    
    def list_files(self, dir_path: Union[str, Path], pattern: str = None, recursive: bool = True) -> List[Path]:
        """
        List files in a directory.
        
        Args:
            dir_path: Path to the directory
            pattern: Glob pattern to match files
            recursive: Whether to search recursively
            
        Returns:
            List of paths to the files
        """
        path = self.resolve_path(dir_path)
        
        if pattern:
            if recursive:
                return list(path.glob(f"**/{pattern}"))
            else:
                return list(path.glob(pattern))
        else:
            if recursive:
                return [p for p in path.glob("**/*") if p.is_file()]
            else:
                return [p for p in path.glob("*") if p.is_file()]
    
    def prepare_dataset(self, config: Dict) -> Tuple[Path, Path]:
        """
        Prepare a dataset for training.
        
        Args:
            config: Dataset configuration
            
        Returns:
            Tuple of (images_dir, labels_dir)
        """
        dataset_dir = self.resolve_path(config.get('dataset_dir'), allow_nonexistent=True)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Check if dataset download is required
        if config.get('download', False):
            url = config.get('url')
            if not url:
                raise ValueError("Dataset URL not specified in configuration")
            
            # Download the dataset
            archive_path = dataset_dir / "dataset.zip"
            self.download_file(url, archive_path, expected_hash=config.get('hash'))
            
            # Extract the dataset
            self.extract_archive(archive_path, dataset_dir)
            
            # Delete the archive if specified
            if config.get('delete_archive', False):
                os.remove(archive_path)
        
        # Verify dataset structure
        images_dir = dataset_dir / config.get('images_dir', 'images')
        labels_dir = dataset_dir / config.get('labels_dir', 'labels')
        
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        if not labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
        
        # Log dataset statistics
        image_files = self.list_files(images_dir, pattern="*.jpg", recursive=True)
        label_files = self.list_files(labels_dir, pattern="*.txt", recursive=True)
        
        self.logger.info(f"Dataset prepared with {len(image_files)} images and {len(label_files)} labels")
        
        return images_dir, labels_dir


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    file_io = FileIO()
    
    # Example: Load configuration file
    config_path = file_io.resolve_path('config.yaml')
    if config_path.exists():
        config = file_io.load_yaml(config_path)
        print(f"Loaded configuration with {len(config)} keys")
    
    # Example: List files in a directory
    image_files = file_io.list_files('data', pattern="*.jpg", recursive=True)
    print(f"Found {len(image_files)} image files")
