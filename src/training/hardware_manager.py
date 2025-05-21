import os
import platform
import sys
import logging
import subprocess
import re
import torch
import psutil
import yaml
from typing import Dict, Optional, Tuple, Union, List
import time

class HardwareManager:
    """
    Hardware Manager for training optimization.
    
    This class handles the automatic detection of hardware capabilities (CUDA, ROCm, CPU)
    and optimization of training parameters based on the hardware configuration.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the hardware manager.
        
        Args:
            config_path: Path to the configuration file
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load configuration if provided
        self.config = self._load_config(config_path) if config_path else {}
        
        # Determine the operating system
        self.os_name = platform.system()
        self.logger.info(f"Operating system: {self.os_name}")
        
        # Detect hardware capabilities
        self.has_cuda = self._detect_cuda()
        self.has_rocm = self._detect_rocm()
        
        # Determine the device to use
        self.device = self._determine_device()
        
        # Get hardware details
        self.gpu_name = self._get_gpu_name()
        self.vram_total = self._get_vram_total()
        self.ram_total = psutil.virtual_memory().total
        
        self.logger.info(f"Selected device: {self.device}")
        if self.device != 'cpu':
            self.logger.info(f"GPU: {self.gpu_name}")
            self.logger.info(f"VRAM: {self.vram_total / (1024**3):.2f} GB")
        self.logger.info(f"RAM: {self.ram_total / (1024**3):.2f} GB")
    
    def _load_config(self, config_path: str) -> Dict:
        """
        Load the configuration from a YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Dictionary containing the configuration
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config
        except Exception as e:
            self.logger.error(f"Failed to load configuration from {config_path}: {e}")
            return {}
    
    def _detect_cuda(self) -> bool:
        """
        Detect if CUDA is available.
        
        Returns:
            True if CUDA is available, False otherwise
        """
        try:
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                self.logger.info(f"CUDA is available. CUDA version: {torch.version.cuda}")
                self.cuda_version = torch.version.cuda
                self.cuda_device_count = torch.cuda.device_count()
                self.logger.info(f"Found {self.cuda_device_count} CUDA device(s)")
            else:
                self.logger.info("CUDA is not available.")
                self.cuda_version = None
                self.cuda_device_count = 0
            return cuda_available
        except Exception as e:
            self.logger.warning(f"Error detecting CUDA: {e}")
            self.cuda_version = None
            self.cuda_device_count = 0
            return False
    
    def _detect_rocm(self) -> bool:
        """
        Detect if ROCm is available.
        
        Returns:
            True if ROCm is available, False otherwise
        """
        # Check if PyTorch was built with ROCm
        has_hip = hasattr(torch, 'hip') and torch.hip.is_available()
        
        if has_hip:
            self.logger.info("ROCm (HIP) is available through PyTorch.")
            return True
        
        # Try to detect ROCm through system commands
        try:
            if self.os_name == 'Linux':
                # Try to run rocm-smi
                rocm_process = subprocess.run(
                    ["rocm-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                if rocm_process.returncode == 0:
                    self.logger.info("ROCm is available (detected via rocm-smi).")
                    return True
            
            # Check if any AMD GPU is present on Windows
            elif self.os_name == 'Windows':
                # Use Windows Management Instrumentation (WMI) to detect AMD GPUs
                try:
                    import wmi
                    c = wmi.WMI()
                    for gpu in c.Win32_VideoController():
                        if 'AMD' in gpu.Name or 'Radeon' in gpu.Name:
                            self.logger.info(f"AMD GPU detected: {gpu.Name}")
                            return True
                except ImportError:
                    self.logger.warning("WMI module not available, cannot detect AMD GPUs on Windows")
        
        except Exception as e:
            self.logger.warning(f"Error detecting ROCm: {e}")
        
        self.logger.info("ROCm is not available.")
        return False
    
    def _determine_device(self) -> str:
        """
        Determine the device to use for training based on availability and configuration.
        
        Returns:
            Device string ('cuda', 'rocm', or 'cpu')
        """
        # Get the configured device preference
        device_preference = self.config.get('hardware', {}).get('device', 'auto')
        
        if device_preference == 'auto':
            # Auto-detect the best available device
            if self.has_cuda:
                return 'cuda'
            elif self.has_rocm:
                return 'rocm'
            else:
                return 'cpu'
        else:
            # Use the explicitly configured device if available
            if device_preference == 'cuda' and self.has_cuda:
                return 'cuda'
            elif device_preference == 'rocm' and self.has_rocm:
                return 'rocm'
            elif device_preference == 'cpu':
                return 'cpu'
            else:
                # Fallback to the best available device
                self.logger.warning(
                    f"Configured device '{device_preference}' is not available. "
                    f"Falling back to auto-detection."
                )
                return self._determine_device_auto()
    
    def _determine_device_auto(self) -> str:
        """
        Auto-detect the best available device.
        
        Returns:
            Device string ('cuda', 'rocm', or 'cpu')
        """
        if self.has_cuda:
            return 'cuda'
        elif self.has_rocm:
            return 'rocm'
        else:
            return 'cpu'
    
    def _get_gpu_name(self) -> Optional[str]:
        """
        Get the name of the GPU being used.
        
        Returns:
            Name of the GPU or None if no GPU is available
        """
        if self.device == 'cuda':
            try:
                # Get the name of the first CUDA device
                device_index = 0
                return torch.cuda.get_device_name(device_index)
            except Exception as e:
                self.logger.warning(f"Error getting CUDA device name: {e}")
                return "NVIDIA GPU (Unknown model)"
        
        elif self.device == 'rocm':
            try:
                # Try to get the AMD GPU name
                if self.os_name == 'Linux':
                    # Use rocm-smi to get the GPU name
                    rocm_process = subprocess.run(
                        ["rocm-smi", "--showproductname"], 
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                    )
                    if rocm_process.returncode == 0:
                        # Extract the GPU name from the output
                        match = re.search(r'GPU\[.*\]\s*:\s*(.+)', rocm_process.stdout)
                        if match:
                            return match.group(1).strip()
                
                elif self.os_name == 'Windows':
                    # Use WMI to get the AMD GPU name
                    try:
                        import wmi
                        c = wmi.WMI()
                        for gpu in c.Win32_VideoController():
                            if 'AMD' in gpu.Name or 'Radeon' in gpu.Name:
                                return gpu.Name
                    except ImportError:
                        pass
                
                return "AMD GPU (Unknown model)"
            
            except Exception as e:
                self.logger.warning(f"Error getting ROCm device name: {e}")
                return "AMD GPU (Unknown model)"
        
        return None  # No GPU available
    
    def _get_vram_total(self) -> int:
        """
        Get the total amount of VRAM in bytes.
        
        Returns:
            Total VRAM in bytes or 0 if no GPU is available
        """
        if self.device == 'cuda':
            try:
                device_index = 0
                return torch.cuda.get_device_properties(device_index).total_memory
            except Exception as e:
                self.logger.warning(f"Error getting CUDA VRAM: {e}")
                return 0
        
        elif self.device == 'rocm':
            try:
                if self.os_name == 'Linux':
                    # Use rocm-smi to get the VRAM
                    rocm_process = subprocess.run(
                        ["rocm-smi", "--showmeminfo", "vram"], 
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                    )
                    if rocm_process.returncode == 0:
                        # Extract the VRAM total from the output
                        match = re.search(r'vram\s+total.*?:\s+(\d+)\s+(\w+)', rocm_process.stdout, re.IGNORECASE)
                        if match:
                            value = int(match.group(1))
                            unit = match.group(2).upper()
                            
                            # Convert to bytes
                            if unit == 'MB':
                                return value * 1024 * 1024
                            elif unit == 'GB':
                                return value * 1024 * 1024 * 1024
                            else:
                                return value
                
                # Fallback for Windows or if rocm-smi fails
                # This is a rough estimate based on typical AMD GPUs
                # The RX7700S typically has around 12GB VRAM
                if 'RX7700S' in str(self.gpu_name):
                    return 12 * 1024 * 1024 * 1024  # 12GB in bytes
                
                # Generic fallback
                return 8 * 1024 * 1024 * 1024  # Assume 8GB VRAM as a safe default
            
            except Exception as e:
                self.logger.warning(f"Error getting ROCm VRAM: {e}")
                return 8 * 1024 * 1024 * 1024  # Default to 8GB
        
        return 0  # No GPU available
    
    def get_vram_usage(self) -> Tuple[int, int]:
        """
        Get the current VRAM usage.
        
        Returns:
            Tuple of (used VRAM in bytes, total VRAM in bytes)
        """
        if self.device == 'cuda':
            try:
                device_index = 0
                total = torch.cuda.get_device_properties(device_index).total_memory
                # Reserved memory is what PyTorch has already allocated
                reserved = torch.cuda.memory_reserved(device_index)
                # Allocated memory is what PyTorch is actually using
                allocated = torch.cuda.memory_allocated(device_index)
                # Used memory is what's actually being used
                used = allocated
                return used, total
            except Exception as e:
                self.logger.warning(f"Error getting CUDA VRAM usage: {e}")
                return 0, self.vram_total
        
        elif self.device == 'rocm':
            try:
                if self.os_name == 'Linux':
                    # Use rocm-smi to get the VRAM usage
                    rocm_process = subprocess.run(
                        ["rocm-smi", "--showmeminfo", "vram"], 
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                    )
                    if rocm_process.returncode == 0:
                        # Extract the VRAM used from the output
                        match = re.search(r'vram\s+used.*?:\s+(\d+)\s+(\w+)', rocm_process.stdout, re.IGNORECASE)
                        if match:
                            value = int(match.group(1))
                            unit = match.group(2).upper()
                            
                            # Convert to bytes
                            if unit == 'MB':
                                used = value * 1024 * 1024
                            elif unit == 'GB':
                                used = value * 1024 * 1024 * 1024
                            else:
                                used = value
                            
                            return used, self.vram_total
                
                # Fallback if we can't get actual usage
                return 0, self.vram_total
            
            except Exception as e:
                self.logger.warning(f"Error getting ROCm VRAM usage: {e}")
                return 0, self.vram_total
        
        return 0, 0  # No GPU available
    
    def calculate_optimal_batch_size(self, base_batch_size: int = 16, image_size: int = 640) -> int:
        """
        Calculate the optimal batch size based on available VRAM.
        
        Args:
            base_batch_size: Base batch size for reference
            image_size: Image size in pixels (assuming square images)
            
        Returns:
            Optimal batch size
        """
        # If batch size is explicitly set in config, use that
        configured_batch_size = self.config.get('hardware', {}).get('batch_size', 0)
        if configured_batch_size > 0:
            return configured_batch_size
        
        if self.device == 'cpu':
            # For CPU, use a small batch size to avoid memory issues
            ram_gb = self.ram_total / (1024**3)
            if ram_gb >= 32:
                return 8
            elif ram_gb >= 16:
                return 4
            else:
                return 2
        
        # For GPU, calculate based on VRAM
        vram_gb = self.vram_total / (1024**3)
        
        # YOLOv8m model size is roughly 90MB, but training requires more memory
        # Each image takes approximately (image_size^2 * 3 * 4 bytes) memory
        # We also need to consider memory for gradients, optimizer states, etc.
        
        # Basic heuristic based on VRAM
        if vram_gb >= 24:
            batch_size = 64
        elif vram_gb >= 16:
            batch_size = 32
        elif vram_gb >= 12:
            batch_size = 24
        elif vram_gb >= 8:
            batch_size = 16
        elif vram_gb >= 6:
            batch_size = 12
        elif vram_gb >= 4:
            batch_size = 8
        else:
            batch_size = 4
        
        # Adjust for image size (default is 640x640)
        image_factor = (image_size / 640) ** 2
        batch_size = int(batch_size / image_factor)
        
        # Ensure batch size is at least 1
        batch_size = max(1, batch_size)
        
        # Apply memory threshold from config
        memory_threshold = self.config.get('hardware', {}).get('memory_threshold', 0.85)
        batch_size = int(batch_size * memory_threshold)
        
        self.logger.info(f"Calculated optimal batch size: {batch_size}")
        return batch_size
    
    def calculate_optimal_workers(self) -> int:
        """
        Calculate the optimal number of data loader workers.
        
        Returns:
            Optimal number of workers
        """
        # If workers count is explicitly set in config, use that
        configured_workers = self.config.get('hardware', {}).get('workers', 0)
        if configured_workers > 0:
            return configured_workers
        
        # Use CPU count with some limitations
        cpu_count = psutil.cpu_count(logical=False)  # Physical cores only
        if cpu_count is None:
            cpu_count = psutil.cpu_count(logical=True)  # Logical cores as fallback
        
        if cpu_count is None:
            # If we can't determine CPU count, use a safe default
            return 2
        
        # Typically, using workers equal to CPU count is optimal
        # but we'll cap it to avoid excessive resource usage
        if self.device == 'cpu':
            # When training on CPU, use fewer workers
            workers = max(1, cpu_count // 2)
        else:
            # When training on GPU, use more workers
            workers = min(cpu_count, 8)  # Cap at 8 workers
        
        self.logger.info(f"Calculated optimal workers: {workers}")
        return workers
    
    def get_torch_device(self) -> torch.device:
        """
        Get the appropriate torch device.
        
        Returns:
            torch.device object for the selected device
        """
        if self.device == 'cuda':
            return torch.device('cuda')
        elif self.device == 'rocm':
            # PyTorch uses 'cuda' device even for ROCm when HIP is enabled
            if hasattr(torch, 'hip') and torch.hip.is_available():
                return torch.device('cuda')
            else:
                self.logger.warning("ROCm selected but not available in PyTorch. Falling back to CPU.")
                return torch.device('cpu')
        else:
            return torch.device('cpu')
    
    def get_training_params(self, image_size: int = 640, empirical: bool = False, data_path: str = None) -> Dict:
        """
        Get all hardware-optimized training parameters.
        
        Args:
            image_size: Image size for training
            empirical: Whether to perform empirical benchmarking
            data_path: Path to data.yaml file (required if empirical=True)
            
        Returns:
            Dictionary of optimized training parameters
        """
        if empirical and data_path:
            # Get empirically determined optimal parameters
            optimal = self.find_optimal_training_params(data_path, image_size)
            return {
                'device': self.get_torch_device(),
                'batch_size': optimal['batch_size'],
                'workers': optimal['workers'],
                'is_gpu': self.device != 'cpu',
                'device_type': self.device
            }
        else:
            # Use heuristic optimization
            return {
                'device': self.get_torch_device(),
                'batch_size': self.calculate_optimal_batch_size(image_size=image_size),
                'workers': self.calculate_optimal_workers(),
                'is_gpu': self.device != 'cpu',
                'device_type': self.device
            }
    
    def print_hardware_summary(self):
        """
        Print a summary of detected hardware and optimized parameters.
        """
        print("\nHardware Summary:")
        print("-----------------")
        print(f"Operating System: {self.os_name}")
        print(f"Device: {self.device}")
        
        if self.device != 'cpu':
            print(f"GPU: {self.gpu_name}")
            vram_gb = self.vram_total / (1024**3)
            print(f"VRAM: {vram_gb:.2f} GB")
        
        ram_gb = self.ram_total / (1024**3)
        print(f"RAM: {ram_gb:.2f} GB")
        
        training_params = self.get_training_params()
        print("\nOptimized Training Parameters:")
        print(f"Batch Size: {training_params['batch_size']}")
        print(f"Workers: {training_params['workers']}")
        print(f"Device: {training_params['device']}")
        print("-----------------\n")
    
    def benchmark_batch_size(self, 
                             data_path: str, 
                             img_size: int = 640, 
                             min_batch: int = 1, 
                             max_batch: int = 64, 
                             iterations: int = 3) -> Dict:
        """
        Benchmark different batch sizes and measure throughput and memory usage.
        
        Args:
            data_path: Path to data.yaml file
            img_size: Image size for training
            min_batch: Minimum batch size to test
            max_batch: Maximum batch size to test
            iterations: Number of iterations to run for each batch size
            
        Returns:
            Dictionary with optimal batch size and benchmark results
        """
        self.logger.info(f"Benchmarking batch sizes from {min_batch} to {max_batch} on {self.device}")
        
        # Import here to avoid circular imports
        try:
            from ultralytics import YOLO
        except ImportError:
            self.logger.error("Failed to import YOLO. Is ultralytics installed?")
            return {'optimal_batch': self.calculate_optimal_batch_size(image_size=img_size)}
        
        import torch
        
        results = []
        if self.device == 'cuda':
            max_memory = torch.cuda.get_device_properties(0).total_memory
        else:
            max_memory = 0
        
        # Determine batch sizes to test (use exponential growth for efficiency)
        if max_batch > 32:
            batch_sizes = [min_batch] + [b for b in range(2, 9, 2)] + [b for b in range(10, 33, 4)] + [b for b in range(36, max_batch + 1, 8)]
        else:
            batch_sizes = [b for b in range(min_batch, max_batch + 1, 2)]
        
        # Ensure max_batch is included
        if max_batch not in batch_sizes:
            batch_sizes.append(max_batch)
        
        for batch_size in batch_sizes:
            try:
                torch.cuda.empty_cache() if self.device == 'cuda' else None
                model = YOLO('yolov8n.yaml')  # Use small model for benchmarking
                
                self.logger.info(f"Testing batch size {batch_size}...")
                
                # Warm-up - FIXED: removed invalid parameters
                model.train(data=data_path, imgsz=img_size, epochs=1, batch=batch_size, 
                           device=self.device, verbose=False, val=False, plots=False, 
                           save=False)
                
                # Benchmark - FIXED: removed invalid parameters
                start_time = time.time()
                for i in range(iterations):
                    model.train(data=data_path, imgsz=img_size, epochs=1, batch=batch_size, 
                               device=self.device, verbose=False, val=False, plots=False, 
                               save=False)
                duration = (time.time() - start_time) / iterations
                
                # Memory usage
                if self.device == 'cuda':
                    used_memory = torch.cuda.max_memory_allocated()
                    memory_percent = 100 * used_memory / max_memory
                else:
                    used_memory = 0
                    memory_percent = 0
                
                # Calculate images per second (estimate from a small number of batches)
                images_per_sec = 100 * batch_size / duration  # Assuming ~100 batches in an epoch
                
                results.append({
                    'batch_size': batch_size,
                    'duration': duration,
                    'images_per_sec': images_per_sec,
                    'memory_used': used_memory,
                    'memory_percent': memory_percent
                })
                
                self.logger.info(f"Batch {batch_size}: {images_per_sec:.2f} img/s, {memory_percent:.2f}% VRAM")
                
                del model
                torch.cuda.empty_cache() if self.device == 'cuda' else None
                
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    self.logger.warning(f"Batch size {batch_size} caused OOM error.")
                    break
                else:
                    self.logger.error(f"Error with batch size {batch_size}: {e}")
                    break
        
        # Find optimal batch size (highest throughput with memory under 90%)
        valid_results = [r for r in results if r.get('memory_percent', 0) < 90]
        if valid_results:
            optimal = max(valid_results, key=lambda x: x['images_per_sec'])
            self.logger.info(f"Optimal batch size: {optimal['batch_size']} "
                           f"({optimal['images_per_sec']:.2f} img/s, "
                           f"{optimal['memory_percent']:.1f}% VRAM)")
            return {'optimal_batch': optimal['batch_size'], 'results': results}
        elif results:
            # If all results exceed 90% memory, choose the one with highest throughput
            optimal = max(results, key=lambda x: x['images_per_sec'])
            self.logger.info(f"All batch sizes use >90% VRAM. Best: {optimal['batch_size']}")
            return {'optimal_batch': optimal['batch_size'], 'results': results}
        else:
            self.logger.error("No valid results found")
            return {'optimal_batch': self.calculate_optimal_batch_size(image_size=img_size)}
    
    def benchmark_workers(self, 
                         data_path: str, 
                         img_size: int = 640, 
                         batch_size: int = None,
                         min_workers: int = 0, 
                         max_workers: int = None,
                         iterations: int = 3) -> Dict:
        """
        Benchmark different worker counts and measure throughput.
        
        Args:
            data_path: Path to data.yaml file
            img_size: Image size for training
            batch_size: Batch size to use for testing (if None, use calculated optimal)
            min_workers: Minimum worker count to test
            max_workers: Maximum worker count to test
            iterations: Number of iterations to run for each worker count
            
        Returns:
            Dictionary with optimal worker count and benchmark results
        """
        # Set default batch size if not provided
        if batch_size is None:
            batch_size = self.calculate_optimal_batch_size(image_size=img_size)
        
        # Set default max_workers if not provided
        if max_workers is None:
            cpu_count = psutil.cpu_count(logical=True)
            max_workers = min(cpu_count, 16) if cpu_count else 8
        
        self.logger.info(f"Benchmarking workers from {min_workers} to {max_workers} with batch size {batch_size}")
        
        # Import here to avoid circular imports
        try:
            from ultralytics import YOLO
            import torch
        except ImportError:
            self.logger.error("Failed to import YOLO. Is ultralytics installed?")
            return {'optimal_workers': self.calculate_optimal_workers()}
        
        results = []
        workers_list = list(range(min_workers, max_workers + 1, 1))
        
        for workers in workers_list:
            try:
                torch.cuda.empty_cache() if self.device == 'cuda' else None
                model = YOLO('yolov8n.yaml')  # Use small model for benchmarking
                
                self.logger.info(f"Testing workers={workers}...")
                
                # Warm-up
                model.train(data=data_path, imgsz=img_size, epochs=1, batch=batch_size, 
                            device=self.device, workers=workers, verbose=False, val=False, 
                            plots=False, save=False)
                
                # Benchmark
                start_time = time.time()
                for i in range(iterations):
                    model.train(data=data_path, imgsz=img_size, epochs=1, batch=batch_size, 
                               device=self.device, workers=workers, verbose=False, val=False, 
                               plots=False, save=False)
                duration = (time.time() - start_time) / iterations
                
                # Calculate images per second (estimate from a small number of batches)
                images_per_sec = 100 * batch_size / duration  # Assuming ~100 batches in an epoch
                
                results.append({
                    'workers': workers,
                    'duration': duration,
                    'images_per_sec': images_per_sec,
                })
                
                self.logger.info(f"Workers {workers}: {images_per_sec:.2f} img/s")
                
                del model
                torch.cuda.empty_cache() if self.device == 'cuda' else None
                
            except Exception as e:
                self.logger.error(f"Error with workers={workers}: {e}")
                continue
        
        # Find optimal worker count (highest throughput)
        if results:
            optimal = max(results, key=lambda x: x['images_per_sec'])
            self.logger.info(f"Optimal workers: {optimal['workers']} "
                           f"({optimal['images_per_sec']:.2f} img/s)")
            return {'optimal_workers': optimal['workers'], 'results': results}
        else:
            self.logger.error("No valid results found")
            return {'optimal_workers': self.calculate_optimal_workers()}
    
    def find_optimal_training_params(self, 
                               data_path: str, 
                               img_size: int = 640,
                               max_batch: int = 64,
                               max_workers: int = None,
                               save_results: bool = True,
                               target_model: str = 'yolov8m') -> Dict:
        """
        Find optimal training parameters through empirical benchmarking.
        
        Args:
            data_path: Path to data.yaml file
            img_size: Image size for training
            max_batch: Maximum batch size to test
            max_workers: Maximum worker count to test
            save_results: Whether to save benchmark results to a file
            target_model: Target model for training (used for batch size adjustment)
            
        Returns:
            Dictionary with optimal training parameters
        """
        start_time = time.time()
        self.logger.info("Starting empirical benchmarking to find optimal training parameters")
        
        # First find optimal batch size
        batch_results = self.benchmark_batch_size(data_path, img_size, max_batch=max_batch)
        optimal_batch = batch_results['optimal_batch']
        
        # Then find optimal worker count using the optimal batch size
        worker_results = self.benchmark_workers(data_path, img_size, batch_size=optimal_batch, max_workers=max_workers)
        optimal_workers = worker_results['optimal_workers']
          # Apply model size adjustment factor
        model_size_factors = {
            'yolov8n': 1.0,   # nano (baseline for benchmarking)
            'yolov8s': 0.75,  # small
            'yolov8m': 0.5,   # medium
            'yolov8l': 0.35,  # large
            'yolov8x': 0.25,  # xlarge
            'yolov11s': 0.75, # YOLOv11 small (similar to v8s)
            'yolov11m': 0.5,  # YOLOv11 medium (similar to v8m)
            'yolov11l': 0.35  # YOLOv11 large (similar to v8l)
        }
        
        # Get adjustment factor for target model, default to 0.5 if unknown
        size_factor = model_size_factors.get(target_model, 0.5)
        
        # Apply size factor to optimal batch size
        original_batch = optimal_batch
        optimal_batch = max(1, int(optimal_batch * size_factor))
        
        self.logger.info(f"Adjusted batch size from {original_batch} to {optimal_batch} for {target_model}")
        
        # Create result summary
        results = {
            'optimal_params': {
                'batch_size': optimal_batch,  # Use adjusted batch size
                'workers': optimal_workers,
                'img_size': img_size,
                'device': self.device,
                'original_batch': original_batch,  # Store original for reference
                'adjustment_factor': size_factor
            },
            'benchmarks': {
                'batch_sizes': batch_results.get('results', []),
                'workers': worker_results.get('results', [])
            },
            'hardware': {
                'device': self.device,
                'gpu_name': self.gpu_name,
                'vram_total_gb': self.vram_total / (1024**3) if self.vram_total else 0,
                'ram_total_gb': self.ram_total / (1024**3)
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save results if requested
        if save_results:
            import yaml
            import os
            output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_path = os.path.join(output_dir, 'optimal_training_params.yaml')
            try:
                with open(output_path, 'w') as f:
                    yaml.dump(results, f, default_flow_style=False)
                self.logger.info(f"Saved benchmark results to {output_path}")
            except Exception as e:
                self.logger.error(f"Failed to save benchmark results: {e}")
        
        elapsed = time.time() - start_time
        self.logger.info(f"Benchmarking completed in {elapsed:.1f}s. Optimal parameters: batch={optimal_batch}, workers={optimal_workers}")
        
        return results['optimal_params']


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize hardware manager
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yaml")
    
    if os.path.exists(config_path):
        manager = HardwareManager(config_path)
    else:
        manager = HardwareManager()
    
    # Print hardware summary
    manager.print_hardware_summary()
