"""
Reproducibility Utilities

Ensures deterministic training for exact reproducibility.
"""

import os
import random
import platform
import subprocess
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
        deterministic: Whether to use deterministic algorithms (slower but reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        # Enable deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set deterministic algorithms (PyTorch 1.8+)
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True)
            except RuntimeError:
                # Some operations don't have deterministic implementations
                os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
                torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_reproducibility_info() -> Dict[str, Any]:
    """
    Get system information for reproducibility tracking.
    
    Returns:
        Dictionary with version info, hardware info, etc.
    """
    info = {
        # Python
        "python_version": platform.python_version(),
        
        # PyTorch
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        
        # Hardware
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    
    # GPU info
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_names"] = [
            torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())
        ]
    
    # Git info
    try:
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        info["git_hash"] = git_hash
        
        git_diff = subprocess.check_output(
            ['git', 'diff', '--stat'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        info["has_uncommitted_changes"] = len(git_diff) > 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        info["git_hash"] = None
    
    return info


def get_pip_freeze() -> str:
    """Get pip freeze output for dependency tracking."""
    try:
        result = subprocess.check_output(
            ['pip', 'freeze'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8')
        return result
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


class ReproducibilityManager:
    """
    Manager for ensuring reproducibility across experiments.
    """
    
    def __init__(self, seed: int, output_dir: str):
        self.seed = seed
        self.output_dir = output_dir
        self.info = get_reproducibility_info()
        
        # Set seed
        set_seed(seed, deterministic=True)
    
    def save_info(self):
        """Save reproducibility information to output directory."""
        import json
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save system info
        with open(os.path.join(self.output_dir, 'reproducibility.json'), 'w') as f:
            json.dump(self.info, f, indent=2)
        
        # Save pip freeze
        with open(os.path.join(self.output_dir, 'requirements_snapshot.txt'), 'w') as f:
            f.write(get_pip_freeze())
    
    def verify_reproducibility(self, other_info: Dict) -> bool:
        """
        Verify that current environment matches saved info.
        
        Args:
            other_info: Previously saved reproducibility info
            
        Returns:
            True if environments match
        """
        critical_keys = [
            'torch_version',
            'cuda_version',
            'cudnn_version',
        ]
        
        for key in critical_keys:
            if self.info.get(key) != other_info.get(key):
                print(f"Warning: {key} mismatch: {self.info.get(key)} vs {other_info.get(key)}")
                return False
        
        return True


def worker_init_fn(worker_id: int):
    """
    Initialize worker with deterministic seed.
    
    Use as DataLoader worker_init_fn.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
