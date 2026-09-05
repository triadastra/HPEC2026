"""
Device management utilities.
"""

import torch
from typing import Optional


def get_device(prefer_cuda: bool = True, cuda_id: Optional[int] = None) -> torch.device:
    """
    Get the best available device for computation.

    Args:
        prefer_cuda: Whether to prefer CUDA if available
        cuda_id: Specific CUDA device ID to use (if None, uses cuda:0)

    Returns:
        torch.device object
    """
    if prefer_cuda and torch.cuda.is_available():
        if cuda_id is not None:
            device = torch.device(f"cuda:{cuda_id}")
        else:
            device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Apple Silicon GPU support
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device


def get_device_name(device: torch.device) -> str:
    """
    Get human-readable device name.

    Args:
        device: torch.device object

    Returns:
        Device name string
    """
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    elif device.type == "mps":
        return "Apple Silicon GPU"
    else:
        return "CPU"


def get_memory_info(device: torch.device) -> dict:
    """
    Get memory information for device.

    Args:
        device: torch.device object

    Returns:
        Dictionary with memory info (total, reserved, allocated) in GB
    """
    if device.type == "cuda":
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        return {
            "total_gb": total,
            "reserved_gb": reserved,
            "allocated_gb": allocated,
            "free_gb": total - reserved,
        }
    else:
        return {"device": "cpu", "memory": "N/A"}


def clear_device_cache(device: torch.device) -> None:
    """
    Clear device cache (for CUDA/MPS).

    Args:
        device: torch.device object
    """
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def set_device_seed(device: torch.device, seed: int) -> None:
    """
    Set random seed for device-specific operations.

    Args:
        device: torch.device object
        seed: Random seed value
    """
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
