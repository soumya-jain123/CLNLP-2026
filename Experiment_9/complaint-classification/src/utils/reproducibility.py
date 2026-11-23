"""
Reproducibility utilities for Phase 5 transformer experiments.
Configures seeds across Python, NumPy, PyTorch, and CUDA (when available).
"""

import os
import random
from typing import Dict

import numpy as np
import torch


def set_global_seed(seed: int = 42, deterministic_cudnn: bool = True) -> Dict[str, object]:
    """
    Set random seeds across Python, NumPy, and PyTorch for reproducible runs.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    return {
        "seed": seed,
        "cuda_available": cuda_available,
        "deterministic_cudnn": deterministic_cudnn if cuda_available else "N/A (CPU)",
    }
