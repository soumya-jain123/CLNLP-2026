"""
Hardware detection and resource-aware configuration utilities.
Inspects CPU, RAM, Disk, PyTorch, CUDA GPU, and mixed-precision support.
"""

import os
import shutil
from pathlib import Path
from typing import Any, Dict

import psutil
import torch


def detect_hardware(project_root: Path = Path(".")) -> Dict[str, Any]:
    """
    Inspect available compute hardware and return structured capability report.
    """
    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
    gpu_vram_gb = [
        round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2)
        for i in range(gpu_count)
    ]

    bf16_supported = bool(
        cuda_available and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
    )
    fp16_supported = bool(cuda_available)

    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(str(project_root))

    if cuda_available:
        device = "cuda"
        precision = "bf16" if bf16_supported else "fp16"
        recommended_batch_size = 16 if (gpu_vram_gb and gpu_vram_gb[0] >= 8.0) else 8
        runtime_note = f"CUDA GPU detected ({gpu_names[0]}, {gpu_vram_gb[0]} GB VRAM). Mixed precision ({precision}) enabled."
    else:
        device = "cpu"
        precision = "fp32"
        recommended_batch_size = 16
        runtime_note = (
            "No CUDA GPU detected (CPU-only PyTorch). Running in FP32 with dynamic padding, "
            "lightweight/distilled transformer backbones, and compact sequence lengths for feasible CPU runtime."
        )

    return {
        "pytorch_version": torch.__version__,
        "device": device,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda if cuda_available else None,
        "gpu_count": gpu_count,
        "gpu_names": gpu_names,
        "gpu_vram_gb": gpu_vram_gb,
        "fp16_supported": fp16_supported,
        "bf16_supported": bf16_supported,
        "recommended_precision": precision,
        "recommended_batch_size": recommended_batch_size,
        "cpu_cores_logical": os.cpu_count(),
        "ram_total_gb": round(vm.total / (1024**3), 2),
        "ram_available_gb": round(vm.available / (1024**3), 2),
        "disk_free_gb": round(disk.free / (1024**3), 2),
        "runtime_note": runtime_note,
    }
