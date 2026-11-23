"""Unit tests for Phase 5 reproducibility, hardware detection, and config validation."""

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.hardware_detection import detect_hardware
from src.utils.reproducibility import set_global_seed


def test_reproducibility_seed():
    set_global_seed(42)
    a1 = np.random.rand(5)
    t1 = torch.rand(5)
    set_global_seed(42)
    a2 = np.random.rand(5)
    t2 = torch.rand(5)
    assert np.allclose(a1, a2)
    assert torch.allclose(t1, t2)


def test_hardware_detection_fields():
    hw = detect_hardware(PROJECT_ROOT)
    assert hw["device"] in ("cpu", "cuda")
    assert hw["cpu_cores_logical"] >= 1
    assert hw["disk_free_gb"] > 0
