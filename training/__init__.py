"""
Training package for ML Manager.

This package contains the Trainer class for training YOLO and VideoMAE models,
along with training utilities.
"""

from training.trainer import UnifiedTrainer
from . import utils

__all__ = ["UnifiedTrainer", "utils"]
