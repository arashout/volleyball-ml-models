"""
Settings package for ML Manager.

This package contains all configuration classes for the ML Manager module.
"""

from settings.weights_config import ModelWeightsConfig
from settings.yolo_config import YOLOTrainingConfig
from settings.videomae_config import VideoMAETrainingConfig

__all__ = [
    "ModelWeightsConfig",
    "YOLOTrainingConfig", 
    "VideoMAETrainingConfig"
]
