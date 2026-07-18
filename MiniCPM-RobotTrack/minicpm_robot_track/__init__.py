"""MiniCPM-RobotTrack training and evaluation package."""

from .config import ModelConfig
from .modeling import MiniCPMRobotTrack

__all__ = ["MiniCPMRobotTrack", "ModelConfig"]
__version__ = "0.1.0"
