from transformers import AutoConfig, AutoModel

from .configuration import MiniCPMRobotTrackConfig
from .modeling import MiniCPMRobotTrackForWaypoint

__all__ = ["MiniCPMRobotTrackConfig", "MiniCPMRobotTrackForWaypoint"]


def _register_with_auto() -> None:
    try:
        AutoConfig.register(MiniCPMRobotTrackConfig.model_type, MiniCPMRobotTrackConfig)
    except ValueError:
        pass
    try:
        AutoModel.register(MiniCPMRobotTrackConfig, MiniCPMRobotTrackForWaypoint)
    except ValueError:
        pass


_register_with_auto()
