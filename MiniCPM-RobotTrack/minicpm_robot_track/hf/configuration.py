from __future__ import annotations

from transformers import PretrainedConfig


class MiniCPMRobotTrackConfig(PretrainedConfig):
    """Hugging Face configuration for the published funnel tracking policy."""

    model_type = "minicpm_robot_track"

    def __init__(
        self,
        backbone_name: str = "openbmb/MiniCPM4-0.5B",
        vision_feature_dim: int = 1536,
        history_frames: int = 31,
        coarse_tokens_per_frame: int = 4,
        fine_tokens_current_frame: int = 64,
        num_waypoints: int = 8,
        action_dim: int = 3,
        max_text_tokens: int = 128,
        max_time_steps: int = 4096,
        trajectory_dropout: float = 0.4,
        xy_scale: float = 2.0,
        use_tanh_actions: bool = False,
        freeze_backbone: bool = False,
        gradient_checkpointing: bool = False,
        trust_remote_code: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.backbone_name = backbone_name
        self.vision_feature_dim = vision_feature_dim
        self.history_frames = history_frames
        self.coarse_tokens_per_frame = coarse_tokens_per_frame
        self.fine_tokens_current_frame = fine_tokens_current_frame
        self.num_waypoints = num_waypoints
        self.action_dim = action_dim
        self.max_text_tokens = max_text_tokens
        self.max_time_steps = max_time_steps
        self.trajectory_dropout = trajectory_dropout
        self.xy_scale = xy_scale
        self.use_tanh_actions = use_tanh_actions
        self.freeze_backbone = freeze_backbone
        self.gradient_checkpointing = gradient_checkpointing
        self.trust_remote_code = trust_remote_code
