"""MedbotWorld-style bimanual LeRobot modality configuration for GR00T N1.6 / GR00T-H.

This matches the Medbot LeRobot tutorial layout (GR00T-Dreams–style bimanual channels):
20D state/action (per arm: xyz, 6D rotation, jaw), one endoscope view, language via
``tasks.jsonl`` + ``task_index``.

Actions are modeled as **absolute** targets with ``temporal_meanstd`` normalization per
channel group (no REL_XYZ_ROT6D pipeline — rotations are already 6D in the dataset).
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

from open_h.embodiments.temporal_layout import (
    OPEN_H_ACTION_DELTA_INDICES,
    OPEN_H_VIDEO_DELTA_INDICES,
)

def _abs_temporal() -> ActionConfig:
    return ActionConfig(
        rep=ActionRepresentation.ABSOLUTE,
        type=ActionType.NON_EEF,
        format=ActionFormat.DEFAULT,
        normalization_type="temporal_meanstd",
    )


medbot_config = {
    "video": ModalityConfig(
        delta_indices=OPEN_H_VIDEO_DELTA_INDICES,
        modality_keys=[
            "left_endo_image",
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_cartesian",
            "left_rotation",
            "left_jaw",
            "right_cartesian",
            "right_rotation",
            "right_jaw",
        ],
        mean_std_embedding_keys=[
            "left_cartesian",
            "left_rotation",
            "left_jaw",
            "right_cartesian",
            "right_rotation",
            "right_jaw",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=OPEN_H_ACTION_DELTA_INDICES,
        modality_keys=[
            "left_cartesian",
            "left_rotation",
            "left_jaw",
            "right_cartesian",
            "right_rotation",
            "right_jaw",
        ],
        action_configs=[
            _abs_temporal(),
            _abs_temporal(),
            _abs_temporal(),
            _abs_temporal(),
            _abs_temporal(),
            _abs_temporal(),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["task"],
    ),
}

register_modality_config(medbot_config, embodiment_tag=EmbodimentTag.MEDBOT)
