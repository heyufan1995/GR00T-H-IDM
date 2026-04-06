"""Shared Open-H temporal layout for GR00T-H-style training.

- **Video**: frames at anchor ``i`` and ``i + VIDEO_SPACING`` (``delta_indices=[0, 16]``).
- **Actions**: ``OPEN_H_ACTION_HORIZON`` steps with deltas ``0..H-1`` (targets at
  ``i .. i+H-1``), i.e. the chunk between the two video endpoints.
- **State** (proprio): remains ``delta_indices=[0]`` at the anchor for preprocessing
  (e.g. REL_XYZ_ROT6D). The policy model can ignore it when
  ``state_dropout_prob_per_embodiment`` is 1.0 in the training YAML.
"""

OPEN_H_ACTION_HORIZON: int = 16
VIDEO_SPACING: int = 16
OPEN_H_VIDEO_DELTA_INDICES: list[int] = [0, VIDEO_SPACING]
OPEN_H_ACTION_DELTA_INDICES: list[int] = list(range(OPEN_H_ACTION_HORIZON))
