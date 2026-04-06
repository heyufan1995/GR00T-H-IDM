# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
LeRobot IDM-style evaluation for GR00T N1.6 / GR00T-H checkpoints.

Runs open-loop action prediction on a LeRobot dataset (typically a held-out test set),
reusing :func:`gr00t.eval.open_loop_eval.evaluate_single_trajectory` and
:class:`gr00t.policy.gr00t_policy.Gr00tPolicy` — the same stack as ``open_loop_eval.py``.

This mirrors the workflow of ``GR00T-Dreams/scripts/idm_inference_simple.py`` (MSE/MAE on
GT actions in dataset space) without duplicating the policy or loader logic.

Example:

.. code-block:: bash

    uv run python gr00t/eval/idm_lerobot_eval.py \\
        --checkpoint /path/to/checkpoint-20000 \\
        --dataset /path/to/test_lerobot \\
        --output-dir /path/to/idm_eval_out \\
        --embodiment-tag NEW_EMBODIMENT \\
        --include-splits test

Finetune your downstream dataset with ``gr00t/experiment/launch_finetune.py`` first; use the
same ``--embodiment-tag`` and ensure the test set has ``meta/stats.json`` (and temporal stats
if required by your training recipe).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import tyro

# Register Open-H embodiment configs when present (optional for core-only tags).
try:
    import open_h.embodiments  # noqa: F401
except ImportError:
    pass

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.split_utils import load_info_json, resolve_episode_indices
from gr00t.eval.open_loop_eval import evaluate_single_trajectory
from gr00t.policy.gr00t_policy import Gr00tPolicy


@dataclass
class ArgsConfig:
    """IDM-style LeRobot eval (predicted vs dataset GT actions, unnormalized MSE/MAE)."""

    checkpoint: str
    """Path to finetuned model checkpoint directory (HuggingFace-style folder)."""

    dataset: str
    """Path to LeRobot dataset root (must contain meta/stats.json)."""

    output_dir: str
    """Directory for ``inference_summary.json`` and optional trajectory plots."""

    embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
    """Must match the tag used during finetuning."""

    steps: int = 10_000
    """Max steps per trajectory (capped by episode length in the loader)."""

    action_horizon: int = 16
    """Action chunk horizon; must match training / checkpoint."""

    traj_ids: list[int] | None = None
    """If set, evaluate only these trajectory indices (overrides split filtering)."""

    include_splits: list[str] | None = None
    """Episode indices from ``meta/info.json`` splits (e.g. ``test``). Ignored if ``traj_ids`` is set."""

    exclude_splits: list[str] | None = None
    """Exclude these splits when building the episode list from info.json."""

    max_episodes: int | None = None
    """After resolving the episode list, keep at most this many (order preserved)."""

    modality_keys: list[str] | None = None
    """Subset of action keys to concatenate for metrics; default = all action keys."""

    video_backend: str = "torchcodec"
    """Video backend passed to :class:`~gr00t.data.dataset.lerobot_episode_loader.LeRobotEpisodeLoader`."""

    no_plots: bool = False
    """If set, skip matplotlib trajectory plots (faster for large test sets)."""

    device: str | None = None
    """Device for the policy (default: cuda if available else cpu)."""


def _resolve_traj_ids(args: ArgsConfig, dataset_path: Path, num_episodes: int) -> list[int]:
    if args.traj_ids is not None:
        out = list(args.traj_ids)
    elif args.include_splits or args.exclude_splits:
        info = load_info_json(dataset_path)
        total = int(info.get("total_episodes", num_episodes))
        resolved = resolve_episode_indices(
            info,
            include_splits=args.include_splits or None,
            exclude_splits=args.exclude_splits or None,
            total_episodes=total,
        )
        assert resolved is not None
        out = [int(i) for i in resolved.tolist()]
    else:
        out = list(range(num_episodes))

    if args.max_episodes is not None:
        out = out[: args.max_episodes]
    return out


def main(args: ArgsConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    policy = Gr00tPolicy(
        embodiment_tag=args.embodiment_tag,
        model_path=args.checkpoint,
        device=device,
    )
    modality = policy.get_modality_config()

    dataset_path = Path(args.dataset)
    loader = LeRobotEpisodeLoader(
        dataset_path=args.dataset,
        modality_configs=modality,
        video_backend=args.video_backend,
        video_backend_kwargs=None,
    )

    traj_ids = _resolve_traj_ids(args, dataset_path, len(loader))
    per_episode: list[dict[str, float | int]] = []
    all_mse: list[float] = []
    all_mae: list[float] = []

    for traj_id in traj_ids:
        if traj_id < 0 or traj_id >= len(loader):
            logging.warning("Skipping out-of-range traj_id=%s (dataset has %s episodes)", traj_id, len(loader))
            continue
        plot_path = None if args.no_plots else str(out / f"traj_{traj_id}.jpeg")
        mse, mae = evaluate_single_trajectory(
            policy,
            loader,
            traj_id,
            args.embodiment_tag,
            args.modality_keys,
            steps=args.steps,
            action_horizon=args.action_horizon,
            save_plot_path=plot_path,
            save_plots=not args.no_plots,
        )
        all_mse.append(float(mse))
        all_mae.append(float(mae))
        per_episode.append({"traj_id": traj_id, "mse": float(mse), "mae": float(mae)})
        logging.info("traj_id=%s MSE=%s MAE=%s", traj_id, mse, mae)

    summary = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "embodiment_tag": args.embodiment_tag.value,
        "action_horizon": args.action_horizon,
        "metrics": {
            "average_mse": float(np.mean(all_mse)) if all_mse else None,
            "average_mae": float(np.mean(all_mae)) if all_mae else None,
            "per_episode_mse": all_mse,
            "per_episode_mae": all_mae,
        },
        "episodes_evaluated": per_episode,
        "notes": "MSE/MAE are in unnormalized dataset space (same as open_loop_eval).",
    }

    summary_path = out / "inference_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info("Wrote %s", summary_path)
    if all_mse:
        logging.info("Average MSE: %s  Average MAE: %s", summary["metrics"]["average_mse"], summary["metrics"]["average_mae"])


if __name__ == "__main__":
    main(tyro.cli(ArgsConfig))
