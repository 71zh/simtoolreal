#!/usr/bin/env python3
"""Load the latest checkpoint and run the policy in Isaac Gym inference mode (`test=True`).

Use this **after** a short training run to eyeball behaviour in simulation.

Examples (repo root, venv activated)::

    # Phase 1 (trajectory-only), open viewer locally
    python scripts/play_trained_peeler.py --headless false --num-envs 1

    # AutoDL / SSH (no monitor): headless rollout + optionally save clips
    python scripts/play_trained_peeler.py --headless true --capture-video True --num-envs 4

`--checkpoint` defaults to ``runs/<train_cfg>/nn/last.pth`` for the matching profile.

# isort: skip_file
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import tyro


@dataclass
class PlayTrainedPeelerArgs:
    profile: Literal["phase1_trajectory", "phase2_peel_surrogate"] = (
        "phase1_trajectory"
    )
    checkpoint: Optional[Path] = None
    """If omitted, uses ``runs/<preset>/nn/last.pth``."""

    headless: bool = False
    """False = Isaac viewer window on a machine with a display (recommended locally)."""

    num_envs: int = 1
    """1 is easiest for visual inspection; bump for faster batched rollout stats."""

    capture_video: bool = False
    """If True (and Isaac can render), snippets go next to checkpoint under *_videos."""

    wandb_activate: bool = False


def _default_ckpt(root: Path, profile: str) -> Path:
    sub = (
        "SimToolRealKnifePhase1LSTMAsymmetricPPO"
        if profile == "phase1_trajectory"
        else "SimToolRealKnifeFruitLSTMAsymmetricPPO"
    )
    return root / "runs" / sub / "nn" / "last.pth"


def main() -> None:
    args = tyro.cli(PlayTrainedPeelerArgs)
    root = Path(__file__).resolve().parent.parent

    ckpt = args.checkpoint or _default_ckpt(root, args.profile)
    if not ckpt.is_file():
        raise SystemExit(
            f"No checkpoint at {ckpt}. Train first:\n"
            "  python scripts/knife_fruit_quick_train.py "
            "--max-epochs 5 --num-envs 256 --save-frequency 1"
        )

    if args.profile == "phase1_trajectory":
        task = "SimToolRealKnifePhase1Trajectory"
        hydra_train = "SimToolRealKnifePhase1LSTMAsymmetricPPO"
    else:
        task = "SimToolRealKnifeFruit"
        hydra_train = "SimToolRealKnifeFruitLSTMAsymmetricPPO"

    cmd: list[str] = [
        sys.executable,
        "-m",
        "isaacgymenvs.train",
        f"task={task}",
        f"train={hydra_train}",
        "test=True",
        f"checkpoint={ckpt}",
        f"num_envs={args.num_envs}",
        f"task.env.numEnvs={args.num_envs}",
        f"headless={args.headless}",
        f"wandb_activate={args.wandb_activate}",
        f"capture_video={args.capture_video}",
        "multi_gpu=False",
        "sim_device=cuda:0",
        "rl_device=cuda:0",
        "graphics_device_id=0",
    ]

    print("Running:\n  " + " \\\n  ".join(cmd))
    raise SystemExit(subprocess.run(cmd, cwd=root).returncode)


if __name__ == "__main__":
    main()
