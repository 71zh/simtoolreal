#!/usr/bin/env python3
"""Evaluate the latest training checkpoint under ``runs/<train_cfg_name>/``.

- **phase2_peel_surrogate** (default): runs ``dextoolbench/eval_knife_fruit.py`` (`SimToolRealKnifeFruitLSTMAsymmetricPPO` runs dir).
- **phase1_trajectory**: only verifies checkpoint exists and prints a one-liner to run Isaac ``test=True`` rollout (knife-fruit eval script does not apply).

Example::

    python scripts/knife_fruit_eval_latest.py --num-episodes 10
    python scripts/knife_fruit_eval_latest.py --profile phase1_trajectory

# isort: skip_file
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro


@dataclass
class EvalLatestArgs:
    profile: Literal["phase1_trajectory", "phase2_peel_surrogate"] = (
        "phase2_peel_surrogate"
    )
    """phase2 runs eval_knife_fruit metrics; phase1 only prints rollout command."""

    num_episodes: int = 8
    output_json: Path = Path("knife_fruit_runs/eval_latest.json")


def main() -> None:
    args = tyro.cli(EvalLatestArgs)
    root = Path(__file__).resolve().parent.parent

    exp_name = (
        "SimToolRealKnifePhase1LSTMAsymmetricPPO"
        if args.profile == "phase1_trajectory"
        else "SimToolRealKnifeFruitLSTMAsymmetricPPO"
    )
    run_dir = root / "runs" / exp_name
    cfg = run_dir / "config.yaml"
    ckpt = run_dir / "nn" / "last.pth"

    if not cfg.is_file():
        raise SystemExit(
            f"Missing {cfg}. Train first with scripts/knife_fruit_quick_train.py "
            f"(matching profile)."
        )
    if not ckpt.is_file():
        raise SystemExit(
            f"Missing {ckpt}. Wait until training saves at least once "
            "(see save_frequency in the train script)."
        )

    if args.profile == "phase1_trajectory":
        print(f"Checkpoint: {ckpt}")
        print("\nInference smoke (headless rollout, rl_games will print episodic stats):")
        print(
            "  python -m isaacgymenvs.train \\\n"
            "    task=SimToolRealKnifePhase1Trajectory \\\n"
            "    train=SimToolRealKnifePhase1LSTMAsymmetricPPO \\\n"
            "    test=True \\\n"
            f"    checkpoint={ckpt} \\\n"
            "    headless=True num_envs=64 task.env.numEnvs=64 \\\n"
            "    sim_device=cuda:0 rl_device=cuda:0"
        )
        return

    cmd = [
        sys.executable,
        str(root / "dextoolbench" / "eval_knife_fruit.py"),
        "--config-path",
        str(cfg),
        "--checkpoint-path",
        str(ckpt),
        "--num-episodes",
        str(args.num_episodes),
        "--output-json",
        str(root / args.output_json),
    ]
    subprocess.run(cmd, cwd=root, check=True)


if __name__ == "__main__":
    main()
