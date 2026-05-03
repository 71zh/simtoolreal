#!/usr/bin/env python3
"""Run ``dextoolbench/eval_knife_fruit.py`` using the latest training checkpoint.

Expects training output under::

    runs/SimToolRealKnifeFruitLSTMAsymmetricPPO/config.yaml
    runs/SimToolRealKnifeFruitLSTMAsymmetricPPO/nn/last.pth

Run from repo root with venv + Isaac Gym active.

Example::

    python scripts/knife_fruit_eval_latest.py --num-episodes 10

# isort: skip_file
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import tyro


@dataclass
class EvalLatestArgs:
    num_episodes: int = 8
    output_json: Path = Path("knife_fruit_runs/eval_latest.json")


def main() -> None:
    args = tyro.cli(EvalLatestArgs)
    root = Path(__file__).resolve().parent.parent

    exp_name = "SimToolRealKnifeFruitLSTMAsymmetricPPO"
    run_dir = root / "runs" / exp_name
    cfg = run_dir / "config.yaml"
    ckpt = run_dir / "nn" / "last.pth"

    if not cfg.is_file():
        raise SystemExit(
            f"Missing {cfg}. Train first with scripts/knife_fruit_quick_train.py"
        )
    if not ckpt.is_file():
        raise SystemExit(
            f"Missing {ckpt}. Wait until training saves at least once "
            "(see save_frequency in the train script)."
        )

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
