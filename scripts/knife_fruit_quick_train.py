#!/usr/bin/env python3
"""Quick training for peel-related tasks (two curriculum profiles).

- **phase1_trajectory** (default): grasp the peeler and follow fixed goal poses
  (existing keypoint / lifting / velocity-penalties / anti-drop resets). See
  `isaacgymenvs/cfg/task/SimToolRealKnifePhase1Trajectory.yaml`.
- **phase2_peel_surrogate**: Fruit actor + peel-bin surrogate rewards —
  `SimToolRealKnifeFruit.yaml`.

Run from the **repository root** with the project venv activated (Python 3.8 + Isaac Gym installed).

Example:

    source .venv/bin/activate
    python scripts/knife_fruit_quick_train.py
    python scripts/knife_fruit_quick_train.py --profile phase2_peel_surrogate
    python scripts/knife_fruit_quick_train.py --max-epochs 200 --num-envs 4096

Minimal train (cheap sanity check before ``play_trained_peeler.py`` — policy will be weak)::

    python scripts/knife_fruit_quick_train.py --max-epochs 10 --num-envs 512 --save-frequency 1

**AutoDL**

1. ``cd /root/autodl-tmp/simtoolreal`` (or your clone path), activate venv, then
   ``source /etc/network_turbo`` for faster downloads.
2. Run this script the same way as locally. This machine cannot SSH into your
   instance; you run commands on AutoDL (or Cursor Remote-SSH into it).
3. **Visualize learning curves:** in another shell on the instance::

       tensorboard --logdir runs --bind_all 0.0.0.0 --port 6006

   In the AutoDL console, add a **custom service** mapping host port to **6006**, then open the URL in your browser.
4. **Evaluate the last checkpoint ( scalars, no 3D viewer )**::

       python scripts/knife_fruit_eval_latest.py

   Outputs JSON under ``knife_fruit_runs/eval_latest.json``.

4b. **Watch the robot in Isaac Gym after training**

   After ``runs/.../nn/last.pth`` exists::

       python scripts/play_trained_peeler.py --profile phase1_trajectory --headless false --num-envs 1

   On AutoDL without a monitor, keep ``--headless true`` and enable ``--capture-video True`` so short clips save next to the checkpoint path (see Isaac Gym RecordVideo docs in ``isaacgymenvs/train.py``).

5. Isaac Gym preview is **headless** on clouds; live 3D needs X11 forwarding or a local replay pipeline (NPZ + ``recorded_data/visualize.py``) if you enable recording in the task YAML.

NOTE: Observation size differs from the original SimToolReal-only policy — do **not**
expect pretrained ``pretrained_policy`` weights to load without architecture changes.

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
class KnifeFruitQuickTrainArgs:
    profile: Literal["phase1_trajectory", "phase2_peel_surrogate"] = "phase1_trajectory"
    """phase1 = tool trajectory + smoothing + anti-drop only; phase2 = pseudo-peeling env."""

    max_epochs: int = 80
    """Number of RL learning epochs (rl_games). Raise for longer runs."""

    num_envs: int = 2048
    """Parallel environments. Try 4096 on a 24GB GPU; reduce if OOM."""

    headless: bool = True
    """Set False only if you have a display (local desktop)."""

    wandb_activate: bool = False
    """Set True and run ``wandb login`` on the machine first."""

    save_frequency: int = 20
    """Save checkpoint every this many epochs (small = easier smoke eval)."""


def _choose_minibatch(num_envs: int, horizon: int = 16) -> int:
    """Pick a minibatch size that divides num_envs * horizon (PPO rollout size)."""
    rollout = num_envs * horizon
    for mb in (8192, 4096, 2048, 1024, 512, 256):
        if rollout % mb == 0:
            return mb
    return 256


def main() -> None:
    args = tyro.cli(KnifeFruitQuickTrainArgs)
    repo_root = Path(__file__).resolve().parent.parent

    minibatch = _choose_minibatch(args.num_envs)

    if args.profile == "phase1_trajectory":
        hydra_task = "SimToolRealKnifePhase1Trajectory"
        hydra_train = "SimToolRealKnifePhase1LSTMAsymmetricPPO"
    else:
        hydra_task = "SimToolRealKnifeFruit"
        hydra_train = "SimToolRealKnifeFruitLSTMAsymmetricPPO"

    cmd: list[str] = [
        sys.executable,
        "-m",
        "isaacgymenvs.train",
        f"task={hydra_task}",
        f"train={hydra_train}",
        f"num_envs={args.num_envs}",
        f"task.env.numEnvs={args.num_envs}",
        f"headless={args.headless}",
        f"wandb_activate={args.wandb_activate}",
        f"train.params.config.minibatch_size={minibatch}",
        f"train.params.config.central_value_config.minibatch_size={minibatch}",
        f"train.params.config.max_epochs={args.max_epochs}",
        f"train.params.config.save_frequency={args.save_frequency}",
        "sim_device=cuda:0",
        "rl_device=cuda:0",
    ]

    print("cwd:", repo_root)
    print("Running:\n  " + " \\\n  ".join(cmd))
    completed = subprocess.run(cmd, cwd=repo_root)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
