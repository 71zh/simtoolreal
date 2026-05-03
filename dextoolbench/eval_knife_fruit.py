"""Batch evaluator for knife-fruit peeling policies."""

# NOTE: torch must be imported AFTER isaacgym imports
# isort: off
import torch
# isort: on

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import tyro

from deployment.isaac.isaac_env import create_env
from deployment.rl_player import RlPlayer


@dataclass
class KnifeFruitEvalArgs:
    config_path: Path
    """Path to policy config YAML."""

    checkpoint_path: Path
    """Path to policy checkpoint."""

    output_json: Path
    """Output JSON path for aggregate and per-episode metrics."""

    num_episodes: int = 20
    """Number of episodes to evaluate."""

    deterministic_actions: bool = True
    """Whether to use deterministic policy actions."""

    object_name: str = "paring_knife"
    """Knife object name from dextoolbench.objects."""

    fruit_object_name: str = "apple_proxy"
    """Fruit object name from dextoolbench.objects."""

    success_progress_threshold: float = 0.75
    """Episode is successful only when final peel progress >= this threshold."""

    max_damage_threshold: float = 0.03
    """Episode is successful only when accumulated fruit damage <= this threshold."""


def reset_env_for_eval(env, device: str):
    """Reset helper matching existing dextoolbench eval behavior."""
    obs, _, _, _ = env.step(torch.zeros((env.num_envs, env.num_acts), device=device))
    return obs["obs"]


def main():
    args = tyro.cli(KnifeFruitEvalArgs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = create_env(
        config_path=str(args.config_path),
        headless=True,
        device=device,
        overrides={
            "task.env.objectName": args.object_name,
            "task.env.fruitObjectName": args.fruit_object_name,
            "task.env.enableKnifeFruitTask": True,
            "task.env.numEnvs": 1,
            "task.env.useActionDelay": False,
            "task.env.useObsDelay": False,
            "task.env.useObjectStateDelayNoise": False,
            "task.env.randomizeObjectRotation": True,
            "task.env.forceScale": 0.0,
            "task.env.torqueScale": 0.0,
            "task.env.linVelImpulseScale": 0.0,
            "task.env.angVelImpulseScale": 0.0,
            "task.env.objectScaleNoiseMultiplierRange": [1.0, 1.0],
        },
    )

    checkpoint = torch.load(args.checkpoint_path)
    if (
        isinstance(checkpoint, list)
        and len(checkpoint) > 0
        and isinstance(checkpoint[0], dict)
        and "env_state" in checkpoint[0]
    ):
        env.set_env_state(checkpoint[0]["env_state"])

    player = RlPlayer(
        num_observations=env.num_obs,
        num_actions=env.num_acts,
        config_path=str(args.config_path),
        checkpoint_path=str(args.checkpoint_path),
        device=device,
        num_envs=env.num_envs,
    )

    episode_rows: List[Dict[str, float]] = []
    for episode_idx in range(args.num_episodes):
        player.reset()
        obs = reset_env_for_eval(env, device=device)

        done = False
        steps = 0
        max_peel_progress = 0.0
        max_contact_depth = 0.0

        while not done:
            action = player.get_normalized_action(
                obs=obs, deterministic_actions=args.deterministic_actions
            )
            obs_dict, _, done_tensor, _ = env.step(action)
            obs = obs_dict["obs"]
            done = bool(done_tensor[0].item())
            steps += 1

            max_peel_progress = max(max_peel_progress, env.peel_progress[0].item())
            max_contact_depth = max(max_contact_depth, env.knife_contact_depth[0].item())

        final_peel_progress = float(env.peel_progress[0].item())
        final_fruit_damage = float(env.fruit_damage[0].item())
        final_successes = float(env.successes[0].item())
        task_success = (
            final_peel_progress >= args.success_progress_threshold
            and final_fruit_damage <= args.max_damage_threshold
        )

        row = {
            "episode": float(episode_idx),
            "steps": float(steps),
            "final_peel_progress": final_peel_progress,
            "max_peel_progress": float(max_peel_progress),
            "final_fruit_damage": final_fruit_damage,
            "max_contact_depth": float(max_contact_depth),
            "env_successes": final_successes,
            "task_success": float(task_success),
        }
        episode_rows.append(row)
        print(
            f"Episode {episode_idx + 1}/{args.num_episodes}: "
            f"progress={final_peel_progress:.3f}, damage={final_fruit_damage:.4f}, "
            f"task_success={task_success}"
        )

    def _avg(key: str) -> float:
        return float(np.mean([row[key] for row in episode_rows]))

    summary = {
        "num_episodes": args.num_episodes,
        "avg_steps": _avg("steps"),
        "avg_final_peel_progress": _avg("final_peel_progress"),
        "avg_max_peel_progress": _avg("max_peel_progress"),
        "avg_final_fruit_damage": _avg("final_fruit_damage"),
        "avg_max_contact_depth": _avg("max_contact_depth"),
        "avg_env_successes": _avg("env_successes"),
        "task_success_rate": _avg("task_success"),
    }

    output = {
        "args": asdict(args),
        "summary": summary,
        "episodes": episode_rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)

    print("Saved knife-fruit evaluation to:", args.output_json)
    print("Summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
