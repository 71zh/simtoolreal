# Modification Log

## 2026-05-03 - Knife-Fruit RL task scaffold

### Goals
- Add a first-pass RL scaffold for a knife-based fruit peeling task.
- Keep backward compatibility with existing `SimToolReal` training behavior.

### Code Changes
- Updated `isaacgymenvs/tasks/simtoolreal/env.py`:
  - Added knife-fruit task config switches and hyperparameters.
  - Added new observation channels:
    - `knife_tip_rel_fruit`
    - `knife_blade_tangent_align`
    - `knife_normal_contact_depth`
    - `peel_progress`
    - `peel_progress_delta`
  - Added knife-fruit internal state buffers (fruit center/radius, peel bin coverage, tip speed, damage accumulation).
  - Added reset logic for knife-fruit states per environment.
  - Added derived geometry updates for knife tip vs fruit surface each step.
  - Added new reward components:
    - alignment reward
    - contact-depth reward
    - peel-progress reward
    - fruit-damage penalty
  - Added extra reset rules for knife loss and over-penetration.
  - Added extra metrics in `extras`: `peel_progress`, `fruit_damage`.
  - Extended reward episode logging dictionary with raw and scaled knife-fruit terms.

- Updated `isaacgymenvs/tasks/simtoolreal/object_size_distributions.py`:
  - Added `knife` to supported object type literals.
  - Added two knife distributions (cuboid handle and cylindrical handle variants).

- Updated `dextoolbench/objects.py`:
  - Added `paring_knife_proxy` object entry in `NAME_TO_OBJECT`.
  - Uses existing `flat_spatula` asset as a temporary knife proxy to keep runs executable.

### New Config Files
- Added `isaacgymenvs/cfg/task/SimToolRealKnifeFruit.yaml`:
  - Enables knife-fruit mode.
  - Defines reward scales, safety thresholds, fruit randomization, peel discretization, and extended obs/state lists.

- Added `isaacgymenvs/cfg/train/SimToolRealKnifeFruitLSTMAsymmetricPPO.yaml`:
  - Inherits from LSTM asymmetric PPO config.
  - Uses moderate entropy and reduced minibatch size for initial stabilization.

### Notes
- This is a proxy peeling formulation (surface-coverage progress), not material cutting simulation.
- Dedicated knife/fruit URDF assets are still recommended for realism.

## 2026-05-03 - Dedicated assets and evaluator

### Asset Additions
- Added dedicated knife asset:
  - `assets/urdf/dextoolbench/knife/paring_knife/paring_knife.urdf`
  - `assets/urdf/dextoolbench/knife/paring_knife/paring_knife.obj`
- Added dedicated fruit proxy asset:
  - `assets/urdf/dextoolbench/fruit/apple_proxy/apple_proxy.urdf`
  - `assets/urdf/dextoolbench/fruit/apple_proxy/apple_proxy.obj`

### Object Registry Updates
- Updated `dextoolbench/objects.py`:
  - Replaced temporary `paring_knife_proxy` entry with dedicated `paring_knife`.
  - Added `apple_proxy` object entry for fruit-related experiments.

### Config Updates
- Updated `isaacgymenvs/cfg/task/SimToolRealKnifeFruit.yaml`:
  - `objectName` now defaults to `paring_knife`.
  - Added `fruitObjectName: apple_proxy` metadata field.

### New Evaluation Script
- Added `dextoolbench/eval_knife_fruit.py`:
  - Runs batch episodes without trajectory JSON dependency.
  - Collects and saves:
    - peel progress
    - fruit damage
    - contact depth
    - task success rate (progress + damage thresholds)

## 2026-05-03 - Viser replay and recording: apple / fruit actor

- Extended `recorded_data/core.py` `RecordedData` with optional `fruit_root_states_array` (T, 13) and `fruit_object_name`.
- `to_file` / `from_file` / `slice` updated; optional keys omitted when absent.
- `isaacgymenvs/tasks/simtoolreal/env.py` `_record_data` now logs per-step fruit root state when knife-fruit mode is active and saves both fields in the NPZ.
- `recorded_data/visualize.py` loads a red-tinted fruit URDF at `/fruit` when the NPZ contains `fruit_root_states_array`; CLI adds `fruit_object_name` override.
- `dextoolbench/eval.py` `ViserServer` optionally adds `/fruit` from `fruitObjectName` when `enableKnifeFruitTask` is true; `EvalRunner` passes `fruit_pose` each frame and includes fruit in video frame caches when present.
- `isaacgymenvs/cfg/task/SimToolRealKnifeFruit.yaml` comment documents enabling `record_data` for replay.

### Environment Integration
- Updated `isaacgymenvs/tasks/simtoolreal/env.py` to make fruit a real simulation actor:
  - Added `fruit_indices` tracking and `fruitObjectName` config support.
  - Extended additional-asset loading to include a fruit asset in knife-fruit mode.
  - Created a `fruit_object` actor per environment in `_create_additional_objects`.
  - Converted fruit indices to tensors after env creation.
  - Added `fruit_init_state` tensor creation and reset handling.
  - Included fruit indices in deferred actor-root state updates through `_extra_object_indices`.
  - Updated knife-fruit buffer updates to read fruit center directly from actor root states each step.

### Evaluator Update
- Updated `dextoolbench/eval_knife_fruit.py`:
  - Added `fruit_object_name` CLI argument.
  - Passes `task.env.fruitObjectName` override to environment creation.

## 2026-05-03 - Fruit actor in Isaac Gym debug visualization

- Updated `isaacgymenvs/tasks/simtoolreal/env.py` `post_physics_step` debug viewer branch:
  - When `enableDebugVis` is true and knife-fruit mode is active, draws the fruit actor pose (RGB axes via `_draw_transform`) and an orange wireframe sphere with radius `max(0.02, fruit_radius)` to match the peeling proxy geometry.
  - The fruit rigid-body mesh remains visible in the viewer whenever rendering is enabled; the extra draw aids alignment with the knife and goal.

## 2026-05-04 - Curriculum: Phase‑1 trajectory vs Phase‑2 pseudo‑peeling

### Intent
- **Phase 1 (do first)** – operational meaning: grasp the peeler and **track a WORLD‑frame pose sequence**
  (“走刀”). Use only the legacy SimToolReal stack already in `env.py`:
  **keypoints vs sequential goals** (+ lifting / finger deltas) + **`_action_penalties` joint‑velocity smoothing**
  + **`_compute_resets` fall/drop/hand‑far** guarding.
- **Phase 2 (later)** – **pseudo peeling**: retained in `SimToolRealKnifeFruit` (bins + optional fruit actor physics),
  or future rigid peel shards / trajectory‑only success.

### Assets / configs added
- `dextoolbench/trajectories/knife/paring_knife/phase1_tool_trajectory_proxy.json` – trimmed proxy arc (from spatula envelope);
  replace with peeler‑specific poses (`interactive_create_task_trajectory.py` recommended).
- `isaacgymenvs/cfg/task/SimToolRealKnifePhase1Trajectory.yaml` – Phase‑1 Hydra task: `paring_knife`, `useFixedGoalStates`, JSON goals.
- `isaacgymenvs/cfg/train/SimToolRealKnifePhase1LSTMAsymmetricPPO.yaml` – LSTM asymmetric PPO config for Phase‑1.
- `scripts/knife_fruit_quick_train.py` – `--profile phase1_trajectory` (default) vs `phase2_peel_surrogate`.
- `scripts/knife_fruit_eval_latest.py` – `--profile` chooses run directory; Phase‑1 prints `test=True` rollout command instead of peel JSON eval.