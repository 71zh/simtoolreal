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

## 2026-05-03 - Fruit actor integration

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
