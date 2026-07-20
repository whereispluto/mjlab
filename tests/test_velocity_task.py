"""Tests specific to velocity tasks."""

import pytest

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, GO1_ACTION_SCALE
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg


@pytest.fixture(scope="module")
def velocity_task_ids() -> list[str]:
  """Get all velocity task IDs."""
  return [t for t in list_tasks() if "Velocity" in t]


@pytest.fixture(scope="module")
def g1_velocity_task_ids(velocity_task_ids: list[str]) -> list[str]:
  """Get all G1 velocity task IDs."""
  return [t for t in velocity_task_ids if "G1" in t]


@pytest.fixture(scope="module")
def go1_velocity_task_ids(velocity_task_ids: list[str]) -> list[str]:
  """Get all Go1 velocity task IDs."""
  return [t for t in velocity_task_ids if "Go1" in t]


@pytest.fixture(scope="module")
def rough_velocity_task_ids(velocity_task_ids: list[str]) -> list[str]:
  """Get all rough terrain velocity task IDs."""
  return [t for t in velocity_task_ids if "Rough" in t]


@pytest.fixture(scope="module")
def flat_velocity_task_ids(velocity_task_ids: list[str]) -> list[str]:
  """Get all flat terrain velocity task IDs."""
  return [t for t in velocity_task_ids if "Flat" in t]


def test_velocity_tasks_have_twist_command(velocity_task_ids: list[str]) -> None:
  """All velocity tasks should have a velocity command."""
  for task_id in velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert "twist" in cfg.commands, f"Task {task_id} missing 'twist' command"

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg), (
      f"Task {task_id} twist command is not UniformVelocityCommandCfg"
    )


def test_custom_biped_no_lin_vel_starts_with_slow_forward_commands() -> None:
  """Custom biped curriculum should preserve its initial 0.1-0.2 m/s range."""
  task_id = "Mjlab-Velocity-Flat-Forward-Custom-Biped-NoLinVel"
  cfg = load_env_cfg(task_id)

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  assert twist_cmd.rel_forward_envs == 0.0
  assert twist_cmd.forward_velocity_joint_name == "base_x"

  command_curriculum = cfg.curriculum["command_vel"]
  first_stage = command_curriculum.params["velocity_stages"][0]
  assert first_stage["lin_vel_x"] == (0.1, 0.2)

  track_reward = cfg.rewards["track_linear_velocity"]
  assert track_reward.params["asset_cfg"].joint_names == ("base_x", "base_z")
  assert track_reward.weight == 2.0
  assert cfg.rewards["forward_velocity"].weight == 0.5
  assert cfg.rewards["pose"].weight == 0.03
  assert cfg.rewards["pose"].params["std_walking"][r".*_knee_joint.*"] == 0.7
  assert cfg.rewards["pose"].params["std_walking"][r".*_ankle_joint.*"] == 1.0
  assert cfg.rewards["air_time"].weight == 0.8
  assert cfg.rewards["foot_slip"].weight == -0.5
  assert cfg.rewards["alternating_feet"].weight == 3.0
  assert cfg.rewards["alternating_feet"].params["target_step_length"] == 0.08
  assert cfg.rewards["foot_lead_switch"].weight == 1.0
  assert cfg.rewards["foot_lead_switch"].params["maximum_stagnation_time"] == 0.6
  assert cfg.rewards["foot_lead_switch"].params["maximum_penalty_scale"] == 2.0
  assert cfg.rewards["phase_foot_position"].weight == -1.0
  assert cfg.rewards["phase_foot_position"].params["cycle_time"] == 1.0
  assert cfg.rewards["phase_foot_alignment"].weight == 4.0
  assert cfg.rewards["phase_foot_contact"].weight == 4.0
  assert cfg.rewards["phase_foot_contact"].params["cycle_time"] == 1.0
  assert cfg.rewards["phase_foot_height"].weight == -2.0
  assert cfg.rewards["phase_foot_height"].params["target_height"] == 0.04
  assert cfg.rewards["phase_joint_position"].weight == -3.0
  assert cfg.rewards["phase_joint_position"].params["hip_amplitude"] == 0.12
  assert cfg.rewards["phase_joint_position"].params["tolerance"] == 0.2
  assert cfg.rewards["termination_penalty"].weight == -200.0
  assert cfg.rewards["alive"].weight == 1.0
  assert "gait_phase" in cfg.observations["actor"].terms
  assert "gait_phase" in cfg.observations["critic"].terms
  actuated_joint_names = (
    "left_leg_joint",
    "left_knee_joint",
    "left_ankle_joint",
    "right_leg_joint",
    "right_knee_joint",
    "right_ankle_joint",
  )
  actor_terms = cfg.observations["actor"].terms
  assert actor_terms["joint_pos"].params["asset_cfg"].joint_names == (
    actuated_joint_names
  )
  assert actor_terms["joint_pos"].params["asset_cfg"].preserve_order
  assert actor_terms["joint_vel"].params["asset_cfg"].joint_names == (
    actuated_joint_names
  )
  assert actor_terms["joint_vel"].params["asset_cfg"].preserve_order
  critic_terms = cfg.observations["critic"].terms
  assert "asset_cfg" not in critic_terms["joint_pos"].params
  assert critic_terms["joint_pos"].history_length == 4
  assert "asset_cfg" not in critic_terms["joint_vel"].params
  assert critic_terms["joint_vel"].history_length == 4
  assert cfg.rewards["swing_forward_velocity"].weight == 1.0
  assert cfg.rewards["foot_flatness"].weight == -3.0
  assert cfg.rewards["both_feet_contact"].weight == -0.2


def test_g1_velocity_has_required_sensors(g1_velocity_task_ids: list[str]) -> None:
  """G1 velocity tasks should have feet/ground and self collision sensors."""
  for task_id in g1_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.sensors is not None, f"Task {task_id} has no sensors"

    sensor_names = {s.name for s in cfg.scene.sensors}
    assert "feet_ground_contact" in sensor_names, (
      f"Task {task_id} missing feet_ground_contact sensor"
    )
    assert "self_collision" in sensor_names, (
      f"Task {task_id} missing self_collision sensor"
    )


def test_go1_velocity_has_required_sensors(go1_velocity_task_ids: list[str]) -> None:
  """Go1 velocity tasks should have feet/ground and collision sensors."""
  for task_id in go1_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.sensors is not None, f"Task {task_id} has no sensors"

    sensor_names = {s.name for s in cfg.scene.sensors}
    assert "feet_ground_contact" in sensor_names, (
      f"Task {task_id} missing feet_ground_contact sensor"
    )
    if "Rough" in task_id:
      for name in (
        "self_collision",
        "thigh_ground_touch",
        "shank_ground_touch",
        "trunk_ground_touch",
      ):
        assert name in sensor_names, f"Task {task_id} missing {name} sensor"


def test_flat_velocity_tasks_have_plane_terrain(
  flat_velocity_task_ids: list[str],
) -> None:
  """Flat velocity tasks should have terrain_type='plane' and no terrain_generator."""
  for task_id in flat_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.terrain is not None, f"Task {task_id} has no terrain config"
    assert cfg.scene.terrain.terrain_type == "plane", (
      f"Task {task_id} terrain_type={cfg.scene.terrain.terrain_type}, expected 'plane'"
    )
    assert cfg.scene.terrain.terrain_generator is None, (
      f"Task {task_id} has terrain_generator, expected None for flat terrain"
    )


def test_rough_velocity_tasks_have_generator_terrain(
  rough_velocity_task_ids: list[str],
) -> None:
  """Rough velocity tasks should have generator terrain."""
  for task_id in rough_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.terrain is not None, f"Task {task_id} has no terrain config"
    assert cfg.scene.terrain.terrain_type == "generator", (
      f"Task {task_id} terrain_type={cfg.scene.terrain.terrain_type}, "
      "expected 'generator'"
    )
    assert cfg.scene.terrain.terrain_generator is not None, (
      f"Task {task_id} has no terrain_generator, expected one for rough terrain"
    )


def test_rough_velocity_training_has_curriculum_enabled() -> None:
  """Rough velocity training tasks should have terrain curriculum enabled."""
  rough_training_tasks = [
    "Mjlab-Velocity-Rough-Unitree-G1",
    "Mjlab-Velocity-Rough-Unitree-Go1",
  ]

  for task_id in rough_training_tasks:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.terrain is not None, f"Task {task_id} has no terrain config"
    assert cfg.scene.terrain.terrain_generator is not None, (
      f"Task {task_id} has no terrain_generator"
    )
    assert cfg.scene.terrain.terrain_generator.curriculum is True, (
      f"Task {task_id} curriculum={cfg.scene.terrain.terrain_generator.curriculum}, "
      "expected True"
    )


def test_rough_velocity_play_has_curriculum_disabled() -> None:
  """Rough velocity play tasks should have terrain curriculum disabled."""
  rough_training_tasks = [
    "Mjlab-Velocity-Rough-Unitree-G1",
    "Mjlab-Velocity-Rough-Unitree-Go1",
  ]

  for task_id in rough_training_tasks:
    cfg = load_env_cfg(task_id, play=True)

    assert cfg.scene.terrain is not None, (
      f"Task {task_id} (play mode) has no terrain config"
    )
    assert cfg.scene.terrain.terrain_generator is not None, (
      f"Task {task_id} (play mode) has no terrain_generator"
    )
    assert cfg.scene.terrain.terrain_generator.curriculum is False, (
      f"Task {task_id} (play mode) curriculum={cfg.scene.terrain.terrain_generator.curriculum}, "
      "expected False"
    )


def test_g1_velocity_has_correct_action_scale(g1_velocity_task_ids: list[str]) -> None:
  """G1 velocity tasks should use G1_ACTION_SCALE."""
  for task_id in g1_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert "joint_pos" in cfg.actions, f"Task {task_id} missing 'joint_pos' action"

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg), (
      f"Task {task_id} joint_pos action is not JointPositionActionCfg"
    )

    assert joint_pos_action.scale == G1_ACTION_SCALE, (
      f"Task {task_id} action scale mismatch, expected G1_ACTION_SCALE"
    )


def test_go1_velocity_has_correct_action_scale(
  go1_velocity_task_ids: list[str],
) -> None:
  """Go1 velocity tasks should use GO1_ACTION_SCALE."""
  for task_id in go1_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert "joint_pos" in cfg.actions, f"Task {task_id} missing 'joint_pos' action"

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg), (
      f"Task {task_id} joint_pos action is not JointPositionActionCfg"
    )

    assert joint_pos_action.scale == GO1_ACTION_SCALE, (
      f"Task {task_id} action scale mismatch, expected GO1_ACTION_SCALE"
    )
