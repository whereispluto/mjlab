import mujoco
import pytest

from mjlab.asset_zoo.robots import (
  get_custom_biped_robot_cfg,
  get_g1_robot_cfg,
  get_go1_robot_cfg,
)
from mjlab.entity import Entity


@pytest.mark.parametrize(
  "robot_name,robot_cfg_fn",
  [
    ("Custom biped", get_custom_biped_robot_cfg),
    ("G1", get_g1_robot_cfg),
    ("GO1", get_go1_robot_cfg),
  ],
)
def test_robot_compiles_parametrized(robot_name: str, robot_cfg_fn) -> None:
  """Tests that all robots in the asset zoo compile without errors."""
  robot_cfg = robot_cfg_fn()
  assert isinstance(Entity(robot_cfg).compile(), mujoco.MjModel)


def test_custom_biped_home_pose_is_bilaterally_symmetric() -> None:
  """The posture reference must not permanently favor either leading foot."""
  init_state = get_custom_biped_robot_cfg().init_state
  joint_pos = init_state.joint_pos
  assert joint_pos is not None

  for joint in ("leg_joint", "knee_joint", "ankle_joint"):
    assert joint_pos[f"left_{joint}"] == joint_pos[f"right_{joint}"]
  assert sum(
    joint_pos[f"left_{joint}"]
    for joint in (
      "leg_joint",
      "knee_joint",
      "ankle_joint",
    )
  ) == pytest.approx(0.0)
