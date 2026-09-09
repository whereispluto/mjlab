import mujoco
import pytest

from mjlab.actuator import DcMotorActuatorCfg
from mjlab.asset_zoo.robots import (
  get_custom_biped_robot_cfg,
  get_g1_robot_cfg,
  get_go1_robot_cfg,
)
from mjlab.asset_zoo.robots.custom_biped.biped_constants import (
  CUSTOM_BIPED_ACTION_SCALE,
  HTDW4438_DAMPING_RATIO,
  HTDW4438_NATURAL_FREQ,
  HTDW4438_NO_LOAD_SPEED,
  HTDW4438_POSITION_DAMPING,
  HTDW4438_POSITION_STIFFNESS,
  HTDW4438_RATED_TORQUE,
  HTDW4438_REDUCTION_RATIO,
  HTDW4438_REFLECTED_INERTIA,
  HTDW4438_ROTOR_INERTIA,
  HTDW4438_STALL_TORQUE,
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


def test_custom_biped_foot_meshes_use_g1_sliding_friction() -> None:
  model = Entity(get_custom_biped_robot_cfg()).compile()

  for name in ("left_foot_collision", "right_foot_collision"):
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    assert geom_id >= 0
    assert model.geom_friction[geom_id, 0] == pytest.approx(0.6)


def test_custom_biped_uses_htdw4438_motor_limits() -> None:
  """All six drive joints should use the HTDW-4438-30 output ratings."""
  articulation = get_custom_biped_robot_cfg().articulation
  assert articulation is not None

  actuators = articulation.actuators
  assert len(actuators) == 3
  for actuator in actuators:
    assert isinstance(actuator, DcMotorActuatorCfg)
    assert actuator.effort_limit == pytest.approx(HTDW4438_RATED_TORQUE)
    assert actuator.saturation_effort == pytest.approx(HTDW4438_STALL_TORQUE)
    assert actuator.velocity_limit == pytest.approx(HTDW4438_NO_LOAD_SPEED)
    assert actuator.stiffness == pytest.approx(HTDW4438_POSITION_STIFFNESS)
    assert actuator.damping == pytest.approx(HTDW4438_POSITION_DAMPING)
    assert actuator.armature == pytest.approx(HTDW4438_REFLECTED_INERTIA)


def test_custom_biped_reflects_htdw4438_rotor_inertia_through_reducer() -> None:
  assert HTDW4438_REFLECTED_INERTIA == pytest.approx(
    HTDW4438_ROTOR_INERTIA * HTDW4438_REDUCTION_RATIO**2
  )


def test_custom_biped_pd_gains_follow_natural_frequency_design() -> None:
  assert HTDW4438_POSITION_STIFFNESS == pytest.approx(
    HTDW4438_REFLECTED_INERTIA * HTDW4438_NATURAL_FREQ**2
  )
  assert HTDW4438_POSITION_DAMPING == pytest.approx(
    2.0 * HTDW4438_DAMPING_RATIO * HTDW4438_REFLECTED_INERTIA * HTDW4438_NATURAL_FREQ
  )


def test_custom_biped_action_scale_covers_gait_reference() -> None:
  """Position targets should cover gait motion; actuators limit effort separately."""
  assert CUSTOM_BIPED_ACTION_SCALE[".*_leg_joint"] >= 0.12
  assert CUSTOM_BIPED_ACTION_SCALE[".*_knee_joint"] >= 0.25
  assert CUSTOM_BIPED_ACTION_SCALE[".*_ankle_joint"] >= 0.25
