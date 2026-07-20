"""Custom biped robot constants."""

import math
from pathlib import Path

import mujoco

from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

CUSTOM_BIPED_XML: Path = Path(__file__).resolve().parent / "mjcf" / "biped.xml"
assert CUSTOM_BIPED_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(CUSTOM_BIPED_XML))


# HTDW-4438-30-NE output-side specifications. The MJCF joints are already on the
# reducer output side, so the 30:1 reduction ratio must not be applied again.
HTDW4438_RATED_TORQUE = 2.0  # N m
HTDW4438_STALL_TORQUE = 10.0  # N m
HTDW4438_RATED_SPEED = 40.0 * 2.0 * math.pi / 60.0  # rad/s
HTDW4438_NO_LOAD_SPEED = 160.0 * 2.0 * math.pi / 60.0  # rad/s
HTDW4438_RATED_OUTPUT_POWER = 8.5  # W

# The datasheet does not specify reflected rotor inertia or closed-loop gains. These
# PD gains are therefore explicit controller settings, shared with the STM32 true
# motion-control mode, and still need validation on a suspended joint.
HTDW4438_REFLECTED_INERTIA_ESTIMATE = 0.01  # kg m^2; pending identification
# These values map exactly to the M4438_30 int16 protocol gain codes (3020, 240),
# avoiding a small sim/firmware mismatch from protocol quantization.
HTDW4438_POSITION_STIFFNESS = 25.2628551029  # Kp, N m/rad
HTDW4438_POSITION_DAMPING = 2.0076441141  # Kd, N m s/rad

CUSTOM_BIPED_ACTUATOR_LEG = DcMotorActuatorCfg(
  target_names_expr=(".*_leg_joint",),
  stiffness=HTDW4438_POSITION_STIFFNESS,
  damping=HTDW4438_POSITION_DAMPING,
  effort_limit=HTDW4438_RATED_TORQUE,
  saturation_effort=HTDW4438_STALL_TORQUE,
  velocity_limit=HTDW4438_NO_LOAD_SPEED,
  armature=HTDW4438_REFLECTED_INERTIA_ESTIMATE,
)
CUSTOM_BIPED_ACTUATOR_KNEE = DcMotorActuatorCfg(
  target_names_expr=(".*_knee_joint",),
  stiffness=HTDW4438_POSITION_STIFFNESS,
  damping=HTDW4438_POSITION_DAMPING,
  effort_limit=HTDW4438_RATED_TORQUE,
  saturation_effort=HTDW4438_STALL_TORQUE,
  velocity_limit=HTDW4438_NO_LOAD_SPEED,
  armature=HTDW4438_REFLECTED_INERTIA_ESTIMATE,
)
CUSTOM_BIPED_ACTUATOR_ANKLE = DcMotorActuatorCfg(
  target_names_expr=(".*_ankle_joint",),
  stiffness=HTDW4438_POSITION_STIFFNESS,
  damping=HTDW4438_POSITION_DAMPING,
  effort_limit=HTDW4438_RATED_TORQUE,
  saturation_effort=HTDW4438_STALL_TORQUE,
  velocity_limit=HTDW4438_NO_LOAD_SPEED,
  armature=HTDW4438_REFLECTED_INERTIA_ESTIMATE,
)

CUSTOM_BIPED_HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.522),
  joint_pos={
    "base_x": 0.0,
    "base_z": 0.0,
    "base_pitch": math.radians(0.0),
    "left_leg_joint": math.radians(10),
    "left_knee_joint": math.radians(-20),
    "left_ankle_joint": math.radians(10),
    "right_leg_joint": math.radians(10),
    "right_knee_joint": math.radians(-20),
    "right_ankle_joint": math.radians(10),
  },
  joint_vel={".*": 0.0},
)

CUSTOM_BIPED_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    CUSTOM_BIPED_ACTUATOR_LEG,
    CUSTOM_BIPED_ACTUATOR_KNEE,
    CUSTOM_BIPED_ACTUATOR_ANKLE,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_custom_biped_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=CUSTOM_BIPED_HOME_KEYFRAME,
    spec_fn=get_spec,
    articulation=CUSTOM_BIPED_ARTICULATION,
  )


# Action range is a kinematic policy interface, not a motor torque limit. Keep the
# existing actor output mapping when changing to the real 2 N m continuous rating.
CUSTOM_BIPED_ACTION_SCALE: dict[str, float] = {
  ".*_leg_joint": 0.2473661710,
  ".*_knee_joint": 0.3957858736,
  ".*_ankle_joint": 0.2671554647,
}
