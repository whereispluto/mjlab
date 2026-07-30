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
# Calling the firmware with Kp=Kd=1.0 maps to M4438_30 int16 gain codes (19, 19).
# These are the equivalent output-side gains after protocol quantization.
HTDW4438_POSITION_STIFFNESS = 0.15893849236929034  # Kp, N m/rad
HTDW4438_POSITION_DAMPING = 0.15893849236929034  # Kd, N m s/rad

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


# With the low real-controller gain, the old 0.25--0.40 rad action range could
# produce only 0.04--0.06 N m of static PD torque. That is below the roughly
# 0.32 N m needed at each hip in the nominal crouch. Keep the real Kp/Kd unchanged
# and scale the position target range so the policy can access most, but not all,
# of the motor's continuous torque through position error.
CUSTOM_BIPED_POLICY_EFFORT_FRACTION = 0.8
CUSTOM_BIPED_POSITION_ACTION_SCALE = (
  CUSTOM_BIPED_POLICY_EFFORT_FRACTION
  * HTDW4438_RATED_TORQUE
  / HTDW4438_POSITION_STIFFNESS
)
CUSTOM_BIPED_ACTION_SCALE: dict[str, float] = {
  ".*_leg_joint": CUSTOM_BIPED_POSITION_ACTION_SCALE,
  ".*_knee_joint": CUSTOM_BIPED_POSITION_ACTION_SCALE,
  ".*_ankle_joint": CUSTOM_BIPED_POSITION_ACTION_SCALE,
}
