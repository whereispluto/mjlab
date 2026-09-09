"""Custom biped robot constants."""

import math
from pathlib import Path

import mujoco

from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia, rpm_to_rad

CUSTOM_BIPED_XML: Path = Path(__file__).resolve().parent / "mjcf" / "biped.xml"
assert CUSTOM_BIPED_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(CUSTOM_BIPED_XML))


# HTDW-4438-30-NE motor and output-side specifications. The rotor inertia is
# converted from 9.1498 kg mm^2 to kg m^2. The MJCF joints are on the reducer output
# side, so their armature must include the inertia reflected through the 30:1 reducer.
HTDW4438_ROTOR_INERTIA = 9.1498e-6  # kg m^2
HTDW4438_REDUCTION_RATIO = 30.0
HTDW4438_RATED_TORQUE = 2.0  # N m
HTDW4438_STALL_TORQUE = 10.0  # N m
HTDW4438_RATED_SPEED = rpm_to_rad(40.0)
HTDW4438_NO_LOAD_SPEED = rpm_to_rad(160.0)
HTDW4438_RATED_OUTPUT_POWER = 8.5  # W
HTDW4438_REFLECTED_INERTIA = reflected_inertia(
  HTDW4438_ROTOR_INERTIA, HTDW4438_REDUCTION_RATIO
)

HTDW4438_ACTUATOR = ElectricActuator(
  reflected_inertia=HTDW4438_REFLECTED_INERTIA,
  velocity_limit=HTDW4438_NO_LOAD_SPEED,
  effort_limit=HTDW4438_RATED_TORQUE,
)

HTDW4438_NATURAL_FREQ = 5.0 * 2.0 * math.pi  # 5 Hz
HTDW4438_DAMPING_RATIO = 1.0

HTDW4438_POSITION_STIFFNESS = (
  HTDW4438_ACTUATOR.reflected_inertia * HTDW4438_NATURAL_FREQ**2
)
HTDW4438_POSITION_DAMPING = (
  2.0
  * HTDW4438_DAMPING_RATIO
  * HTDW4438_ACTUATOR.reflected_inertia
  * HTDW4438_NATURAL_FREQ
)

CUSTOM_BIPED_ACTUATOR_LEG = DcMotorActuatorCfg(
  target_names_expr=(".*_leg_joint",),
  stiffness=HTDW4438_POSITION_STIFFNESS,
  damping=HTDW4438_POSITION_DAMPING,
  effort_limit=HTDW4438_ACTUATOR.effort_limit,
  saturation_effort=HTDW4438_STALL_TORQUE,
  velocity_limit=HTDW4438_ACTUATOR.velocity_limit,
  armature=HTDW4438_ACTUATOR.reflected_inertia,
)
CUSTOM_BIPED_ACTUATOR_KNEE = DcMotorActuatorCfg(
  target_names_expr=(".*_knee_joint",),
  stiffness=HTDW4438_POSITION_STIFFNESS,
  damping=HTDW4438_POSITION_DAMPING,
  effort_limit=HTDW4438_ACTUATOR.effort_limit,
  saturation_effort=HTDW4438_STALL_TORQUE,
  velocity_limit=HTDW4438_ACTUATOR.velocity_limit,
  armature=HTDW4438_ACTUATOR.reflected_inertia,
)
CUSTOM_BIPED_ACTUATOR_ANKLE = DcMotorActuatorCfg(
  target_names_expr=(".*_ankle_joint",),
  stiffness=HTDW4438_POSITION_STIFFNESS,
  damping=HTDW4438_POSITION_DAMPING,
  effort_limit=HTDW4438_ACTUATOR.effort_limit,
  saturation_effort=HTDW4438_STALL_TORQUE,
  velocity_limit=HTDW4438_ACTUATOR.velocity_limit,
  armature=HTDW4438_ACTUATOR.reflected_inertia,
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


# The action is a position target, so its range must cover the required gait motion.
# Motor effort remains limited independently by the actuator's 2 N m effort limit.
# Using effort / stiffness here restricted every joint to 0.049 rad after the PD
# retune, below the 0.12/0.25 rad hip/knee gait references.
CUSTOM_BIPED_ACTION_SCALE: dict[str, float] = {
  ".*_leg_joint": 0.18,
  ".*_knee_joint": 0.35,
  ".*_ankle_joint": 0.25,
}
