"""Custom biped robot constants."""

import math
from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator

CUSTOM_BIPED_XML: Path = Path(__file__).resolve().parent / "mjcf" / "biped.xml"
assert CUSTOM_BIPED_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(CUSTOM_BIPED_XML))


LEG_ACTUATOR = ElectricActuator(
  reflected_inertia=0.01,
  velocity_limit=25.0,
  effort_limit=25.0,
)
KNEE_ACTUATOR = ElectricActuator(
  reflected_inertia=0.01,
  velocity_limit=40.0,
  effort_limit=40.0,
)
ANKLE_ACTUATOR = ElectricActuator(
  reflected_inertia=0.01,
  velocity_limit=27.0,
  effort_limit=27.0,
)

NATURAL_FREQ = 8.0 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0

STIFFNESS_LEG = LEG_ACTUATOR.reflected_inertia * NATURAL_FREQ**2
DAMPING_LEG = 2.0 * DAMPING_RATIO * LEG_ACTUATOR.reflected_inertia * NATURAL_FREQ

STIFFNESS_KNEE = KNEE_ACTUATOR.reflected_inertia * NATURAL_FREQ**2
DAMPING_KNEE = 2.0 * DAMPING_RATIO * KNEE_ACTUATOR.reflected_inertia * NATURAL_FREQ

STIFFNESS_ANKLE = ANKLE_ACTUATOR.reflected_inertia * NATURAL_FREQ**2
DAMPING_ANKLE = 2.0 * DAMPING_RATIO * ANKLE_ACTUATOR.reflected_inertia * NATURAL_FREQ

CUSTOM_BIPED_ACTUATOR_LEG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_leg_joint",),
  stiffness=STIFFNESS_LEG,
  damping=DAMPING_LEG,
  effort_limit=LEG_ACTUATOR.effort_limit,
  armature=LEG_ACTUATOR.reflected_inertia,
)
CUSTOM_BIPED_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_knee_joint",),
  stiffness=STIFFNESS_KNEE,
  damping=DAMPING_KNEE,
  effort_limit=KNEE_ACTUATOR.effort_limit,
  armature=KNEE_ACTUATOR.reflected_inertia,
)
CUSTOM_BIPED_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_joint",),
  stiffness=STIFFNESS_ANKLE,
  damping=DAMPING_ANKLE,
  effort_limit=ANKLE_ACTUATOR.effort_limit,
  armature=ANKLE_ACTUATOR.reflected_inertia,
)

CUSTOM_BIPED_HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.512),
  joint_pos={
    "base_x": 0.0,
    "base_z": 0.0,
    "base_pitch": math.radians(0.0),
    "left_leg_joint": math.radians(19),
    "left_knee_joint": math.radians(-20),
    "left_ankle_joint": math.radians(1),
    "right_leg_joint": math.radians(-17),
    "right_knee_joint": math.radians(-9),
    "right_ankle_joint": math.radians(26),
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


CUSTOM_BIPED_ACTION_SCALE: dict[str, float] = {}
for actuator in CUSTOM_BIPED_ARTICULATION.actuators:
  assert isinstance(actuator, BuiltinPositionActuatorCfg)
  effort = actuator.effort_limit
  stiffness = actuator.stiffness
  assert effort is not None
  for name in actuator.target_names_expr:
    CUSTOM_BIPED_ACTION_SCALE[name] = 0.25 * effort / stiffness
