"""Custom biped velocity environment configurations."""

import math

from mjlab.asset_zoo.robots import (
  CUSTOM_BIPED_ACTION_SCALE,
  get_custom_biped_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
  BuiltinSensorCfg,
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def _custom_biped_flat_forward_env_cfg(
  play: bool = False,
  include_actor_base_lin_vel: bool = True,
) -> ManagerBasedRlEnvCfg:
  cfg = make_velocity_env_cfg()

  cfg.sim.njmax = 200
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  cfg.scene.entities = {"robot": get_custom_biped_robot_cfg()}
  # Fixed-base custom biped cannot accept root velocity writes; disable push event.
  cfg.events.pop("push_robot", None)
  # Provide IMU sensor aliases expected by the velocity task observations.
  imu_vel_cfg = BuiltinSensorCfg(
    name="imu_lin_vel",
    sensor_type="velocimeter",
    obj=ObjRef(type="site", name="imu", entity="robot"),
  )
  imu_gyro_cfg = BuiltinSensorCfg(
    name="imu_ang_vel",
    sensor_type="gyro",
    obj=ObjRef(type="site", name="imu", entity="robot"),
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (imu_vel_cfg, imu_gyro_cfg)
  cfg.scene.sensors = tuple(
    sensor
    for sensor in (cfg.scene.sensors or ())
    if sensor.name not in {"terrain_scan"}
  )

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = (
        ObjRef(type="site", name="left_foot_site", entity="robot"),
        ObjRef(type="site", name="right_foot_site", entity="robot"),
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="body",
      pattern=("left_ankle_link", "right_ankle_link"),
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  base_ground_cfg = ContactSensorCfg(
    name="base_ground_touch",
    primary=ContactMatch(
      mode="body",
      pattern="base_link",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
    base_ground_cfg,
  )

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = CUSTOM_BIPED_ACTION_SCALE

  cfg.viewer.body_name = "base_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.heading_command = False
  twist_cmd.rel_heading_envs = 0.0
  twist_cmd.rel_world_envs = 0.0
  twist_cmd.rel_forward_envs = 1.0
  twist_cmd.rel_standing_envs = 0.0
  twist_cmd.ranges.heading = None
  twist_cmd.ranges.lin_vel_x = (0.0, 1.2)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)

  cfg.events.pop("foot_friction", None)
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)

  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*_leg_joint.*": 0.2,
    r".*_knee_joint.*": 0.35,
    r".*_ankle_joint.*": 0.15,
    r"^(?!(.*_leg_joint.*|.*_knee_joint.*|.*_ankle_joint.*)).*$": 0.05,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*_leg_joint.*": 0.3,
    r".*_knee_joint.*": 0.5,
    r".*_ankle_joint.*": 0.25,
    r"^(?!(.*_leg_joint.*|.*_knee_joint.*|.*_ankle_joint.*)).*$": 0.05,
  }

  cfg.rewards["track_linear_velocity"].weight = 4.0
  cfg.rewards["track_angular_velocity"].weight = 0.25
  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.15

  if "action_rate_l2" in cfg.rewards:
    cfg.rewards["action_rate_l2"].weight = -0.02

  if "pose" in cfg.rewards:
    cfg.rewards["pose"].weight = 0.3
  if "upright" in cfg.rewards:
    cfg.rewards["upright"].weight = 0.7

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )
  cfg.rewards["base_contact"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-10.0,
    params={
      "sensor_name": base_ground_cfg.name,
      "force_threshold": 10.0,
    },
  )

  site_names = ("left_foot_site", "right_foot_site")
  if "foot_clearance" in cfg.rewards:
    cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  if "foot_slip" in cfg.rewards:
    cfg.rewards["foot_slip"].params["asset_cfg"] = SceneEntityCfg(
      "robot", site_names=site_names
    )

  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.terminations["fell_over"].params["limit_angle"] = math.radians(75.0)
  # Terminate when the base drops below 0.30 m (nominal height ≈ 0.53 m),
  # which gives the robot more time to recover from minor collapses.
  cfg.terminations["base_too_low"] = TerminationTermCfg(
    func=mdp.root_height_below_minimum,
    params={"minimum_height": 0.30},
  )
  # Terminate when the base touches the ground.
  cfg.terminations["base_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={
      "sensor_name": base_ground_cfg.name,
      "force_threshold": 10.0,
    },
  )
  cfg.curriculum.pop("terrain_levels", None)

  actor_terms = cfg.observations["actor"].terms
  critic_terms = cfg.observations["critic"].terms
  actor_terms.pop("height_scan", None)
  critic_terms.pop("height_scan", None)
  critic_terms.pop("foot_height", None)

  if include_actor_base_lin_vel:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {"step": 0, "lin_vel_x": (0.0, 0.4), "ang_vel_z": (0.0, 0.0)},
      {"step": 5000 * 24, "lin_vel_x": (0.2, 0.8), "ang_vel_z": (0.0, 0.0)},
      {"step": 10000 * 24, "lin_vel_x": (0.4, 1.2), "ang_vel_z": (0.0, 0.0)},
    ]
  else:
    actor_terms.pop("base_lin_vel", None)
    actor_terms["joint_pos"].history_length = 4
    actor_terms["joint_pos"].flatten_history_dim = True
    actor_terms["joint_vel"].history_length = 4
    actor_terms["joint_vel"].flatten_history_dim = True
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {"step": 0, "lin_vel_x": (0.0, 0.5), "ang_vel_z": (0.0, 0.0)},
      {"step": 5000 * 24, "lin_vel_x": (0.0, 1.0), "ang_vel_z": (0.0, 0.0)},
      {"step": 10000 * 24, "lin_vel_x": (0.0, 1.5), "ang_vel_z": (0.0, 0.0)},
    ]
    twist_cmd.ranges.lin_vel_x = (0.0, 1.5)

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    twist_cmd.ranges.lin_vel_x = (0.3, 0.3) # Set to a fixed value for playing
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (0.0, 0.0)

  return cfg


def custom_biped_flat_forward_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _custom_biped_flat_forward_env_cfg(
    play=play,
    include_actor_base_lin_vel=True,
  )


def custom_biped_flat_forward_no_lin_vel_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  return _custom_biped_flat_forward_env_cfg(
    play=play,
    include_actor_base_lin_vel=False,
  )
