"""Custom biped velocity environment configurations."""

import math
from copy import deepcopy

from mjlab.asset_zoo.robots import (
  CUSTOM_BIPED_ACTION_SCALE,
  get_custom_biped_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationTermCfg
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
  cfg.events["reset_base"].params["pose_range"]["z"] = (0.0, 0.0)
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
  fall_ground_cfg = ContactSensorCfg(
    name="fall_ground_touch",
    primary=ContactMatch(
      mode="body",
      pattern=(
        "base_link",
        "left_leg_Link",
        "left_knee_link",
        "right_leg_link",
        "right_knee_link",
      ),
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
    fall_ground_cfg,
  )

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = CUSTOM_BIPED_ACTION_SCALE

  cfg.viewer.body_name = "base_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.forward_velocity_joint_name = "base_x"
  twist_cmd.heading_command = False
  twist_cmd.rel_heading_envs = 0.0
  twist_cmd.rel_world_envs = 0.0
  # The configured ranges are already forward-only. Keep the generic forward-only
  # sampler disabled because it clamps linear x commands to at least 0.3 m/s.
  twist_cmd.rel_forward_envs = 0.0
  twist_cmd.rel_standing_envs = 0.0
  twist_cmd.ranges.heading = None
  twist_cmd.ranges.lin_vel_x = (0, 0.2)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
    "left_foot_collision",
    "right_foot_collision",
  )
  cfg.events["foot_friction"].params["ranges"] = (0.4, 1.0)
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

  cfg.rewards["track_linear_velocity"].weight = 6.0
  cfg.rewards["track_linear_velocity"].params["std"] = 0.25
  cfg.rewards["track_angular_velocity"].weight = 0.25
  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.3
  cfg.rewards["air_time"].params["command_threshold"] = 0.05

  if "action_rate_l2" in cfg.rewards:
    cfg.rewards["action_rate_l2"].weight = -0.02

  if "pose" in cfg.rewards:
    cfg.rewards["pose"].weight = 0.1
  if "upright" in cfg.rewards:
    cfg.rewards["upright"].weight = 0.5

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
  cfg.terminations["fell_over"].params["limit_angle"] = math.radians(50.0)
  # Terminate when the slide-root drops below the nominal foot-ground offset.
  cfg.terminations["base_too_low"] = TerminationTermCfg(
    func=mdp.root_height_below_minimum,
    params={"minimum_height": 0.01},
  )
  # Terminate when non-foot links touch the ground.
  cfg.terminations["fall_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={
      "sensor_name": fall_ground_cfg.name,
      "force_threshold": 5.0,
    },
  )
  cfg.curriculum.pop("terrain_levels", None)

  actor_terms = cfg.observations["actor"].terms
  critic_terms = cfg.observations["critic"].terms
  projected_gravity_cfg = SceneEntityCfg("robot", body_names=("base_link",))
  actor_terms["projected_gravity"].params["asset_cfg"] = projected_gravity_cfg
  critic_terms["projected_gravity"].params["asset_cfg"] = deepcopy(
    projected_gravity_cfg
  )
  actor_terms.pop("height_scan", None)
  critic_terms.pop("height_scan", None)
  critic_terms.pop("foot_height", None)

  if include_actor_base_lin_vel:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {"step": 0, "lin_vel_x": (0.0, 0.4), "ang_vel_z": (0.0, 0.0)},
      {"step": 5000 * 24, "lin_vel_x": (-0.2, 0.2), "ang_vel_z": (0.0, 0.0)},
      {"step": 10000 * 24, "lin_vel_x": (-0.2, 0.6), "ang_vel_z": (0.0, 0.0)},
    ]
  else:
    actor_terms.pop("base_lin_vel", None)
    actor_terms["joint_pos"] = deepcopy(actor_terms["joint_pos"])
    actor_terms["joint_vel"] = deepcopy(actor_terms["joint_vel"])
    actuated_joint_names = (
      "left_leg_joint",
      "left_knee_joint",
      "left_ankle_joint",
      "right_leg_joint",
      "right_knee_joint",
      "right_ankle_joint",
    )
    actor_terms["joint_pos"].params["asset_cfg"] = SceneEntityCfg(
      "robot", joint_names=actuated_joint_names, preserve_order=True
    )
    actor_terms["joint_pos"].history_length = 4
    actor_terms["joint_pos"].flatten_history_dim = True
    actor_terms["joint_vel"].params["asset_cfg"] = SceneEntityCfg(
      "robot", joint_names=actuated_joint_names, preserve_order=True
    )
    actor_terms["joint_vel"].history_length = 4
    actor_terms["joint_vel"].flatten_history_dim = True
    critic_terms["joint_pos"].history_length = 4
    critic_terms["joint_pos"].flatten_history_dim = True
    critic_terms["joint_vel"].history_length = 4
    critic_terms["joint_vel"].flatten_history_dim = True
    actor_terms["gait_phase"] = ObservationTermCfg(
      func=mdp.gait_phase,
      params={"cycle_time": 1.0},
    )
    critic_terms["gait_phase"] = ObservationTermCfg(
      func=mdp.gait_phase,
      params={"cycle_time": 1.0},
    )
    cfg.rewards["pose"].weight = 0.03
    cfg.rewards["pose"].params["std_walking"] = {
      r".*_leg_joint.*": 0.5,
      r".*_knee_joint.*": 0.7,
      r".*_ankle_joint.*": 1.0,
      r"^(?!(.*_leg_joint.*|.*_knee_joint.*|.*_ankle_joint.*)).*$": 0.1,
    }
    cfg.rewards["track_linear_velocity"].func = mdp.track_planar_joint_velocity
    cfg.rewards["track_linear_velocity"].weight = 6.0
    cfg.rewards["track_linear_velocity"].params["std"] = 0.2
    cfg.rewards["track_linear_velocity"].params["asset_cfg"] = SceneEntityCfg(
      "robot",
      joint_names=("base_x", "base_z"),
      preserve_order=True,
    )
    cfg.rewards["velocity_error"] = RewardTermCfg(
      func=mdp.planar_joint_velocity_error,
      weight=-8.0,
      params={
        "command_name": "twist",
        "beta": 0.1,
        "vertical_velocity_weight": 0.25,
        "asset_cfg": SceneEntityCfg(
          "robot", joint_names=("base_x", "base_z"), preserve_order=True
        ),
      },
    )
    cfg.rewards["forward_velocity"] = RewardTermCfg(
      func=mdp.forward_velocity,
      weight=2.0,
      params={
        "command_name": "twist",
        "asset_cfg": SceneEntityCfg("robot", joint_names=("base_x",)),
      },
    )
    cfg.rewards["action_rate_l2"].weight = -0.02
    cfg.rewards["action_acc_l2"] = RewardTermCfg(
      func=mdp.action_acc_l2,
      weight=-0.01,
    )
    cfg.rewards["motor_effort_l2"] = RewardTermCfg(
      func=mdp.joint_torques_l2,
      weight=-0.005,
      params={"asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards["air_time"].weight = 0.8
    cfg.rewards["air_time"].params["threshold_min"] = 0.2
    cfg.rewards["air_time"].params["threshold_max"] = 0.5
    cfg.rewards["foot_clearance"].weight = -1.0
    cfg.rewards["foot_clearance"].params["target_height"] = 0.04
    cfg.rewards["foot_swing_height"].weight = -0.5
    cfg.rewards["foot_swing_height"].params["target_height"] = 0.04
    cfg.rewards["foot_slip"].weight = -0.5
    cfg.rewards["alternating_feet"] = RewardTermCfg(
      func=mdp.alternating_feet,
      weight=1.5,
      params={
        "sensor_name": feet_ground_cfg.name,
        "command_name": "twist",
        "command_threshold": 0.05,
        "minimum_air_time": 0.25,
        "minimum_step_length": 0.02,
        "target_step_length": 0.08,
        "failed_step_penalty": 1.0,
        "repeated_landing_penalty": 0.2,
        "rapid_landing_penalty": 1.0,
        "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
      },
    )
    cfg.rewards["foot_lead_switch"] = RewardTermCfg(
      func=mdp.alternating_foot_lead,
      weight=0.25,
      params={
        "command_name": "twist",
        "minimum_lead": 0.02,
        "maximum_stagnation_time": 0.6,
        "stagnation_penalty": 0.5,
        "maximum_penalty_scale": 2.0,
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
      },
    )
    cfg.rewards["phase_foot_position"] = RewardTermCfg(
      func=mdp.feet_phase_position,
      weight=-0.5,
      params={
        "command_name": "twist",
        "cycle_time": 1.0,
        "target_step_length": 0.06,
        "reference_velocity": 0.3,
        "tolerance": 0.12,
        "maximum_error_scale": 2.0,
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
      },
    )
    cfg.rewards["phase_foot_alignment"] = RewardTermCfg(
      func=mdp.feet_phase_alignment,
      weight=2.0,
      params={
        "command_name": "twist",
        "cycle_time": 1.0,
        "target_step_length": 0.06,
        "reference_velocity": 0.3,
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
      },
    )
    cfg.rewards["phase_foot_contact"] = RewardTermCfg(
      func=mdp.feet_phase_contact,
      weight=2.0,
      params={
        "sensor_name": feet_ground_cfg.name,
        "command_name": "twist",
        "cycle_time": 1.0,
        "command_threshold": 0.05,
      },
    )
    cfg.rewards["phase_foot_height"] = RewardTermCfg(
      func=mdp.feet_phase_height,
      weight=-2.0,
      params={
        "height_sensor_name": "foot_height_scan",
        "command_name": "twist",
        "cycle_time": 1.0,
        "target_height": 0.04,
        "maximum_error_scale": 2.0,
        "command_threshold": 0.05,
      },
    )
    cfg.rewards["phase_joint_position"] = RewardTermCfg(
      func=mdp.joints_phase_position,
      weight=-0.5,
      params={
        "command_name": "twist",
        "cycle_time": 1.0,
        "hip_amplitude": 0.12,
        "knee_amplitude": 0.25,
        "reference_velocity": 0.3,
        "tolerance": 0.2,
        "maximum_error_scale": 2.0,
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg(
          "robot", joint_names=actuated_joint_names, preserve_order=True
        ),
      },
    )
    cfg.rewards["swing_forward_velocity"] = RewardTermCfg(
      func=mdp.feet_phase_swing_velocity,
      weight=1.0,
      params={
        "command_name": "twist",
        "cycle_time": 1.0,
        "command_threshold": 0.05,
        "target_velocity": 0.3,
        "reference_velocity": 0.3,
        "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
      },
    )
    cfg.rewards["foot_flatness"] = RewardTermCfg(
      func=mdp.feet_contact_flatness,
      weight=-3.0,
      params={
        "sensor_name": feet_ground_cfg.name,
        "command_name": "twist",
        "settle_time": 0.04,
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
      },
    )
    cfg.rewards["both_feet_contact"] = RewardTermCfg(
      func=mdp.both_feet_contact,
      weight=-0.2,
      params={
        "sensor_name": feet_ground_cfg.name,
        "command_name": "twist",
        "command_threshold": 0.05,
      },
    )
    cfg.rewards["termination_penalty"] = RewardTermCfg(
      func=mdp.is_terminated,
      weight=-200.0,
    )
    cfg.rewards["alive"] = RewardTermCfg(
      func=mdp.is_alive,
      weight=1.0,
    )
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {"step": 0, "lin_vel_x": (0.12, 0.18), "ang_vel_z": (0.0, 0.0)},
      {"step": 10000 * 24, "lin_vel_x": (0.1, 0.25), "ang_vel_z": (0.0, 0.0)},
      {"step": 20000 * 24, "lin_vel_x": (0.1, 0.4), "ang_vel_z": (0.0, 0.0)},
    ]
    twist_cmd.ranges.lin_vel_x = (0.1, 0.4)

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations = {}
    cfg.curriculum = {}
    twist_cmd.ranges.lin_vel_x = (0.3, 0.3)  # Set to a fixed value for playing
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
