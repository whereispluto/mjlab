from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.tasks.velocity.mdp.terrain_utils import terrain_normal_from_sensors
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _command_scale(
  command: torch.Tensor,
  reference_velocity: float | None,
  *,
  minimum: float = 0.25,
  maximum: float = 1.5,
) -> torch.Tensor:
  if reference_velocity is None:
    return torch.ones_like(command[:, 0])
  assert reference_velocity > 0.0
  return torch.clamp(
    torch.abs(command[:, 0]) / reference_velocity,
    min=minimum,
    max=maximum,
  )


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + z_error
  return torch.exp(-lin_vel_error / std**2)


def track_planar_joint_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Track forward velocity for a planar base represented by x and z joints."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  planar_velocity = asset.data.joint_vel[:, asset_cfg.joint_ids]
  assert planar_velocity.shape[1] == 2, "Expected ordered base x and z joints."
  x_error = torch.square(command[:, 0] - planar_velocity[:, 0])
  y_error = torch.square(command[:, 1])
  z_error = torch.square(planar_velocity[:, 1])
  absolute_x_error = torch.abs(command[:, 0] - planar_velocity[:, 0])
  env.extras["log"]["Metrics/command_velocity_x_mean"] = torch.mean(command[:, 0])
  env.extras["log"]["Metrics/actual_velocity_x_mean"] = torch.mean(
    planar_velocity[:, 0]
  )
  env.extras["log"]["Metrics/velocity_x_absolute_error"] = torch.mean(absolute_x_error)
  for speed in (0.1, 0.2, 0.3, 0.4):
    mask = torch.abs(command[:, 0] - speed) < 0.05
    if torch.any(mask):
      env.extras["log"][f"Metrics/velocity_x_mae_{speed:.1f}"] = torch.mean(
        absolute_x_error[mask]
      )
  return torch.exp(-(x_error + y_error + z_error) / std**2)


def planar_joint_velocity_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  beta: float,
  vertical_velocity_weight: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Symmetric velocity cost that keeps a gradient for under- and overspeeding."""
  assert beta > 0.0
  assert vertical_velocity_weight >= 0.0
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  planar_velocity = asset.data.joint_vel[:, asset_cfg.joint_ids]
  assert planar_velocity.shape[1] == 2, "Expected ordered base x and z joints."
  forward_error = torch.nn.functional.smooth_l1_loss(
    planar_velocity[:, 0],
    command[:, 0],
    reduction="none",
    beta=beta,
  )
  vertical_error = torch.abs(planar_velocity[:, 1])
  return forward_error + vertical_velocity_weight * vertical_error


def forward_velocity(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward forward progress as a fraction of the commanded forward speed.

  This term is intended for forward-only tasks. Unlike the exponential tracking
  reward, it retains a constant learning gradient between zero and the target speed.
  Backward motion receives no reward and overspeeding is capped at one.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  target_speed = command[:, 0].clamp(min=0.05)
  if asset_cfg.joint_names is None:
    actual_speed = asset.data.root_link_lin_vel_b[:, 0]
  else:
    joint_velocity = asset.data.joint_vel[:, asset_cfg.joint_ids]
    assert joint_velocity.shape[1] == 1, "Expected one forward-velocity joint."
    actual_speed = joint_velocity[:, 0]
  env.extras["log"]["Metrics/forward_velocity_mean"] = torch.mean(actual_speed)
  return torch.clamp(actual_speed / target_speed, min=0.0, max=1.0)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error + xy_error
  return torch.exp(-ang_vel_error / std**2)


class upright:
  """Reward for keeping the base upright.

  Without ``terrain_sensor_names``, penalizes tilt relative to world up (correct for
  flat ground).

  With ``terrain_sensor_names``, penalizes tilt relative to the terrain surface normal.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._terrain_sensor_names: tuple[str, ...] | None = cfg.params.get(
      "terrain_sensor_names"
    )
    self._debug_vis_enabled = True
    self._env = env
    self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    terrain_sensor_names: tuple[str, ...] | None = None,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    if asset_cfg.body_ids:
      body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
      body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    else:
      body_quat_w = asset.data.root_link_quat_w  # [B, 4]

    if terrain_sensor_names is not None:
      terrain_normal = terrain_normal_from_sensors(env, terrain_sensor_names)  # [B, 3]
      # Project terrain normal into body frame. When aligned with the terrain surface
      # this should be (0, 0, 1); XY measures tilt.
      target_b = quat_apply_inverse(body_quat_w, terrain_normal)  # [B, 3]
      xy_squared = torch.sum(torch.square(target_b[:, :2]), dim=1)
    else:
      gravity_w = asset.data.gravity_vec_w  # [3]
      projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
      xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)

    return torch.exp(-xy_squared / std**2)

  def reset(self, env_ids: torch.Tensor) -> None:
    del env_ids  # Unused.

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    if not self._debug_vis_enabled or self._terrain_sensor_names is None:
      return

    env = self._env
    asset: Entity = env.scene[self._asset_cfg.name]

    env_indices = list(visualizer.get_env_indices(env.num_envs))
    if not env_indices:
      return

    terrain_normal = terrain_normal_from_sensors(env, self._terrain_sensor_names)
    if self._asset_cfg.body_ids:
      body_quat_w = asset.data.body_link_quat_w[:, self._asset_cfg.body_ids, :].squeeze(
        1
      )
    else:
      body_quat_w = asset.data.root_link_quat_w
    up_local = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(
      body_quat_w[:, :3]
    )
    body_up_w = quat_apply(body_quat_w, up_local)

    positions = asset.data.root_link_pos_w.cpu().numpy()
    offset = np.array([0.0, 0.3, 0.0])
    terrain_normal_np = terrain_normal.cpu().numpy()
    body_up_np = body_up_w.cpu().numpy()
    scale = 0.25

    for i in env_indices:
      origin = positions[i] + offset
      # Terrain normal (magenta).
      visualizer.add_arrow(
        start=origin,
        end=origin + terrain_normal_np[i] * scale,
        color=(0.8, 0.2, 0.8, 0.8),
        width=0.01,
      )
      # Body up (orange).
      visualizer.add_arrow(
        start=origin,
        end=origin + body_up_np[i] * scale,
        color=(1.0, 0.5, 0.0, 0.8),
        width=0.01,
      )


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.sum(dim=-1).float()


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold_min: float = 0.05,
  threshold_max: float = 0.5,
  command_name: str | None = None,
  command_threshold: float = 0.5,
) -> torch.Tensor:
  """Reward a bounded swing duration once when each foot lands."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  last_air_time = sensor_data.last_air_time
  assert last_air_time is not None
  first_contact = sensor.compute_first_contact(dt=env.step_dt)
  swing_duration = torch.clamp(
    last_air_time - threshold_min,
    min=0.0,
    max=threshold_max - threshold_min,
  )
  reward = torch.sum(swing_duration * first_contact.float(), dim=1)
  num_landings = torch.sum(first_contact.float())
  air_time_at_landing = last_air_time * first_contact.float()
  mean_air_time = torch.sum(air_time_at_landing) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/air_time_mean"] = mean_air_time
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward


class alternating_feet:
  """Reward alternating single-foot landings and discourage repeated landings."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.last_landing_foot = torch.full(
      (env.num_envs,), -1, device=env.device, dtype=torch.long
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.05,
    minimum_air_time: float = 0.08,
    minimum_step_length: float = 0.02,
    target_step_length: float = 0.08,
    failed_step_penalty: float = 1.0,
    repeated_landing_penalty: float = 0.2,
    rapid_landing_penalty: float = 0.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    assert target_step_length > minimum_step_length >= 0.0
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    assert first_contact.shape[1] == 2, (
      "alternating_feet requires a contact sensor with exactly two feet"
    )
    last_air_time = contact_sensor.data.last_air_time
    assert last_air_time is not None

    single_landing = torch.sum(first_contact, dim=1) == 1
    landing_foot = torch.argmax(first_contact.to(torch.int64), dim=1)
    landing_air_time = torch.gather(
      last_air_time, dim=1, index=landing_foot.unsqueeze(1)
    ).squeeze(1)
    valid_landing = single_landing & (landing_air_time >= minimum_air_time)
    has_previous_landing = self.last_landing_foot >= 0
    alternating = (
      valid_landing & has_previous_landing & (landing_foot != self.last_landing_foot)
    )
    repeated = (
      valid_landing & has_previous_landing & (landing_foot == self.last_landing_foot)
    )
    rapid_landing = (
      single_landing & has_previous_landing & (landing_air_time < minimum_air_time)
    )

    foot_x = asset.data.site_pos_w[:, asset_cfg.site_ids, 0]
    assert foot_x.shape[1] == 2, (
      "alternating_feet requires exactly two ordered foot sites"
    )
    other_foot = 1 - landing_foot
    landing_x = torch.gather(foot_x, dim=1, index=landing_foot.unsqueeze(1)).squeeze(1)
    other_x = torch.gather(foot_x, dim=1, index=other_foot.unsqueeze(1)).squeeze(1)
    landing_step_length = landing_x - other_x
    successful_step = alternating & (landing_step_length >= minimum_step_length)
    failed_step = alternating & ~successful_step
    step_progress = torch.clamp(
      (landing_step_length - minimum_step_length)
      / (target_step_length - minimum_step_length),
      min=0.0,
      max=1.0,
    )
    reward = (
      step_progress * successful_step.float()
      - failed_step_penalty * failed_step.float()
      - repeated_landing_penalty * repeated.float()
      - rapid_landing_penalty * rapid_landing.float()
    )
    command = env.command_manager.get_command(command_name)
    assert command is not None
    total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    active = total_command > command_threshold
    reward *= active.float()

    update = valid_landing & active
    self.last_landing_foot = torch.where(update, landing_foot, self.last_landing_foot)
    env.extras["log"]["Metrics/alternating_landing_rate"] = torch.mean(
      alternating.float()
    )
    env.extras["log"]["Metrics/successful_step_rate"] = torch.mean(
      successful_step.float()
    )
    env.extras["log"]["Metrics/rapid_landing_rate"] = torch.mean(rapid_landing.float())
    alternating_count = torch.clamp(torch.sum(alternating.float()), min=1.0)
    env.extras["log"]["Metrics/landing_step_length_mean"] = (
      torch.sum(landing_step_length * alternating.float()) / alternating_count
    )
    env.extras["log"]["Metrics/left_foot_ahead_fraction"] = torch.mean(
      (foot_x[:, 0] > foot_x[:, 1]).float()
    )
    for foot_id, foot_name in enumerate(("left", "right")):
      landing_mask = alternating & (landing_foot == foot_id)
      landing_count = torch.clamp(torch.sum(landing_mask.float()), min=1.0)
      env.extras["log"][f"Metrics/{foot_name}_landing_step_length"] = (
        torch.sum(landing_step_length * landing_mask.float()) / landing_count
      )
      env.extras["log"][f"Metrics/{foot_name}_step_success_fraction"] = (
        torch.sum((successful_step & (landing_foot == foot_id)).float()) / landing_count
      )
    return reward

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.last_landing_foot[env_ids] = -1


class alternating_foot_lead:
  """Reward switching which foot leads and penalize prolonged lead stagnation."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.last_leading_foot = torch.full(
      (env.num_envs,), -1, device=env.device, dtype=torch.long
    )
    self.time_since_switch = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    minimum_lead: float = 0.02,
    maximum_stagnation_time: float = 0.6,
    stagnation_penalty: float = 0.5,
    maximum_penalty_scale: float = 4.0,
    command_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    assert minimum_lead > 0.0
    assert maximum_stagnation_time > 0.0
    assert maximum_penalty_scale >= 1.0
    asset: Entity = env.scene[asset_cfg.name]
    foot_x = asset.data.site_pos_w[:, asset_cfg.site_ids, 0]
    assert foot_x.shape[1] == 2, (
      "alternating_foot_lead requires exactly two ordered foot sites"
    )

    separation = foot_x[:, 0] - foot_x[:, 1]
    candidate_leader = torch.full_like(self.last_leading_foot, -1)
    candidate_leader = torch.where(
      separation >= minimum_lead,
      torch.zeros_like(candidate_leader),
      candidate_leader,
    )
    candidate_leader = torch.where(
      separation <= -minimum_lead,
      torch.ones_like(candidate_leader),
      candidate_leader,
    )

    command = env.command_manager.get_command(command_name)
    assert command is not None
    total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    active = total_command > command_threshold
    candidate_valid = candidate_leader >= 0
    initialized = active & candidate_valid & (self.last_leading_foot < 0)
    switched = (
      active
      & candidate_valid
      & (self.last_leading_foot >= 0)
      & (candidate_leader != self.last_leading_foot)
    )

    self.time_since_switch = torch.where(
      active,
      self.time_since_switch + self.step_dt,
      torch.zeros_like(self.time_since_switch),
    )
    self.time_since_switch = torch.where(
      initialized | switched,
      torch.zeros_like(self.time_since_switch),
      self.time_since_switch,
    )
    update_leader = active & candidate_valid
    self.last_leading_foot = torch.where(
      update_leader, candidate_leader, self.last_leading_foot
    )

    has_leader = self.last_leading_foot >= 0
    stagnation = torch.clamp(
      (self.time_since_switch - maximum_stagnation_time) / maximum_stagnation_time,
      min=0.0,
      max=maximum_penalty_scale,
    )
    reward = switched.float() - stagnation_penalty * stagnation * has_leader.float()
    env.extras["log"]["Metrics/lead_switch_rate"] = torch.mean(switched.float())
    env.extras["log"]["Metrics/lead_stagnation_time_mean"] = torch.mean(
      self.time_since_switch * has_leader.float()
    )
    env.extras["log"]["Metrics/lead_stagnation_penalty_mean"] = torch.mean(
      stagnation * has_leader.float()
    )
    return reward

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.last_leading_foot[env_ids] = -1
    self.time_since_switch[env_ids] = 0.0


def feet_phase_position(
  env: ManagerBasedRlEnv,
  command_name: str,
  cycle_time: float,
  target_step_length: float,
  tolerance: float,
  reference_velocity: float | None = None,
  maximum_error_scale: float = 2.0,
  command_threshold: float = 0.05,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Track a periodic target for which foot should lead in the sagittal plane."""
  assert cycle_time > 0.0
  assert target_step_length > 0.0
  assert tolerance > 0.0
  assert maximum_error_scale >= 1.0
  asset: Entity = env.scene[asset_cfg.name]
  foot_x = asset.data.site_pos_w[:, asset_cfg.site_ids, 0]
  assert foot_x.shape[1] == 2, (
    "feet_phase_position requires exactly two ordered foot sites"
  )

  phase = 2.0 * torch.pi * env.episode_length_buf.float() * env.step_dt / cycle_time
  command = env.command_manager.get_command(command_name)
  assert command is not None
  speed_scale = _command_scale(command, reference_velocity)
  target_separation = target_step_length * speed_scale * torch.cos(phase)
  actual_separation = foot_x[:, 0] - foot_x[:, 1]
  tracking_error = torch.abs(actual_separation - target_separation)
  normalized_error = torch.clamp(
    tracking_error / tolerance, min=0.0, max=maximum_error_scale
  )
  cost = torch.square(normalized_error)

  total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  cost *= (total_command > command_threshold).float()
  env.extras["log"]["Metrics/phase_foot_separation_error"] = torch.mean(tracking_error)
  env.extras["log"]["Metrics/phase_foot_separation_target"] = torch.mean(
    target_separation
  )
  return cost


def feet_phase_alignment(
  env: ManagerBasedRlEnv,
  command_name: str,
  cycle_time: float,
  target_step_length: float,
  reference_velocity: float | None = None,
  command_threshold: float = 0.05,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward the correct foot leading during each half of the gait cycle."""
  assert cycle_time > 0.0
  assert target_step_length > 0.0
  asset: Entity = env.scene[asset_cfg.name]
  foot_x = asset.data.site_pos_w[:, asset_cfg.site_ids, 0]
  assert foot_x.shape[1] == 2, (
    "feet_phase_alignment requires exactly two ordered foot sites"
  )

  phase = 2.0 * torch.pi * env.episode_length_buf.float() * env.step_dt / cycle_time
  desired_lead_sign = torch.where(
    torch.cos(phase) >= 0.0,
    torch.ones_like(phase),
    -torch.ones_like(phase),
  )
  actual_separation = foot_x[:, 0] - foot_x[:, 1]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  target_separation = target_step_length * _command_scale(command, reference_velocity)
  # A hard clamp has zero gradient once a foot is farther ahead than the target,
  # which is exactly where a policy stuck in a split stance needs correction.
  # Tanh keeps the reward bounded while retaining a smooth recovery gradient.
  normalized_separation = torch.tanh(actual_separation / target_separation)
  reward = desired_lead_sign * normalized_separation

  total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  reward *= (total_command > command_threshold).float()
  env.extras["log"]["Metrics/phase_lead_alignment"] = torch.mean(reward)
  return reward


def feet_phase_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  cycle_time: float,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Reward phase-scheduled single support with smooth transition windows.

  During the first half-cycle the left foot supports while the right foot
  swings; the roles reverse during the second half-cycle. Multiplying the
  contact difference by a sine clock makes the reward vanish at phase
  transitions, allowing a short, natural double-support period around landing.
  """
  assert cycle_time > 0.0
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.found is not None
  in_contact = contact_sensor.data.found > 0
  assert in_contact.shape[1] == 2, (
    "feet_phase_contact requires a contact sensor with exactly two feet"
  )

  phase = 2.0 * torch.pi * env.episode_length_buf.float() * env.step_dt / cycle_time
  support_clock = torch.sin(phase)
  contact_difference = in_contact[:, 0].float() - in_contact[:, 1].float()
  alignment = support_clock * contact_difference

  command = env.command_manager.get_command(command_name)
  assert command is not None
  total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  active = total_command > command_threshold
  reward = alignment * active.float()

  scheduled_single_support = alignment > 0.0
  env.extras["log"]["Metrics/phase_contact_alignment"] = torch.mean(reward)
  env.extras["log"]["Metrics/scheduled_single_support_rate"] = torch.mean(
    (scheduled_single_support & active).float()
  )
  env.extras["log"]["Metrics/left_contact_rate"] = torch.mean(in_contact[:, 0].float())
  env.extras["log"]["Metrics/right_contact_rate"] = torch.mean(in_contact[:, 1].float())
  return reward


def feet_phase_height(
  env: ManagerBasedRlEnv,
  height_sensor_name: str,
  command_name: str,
  cycle_time: float,
  target_height: float,
  maximum_error_scale: float = 2.0,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Track alternating sinusoidal swing-foot clearance targets."""
  assert cycle_time > 0.0
  assert target_height > 0.0
  assert maximum_error_scale >= 1.0
  height_sensor = env.scene[height_sensor_name]
  assert isinstance(height_sensor, TerrainHeightSensor), (
    f"feet_phase_height requires a TerrainHeightSensor, "
    f"got {type(height_sensor).__name__}"
  )
  foot_height = height_sensor.data.heights
  assert foot_height.shape[1] == 2, (
    "feet_phase_height requires exactly two ordered foot heights"
  )

  phase = 2.0 * torch.pi * env.episode_length_buf.float() * env.step_dt / cycle_time
  support_clock = torch.sin(phase)
  swing_weight = torch.stack(
    (torch.clamp(-support_clock, min=0.0), torch.clamp(support_clock, min=0.0)),
    dim=1,
  )
  target = target_height * swing_weight
  normalized_error = torch.clamp(
    torch.abs(foot_height - target) / target_height,
    min=0.0,
    max=maximum_error_scale,
  )
  cost = torch.sum(torch.square(normalized_error), dim=1)

  command = env.command_manager.get_command(command_name)
  assert command is not None
  total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  active = total_command > command_threshold
  cost *= active.float()
  env.extras["log"]["Metrics/phase_foot_height_error"] = torch.mean(
    torch.abs(foot_height - target)
  )
  env.extras["log"]["Metrics/left_foot_height_mean"] = torch.mean(foot_height[:, 0])
  env.extras["log"]["Metrics/right_foot_height_mean"] = torch.mean(foot_height[:, 1])
  return cost


def feet_phase_swing_velocity(
  env: ManagerBasedRlEnv,
  command_name: str,
  cycle_time: float,
  target_velocity: float,
  reference_velocity: float | None = None,
  command_threshold: float = 0.05,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward forward motion of the foot scheduled to swing by the gait clock.

  Unlike contact-gated swing rewards, this term cannot be collected by keeping
  the same foot airborne. The phase clock assigns the right foot to swing in
  the first half-cycle and the left foot in the second half-cycle.
  """
  assert cycle_time > 0.0
  assert target_velocity > 0.0
  asset: Entity = env.scene[asset_cfg.name]
  foot_velocity_x = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, 0]
  assert foot_velocity_x.shape[1] == 2, (
    "feet_phase_swing_velocity requires exactly two ordered foot sites"
  )

  phase = 2.0 * torch.pi * env.episode_length_buf.float() * env.step_dt / cycle_time
  support_clock = torch.sin(phase)
  swing_weight = torch.stack(
    (torch.clamp(-support_clock, min=0.0), torch.clamp(support_clock, min=0.0)),
    dim=1,
  )
  relative_velocity = foot_velocity_x - torch.flip(foot_velocity_x, dims=(1,))
  command = env.command_manager.get_command(command_name)
  assert command is not None
  scaled_target_velocity = target_velocity * _command_scale(command, reference_velocity)
  normalized_velocity = torch.clamp(
    relative_velocity / scaled_target_velocity.unsqueeze(1), min=-1.0, max=1.0
  )
  reward = torch.sum(normalized_velocity * swing_weight, dim=1)

  total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  active = total_command > command_threshold
  reward *= active.float()

  active_swing_weight = swing_weight * active.unsqueeze(1).float()
  total_weight = torch.clamp(torch.sum(active_swing_weight), min=1.0)
  env.extras["log"]["Metrics/phase_swing_velocity_mean"] = (
    torch.sum(relative_velocity * active_swing_weight) / total_weight
  )
  for foot_id, foot_name in enumerate(("left", "right")):
    foot_weight = active_swing_weight[:, foot_id]
    weight_sum = torch.clamp(torch.sum(foot_weight), min=1.0)
    env.extras["log"][f"Metrics/{foot_name}_phase_swing_velocity_mean"] = (
      torch.sum(relative_velocity[:, foot_id] * foot_weight) / weight_sum
    )
  return reward


def joints_phase_position(
  env: ManagerBasedRlEnv,
  command_name: str,
  cycle_time: float,
  hip_amplitude: float,
  knee_amplitude: float,
  tolerance: float,
  reference_velocity: float | None = None,
  maximum_error_scale: float = 2.0,
  command_threshold: float = 0.05,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Track a symmetric joint-space reference for alternating flat-foot steps."""
  assert cycle_time > 0.0
  assert hip_amplitude > 0.0
  assert knee_amplitude > 0.0
  assert tolerance > 0.0
  assert maximum_error_scale >= 1.0
  asset: Entity = env.scene[asset_cfg.name]
  joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  default_joint_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  assert joint_pos.shape[1] == 6, (
    "joints_phase_position requires ordered left and right hip, knee, ankle joints"
  )

  phase = 2.0 * torch.pi * env.episode_length_buf.float() * env.step_dt / cycle_time
  command = env.command_manager.get_command(command_name)
  assert command is not None
  speed_scale = _command_scale(command, reference_velocity)
  lead_clock = torch.cos(phase)
  support_clock = torch.sin(phase)
  left_swing = torch.clamp(-support_clock, min=0.0)
  right_swing = torch.clamp(support_clock, min=0.0)

  target = default_joint_pos.clone()
  left_hip_delta = hip_amplitude * speed_scale * lead_clock
  right_hip_delta = -left_hip_delta
  left_knee_delta = -knee_amplitude * speed_scale * left_swing
  right_knee_delta = -knee_amplitude * speed_scale * right_swing
  target[:, 0] += left_hip_delta
  target[:, 1] += left_knee_delta
  target[:, 2] -= left_hip_delta + left_knee_delta
  target[:, 3] += right_hip_delta
  target[:, 4] += right_knee_delta
  target[:, 5] -= right_hip_delta + right_knee_delta

  normalized_error = torch.clamp(
    torch.abs(joint_pos - target) / tolerance,
    min=0.0,
    max=maximum_error_scale,
  )
  cost = torch.mean(torch.square(normalized_error), dim=1)

  total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  active = total_command > command_threshold
  cost *= active.float()
  env.extras["log"]["Metrics/phase_joint_position_error"] = torch.mean(
    torch.abs(joint_pos - target)
  )
  return cost


def feet_swing_forward_velocity(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  target_velocity: float,
  command_threshold: float = 0.05,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward swing-foot velocity relative to the stance foot."""
  assert target_velocity > 0.0, "target_velocity must be positive"
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.found is not None
  in_contact = contact_sensor.data.found > 0
  in_air = ~in_contact
  foot_velocity_x = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, 0]
  assert in_air.shape == foot_velocity_x.shape, (
    "feet_swing_forward_velocity requires one contact slot per foot site"
  )

  single_support = in_air & torch.flip(in_contact, dims=(1,))
  relative_velocity = foot_velocity_x - torch.flip(foot_velocity_x, dims=(1,))
  normalized_velocity = torch.clamp(
    relative_velocity / target_velocity, min=-1.0, max=1.0
  )
  reward = torch.sum(normalized_velocity * single_support.float(), dim=1)

  command = env.command_manager.get_command(command_name)
  assert command is not None
  total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  active = (total_command > command_threshold).float()
  reward *= active
  env.extras["log"]["Metrics/swing_forward_velocity_mean"] = torch.mean(
    relative_velocity * single_support.float()
  )
  for foot_id, foot_name in enumerate(("left", "right")):
    swing_mask = single_support[:, foot_id].float()
    num_swings = torch.clamp(torch.sum(swing_mask), min=1.0)
    env.extras["log"][f"Metrics/{foot_name}_swing_velocity_mean"] = (
      torch.sum(relative_velocity[:, foot_id] * swing_mask) / num_swings
    )
  return reward


def feet_contact_flatness(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  settle_time: float = 0.04,
  command_threshold: float = 0.05,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot pitch after heel strike has had time to settle."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  current_contact_time = contact_sensor.data.current_contact_time
  assert current_contact_time is not None
  assert current_contact_time.shape[1] == 2, (
    "feet_contact_flatness requires a contact sensor with exactly two feet"
  )

  foot_quat_w = asset.data.site_quat_w[:, asset_cfg.site_ids]
  assert foot_quat_w.shape[1] == 2, (
    "feet_contact_flatness requires exactly two ordered foot sites"
  )
  local_forward = torch.zeros_like(foot_quat_w[..., :3])
  local_forward[..., 0] = 1.0
  forward_axis_w = quat_apply(foot_quat_w, local_forward)
  pitch_error = torch.abs(forward_axis_w[..., 2])
  settled_contact = current_contact_time >= settle_time
  cost = torch.sum(pitch_error * settled_contact.float(), dim=1)

  command = env.command_manager.get_command(command_name)
  assert command is not None
  total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  cost *= (total_command > command_threshold).float()

  for foot_id, foot_name in enumerate(("left", "right")):
    contact_mask = settled_contact[:, foot_id].float()
    num_contacts = torch.clamp(torch.sum(contact_mask), min=1.0)
    env.extras["log"][f"Metrics/{foot_name}_foot_pitch_error"] = (
      torch.sum(pitch_error[:, foot_id] * contact_mask) / num_contacts
    )
  return cost


def both_feet_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize double support under a motion command to initiate foot lift."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.found is not None
  assert contact_sensor.data.found.shape[1] == 2, (
    "both_feet_contact requires a contact sensor with exactly two feet"
  )
  double_support = torch.all(contact_sensor.data.found > 0, dim=1)

  command = env.command_manager.get_command(command_name)
  assert command is not None
  total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  active = total_command > command_threshold
  cost = double_support.float() * active.float()
  env.extras["log"]["Metrics/double_support_rate"] = torch.mean(double_support.float())
  return cost


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  height_sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  height_sensor = env.scene[height_sensor_name]
  assert isinstance(height_sensor, TerrainHeightSensor), (
    f"feet_clearance requires a TerrainHeightSensor, got {type(height_sensor).__name__}"
  )
  foot_height = height_sensor.data.heights  # [B, F]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, F, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, F]
  delta = torch.abs(foot_height - target_height)  # [B, F]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    height_sensor = env.scene[cfg.params["height_sensor_name"]]
    assert isinstance(height_sensor, TerrainHeightSensor), (
      f"feet_swing_height requires a TerrainHeightSensor, got {type(height_sensor).__name__}"
    )
    num_feet = height_sensor.num_frames
    self.peak_heights = torch.zeros(
      (env.num_envs, num_feet), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    height_sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
  ) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    height_sensor: TerrainHeightSensor = env.scene[height_sensor_name]
    foot_heights = height_sensor.data.heights
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class variable_posture:
  """Penalize deviation from default pose with speed-dependent tolerance.

  Uses per-joint standard deviations to control how much each joint can deviate
  from default pose. Smaller std = stricter (less deviation allowed), larger
  std = more forgiving. The reward is: exp(-mean(error² / std²))

  Three speed regimes (based on linear + angular command velocity):
    - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
    - std_walking (walking_threshold <= speed < running_threshold): Moderate.
    - std_running (speed >= running_threshold): Loose tolerance for large motion.

  Tune std values per joint based on how much motion that joint needs at each
  speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))
