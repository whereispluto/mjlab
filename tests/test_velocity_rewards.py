"""Tests for velocity task reward functions."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, PropertyMock

import torch

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import RayCastData, RayCastSensor, TerrainHeightSensor
from mjlab.tasks.velocity.mdp.observations import gait_phase
from mjlab.tasks.velocity.mdp.rewards import (
  alternating_feet,
  alternating_foot_lead,
  both_feet_contact,
  feet_contact_flatness,
  feet_phase_alignment,
  feet_phase_contact,
  feet_phase_height,
  feet_phase_position,
  feet_phase_swing_velocity,
  feet_swing_forward_velocity,
  joints_phase_position,
  planar_joint_velocity_error,
  upright,
)
from mjlab.utils.lab_api.math import quat_from_euler_xyz

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _identity_quat(B: int) -> torch.Tensor:
  """(w, x, y, z) = (1, 0, 0, 0)."""
  q = torch.zeros(B, 4)
  q[:, 0] = 1.0
  return q


def _quat_from_roll(roll_rad: float, B: int = 1) -> torch.Tensor:
  roll = torch.full((B,), roll_rad)
  zero = torch.zeros(B)
  return quat_from_euler_xyz(roll, zero, zero)


def _quat_from_pitch(pitch_rad: float, B: int = 1) -> torch.Tensor:
  pitch = torch.full((B,), pitch_rad)
  zero = torch.zeros(B)
  return quat_from_euler_xyz(zero, pitch, zero)


def _make_env_and_reward(
  terrain_sensor_names: tuple[str, ...] | None = None,
  body_quat_w: torch.Tensor | None = None,
  terrain_hit_z: float = 0.0,
  terrain_slope_x: float = 0.0,
):
  """Build mocked env + upright reward instance.

  Args:
    terrain_sensor_names: If set, enables terrain-aware mode.
    body_quat_w: [B, 4] root orientation. Defaults to identity.
    terrain_hit_z: Z value for flat terrain hits.
    terrain_slope_x: Slope in X (z = terrain_slope_x * x).
  """
  B = 1 if body_quat_w is None else body_quat_w.shape[0]
  if body_quat_w is None:
    body_quat_w = _identity_quat(B)

  # Mock asset data. Use explicit asset_cfg with no body_names so
  # body_ids stays None and the reward uses root_link_quat_w.
  asset = MagicMock()
  asset.data.root_link_quat_w = body_quat_w
  asset.data.root_link_pos_w = torch.zeros(B, 3)
  asset.data.gravity_vec_w = torch.tensor([0.0, 0.0, -1.0]).expand(B, 3)
  asset_cfg = SceneEntityCfg("robot", body_names=None, body_ids=[])

  # Mock terrain sensor if needed.
  sensors: dict = {"robot": asset}
  if terrain_sensor_names is not None:
    N = 100
    torch.manual_seed(0)
    hit_pos = torch.zeros(B, N, 3)
    hit_pos[:, :, 0] = torch.randn(B, N)
    hit_pos[:, :, 1] = torch.randn(B, N)
    hit_pos[:, :, 2] = terrain_hit_z + terrain_slope_x * hit_pos[:, :, 0]

    raycast_sensor = MagicMock(spec=RayCastSensor)
    raycast_data = RayCastData(
      distances=torch.ones(B, N),
      normals_w=torch.zeros(B, N, 3),
      hit_pos_w=hit_pos,
      pos_w=torch.zeros(B, 3),
      quat_w=torch.zeros(B, 4),
      frame_pos_w=torch.zeros(B, 1, 3),
      frame_quat_w=torch.zeros(B, 1, 4),
    )
    type(raycast_sensor).data = PropertyMock(return_value=raycast_data)
    for name in terrain_sensor_names:
      sensors[name] = raycast_sensor

  env = MagicMock()
  env.scene.__getitem__ = MagicMock(side_effect=lambda n: sensors[n])

  params: dict = {"std": 1.0, "asset_cfg": asset_cfg}
  if terrain_sensor_names is not None:
    params["terrain_sensor_names"] = terrain_sensor_names
  cfg = MagicMock(spec=RewardTermCfg)
  cfg.params = params

  reward_fn = upright(cfg, env)
  return env, reward_fn, params


def test_world_up_identity_gives_max_reward():
  """Perfectly upright robot on flat ground → reward ≈ 1."""
  env, reward, params = _make_env_and_reward()
  r = reward(env, std=params["std"], asset_cfg=params["asset_cfg"])
  assert r.shape == (1,)
  assert r.item() > 0.99


def test_world_up_tilted_gives_lower_reward():
  """30° roll → reward significantly below 1."""
  quat = _quat_from_roll(math.radians(30))
  env, reward, params = _make_env_and_reward(body_quat_w=quat)
  r = reward(env, std=params["std"], asset_cfg=params["asset_cfg"])
  assert r.item() < 0.8


def test_terrain_aware_aligned_with_slope():
  """Robot pitched to match a slope → terrain-aware reward ≈ 1."""
  slope = 0.5  # z = 0.5 * x
  tilt = math.atan(slope)  # Pitch to match slope in XZ plane.
  quat = _quat_from_pitch(-tilt)
  env, reward, params = _make_env_and_reward(
    terrain_sensor_names=("terrain_scan",),
    body_quat_w=quat,
    terrain_slope_x=slope,
  )
  r = reward(
    env,
    std=params["std"],
    asset_cfg=params["asset_cfg"],
    terrain_sensor_names=params["terrain_sensor_names"],
  )
  # Should be close to 1 since robot matches terrain.
  assert r.item() > 0.9


def test_terrain_aware_upright_on_slope_penalized():
  """Robot staying vertical on a slope → terrain-aware reward < 1."""
  slope = 0.5
  quat = _identity_quat(1)  # Robot is world-vertical, not matching slope.
  env, reward, params = _make_env_and_reward(
    terrain_sensor_names=("terrain_scan",),
    body_quat_w=quat,
    terrain_slope_x=slope,
  )
  r = reward(
    env,
    std=params["std"],
    asset_cfg=params["asset_cfg"],
    terrain_sensor_names=params["terrain_sensor_names"],
  )
  # Should be penalized since robot doesn't match terrain.
  assert r.item() < 0.95


def test_terrain_aware_flat_ground_matches_world_up():
  """On flat terrain, terrain-aware and world-up should give same reward."""
  quat = _quat_from_roll(math.radians(15))
  env_t, reward_t, params_t = _make_env_and_reward(
    terrain_sensor_names=("terrain_scan",),
    body_quat_w=quat,
  )
  env_w, reward_w, params_w = _make_env_and_reward(body_quat_w=quat)

  r_terrain = reward_t(
    env_t,
    std=params_t["std"],
    asset_cfg=params_t["asset_cfg"],
    terrain_sensor_names=params_t["terrain_sensor_names"],
  )
  r_world = reward_w(env_w, std=params_w["std"], asset_cfg=params_w["asset_cfg"])

  torch.testing.assert_close(r_terrain, r_world, atol=0.02, rtol=0.02)


def test_batch_consistency():
  """Multiple envs with different orientations get independent rewards."""
  B = 4
  quats = torch.zeros(B, 4)
  quats[:, 0] = 1.0  # All identity.
  # Tilt env 2 by 45°.
  quats[2] = _quat_from_roll(math.radians(45))[0]

  env, reward, params = _make_env_and_reward(body_quat_w=quats)
  r = reward(env, std=params["std"], asset_cfg=params["asset_cfg"])

  assert r.shape == (B,)
  # Env 0, 1, 3 should be ~1, env 2 should be lower.
  assert r[0].item() > 0.99
  assert r[1].item() > 0.99
  assert r[2].item() < 0.7
  assert r[3].item() > 0.99


def test_alternating_feet_rewards_only_opposite_single_landings():
  """Alternation is rewarded, repeated landings are penalized, and pairs ignored."""
  contact_sensor = MagicMock()
  contact_sensor.data.last_air_time = torch.full((3, 2), 0.15)
  contact_sensor.compute_first_contact.side_effect = (
    torch.tensor([[True, False], [False, True], [True, False]]),
    torch.tensor([[False, True], [False, True], [True, True]]),
  )
  command_manager = MagicMock()
  command_manager.get_command.return_value = torch.tensor([[0.2, 0.0, 0.0]] * 3)
  asset = SimpleNamespace(
    data=SimpleNamespace(
      site_pos_w=torch.tensor([[[0.0, 0.0, 0.0], [0.08, 0.0, 0.0]]] * 3)
    )
  )
  env = cast(
    "ManagerBasedRlEnv",
    SimpleNamespace(
      num_envs=3,
      device="cpu",
      step_dt=0.02,
      scene={"feet": contact_sensor, "robot": asset},
      command_manager=command_manager,
      extras={"log": {}},
    ),
  )
  reward = alternating_feet(MagicMock(spec=RewardTermCfg), env)
  asset_cfg = SceneEntityCfg("robot", site_names=("left", "right"), site_ids=[0, 1])

  first = reward(
    env, "feet", "twist", repeated_landing_penalty=0.2, asset_cfg=asset_cfg
  )
  second = reward(
    env, "feet", "twist", repeated_landing_penalty=0.2, asset_cfg=asset_cfg
  )

  torch.testing.assert_close(first, torch.zeros(3))
  torch.testing.assert_close(second, torch.tensor([1.0, -0.2, 0.0]))
  torch.testing.assert_close(reward.last_landing_foot, torch.tensor([1, 1, 0]))
  torch.testing.assert_close(
    env.extras["log"]["Metrics/right_landing_step_length"], torch.tensor(0.08)
  )
  torch.testing.assert_close(
    env.extras["log"]["Metrics/right_step_success_fraction"], torch.tensor(1.0)
  )


def test_alternating_feet_reset_clears_selected_history():
  """Reset environments should not inherit a landing from the prior episode."""
  contact_sensor = MagicMock()
  contact_sensor.data.last_air_time = torch.full((2, 2), 0.15)
  contact_sensor.compute_first_contact.side_effect = (
    torch.tensor([[True, False], [False, True]]),
    torch.tensor([[False, True], [True, False]]),
  )
  command_manager = MagicMock()
  command_manager.get_command.return_value = torch.tensor([[0.2, 0.0, 0.0]] * 2)
  asset = SimpleNamespace(
    data=SimpleNamespace(
      site_pos_w=torch.tensor(
        [
          [[0.0, 0.0, 0.0], [0.08, 0.0, 0.0]],
          [[0.08, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
      )
    )
  )
  env = cast(
    "ManagerBasedRlEnv",
    SimpleNamespace(
      num_envs=2,
      device="cpu",
      step_dt=0.02,
      scene={"feet": contact_sensor, "robot": asset},
      command_manager=command_manager,
      extras={"log": {}},
    ),
  )
  reward = alternating_feet(MagicMock(spec=RewardTermCfg), env)
  asset_cfg = SceneEntityCfg("robot", site_names=("left", "right"), site_ids=[0, 1])
  reward(env, "feet", "twist", asset_cfg=asset_cfg)
  reward.reset(torch.tensor([0]))

  result = reward(env, "feet", "twist", asset_cfg=asset_cfg)

  torch.testing.assert_close(result, torch.tensor([0.0, 1.0]))


def test_alternating_feet_penalizes_contact_jitter_after_short_air_time():
  """A rapid contact toggle should be penalized, not exploited as a landing."""
  contact_sensor = MagicMock()
  contact_sensor.data.last_air_time = torch.tensor([[0.15, 0.15]])
  contact_sensor.compute_first_contact.side_effect = (
    torch.tensor([[True, False]]),
    torch.tensor([[False, True]]),
  )
  command_manager = MagicMock()
  command_manager.get_command.return_value = torch.tensor([[0.2, 0.0, 0.0]])
  asset = SimpleNamespace(
    data=SimpleNamespace(site_pos_w=torch.tensor([[[0.0, 0.0, 0.0], [0.08, 0.0, 0.0]]]))
  )
  env = cast(
    "ManagerBasedRlEnv",
    SimpleNamespace(
      num_envs=1,
      device="cpu",
      step_dt=0.02,
      scene={"feet": contact_sensor, "robot": asset},
      command_manager=command_manager,
      extras={"log": {}},
    ),
  )
  reward = alternating_feet(MagicMock(spec=RewardTermCfg), env)
  asset_cfg = SceneEntityCfg("robot", site_names=("left", "right"), site_ids=[0, 1])
  reward(env, "feet", "twist", minimum_air_time=0.08, asset_cfg=asset_cfg)
  contact_sensor.data.last_air_time = torch.tensor([[0.15, 0.03]])

  result = reward(
    env,
    "feet",
    "twist",
    minimum_air_time=0.08,
    rapid_landing_penalty=0.5,
    asset_cfg=asset_cfg,
  )

  torch.testing.assert_close(result, torch.tensor([-0.5]))
  torch.testing.assert_close(reward.last_landing_foot, torch.tensor([0]))
  torch.testing.assert_close(
    env.extras["log"]["Metrics/rapid_landing_rate"], torch.tensor(1.0)
  )


def test_alternating_foot_lead_penalizes_stagnation_and_rewards_switch():
  """A leading foot held too long is penalized until the opposite foot passes."""
  asset = SimpleNamespace(
    data=SimpleNamespace(
      site_pos_w=torch.tensor(
        [
          [[0.04, 0.0, 0.0], [0.0, 0.0, 0.0]],
          [[0.0, 0.0, 0.0], [0.04, 0.0, 0.0]],
        ]
      )
    )
  )
  command_manager = MagicMock()
  command_manager.get_command.return_value = torch.tensor([[0.2, 0.0, 0.0]] * 2)
  env = cast(
    "ManagerBasedRlEnv",
    SimpleNamespace(
      num_envs=2,
      device="cpu",
      step_dt=0.1,
      scene={"robot": asset},
      command_manager=command_manager,
      extras={"log": {}},
    ),
  )
  reward = alternating_foot_lead(MagicMock(spec=RewardTermCfg), env)
  asset_cfg = SceneEntityCfg("robot", site_names=("left", "right"), site_ids=[0, 1])

  stagnation_reward = reward(
    env,
    "twist",
    minimum_lead=0.02,
    maximum_stagnation_time=0.2,
    stagnation_penalty=0.5,
    asset_cfg=asset_cfg,
  )
  for _ in range(6):
    stagnation_reward = reward(
      env,
      "twist",
      minimum_lead=0.02,
      maximum_stagnation_time=0.2,
      stagnation_penalty=0.5,
      asset_cfg=asset_cfg,
    )

  torch.testing.assert_close(stagnation_reward, torch.tensor([-1.0, -1.0]))

  asset.data.site_pos_w = torch.flip(asset.data.site_pos_w, dims=(1,))
  switch_reward = reward(
    env,
    "twist",
    minimum_lead=0.02,
    maximum_stagnation_time=0.2,
    stagnation_penalty=0.5,
    asset_cfg=asset_cfg,
  )

  torch.testing.assert_close(switch_reward, torch.ones(2))
  torch.testing.assert_close(reward.time_since_switch, torch.zeros(2))


def test_gait_phase_observation_tracks_episode_time():
  """One-second gait phase should advance by a quarter cycle every 0.25 s."""
  env = cast(
    "ManagerBasedRlEnv",
    SimpleNamespace(
      episode_length_buf=torch.tensor([0, 1, 2, 3]),
      step_dt=0.25,
    ),
  )

  result = gait_phase(env, cycle_time=1.0)

  expected = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]])
  torch.testing.assert_close(result, expected, atol=1e-6, rtol=0)


def test_feet_phase_position_rewards_alternating_lead_targets():
  """The left and right foot should lead on opposite halves of the gait cycle."""
  asset = SimpleNamespace(
    data=SimpleNamespace(
      site_pos_w=torch.tensor(
        [
          [[0.03, 0.0, 0.0], [-0.03, 0.0, 0.0]],
          [[-0.03, 0.0, 0.0], [0.03, 0.0, 0.0]],
          [[0.03, 0.0, 0.0], [-0.03, 0.0, 0.0]],
        ]
      )
    )
  )
  command_manager = MagicMock()
  command_manager.get_command.return_value = torch.tensor([[0.2, 0.0, 0.0]] * 3)
  env = cast(
    "ManagerBasedRlEnv",
    SimpleNamespace(
      episode_length_buf=torch.tensor([0, 2, 2]),
      step_dt=0.25,
      scene={"robot": asset},
      command_manager=command_manager,
      extras={"log": {}},
    ),
  )
  asset_cfg = SceneEntityCfg("robot", site_names=("left", "right"), site_ids=[0, 1])

  result = feet_phase_position(
    env,
    command_name="twist",
    cycle_time=1.0,
    target_step_length=0.06,
    tolerance=0.12,
    asset_cfg=asset_cfg,
  )

  torch.testing.assert_close(result, torch.tensor([0.0, 0.0, 1.0]))


def test_feet_phase_position_scales_stride_with_forward_command():
  asset = SimpleNamespace(
    data=SimpleNamespace(
      site_pos_w=torch.tensor(
        [
          [[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]],
          [[0.04, 0.0, 0.0], [-0.04, 0.0, 0.0]],
        ]
      )
    )
  )
  command_manager = MagicMock()
  command_manager.get_command.return_value = torch.tensor(
    [[0.1, 0.0, 0.0], [0.4, 0.0, 0.0]]
  )
  env = cast(
    "ManagerBasedRlEnv",
    SimpleNamespace(
      episode_length_buf=torch.zeros(2, dtype=torch.long),
      step_dt=0.02,
      scene={"robot": asset},
      command_manager=command_manager,
      extras={"log": {}},
    ),
  )
  asset_cfg = SceneEntityCfg("robot", site_names=("left", "right"), site_ids=[0, 1])

  result = feet_phase_position(
    env,
    command_name="twist",
    cycle_time=1.0,
    target_step_length=0.06,
    tolerance=0.12,
    reference_velocity=0.3,
    asset_cfg=asset_cfg,
  )

  torch.testing.assert_close(result, torch.zeros(2), atol=1e-6, rtol=0)


def test_planar_joint_velocity_error_penalizes_under_and_overspeed_symmetrically():
  asset = SimpleNamespace(
    data=SimpleNamespace(joint_vel=torch.tensor([[0.1, 0.0], [0.3, 0.0], [0.2, 0.1]]))
  )
  command_manager = MagicMock()
  command_manager.get_command.return_value = torch.tensor([[0.2, 0.0, 0.0]] * 3)
  env = cast(
    "ManagerBasedRlEnv",
    SimpleNamespace(scene={"robot": asset}, command_manager=command_manager),
  )
  asset_cfg = SceneEntityCfg(
    "robot", joint_names=("base_x", "base_z"), joint_ids=[0, 1]
  )

  result = planar_joint_velocity_error(
    env,
    command_name="twist",
    beta=0.1,
    vertical_velocity_weight=0.25,
    asset_cfg=asset_cfg,
  )

  torch.testing.assert_close(result, torch.tensor([0.05, 0.05, 0.025]))


def test_feet_phase_alignment_requires_lead_direction_to_switch():
  """A fixed leading foot is rewarded in one half-cycle and penalized in the other."""
  asset = SimpleNamespace(
    data=SimpleNamespace(
      site_pos_w=torch.tensor(
        [
          [[0.03, 0.0, 0.0], [-0.03, 0.0, 0.0]],
          [[-0.03, 0.0, 0.0], [0.03, 0.0, 0.0]],
          [[0.03, 0.0, 0.0], [-0.03, 0.0, 0.0]],
        ]
      )
    )
  )
  command_manager = MagicMock()
  command_manager.get_command.return_value = torch.tensor([[0.2, 0.0, 0.0]] * 3)
  env = cast(
    "ManagerBasedRlEnv",
    SimpleNamespace(
      episode_length_buf=torch.tensor([0, 2, 2]),
      step_dt=0.25,
      scene={"robot": asset},
      command_manager=command_manager,
      extras={"log": {}},
    ),
  )
  asset_cfg = SceneEntityCfg("robot", site_names=("left", "right"), site_ids=[0, 1])

  result = feet_phase_alignment(
    env,
    command_name="twist",
    cycle_time=1.0,
    target_step_length=0.06,
    asset_cfg=asset_cfg,
  )

  expected_magnitude = math.tanh(1.0)
  torch.testing.assert_close(
    result,
    torch.tensor([expected_magnitude, expected_magnitude, -expected_magnitude]),
  )


def test_feet_phase_alignment_retains_gradient_beyond_target_step_length():
  """A split stance beyond the target must remain recoverable by optimization."""
  actual_separation = torch.tensor(0.12, requires_grad=True)
  asset = SimpleNamespace(
    data=SimpleNamespace(
      site_pos_w=torch.stack(
        (
          torch.stack((actual_separation / 2, torch.tensor(0.0), torch.tensor(0.0))),
          torch.stack((-actual_separation / 2, torch.tensor(0.0), torch.tensor(0.0))),
        )
      ).unsqueeze(0)
    )
  )
  command_manager = MagicMock()
  command_manager.get_command.return_value = torch.tensor([[0.2, 0.0, 0.0]])
  env = cast(
    "ManagerBasedRlEnv",
    SimpleNamespace(
      episode_length_buf=torch.tensor([0]),
      step_dt=0.25,
      scene={"robot": asset},
      command_manager=command_manager,
      extras={"log": {}},
    ),
  )
  asset_cfg = SceneEntityCfg("robot", site_names=("left", "right"), site_ids=[0, 1])

  result = feet_phase_alignment(
    env,
    command_name="twist",
    cycle_time=1.0,
    target_step_length=0.06,
    asset_cfg=asset_cfg,
  )
  result.sum().backward()

  assert actual_separation.grad is not None
  assert actual_separation.grad > 0.0


def test_feet_phase_contact_schedules_opposite_single_support():
  """Contact roles should swap between the two halves of the gait cycle."""
  contact_sensor = SimpleNamespace(
    data=SimpleNamespace(found=torch.tensor([[1, 0], [0, 1], [0, 1], [1, 0]]))
  )
  command_manager = MagicMock()
  command_manager.get_command.return_value = torch.tensor(
    [[0.2, 0.0, 0.0], [0.2, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.0, 0.0]]
  )
  env = cast(
    "ManagerBasedRlEnv",
    SimpleNamespace(
      episode_length_buf=torch.tensor([1, 1, 3, 1]),
      step_dt=0.25,
      scene={"feet": contact_sensor},
      command_manager=command_manager,
      extras={"log": {}},
    ),
  )

  result = feet_phase_contact(
    env,
    sensor_name="feet",
    command_name="twist",
    cycle_time=1.0,
  )

  torch.testing.assert_close(result, torch.tensor([1.0, -1.0, 1.0, 0.0]))
  torch.testing.assert_close(
    env.extras["log"]["Metrics/scheduled_single_support_rate"],
    torch.tensor(0.5),
  )


def test_feet_phase_height_schedules_opposite_swing_clearance():
  """Only the phase-scheduled swing foot should reach target clearance."""
  height_sensor = MagicMock(spec=TerrainHeightSensor)
  height_sensor.data.heights = torch.tensor(
    [[0.0, 0.04], [0.04, 0.0], [0.04, 0.0], [0.04, 0.0]]
  )
  command_manager = MagicMock()
  command_manager.get_command.return_value = torch.tensor(
    [[0.2, 0.0, 0.0], [0.2, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.0, 0.0]]
  )
  env = cast(
    "ManagerBasedRlEnv",
    SimpleNamespace(
      episode_length_buf=torch.tensor([1, 1, 3, 1]),
      step_dt=0.25,
      scene={"height": height_sensor},
      command_manager=command_manager,
      extras={"log": {}},
    ),
  )

  result = feet_phase_height(
    env,
    height_sensor_name="height",
    command_name="twist",
    cycle_time=1.0,
    target_height=0.04,
  )

  torch.testing.assert_close(result, torch.tensor([0.0, 2.0, 0.0, 0.0]))


def test_feet_phase_swing_velocity_schedules_opposite_forward_swing():
  """Only the phase-scheduled foot should receive forward-swing reward."""
  asset = SimpleNamespace(
    data=SimpleNamespace(
      site_lin_vel_w=torch.tensor(
        [
          [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]],
          [[0.3, 0.0, 0.0], [0.0, 0.0, 0.0]],
          [[0.0, 0.0, 0.0], [-0.3, 0.0, 0.0]],
          [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]],
        ]
      )
    )
  )
  command_manager = MagicMock()
  command_manager.get_command.return_value = torch.tensor(
    [[0.2, 0.0, 0.0], [0.2, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.0, 0.0]]
  )
  env = cast(
    "ManagerBasedRlEnv",
    SimpleNamespace(
      episode_length_buf=torch.tensor([1, 3, 1, 1]),
      step_dt=0.25,
      scene={"robot": asset},
      command_manager=command_manager,
      extras={"log": {}},
    ),
  )
  asset_cfg = SceneEntityCfg("robot", site_names=("left", "right"), site_ids=[0, 1])

  result = feet_phase_swing_velocity(
    env,
    command_name="twist",
    cycle_time=1.0,
    target_velocity=0.3,
    asset_cfg=asset_cfg,
  )

  torch.testing.assert_close(result, torch.tensor([1.0, 1.0, -1.0, 0.0]))
  torch.testing.assert_close(
    env.extras["log"]["Metrics/phase_swing_velocity_mean"], torch.tensor(0.1)
  )


def test_joints_phase_position_tracks_opposite_hip_and_swing_targets():
  """Joint targets should alternate the leading hip and flex the swing knee."""
  target_at_start = [0.1, 0.0, -0.1, -0.1, 0.0, 0.1]
  target_at_quarter_cycle = [0.0, 0.0, 0.0, 0.0, -0.2, 0.2]
  asset = SimpleNamespace(
    data=SimpleNamespace(
      joint_pos=torch.tensor([target_at_start, [0.0] * 6, target_at_quarter_cycle]),
      default_joint_pos=torch.zeros(3, 6),
    )
  )
  command_manager = MagicMock()
  command_manager.get_command.return_value = torch.tensor([[0.2, 0.0, 0.0]] * 3)
  env = cast(
    "ManagerBasedRlEnv",
    SimpleNamespace(
      episode_length_buf=torch.tensor([0, 0, 1]),
      step_dt=0.25,
      scene={"robot": asset},
      command_manager=command_manager,
      extras={"log": {}},
    ),
  )
  asset_cfg = SceneEntityCfg(
    "robot",
    joint_names=(
      "left_hip",
      "left_knee",
      "left_ankle",
      "right_hip",
      "right_knee",
      "right_ankle",
    ),
    joint_ids=list(range(6)),
  )

  result = joints_phase_position(
    env,
    command_name="twist",
    cycle_time=1.0,
    hip_amplitude=0.1,
    knee_amplitude=0.2,
    tolerance=0.1,
    asset_cfg=asset_cfg,
  )

  torch.testing.assert_close(
    result, torch.tensor([0.0, 2.0 / 3.0, 0.0]), atol=1e-6, rtol=0
  )


def test_feet_swing_forward_velocity_is_bounded_and_command_gated():
  """Relative forward swing is rewarded and backward swing is penalized."""
  contact_sensor = SimpleNamespace(
    data=SimpleNamespace(found=torch.tensor([[0, 1], [0, 0], [0, 1]]))
  )
  asset = SimpleNamespace(
    data=SimpleNamespace(
      site_lin_vel_w=torch.tensor(
        [
          [[0.15, 0.0, 0.0], [0.0, 0.0, 0.0]],
          [[-0.2, 0.0, 0.0], [0.6, 0.0, 0.0]],
          [[-0.15, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
      )
    )
  )
  command_manager = MagicMock()
  command_manager.get_command.return_value = torch.tensor(
    [[0.2, 0.0, 0.0], [0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]
  )
  env = cast(
    "ManagerBasedRlEnv",
    SimpleNamespace(
      scene={"robot": asset, "feet": contact_sensor},
      command_manager=command_manager,
      extras={"log": {}},
    ),
  )
  asset_cfg = SceneEntityCfg("robot", site_names=("left", "right"), site_ids=[0, 1])

  result = feet_swing_forward_velocity(
    env,
    sensor_name="feet",
    command_name="twist",
    target_velocity=0.3,
    asset_cfg=asset_cfg,
  )

  torch.testing.assert_close(result, torch.tensor([0.5, 0.0, -0.5]))


def test_feet_contact_flatness_allows_heel_strike_then_penalizes_pitch():
  """Foot pitch is penalized only after contact settles and while moving."""
  half_angle = math.radians(30.0) / 2.0
  flat = [1.0, 0.0, 0.0, 0.0]
  pitched = [math.cos(half_angle), 0.0, math.sin(half_angle), 0.0]
  foot_quat_w = torch.tensor(
    [[flat, pitched], [flat, pitched], [flat, pitched]], dtype=torch.float32
  )
  contact_sensor = SimpleNamespace(
    data=SimpleNamespace(
      current_contact_time=torch.tensor([[0.1, 0.1], [0.02, 0.02], [0.1, 0.1]])
    )
  )
  asset = SimpleNamespace(data=SimpleNamespace(site_quat_w=foot_quat_w))
  command_manager = MagicMock()
  command_manager.get_command.return_value = torch.tensor(
    [[0.2, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.0, 0.0]]
  )
  env = cast(
    "ManagerBasedRlEnv",
    SimpleNamespace(
      scene={"robot": asset, "feet": contact_sensor},
      command_manager=command_manager,
      extras={"log": {}},
    ),
  )
  asset_cfg = SceneEntityCfg("robot", site_names=("left", "right"), site_ids=[0, 1])

  result = feet_contact_flatness(
    env,
    sensor_name="feet",
    command_name="twist",
    settle_time=0.04,
    asset_cfg=asset_cfg,
  )

  torch.testing.assert_close(result, torch.tensor([0.5, 0.0, 0.0]), atol=1e-6, rtol=0)


def test_both_feet_contact_is_command_gated():
  """Double support is penalized only while a motion command is active."""
  contact_sensor = SimpleNamespace(
    data=SimpleNamespace(found=torch.tensor([[1, 1], [1, 0], [1, 1]]))
  )
  command_manager = MagicMock()
  command_manager.get_command.return_value = torch.tensor(
    [[0.2, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.0, 0.0]]
  )
  env = cast(
    "ManagerBasedRlEnv",
    SimpleNamespace(
      scene={"feet": contact_sensor},
      command_manager=command_manager,
      extras={"log": {}},
    ),
  )

  result = both_feet_contact(env, sensor_name="feet", command_name="twist")

  torch.testing.assert_close(result, torch.tensor([1.0, 0.0, 0.0]))
