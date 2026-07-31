"""Tests for play command configuration overrides."""

from types import SimpleNamespace
from typing import cast

import pytest

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scripts.play import _apply_velocity_command_override
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg


def _velocity_command_cfg() -> UniformVelocityCommandCfg:
  return UniformVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(1.0, 2.0),
    rel_standing_envs=0.1,
    rel_heading_envs=0.2,
    rel_world_envs=0.3,
    rel_forward_envs=0.4,
    heading_command=True,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-1.0, 1.0),
      lin_vel_y=(-0.5, 0.5),
      ang_vel_z=(-0.5, 0.5),
      heading=(-3.14, 3.14),
    ),
  )


def test_apply_velocity_command_override_sets_fixed_forward_command() -> None:
  twist_cmd = _velocity_command_cfg()
  env_cfg = cast(ManagerBasedRlEnvCfg, SimpleNamespace(commands={"twist": twist_cmd}))

  _apply_velocity_command_override(env_cfg, 0.2)

  assert twist_cmd.ranges.lin_vel_x == (0.2, 0.2)
  assert twist_cmd.ranges.lin_vel_y == (0.0, 0.0)
  assert twist_cmd.ranges.ang_vel_z == (0.0, 0.0)
  assert twist_cmd.terminal_log_interval_s == 1.0
  assert twist_cmd.ranges.heading is None
  assert not twist_cmd.heading_command
  assert twist_cmd.rel_standing_envs == 0.0
  assert twist_cmd.rel_heading_envs == 0.0
  assert twist_cmd.rel_world_envs == 0.0
  assert twist_cmd.rel_forward_envs == 0.0


def test_apply_velocity_command_override_rejects_non_velocity_task() -> None:
  env_cfg = cast(ManagerBasedRlEnvCfg, SimpleNamespace(commands={}))

  with pytest.raises(ValueError, match="twist command"):
    _apply_velocity_command_override(env_cfg, 0.2)


@pytest.mark.parametrize("command_x", [float("nan"), float("inf")])
def test_apply_velocity_command_override_rejects_nonfinite_speed(
  command_x: float,
) -> None:
  env_cfg = cast(
    ManagerBasedRlEnvCfg,
    SimpleNamespace(commands={"twist": _velocity_command_cfg()}),
  )

  with pytest.raises(ValueError, match="finite number"):
    _apply_velocity_command_override(env_cfg, command_x)
