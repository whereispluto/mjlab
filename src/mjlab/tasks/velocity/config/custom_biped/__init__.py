from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  custom_biped_flat_forward_env_cfg,
  custom_biped_flat_forward_no_lin_vel_env_cfg,
)
from .rl_cfg import custom_biped_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Forward-Custom-Biped",
  env_cfg=custom_biped_flat_forward_env_cfg(),
  play_env_cfg=custom_biped_flat_forward_env_cfg(play=True),
  rl_cfg=custom_biped_ppo_runner_cfg("custom_biped_velocity"),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Forward-Custom-Biped-NoLinVel",
  env_cfg=custom_biped_flat_forward_no_lin_vel_env_cfg(),
  play_env_cfg=custom_biped_flat_forward_no_lin_vel_env_cfg(play=True),
  rl_cfg=custom_biped_ppo_runner_cfg("custom_biped_velocity_nolinvel"),
  runner_cls=VelocityOnPolicyRunner,
)
