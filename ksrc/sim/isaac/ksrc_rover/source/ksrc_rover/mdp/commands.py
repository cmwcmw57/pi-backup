"""Velocity command with raw tracking-error metrics for TensorBoard (2026-10-12).

Isaac Lab's UniformVelocityCommand logs error_vel_* as an episode SUM divided by the
max command duration, so with 2-6 s commands in a 16 s episode it is not m/s (~x2.7) and
moves with episode length. This term keeps those (for continuity) and adds per-step
means in m/s, rad/s, split like scripts/eval_analyze.py:
  err_xy / err_yaw                 all steps
  err_xy_steady / err_yaw_steady   >= TRANSIENT_S after a command change or reset
  err_xy_slick / err_yaw_slick     steady steps with 1-3 slick wheels (mu < SLICK_MU) --
                                   the part a policy can actually improve
  spin_slick / spin_grip           mean |surface - ground speed| [m/s] of slick / grippy wheels
Logged as Metrics/base_velocity/<name>. Values are pooled over the envs that reset at a
step (sum / count), so a short episode does not weigh as much as a full one.
"""

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass

TRANSIENT_S = 1.0
SLICK_MU = 0.2
_KEYS = ("err_xy", "err_yaw", "err_xy_steady", "err_yaw_steady", "err_xy_slick", "err_yaw_slick",
         "spin_slick", "spin_grip")


class RoverVelocityCommand(UniformVelocityCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n, k = self.num_envs, len(_KEYS)
        self._sum = torch.zeros(n, k, device=self.device)
        self._cnt = torch.zeros(n, k, device=self.device)
        self._t_cmd = torch.zeros(n, device=self.device)
        self._last = {key: 0.0 for key in _KEYS}   # logged when no reset env has such steps

    def _update_metrics(self):
        super()._update_metrics()
        from .rewards import _wheel_surface_and_ground_speed

        e_xy = torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1)
        e_yaw = torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        steady = (self._t_cmd >= TRANSIENT_S).float()
        mu4 = getattr(self._env, "_wheel_mu4", None)   # set by randomize_wheel_mu_patch (CPU)
        if mu4 is not None and not torch.isnan(mu4).any():
            slick = mu4.to(self.device) < SLICK_MU                          # (N,4)
        else:
            slick = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        n_slick = slick.sum(dim=1)
        part = steady * ((n_slick >= 1) & (n_slick <= 3)).float()
        surf, v_long = _wheel_surface_and_ground_speed(self._env)
        spin = (surf - v_long).abs()
        sf = slick.float()
        one = torch.ones_like(e_xy)
        vals = torch.stack((e_xy, e_yaw, e_xy * steady, e_yaw * steady, e_xy * part, e_yaw * part,
                            (spin * sf).sum(1), (spin * (1 - sf)).sum(1)), dim=1)
        cnts = torch.stack((one, one, steady, steady, part, part, sf.sum(1), (1 - sf).sum(1)), dim=1)
        self._sum += vals
        self._cnt += cnts
        self._t_cmd += self._env.step_dt

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        self._t_cmd[env_ids] = 0.0

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        s, c = self._sum[ids].sum(dim=0), self._cnt[ids].sum(dim=0)
        extras = super().reset(env_ids)
        for i, key in enumerate(_KEYS):
            if c[i] > 0:
                self._last[key] = (s[i] / c[i]).item()
            extras[key] = self._last[key]
        self._sum[ids] = 0.0
        self._cnt[ids] = 0.0
        return extras


@configclass
class RoverVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = RoverVelocityCommand
