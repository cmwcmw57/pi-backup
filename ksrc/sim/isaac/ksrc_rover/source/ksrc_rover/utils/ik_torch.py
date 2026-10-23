"""Batched port of firmware/common/swerve_kinematics.c, including stateful fold."""

import math

import torch

from .contract import MODULE_XY_M, STEER_LIMIT_RAD, WHEEL_MAX_M_S

FOLD_EPS_RAD = 1e-4     # firmware swerve_kinematics.c SWERVE_FOLD_EPS_RAD


class BatchedSwerveIK:
    def __init__(self, num_envs: int, device: str):
        self.xy = torch.tensor(MODULE_XY_M, device=device)
        self.last_angle = torch.zeros(num_envs, 4, device=device)
        self.last_cmd = torch.zeros(num_envs, 4, device=device)

    def reset(self, env_ids=None):
        self.last_angle[env_ids] = 0.0
        self.last_cmd[env_ids] = 0.0

    def compute(self, command: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vx = command[:, 0:1] - command[:, 2:3] * self.xy[:, 1]
        vy = command[:, 1:2] + command[:, 2:3] * self.xy[:, 0]
        speed = torch.sqrt(vx.square() + vy.square())
        moving = speed >= 0.001
        raw = torch.atan2(vy, vx)
        delta = torch.remainder(raw - self.last_angle + math.pi, 2 * math.pi) - math.pi
        flip = delta.abs() > math.pi / 2
        delta = torch.where(flip, delta - torch.sign(delta) * math.pi, delta)
        speed = torch.where(flip, -speed, speed)
        theta = torch.where(moving, self.last_angle + delta, self.last_angle)
        speed = torch.where(moving, speed, 0.0)
        self.last_angle[:] = theta
        max_speed = speed.abs().amax(dim=1, keepdim=True)
        speed = speed * torch.clamp(WHEEL_MAX_M_S / torch.clamp_min(max_speed, 1e-9), max=1.0)

        n = torch.floor((self.last_cmd - theta) / math.pi + 0.5)
        folded = theta + n * math.pi
        # 2026-10-14: same rounding tolerance as swerve_kinematics.c SWERVE_FOLD_EPS_RAD (a pure sideways
        # command used to toggle +90/-90 every cycle on a 1-ulp overshoot)
        n = torch.where(folded > STEER_LIMIT_RAD + FOLD_EPS_RAD, n - 1, n)
        n = torch.where(folded < -STEER_LIMIT_RAD - FOLD_EPS_RAD, n + 1, n)
        folded = torch.clamp(theta + n * math.pi, -STEER_LIMIT_RAD, STEER_LIMIT_RAD)
        speed = torch.where(torch.remainder(n.abs(), 2) == 1, -speed, speed)
        self.last_cmd[:] = folded
        return folded, speed
