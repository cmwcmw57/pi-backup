"""Sensor pipeline as the real rover computes it (student observations), batched torch.

Same math runs on the Pi from SPI telemetry: pi/common/sensor_fusion.py (numpy twin,
compared in sim/rl/test_actuators.py).  No Isaac Lab import.

  servo_angle      STS3215 position register: 4096 steps / rev (0.088 deg), servo shaft side
  EncoderSpeed     wheel speed = encoder tick difference over one control period (main.c, 20 ms)
  GravityFilter    gravity direction from gyro + accelerometer (complementary filter):
                   the real IMU gives raw rates/specific force, nobody hands you projected gravity
"""

from __future__ import annotations

import math

import torch

from .actuators_torch import ENCODER_TICKS_PER_REV

SERVO_STEPS_PER_REV = 4096
GRAVITY = 9.81
GRAV_TAU_S = 0.5            # accelerometer correction time constant
GRAV_GATE = 1.5             # [m/s^2] skip the correction when | |f| - g | exceeds this (bumps, hard accel)


def servo_angle(angle: torch.Tensor) -> torch.Tensor:
    """Angle as the servo reports it: rounded to its 4096-step grid."""
    q = 2.0 * math.pi / SERVO_STEPS_PER_REV
    return torch.round(angle / q) * q


class EncoderSpeed:
    """(N,4) wheel joint angle each control period -> encoder speed [rad/s].

    Angle increments are wrapped to (-pi, pi] (one period moves < 0.2 rad) and accumulated, so a
    joint position that wraps in the physics engine does not matter.  After reset(ids) the next
    call re-zeroes those envs (speed 0, like a fresh counter on a standing rover)."""

    def __init__(self, num_envs: int, dt: float, device="cpu"):
        self.dt = float(dt)
        self.prev_pos = torch.zeros(num_envs, 4, device=device)
        self.acc = torch.zeros(num_envs, 4, device=device)
        self.prev_ticks = torch.zeros(num_envs, 4, device=device)
        self.need_init = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.out = torch.zeros(num_envs, 4, device=device)

    def reset(self, env_ids=None):
        self.need_init[slice(None) if env_ids is None else env_ids] = True

    def step(self, pos: torch.Tensor) -> torch.Tensor:
        d = torch.remainder(pos - self.prev_pos + math.pi, 2.0 * math.pi) - math.pi
        self.acc += d
        self.prev_pos = pos.clone()
        ticks = torch.floor(self.acc * (ENCODER_TICKS_PER_REV / (2.0 * math.pi)))
        self.out = (ticks - self.prev_ticks) * (2.0 * math.pi / ENCODER_TICKS_PER_REV) / self.dt
        self.prev_ticks = ticks
        self.init_where(self.need_init, pos)
        return self.out

    def init_where(self, m, pos):
        if m.any():
            self.prev_pos[m] = pos[m]
            self.acc[m] = 0.0
            self.prev_ticks[m] = 0.0
            self.out[m] = 0.0
            self.need_init[m] = False


class GravityFilter:
    """Complementary filter: g (N,3) = unit 'down' direction in the body frame (= projected gravity).

    predict  g <- g - dt * (w x g)            (a world-fixed vector seen from a rotating body)
    correct  g <- normalize((1-a) g + a * (-f/|f|)),  a = dt/tau if | |f| - 9.81 | < gate else 0
    f = accelerometer specific force [m/s^2] (+9.81 up when standing), w = gyro [rad/s].
    After reset: g = (0,0,-1) until the first accelerometer sample inside the gate, then g = -f/|f|
    (a rover spawned in the air, or bumped at boot, must not lock onto a free-fall reading)."""

    def __init__(self, num_envs: int, dt: float, device="cpu", tau=GRAV_TAU_S, gate=GRAV_GATE):
        self.dt, self.alpha, self.gate = float(dt), float(dt) / tau, gate
        self.g = torch.zeros(num_envs, 3, device=device)
        self.g[:, 2] = -1.0
        self.need_init = torch.ones(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.need_init[ids] = True
        self.g[ids] = torch.tensor([0.0, 0.0, -1.0], device=self.g.device)

    def init_from(self, acc: torch.Tensor):
        """Initialize waiting envs whose accelerometer sample is inside the gate (no predict step)."""
        fn = torch.linalg.norm(acc, dim=-1, keepdim=True).clamp(min=1e-6)
        ok = (fn[:, 0] - GRAVITY).abs() < self.gate
        m = self.need_init & ok
        self.g = torch.where(m.unsqueeze(-1), -acc / fn, self.g)
        self.need_init &= ~ok
        return self.g

    def step(self, gyro: torch.Tensor, acc: torch.Tensor) -> torch.Tensor:
        fn = torch.linalg.norm(acc, dim=-1, keepdim=True).clamp(min=1e-6)
        a_dir = -acc / fn
        ok = (fn - GRAVITY).abs() < self.gate
        pred = self.g - self.dt * torch.linalg.cross(gyro, self.g, dim=-1)
        a = torch.where(ok, torch.full_like(fn, self.alpha), torch.zeros_like(fn))
        g = (1.0 - a) * pred + a * a_dir
        g = g / torch.linalg.norm(g, dim=-1, keepdim=True).clamp(min=1e-6)
        wait = self.need_init.unsqueeze(-1)
        self.g = torch.where(wait, torch.where(ok, a_dir, self.g), g)
        self.need_init &= ~ok[:, 0]
        return self.g
