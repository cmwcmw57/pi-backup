"""Velocity tracking rewards; action penalties are intentionally kept small."""

import torch

from isaaclab.managers import ManagerTermBase


def track_linear_velocity(env, std: float):
    command = env.command_manager.get_command("base_velocity")[:, :2]
    actual = env.scene["robot"].data.root_lin_vel_b[:, :2]
    return torch.exp(-torch.sum((command - actual).square(), dim=1) / std**2)


def track_angular_velocity(env, std: float):
    command = env.command_manager.get_command("base_velocity")[:, 2]
    actual = env.scene["robot"].data.root_ang_vel_b[:, 2]
    return torch.exp(-(command - actual).square() / std**2)


# --- Bounded linear tracking reward (2026-10-12) ---------------------------------
# w * clip(1 - |err| / e_max, 0, 1): constant slope up to e_max, so errors beyond ~0.15 m/s
# (slick wheels, where exp(std 0.05) is flat) still give a learning signal. Kept >= 0 on
# purpose: a negative per-step term makes an early tilt termination look attractive.


def track_lin_vel_linear(env, e_max: float):
    command = env.command_manager.get_command("base_velocity")[:, :2]
    actual = env.scene["robot"].data.root_lin_vel_b[:, :2]
    err = torch.linalg.norm(command - actual, dim=1)
    return (1.0 - err / e_max).clamp(0.0, 1.0)


def track_ang_vel_linear(env, e_max: float):
    command = env.command_manager.get_command("base_velocity")[:, 2]
    actual = env.scene["robot"].data.root_ang_vel_b[:, 2]
    return (1.0 - (command - actual).abs() / e_max).clamp(0.0, 1.0)


# --- Plan B: dense linear error terms (non-saturating gradient far from goal). ---
# Add as RewTerms ONLY if the tight exp reward stalls early; the exp terms stay as
# the fine "terminal-convergence" shaping, these carry coarse early learning.
def lin_vel_error_abs(env):
    command = env.command_manager.get_command("base_velocity")[:, :2]
    actual = env.scene["robot"].data.root_lin_vel_b[:, :2]
    return -torch.sqrt(torch.sum((command - actual).square(), dim=1) + 1e-8)


def ang_vel_error_abs(env):
    command = env.command_manager.get_command("base_velocity")[:, 2]
    actual = env.scene["robot"].data.root_ang_vel_b[:, 2]
    return -torch.abs(command - actual)


# --- Slip penalty (anti-dig / left-right balance) -------------------------------
# Mean per-wheel longitudinal slip magnitude. Velocity-only reward never asks the
# policy to stop spinning the wheels, so it leaves slip unaddressed (only Direct8
# stumbled into reducing it). On 5 cm sand a spinning wheel digs in and bogs, so we
# penalize slip directly -- and since the high-|Δμ| side slips most, pressing mean
# |slip| down also eases the left-right imbalance. Returns a positive quantity;
# use a NEGATIVE weight.
def slip_magnitude(env):
    return wheel_slip_ratio(env).abs().mean(dim=1)           # (N,) positive


def slip_speed_sq(env, deadzone: float):
    """Sum over wheels of max(|surface - ground speed| - deadzone, 0)^2 [m^2/s^2]. (2026-10-12)

    slip_magnitude barely tells slick wheels from grippy ones (IK eval: 0.265 vs 0.157 per
    step): the ratio blows up at low wheel speed and the 4-wheel mean dilutes one spinning
    wheel. Here absolute speed + deadzone zeroes the everyday slip of grippy wheels (~0.035
    m/s), the square lets the spinning wheel dominate, and the sum keeps it undiluted
    (IK eval: 0.071 vs 0.017). Sim has no sinkage, so this is the only cost of spinning.
    Positive; use a NEGATIVE weight."""
    surf, v_long = _wheel_surface_and_ground_speed(env)
    return ((surf - v_long).abs() - deadzone).clamp(min=0.0).square().sum(dim=1)


def wheel_slip_ratio(env):
    """Signed per-wheel longitudinal slip (N,4); also the teacher's privileged slip obs."""
    surf, v_long = _wheel_surface_and_ground_speed(env)
    denom = torch.maximum(torch.maximum(surf.abs(), v_long.abs()),
                          torch.full_like(surf, 0.05))
    return (surf - v_long) / denom


def _wheel_surface_and_ground_speed(env):
    """(surface speed w*r, ground speed of the wheel center along its rolling direction), (N,4)."""
    from ksrc_rover.utils.contract import MODULE_XY_M, WHEEL_RADIUS_M
    robot = env.scene["robot"]
    term = env.action_manager.get_term("rover")
    steer = robot.data.joint_pos[:, term.steer_ids]          # (N,4) module order
    ww = robot.data.joint_vel[:, term.wheel_ids]             # (N,4)
    vx = robot.data.root_lin_vel_b[:, 0:1]
    vy = robot.data.root_lin_vel_b[:, 1:2]
    wz = robot.data.root_ang_vel_b[:, 2:3]
    xy = torch.as_tensor(MODULE_XY_M, device=steer.device, dtype=steer.dtype)  # (4,2)
    x = xy[:, 0].unsqueeze(0)                                 # (1,4)
    y = xy[:, 1].unsqueeze(0)
    v_long = (vx - wz * y) * torch.cos(steer) + (vy + wz * x) * torch.sin(steer)
    return ww * WHEEL_RADIUS_M, v_long


# --- Smoothness (SLAM camera protection) -----------------------------------------
# Body-shake (roll/pitch rate) and joint-velocity penalties come from legged/arm
# tasks and do not fit here: on rough terrain the shake is mostly terrain-driven,
# and wheels must spin to track. Instead penalize what the policy controls -- how
# its commands change -- plus the yaw acceleration that dominates image flow.
# Both keep their own history and stay 0 until it is filled after a reset, so the
# jump from the zeroed reset action is not punished. Positive; use NEGATIVE weight.
class ActionDiffPenalty(ManagerTermBase):
    """Squared 1st (order=1) or 2nd (order=2) difference of the commanded joint targets.

    Reads the action term's ``commanded_targets`` (steer angle / wheel speed the action
    space produced, before the servo rate limit), normalized by STEER_LIMIT_RAD and
    WHEEL_MAX_RAD_S. Every action space (Direct8 / Res8 / Residual3) ends in these same
    8 joint commands, so the same weights mean the same physical penalty for all; for
    Direct8 the normalized target equals the clamped raw action.
    part="steer" -> wheels' steer [:4], part="wheel" -> wheel speed [4:]. The 2nd
    difference a_t - 2a_{t-1} + a_{t-2} is 0 on a constant-rate ramp, so it leaves smooth
    turns alone and only hits back-and-forth wobble.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        from ksrc_rover.utils.contract import STEER_LIMIT_RAD, WHEEL_MAX_RAD_S
        self._term = env.action_manager.get_term("rover")
        self._scale = torch.tensor([1.0 / STEER_LIMIT_RAD] * 4 + [1.0 / WHEEL_MAX_RAD_S] * 4, device=env.device)
        self._a1 = torch.zeros(env.num_envs, 8, device=env.device)   # a_{t-1}
        self._a2 = torch.zeros(env.num_envs, 8, device=env.device)   # a_{t-2}
        self._count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._a1[env_ids] = 0.0
        self._a2[env_ids] = 0.0
        self._count[env_ids] = 0

    def __call__(self, env, part: str = "steer", order: int = 1):
        a = self._term.commanded_targets * self._scale
        d = a - self._a1 if order == 1 else a - 2.0 * self._a1 + self._a2
        d = d[:, :4] if part == "steer" else d[:, 4:]
        pen = d.square().sum(dim=1) * (self._count >= order)
        self._a2[:] = self._a1
        self._a1[:] = a
        self._count += 1
        return pen


class YawAccelPenalty(ManagerTermBase):
    """Squared body yaw acceleration [(rad/s^2)^2], finite-differenced over one env step."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._prev = torch.zeros(env.num_envs, device=env.device)
        self._count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._prev[env_ids] = 0.0
        self._count[env_ids] = 0

    def __call__(self, env):
        wz = env.scene["robot"].data.root_ang_vel_b[:, 2]
        acc = (wz - self._prev) / env.step_dt
        pen = acc.square() * (self._count >= 1)
        self._prev[:] = wz
        self._count += 1
        return pen
