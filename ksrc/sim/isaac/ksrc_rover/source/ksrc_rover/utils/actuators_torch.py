"""Actuator models shared by the Isaac task (batched, GPU) and sim2sim (MuJoCo, CPU, N=1).

Pure torch, no Isaac Lab import -- like ik_torch.py, so sim/rl/sim2sim_isaac.py and the laptop
tests can use the exact same code the policy was trained with.

  SteerServo   STS3215 12 V as measured 2026-10-13: dead time -> deadband -> trapezoid profile.
               Stepped every physics step (5 ms); output is the PD position target.
  WheelPI      Batched port of firmware/common/wheel_pi.c (FF + reference-model PI, 20 ms).
  DriveGate    Batched port of firmware/common/drive_align_gate.c (hold the drive while the steering
               turns through a big direction change).
               tests/test_actuators.py checks it against the C (sim/libksrc_common.so).
  drive_torque Duty -> wheel torque of the open-loop DC motor (what the PI drives).
  wheel_friction  Gearbox friction (kinetic + Stribeck + stiction, bench 10-13) as an explicit torque.
  encoder_speed  Wheel speed as the Nucleo measures it: tick difference over one control period.
"""

from __future__ import annotations

import math

import torch

from .contract import (
    DRIVE_EFFORT_NM, STEER_ACCEL_RAD_S2, STEER_DEADBAND_RAD, STEER_DELAY_S, STEER_MAX_RAD_S, WHEEL_MAX_RAD_S,
)

ENCODER_TICKS_PER_REV = 8384          # 16 PPR x4 x 131 (firmware main.c TICKS_PER_REV)
PWM_MAX = 799.0
# Gearbox friction in duty units, 2026-10-13 bench (firmware/common/tests/wheel_pi_design.py):
# kinetic 5 + Stribeck 41*exp(-|w|/0.13) (ramp-down stops at ~33 duty), stiction 47 (start 35-52).
FRICTION_DUTY = dict(kinetic=5.0, stribeck=41.0, w_stribeck=0.13, static=47.0)


class SteerServo:
    """goal (N,4) rad per physics step -> PD position target (N,4).

    1) dead time: the goal written ``delay_ticks`` physics steps ago (ring buffer, max 60 ms)
    2) deadband: goals within ``deadband`` of the current profile position are ignored
    3) trapezoid: accelerate/decelerate at ``accel * rate_scale`` up to ``vmax * rate_scale``,
       stop exactly on the goal when it is crossed
    Per-env parameters are tensors so DR (mdp/dr.py) can write them at reset.
    """

    def __init__(self, num_envs: int, phys_dt: float, device="cpu"):
        self.n, self.dt, self.device = num_envs, float(phys_dt), device
        self.applied = torch.zeros(num_envs, 4, device=device)
        self.vel = torch.zeros(num_envs, 4, device=device)
        self.goal_hist = torch.zeros(int(round(0.06 / self.dt)) + 1, num_envs, 4, device=device)
        self.delay_ticks = torch.full((num_envs,), int(round(STEER_DELAY_S / self.dt)), dtype=torch.long, device=device)
        self.deadband = torch.full((num_envs, 1), STEER_DEADBAND_RAD, device=device)
        self.rate_scale = torch.ones(num_envs, 1, device=device)
        self.offset = torch.zeros(num_envs, 4, device=device)     # linkage play / zero error
        self._rows = torch.arange(num_envs, device=device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.applied[ids] = 0.0
        self.vel[ids] = 0.0
        self.goal_hist[:, ids] = 0.0

    def step(self, goal: torch.Tensor) -> torch.Tensor:
        self.goal_hist = torch.roll(self.goal_hist, 1, dims=0)
        self.goal_hist[0] = goal
        d = self.delay_ticks.clamp(max=self.goal_hist.shape[0] - 1)
        g = self.goal_hist[d, self._rows]
        err = g - self.applied
        err = torch.where(err.abs() <= self.deadband, torch.zeros_like(err), err)
        a = STEER_ACCEL_RAD_S2 * self.rate_scale
        v_des = torch.sign(err) * torch.minimum(torch.sqrt(2.0 * a * err.abs()), STEER_MAX_RAD_S * self.rate_scale)
        self.vel += torch.clamp(v_des - self.vel, -a * self.dt, a * self.dt)
        pos = self.applied + self.vel * self.dt
        crossed = (err != 0) & ((g - pos) * err <= 0)          # arrived this step: stop on the goal
        self.applied[:] = torch.where(crossed, g, pos)
        self.vel[:] = torch.where(crossed, torch.zeros_like(pos), self.vel)
        return self.applied + self.offset


class WheelPI:
    """Batched firmware/common/wheel_pi.c.  step(w_ref, w_meas) (N,4) rad/s -> duty (N,4), truncated like
    the C int16 cast.  Defaults = wheel_pi_default_cfg (keep in sync; the test compares against the C).
    Protections as in wheel_pi.h: correction cap, stall (FF only until the wheel turns again or the
    command goes to zero), encoder direction mismatch (FF only for enc_retry_t, then PI again; locked
    after enc_trips_max trips within enc_window_t until clear_fault)."""

    DEFAULTS = dict(dt=0.020, k_duty=0.0097, f_kin=5.0, kick=40.0, w_still=0.01, w_zero=0.001,
                    kp=82.5, ki=2062.0, i_max=200.0, tau_m=0.040, duty_max=799.0,
                    corr_max=250.0, stall_w=0.05, stall_t=0.5, enc_duty=150.0, enc_w=0.3, enc_t=0.3,
                    enc_retry_t=1.0, enc_window_t=10.0, enc_trips_max=3.0)

    def __init__(self, num_envs: int, device="cpu", **cfg):
        unknown = set(cfg) - set(self.DEFAULTS)
        if unknown:
            raise KeyError(unknown)
        self.cfg = dict(self.DEFAULTS, **cfg)
        z = lambda dt: torch.zeros(num_envs, 4, device=device, dtype=dt)
        self.r_m, self.i = z(torch.float32), z(torch.float32)
        # counters as float32 (exact for these sizes), flags as bool
        self.stall_n, self.enc_n = z(torch.float32), z(torch.float32)
        self.enc_hold, self.enc_win, self.enc_trips = z(torch.float32), z(torch.float32), z(torch.float32)
        self.stall, self.enc_dir = z(torch.bool), z(torch.bool)

    def reset(self, env_ids=None, mask=None):
        """wheel_pi_reset for whole envs (env_ids) or single wheels (mask (N,4) bool):
        the direction-mismatch state (enc_dir, retry, window) stays."""
        ids = mask if mask is not None else (slice(None) if env_ids is None else env_ids)
        for t in (self.r_m, self.i, self.stall_n, self.enc_n):
            t[ids] = 0.0
        self.stall[ids] = False

    def clear_fault(self, env_ids=None):
        """wheel_pi_clear_fault: everything to zero (boot / WPI ON / new episode)."""
        ids = slice(None) if env_ids is None else env_ids
        self.reset(env_ids)
        for t in (self.enc_hold, self.enc_win, self.enc_trips):
            t[ids] = 0.0
        self.enc_dir[ids] = False

    def step(self, w_ref: torch.Tensor, w_meas: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        f32 = torch.float32
        w_ref, w_meas = w_ref.to(f32), w_meas.to(f32)
        T = lambda v: torch.tensor(v, dtype=f32, device=w_ref.device)
        dt32 = T(c["dt"])
        off = ~torch.isfinite(w_ref) | ~torch.isfinite(w_meas) | (w_ref.abs() < c["w_zero"])
        if c["tau_m"] > 0:      # same float32 operation order as the C
            r_m = self.r_m + (w_ref - self.r_m) * dt32 / (T(c["tau_m"]) + dt32)
        else:
            r_m = w_ref.clone()
        e = r_m - w_meas
        sg = torch.sign(w_ref)
        ff0 = w_ref / c["k_duty"] + sg * c["f_kin"]
        # protection 3 window: forget the trips enc_window_t after the first one
        win_on = (self.enc_trips > 0) & ~self.enc_dir
        win = torch.where(win_on, self.enc_win + 1, self.enc_win)
        expire = win_on & (win * dt32 >= T(c["enc_window_t"]) - T(1e-4))
        win = torch.where(expire, T(0.0), win)
        trips = torch.where(expire, T(0.0), self.enc_trips)
        # protections 2/3 active: FF only (stall ends when the wheel turns again, retry when the time is up)
        holding = self.enc_hold > 0
        enc_hold = torch.where(holding, self.enc_hold - 1, self.enc_hold)
        freed = ~holding & self.stall & ~self.enc_dir & (w_meas.abs() >= T(0.5) * w_ref.abs()) & (torch.sign(w_meas) == sg)
        stall = self.stall & ~freed
        stall_n = torch.where(freed, torch.zeros_like(self.stall_n), self.stall_n)
        hold = stall | self.enc_dir | (enc_hold > 0)
        ff = ff0 + torch.where(w_meas.abs() < c["w_still"], sg * c["kick"], torch.zeros_like(ff0))
        c_pre = c["kp"] * e + self.i
        u_pre = ff + c_pre
        sat = ((u_pre.abs() >= c["duty_max"]) | (c_pre.abs() >= c["corr_max"])) & (torch.sign(u_pre) == torch.sign(e))
        i = torch.where(sat, self.i, self.i + c["ki"] * e * c["dt"]).clamp(-c["i_max"], c["i_max"])
        corr = (c["kp"] * e + i).clamp(-c["corr_max"], c["corr_max"])
        u = (ff + corr).clamp(-c["duty_max"], c["duty_max"])
        stall_n_new = torch.where((corr * sg >= c["corr_max"]) & (w_meas.abs() < c["stall_w"]), stall_n + 1, T(0.0))
        new_stall = stall_n_new * dt32 >= T(c["stall_t"]) - T(1e-4)
        enc_n = torch.where((u.abs() >= c["enc_duty"]) & (w_meas.abs() >= c["enc_w"]) & (torch.sign(w_meas) == -torch.sign(u)),
                            self.enc_n + 1, T(0.0))
        trip = enc_n * dt32 >= T(c["enc_t"]) - T(1e-4)
        enc_n = torch.where(trip, T(0.0), enc_n)
        win = torch.where(trip & (trips == 0), T(0.0), win)
        trips_new = torch.where(trip, (trips + 1).clamp(max=255.0), trips)
        lock = trip & (trips_new >= c["enc_trips_max"])
        hold_ticks = float(int(torch.tensor(c["enc_retry_t"], dtype=f32) / dt32 + T(0.5)))
        enc_hold_new = torch.where(trip & ~lock, T(hold_ticks), enc_hold)
        i = torch.where(new_stall | trip, torch.zeros_like(i), i)
        zero = torch.zeros_like(u)
        run = ~off & ~hold                       # the PI branch of the C
        self.r_m = torch.where(off, zero, r_m)
        self.i = torch.where(run, i, zero)
        self.stall_n = torch.where(off, zero, torch.where(hold, stall_n, stall_n_new))
        self.enc_n = torch.where(off, zero, torch.where(hold, self.enc_n, enc_n))
        self.stall = torch.where(off, torch.zeros_like(stall), torch.where(hold, stall, new_stall))
        self.enc_win = torch.where(off, self.enc_win, torch.where(hold, win, win))
        self.enc_trips = torch.where(off, self.enc_trips, torch.where(run, trips_new, trips))
        self.enc_hold = torch.where(off, self.enc_hold, torch.where(run, enc_hold_new, enc_hold))
        self.enc_dir = self.enc_dir | (run & lock)
        out = torch.where(hold, ff0.clamp(-c["duty_max"], c["duty_max"]), u)
        return torch.where(off, zero, torch.trunc(torch.nan_to_num(out)))


class DriveGate:
    """Batched firmware/common/drive_align_gate.c, called like main.c Swerve_Control every control tick:
    step(goal, speed, actual) -> allow (N,) bool.  goal = steering goal after IK fold (+ RL residual),
    speed = wheel commands (only != 0 matters), actual = measured servo angle.  A reset env starts
    with no previous goal, so its first drive command waits for alignment, as after power-on.
    main.c values: trigger 60 deg, release 10 deg, max hold 1500 ms / 20 ms = 75 ticks.  The servo
    reads are taken as always valid (actual_ok = 1)."""

    SPEED_EPS = 1e-6

    def __init__(self, num_envs: int, device="cpu", trigger_deg=60.0, release_deg=10.0, max_hold_ticks=75):
        f32 = torch.float32
        self.trigger = torch.tensor(math.radians(trigger_deg), dtype=f32)
        self.release = torch.tensor(math.radians(release_deg), dtype=f32)
        self.max_hold = int(max_hold_ticks)
        z = lambda dt, k=None: torch.zeros((num_envs,) if k is None else (num_envs, k), dtype=dt, device=device)
        self.prev_goal = z(f32, 4)
        self.has_prev, self.pending, self.driving, self.timed_out = (z(torch.bool) for _ in range(4))
        self.hold = z(f32)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.prev_goal[ids] = 0.0
        self.hold[ids] = 0.0
        for t in (self.has_prev, self.pending, self.driving, self.timed_out):
            t[ids] = False

    def step(self, goal: torch.Tensor, speed: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        goal, actual = goal.float(), actual.float()
        # drive_gate_update_goal
        max_delta = (goal - self.prev_goal).abs().amax(dim=1)
        driving = (speed.abs() > self.SPEED_EPS).any(dim=1)
        start = driving & (~self.has_prev | (max_delta > self.trigger))
        self.hold = torch.where(start & ~self.pending, torch.zeros_like(self.hold), self.hold)
        pending = self.pending | start
        timed_out = self.timed_out & ~start
        self.prev_goal = goal.clone()
        self.has_prev = torch.ones_like(self.has_prev)
        # drive_gate_allow (actual_ok = 1)
        max_err = (goal - actual).abs().amax(dim=1)
        meas = ~pending & ~timed_out & driving & (max_err > self.trigger)
        self.hold = torch.where(meas, torch.zeros_like(self.hold), self.hold)
        pending = pending | meas
        release = pending & (max_err <= self.release)
        hold = pending & ~release
        h = torch.where(hold, self.hold + 1, torch.where(release, torch.zeros_like(self.hold), self.hold))
        expire = hold & (self.max_hold > 0) & (h >= self.max_hold)
        self.hold = torch.where(expire, torch.zeros_like(h), h)
        self.pending = pending & ~release & ~expire
        self.timed_out = timed_out | expire
        self.driving = driving
        return ~self.pending


def drive_torque(duty: torch.Tensor, w: torch.Tensor, volt_scale=1.0, damping=None) -> torch.Tensor:
    """Open-loop DC motor (MDD3A duty): tau = tau_stall*V*duty/799 - D*w, D = tau_stall/w_noload (V-independent).

    Same torque-speed line as the Isaac DCMotor drive in velocity mode, but driven by duty, so the
    firmware PI (WheelPI) can sit in front of it.  Friction is not included (see MOTOR_MODELING_PLAN §8)."""
    d = DRIVE_EFFORT_NM / WHEEL_MAX_RAD_S if damping is None else damping
    return DRIVE_EFFORT_NM * volt_scale * duty / PWM_MAX - d * w


def wheel_friction(tau_drive: torch.Tensor, w: torch.Tensor, inertia, dt: float, scale=1.0,
                   w_stick: float = 0.05) -> torch.Tensor:
    """Friction torque [N m] to add to the motor torque, Karnopp stick-slip for an explicit loop.

    Slipping (|w| >= w_stick): -sign(w) * (kinetic + Stribeck(|w|)), in duty units * tau_stall/799.
    Near standstill: the torque that stops the wheel within this step (-tau_drive - J*w/dt) if it is
    within stiction, else the kinetic level against the motor.  So a wheel stays put until the motor
    beats ~47 duty, as on the bench (the firmware PI's kick exists for this).  ``scale`` (N,4) or
    scalar = per-wheel DR; ``inertia`` = joint armature [kg m^2]."""
    c = FRICTION_DUTY
    u = DRIVE_EFFORT_NM / PWM_MAX * scale
    f_kin = (c["kinetic"] + c["stribeck"] * torch.exp(-w.abs() / c["w_stribeck"])) * u
    hold = -tau_drive - inertia * w / dt
    moving = w.abs() >= w_stick
    stick = ~moving & (hold.abs() <= c["static"] * u)
    s = torch.where(moving, torch.sign(w), torch.sign(tau_drive))
    return torch.where(stick, hold, -s * f_kin)


def encoder_ticks(wheel_pos: torch.Tensor) -> torch.Tensor:
    """Cumulative encoder ticks of a wheel angle [rad] (floor, like a quadrature counter)."""
    return torch.floor(wheel_pos * (ENCODER_TICKS_PER_REV / (2.0 * math.pi)))


def encoder_speed(ticks_now: torch.Tensor, ticks_prev: torch.Tensor, dt: float) -> torch.Tensor:
    """Wheel speed [rad/s] from a tick difference over dt (main.c: 20 ms window, 1 tick = 0.0375 rad/s)."""
    return (ticks_now - ticks_prev) * (2.0 * math.pi / ENCODER_TICKS_PER_REV) / dt
