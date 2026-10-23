"""Stage-2 domain randomization: drive motor, steering servo, latency, IMU bias.

Every event writes what it applied into ``env._priv_dr`` (N, 7), which the teacher
reads as a privileged observation (``dr_params``). The layout is fixed and holds
nominal values while stage 2 is off, so a stage-1 teacher resumes into stage 2
without an observation-size change:

    [0] battery voltage scale     (stall torque and no-load speed both scale with V)
    [1] drive damping scale       (open-loop duty ~ voltage: torque/speed slope)
    [2] drive armature / 1e-2     (kg m^2; reflected rotor inertia, measured 9.8e-3; was /1e-3 before 10-13)
    [3] steering servo profile scale (acceleration and top speed together)
    [4] steering servo stiffness scale
    [5] action delay   [policy steps]
    [6] observation delay [policy steps]

Hardware basis (firmware README / Obsidian, 2026-10-11): Nucleo runs a fixed 50 Hz
loop; the SPI telemetry Pi receives was built in the *previous* cycle (obs >= 1 step
old) and a command takes effect from the next exchange, with Pi and Nucleo clocks
unsynchronized (action 0-1 step). Drive SPG30E-GR131: stall 1.96 N m, no-load
7.96 rad/s, driven open-loop by duty.  Rover measurements 2026-10-13 (pi/common/sysid_*):
drive tau 40 ms (armature 9.8e-3), spread +-2.4 %; STS3215 12 V profile 7.7 rad/s^2,
dead time 33-34.5 ms, spread < 1 % lifted (on the ground not yet measured), deadband
~0.25 deg, linkage play ~+-0.6 deg.
"""

from __future__ import annotations

import torch

from isaaclab.managers import ManagerTermBase

from ksrc_rover.utils.contract import DRIVE_ARMATURE

DR_DIM = 7
NOMINAL_ARMATURE = DRIVE_ARMATURE  # rover_cfg drive armature
ARMATURE_UNIT = 1e-2               # privileged [2] = armature / ARMATURE_UNIT (~1)
MAX_DELAY_STEPS = 3                # ring-buffer depth for action and observation delay


def _priv(env):
    if getattr(env, "_priv_dr", None) is None:
        v = torch.zeros(env.num_envs, DR_DIM, device=env.device)
        v[:, [0, 1, 3, 4]] = 1.0
        v[:, 2] = NOMINAL_ARMATURE / ARMATURE_UNIT
        env._priv_dr = v
    return env._priv_dr


def _ids(env, env_ids):
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device)
    return env_ids.to(env.device)


def _u(lo_hi, n, device):
    lo, hi = lo_hi
    return torch.empty(n, device=device).uniform_(lo, hi)


def randomize_drive_motor(env, env_ids, voltage_scale=(0.85, 1.05), motor_spread=0.05,
                          damping_scale=(0.8, 1.2), armature_range=(8.3e-3, 11.3e-3), friction_scale=(1.0, 1.0)):
    """Reset: battery voltage, per-motor spread, open-loop slope and armature for the 4 wheels.

    Voltage scales the DC motor's saturation (stall) torque, effort limit and no-load
    speed together; each motor gets an extra +-motor_spread. The DCMotor actuator keeps
    these as scalars/tensors, so they are promoted to (N, 4) tensors on first use.
    friction_scale: per-wheel gearbox friction scale, used only by the "pi" drive (stiction 35-52 duty
    on the bench depending on where the wheel stopped -> 47 x 0.75-1.1).
    """
    robot = env.scene["robot"]
    act = robot.actuators["drive"]
    ids = _ids(env, env_ids)
    n = len(ids)
    if not torch.is_tensor(act._saturation_effort):
        shape = act.effort_limit.shape
        act._saturation_effort = torch.full(shape, float(act._saturation_effort), device=env.device)
        act._nominal = {"sat": act._saturation_effort.clone(), "eff": act.effort_limit.clone(),
                        "vel": act.velocity_limit.clone(), "damp": act.damping.clone()}
    nom = act._nominal
    volt = _u(voltage_scale, n, env.device)
    per_motor = volt.unsqueeze(1) * _u((1 - motor_spread, 1 + motor_spread), n * 4, env.device).view(n, 4)
    damp = _u(damping_scale, n, env.device).unsqueeze(1)
    act._saturation_effort[ids] = nom["sat"][ids] * per_motor
    act.effort_limit[ids] = nom["eff"][ids] * per_motor
    act.velocity_limit[ids] = nom["vel"][ids] * per_motor
    act.damping[ids] = nom["damp"][ids] * damp
    act._vel_at_effort_lim = act.velocity_limit * (1 + act.effort_limit / act._saturation_effort)
    arm = _u(armature_range, n, env.device)
    wheel_ids = env.action_manager.get_term("rover").wheel_ids
    robot.write_joint_armature_to_sim(arm.unsqueeze(1).expand(n, 4), joint_ids=wheel_ids, env_ids=ids)
    env.action_manager.get_term("rover").friction_scale[ids] = _u(friction_scale, n * 4, env.device).view(n, 4)
    p = _priv(env)
    p[ids, 0] = volt
    p[ids, 1] = damp.squeeze(1)
    p[ids, 2] = arm / ARMATURE_UNIT


def randomize_steer_servo(env, env_ids, rate_scale=(0.85, 1.05), stiffness_scale=(0.8, 1.2),
                          offset_deg=0.6, deadband_deg=(0.15, 0.35), delay_s=(0.030, 0.038)):
    """Reset: STS3215 profile scale (accel + top speed; load/voltage), position-loop stiffness,
    per-wheel angle offset (linkage play / zero error), deadband and dead time."""
    import math
    robot = env.scene["robot"]
    term = env.action_manager.get_term("rover")
    ids = _ids(env, env_ids)
    n = len(ids)
    rate = _u(rate_scale, n, env.device)
    stiff = _u(stiffness_scale, n, env.device)
    term.servo.rate_scale[ids] = rate.unsqueeze(1)
    off = math.radians(offset_deg)
    term.servo.offset[ids] = torch.empty(n, 4, device=env.device).uniform_(-off, off)
    db = _u(tuple(math.radians(x) for x in deadband_deg), n, env.device)
    term.servo.deadband[ids] = db.unsqueeze(1)
    term.servo.delay_ticks[ids] = torch.round(_u(delay_s, n, env.device) / term.servo.dt).long()
    act = robot.actuators["steering"]
    if not hasattr(act, "_nominal_stiffness"):
        act._nominal_stiffness = act.stiffness.clone()
    k = act._nominal_stiffness[ids] * stiff.unsqueeze(1)
    act.stiffness[ids] = k
    robot.write_joint_stiffness_to_sim(k, joint_ids=term.steer_ids, env_ids=ids)
    p = _priv(env)
    p[ids, 3] = rate
    p[ids, 4] = stiff


def randomize_latency(env, env_ids, action_delay=(0, 1), obs_delay=(1, 2)):
    """Reset: integer policy-step delays, inclusive ranges, constant within an episode."""
    ids = _ids(env, env_ids)
    n = len(ids)
    for attr, (lo, hi) in (("_act_delay", action_delay), ("_obs_delay", obs_delay)):
        if getattr(env, attr, None) is None:
            setattr(env, attr, torch.zeros(env.num_envs, dtype=torch.long, device=env.device))
        getattr(env, attr)[ids] = torch.randint(lo, hi + 1, (n,), device=env.device)
    p = _priv(env)
    p[ids, 5] = env._act_delay[ids].float()
    p[ids, 6] = env._obs_delay[ids].float()


def randomize_imu_bias(env, env_ids, gyro_bias=0.01):
    """Reset: constant gyro bias per episode [rad/s], read by DelayedObs(source='gyro')."""
    ids = _ids(env, env_ids)
    if getattr(env, "_gyro_bias", None) is None:
        env._gyro_bias = torch.zeros(env.num_envs, 3, device=env.device)
    env._gyro_bias[ids] = torch.empty(len(ids), 3, device=env.device).uniform_(-gyro_bias, gyro_bias)


def randomize_gravity_error(env, env_ids, tilt_deg=3.0, drift_deg_s=0.3, max_deg=5.0, exact=False):
    """Reset: error of the gravity-direction estimate (2026-10-13).

    The real gravity comes from a filter on a board IMU (mounting angle, filter lag, vibration);
    the sim estimate is cleaner than reality is likely to be.  Per episode: a fixed tilt error
    about a random horizontal body axis, |e0| uniform in [0, tilt_deg] (exactly tilt_deg if exact),
    plus a slow drift at uniform [0, drift_deg_s] in a random horizontal direction; |e| <= max_deg.
    Applied by DelayedObs(grav_err=True) through gravity_error()."""
    import math
    ids = _ids(env, env_ids)
    n = len(ids)
    for attr in ("_grav_err0", "_grav_drift"):
        if getattr(env, attr, None) is None:
            setattr(env, attr, torch.zeros(env.num_envs, 3, device=env.device))
    ang = torch.rand(n, device=env.device) * 2 * math.pi
    mag = torch.full((n,), math.radians(tilt_deg), device=env.device) if exact else \
        torch.rand(n, device=env.device) * math.radians(tilt_deg)
    env._grav_err0[ids] = torch.stack([mag * torch.cos(ang), mag * torch.sin(ang), torch.zeros_like(mag)], 1)
    ang = torch.rand(n, device=env.device) * 2 * math.pi
    rate = torch.rand(n, device=env.device) * math.radians(drift_deg_s)
    env._grav_drift[ids] = torch.stack([rate * torch.cos(ang), rate * torch.sin(ang), torch.zeros_like(rate)], 1)
    env._grav_err_max = math.radians(max_deg)


def gravity_error(env, g):
    """Rotate the (N,3) gravity direction g by this episode's error vector (Rodrigues)."""
    e0 = getattr(env, "_grav_err0", None)
    if e0 is None:
        return g
    t = (env.episode_length_buf.float() * env.step_dt).unsqueeze(1)
    e = e0 + env._grav_drift * t
    th = torch.linalg.norm(e, dim=1, keepdim=True)
    e = e * (th.clamp(max=env._grav_err_max) / th.clamp(min=1e-9))
    th = th.clamp(max=env._grav_err_max)
    k = e / th.clamp(min=1e-9)
    c, s_ = torch.cos(th), torch.sin(th)
    return g * c + torch.linalg.cross(k, g, dim=1) * s_ + k * (k * g).sum(1, keepdim=True) * (1 - c)


def dr_params(env):
    """Privileged: the stage-2 DR values actually applied (layout in module docstring)."""
    return _priv(env)


# --- Delayed proprioception ----------------------------------------------------------
def _sources():
    import isaaclab_tasks.manager_based.locomotion.velocity.mdp as lab_mdp
    from . import observations as obs

    return {
        "gyro": lambda env, cfg: lab_mdp.base_ang_vel(env),
        "gravity": lambda env, cfg: lab_mdp.projected_gravity(env),
        "steer_pos": obs.steering_position,
        "steer_vel": obs.steering_velocity,
        "wheel_vel": obs.wheel_velocity,
        # Student: as the real rover computes them (utils/sensors_torch.py, Pi twin pi/common/sensor_fusion.py)
        "steer_pos_servo": _servo_angle_src,
        "steer_vel_servo": _servo_speed_src,
        "wheel_vel_enc": _EncoderSrc,
        "gravity_imu": _GravityImuSrc,
    }


# --- Real-rover sensor sources (student group) ----------------------------------------
# Stateless ones are plain functions; stateful ones are classes (one instance per obs term)
# called as src(env, asset_cfg, advance): advance=True once per env step, False on the repeat
# call after resets (then only the envs that were just reset get re-initialized).

def _servo_angle_src(env, cfg):
    """Servo position register: the servo shaft (wheel angle minus linkage play/zero error), 4096 grid."""
    from ksrc_rover.utils.sensors_torch import servo_angle
    term = env.action_manager.get_term("rover")
    q = env.scene[cfg.name].data.joint_pos[:, cfg.joint_ids]
    return servo_angle(q - term.servo.offset)


def _servo_speed_src(env, cfg):
    """Servo speed register: step/s (4096 steps = 1 rev)."""
    from ksrc_rover.utils.sensors_torch import servo_angle
    return servo_angle(env.scene[cfg.name].data.joint_vel[:, cfg.joint_ids])


class _EncoderSrc:
    """Encoder tick difference over one env step (firmware main.c, 20 ms)."""

    def __init__(self, env, params):
        from ksrc_rover.utils.sensors_torch import EncoderSpeed
        self.enc = EncoderSpeed(env.num_envs, env.step_dt, env.device)

    def reset(self, env_ids):
        self.enc.reset(env_ids)

    def __call__(self, env, cfg, advance):
        pos = env.scene[cfg.name].data.joint_pos[:, cfg.joint_ids]
        if advance:
            return self.enc.step(pos).clone()
        self.enc.init_where(self.enc.need_init, pos)
        return self.enc.out.clone()


class _GravityImuSrc:
    """Gravity direction from IMU readings averaged over the control period, through the same
    complementary filter as the Pi.

    Why averaged: a single 50 Hz sample of the gyro aliases the chassis vibration on rough
    terrain (PhysX: integrating point samples drifts 24 deg on average, 2026-10-13
    scripts/student_obs_check.py).  The period mean is exact from the pose change:
      gyro = rotation from the previous to the current root orientation / dt   (body frame)
      acc  = R^T ((v - v_prev)/dt + (0, 0, 9.81))                              (specific force)
    The firmware has to deliver the same: average the ISM330 FIFO (208 Hz) over each 20 ms
    cycle instead of reading one sample (MOTOR_MODELING_PLAN / firmware TODO).
    Filter inputs get the gyro bias (DR) and Gaussian noise (params noise_gyro [rad/s],
    noise_acc [m/s^2]); the obs term itself has no extra noise."""

    def __init__(self, env, params):
        from ksrc_rover.utils.sensors_torch import GravityFilter
        self.filt = GravityFilter(env.num_envs, env.step_dt, env.device)
        self.sg, self.sa = float(params.get("noise_gyro", 0.0)), float(params.get("noise_acc", 0.0))
        self.dt = float(env.step_dt)
        self.q_prev = None
        self.v_prev = None
        self.fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids):
        self.filt.reset(env_ids)
        self.fresh[slice(None) if env_ids is None else env_ids] = True

    def _imu(self, env):
        import isaaclab.utils.math as mu
        d = env.scene["robot"].data
        q, v = d.root_quat_w, d.root_lin_vel_w
        if self.q_prev is None:
            self.q_prev, self.v_prev = q.clone(), v.clone()
        f = self.fresh.unsqueeze(-1)            # just reset: no previous pose of this episode
        q_prev = torch.where(f, q, self.q_prev)
        v_prev = torch.where(f, v, self.v_prev)
        gyro = mu.axis_angle_from_quat(mu.quat_mul(mu.quat_inv(q_prev), q)) / self.dt
        g_up = torch.tensor([0.0, 0.0, 9.81], device=v.device)
        acc = mu.quat_apply_inverse(q, (v - v_prev) / self.dt + g_up)
        self.q_prev, self.v_prev = q.clone(), v.clone()
        self.fresh[:] = False
        return gyro, acc

    def __call__(self, env, cfg, advance):
        if advance:
            gyro, acc = self._imu(env)
            if getattr(env, "_gyro_bias", None) is not None:
                gyro = gyro + env._gyro_bias
            if self.sg > 0:
                gyro = gyro + self.sg * torch.randn_like(gyro)
            if self.sa > 0:
                acc = acc + self.sa * torch.randn_like(acc)
            return self.filt.step(gyro, acc).clone()
        if self.filt.need_init.any():           # repeat call after a reset: start from rest (|f| = g)
            import isaaclab.utils.math as mu
            q = env.scene["robot"].data.root_quat_w
            acc = mu.quat_apply_inverse(q, torch.tensor([0.0, 0.0, 9.81], device=q.device).expand(q.shape[0], 3))
            m = self.filt.need_init & self.fresh
            if m.any():
                self.filt.g = torch.where(m.unsqueeze(-1), -acc / 9.81, self.filt.g)
                self.filt.need_init &= ~m
        return self.filt.g.clone()


class DelayedObs(ManagerTermBase):
    """A proprioceptive signal as the Pi would see it: env._obs_delay steps old.

    Keeps a ring buffer of the last MAX_DELAY_STEPS+1 values. Right after a reset the
    buffer holds only the current episode's samples, so the delay is capped by the
    number available (no leakage from the previous episode). The observation manager
    may compute a group more than once per env step (ManagerBasedRLEnv.step calls
    compute() before resets and compute(update_history=True) after), so the buffer
    advances only once per env.common_step_counter; a repeat call in the same step
    just refreshes the newest slot. Adds env._gyro_bias for
    source='gyro'. Without the latency event it is the plain signal. Gaussian noise
    is added by the ObsTerm's own ``noise`` cfg, after this.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        src = _sources()[cfg.params["source"]]
        self._stateful = isinstance(src, type)
        self._fn = src(env, cfg.params) if self._stateful else src
        self._buf = None
        self._last_step = -1
        self._count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._count[env_ids] = 0
        if self._stateful:
            self._fn.reset(env_ids)

    def __call__(self, env, source: str, asset_cfg=None, noise_gyro: float = 0.0, noise_acc: float = 0.0,
                 grav_err: bool = False):
        # noise_* are read by the gravity_imu source at construction (cfg.params); listed here so
        # Isaac Lab's term-parameter check accepts them.  grav_err: apply randomize_gravity_error.
        step = int(env.common_step_counter)
        advance = step != self._last_step    # first call this env step
        self._last_step = step
        x = self._fn(env, asset_cfg, advance) if self._stateful else self._fn(env, asset_cfg)
        if grav_err:
            x = gravity_error(env, x)
        if source == "gyro" and getattr(env, "_gyro_bias", None) is not None:
            x = x + env._gyro_bias
        delay = getattr(env, "_obs_delay", None)
        if delay is None:
            return x
        if self._buf is None:
            self._buf = x.unsqueeze(0).repeat(MAX_DELAY_STEPS + 1, 1, 1)
        if advance:
            self._buf = torch.roll(self._buf, 1, dims=0)
            self._count += 1
        self._buf[0] = x
        d = torch.minimum(delay, self._count - 1).clamp(min=0, max=MAX_DELAY_STEPS)
        return self._buf[d, torch.arange(env.num_envs, device=env.device)]
