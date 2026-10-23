"""Sim2sim: run an Isaac-trained rsl_rl policy (or pure IK) in MuJoCo.

Isaac (PhysX, rigid heightfield, Coulomb μ) and MuJoCo differ in contact solver,
actuator integration and -- with terramech -- the soil model itself (slip-sinkage,
traction peak). A policy that only exploited PhysX quirks falls apart here.

The policy is rebuilt from the run itself, nothing about it is hard-coded:
  <run>/model_XXXX.pt     weights -> network kind
      actor.*                      PPO actor (single-stage or teacher), MLP
      student.* (+ memory_s.rnn.*) distilled student, MLP (or GRU/LSTM + MLP)
  <run>/params/agent.yaml  obs_groups["policy"] = which obs groups the network reads
  <run>/params/env.yaml    per group: term order, history length, noise; action class;
                           command resampling; episode length; wheel-μ event
So a run with a different observation set (teacher 29x2 + privileged 22, student
29xH, GRU 29x1, ...) works as long as every term name maps to a MuJoCo signal in
TERM_SOURCES below. Unknown terms stop with an error naming them.

Mirrored from the Isaac task: 50 Hz policy, physics dt 0.005 (4 substeps), command
distribution, history layout (Isaac Lab CircularBuffer: per term, oldest first,
filled with the first sample after reset), Direct8/Res8/Residual3 processing with the
firmware-style IK, the STS3215 servo model (dead time 34 ms, deadband, 7.7 rad/s^2
trapezoid -- the same SteerServo object as mdp/actions.py, utils/actuators_torch.py),
drive DCMotor clip per substep with D = 0.246 and armature 9.8e-3 (MuJoCo wheel joint
damping removed so droop is not counted twice), steering position actuator kp 12 /
kv 0.3 / 2.94 N m. Optional stage-2 style action/obs delay and the run's Gaussian obs
noise (--obs-delay/--act-delay/--noise).
Hardware-side options (not in the Isaac task yet, for checking what they cost a policy):
  --drive pi       firmware wheel PI (WheelPI = firmware/common/wheel_pi.c) every 20 ms on
                   encoder-quantized speed, driving the duty -> torque motor line
  --wheel-friction-duty F   Coulomb friction on the wheel joints (MuJoCo frictionloss),
                   in duty units (measured: kinetic ~5, start 35-52)
  --sensors real   wheel speed from encoder ticks over 20 ms (1 tick = 0.0375 rad/s),
                   steering angle on the servo's 4096-step grid
NOT mirrored: rocker uses MuJoCo's equality differential; no 5.45 rad/s steering joint
velocity cap; privileged mass/CoM and DR terms report nominal values (MuJoCo model is
not randomized).

Output: npz in the eval_log.py layout (T steps x N episodes) -> eval_analyze.py /
compare_all.py read it unchanged. CLI: sim/isaac/ksrc_rover/CLI.md "Sim2sim (MuJoCo)".
"""

import argparse
import pathlib
import sys
import time

import mujoco
import numpy as np
import torch
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
ISAAC = ROOT / "sim/isaac/ksrc_rover"
sys.path.insert(0, str(ISAAC / "source"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import terrain as terr  # noqa: E402
from config import RoverCfg  # noqa: E402
from mjcf import build  # noqa: E402
from terramech import PRESETS, Terramechanics, sample_terrain  # noqa: E402
from ksrc_rover.utils.actuators_torch import (  # noqa: E402
    PWM_MAX, SteerServo, WheelPI, drive_torque, encoder_speed, encoder_ticks)
from ksrc_rover.utils.sensors_torch import GravityFilter, servo_angle  # noqa: E402
from ksrc_rover.utils.contract import (  # noqa: E402
    CONTROL_DT_S, DRIVE_ARMATURE, DRIVE_EFFORT_NM, MODULE_XY_M, RES8_STEER_SCALE_RAD, RES8_WHEEL_SCALE,
    RESIDUAL3_SCALE, RESIDUAL_RATE_LIMIT, STEER_EFFORT_NM, STEER_LIMIT_RAD, WHEEL_MAX_RAD_S,
    WHEEL_RADIUS_M)
from ksrc_rover.utils.ik_torch import BatchedSwerveIK  # noqa: E402

CORNERS = ("fl", "fr", "rl", "rr")          # Isaac module order
CMD_RANGE = ((-0.25, 0.25), (-0.20, 0.20), (-1.0, 1.0))
STANDING_PROB = 0.05
DRIVE_KD = DRIVE_EFFORT_NM / WHEEL_MAX_RAD_S     # rover_cfg drive damping 0.246 (0.20 until 10-12)
NORM_EPS = 1e-2                             # rsl_rl EmpiricalNormalization eps
ACTION_CLASSES = {"Direct8Action": "Direct8", "Res8Action": "Res8", "Residual3Action": "Residual3"}


# ---------------------------------------------------------------- run config
def load_run_cfg(ckpt):
    """(env.yaml, agent.yaml) saved by Isaac Lab next to the checkpoint, or (None, None)."""
    p = pathlib.Path(ckpt).parent / "params"
    if not (p / "env.yaml").is_file():
        return None, None
    load = lambda f: yaml.load(open(p / f), Loader=yaml.UnsafeLoader)  # !!python/tuple tags
    return load("env.yaml"), load("agent.yaml")


def group_terms(env_cfg, group):
    """[(term name, cfg dict)] in Isaac order, plus the group's history length."""
    g = env_cfg["observations"][group]
    terms = [(k, v) for k, v in g.items() if isinstance(v, dict) and "func" in v]
    hist = g.get("history_length") or 0
    return terms, hist, bool(g.get("enable_corruption"))


# ---------------------------------------------------------------- policy
def _mlp(sd, prefix):
    idx = sorted({int(k[len(prefix):].split(".")[0]) for k in sd
                  if k.startswith(prefix) and k.endswith(".weight")})
    layers = []
    for n, i in enumerate(idx):
        w, b = sd[f"{prefix}{i}.weight"], sd[f"{prefix}{i}.bias"]
        lin = torch.nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(b)
        layers.append(lin)
        if n < len(idx) - 1:
            layers.append(torch.nn.ELU())
    return torch.nn.Sequential(*layers).eval()


class Policy:
    """PPO actor or distilled student (MLP / GRU / LSTM) + its obs normalizer."""

    def __init__(self, path, agent_cfg):
        sd = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
        if any(k.startswith("actor.") for k in sd):
            self.kind, head, norm = "ppo", "actor.", "actor_obs_normalizer."
        elif any(k.startswith("student.") for k in sd):
            self.kind, head, norm = "student", "student.", "student_obs_normalizer."
        else:
            raise SystemExit(f"{path}: actor.* 도 student.* 도 없는 체크포인트")
        self.rnn = None
        if any(k.startswith("memory_s.rnn.") for k in sd):
            rnn_type = (agent_cfg or {}).get("policy", {}).get("rnn_type", "gru").lower()
            w = sd["memory_s.rnn.weight_ih_l0"]
            layers = len({k.split("_l")[-1] for k in sd if k.startswith("memory_s.rnn.weight_ih_l")})
            gates = 3 if rnn_type == "gru" else 4
            cls = torch.nn.GRU if rnn_type == "gru" else torch.nn.LSTM
            self.rnn = cls(w.shape[1], w.shape[0] // gates, layers).eval()
            self.rnn.load_state_dict({k[len("memory_s.rnn."):]: v for k, v in sd.items()
                                      if k.startswith("memory_s.rnn.")})
            self.kind += f"+{rnn_type}"
        self.net = _mlp(sd, head)
        self.obs_dim = self.rnn.input_size if self.rnn else self.net[0].in_features
        self.act_dim = self.net[-1].out_features
        if f"{norm}_mean" in sd:
            self.mean = sd[f"{norm}_mean"].squeeze(0)
            self.std = sd[f"{norm}_std"].squeeze(0) + NORM_EPS
        else:
            self.mean, self.std = torch.zeros(self.obs_dim), torch.ones(self.obs_dim)
        self.reset()

    def reset(self):
        self.h = None

    @torch.no_grad()
    def __call__(self, obs):
        x = (torch.as_tensor(obs, dtype=torch.float32) - self.mean) / self.std
        if self.rnn is not None:
            y, self.h = self.rnn(x.view(1, 1, -1), self.h)
            x = y.view(-1)
        return self.net(x).numpy()


# ---------------------------------------------------------------- actions (mdp/actions.py)
class ActionMap:
    def __init__(self, mode, act_delay=0):
        self.mode = mode
        self.dim = 3 if mode == "Residual3" else 8
        self.ik = BatchedSwerveIK(1, "cpu")
        self.act_delay = act_delay
        self.reset()

    def reset(self):
        self.ik.reset(None)
        self.raw = np.zeros(self.dim)
        self.processed = np.zeros(8)      # targets after the command delay (servo goal + wheel speed)
        self.target = np.zeros(8)         # before delay and rate limit
        self.hist = [np.zeros(8) for _ in range(self.act_delay + 1)]

    def _ik(self, cmd):
        steer, mps = self.ik.compute(torch.as_tensor(cmd, dtype=torch.float32)[None])
        return steer[0].numpy().astype(float), mps[0].numpy().astype(float)

    def process(self, action, cmd):
        a = np.clip(action, -1.0, 1.0)
        if self.mode == "Direct8":
            self.raw = a
            p = np.concatenate((a[:4] * STEER_LIMIT_RAD, a[4:] * WHEEL_MAX_RAD_S))
        elif self.mode == "Res8":
            self.raw = a
            steer, mps = self._ik(cmd)
            s = np.clip(steer + a[:4] * RES8_STEER_SCALE_RAD, -STEER_LIMIT_RAD, STEER_LIMIT_RAD)
            w = np.clip(mps / WHEEL_RADIUS_M * (1.0 + a[4:] * RES8_WHEEL_SCALE), -WHEEL_MAX_RAD_S, WHEEL_MAX_RAD_S)
            p = np.concatenate((s, w))
        else:
            lim = np.clip(a - self.raw, -RESIDUAL_RATE_LIMIT, RESIDUAL_RATE_LIMIT) + self.raw
            self.raw = lim
            steer, mps = self._ik(cmd + lim * np.asarray(RESIDUAL3_SCALE))
            p = np.concatenate((steer, mps / WHEEL_RADIUS_M))
        self.target = p.copy()
        self.hist = self.hist[1:] + [p.copy()]          # command latency (dr.py act delay)
        p = self.hist[0].copy()           # servo dead time / profile run per substep in MjRover.step
        self.processed = p
        return p


# ---------------------------------------------------------------- observations
def wheel_slip(s):
    """Same as mdp.rewards.wheel_slip_ratio (signed, module order)."""
    xy = np.asarray(MODULE_XY_M)
    vx, vy, wz = s["lin_b"][0], s["lin_b"][1], s["ang_b"][2]
    v_long = (vx - wz * xy[:, 1]) * np.cos(s["steer"]) + (vy + wz * xy[:, 0]) * np.sin(s["steer"])
    surf = s["wheel_w"] * WHEEL_RADIUS_M
    return (surf - v_long) / np.maximum(np.maximum(np.abs(surf), np.abs(v_long)), 0.05)


# term -> MuJoCo signal. Keyed by the Isaac func name, or by DelayedObs' params.source.
# "sensor" terms get the obs delay and noise; the others are exact (Pi-known or privileged).
TERM_SOURCES = {
    "generated_commands": ("exact", lambda c: c["cmd"]),
    "base_ang_vel": ("sensor", lambda c: c["s"]["ang_b"]), "gyro": ("sensor", lambda c: c["s"]["ang_b"]),
    "projected_gravity": ("sensor", lambda c: c["s"]["grav_b"]), "gravity": ("sensor", lambda c: c["s"]["grav_b"]),
    "steering_position": ("sensor", lambda c: c["s"]["steer_obs"]), "steer_pos": ("sensor", lambda c: c["s"]["steer_obs"]),
    "steering_velocity": ("sensor", lambda c: c["s"]["steer_v"]), "steer_vel": ("sensor", lambda c: c["s"]["steer_v"]),
    "wheel_velocity": ("sensor", lambda c: c["s"]["wheel_w_obs"]), "wheel_vel": ("sensor", lambda c: c["s"]["wheel_w_obs"]),
    "applied_targets": ("exact", lambda c: c["amap"].processed),
    # student, computed as on the real rover (mdp/dr.py *_src, utils/sensors_torch.py)
    "steer_pos_servo": ("sensor", lambda c: c["s"]["steer_servo"]),
    "steer_vel_servo": ("sensor", lambda c: c["s"]["steer_v_servo"]),
    "wheel_vel_enc": ("sensor", lambda c: c["s"]["wheel_w_enc"]),
    "gravity_imu": ("sensor", lambda c: c["s"]["grav_imu"]),
    # privileged (teacher)
    "base_lin_vel": ("exact", lambda c: c["s"]["lin_b"]),
    "wheel_mu": ("exact", lambda c: c["mu_eff"]),
    "wheel_slip": ("exact", lambda c: wheel_slip(c["s"])),
    "base_mass_com": ("exact", lambda c: np.array([1.0, 0.0, 0.0, 0.0])),
    # mdp/dr.py layout; [2] = armature / 1e-2 (ARMATURE_UNIT since 10-13)
    "dr_params": ("exact", lambda c: np.array([1.0, 1.0, DRIVE_ARMATURE / 1e-2, 1.0, 1.0, c["act_delay"], c["obs_delay"]])),
}


class ObsBuilder:
    """Builds the network input from the run's obs groups, in Isaac's exact layout."""

    def __init__(self, env_cfg, groups, obs_delay=0, noise=False, rng=None):
        self.items, self.rng, self.obs_delay = [], rng, obs_delay
        for gname in groups:
            terms, hist, corrupt = group_terms(env_cfg, gname)
            for name, t in terms:
                key = t.get("params", {}).get("source") or t["func"].split(":")[-1]
                if key not in TERM_SOURCES:
                    raise SystemExit(f"관측 항목 '{gname}.{name}' ({t['func']}) 의 MuJoCo 대응이 없음 "
                                     f"→ sim2sim_isaac.py TERM_SOURCES 에 추가")
                kind, fn = TERM_SOURCES[key]
                std = 0.0
                if noise and corrupt and isinstance(t.get("noise"), dict):
                    std = float(t["noise"].get("std", 0.0))
                self.items.append(dict(name=f"{gname}.{name}", fn=fn, kind=kind,
                                       hist=max(hist or t.get("history_length") or 0, 1), std=std))
        self.reset()

    def reset(self):
        for it in self.items:
            it["buf"] = None          # history (oldest first)
            it["lag"] = []            # raw sensor samples for the obs delay

    def __call__(self, ctx):
        out = []
        for it in self.items:
            x = np.asarray(it["fn"](ctx), dtype=float).copy()
            if it["kind"] == "sensor" and self.obs_delay > 0:
                it["lag"] = (it["lag"] + [x])[-(self.obs_delay + 1):]
                x = it["lag"][0].copy()          # capped by samples since reset, like DelayedObs
            if it["std"] > 0:
                x = x + self.rng.normal(0.0, it["std"], x.shape)
            if it["buf"] is None:
                it["buf"] = [x.copy() for _ in range(it["hist"])]
            else:
                it["buf"] = it["buf"][1:] + [x]
            out.append(np.concatenate(it["buf"]))
        return np.concatenate(out)

    def layout(self):
        return ", ".join(f"{it['name']}×{it['hist']}" for it in self.items)


# ---------------------------------------------------------------- wheel μ
def sample_mu(rng, mode, state, new_episode):
    """'s6' = uniform per wheel 0.1-0.6 (= 'uniform:0.1,0.6'), 'uniform:lo,hi',
    'patch' = randomize_wheel_mu_patch defaults, 'const:<μ>' = all wheels fixed."""
    if mode.startswith("const:"):
        return np.full(4, float(mode.split(":")[1]))
    if mode == "s6":
        mode = "uniform:0.1,0.6"
    if mode.startswith("uniform:"):
        lo, hi = (float(v) for v in mode.split(":")[1].split(","))
        return rng.uniform(lo, hi, 4)
    if new_episode or "base" not in state:
        state["base"], state["uniform"] = rng.uniform(0.25, 0.8), rng.random() < 0.2
    mu = np.full(4, state["base"]) if state["uniform"] else state["base"] * rng.uniform(0.75, 1.25, 4)
    if rng.random() < 0.5:
        idx = [int(rng.integers(4))] if rng.random() < 0.5 else ([0, 2] if rng.random() < 0.5 else [1, 3])
        mu[idx] = rng.uniform(0.1, 0.2, len(idx))
    return np.clip(mu, 0.08, 1.0)


def mu_from_run(env_cfg):
    """The run's own wheel-μ sampler, for --mu auto."""
    ev = (env_cfg or {}).get("events", {}).get("wheel_friction") or {}
    f, p = ev.get("func", ""), ev.get("params", {})
    if f.endswith("randomize_wheel_mu_patch"):
        return "patch"
    if f.endswith("randomize_wheel_mu") and p.get("per") == "wheel":
        lo, hi = p.get("mu_range", (0.1, 0.6))
        return f"uniform:{lo},{hi}"
    return "s6"


class WheelScaledTerramech(Terramechanics):
    """terramech with a per-wheel μ_max factor (module order). Every friction term in
    apply() is linear in μ_max, so scaling the written geom friction is exact."""

    factor = np.ones(4)          # module order fl, fr, rl, rr

    def apply(self, d, dt):
        super().apply(d, dt)
        self.m.geom_friction[self.wg, 0] *= self.factor[[0, 2, 1, 3]]   # terramech order fl, rl, fr, rr


# ---------------------------------------------------------------- MuJoCo rover
class MjRover:
    def __init__(self, scenario, mj_dt, rng, visual=False, drive="isaac", sensors="ideal", wheel_friction_duty=0.0):
        self.scenario, self.rng = scenario, rng
        self.drive, self.sensors = drive, sensors
        self.cfg = RoverCfg()
        assert abs(self.cfg.wheel_r - WHEEL_RADIUS_M) < 1e-6
        self.rough = scenario.startswith("rough")
        self.sand = "sand" in scenario
        if self.rough:
            xml = build(self.cfg, terrain="hfield", hfield_n=terr.HF_N,
                        hfield_size=(terr.ARENA_X / 2, terr.ARENA_Y / 2), visual=visual)
        else:
            xml = build(self.cfg, terrain="plane", visual=visual)
        m = self.m = mujoco.MjModel.from_xml_string(xml)
        m.opt.timestep = mj_dt
        self.d = mujoco.MjData(m)
        self.n_sub = int(round(CONTROL_DT_S / mj_dt))
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        aid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        self.steer_q = np.array([m.jnt_qposadr[jid(f"st_{c}")] for c in CORNERS])
        self.steer_v = np.array([m.jnt_dofadr[jid(f"st_{c}")] for c in CORNERS])
        self.wheel_v = np.array([m.jnt_dofadr[jid(f"wh_{c}")] for c in CORNERS])
        self.wheel_q = np.array([m.jnt_qposadr[jid(f"wh_{c}")] for c in CORNERS])
        self.steer_a = np.array([aid(f"a_st_{c}") for c in CORNERS])
        self.wheel_a = np.array([aid(f"a_wh_{c}") for c in CORNERS])
        m.dof_damping[self.wheel_v] = 0.0          # Isaac DCMotor clips explicitly
        m.dof_armature[self.wheel_v] = DRIVE_ARMATURE  # rotor inertia, measured 10-13 (mjcf.py has 0.0015)
        m.dof_frictionloss[self.wheel_v] = DRIVE_EFFORT_NM * wheel_friction_duty / PWM_MAX
        m.actuator_forcerange[self.steer_a] = [-STEER_EFFORT_NM, STEER_EFFORT_NM]   # 12 V C018 (config.py 1.912)
        self.servo = SteerServo(1, mj_dt)
        self.pi = WheelPI(1)
        self.ticks_prev, self.wheel_w_enc, self.duty = np.zeros(4), np.zeros(4), np.zeros(4)
        self.q_prev = self.v_prev = None
        self.gfilt = GravityFilter(1, CONTROL_DT_S)
        self.grav_imu = np.array([0.0, 0.0, -1.0])
        lo, hi = m.actuator_ctrlrange[self.steer_a].T
        print("[sim2sim] MuJoCo steer ctrlrange deg:",
              np.round(np.degrees(np.stack([lo, hi], 1)), 1).tolist(),
              f"(Isaac ±{np.degrees(STEER_LIMIT_RAD):.0f})", flush=True)
        # wheel tyre geoms + ground lowered (MuJoCo combines μ by max)
        self.tm = WheelScaledTerramech(m, self.cfg.wheel_r, PRESETS["sand"])
        self.tm_g = self.tm.wg[[0, 2, 1, 3]]        # tyre geoms in module order
        self.mu4 = np.ones(4)

    def set_mu(self, mu4):
        self.mu4 = np.asarray(mu4, float)
        if self.sand:
            self.tm.factor = self.mu4 / self.tm.p.mu_max
        else:
            self.m.geom_friction[self.tm_g, 0] = self.mu4

    def reset(self):
        m, d, rng = self.m, self.d, self.rng
        mujoco.mj_resetData(m, d)
        ground = 0.0
        if self.rough:
            Z = terr.generate(0.5, rng)[0]
            z_off = terr.to_hfield(m, Z)
            ex, ey = terr.ARENA_X, terr.ARENA_Y
            drive, _ = terr.drivable_mask(Z, ex, ey, max_slope_deg=20.0, max_step=self.cfg.max_step)
            xs, ys = terr.cell_centers(Z, ex, ey)
            central = drive & (np.abs(xs)[None, :] < 1.5) & (np.abs(ys)[:, None] < 1.5)
            xy = terr.sample_drivable(central if central.any() else drive, Z, ex, ey, rng)[0]
            d.qpos[0:2] = xy
            ground = terr.local_top(Z, ex, ey, xy) + z_off
        d.qpos[2] += ground + 0.01
        yaw = rng.uniform(-np.pi, np.pi)
        d.qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        if self.sand:
            self.tm.p = sample_terrain(rng, 0.5)
            self.tm.reset()
        mujoco.mj_forward(m, d)
        self.servo.reset()
        self.pi.reset()
        self.duty = np.zeros(4)
        self.ticks_prev = self._ticks()
        self.wheel_w_enc = np.zeros(4)
        self.gfilt.reset()
        self.q_prev = self.v_prev = None      # start of episode: gyro 0, |f| = g -> filter starts on the true attitude
        self._step_gravity()

    def _imu(self):
        """IMU averaged over the control period, like mdp/dr.py _GravityImuSrc (body frame):
        gyro = rotation from the previous to the current orientation / dt, acc = R^T(dv/dt + g up)."""
        d = self.d
        q, v = d.qpos[3:7].copy(), d.qvel[0:3].copy()
        if self.q_prev is None:
            self.q_prev, self.v_prev = q, v
        qi = np.empty(4)
        mujoco.mju_negQuat(qi, self.q_prev)
        qr = np.empty(4)
        mujoco.mju_mulQuat(qr, qi, q)
        gyro = np.empty(3)
        mujoco.mju_quat2Vel(gyro, qr, CONTROL_DT_S)
        R = np.empty(9)
        mujoco.mju_quat2Mat(R, q)
        acc = R.reshape(3, 3).T @ ((v - self.v_prev) / CONTROL_DT_S + np.array([0.0, 0.0, 9.81]))
        self.q_prev, self.v_prev = q, v
        return gyro, acc

    def _step_gravity(self):
        gyro, acc = self._imu()
        self.grav_imu = self.gfilt.step(torch.as_tensor(gyro[None], dtype=torch.float32),
                                        torch.as_tensor(acc[None], dtype=torch.float32))[0].numpy().astype(float)

    def _ticks(self):
        return encoder_ticks(torch.as_tensor(self.d.qpos[self.wheel_q])).numpy()

    def step(self, processed):
        """One policy step (20 ms): processed = (servo goal x4, wheel speed target x4) after the command delay."""
        m, d = self.m, self.d
        goal = torch.as_tensor(processed[None, :4], dtype=torch.float32)
        w_t = processed[4:]
        if self.drive == "pi":       # firmware: once per control period, on the encoder speed of the last period
            self.duty = self.pi.step(torch.as_tensor(w_t[None], dtype=torch.float32),
                                     torch.as_tensor(self.wheel_w_enc[None], dtype=torch.float32))[0].numpy()
        for _ in range(self.n_sub):
            d.ctrl[self.steer_a] = self.servo.step(goal)[0].numpy()
            w = d.qvel[self.wheel_v]
            hi = np.minimum(DRIVE_EFFORT_NM, DRIVE_EFFORT_NM * (1.0 - w / WHEEL_MAX_RAD_S))
            lo = np.maximum(-DRIVE_EFFORT_NM, DRIVE_EFFORT_NM * (-1.0 - w / WHEEL_MAX_RAD_S))
            if self.drive == "pi":
                tau = drive_torque(torch.as_tensor(self.duty), torch.as_tensor(w)).numpy()
            else:                    # Isaac DCMotor, velocity target
                tau = DRIVE_KD * (w_t - w)
            d.ctrl[self.wheel_a] = np.clip(tau, lo, hi) / DRIVE_EFFORT_NM
            if self.sand:
                self.tm.apply(d, m.opt.timestep)
            mujoco.mj_step(m, d)
        t = self._ticks()
        self.wheel_w_enc = encoder_speed(torch.as_tensor(t), torch.as_tensor(self.ticks_prev), CONTROL_DT_S).numpy()
        self.ticks_prev = t
        self._step_gravity()

    def state(self):
        d = self.d
        R = np.empty(9)
        mujoco.mju_quat2Mat(R, d.qpos[3:7])
        R = R.reshape(3, 3)
        lin_b = R.T @ d.qvel[0:3]          # free joint: linear vel in world frame
        ang_b = d.qvel[3:6].copy()         # free joint: angular vel already in body frame
        grav_b = R.T @ np.array([0.0, 0.0, -1.0])
        steer, wheel_w = d.qpos[self.steer_q].copy(), d.qvel[self.wheel_v].copy()
        if self.sensors == "real":   # what the Nucleo reports: encoder ticks / 20 ms, servo 4096-step angle
            q = 2 * np.pi / 4096
            steer_obs, wheel_w_obs = np.round(steer / q) * q, self.wheel_w_enc.copy()
        else:
            steer_obs, wheel_w_obs = steer, wheel_w
        return dict(lin_b=lin_b, ang_b=ang_b, grav_b=grav_b, pos=d.qpos[0:3].copy(),
                    quat=d.qpos[3:7].copy(), steer=steer, steer_v=d.qvel[self.steer_v].copy(), wheel_w=wheel_w,
                    steer_obs=steer_obs, wheel_w_obs=wheel_w_obs,
                    steer_servo=servo_angle(torch.as_tensor(steer)).numpy(),
                    steer_v_servo=servo_angle(torch.as_tensor(d.qvel[self.steer_v])).numpy(),
                    wheel_w_enc=self.wheel_w_enc.copy(), grav_imu=self.grav_imu.copy())

    def mu_effective(self):
        """μ the wheels actually have: the set μ, or terramech's peak μ_max × factor."""
        return self.tm.p.mu_max * self.tm.factor if self.sand else self.mu4


# ---------------------------------------------------------------- main loop
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--policy", default="ik", help="rsl_rl model_XXXX.pt (params/ 옆에 있어야 함), 또는 'ik'")
    ap.add_argument("--action", choices=("Direct8", "Res8", "Residual3"), default=None,
                    help="기본: 런의 env.yaml 에서 (ik 면 Res8)")
    ap.add_argument("--scenario", default="flat_mu",
                    choices=("flat_mu", "flat_sand", "rough_mu", "rough_sand"))
    ap.add_argument("--mu", default="auto", help="auto(런의 μ 이벤트) | s6 | patch | uniform:lo,hi | const:μ")
    ap.add_argument("--mu-resample-s", type=float, nargs=2, default=None, help="기본: 런 설정, 없으면 1 1")
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--episode-s", type=float, default=None, help="기본: 런의 episode_length_s")
    ap.add_argument("--cmd-resample-s", type=float, nargs=2, default=None, help="기본: 런의 resampling_time_range")
    ap.add_argument("--obs-delay", type=int, default=0, help="센서 관측 지연 [정책 스텝]")
    ap.add_argument("--act-delay", type=int, default=0, help="명령 지연 [정책 스텝]")
    ap.add_argument("--noise", action="store_true", help="런에 enable_corruption 이 켜져 있으면 그 노이즈 σ 적용")
    ap.add_argument("--drive", choices=("isaac", "pi"), default="isaac",
                    help="isaac = 학습과 같은 DCMotor 속도 목표 / pi = 펌웨어 바퀴 PI (wheel_pi.c) + duty 모터")
    ap.add_argument("--wheel-friction-duty", type=float, default=0.0,
                    help="바퀴 관절 쿨롱 마찰 [duty 환산] (실측 운동 ~5, 기동 35-52). 기본 0 = 학습과 같음")
    ap.add_argument("--sensors", choices=("ideal", "real"), default="ideal",
                    help="real = 바퀴 속도를 엔코더 20 ms 차분으로, 조향각을 서보 4096 격자로")
    ap.add_argument("--mj-dt", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--out", default=None, help="npz 경로. --view 면 생략 가능(저장 안 함)")
    ap.add_argument("--view", action="store_true", help="MuJoCo 뷰어로 실시간 재생 (창 닫으면 종료)")
    ap.add_argument("--speed", type=float, default=1.0, help="--view 재생 배속 (2 = 2배 빠르게)")
    args = ap.parse_args()
    if args.out is None and not args.view:
        ap.error("--out 이 필요합니다 (--view 일 때만 생략 가능)")

    rng = np.random.default_rng(args.seed)
    env_cfg, agent_cfg = (None, None) if args.policy == "ik" else load_run_cfg(args.policy)
    if args.policy != "ik" and env_cfg is None:
        raise SystemExit(f"{pathlib.Path(args.policy).parent}/params/env.yaml 없음 — 체크포인트와 params 를 같이 받을 것")
    action = args.action or (ACTION_CLASSES[env_cfg["actions"]["rover"]["class_type"].split(":")[-1]]
                             if env_cfg else "Res8")
    episode_s = args.episode_s or (env_cfg or {}).get("episode_length_s", 8.0)
    cmd_rs = args.cmd_resample_s or tuple((env_cfg or {}).get("commands", {}).get(
        "base_velocity", {}).get("resampling_time_range", (8.0, 8.0)))
    mu_mode = mu_from_run(env_cfg) if args.mu == "auto" else args.mu
    if args.mu_resample_s:
        mu_rs = tuple(args.mu_resample_s)
    else:
        iv = ((env_cfg or {}).get("events", {}).get("wheel_friction_interval") or {}).get("interval_range_s")
        mu_rs = tuple(iv) if iv else (1.0, 1.0)

    if abs(args.mj_dt - 0.005) > 1e-9:
        print("[sim2sim] 경고: 서보 모델은 물리 스텝마다 돈다 -- Isaac(0.005)과 다르면 지연 틱 수가 바뀐다", flush=True)
    rover = MjRover(args.scenario, args.mj_dt, rng, visual=args.view, drive=args.drive, sensors=args.sensors,
                    wheel_friction_duty=args.wheel_friction_duty)
    viewer = None
    if args.view:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(rover.m, rover.d)
        viewer.cam.distance, viewer.cam.elevation = 2.0, -25
    amap = ActionMap(action, args.act_delay)
    policy, obs = None, None
    if args.policy != "ik":
        policy = Policy(args.policy, agent_cfg)
        groups = agent_cfg["obs_groups"]["policy"]
        obs = ObsBuilder(env_cfg, groups, args.obs_delay, args.noise, rng)
        probe = obs(dict(cmd=np.zeros(3), s=rover.state(), amap=amap, mu_eff=np.ones(4),
                         act_delay=args.act_delay, obs_delay=args.obs_delay))
        if probe.size != policy.obs_dim or policy.act_dim != amap.dim:
            raise SystemExit(f"크기 불일치: 관측 {probe.size} vs 정책 {policy.obs_dim}, "
                             f"행동 {amap.dim}({action}) vs 정책 {policy.act_dim}")
        print(f"[sim2sim] policy {policy.kind}, obs {policy.obs_dim} = {obs.layout()}", flush=True)
    print(f"[sim2sim] action {action}, scenario {args.scenario}, μ {mu_mode} every {mu_rs}s, "
          f"episode {episode_s}s, command every {cmd_rs}s, delay obs {args.obs_delay}/act {args.act_delay}, "
          f"drive {args.drive} (friction {args.wheel_friction_duty:g} duty), sensors {args.sensors}", flush=True)
    if action == "Direct8" and policy is None:
        print("[sim2sim] 경고: Direct8 + zero action = 정지. IK 기준은 --action Res8 --policy ik", flush=True)

    T = int(round(episode_s / CONTROL_DT_S))
    keys = ("cmd", "vel", "steer", "wheel_w", "root_pos", "root_quat", "action", "ang",
            "mu_l", "mu_r", "mu4", "done")
    buf = {k: [] for k in keys}
    new_cmd = lambda: (np.zeros(3) if rng.random() < STANDING_PROB
                       else np.array([rng.uniform(*r) for r in CMD_RANGE]))
    for ep in range(args.episodes):
        if viewer is not None and not viewer.is_running():
            break
        rover.reset()
        amap.reset()
        mu_state = {}
        rover.set_mu(sample_mu(rng, mu_mode, mu_state, True))
        t_mu = rng.uniform(*mu_rs)
        cmd, t_cmd = new_cmd(), rng.uniform(*cmd_rs)
        if policy is not None:
            policy.reset()
            obs.reset()

        def observe():
            s = rover.state()
            if obs is None:
                return s, None
            return s, obs(dict(cmd=cmd, s=s, amap=amap, mu_eff=rover.mu_effective(),
                               act_delay=args.act_delay, obs_delay=args.obs_delay))

        s, o = observe()
        if viewer is not None:
            if rover.rough:
                viewer.update_hfield(0)          # new terrain this episode -> GPU copy
            print(f"[view] ep {ep + 1}: cmd vx {cmd[0]:+.2f} vy {cmd[1]:+.2f} wz {cmd[2]:+.2f}, "
                  f"μ {np.round(rover.mu4, 2).tolist()}", flush=True)
        rows = {k: [] for k in keys}
        for k in range(T):
            t = k * CONTROL_DT_S
            if t >= t_cmd:
                cmd, t_cmd = new_cmd(), t + rng.uniform(*cmd_rs)
            if mu_rs[0] > 0 and t >= t_mu:
                rover.set_mu(sample_mu(rng, mu_mode, mu_state, False))
                t_mu = t + rng.uniform(*mu_rs)
            a = policy(o) if policy is not None else np.zeros(amap.dim)
            row = dict(cmd=cmd.copy(), vel=np.array([s["lin_b"][0], s["lin_b"][1], s["ang_b"][2]]),
                       steer=s["steer"], wheel_w=s["wheel_w"], root_pos=s["pos"], root_quat=s["quat"],
                       action=np.asarray(a, float), ang=s["ang_b"],
                       mu_l=rover.mu4[[0, 2]].mean(), mu_r=rover.mu4[[1, 3]].mean(), mu4=rover.mu4.copy())
            t_wall = time.time()
            rover.step(amap.process(a, cmd))
            s, o = observe()
            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.cam.lookat[:] = s["pos"]
                viewer.sync()
                time.sleep(max(0.0, CONTROL_DT_S / args.speed - (time.time() - t_wall)))
            tilt = np.degrees(np.arccos(np.clip(-s["grav_b"][2], -1, 1)))
            done = (k == T - 1) or tilt > 60.0 or not np.isfinite(s["pos"]).all()
            row["done"] = np.int8(done)
            for kk, v in row.items():
                rows[kk].append(np.asarray(v))
            if done:
                break
        if viewer is not None and not viewer.is_running():
            break                                   # window closed mid-episode: drop it
        n = len(rows["done"])
        for kk in keys:   # pad an early-terminated episode with its done row (masked out later)
            arr = np.stack(rows[kk])
            if n < T:
                arr = np.concatenate([arr, np.repeat(arr[-1:], T - n, axis=0)])
            buf[kk].append(arr)
        print(f"[sim2sim] ep {ep + 1}/{args.episodes} steps {n}{' (전복)' if n < T else ''}", flush=True)

    if viewer is not None:
        viewer.close()
    if args.out is None or not buf["done"]:
        return
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, task=f"mujoco:{args.scenario}:{action}", is_baseline=(policy is None),
        wheel_seed=args.seed, wheel_radius_m=WHEEL_RADIUS_M, module_xy=np.asarray(MODULE_XY_M),
        policy=str(args.policy), mu_mode=mu_mode, drive=args.drive, sensors=args.sensors,
        wheel_friction_duty=args.wheel_friction_duty,
        **{k: np.stack(v, axis=1) for k, v in buf.items()})   # (T, N_episodes, ...)
    print(f"SIM2SIM_SAVED {out} episodes={args.episodes} T={T}", flush=True)


if __name__ == "__main__":
    main()
