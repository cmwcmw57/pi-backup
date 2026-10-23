"""RL Res8 student on the rover: observation assembly + residual command (2026-10-13, structure B).

Structure B (firmware does IK):  Pi sends (vx, vy, w) + per-wheel residual (SPI type 0x02),
the Nucleo runs its own IK + fold, adds the residual, then gate -> wheel PI.  The Pi still needs
the IK result for the "applied_targets" observation, so it runs the **same C IK** as the firmware
(sim/libksrc_common.so, built from firmware/common on the Pi with sim/build.sh).

Observation = exactly what the Isaac student group computes (rover_env_cfg.py StudentCfg):
  command        (vx, vy, w) commanded this step                        [m/s, m/s, rad/s]
  gyro           IMU gyro, body frame, 20 ms mean, boot bias removed     [rad/s]
  gravity        complementary filter (sensor_fusion.GravityFilter)      [unit]
  steering       servo position register (telemetry steer_meas_rad)      [rad]
  steering_rate  servo speed register step/s * 2 pi / 4096                [rad/s]
  wheel_rate     encoder 20 ms difference (telemetry wheel_mps / r)       [rad/s]
  applied_targets  steering goal x4 [rad] + wheel speed target x4 [rad/s] the Pi just commanded
                   (= IK + residual, before the Nucleo's command delay and drive gate)
each term with STUDENT_HISTORY frames, oldest first (Isaac Lab flatten_history_dim); after a
reset the buffer is filled with the first frame (Isaac Lab CircularBuffer).  The layout comes
from the exported <policy>.json (sim/isaac/ksrc_rover/scripts/export_student.py).
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "pi" / "common"))

import ksrc_common as kc  # noqa: E402
import nucleo_link as nl  # noqa: E402
from sensor_fusion import GravityFilter, WHEEL_RADIUS_M  # noqa: E402


def _load_contract():
    """utils/contract.py of the Isaac task (plain Python, no Isaac import): one source of the constants."""
    p = ROOT / "sim/isaac/ksrc_rover/source/ksrc_rover/utils/contract.py"
    spec = importlib.util.spec_from_file_location("ksrc_contract", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load_contract()
STEER_STEP_RAD = 2.0 * math.pi / 4096
FW_MAX_WHEEL_MPS = 0.378          # main.c SWERVE_MAX_WHEEL_MPS
FW_STEER_LIMIT_RAD = math.radians(90.0)   # joint.c min/max_deg (all four)


class FirmwareIK:
    """main.c Swerve_Control up to the residual: swerve_ik_compute -> swerve_fold_to_range (same C)."""

    def __init__(self):
        self.modules = (kc.SwerveModulePos * 4)(*[kc.SwerveModulePos(x, y) for x, y in C.MODULE_XY_M])
        self.reset()

    def reset(self):
        self.state = (kc.SwerveModuleState * 4)()

    def sync(self, steer_cmd_rad):
        """Start from the angles the Nucleo is already holding (telemetry steer_cmd_rad, 2026-10-15).
        A fresh Pi state holds 0 while standing, but the firmware keeps the last run's angles -> the
        applied_targets obs would be wrong until the first move (and at the +-90 deg seam the fold could
        pick the other side)."""
        for m, a in enumerate(steer_cmd_rad):
            self.state[m].last_angle_rad = float(a)
            self.state[m].last_cmd_rad = float(a)
            self.state[m].initialized = 1

    def step(self, vx, vy, w, d_steer=None, wheel_gain=None):
        """-> (steer goal [rad] x4, wheel speed [m/s] x4) as the firmware applies them."""
        out = kc.swerve_ik_compute(vx, vy, w, self.modules, FW_MAX_WHEEL_MPS, self.state)
        lo, hi = [-FW_STEER_LIMIT_RAD] * 4, [FW_STEER_LIMIT_RAD] * 4
        kc.swerve_fold_to_range(lo, hi, self.state, out)
        goal = np.array([out[m].angle_rad for m in range(4)], np.float32)
        speed = np.array([out[m].speed_mps for m in range(4)], np.float32)
        if d_steer is not None:
            goal = np.clip(goal + np.asarray(d_steer, np.float32), -FW_STEER_LIMIT_RAD, FW_STEER_LIMIT_RAD)
            speed = np.clip(speed * np.asarray(wheel_gain, np.float32), -FW_MAX_WHEEL_MPS, FW_MAX_WHEEL_MPS)
        return goal.astype(np.float32), speed.astype(np.float32)


def action_to_residual(action, steer_scale_rad=None):
    """Policy output (8,) in [-1, 1] -> (d_steer [rad] x4, wheel_gain x4), as Res8Action does.
    steer_scale_rad: 조향 잔차 폭 덮어쓰기 (10-16 모래 비교 --res-steer-deg). None = 학습값."""
    a = np.clip(np.nan_to_num(np.asarray(action, np.float32), nan=0.0), -1.0, 1.0)
    k = C.RES8_STEER_SCALE_RAD if steer_scale_rad is None else steer_scale_rad
    return (a[:4] * k).astype(np.float32), 1.0 + a[4:] * C.RES8_WHEEL_SCALE


class StudentObs:
    """Builds the student input vector from SPI telemetry, in the exported layout."""

    TERMS = ("command", "gyro", "gravity", "steering", "steering_rate", "wheel_rate", "applied_targets", "vo")
    VO_AGE_CAP = 1.0      # rover_env_cfg.VO_DR["age_cap"]: what the sim feeds when there is no VO

    def __init__(self, meta: dict, dt: float = 0.02):
        self.hist = int(meta["student_history"])
        self.layout = []
        for t in meta["terms"]:
            total = int(np.prod(t["shape"]))
            if t["name"] not in self.TERMS or total % self.hist:
                raise ValueError(f"모르는 관측 항목/크기: {t}")
            self.layout.append((t["name"], total // self.hist))
        if sum(d * self.hist for _, d in self.layout) != int(meta["obs_dim"]):
            raise ValueError("obs_dim 불일치")
        self.gfilt = GravityFilter(dt=dt)
        self.reset()

    def reset(self):
        self.buf = None
        self.gfilt.reset()

    def frame(self, cmd, tlm, gyro_bias, applied, vo=None):
        """One 20 ms frame (dict name -> vector).  applied = (steer goal x4 rad, wheel target x4 rad/s).
        vo = (vx, vy [m/s, body], age [s]) of the latest visual-odometry velocity, or None (no VO yet) -> the
        sim's "no VO" value [0, 0, 0, age_cap].  Only used by students exported with the vo term (v3)."""
        gyro = np.asarray(tlm["gyro_dps"], float) * math.pi / 180.0 - gyro_bias
        acc = np.asarray(tlm["acc_g"], float) * 9.81
        grav = self.gfilt.step(gyro, acc)
        return {
            "command": np.asarray(cmd, np.float32),
            "gyro": gyro.astype(np.float32),
            "gravity": np.asarray(grav, np.float32),
            "steering": np.asarray(tlm["steer_meas_rad"], np.float32),
            "steering_rate": np.asarray(tlm["steer_speed"], np.float32) * STEER_STEP_RAD,
            "wheel_rate": np.asarray(tlm["wheel_mps"], np.float32) / WHEEL_RADIUS_M,
            "applied_targets": np.concatenate([np.asarray(applied[0], np.float32),
                                               np.asarray(applied[1], np.float32)]),
            "vo": (np.array([0.0, 0.0, 0.0, self.VO_AGE_CAP], np.float32) if vo is None or vo[2] >= 0.5 else
                   np.array([vo[0], vo[1], 1.0, min(float(vo[2]), self.VO_AGE_CAP)], np.float32)),
        }

    def push(self, fr) -> np.ndarray:
        """Append a frame, return the flat observation (1, obs_dim)."""
        if self.buf is None:
            self.buf = {n: np.repeat(fr[n][None].astype(np.float32), self.hist, 0) for n, _ in self.layout}
        else:
            for n, _ in self.layout:
                self.buf[n] = np.concatenate([self.buf[n][1:], fr[n][None].astype(np.float32)], 0)
        for n, d in self.layout:
            if self.buf[n].shape[1] != d:
                raise ValueError(f"{n}: {self.buf[n].shape[1]} != {d}")
        return np.concatenate([self.buf[n].reshape(-1) for n, _ in self.layout])[None]


def applied_for_obs(ik_goal, ik_speed_mps, tlm=None):
    """What the sim's applied_targets holds (2026-10-14): the targets the policy just commanded -- IK + residual,
    before the Nucleo's command delay and drive gate.  ``tlm`` is unused (kept for the call sites)."""
    w = np.asarray(ik_speed_mps, np.float32) / np.float32(WHEEL_RADIUS_M)
    return np.asarray(ik_goal, np.float32), w
