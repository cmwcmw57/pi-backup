"""Flat-ground pure IK rollout matching the Isaac task's low-level actuator model."""

import argparse
import pathlib
import sys

import mujoco
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
ISAAC = ROOT / "sim/isaac/ksrc_rover"
sys.path.insert(0, str(ISAAC / "source"))
sys.path.insert(0, str(ISAAC / "scripts"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from ik_benchmark_common import DT, schedule  # noqa: E402
from config import RoverCfg  # noqa: E402
from ksrc_rover.utils.contract import DRIVE_EFFORT_NM, STEER_MAX_RAD_S, WHEEL_MAX_RAD_S, WHEEL_RADIUS_M  # noqa: E402
from ksrc_rover.utils.ik_torch import BatchedSwerveIK  # noqa: E402
from mjcf import build  # noqa: E402


def ids(model, kind, names):
    return np.asarray([mujoco.mj_name2id(model, kind, name) for name in names], dtype=int)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("/tmp/pure_ik_mujoco.npz"))
    parser.add_argument("--drive-kd", type=float, default=0.20)
    parser.add_argument("--scenario", choices=("full", "arc_left", "arc_right"), default="full")
    args = parser.parse_args()
    cfg = RoverCfg()
    assert abs(cfg.wheel_r - WHEEL_RADIUS_M) < 1e-6
    model = mujoco.MjModel.from_xml_string(build(cfg, terrain="plane"))
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)
    corners = ("fl", "fr", "rl", "rr")
    steer_j = ids(model, mujoco.mjtObj.mjOBJ_JOINT, [f"st_{n}" for n in corners])
    wheel_j = ids(model, mujoco.mjtObj.mjOBJ_JOINT, [f"wh_{n}" for n in corners])
    steer_q = np.asarray([model.jnt_qposadr[i] for i in steer_j])
    wheel_v = np.asarray([model.jnt_dofadr[i] for i in wheel_j])
    steer_a = ids(model, mujoco.mjtObj.mjOBJ_ACTUATOR, [f"a_st_{n}" for n in corners])
    wheel_a = ids(model, mujoco.mjtObj.mjOBJ_ACTUATOR, [f"a_wh_{n}" for n in corners])
    rocker_j = ids(model, mujoco.mjtObj.mjOBJ_JOINT, ["rock_l", "rock_r"])
    rocker_q = np.asarray([model.jnt_qposadr[i] for i in rocker_j])
    # Isaac DCMotor applies torque explicitly; remove MuJoCo's passive DC droop
    # so the same torque-speed clipping is not counted twice.
    model.dof_damping[wheel_v] = 0.0
    data.qpos[2] = 0.15
    data.qpos[3] = 1.0
    mujoco.mj_forward(model, data)
    ik = BatchedSwerveIK(1, "cpu")
    steer_applied = np.zeros(4)
    names, commands = schedule(args.scenario)
    rows = {k: [] for k in ("body", "xy", "yaw", "steer_target", "steer_actual", "wheel_target", "wheel_actual", "rocker_actual")}
    for command in commands:
        theta, wheel_mps = ik.compute(torch.as_tensor(command[None, :]))
        steer_goal = theta[0].numpy()
        steer_applied += np.clip(steer_goal - steer_applied, -STEER_MAX_RAD_S * DT, STEER_MAX_RAD_S * DT)
        steer_target = steer_applied.copy()
        wheel_target = wheel_mps[0].numpy() / WHEEL_RADIUS_M
        data.ctrl[steer_a] = steer_target
        for _ in range(round(DT / model.opt.timestep)):
            wheel_actual = data.qvel[wheel_v]
            effort = args.drive_kd * (wheel_target - wheel_actual)
            hi = np.minimum(DRIVE_EFFORT_NM, DRIVE_EFFORT_NM * (1.0 - wheel_actual / WHEEL_MAX_RAD_S))
            lo = np.maximum(-DRIVE_EFFORT_NM, DRIVE_EFFORT_NM * (-1.0 - wheel_actual / WHEEL_MAX_RAD_S))
            data.ctrl[wheel_a] = np.clip(effort, lo, hi) / DRIVE_EFFORT_NM
            mujoco.mj_step(model, data)
        rot = np.empty(9)
        mujoco.mju_quat2Mat(rot, data.qpos[3:7])
        rot = rot.reshape(3, 3)
        body_lin = rot.T @ data.qvel[:3]
        body_ang = rot.T @ data.qvel[3:6]
        sample = {
            "body": np.array([body_lin[0], body_lin[1], body_ang[2]]),
            "xy": data.qpos[:2].copy(),
            "yaw": np.array([np.arctan2(rot[1, 0], rot[0, 0])]),
            "steer_target": steer_target,
            "steer_actual": data.qpos[steer_q].copy(),
            "wheel_target": wheel_target,
            "wheel_actual": data.qvel[wheel_v].copy(),
            "rocker_actual": data.qpos[rocker_q].copy(),
        }
        for key, value in sample.items():
            rows[key].append(np.asarray(value).copy())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, phase=names, command=commands, dt=DT,
                        **{key: np.stack(value) for key, value in rows.items()})
    print(f"IK_BENCHMARK_OK {args.out} steps={len(names)}", flush=True)


if __name__ == "__main__":
    main()
