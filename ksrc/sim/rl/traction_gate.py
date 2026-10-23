#!/usr/bin/env python3
"""Check whether slip-aware wheel control has room to improve speed tracking.

This is a diagnostic oracle: it reads simulated per-wheel slip, which the rover
cannot measure directly. A negative result rules out training on this condition;
a positive result establishes only a possible gain, not a deployable policy.
"""
import argparse

import mujoco
import numpy as np

import terrain
from env import RoverEnv
from terramech import PRESETS


def set_slope(env, degrees):
    if degrees == 0:
        return
    nrow, ncol = (int(env.m.hfield_nrow[0]), int(env.m.hfield_ncol[0]))
    x = np.linspace(-env.hf_ex / 2, env.hf_ex / 2, ncol)
    z = np.broadcast_to(np.tan(np.radians(degrees)) * x, (nrow, ncol)).copy()
    old_top = terrain.local_top(env.hf_Z, env.hf_ex, env.hf_ey,
                                env.d.qpos[:2]) + env.hf_off
    base_height = env.d.qpos[2] - old_top
    new_off = terrain.to_hfield(env.m, z)
    env.hf_Z, env.hf_off = z, new_off
    env.d.qpos[:2] = (-2.0, 0.0)
    env.d.qpos[2] = (terrain.local_top(z, env.hf_ex, env.hf_ey,
                                        env.d.qpos[:2]) + new_off + base_height + 0.01)
    p = np.radians(degrees)
    env.d.qpos[3:7] = [np.cos(p / 2), 0, -np.sin(p / 2), 0]
    env.d.qvel[:] = 0
    mujoco.mj_forward(env.m, env.d)
    env.tm.reset()
    for _ in range(int(0.4 / env.m.opt.timestep)):
        env.tm.apply(env.d, env.m.opt.timestep)
        mujoco.mj_step(env.m, env.d)
    env.last_xy = env.d.qpos[:2].copy()


def rollout(seed, mode, args):
    env = RoverEnv(task="cmdvel", action_mode="wheel", randomize=False,
                   arena_eval=False, difficulty=0.8, episode_s=args.seconds,
                   seed=seed)
    env.sample_cmd = lambda: np.array([args.speed, 0.0, 0.0])
    env.cmd_hold = 100000
    env.cmd_a = 1.0
    env.reset(seed=seed)
    env.tm.p = env.terr_p = PRESETS[args.terrain]
    set_slope(env, args.slope_deg)
    samples = []
    for _ in range(env.episode_steps):
        if mode == "ik":
            action = np.zeros(4, np.float32)
        else:
            gain, target = {"tc1": (1, 0.2), "tc2": (2, 0.2),
                            "tc4": (4, 0.2), "tc4low": (4, 0.1)}[mode]
            action = np.clip(-gain * np.maximum(np.abs(env.tm.slip) - target, 0),
                             -1, 0).astype(np.float32)
        _, _, term, trunc, info = env.step(action)
        twist = env.body_twist()
        samples.append((abs(twist[0] - args.speed), abs(twist[1]),
                        abs(twist[2]), np.abs(env.tm.slip).mean(),
                        env.tm.z.mean(), np.abs(action).mean()))
        if term or trunc:
            break
    data = np.asarray(samples)
    out = {"vx_error_mps": data[:, 0].mean(),
           "vy_error_mps": data[:, 1].mean(), "yaw_error_rps": data[:, 2].mean(),
           "slip": data[:, 3].mean(), "sink_m": data[:, 4].mean(),
           "action_use": data[:, 5].mean(), "seconds": len(data) / 50,
           "stuck": int(info["stuck"]), "tip": int(info["tip"]),
           "oob": int(info["oob"])}
    env.close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--speed", type=float, default=0.25)
    ap.add_argument("--terrain", choices=("sand", "loose"), default="loose")
    ap.add_argument("--slope-deg", type=float, default=0)
    ap.add_argument("--seed0", type=int, default=500)
    args = ap.parse_args()
    modes = ("ik", "tc1", "tc2", "tc4", "tc4low")
    rows = {m: [] for m in modes}
    for seed in range(args.seed0, args.seed0 + args.episodes):
        for mode in modes:
            rows[mode].append(rollout(seed, mode, args))
        print(f"seed {seed} complete", flush=True)
    for mode in modes:
        print(mode, {k: round(np.mean([r[k] for r in rows[mode]]), 5)
                     for k in rows[mode][0]}, flush=True)
    for mode in modes[1:]:
        print("delta", mode,
              {k: round(np.mean([rows[mode][i][k] - rows["ik"][i][k]
                                 for i in range(args.episodes)]), 5)
               for k in ("vx_error_mps", "slip", "sink_m", "stuck")},
              flush=True)


if __name__ == "__main__":
    main()
