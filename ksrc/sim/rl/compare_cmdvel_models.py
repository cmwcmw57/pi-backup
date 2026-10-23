#!/usr/bin/env python3
"""Paired cmd_vel evaluation of IK, two saved body policies, and truth P.

The policy files must have the same 3D action and observation interface.
Each arm gets the same reset seed and command schedule; report episode-paired
confidence intervals for the new policy relative to the previous policy.
"""
import argparse
import pickle

import numpy as np
from stable_baselines3 import PPO

from env import CTRL_HZ, OBS_PER_FRAME, PRIV_DIM, RoverEnv


FIELDS = (("v_err_m", 100), ("v_err_p90", 100),
          ("om_err_m", 100), ("om_err_p90", 100),
          ("slip_m", 1), ("slip_p90", 1), ("sink_m", 100),
          ("stuck", 100), ("tip", 100), ("oob", 100),
          ("a_use", 100), ("cmd_sat", 100))


def load_policy(model_path, norm_path):
    model = PPO.load(model_path, device="cpu")
    with open(norm_path, "rb") as f:
        norm = pickle.load(f).obs_rms
    if model.action_space.shape != (3,) or norm.mean.shape != model.observation_space.shape:
        raise ValueError(f"policy/norm interface mismatch: {model_path}")
    return model, norm


def rollout(env, seed, policy, oracle_gain, action_scale=1.0):
    obs, _ = env.reset(seed=seed)
    commands = []
    while True:
        commands.append(env.cmd_nom.copy())
        if policy is None:
            if oracle_gain is None:
                action = np.zeros(3, np.float32)
            else:
                lim = np.array([env.cfg.d_vx_max, env.cfg.d_vy_max,
                                env.cfg.d_om_max])
                action = np.clip(oracle_gain * (env.cmd_nom-env.body_twist()) / lim,
                                 -1, 1).astype(np.float32)
        else:
            model, norm = policy
            obs_n = np.clip((obs-norm.mean) / np.sqrt(norm.var+1e-8),
                            -10, 10).astype(np.float32)
            action = np.clip(action_scale * model.predict(
                obs_n, deterministic=True)[0], -1, 1)
        obs, _, term, trunc, info = env.step(action)
        if term or trunc:
            return info, np.asarray(commands)


def ci(x, seed=0):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(x), size=(10000, len(x)))
    return np.quantile(x[idx].mean(axis=1), [0.025, 0.975])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("old", "new"):
        p.add_argument(f"--{name}-model", required=True)
        p.add_argument(f"--{name}-vecnorm", required=True)
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--seed0", type=int, default=35000)
    p.add_argument("--episode-s", type=float, default=8.0)
    p.add_argument("--difficulty", type=float, default=0.8)
    p.add_argument("--difficulty-min-frac", type=float, default=0.6,
                   help="episode difficulty is sampled in [fraction*difficulty, difficulty]")
    p.add_argument("--soil-amp", type=float, default=None)
    p.add_argument("--push-force", type=float, default=0.0)
    p.add_argument("--cmd-filter-tau", type=float, default=0.10)
    p.add_argument("--history-frames", type=int, default=5)
    p.add_argument("--oracle-gain", type=float, default=0.30)
    p.add_argument("--new-action-scale", type=float, default=1.0,
                   help="multiply only the new policy residual for an ablation")
    args = p.parse_args()
    if (args.episodes < 2 or args.episode_s <= 0 or
            not np.isfinite(args.push_force) or args.push_force < 0 or
            not np.isfinite(args.new_action_scale) or
            not 0 <= args.new_action_scale <= 1 or
            not np.isfinite(args.difficulty_min_frac) or
            not 0 <= args.difficulty_min_frac <= 1):
        p.error("episodes >=2, episode-s positive, push-force >=0, "
                "new-action-scale and difficulty-min-frac in [0,1] required")
    policies = {name: load_policy(getattr(args, f"{name}_model"),
                                  getattr(args, f"{name}_vecnorm"))
                for name in ("old", "new")}
    expected = OBS_PER_FRAME * args.history_frames + PRIV_DIM
    for name, (model, _) in policies.items():
        if model.observation_space.shape != (expected,):
            p.error(f"{name} observation must be {expected}D for this comparison")
    settings = dict(task="cmdvel", action_mode="body", privileged=True,
                    history_frames=args.history_frames, randomize=True,
                    arena_eval=False, episode_s=args.episode_s,
                    difficulty=args.difficulty, soil_amp=args.soil_amp,
                    difficulty_min_frac=args.difficulty_min_frac,
                    cmd_filter_tau=args.cmd_filter_tau,
                    push_force_max=args.push_force)
    names = ("IK", "old", "new", "oracle_P")
    envs = {name: RoverEnv(**settings) for name in names}
    rows = {name: [] for name in names}
    cmd_mismatch = 0.0
    try:
        for seed in range(args.seed0, args.seed0+args.episodes):
            commands = {}
            for name in names:
                policy = policies.get(name)
                gain = args.oracle_gain if name == "oracle_P" else None
                scale = args.new_action_scale if name == "new" else 1.0
                info, cmds = rollout(envs[name], seed, policy, gain, scale)
                rows[name].append(info)
                commands[name] = cmds
            for name in names[1:]:
                n = min(len(commands["IK"]), len(commands[name]))
                cmd_mismatch = max(cmd_mismatch, float(np.max(np.abs(
                    commands["IK"][:n] - commands[name][:n]))))
            if (seed-args.seed0+1) % 10 == 0:
                print(f"{seed-args.seed0+1}/{args.episodes} pairs", flush=True)
    finally:
        for env in envs.values():
            env.close()
    print(f"command mismatch max={cmd_mismatch:.6f}")
    print("metric          IK      old      new   oracle   new-old [95% CI]"
          "    new-IK [95% CI]")
    fields = (FIELDS + (("push_v_err_m", 100), ("push_om_err_m", 100))
              if args.push_force > 0 else FIELDS)
    for field, scale in fields:
        x = {name: np.array([row[field] for row in rows[name]])*scale
             for name in names}
        if not all(np.isfinite(a).all() for a in x.values()):
            print(f"{field:<13} N/A (some episodes ended before the push recovery window)")
            continue
        d_old = x["new"]-x["old"]
        d_ik = x["new"]-x["IK"]
        lo, hi = ci(d_old)
        li, ui = ci(d_ik)
        print(f"{field:<13} {x['IK'].mean():7.3f} {x['old'].mean():7.3f} "
              f"{x['new'].mean():7.3f} {x['oracle_P'].mean():7.3f} "
              f"{d_old.mean():+7.3f} [{lo:+.3f},{hi:+.3f}] "
              f"{d_ik.mean():+7.3f} [{li:+.3f},{ui:+.3f}]")


if __name__ == "__main__":
    main()
