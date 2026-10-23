#!/usr/bin/env python3
"""Locate cmdvel tracking tails within each two-second command interval."""
import argparse
import pickle

import numpy as np
from stable_baselines3 import PPO

from env import OBS_PER_FRAME, PRIV_DIM, RoverEnv


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--vecnorm", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=14000)
    ap.add_argument("--history-frames", type=int, default=5)
    ap.add_argument("--episode-s", type=float, default=8)
    ap.add_argument("--difficulty", type=float, default=0.8)
    ap.add_argument("--soil-amp", type=float, default=None)
    ap.add_argument("--cmd-filter-tau", type=float, default=0.28)
    ap.add_argument("--oracle-gain", type=float, default=None,
                    help="use simulator-truth body P instead of the saved policy")
    args = ap.parse_args()
    model = PPO.load(args.model, device="cpu")
    with open(args.vecnorm, "rb") as f:
        norm = pickle.load(f).obs_rms
    regular_dim = OBS_PER_FRAME * args.history_frames
    if model.observation_space.shape[0] not in (regular_dim, regular_dim + PRIV_DIM):
        ap.error("model observation does not match requested history")
    env_kw = dict(task="cmdvel", action_mode="body", history_frames=args.history_frames,
                  privileged=model.observation_space.shape[0] > regular_dim,
                  difficulty=args.difficulty, soil_amp=args.soil_amp,
                  episode_s=args.episode_s, randomize=True, arena_eval=False,
                  cmd_filter_tau=args.cmd_filter_tau)
    records = {}
    for name, assist in (("IK", False), ("teacher", True)):
        env = RoverEnv(**env_kw)
        bins = {"0-.25s": [], ".25-.75s": [], ".75-2s": []}
        for seed in range(args.seed0, args.seed0 + args.episodes):
            obs, _ = env.reset(seed=seed)
            while True:
                phase = (env._cmd_k - 1) % env.cmd_hold
                key = ("0-.25s" if phase < 13 else
                       ".25-.75s" if phase < 38 else ".75-2s")
                if assist:
                    if args.oracle_gain is not None:
                        limits = np.array([env.cfg.d_vx_max, env.cfg.d_vy_max,
                                           env.cfg.d_om_max])
                        action = np.clip(args.oracle_gain *
                                         (env.cmd_nom - env.body_twist()) / limits,
                                         -1, 1).astype(np.float32)
                    else:
                        obs_n = np.clip((obs - norm.mean) / np.sqrt(norm.var + 1e-8),
                                        -10, 10).astype(np.float32)
                        action = model.predict(obs_n, deterministic=True)[0]
                else:
                    action = np.zeros(3, np.float32)
                obs, _, term, trunc, _ = env.step(action)
                v_err, om_err = env.vel_err_hist[-1]
                bins[key].append((v_err, om_err,
                                  float(np.abs(env.tm.slip).mean())))
                if term or trunc:
                    break
        records[name] = {k: np.asarray(v) for k, v in bins.items()}
        env.close()
    print("phase       metric       IK       teacher    delta  (all-step means/p90)")
    for phase in records["IK"]:
        for col, label, scale in ((0, "v mean", 100), (0, "v p90", 100),
                                  (1, "om mean", 100), (1, "om p90", 100),
                                  (2, "slip mean", 1)):
            x, y = records["IK"][phase][:, col], records["teacher"][phase][:, col]
            fn = np.quantile if "p90" in label else np.mean
            b = fn(x, .9) if fn is np.quantile else fn(x)
            r = fn(y, .9) if fn is np.quantile else fn(y)
            print(f"{phase:<11} {label:<9} {b*scale:7.2f} {r*scale:11.2f} "
                  f"{(r-b)*scale:+8.2f}")


if __name__ == "__main__":
    main()
