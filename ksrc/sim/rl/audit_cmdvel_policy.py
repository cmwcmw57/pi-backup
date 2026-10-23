#!/usr/bin/env python3
"""Inspect whether a cmdvel policy's residual points against measured speed error."""
import argparse
import pickle

import numpy as np
from stable_baselines3 import PPO

from env import HISTORY, OBS_PER_FRAME, PRIV_DIM, RoverEnv


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--vecnorm", required=True)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=3000)
    ap.add_argument("--difficulty", type=float, default=0.8)
    ap.add_argument("--velocity-obs", action="store_true")
    ap.add_argument("--history-frames", type=int, default=HISTORY)
    args = ap.parse_args()
    model = PPO.load(args.model, device="cpu")
    regular_dim = (OBS_PER_FRAME + 3 * args.velocity_obs) * args.history_frames
    obs_dim = model.observation_space.shape[0]
    if obs_dim not in (regular_dim, regular_dim + PRIV_DIM):
        ap.error(f"model observation {obs_dim}D does not match {regular_dim}D or "
                 f"{regular_dim + PRIV_DIM}D")
    with open(args.vecnorm, "rb") as f:
        norm = pickle.load(f)
    env = RoverEnv(task="cmdvel", action_mode="body", velocity_obs=args.velocity_obs,
                   difficulty=args.difficulty, episode_s=8, randomize=True,
                   arena_eval=False, history_frames=args.history_frames,
                   privileged=obs_dim > regular_dim)
    errors, actions, priv_deltas = [], [], []
    for seed in range(args.seed0, args.seed0 + args.episodes):
        obs, _ = env.reset(seed=seed)
        while True:
            err = env.cmd_nom - env.body_twist()
            obs_n = np.clip((obs - norm.obs_rms.mean) /
                            np.sqrt(norm.obs_rms.var + 1e-8), -10, 10).astype(np.float32)
            action = np.clip(model.predict(obs_n, deterministic=True)[0], -1, 1)
            if obs_dim > regular_dim:
                ablated = obs_n.copy()
                ablated[regular_dim:] = 0.0
                a_mean = np.clip(model.predict(ablated, deterministic=True)[0], -1, 1)
                priv_deltas.append(np.abs(action - a_mean))
            errors.append(err)
            actions.append(action)
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break
    env.close()
    errors = np.asarray(errors)
    actions = np.asarray(actions)
    scale = np.array([env.cfg.d_vx_max, env.cfg.d_vy_max, env.cfg.d_om_max])
    for j, name in enumerate(("vx", "vy", "omega")):
        corr = np.corrcoef(errors[:, j], actions[:, j])[0, 1]
        err_in_action_units = errors[:, j] / scale[j]
        gain = np.dot(err_in_action_units, actions[:, j]) / max(
            np.dot(err_in_action_units, err_in_action_units), 1e-12)
        print(f"{name:5s}: error_mean={errors[:,j].mean():+.4f} "
              f"action_mean={actions[:,j].mean():+.4f} "
              f"corr(error,action)={corr:+.3f} "
              f"effective_gain={gain:.3f} "
              f"positive_feedback_fraction="
              f"{np.mean(errors[:,j] * actions[:,j] > 0):.2%} "
              f"delta_magnitude={np.mean(np.abs(actions[:,j]*scale[j])):.4f}")
    if priv_deltas:
        print("privilege ablation mean |delta action| by axis:",
              np.mean(priv_deltas, axis=0).round(4),
              "(normalized action range is [-1, 1])")


if __name__ == "__main__":
    main()
