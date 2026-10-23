#!/usr/bin/env python3
"""SAC feasibility pilot for cmd_vel body residuals above the truth-P teacher.

This measures whether SAC can improve a fixed feedback teacher on hard sand.
The feedback teacher uses simulator-truth velocity; this is not deployable.
"""
import argparse
import json
from pathlib import Path
import pickle
import shutil

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

from train_body_residual_td3 import BodyResidualEnv


class FrozenObsNorm(gym.ObservationWrapper):
    """Use the existing PPO teacher's observation scale without updating it."""

    def __init__(self, env, rms):
        super().__init__(env)
        self.mean = np.asarray(rms.mean, dtype=np.float32)
        self.std = np.sqrt(np.asarray(rms.var, dtype=np.float32) + 1e-8)
        if self.mean.shape != env.observation_space.shape:
            raise ValueError("teacher normalization and SAC observation mismatch")
        self.observation_space = spaces.Box(-10, 10, self.mean.shape, np.float32)

    def observation(self, obs):
        return np.clip((obs-self.mean)/self.std, -10, 10).astype(np.float32)


def evaluate(model, rms, settings, episodes, seed0):
    arms = {
        "IK": BodyResidualEnv(**{**settings, "body_gain": 0.0}),
        "P": BodyResidualEnv(**settings),
        "P+SAC": FrozenObsNorm(BodyResidualEnv(**settings), rms),
    }
    rows = {name: [] for name in arms}
    command_diff = 0.0
    try:
        for seed in range(seed0, seed0+episodes):
            cmds = {}
            for name, env in arms.items():
                obs, _ = env.reset(seed=seed)
                commands = []
                while True:
                    core = env.unwrapped.core
                    commands.append(core.cmd_nom.copy())
                    action = (model.predict(obs, deterministic=True)[0]
                              if name == "P+SAC" else np.zeros(3, np.float32))
                    obs, _, term, trunc, info = env.step(action)
                    if term or trunc:
                        rows[name].append(info)
                        cmds[name] = np.asarray(commands)
                        break
            for name in ("P", "P+SAC"):
                n = min(len(cmds["IK"]), len(cmds[name]))
                command_diff = max(command_diff, float(np.max(
                    np.abs(cmds["IK"][:n]-cmds[name][:n]))))
            if (seed-seed0+1) % 5 == 0:
                print(f"eval {seed-seed0+1}/{episodes}", flush=True)
    finally:
        for env in arms.values():
            env.close()
    rng = np.random.default_rng(17)
    index = rng.integers(episodes, size=(10000, episodes))
    print(f"command mismatch max={command_diff:.6f}")
    print("metric              IK        P    P+SAC    SAC-P [95% CI]")
    for key, scale in (("v_err_m", 100), ("v_err_p90", 100),
                       ("om_err_m", 100), ("om_err_p90", 100),
                       ("slip_p90", 1), ("sink_m", 100),
                       ("stuck", 100), ("tip", 100), ("oob", 100)):
        x = {name: np.asarray([r[key] for r in rows[name]])*scale
             for name in arms}
        d = x["P+SAC"]-x["P"]
        lo, hi = np.quantile(d[index].mean(axis=1), [.025, .975])
        print(f"{key:<16} {x['IK'].mean():7.3f} {x['P'].mean():8.3f} "
              f"{x['P+SAC'].mean():8.3f} {d.mean():+8.3f} "
              f"[{lo:+.3f},{hi:+.3f}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--eval-model", type=Path)
    ap.add_argument("--vecnorm", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=55000)
    ap.add_argument("--seed", type=int, default=52000)
    ap.add_argument("--difficulty", type=float, default=1.0)
    ap.add_argument("--soil-amp", type=float, default=1.0)
    ap.add_argument("--episode-s", type=float, default=12.0)
    ap.add_argument("--cmd-filter-tau", type=float, default=0.10)
    ap.add_argument("--body-gain", type=float, default=0.30)
    ap.add_argument("--residual-scale", type=float, default=0.25)
    ap.add_argument("--action-hold", type=int, default=5)
    ap.add_argument("--w-tail", type=float, default=0.03)
    ap.add_argument("--push-force", type=float, default=0.0)
    args = ap.parse_args()
    if args.episodes < 2 or args.steps < 1 or args.out is None and args.eval_model is None:
        ap.error("episodes>=2 and training requires --steps and --out")
    with open(args.vecnorm, "rb") as f:
        rms = pickle.load(f).obs_rms
    settings = {name: getattr(args, name) for name in
                ("difficulty", "soil_amp", "episode_s", "cmd_filter_tau",
                 "body_gain", "residual_scale", "action_hold", "w_tail",
                 "push_force")}
    if args.eval_model is not None:
        evaluate(SAC.load(args.eval_model, device="cpu"), rms, settings,
                 args.episodes, args.seed0)
        return
    if args.out.exists() and any(args.out.iterdir()):
        ap.error("--out must be new or empty")
    args.out.mkdir(parents=True, exist_ok=True)
    src = args.out/"src"
    src.mkdir(exist_ok=True)
    for name in ("train_body_residual_sac.py", "train_body_residual_td3.py",
                 "env.py", "config.py", "reward.py", "terramech.py"):
        shutil.copy2(Path(__file__).with_name(name), src/name)
    shutil.copy2(args.vecnorm, args.out/"vecnorm.pkl")
    (args.out/"manifest.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        indent=2)+"\n")
    env = FrozenObsNorm(BodyResidualEnv(**settings), rms)
    model = SAC("MlpPolicy", env, seed=args.seed, device="cpu", verbose=1,
                learning_rate=3e-4, buffer_size=100000, learning_starts=1000,
                batch_size=256, train_freq=(1, "step"), gradient_steps=1,
                gamma=0.99, ent_coef="auto_0.01",
                policy_kwargs=dict(net_arch=[256, 256]))
    try:
        model.learn(total_timesteps=args.steps,
                    callback=CheckpointCallback(
                        save_freq=5000, save_path=str(args.out), name_prefix="sac"))
        model.save(args.out/"model")
    finally:
        env.close()
    print(f"saved {args.out/'model.zip'}")


if __name__ == "__main__":
    main()
