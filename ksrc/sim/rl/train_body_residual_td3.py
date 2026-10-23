#!/usr/bin/env python3
"""Pilot: off-policy body residual above a fixed simulator-truth P controller.

Teacher feasibility only: body_twist() is MuJoCo truth. The learned residual
does not replace a validated ground-velocity estimator on the physical rover.
The residual action is held for several 50 Hz physics-control cycles so that
the policy can observe its delayed effect; the P base is recomputed each cycle.
"""
import argparse
import json
from pathlib import Path
import shutil

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise

from env import RoverEnv
from reward import VelRewardCfg


class BodyResidualEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, *, difficulty=0.8, soil_amp=0.25, episode_s=8.0,
                 cmd_filter_tau=0.10, body_gain=0.3, residual_scale=0.5,
                 action_hold=5, w_tail=0.03, push_force=0.0):
        if action_hold < 1 or residual_scale <= 0 or body_gain < 0:
            raise ValueError("action_hold>=1, residual_scale>0, body_gain>=0 required")
        self.core = RoverEnv(task="cmdvel", action_mode="body", privileged=True,
                             history_frames=5, randomize=True, arena_eval=False,
                             difficulty=difficulty, soil_amp=soil_amp,
                             episode_s=episode_s, cmd_filter_tau=cmd_filter_tau,
                             vel_rew=VelRewardCfg(w_tail=w_tail),
                             push_force_max=push_force)
        self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)
        self.observation_space = self.core.observation_space
        self.body_gain = float(body_gain)
        self.residual_scale = float(residual_scale)
        self.action_hold = int(action_hold)
        self.previous_residual = np.zeros(3)

    def reset(self, *, seed=None, options=None):
        self.previous_residual[:] = 0
        return self.core.reset(seed=seed, options=options)

    def step(self, action):
        residual = np.clip(np.asarray(action, dtype=float), -1, 1)
        total_reward = 0.0
        n = 0
        for _ in range(self.action_hold):
            core = self.core
            limits = np.array([core.cfg.d_vx_max, core.cfg.d_vy_max,
                               core.cfg.d_om_max])
            p = np.clip(self.body_gain *
                        (core.cmd_nom-core.body_twist()) / limits, -1, 1)
            physical = np.clip(p + self.residual_scale * residual, -1, 1)
            obs, reward, term, trunc, info = core.step(physical.astype(np.float32))
            total_reward += reward
            n += 1
            if term or trunc:
                break
        # Penalize only the *extra* command. The base controller is identical
        # in the policy and zero-residual comparison arms.
        total_reward = total_reward / n - 0.01 * float(np.mean(residual ** 2)) \
            - 0.005 * float(np.mean((residual-self.previous_residual) ** 2))
        self.previous_residual = residual.copy()
        return obs, total_reward, term, trunc, info

    def close(self):
        self.core.close()


def evaluate(model, settings, episodes, seed0):
    names = ("IK", "body_P", "body_P+TD3")
    envs = {"IK": RoverEnv(task="cmdvel", action_mode="body", privileged=True,
                            history_frames=5, randomize=True, arena_eval=False,
                            difficulty=settings["difficulty"],
                            soil_amp=settings["soil_amp"],
                            episode_s=settings["episode_s"],
                            cmd_filter_tau=settings["cmd_filter_tau"],
                            push_force_max=settings["push_force"]),
            "body_P": BodyResidualEnv(**settings),
            "body_P+TD3": BodyResidualEnv(**settings)}
    rows = {name: [] for name in names}
    try:
        for seed in range(seed0, seed0+episodes):
            for name in names:
                obs, _ = envs[name].reset(seed=seed)
                while True:
                    action = (model.predict(obs, deterministic=True)[0]
                              if name == "body_P+TD3" else np.zeros(3, np.float32))
                    obs, _, term, trunc, info = envs[name].step(action)
                    if term or trunc:
                        rows[name].append(info)
                        break
            if (seed-seed0+1) % 5 == 0:
                print(f"eval {seed-seed0+1}/{episodes}", flush=True)
    finally:
        for env in envs.values():
            env.close()
    rng = np.random.default_rng(17)
    index = rng.integers(episodes, size=(10000, episodes))
    print("metric            IK     body_P   body_P+TD3   TD3-P [95% CI]")
    for key, scale in (("v_err_m", 100), ("v_err_p90", 100),
                       ("om_err_m", 100), ("om_err_p90", 100),
                       ("slip_p90", 1), ("sink_m", 100),
                       ("stuck", 100), ("tip", 100), ("oob", 100)):
        vals = {name: np.array([i[key] for i in rows[name]])*scale for name in names}
        diff = vals["body_P+TD3"]-vals["body_P"]
        lo, hi = np.quantile(diff[index].mean(axis=1), [.025, .975])
        print(f"{key:<15} {vals['IK'].mean():7.3f} {vals['body_P'].mean():8.3f} "
              f"{vals['body_P+TD3'].mean():11.3f} {diff.mean():+8.3f} "
              f"[{lo:+.3f},{hi:+.3f}]")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=20000,
                   help="macro-actions; each executes action-hold 50 Hz cycles")
    p.add_argument("--out", type=Path)
    p.add_argument("--eval-model", type=Path)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed0", type=int, default=47000)
    p.add_argument("--seed", type=int, default=42000)
    p.add_argument("--difficulty", type=float, default=0.8)
    p.add_argument("--soil-amp", type=float, default=0.25)
    p.add_argument("--episode-s", type=float, default=8.0)
    p.add_argument("--cmd-filter-tau", type=float, default=0.10)
    p.add_argument("--body-gain", type=float, default=0.30)
    p.add_argument("--residual-scale", type=float, default=0.50)
    p.add_argument("--action-hold", type=int, default=5)
    p.add_argument("--w-tail", type=float, default=0.03)
    p.add_argument("--push-force", type=float, default=0.0)
    args = p.parse_args()
    settings = {name: getattr(args, name) for name in
                ("difficulty", "soil_amp", "episode_s", "cmd_filter_tau",
                 "body_gain", "residual_scale", "action_hold", "w_tail",
                 "push_force")}
    if args.episodes < 2:
        p.error("episodes must be >=2")
    if args.eval_model is not None:
        model = TD3.load(args.eval_model, device="cpu")
        evaluate(model, settings, args.episodes, args.seed0)
        return
    if args.out is None or args.steps < 1:
        p.error("training requires --out and --steps >0")
    if args.out.exists() and any(args.out.iterdir()):
        p.error("--out must be new or empty")
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.out / "src"
    source.mkdir(exist_ok=True)
    for name in ("train_body_residual_td3.py", "env.py", "config.py",
                 "reward.py", "terramech.py"):
        shutil.copy2(Path(__file__).with_name(name), source/name)
    (args.out/"manifest.json").write_text(json.dumps(
        {"args": {k: str(v) if isinstance(v, Path) else v
                  for k, v in vars(args).items()}, "environment": settings}, indent=2))
    env = BodyResidualEnv(**settings)
    noise = NormalActionNoise(mean=np.zeros(3), sigma=np.full(3, 0.25))
    model = TD3("MlpPolicy", env, seed=args.seed, device="cpu", verbose=1,
                learning_rate=3e-4, buffer_size=100000, learning_starts=1000,
                batch_size=256, train_freq=(1, "step"), gradient_steps=1,
                gamma=0.99, action_noise=noise,
                policy_kwargs=dict(net_arch=[256, 256]))
    try:
        model.learn(total_timesteps=args.steps,
                    callback=CheckpointCallback(
                        save_freq=5000, save_path=str(args.out), name_prefix="td3"))
        model.save(args.out/"model")
    finally:
        env.close()
    print(f"saved {args.out/'model.zip'}")


if __name__ == "__main__":
    main()
