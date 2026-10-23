#!/usr/bin/env python3
"""Pilot: train wheel assistance on top of a fixed truth P body controller.

The body controller and privileged observation use simulator truth. This is a
teacher feasibility experiment, not a deployable student or a hardware command.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import shutil

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env import CTRL_HZ, RoverEnv
from reward import VelRewardCfg


def wheel_asymmetry(core, metric):
    if metric == "raw":
        return float(np.std(core.tm.slip))
    valid = ((np.abs(core.wheel_ref_applied) > 0.02 * core.v_max) &
             (core.tm.Fn > 1e-6))
    if np.count_nonzero(valid) < 3:
        return 0.0
    traction = core.tm.slip * np.sign(core.wheel_ref_applied)
    return float(np.std(traction[valid]))


class DifferentialWheelEnv(gym.ActionWrapper):
    """A 4D wheel policy with identical fixed body feedback in both arms."""

    def __init__(self, *, body_gain=0.3, difficulty=0.8, episode_s=8.0,
                 soil_amp=0.25, cmd_filter_tau=0.10, w_tail=0.03,
                 allow_wheel_common=False, gate_asym=None,
                 gate_before_s=0.75, gate_metric="raw"):
        env = RoverEnv(task="cmdvel", action_mode="body+wheel", privileged=True,
                       history_frames=5, difficulty=difficulty,
                       episode_s=episode_s, soil_amp=soil_amp,
                       cmd_filter_tau=cmd_filter_tau, randomize=True,
                       arena_eval=False, vel_rew=VelRewardCfg(w_tail=w_tail))
        super().__init__(env)
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)
        self.body_gain = float(body_gain)
        self.allow_wheel_common = bool(allow_wheel_common)
        self.gate_asym = gate_asym
        self.gate_before_s = float(gate_before_s)
        self.gate_metric = gate_metric

    def action(self, wheel_action):
        core = self.env
        limits = np.array([core.cfg.d_vx_max, core.cfg.d_vy_max,
                           core.cfg.d_om_max])
        body = np.clip(self.body_gain *
                       (core.cmd_nom - core.body_twist()) / limits, -1.0, 1.0)
        wheel = np.clip(np.asarray(wheel_action, dtype=float), -1.0, 1.0)
        if not self.allow_wheel_common:
            wheel -= np.mean(wheel)
            wheel /= max(1.0, float(np.max(np.abs(wheel))))
        if self.gate_asym is not None:
            phase = ((core._cmd_k - 1) % core.cmd_hold) / CTRL_HZ
            if (phase >= self.gate_before_s or
                    wheel_asymmetry(core, self.gate_metric) <= self.gate_asym):
                wheel[:] = 0.0
        return np.concatenate([body, wheel]).astype(np.float32)


def evaluate(model_path, norm_path, args):
    model = PPO.load(model_path, device="cpu")
    with open(norm_path, "rb") as f:
        rms = pickle.load(f).obs_rms
    settings = dict(body_gain=args.body_gain, difficulty=args.difficulty,
                    episode_s=args.episode_s, soil_amp=args.soil_amp,
                    cmd_filter_tau=args.cmd_filter_tau, w_tail=args.w_tail,
                    allow_wheel_common=args.allow_wheel_common,
                    gate_asym=args.gate_asym,
                    gate_before_s=args.gate_before_s,
                    gate_metric=args.gate_metric)
    env = DifferentialWheelEnv(**settings)
    records = {"IK": [], "body": [], "body+wheel": []}
    traces = {"IK": [], "body": [], "body+wheel": []}
    max_command_mismatch = 0.0
    try:
        for seed in range(args.seed0, args.seed0 + args.episodes):
            commands = {}
            for name in records:
                obs, _ = env.reset(seed=seed)
                trace = []
                cmds = []
                while True:
                    core = env.env
                    phase = ((core._cmd_k - 1) % core.cmd_hold) / CTRL_HZ
                    asym = wheel_asymmetry(core, args.gate_metric)
                    cmds.append(core.cmd_nom.copy())
                    if name in ("IK", "body"):
                        wheel = np.zeros(4, np.float32)
                    else:
                        obs_n = np.clip((obs-rms.mean) / np.sqrt(rms.var+1e-8),
                                        -10, 10).astype(np.float32)
                        wheel = model.predict(obs_n, deterministic=True)[0]
                        if args.wheel_component == "common":
                            wheel = np.full(4, float(np.mean(wheel)), np.float32)
                        elif args.wheel_component == "differential":
                            wheel = wheel - np.mean(wheel)
                    if name == "IK":
                        obs, _, term, trunc, info = core.step(
                            np.zeros(7, np.float32))
                    else:
                        obs, _, term, trunc, info = env.step(wheel)
                    trace.append((phase, asym, *core.vel_err_hist[-1]))
                    if term or trunc:
                        break
                records[name].append(info)
                traces[name].append(np.asarray(trace))
                commands[name] = np.asarray(cmds)
            for name in ("body", "body+wheel"):
                n = min(len(commands["IK"]), len(commands[name]))
                max_command_mismatch = max(
                    max_command_mismatch,
                    float(np.max(np.abs(commands["IK"][:n] -
                                        commands[name][:n]))))
            print(f"seed {seed} complete", flush=True)
    finally:
        env.close()
    rng = np.random.default_rng(123)
    def ci_mean(values):
        values = np.asarray(values, dtype=float)
        if not len(values):
            return (float("nan"), float("nan"))
        idx = rng.integers(0, len(values), size=(10000, len(values)))
        return tuple(np.percentile(values[idx].mean(axis=1), (2.5, 97.5)))
    for field, scale in (("v_err_m", 100), ("v_err_p90", 100),
                         ("om_err_m", 100), ("om_err_p90", 100),
                         ("slip_p90", 1), ("sink_m", 100),
                         ("stuck", 100), ("tip", 100),
                         ("wheel_use", 100), ("wheel_desat", 100)):
        body = np.array([i[field] for i in records["body"]]) * scale
        ik = np.array([i[field] for i in records["IK"]]) * scale
        both = np.array([i[field] for i in records["body+wheel"]]) * scale
        delta = both-body
        lo, hi = ci_mean(delta)
        print(f"{field:<15} IK={ik.mean():8.3f} body={body.mean():8.3f} "
              f"body+wheel={both.mean():8.3f} delta={delta.mean():+8.3f} "
              f"CI=[{lo:+.3f},{hi:+.3f}] "
              f"better={np.count_nonzero(delta<0)}/{len(delta)} "
              f"worst_loss={delta.max():+8.3f}")
        if field in ("v_err_m", "v_err_p90", "om_err_m", "om_err_p90"):
            body_minus_ik = body-ik
            lo_ik, hi_ik = ci_mean(body_minus_ik)
            print(f"  body-IK={body_minus_ik.mean():+.3f} "
                  f"CI=[{lo_ik:+.3f},{hi_ik:+.3f}]")
    print(f"command mismatch max={max_command_mismatch:.6f}; "
          "nonzero weakens exact pairing")
    # Select challenge intervals using only the fixed body controller's state.
    for label, mask_fn in (
            ("transition <0.75s", lambda x: x[:, 0] < 0.75),
            ("steady >=0.75s", lambda x: x[:, 0] >= 0.75),
            ("high asym >0.15", lambda x: x[:, 1] > 0.15),
            ("transition+high asym", lambda x: (x[:, 0] < 0.75) & (x[:, 1] > 0.15)),
            ("steady+high asym", lambda x: (x[:, 0] >= 0.75) & (x[:, 1] > 0.15))):
        differences = []
        episode_differences = []
        for base, assisted in zip(traces["body"], traces["body+wheel"]):
            n = min(len(base), len(assisted))
            local = (assisted[:n, 2:4] - base[:n, 2:4])[mask_fn(base[:n])]
            differences.extend(local)
            if len(local):
                episode_differences.append(np.mean(local, axis=0))
        d = np.asarray(differences).reshape(-1, 2)
        if len(d):
            ep_d = np.asarray(episode_differences)
            v_ci = ci_mean(ep_d[:, 0] * 100)
            om_ci = ci_mean(ep_d[:, 1] * 100)
            print(f"{label}: n={len(d)} steps; step-weighted delta "
                  f"v={d[:,0].mean()*100:+.3f}, "
                  f"om={d[:,1].mean()*100:+.3f} %max; "
                  f"v better={np.mean(d[:,0] < 0)*100:.1f}%; "
                  f"episode-mean v={ep_d[:,0].mean()*100:+.3f} "
                  f"om={ep_d[:,1].mean()*100:+.3f}; "
                  f"episode-mean CI v=[{v_ci[0]:+.3f},{v_ci[1]:+.3f}] "
                  f"om=[{om_ci[0]:+.3f},{om_ci[1]:+.3f}] "
                  f"({len(ep_d)} episodes)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--out", type=Path)
    p.add_argument("--resume-model", type=Path,
                   help="continue PPO from a prior model; --steps are additional")
    p.add_argument("--resume-vecnorm", type=Path)
    p.add_argument("--checkpoint-every", type=int, default=50000)
    p.add_argument("--eval-model", type=Path)
    p.add_argument("--vecnorm", type=Path)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed0", type=int, default=26000)
    p.add_argument("--difficulty", type=float, default=0.8)
    p.add_argument("--soil-amp", type=float, default=0.25)
    p.add_argument("--episode-s", type=float, default=8.0)
    p.add_argument("--cmd-filter-tau", type=float, default=0.10)
    p.add_argument("--body-gain", type=float, default=0.3)
    p.add_argument("--w-tail", type=float, default=0.03)
    p.add_argument("--allow-wheel-common", action="store_true",
                   help="do not project wheel actions to zero mean")
    p.add_argument("--wheel-component", choices=("full", "common", "differential"),
                   default="full", help="evaluation ablation of learned 4D action")
    p.add_argument("--gate-asym", type=float,
                   help="wheel policy active when slip std exceeds this")
    p.add_argument("--gate-before-s", type=float, default=0.75,
                   help="wheel policy active only before this command phase")
    p.add_argument("--gate-metric", choices=("raw", "traction"), default="traction",
                   help="asymmetry from raw signed slip or drive-aligned loaded slip")
    p.add_argument("--seed", type=int, default=25000)
    args = p.parse_args()
    if args.gate_asym is not None and not 0.0 <= args.gate_asym <= 1.0:
        p.error("--gate-asym must be in [0,1]")
    if not 0.0 < args.gate_before_s <= 2.0:
        p.error("--gate-before-s must be in (0,2]")
    if args.eval_model is not None:
        if args.vecnorm is None:
            p.error("--eval-model requires --vecnorm")
        evaluate(args.eval_model, args.vecnorm, args)
        return
    if args.out is None or args.steps <= 0:
        p.error("training requires --out and positive --steps")
    if (args.resume_model is None) != (args.resume_vecnorm is None):
        p.error("--resume-model and --resume-vecnorm must be given together")
    if args.checkpoint_every <= 0:
        p.error("--checkpoint-every must be positive")
    if args.out.exists() and any(args.out.iterdir()):
        p.error("--out must be an empty or new directory")
    args.out.mkdir(parents=True, exist_ok=True)
    settings = dict(body_gain=args.body_gain, difficulty=args.difficulty,
                    episode_s=args.episode_s, soil_amp=args.soil_amp,
                    cmd_filter_tau=args.cmd_filter_tau, w_tail=args.w_tail,
                    allow_wheel_common=args.allow_wheel_common,
                    gate_asym=args.gate_asym,
                    gate_before_s=args.gate_before_s,
                    gate_metric=args.gate_metric)
    raw = DummyVecEnv([lambda: Monitor(DifferentialWheelEnv(**settings))])
    if args.resume_model is None:
        vec = VecNormalize(raw, norm_obs=True, norm_reward=True, clip_obs=10.0,
                           gamma=0.998)
    else:
        vec = VecNormalize.load(str(args.resume_vecnorm), raw)
        vec.training = True
        vec.norm_reward = True
    here = Path(__file__).resolve().parent
    hashes = {name: hashlib.sha256((here/name).read_bytes()).hexdigest()[:16]
              for name in ("env.py", "reward.py", "config.py", "terramech.py",
                           "train_wheel_residual.py")}
    if args.resume_model is not None:
        prior = args.resume_model.parent/"manifest.json"
        if prior.is_file():
            prior_hashes = json.loads(prior.read_text())["source_hashes"]
            changed = [name for name in ("env.py", "reward.py", "config.py",
                                         "terramech.py")
                       if prior_hashes.get(name) != hashes[name]]
            if changed:
                p.error(f"resume environment changed: {changed}")
    snapshot = args.out/"src"
    snapshot.mkdir(exist_ok=True)
    for name in ("env.py", "reward.py", "config.py", "terramech.py",
                 "train_wheel_residual.py"):
        shutil.copy2(here/name, snapshot/name)
    (args.out/"manifest.json").write_text(json.dumps(
        {"args": {k: str(v) if isinstance(v, Path) else v
                  for k, v in vars(args).items()}, "source_hashes": hashes,
         "controller": ("truth P body + unrestricted 4D wheel PPO"
                        if args.allow_wheel_common else
                        "truth P body + zero-mean 4D wheel PPO"),
         "gate": (None if args.gate_asym is None else
                  {"metric": args.gate_metric,
                   "slip_std_gt": args.gate_asym,
                   "command_phase_lt_s": args.gate_before_s})}, indent=2))
    if args.resume_model is None:
        model = PPO("MlpPolicy", vec, verbose=1, seed=args.seed, device="cpu",
                    n_steps=512, batch_size=512, n_epochs=6, learning_rate=3e-4,
                    gamma=0.998, gae_lambda=0.95, clip_range=0.2, ent_coef=0.002,
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                                       log_std_init=-2.0))
    else:
        model = PPO.load(args.resume_model, env=vec, device="cpu", verbose=1)
        print(f"resuming from {model.num_timesteps} steps", flush=True)
    try:
        model.learn(total_timesteps=args.steps,
                    reset_num_timesteps=args.resume_model is None,
                    callback=CheckpointCallback(
                        save_freq=args.checkpoint_every, save_path=str(args.out),
                        name_prefix="ppo", save_vecnormalize=True))
        model.save(args.out/"model")
        vec.save(str(args.out/"vecnorm.pkl"))
    finally:
        vec.close()
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
