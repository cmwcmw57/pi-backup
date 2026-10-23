#!/usr/bin/env python3
"""Diagnostic: imitate a simulator-truth velocity feedback controller.

The target action uses MuJoCo body velocity. This is an oracle warm start,
not a deployable rover policy unless a validated ground-speed estimate exists.
"""
import argparse
import json
import pathlib
import pickle
import shutil

import numpy as np
import torch
from stable_baselines3 import PPO

from env import CTRL_HZ, HISTORY, RoverEnv


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--vecnorm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--velocity-obs", action="store_true")
    ap.add_argument("--privileged", action="store_true",
                    help="43D 특권 관측 teacher 모델을 oracle 제어기로 warm start")
    ap.add_argument("--history-frames", type=int, default=HISTORY)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=1,
                    help="1=oracle 시범만; 2 이상=정책 방문 상태를 모아 DAgger 반복")
    ap.add_argument("--gain", type=float, default=0.6)
    ap.add_argument("--difficulty", type=float, default=0.8)
    ap.add_argument("--cmd-filter-tau", type=float, default=0.28,
                    help="cmdvel 명령 필터 [s]; 0.10=ROS 기본값")
    ap.add_argument("--seed0", type=int, default=5000)
    ap.add_argument("--steady-tail-weight", type=float, default=0.0,
                    help="extra imitation weight for large translational errors "
                         "after steering/command settling (0 keeps old behavior)")
    ap.add_argument("--steady-after-s", type=float, default=0.75)
    args = ap.parse_args()
    if args.episodes < 2 or args.epochs < 1 or args.rounds < 1:
        ap.error("--episodes >= 2, --epochs >= 1, --rounds >= 1 required")
    if not 1 <= args.history_frames <= HISTORY:
        ap.error(f"--history-frames must be in 1..{HISTORY}")
    if not np.isfinite(args.cmd_filter_tau) or args.cmd_filter_tau < 0:
        ap.error("--cmd-filter-tau must be nonnegative and finite")
    if (not np.isfinite(args.steady_tail_weight) or args.steady_tail_weight < 0 or
            not np.isfinite(args.steady_after_s) or
            not 0 <= args.steady_after_s < 2.0):
        ap.error("steady-tail-weight must be nonnegative; steady-after-s in [0,2)")
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    model = PPO.load(args.model, device="cpu")
    with open(args.vecnorm, "rb") as f:
        norm = pickle.load(f)
    env = RoverEnv(task="cmdvel", action_mode="body", velocity_obs=args.velocity_obs,
                   difficulty=args.difficulty, episode_s=8, randomize=True,
                   arena_eval=False, history_frames=args.history_frames,
                   privileged=args.privileged,
                   cmd_filter_tau=args.cmd_filter_tau)
    assert model.action_space.shape == env.action_space.shape
    assert model.observation_space.shape == env.observation_space.shape
    lim = np.array([env.cfg.d_vx_max, env.cfg.d_vy_max, env.cfg.d_om_max])
    observations, labels, weights, episode_ids = [], [], [], []
    rng = np.random.default_rng(0)
    # Neighboring stacked observations share almost every frame. Split entire
    # episodes so validation does not reuse the same trajectory as training.
    split_episode = max(1, int(args.episodes * 0.8))
    params = list(model.policy.mlp_extractor.policy_net.parameters()) + \
             list(model.policy.action_net.parameters())
    optimizer = torch.optim.Adam(params, lr=3e-4)
    for round_idx in range(args.rounds):
        for episode_id, seed in enumerate(range(args.seed0, args.seed0 + args.episodes)):
            obs, _ = env.reset(seed=seed)
            while True:
                err = env.cmd_nom - env.body_twist()
                target = np.clip(args.gain * err / lim, -1, 1).astype(np.float32)
                obs_n = np.clip((obs - norm.obs_rms.mean) /
                                np.sqrt(norm.obs_rms.var + 1e-8),
                                -10, 10).astype(np.float32)
                observations.append(obs_n)
                labels.append(target)
                phase = ((env._cmd_k - 1) % env.cmd_hold) / CTRL_HZ
                v_err_n = float(np.linalg.norm(err[:2]) / max(env.v_max, 1e-9))
                weight = (1.0 + args.steady_tail_weight *
                          min(v_err_n / 0.16, 2.0)
                          if phase >= args.steady_after_s else 1.0)
                weights.append(weight)
                episode_ids.append(episode_id)
                # Later rounds label the states reached by the learner itself.
                # This matters because tiny imitation errors compound in closed loop.
                action = (target if round_idx == 0 else
                          model.predict(obs_n, deterministic=True)[0])
                obs, _, term, trunc, _ = env.step(action)
                if term or trunc:
                    break
            print(f"round {round_idx + 1} collected seed {seed}", flush=True)
        x = torch.from_numpy(np.asarray(observations))
        y = torch.from_numpy(np.asarray(labels))
        w = torch.from_numpy(np.asarray(weights, dtype=np.float32))
        ep_ids = np.asarray(episode_ids)
        train_idx = np.flatnonzero(ep_ids < split_episode)
        test_idx = np.flatnonzero(ep_ids >= split_episode)
        for epoch in range(args.epochs):
            batch_order = rng.permutation(train_idx)
            for start in range(0, len(batch_order), 512):
                idx = batch_order[start:start + 512]
                pred = model.policy._predict(x[idx], deterministic=True)
                per_sample = torch.mean((pred - y[idx]) ** 2, dim=1)
                loss = torch.mean(w[idx] * per_sample) / torch.mean(w[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                val = torch.mean((model.policy._predict(x[test_idx],
                                                        deterministic=True) - y[test_idx]) ** 2)
            print(f"round {round_idx + 1} epoch {epoch + 1}: "
                  f"val_mse={val.item():.6f}", flush=True)
        model.save(out / f"round_{round_idx + 1}")
    env.close()
    model.save(out / "model")
    shutil.copy2(args.vecnorm, out / "vecnorm.pkl")
    source = out / "src"
    source.mkdir(exist_ok=True)
    for name in ("behavior_clone_cmdvel.py", "env.py", "config.py",
                 "reward.py", "terramech.py"):
        shutil.copy2(pathlib.Path(__file__).with_name(name), source / name)
    (out / "manifest.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    print(f"saved {out}  samples={len(observations)}")


if __name__ == "__main__":
    main()
