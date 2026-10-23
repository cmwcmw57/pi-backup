#!/usr/bin/env python3
"""Measure how often sand creates distinct, loaded wheel traction states.

Uses the simulator-truth body P controller only. This is a distribution audit,
not evidence that a particular wheel intervention would improve tracking.
"""
import argparse

import numpy as np

from env import CTRL_HZ, RoverEnv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed0", type=int, default=36000)
    p.add_argument("--episode-s", type=float, default=20)
    p.add_argument("--difficulty", type=float, default=1.0)
    p.add_argument("--soil-amp", type=float, default=0.35)
    p.add_argument("--cmd-filter-tau", type=float, default=0.10)
    args = p.parse_args()
    env = RoverEnv(task="cmdvel", action_mode="body+wheel", randomize=True,
                   arena_eval=False, difficulty=args.difficulty,
                   episode_s=args.episode_s, soil_amp=args.soil_amp,
                   cmd_filter_tau=args.cmd_filter_tau)
    records = []
    try:
        for seed in range(args.seed0, args.seed0 + args.episodes):
            env.reset(seed=seed)
            while True:
                phase = ((env._cmd_k - 1) % env.cmd_hold) / CTRL_HZ
                raw_spread = float(np.std(env.tm.slip))
                valid = ((np.abs(env.wheel_ref_applied) > 0.02 * env.v_max) &
                         (env.tm.Fn > 1e-6))
                traction = env.tm.slip * np.sign(env.wheel_ref_applied)
                if np.count_nonzero(valid) >= 3:
                    active_slip = traction[valid]
                    spread = float(np.std(active_slip))
                    mu = env.tm.mu_w[valid]
                    mu_range = float((np.max(mu) - np.min(mu)) /
                                     max(np.mean(mu), 1e-9))
                    mixed = bool(np.any(active_slip > 0.30) and
                                 np.any(active_slip < 0.10))
                    high_asym = spread > 0.15
                    spin = bool(np.any(active_slip > 0.30))
                else:
                    spread = mu_range = 0.0
                    mixed = high_asym = spin = False
                limits = np.array([env.cfg.d_vx_max, env.cfg.d_vy_max,
                                   env.cfg.d_om_max])
                body = np.clip(0.3 * (env.cmd_nom - env.body_twist()) / limits,
                               -1.0, 1.0)
                action = np.concatenate([body, np.zeros(4)]).astype(np.float32)
                _, _, term, trunc, _ = env.step(action)
                v_err, om_err = env.vel_err_hist[-1]
                records.append((phase >= 0.75, np.count_nonzero(valid) >= 3,
                                spread, mu_range, mixed, high_asym, spin,
                                v_err, om_err, raw_spread))
                if term or trunc:
                    break
            print(f"seed {seed} complete", flush=True)
    finally:
        env.close()
    x = np.asarray(records)
    names = (("all", np.ones(len(x), dtype=bool)),
             ("transition", x[:, 0] == 0),
             ("steady", x[:, 0] == 1),
             ("loaded<3", x[:, 1] == 0),
             ("loaded>=3", x[:, 1] == 1))
    print("group         steps  loaded>=3  slip_sd raw_sd  mu_range  mixed_spin  "
          "asym>.05 asym>.15 raw>.15  any_spin  v_err%  om_err%")
    for name, mask in names:
        y = x[mask]
        print(f"{name:<12} {len(y):5d} {y[:,1].mean():10.3f} "
              f"{y[:,2].mean():8.3f} {y[:,9].mean():6.3f} {y[:,3].mean():9.3f} "
              f"{y[:,4].mean():11.3f} {np.mean(y[:,2] > 0.05):8.3f} "
              f"{y[:,5].mean():8.3f} "
              f"{np.mean(y[:,9] > 0.15):9.3f} {y[:,6].mean():9.3f} {y[:,7].mean()*100:7.2f} "
              f"{y[:,8].mean()*100:8.2f}")
    for label, mask in (("mixed-spin", x[:,4] > 0),
                        ("high-asym", x[:,5] > 0),
                        ("raw-high-but-not-traction-high", (x[:,9] > 0.15) &
                         (x[:,5] == 0)),
                        ("other-loaded", (x[:,1] > 0) & (x[:,5] == 0))):
        if np.any(mask):
            y = x[mask]
            print(f"{label}: {len(y)} steps, v_err={np.mean(y[:,7])*100:.2f}%vmax, "
                  f"om_err={np.mean(y[:,8])*100:.2f}%ommax")


if __name__ == "__main__":
    main()
