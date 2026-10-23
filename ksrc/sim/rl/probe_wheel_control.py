#!/usr/bin/env python3
"""Test whether individual wheel-speed residuals help beyond body feedback.

This is a simulator-truth control probe, not a trained or deployable policy.
Every arm uses the same 7D action interface so that only the action differs.
"""
import argparse

import numpy as np

from env import CTRL_HZ, RoverEnv


def run_episode(env, seed, kind, body_gain, wheel_gain, slip_target, wheel_mode):
    env.reset(seed=seed)
    commands = []
    slip_asym = []
    traction_slip = []
    wheel_active = []
    wheel_common = []
    wheel_differential = []
    trace = []
    while True:
        action = np.zeros(7, dtype=np.float32)
        if kind in ("body", "body+wheel"):
            limits = np.array([env.cfg.d_vx_max, env.cfg.d_vy_max,
                               env.cfg.d_om_max])
            error = (env.cmd_nom - env.body_twist()) / limits
            action[:3] = np.clip(body_gain * error, -1.0, 1.0)
        signed_slip = env.tm.slip * np.sign(env.wheel_ref_applied)
        spinning = ((np.abs(env.wheel_ref_applied) > 0.02 * env.v_max) &
                    (env.tm.Fn > 1e-6))
        phase = ((env._cmd_k - 1) % env.cmd_hold) / CTRL_HZ
        # Reversing wheels have opposite raw slip signs even when traction is
        # identical. Compare drive-aligned slip only on loaded, driven wheels.
        pre_asym = (float(np.std(signed_slip[spinning]))
                    if np.count_nonzero(spinning) >= 3 else 0.0)
        traction_slip.append(float(np.mean(signed_slip[spinning]))
                             if np.any(spinning) else 0.0)
        wheel_active.append(float(np.mean(spinning)))
        if kind in ("wheel", "body+wheel"):
            # Positive traction slip means the wheel surface is overrunning the
            # ground, regardless of forward/reverse command sign. A dragging
            # or braking wheel must not be slowed further.
            if wheel_mode == "slow":
                wheel_error = np.maximum(signed_slip - slip_target, 0.0)
            elif wheel_mode == "target":
                wheel_error = signed_slip - slip_target
            elif wheel_mode == "balance":
                wheel_error = signed_slip - np.median(signed_slip[spinning]) \
                    if np.any(spinning) else np.zeros(4)
            elif wheel_mode == "common":
                wheel_error = (np.mean(signed_slip[spinning]) - slip_target) \
                    if np.any(spinning) else 0.0
            else:  # differential: strictly zero-mean wheel correction
                wheel_error = signed_slip - np.mean(signed_slip[spinning]) \
                    if np.any(spinning) else np.zeros(4)
            action[3:] = np.where(
                spinning,
                np.clip(-wheel_gain * wheel_error, -1.0, 1.0),
                0.0)
        wheel_common.append(float(np.mean(action[3:])))
        wheel_differential.append(float(np.std(action[3:])))
        commands.append(env.cmd_nom.copy())
        _, _, term, trunc, info = env.step(action)
        trace.append((phase, pre_asym, *env.vel_err_hist[-1]))
        slip_asym.append(float(np.std(env.tm.slip)))
        if term or trunc:
            break
    info["slip_asym_m"] = float(np.mean(slip_asym))
    info["traction_slip_m"] = float(np.mean(traction_slip)) if traction_slip else 0.0
    info["wheel_active"] = float(np.mean(wheel_active)) if wheel_active else 0.0
    info["wheel_common"] = float(np.mean(wheel_common))
    info["wheel_differential"] = float(np.mean(wheel_differential))
    return info, np.asarray(commands), np.asarray(trace)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--seed0", type=int, default=23000)
    p.add_argument("--episode-s", type=float, default=8.0)
    p.add_argument("--difficulty", type=float, default=0.8)
    p.add_argument("--soil-amp", type=float, default=0.25)
    p.add_argument("--cmd-filter-tau", type=float, default=0.10)
    p.add_argument("--body-gain", type=float, default=0.30)
    p.add_argument("--wheel-gain", type=float, default=2.0)
    p.add_argument("--slip-target", type=float, default=0.30)
    p.add_argument("--wheel-mode", choices=("slow", "target", "balance",
                                            "common", "differential"),
                   default="slow")
    args = p.parse_args()
    if args.episodes < 1 or args.episode_s <= 0:
        p.error("episodes and episode-s must be positive")
    if min(args.soil_amp, args.body_gain, args.wheel_gain, args.slip_target) < 0:
        p.error("gains, soil-amp and slip-target must be nonnegative")
    names = ("IK", "body", "wheel", "body+wheel")
    kwargs = dict(task="cmdvel", action_mode="body+wheel", randomize=True,
                  arena_eval=False, episode_s=args.episode_s,
                  difficulty=args.difficulty, soil_amp=args.soil_amp,
                  cmd_filter_tau=args.cmd_filter_tau)
    envs = {name: RoverEnv(**kwargs) for name in names}
    records = {name: [] for name in names}
    traces = {name: [] for name in names}
    mismatches = []
    try:
        for seed in range(args.seed0, args.seed0 + args.episodes):
            cmds = {}
            for name in names:
                info, commands, trace = run_episode(
                    envs[name], seed, "none" if name == "IK" else name,
                    args.body_gain, args.wheel_gain, args.slip_target,
                    args.wheel_mode)
                records[name].append(info)
                traces[name].append(trace)
                cmds[name] = commands
            for name in names[1:]:
                n = min(len(cmds["IK"]), len(cmds[name]))
                mismatches.append(float(np.max(np.abs(cmds["IK"][:n] -
                                                      cmds[name][:n]))))
            print(f"seed {seed} complete", flush=True)
    finally:
        for env in envs.values():
            env.close()

    print(f"command mismatch max={max(mismatches):.6f} m/s-or-rad/s "
          "(nonzero weakens exact pairing)")
    fields = (("v_err_m", 100), ("v_err_p90", 100),
              ("om_err_m", 100), ("om_err_p90", 100),
              ("slip_m", 1), ("slip_p90", 1),
              ("sink_m", 100), ("slip_asym_m", 1),
              ("traction_slip_m", 1), ("wheel_active", 100),
              ("wheel_common", 100), ("wheel_differential", 100),
              ("stuck", 1), ("tip", 1), ("oob", 1),
              ("wheel_use", 100), ("wheel_desat", 100))
    print("metric                   IK      body     wheel    body+wheel")
    for field, scale in fields:
        values = [np.mean([r[field] for r in records[name]]) * scale
                  for name in names]
        print(f"{field:<20}" + "".join(f" {v:9.3f}" for v in values))
    for field in ("v_err_m", "v_err_p90", "om_err_m", "om_err_p90"):
        b = np.array([r[field] for r in records["body"]])
        w = np.array([r[field] for r in records["body+wheel"]])
        delta = (w - b) * 100
        print(f"body+wheel minus body {field}: mean {delta.mean():+.3f} %max, "
              f"better {np.count_nonzero(delta < 0)}/{len(delta)} episodes, "
              f"worst loss {delta.max():+.3f} %max")
    # These masks are defined only by the IK trajectory, never by the policy
    # outcome. This avoids selecting the "recoverable" states after seeing RL.
    for label, mask_fn in (
            ("transition <0.75s", lambda x: x[:, 0] < 0.75),
            ("steady >=0.75s", lambda x: x[:, 0] >= 0.75),
            ("high slip asym >0.15", lambda x: x[:, 1] > 0.15),
            ("steady+high asym", lambda x: (x[:, 0] >= 0.75) & (x[:, 1] > 0.15))):
        differences = []
        for ik, body, both in zip(traces["IK"], traces["body"],
                                  traces["body+wheel"]):
            n = min(len(ik), len(body), len(both))
            mask = mask_fn(ik[:n])
            differences.extend((both[:n, 2:4] - body[:n, 2:4])[mask])
        d = np.asarray(differences).reshape(-1, 2)
        if len(d):
            print(f"{label}: n={len(d)} steps; body+wheel-body "
                  f"v={np.mean(d[:, 0])*100:+.3f}, "
                  f"om={np.mean(d[:, 1])*100:+.3f} %max; "
                  f"v better={np.mean(d[:, 0] < 0)*100:.1f}%")


if __name__ == "__main__":
    main()
