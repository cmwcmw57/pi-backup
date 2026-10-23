#!/usr/bin/env python3
"""Branch the exact same sand state to measure wheel-residual control authority.

This is a hindsight diagnostic, not a trained controller or deployment result.
The 3D body-truth P controller is identical in every branch. Only the four
wheel-speed multipliers change for a short fixed horizon. Each branch starts
from a deepcopy of the same MuJoCo model/data, soil memory, IK fold state,
command schedule, actuator delay, and RNG state.
"""
import argparse
import copy
import json
import pickle

import numpy as np

from env import CTRL_HZ, RoverEnv


def body_p(env, gain=0.30):
    limits = np.array([env.cfg.d_vx_max, env.cfg.d_vy_max, env.cfg.d_om_max])
    return np.clip(gain * (env.cmd_nom - env.body_twist()) / limits, -1, 1)


def traction_state(env):
    valid = ((np.abs(env.wheel_ref_applied) > 0.02 * env.v_max) &
             (env.tm.Fn > 1e-6))
    traction = env.tm.slip * np.sign(env.wheel_ref_applied)
    spread = float(np.std(traction[valid])) if np.count_nonzero(valid) >= 3 else 0.0
    return valid, traction, spread


def _zero_mean(x, valid):
    x = np.where(valid, x, 0.0)
    x[valid] -= np.mean(x[valid])
    largest = np.max(np.abs(x))
    return x / largest if largest > 1e-12 else x


def candidates(env, valid, traction):
    bank = {"body_only": np.zeros(4)}
    # Differential actions preserve the common IK wheel command to first order.
    for i in range(4):
        if valid[i]:
            for sign in (-1, 1):
                a = np.zeros(4)
                a[valid] = -sign / (np.count_nonzero(valid) - 1)
                a[i] = sign
                bank[f"wheel_{i}_{sign:+d}"] = a
    for i in range(4):
        for j in range(i + 1, 4):
            if valid[i] and valid[j]:
                a = np.zeros(4)
                a[i], a[j] = 1.0, -1.0
                bank[f"pair_{i}_{j}"] = a
                bank[f"pair_{j}_{i}"] = -a
    slip = _zero_mean(traction.copy(), valid)
    mu = _zero_mean(env.tm.mu_w.copy(), valid)
    bank["slow_high_slip"] = -slip
    bank["boost_high_slip"] = slip
    bank["boost_high_mu"] = mu
    bank["boost_low_mu"] = -mu
    # Common-mode changes are a separate comparator, not wheel selectivity.
    bank["all_boost"] = np.ones(4)
    bank["all_slow"] = -np.ones(4)
    return bank


def policy_wheel(env, policy):
    model, rms, history, gate, before_s = policy
    obs = np.concatenate([*env.hist[-history:], env._privileged()]).astype(np.float32)
    obs_n = np.clip((obs-rms.mean) / np.sqrt(rms.var+1e-8), -10, 10).astype(np.float32)
    wheel = np.asarray(model.predict(obs_n, deterministic=True)[0], dtype=float)
    wheel -= np.mean(wheel)
    wheel /= max(1.0, float(np.max(np.abs(wheel))))
    phase = ((env._cmd_k - 1) % env.cmd_hold) / CTRL_HZ
    _, _, spread = traction_state(env)
    if phase >= before_s or spread <= gate:
        wheel[:] = 0.0
    return wheel


def rollout(snapshot, wheel, horizon, policy=None):
    env = copy.deepcopy(snapshot)
    points = []
    initial_cmd = env.cmd_tgt.copy()
    for _ in range(horizon):
        applied_wheel = policy_wheel(env, policy) if policy is not None else wheel
        action = np.concatenate([body_p(env), applied_wheel]).astype(np.float32)
        _, _, term, trunc, _ = env.step(action)
        v, om = env.vel_err_hist[-1]
        points.append((v, om, float(np.mean(env.tm.z)),
                       float(np.mean(np.abs(env.tm.slip)))))
        if term or trunc:
            break
    x = np.asarray(points)
    result = dict(v=float(x[:, 0].mean()), om=float(x[:, 1].mean()),
                  v_final=float(x[-1, 0]), om_final=float(x[-1, 1]),
                  sink=float(x[:, 2].mean()), slip=float(x[:, 3].mean()),
                  steps=len(x), command_changed=bool(np.any(env.cmd_tgt != initial_cmd)))
    env.close()
    return result


def summarize(rows):
    if not rows:
        return {}
    values = np.array([[r["diff_best_v"], r["diff_best_om"],
                        r["common_best_v"], r["common_best_om"],
                        r["slip_rule_v"], r["slip_rule_om"]] for r in rows])
    result = {
        "n_states": len(rows), "n_seeds": len(set(r["seed"] for r in rows)),
        "baseline_v_mean_percent_vmax": round(100 * np.mean([r["baseline"]["v"] for r in rows]), 3),
        "baseline_om_mean_percent_ommax": round(100 * np.mean([r["baseline"]["om"] for r in rows]), 3),
        "diff_best_v_delta_percent_vmax": round(100 * values[:, 0].mean(), 3),
        "diff_best_om_delta_percent_ommax": round(100 * values[:, 1].mean(), 3),
        "common_best_v_delta_percent_vmax": round(100 * values[:, 2].mean(), 3),
        "common_best_om_delta_percent_ommax": round(100 * values[:, 3].mean(), 3),
        "slip_rule_v_delta_percent_vmax": round(100 * values[:, 4].mean(), 3),
        "slip_rule_om_delta_percent_ommax": round(100 * values[:, 5].mean(), 3),
        "diff_v_improved_fraction": round(float(np.mean(values[:, 0] < -1e-5)), 3),
        "diff_v_and_om_improved_fraction": round(float(np.mean((values[:, 0] < -1e-5) &
                                                                (values[:, 1] < -1e-5))), 3),
    }
    if rows[0]["learned_v"] is not None:
        result["learned_v_delta_percent_vmax"] = round(
            100 * np.mean([r["learned_v"] for r in rows]), 3)
        result["learned_om_delta_percent_ommax"] = round(
            100 * np.mean([r["learned_om"] for r in rows]), 3)
        result["learned_both_improved_fraction"] = round(float(np.mean(
            [r["learned_v"] < 0 and r["learned_om"] < 0 for r in rows])), 3)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed0", type=int, default=41000)
    p.add_argument("--episode-s", type=float, default=10.0)
    p.add_argument("--soil-amp", type=float, default=1.0)
    p.add_argument("--difficulty", type=float, default=1.0)
    p.add_argument("--spread", type=float, default=0.12)
    p.add_argument("--horizon", type=int, default=15,
                   help="20 ms control steps per branch")
    p.add_argument("--per-seed", type=int, default=2)
    p.add_argument("--out", type=str, default="/tmp/ksrc_wheel_same_state.json")
    p.add_argument("--policy-model", type=str)
    p.add_argument("--policy-vecnorm", type=str)
    p.add_argument("--policy-history", type=int, default=5)
    p.add_argument("--policy-gate", type=float, default=0.05)
    p.add_argument("--policy-before-s", type=float, default=2.0)
    args = p.parse_args()
    if args.episodes < 1 or args.horizon < 1 or args.per_seed < 1:
        p.error("episodes, horizon and per-seed must be positive")
    if args.spread <= 0 or args.episode_s <= 0 or args.soil_amp < 0:
        p.error("spread and episode-s must be positive, soil-amp nonnegative")
    if (args.policy_model is None) != (args.policy_vecnorm is None):
        p.error("--policy-model and --policy-vecnorm must be used together")
    policy = None
    if args.policy_model is not None:
        from stable_baselines3 import PPO
        model = PPO.load(args.policy_model, device="cpu")
        with open(args.policy_vecnorm, "rb") as f:
            rms = pickle.load(f).obs_rms
        policy = (model, rms, args.policy_history, args.policy_gate,
                  args.policy_before_s)
    env = RoverEnv(task="cmdvel", action_mode="body+wheel", randomize=True,
                   arena_eval=False, episode_s=args.episode_s,
                   difficulty=args.difficulty, soil_amp=args.soil_amp,
                   cmd_filter_tau=0.10)
    rows = []
    total_eligible = 0
    try:
        for seed in range(args.seed0, args.seed0 + args.episodes):
            env.reset(seed=seed)
            selected = 0
            last_k = -100000
            while True:
                phase = ((env._cmd_k - 1) % env.cmd_hold) / CTRL_HZ
                valid, traction, spread = traction_state(env)
                eligible = (0.75 <= phase <= 1.50 and spread >= args.spread and
                            np.count_nonzero(valid) >= 3 and
                            not env.drive_alignment_pending and
                            env.k - last_k >= 0.75 * CTRL_HZ)
                if eligible:
                    total_eligible += 1
                    if selected < args.per_seed:
                        snapshot = copy.deepcopy(env)
                        bank = candidates(snapshot, valid, traction)
                        result = {name: rollout(snapshot, action, args.horizon)
                                  for name, action in bank.items()}
                        if policy is not None:
                            result["learned_policy"] = rollout(
                                snapshot, np.zeros(4), args.horizon, policy=policy)
                        ref = result["body_only"]
                        if any(x["steps"] != args.horizon or x["command_changed"]
                               for x in result.values()):
                            raise RuntimeError("probe ended or command switched within horizon")
                        differential = [name for name in bank if name not in
                                        ("body_only", "all_boost", "all_slow")]
                        best = min(differential, key=lambda n: result[n]["v"])
                        common = min(("all_boost", "all_slow"),
                                     key=lambda n: result[n]["v"])
                        slow = result["slow_high_slip"]
                        learned = result.get("learned_policy")
                        rows.append(dict(seed=seed, step=env.k, phase=round(phase, 3),
                                         spread=round(spread, 4),
                                         valid_wheels=int(np.count_nonzero(valid)),
                                         baseline=ref, best_diff=best, best_common=common,
                                         diff_best_v=result[best]["v"]-ref["v"],
                                         diff_best_om=result[best]["om"]-ref["om"],
                                         common_best_v=result[common]["v"]-ref["v"],
                                         common_best_om=result[common]["om"]-ref["om"],
                                         slip_rule_v=slow["v"]-ref["v"],
                                         slip_rule_om=slow["om"]-ref["om"],
                                         learned_v=(None if learned is None else
                                                    learned["v"]-ref["v"]),
                                         learned_om=(None if learned is None else
                                                     learned["om"]-ref["om"]),
                                         arms=result))
                        selected += 1
                        last_k = env.k
                action = np.concatenate([body_p(env), np.zeros(4)]).astype(np.float32)
                _, _, term, trunc, _ = env.step(action)
                if term or trunc:
                    break
            print(f"seed={seed} probes={selected}", flush=True)
    finally:
        env.close()
    out = dict(args=vars(args), eligible_steps=total_eligible,
               summary=summarize(rows), states=rows)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["summary"], indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
