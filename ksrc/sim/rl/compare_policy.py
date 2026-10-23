#!/usr/bin/env python3
"""Compare a saved residual policy with pure IK on paired episodes.

Default path task: unseen fixed paths on the competition terrain.
cmdvel task: randomized body-velocity commands on procedural sand by default.
"""
import argparse
import math
import pickle

import numpy as np
from stable_baselines3 import PPO

from config import RoverCfg
from env import CTRL_HZ, HISTORY, OBS_PER_FRAME, PRIV_DIM, RoverEnv
from reward import VelRewardCfg
from train import stage1_paths


# cmdvel 과제의 지표. v_err/om_err 는 v_max/om_max 로 나눈 무차원 오차다.
# 100 을 곱한 값은 명령 크기 대비 비율이 아니라 최대속도 대비 백분율이다.
METRICS_CMDVEL = (
    ("r_step_m", "보상/스텝", 1.0, ""),
    ("v_err_m", "병진속도오차 평균", 100.0, "%vmax"),
    ("v_err_p90", "병진속도오차 p90", 100.0, "%vmax"),
    ("om_err_m", "요속도오차 평균", 100.0, "%ommax"),
    ("om_err_p90", "요속도오차 p90", 100.0, "%ommax"),
    ("slip_m", "슬립 평균", 1.0, ""),
    ("slip_p90", "슬립 p90", 1.0, ""),
    ("sink_m", "침하 평균", 100.0, "%zmax"),
    ("stuck", "고착", 100.0, "%"),
    ("tip", "전복", 100.0, "%"),
    ("oob", "경계 이탈", 100.0, "%"),
    ("a_use", "잔차 사용률", 100.0, "%"),
    ("a_p95", "잔차 크기 p95", 100.0, "%"),
    ("wheel_use", "바퀴 잔차 사용률", 100.0, "%"),
    ("cmd_sat", "차체명령 포화", 100.0, "%"),
    ("t_elapsed", "에피소드 길이", 1.0, "s"),
)
METRICS_CMDVEL_PUSH = (
    ("push_v_err_m", "외력 후 병진오차", 100.0, "%vmax"),
    ("push_om_err_m", "외력 후 요오차", 100.0, "%ommax"),
)

METRICS = (
    ("success", "완주율", 100.0, "%p"),
    ("frac", "경로 진행률", 100.0, "%p"),
    ("e_y_m", "횡오차 평균", 1000.0, "mm"),
    ("e_y_p90", "횡오차 p90", 1000.0, "mm"),
    ("e_psi_m", "기수오차 평균", 1.0, "deg"),
    ("slip_m", "슬립 평균", 1.0, ""),
    ("slip_p90", "슬립 p90", 1.0, ""),
    # 침하는 슬립의 대가다 (terramech: zdot = alpha|i||v| - beta z,
    # Rn = Fn(c_r + k_z z), k_z 8.0).  2026-10-04 까지 비교표에 없어서
    # 슬립 +26% 를 쓰는 정책의 비용이 측정되지 않았다.
    ("sink_m", "침하 평균", 1000.0, "mm"),
    ("t_elapsed", "소요시간", 1.0, "s"),
    ("a_use", "잔차 사용률", 100.0, "%"),
    ("a_p95", "잔차 크기 p95", 100.0, "%"),
    ("cmd_sat", "차체명령 포화", 100.0, "%"),
    ("drive_gate", "조향 대기 비율", 100.0, "%"),
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--vecnorm", required=True)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--terrain", choices=("sand", "rock"), default="sand")
    p.add_argument("--path-seed", type=int, default=4321,
                   help="holdout path seed; training default is 1234")
    p.add_argument("--n-paths", type=int, default=30)
    p.add_argument("--seed0", type=int, default=1000)
    p.add_argument("--episode-s", type=float, default=20.0)
    p.add_argument("--difficulty", type=float, default=0.8)
    p.add_argument("--drive-align-gate-deg", type=float, default=10.0,
                   help="학습/ROS와 같은 조향 게이트 [deg], 0=이전 동작")
    p.add_argument("--min-slope-deg", type=float, default=0.0,
                   help="경로가 지나는 평균 지형경사의 하한 [deg]. 0 이면 기존 동작(경사 무시). 2026-10-04 측정: 하한이 없으면 경로 경사 중앙값이 0도라 견인력 수요가 R/N=0.057 뿐이고, 바퀴별 mu 를 +-16%% 흔들어도 목적 현상이 발생하지 않는다. 8 이면 실제 평균 9.5도(수요 0.165)로 가용 mu(i) 0.17~0.39 와 같은 범위가 된다")
    p.add_argument("--path-d-ep", type=float, default=0.0,
                   help="make_path 의 급사면 선호 확률. min-slope-deg 와 함께 쓴다 (0.9 권장)")
    p.add_argument("--soil-amp", type=float, default=None,
                   help="모래 비대칭 장(바퀴별 mu 배율) 진폭. **이 RL 의 과제 본체**다 (terrain.soil_field 참고). --stage1 은 randomize 를 통째로 꺼서 이것까지 끄므로, Stage 1 에서 목적 현상을 쓰려면 이 값을 명시한다. 미지정이면 기존 동작(randomize 에 종속). 0.25 권장, 0.5 는 장이 음수가 된다")
    p.add_argument("--randomize", action="store_true",
                   help="enable motor/sensor/soil randomization and noise")
    # 기준선과 정책에 **같은** lookahead 를 적용한다.  정책은 cmd_nom 위에
    # 잔차를 얹으므로 lookahead 가 바뀌면 정책이 보는 명령도 바뀐다 -- 즉
    # 이것은 "튜닝된 기준선 위에서도 RL 이 뭘 더하는가" 를 묻는 노브다.
    # 2026-10-04 스윕: 순수 IK 가 L=0.264 에서 20.72mm, L=0.12 에서 14.81mm.
    # L 하나가 5.91mm 를 움직이는데 RL 이 주장한 개선은 1.88mm 였다.
    p.add_argument("--task", default="path", choices=("path", "cmdvel"),
                   help="학습 때와 **같은** 값이어야 한다. cmdvel 은 경로 지표 "
                        "대신 속도추종 지표를 낸다.")
    p.add_argument("--action-mode", default="body",
                   choices=("body", "wheel", "body+wheel"),
                   help="학습한 정책과 같은 잔차 모드")
    p.add_argument("--velocity-obs", action="store_true",
                   help="학습 시 --velocity-obs 를 썼을 때만. MuJoCo 참값 진단")
    p.add_argument("--push-force", type=float, default=0.0,
                   help="에피소드마다 0.3초간 가하는 무작위 평면 외력 상한 [N]")
    p.add_argument("--cmd-filter-tau", type=float, default=0.28,
                   help="cmdvel 목표속도 필터 [s]; 0.28=기존 학습, 0.10=ROS 기본값")
    p.add_argument("--w-tail", type=float, default=0.0,
                   help="cmdvel 평가 보상에 쓰는 큰 오차 이차 비용 계수")
    p.add_argument("--history-frames", type=int, default=HISTORY,
                   help="학습 시 사용한 관측 이력 프레임 수")
    p.add_argument("--oracle-gain", type=float, default=None,
                   help="진단 전용: 모델 대신 MuJoCo 참속도 오차에 이득을 곱한 잔차")
    p.add_argument("--oracle-ki", type=float, default=0.0,
                   help="진단 전용: 참속도 오차의 누적 보정 이득 [1/s]; 기본 0=P만")
    p.add_argument("--oracle-i-tau", type=float, default=1.0,
                   help="진단 전용: 누적 보정의 누설 시정수 [s]; 명령 전환 시 windup 억제")
    p.add_argument("--assist-cap", type=float, default=1.0,
                   help="평가 시 보조 행동 절댓값 상한(0..1), 기본 1=학습 때 상한")
    p.add_argument("--assist-delay-s", type=float, default=0.0,
                   help="cmdvel 진단: 명령 목표가 바뀐 직후 이 시간 동안 잔차를 0으로 둠")
    p.add_argument("--priv-ablate", choices=("none", "mean"), default="none",
                   help="teacher 진단: 정규화된 특권값을 학습 평균으로 대체")
    p.add_argument("--eval-geometry", choices=("arena", "proc"), default=None,
                   help="cmdvel 기본은 미학습 절차지형. arena 는 실제 규사맵 크기 그대로")
    p.add_argument("--d-rate-max", type=float, default=None,
                   help="잔차 변화율 제한 [스텝당 |da|]. 속도 액션은 급격한 변화를 요구할 수 있어 실기에서 깨진 사례가 있다 (arXiv:2504.02420). 0.25 = 80ms 에 -1 -> +1. 미지정이면 제한 없음(기존 동작)")
    p.add_argument("--pp-k-v", type=float, default=None,
                   help="pure pursuit lookahead 속도계수 (기본 1.2)")
    p.add_argument("--pp-l-min", type=float, default=None,
                   help="lookahead 하한 [m] (기본 0.18)")
    p.add_argument("--pp-l-max", type=float, default=None,
                   help="lookahead 상한 [m] (기본 0.45)")
    args = p.parse_args()
    if args.episodes < 1 or args.n_paths < 1:
        p.error("episodes and n-paths must be positive")
    if not 1 <= args.history_frames <= HISTORY:
        p.error(f"--history-frames 는 1..{HISTORY} 이어야 합니다")
    if not math.isfinite(args.push_force) or args.push_force < 0:
        p.error("--push-force 는 0 이상의 유한한 숫자여야 합니다")
    if not math.isfinite(args.cmd_filter_tau) or args.cmd_filter_tau < 0:
        p.error("--cmd-filter-tau 는 0 이상의 유한한 숫자여야 합니다")
    if not math.isfinite(args.w_tail) or args.w_tail < 0:
        p.error("--w-tail 은 0 이상의 유한한 숫자여야 합니다")
    if args.task != "cmdvel" and args.w_tail != 0:
        p.error("--w-tail 은 cmdvel 과제에서만 사용합니다")
    if args.oracle_gain is not None and (
            args.task != "cmdvel" or args.action_mode != "body" or
            not math.isfinite(args.oracle_gain) or args.oracle_gain < 0):
        p.error("--oracle-gain 은 cmdvel/body 에서 0 이상의 유한한 값이어야 합니다")
    if (not math.isfinite(args.oracle_ki) or args.oracle_ki < 0 or
            (args.oracle_ki > 0 and args.oracle_gain is None)):
        p.error("--oracle-ki 는 --oracle-gain 과 함께 쓰는 0 이상의 유한한 값이어야 합니다")
    if not math.isfinite(args.oracle_i_tau) or args.oracle_i_tau <= 0:
        p.error("--oracle-i-tau 는 양의 유한한 초 단위여야 합니다")
    if not math.isfinite(args.assist_cap) or not 0 <= args.assist_cap <= 1:
        p.error("--assist-cap 은 0..1 이어야 합니다")
    if (not math.isfinite(args.assist_delay_s) or
            not 0 <= args.assist_delay_s < 2.0 or
            (args.assist_delay_s > 0 and args.task != "cmdvel")):
        p.error("--assist-delay-s 는 cmdvel 에서 0..2초 미만이어야 합니다")
    if not math.isfinite(args.drive_align_gate_deg) or args.drive_align_gate_deg < 0:
        p.error("--drive-align-gate-deg 는 0 이상의 유한한 숫자여야 합니다")

    model = PPO.load(args.model, device="cpu")
    n_action = {"body": 3, "wheel": 4, "body+wheel": 7}[args.action_mode]
    if model.action_space.shape != (n_action,):
        p.error(f"{args.action_mode} 정책은 {n_action}D 여야 합니다: "
                f"현재 {model.action_space.shape}")
    obs_dim = int(model.observation_space.shape[0])
    regular_dim = ((OBS_PER_FRAME - 3 + n_action + 3 * args.velocity_obs) *
                   args.history_frames)
    if obs_dim < regular_dim:
        p.error(f"model observation has {obs_dim} values; current env needs {regular_dim}")
    if obs_dim not in (regular_dim, regular_dim + PRIV_DIM):
        p.error("관측 차원은 고유수용감각 또는 고유수용감각+43D 특권정보여야 합니다")
    if args.priv_ablate != "none" and obs_dim == regular_dim:
        p.error("--priv-ablate 는 특권 관측 teacher 모델에서만 사용합니다")
    with open(args.vecnorm, "rb") as f:
        norm = pickle.load(f)
    if norm.obs_rms.mean.shape != (obs_dim,):
        p.error("VecNormalize statistics do not match the model observation")

    over = {}
    for flag in ("pp_k_v", "pp_l_min", "pp_l_max"):
        v = getattr(args, flag)
        if v is not None:
            if not math.isfinite(v) or v <= 0:
                p.error(f"--{flag.replace('_', '-')} 는 양의 유한한 값이어야 합니다")
            over[flag] = float(v)
    if args.d_rate_max is not None:
        if not math.isfinite(args.d_rate_max) or not 0 < args.d_rate_max <= 2.0:
            p.error("--d-rate-max 는 0 초과 2.0 이하여야 합니다 (액션 범위가 [-1,1])")
        over["d_rate_max"] = float(args.d_rate_max)
    _base = RoverCfg()
    if over.get("pp_l_min", _base.pp_l_min) > over.get("pp_l_max", _base.pp_l_max):
        p.error("--pp-l-min 이 --pp-l-max 보다 큽니다")
    cfg = RoverCfg(**over) if over else None

    eval_geometry = args.eval_geometry or ("proc" if args.task == "cmdvel" else "arena")
    paths = (stage1_paths(args.terrain, args.path_seed, args.n_paths,
                          min_slope_deg=args.min_slope_deg, d_ep=args.path_d_ep)
             if eval_geometry == "arena" else None)
    env_args = dict(cfg=cfg, task=args.task, action_mode=args.action_mode,
                    vel_rew=VelRewardCfg(w_tail=args.w_tail),
                    velocity_obs=args.velocity_obs,
                    history_frames=args.history_frames,
                    difficulty=args.difficulty, episode_s=args.episode_s,
                    arena_eval=eval_geometry == "arena", eval_kind=args.terrain, paths=paths,
                    randomize=args.randomize, privileged=obs_dim > regular_dim,
                    drive_align_gate_deg=args.drive_align_gate_deg,
                    soil_amp=args.soil_amp)
    env_args["cmd_filter_tau"] = args.cmd_filter_tau
    env_args["push_force_max"] = args.push_force
    base_env = RoverEnv(**env_args, seed=args.seed0)
    rl_env = RoverEnv(**env_args, seed=args.seed0)
    if base_env.observation_space.shape != (obs_dim,):
        p.error(f"model observation has {obs_dim} values but current env has "
                f"{base_env.observation_space.shape[0]}")

    def episode(env, seed, assist):
        obs, _ = env.reset(seed=seed)
        done = False
        err_i = np.zeros(3)
        while not done:
            if assist:
                if args.oracle_gain is not None:
                    lim = np.array([env.cfg.d_vx_max, env.cfg.d_vy_max,
                                    env.cfg.d_om_max])
                    err_n = (env.cmd_nom - env.body_twist()) / lim
                    err_i = (math.exp(-1.0 / (CTRL_HZ * args.oracle_i_tau)) *
                             err_i + err_n / CTRL_HZ)
                    # 누설과 상태 상한은 지속 슬립에서 누적기가 한계로 달려
                    # 명령 전환 후 과보정하는 것을 막는다.
                    err_i = np.clip(err_i, -args.oracle_i_tau, args.oracle_i_tau)
                    action = np.clip(args.oracle_gain * err_n +
                                     args.oracle_ki * err_i, -1.0, 1.0).astype(np.float32)
                else:
                    normalized = np.clip(
                        (obs - norm.obs_rms.mean) / np.sqrt(norm.obs_rms.var + 1e-8),
                        -10, 10).astype(np.float32)
                    if args.priv_ablate == "mean":
                        normalized[regular_dim:] = 0.0
                    action = model.predict(normalized, deterministic=True)[0]
                action = np.clip(action, -args.assist_cap, args.assist_cap)
                if ((env._cmd_k - 1) % env.cmd_hold <
                        args.assist_delay_s * CTRL_HZ):
                    action = np.zeros_like(action)
            else:
                action = np.zeros(n_action, dtype=np.float32)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        info["r_step_m"] = (sum(v for k, v in info.items() if k.startswith("rterm_")) /
                            max(info["t_elapsed"] * 50.0, 1.0))
        return info

    base, rl = [], []
    # soil_amp 미지정이면 env 가 randomize 에 종속시켜 cfg.soil_amp 를 쓴다.
    # 헤더에 유효값을 찍지 않으면 soil 0.25 와 0 을 구분할 수 없다 (baseline.py
    # 의 라벨 버그와 같은 뿌리).
    _soil_eff = RoverCfg().soil_amp if args.randomize else 0.0
    print(f"{args.terrain} | geometry={eval_geometry} | "
          f"holdout paths seed={args.path_seed if paths is not None else 'none'} | "
          f"randomize={args.randomize} | episodes={args.episodes} | "
          f"drive_align_gate_deg={args.drive_align_gate_deg} | "
          f"soil_amp={args.soil_amp} push_force_max={args.push_force}N"
          f"{'' if args.soil_amp is not None else f' -> 유효 {_soil_eff}'} | "
          f"w_tail={args.w_tail} | "
          f"cmd_filter_tau={args.cmd_filter_tau} | "
          f"assist_cap={args.assist_cap} | "
          f"assist_delay_s={args.assist_delay_s} | "
          f"priv_ablate={args.priv_ablate} | "
          f"lookahead k_v={over.get('pp_k_v', _base.pp_k_v)} "
          f"L={over.get('pp_l_min', _base.pp_l_min):.2f}"
          f"~{over.get('pp_l_max', _base.pp_l_max):.2f}m",
          flush=True)
    for i in range(args.episodes):
        seed = args.seed0 + i
        base.append(episode(base_env, seed, False))
        rl.append(episode(rl_env, seed, True))
        if (i + 1) % 10 == 0 or i + 1 == args.episodes:
            print(f"{i + 1}/{args.episodes} pairs complete", flush=True)

    print(f"{'지표':<17} {'순수 IK':>10} "
          f"{'참값 oracle' if args.oracle_gain is not None else '보조 정책':>10} "
          f"{'차이(RL-IK)':>14} {'95% CI (짝 bootstrap)':>24}")
    rng = np.random.default_rng(0)
    resamples = rng.integers(0, args.episodes, size=(10000, args.episodes))
    metric_list = (METRICS_CMDVEL + (METRICS_CMDVEL_PUSH if args.push_force > 0 else ())
                   if args.task == "cmdvel" else METRICS)
    for key, label, scale, unit in metric_list:
        b = np.array([v[key] for v in base], dtype=float) * scale
        r = np.array([v[key] for v in rl], dtype=float) * scale
        valid = np.isfinite(b) & np.isfinite(r)
        if not np.all(valid):
            # 외력이 끝나기 전에 실패 종료하면 '외력 후 1초'가 존재하지 않는다.
            # NaN 을 bootstrap 에 넣으면 표 전체가 NaN 이 된다. 유효 짝 수를
            # 명시하고 실패율은 별도 행으로 함께 보여 준다.
            if not valid.any():
                print(f"{label:<17} {'N/A':>10} {'N/A':>10} "
                      f"{'N/A':>13} {'N/A':>24} {unit} (유효 0/{args.episodes})")
                continue
            b, r = b[valid], r[valid]
            sample_idx = rng.integers(0, len(b), size=(10000, len(b)))
            count_note = f" (유효 {len(b)}/{args.episodes})"
        else:
            sample_idx = resamples
            count_note = ""
        delta = r - b
        lo, hi = np.percentile(delta[sample_idx].mean(axis=1), (2.5, 97.5))
        print(f"{label:<17} {np.mean(b):10.2f} {np.mean(r):10.2f} "
              f"{np.mean(delta):+13.2f} [{lo:+.2f}, {hi:+.2f}] {unit}{count_note}")
    if args.task == "path":
        wins = sum(not b["success"] and r["success"] for b, r in zip(base, rl))
        losses = sum(b["success"] and not r["success"] for b, r in zip(base, rl))
        discordant = wins + losses
        p_exact = min(1.0, 2.0 * sum(math.comb(discordant, k)
                                   for k in range(min(wins, losses) + 1))
                      / (2 ** discordant)) if discordant else 1.0
        print(f"성공 짝비교 RL만/IK만: {wins}/{losses}; "
              f"McNemar exact p={p_exact:.3f}")
    print("같은 seed와 경로를 짝지었지만, 주행 결과가 갈라지면 이후 노이즈의 "
          "난수 소비는 완전히 같지 않을 수 있습니다.")


if __name__ == "__main__":
    main()
