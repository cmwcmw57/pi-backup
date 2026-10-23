#!/usr/bin/env python3
"""고전 P 피드백 기준선: 바퀴별 **감속 전용** 비례제어를 순수 IK 와 짝비교한다.

왜 이 도구가 필요한가
---------------------
2026-10-07, arXiv:2609.38715 (Matsuoka/Takehana/Uno/Kuwahara/Yoshida, 도호쿠대).
4륜 독립구동 로버 EX1 로 좌우 그라우저를 15.0mm vs 5.4mm 로 달아 견인력
비대칭을 **물리적으로** 만들고, 바퀴별 속도를 깎아서 경로이탈을 잡은
하드웨어 실험이다.  제어기 전체가 이것뿐이다:

    v_cmd,L = max(v_min, v_ref - max(0, -u))
    v_cmd,R = max(v_min, v_ref - max(0, +u)),   u = Kp * e_psi
    Kp = 3.0 m/(s*rad),  v_min = 0.2 m/s

즉 **P 게인 하나**다.  우리 oracle 도 차체수준 PI 루프(K=0.6)로 속도오차
14.18% -> 12.65% (-11%) 를 뽑았다.  그러면 RL 이 이겨야 하는 상대는
"순수 IK" 가 아니라 **이 P 제어기**다.  순수 IK 를 상대로 재는 건
기준선을 한 번도 튜닝하지 않은 채 재는 것과 같은 실수다 (s13 철회의
직접적 원인이었다).

이 도구는 세 신호를 지원한다.
  yaw   요속도 오차.  **측정된 자이로**를 쓴다 (정책이 보는 것과 동일).
        cmdvel 과제의 자연스러운 신호.
  psi   기수 오차.  **측위 추정치** e_psi_est 를 쓴다.  Tohoku 논문의
        신호 그 자체이고, path 과제의 자연스러운 신호.
  slip  바퀴별 **참 슬립**.  실기에서 못 얻으므로 배포 불가 -- 이건
        "바퀴별 감속에 남은 여지의 상한" 을 재는 oracle 이다.

yaw/psi 는 좌우 **차동 감속**이다.  양쪽을 같이 깎으면 그냥 느려질 뿐
요를 못 만든다.  slip 은 바퀴별 독립 감속이다.

구조적 한계 (한계가 아니라 분업)
--------------------------------
감속만 하면 |v_ref| 가 커질 수 없으므로 **병진속도 부족은 고칠 수 없다**.
cmdvel 에서 v_err 은 거의 전부 부족(actual < cmd)이라, 이 제어기는
v_err 을 줄이는 게 아니라 **악화**시키는 쪽이 정상이다.  그 대가로
슬립·침하·요오차를 줄인다.  그래서 표를 한 줄만 보면 안 된다.

사용법:
    python3 pfb_baseline.py --task cmdvel --signal yaw  --gains 0.5 1 2 4 8
    python3 pfb_baseline.py --task path   --signal psi  --gains 2 4 6 10 --randomize
    python3 pfb_baseline.py --task cmdvel --signal slip --gains 1 2 4 8 --slip-star 0.2
"""
import argparse
import math
from dataclasses import replace

import numpy as np

from compare_policy import METRICS, METRICS_CMDVEL
from config import RoverCfg
from env import CTRL_HZ, RoverEnv
from train import stage1_paths

# v_ref 는 MJCF 순서다 (env.MJ_WHEELS = fl, rl, fr, rr -> C2MJ 재배열 후).
# mjcf.CORNERS 의 좌우부호: fl,rl = +1(좌, +y) / fr,rr = -1(우).
LEFT = np.array([True, True, False, False])

# 요 부호: +omega = CCW = 좌선회.  CCW 를 더 만들려면 **좌측**을 깎는다
# (전차처럼 느린 쪽으로 돈다).  Tohoku 논문의 max(0,-u)/max(0,+u) 와 같은 구조.


def controller_omega(env, kp):
    """**차체 수준 요속도 P 루프.**  측정 자이로만 쓴다 (실기 배포 가능).

    바퀴별 차동 감속(signal=yaw)과 **다른 메커니즘**이다.  저쪽은 한쪽을
    깎아서 요를 만드는데 그러면 병진이 같이 느려진다.  이쪽은 IK 를 통해
    요 명령 자체를 키우므로 네 바퀴가 제대로 재배분된다.

    a = clip(Kp * (om_cmd - om_meas) / d_om_max, -1, 1) 을 요 축에만 넣는다.
    """
    e_om = float(env.cmd_nom[2] - env.gyro_meas[2])
    return np.array([0.0, 0.0,
                     np.clip(kp * e_om / env.cfg.d_om_max, -1.0, 1.0)],
                    dtype=np.float32)


def controller(env, signal, kp, slip_star):
    """[0,1]^4 감속 요구.  음수는 env 에서 max(0,a) 로 걸러지므로 안 쓴다."""
    a = np.zeros(4)
    if signal == "slip":
        # 슬립이 문턱을 넘은 바퀴만, 넘은 만큼 깎는다.
        a = kp * np.maximum(np.abs(env.tm.slip) - slip_star, 0.0)
    else:
        # 두 신호를 **같은 부호 규약**으로 모은다: e_ccw = "CCW(+omega) 를
        # 얼마나 더 만들어야 하는가".  규약을 통일하지 않으면 신호마다
        # 부호가 뒤집혀 제어기가 오차를 키운다.
        if signal == "yaw":
            # 명령 요속도 - 측정 요속도.  om_max 로 무차원화해서 kp 를
            # 지형·속도 설정과 무관하게 만든다.
            e_ccw = (env.cmd_nom[2] - env.gyro_meas[2]) / env.om_max
        else:  # psi -- 측위가 추정한 기수오차 [rad]
            # e_psi_est = yaw_est - course.  양수면 차체가 기준코스보다
            # **좌측(CCW)** 을 보고 있으므로 **CW 로 돌아야** 한다.
            e_ccw = -env.e_psi_est
        u = kp * float(e_ccw)
        # CCW 를 더 만들려면 **좌측**을 깎는다 (전차처럼 느린 쪽으로 돈다).
        a = np.where(LEFT, u, -u)
    return np.clip(a, 0.0, 1.0).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="cmdvel", choices=("path", "cmdvel"))
    p.add_argument("--signal", default="yaw",
                   choices=("yaw", "psi", "slip", "omega"))
    p.add_argument("--gains", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0, 8.0],
                   help="비례게인 목록. 각각이 하나의 arm 이 되어 같은 기준선과 짝비교된다")
    p.add_argument("--slip-star", type=float, default=0.2,
                   help="--signal slip 의 문턱. 이 아래 슬립은 건드리지 않는다")
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--terrain", default="sand", choices=("sand", "rock"))
    p.add_argument("--path-seed", type=int, default=4321,
                   help="holdout 경로 seed (학습 기본값 1234 와 달라야 한다)")
    p.add_argument("--n-paths", type=int, default=30)
    p.add_argument("--seed0", type=int, default=1000)
    p.add_argument("--episode-s", type=float, default=20.0)
    p.add_argument("--difficulty", type=float, default=0.8)
    p.add_argument("--drive-align-gate-deg", type=float, default=10.0)
    p.add_argument("--min-slope-deg", type=float, default=0.0)
    p.add_argument("--path-d-ep", type=float, default=0.0)
    p.add_argument("--randomize", action="store_true",
                   help="센서·구동·측위 잡음 전부 켬. --signal yaw/psi 는 "
                        "**측정값**을 쓰므로 이걸 켠 숫자가 실제 성능이다")
    p.add_argument("--soil-amp", type=float, default=None)
    p.add_argument("--eval-geometry", choices=("arena", "proc"), default=None,
                   help="cmdvel 기본은 미학습 절차지형, path 기본은 규사맵")
    p.add_argument("--push-force", type=float, default=0.0)
    # 자이로 P 루프의 생사는 자이로 잡음이 결정한다.  시뮬 기본값은 ISM330DHCX
    # **데이터시트** 수준(백색 0.010 rad/s, 바이어스 ±0.02)이고 실측이 아니다.
    # 실기가 더 나쁘면 이득이 어디서 사라지는지 알아야 배포 판단을 할 수 있다.
    # **정착 구간 게이트.**  cmdvel 의 요오차는 슬립이 아니라 조향 재정렬
    # 지연이 지배한다 -- 조향 최고속도 5.2 rad/s 로 180도 돌리는 데 0.60초가
    # 걸리고, 그 구간의 오차는 플랜트 한계지 피드백으로 고칠 수 있는 게
    # 아니다.  거기에 "요가 부족하다" 고 더 밀면 아직 정렬 안 된 바퀴를 더
    # 돌릴 뿐이다 (슬립 증가, 병진 손해).  2026-10-07 측정에서 게이트 없는
    # 요 P 루프가 정확히 그렇게 실패했다: 요 평균 -0.36 을 얻는 대가로
    # 병진 +0.52, 요 p90 +0.40~+2.70, 슬립 +0.01~+0.05.
    p.add_argument("--gate-settle-s", type=float, default=0.0,
                   help="명령이 바뀐 뒤 이 시간 동안 보정을 끈다 [s]. "
                        "**시뮬 전용 진단**이다 -- 실기 Nav2 명령은 연속이라 "
                        "'명령이 바뀐 뒤 몇 초' 가 정의되지 않는다. 배포에는 "
                        "--gate-align-deg 를 쓴다")
    p.add_argument("--gate-align-deg", type=float, default=0.0,
                   help="조향 정렬 오차가 이 값보다 크면 보정을 끈다 [deg]. "
                        "**배포 가능한 게이트**다 -- 조향 엔코더로 재고 "
                        "펌웨어 drive_align_gate 가 이미 같은 양을 쓴다. "
                        "10 권장 (게이트 해제 임계와 동일)")
    p.add_argument("--cmd-filter-tau", type=float, default=0.28,
                   help="cmdvel 목표속도 필터 [s]; 0.10=ROS 기본값·s45 이후 학습값")
    p.add_argument("--gyro-noise", type=float, default=None,
                   help="자이로 백색잡음 [rad/s] 덮어쓰기 (기본 0.010)")
    p.add_argument("--gyro-bias", type=float, default=None,
                   help="자이로 바이어스 상한 [rad/s] 덮어쓰기 (기본 0.02). "
                        "실기는 정지 보정으로 뺄 수 있으므로 0 도 의미가 있다")
    p.add_argument("--dump", default=None,
                   help="에피소드별 지표를 .npz 로 저장. 표를 다시 그리거나 "
                        "분포(어느 에피소드에서 이겼나)를 따로 보려면 필요하다 "
                        "-- 스윕 재실행이 수십 분이라 raw 를 남겨야 한다")
    args = p.parse_args()

    if args.episodes < 1 or args.n_paths < 1:
        p.error("--episodes 와 --n-paths 는 양수여야 합니다")
    if not all(math.isfinite(g) and g >= 0 for g in args.gains):
        p.error("--gains 는 0 이상의 유한한 값이어야 합니다")
    if len(set(args.gains)) != len(args.gains):
        p.error("--gains 에 중복이 있습니다")
    if not math.isfinite(args.slip_star) or not 0 <= args.slip_star < 1:
        p.error("--slip-star 는 0..1 이어야 합니다")
    if args.gate_align_deg and not (0 < args.gate_align_deg <= 90):
        p.error("--gate-align-deg 는 0 초과 90 이하여야 합니다")
    if args.signal == "omega" and args.task != "cmdvel":
        p.error("--signal omega 는 cmdvel 과제용입니다")
    if args.signal == "psi" and args.task != "path":
        p.error("--signal psi 는 path 과제에서만 의미가 있습니다 "
                "(cmdvel 에는 기수 기준이 없습니다)")

    # 이 제어기는 Tohoku 논문 그대로 **감속만** 한다 (명령이 v_ref 를 넘지
    # 않는다).  env 기본값은 양방향이므로 여기서 명시적으로 켠다.
    cut_cfg = RoverCfg(d_wheel_cut_only=True)
    geom = args.eval_geometry or ("proc" if args.task == "cmdvel" else "arena")
    paths = (stage1_paths(args.terrain, args.path_seed, args.n_paths,
                          min_slope_deg=args.min_slope_deg, d_ep=args.path_d_ep)
             if geom == "arena" or args.task == "path" else None)
    nz = None
    if args.gyro_noise is not None or args.gyro_bias is not None:
        from config import NoiseCfg
        base = NoiseCfg()
        kw = {}
        if args.gyro_noise is not None:
            if not math.isfinite(args.gyro_noise) or args.gyro_noise < 0:
                p.error("--gyro-noise 는 0 이상이어야 합니다")
            kw["gyro_noise"] = float(args.gyro_noise)
        if args.gyro_bias is not None:
            if not math.isfinite(args.gyro_bias) or args.gyro_bias < 0:
                p.error("--gyro-bias 는 0 이상이어야 합니다")
            kw["gyro_bias"] = (-float(args.gyro_bias), float(args.gyro_bias))
        nz = replace(base, **kw)

    mode = "body" if args.signal == "omega" else "wheel"
    env_args = dict(cfg=(RoverCfg() if mode == "body" else cut_cfg),
                    task=args.task, action_mode=mode,
                    difficulty=args.difficulty, episode_s=args.episode_s,
                    arena_eval=geom == "arena", eval_kind=args.terrain, paths=paths,
                    randomize=args.randomize, soil_amp=args.soil_amp, noise=nz,
                    cmd_filter_tau=args.cmd_filter_tau,
                    drive_align_gate_deg=args.drive_align_gate_deg,
                    push_force_max=args.push_force)

    # arm 마다 **독립 env** 를 쓴다.  같은 seed 로 reset 하면 경로·지형·
    # 잡음 시드가 같으므로 짝비교가 성립한다 (compare_policy 와 같은 방식).
    arms = [("순수 IK", None)] + [(f"Kp={g:g}", g) for g in args.gains]
    envs = [RoverEnv(**env_args, seed=args.seed0) for _ in arms]

    cfg = cut_cfg
    soil_eff = args.soil_amp if args.soil_amp is not None else (
        cfg.soil_amp if args.randomize else 0.0)
    print(f"고전 P 피드백 기준선 (감속 전용) | task={args.task} | "
          f"signal={args.signal}"
          f"{f' s*={args.slip_star}' if args.signal == 'slip' else ''} | "
          f"{args.terrain} geometry={geom} | "
          f"holdout seed={args.path_seed if paths is not None else 'none'} | "
          f"{args.episodes} 에피소드 | d={args.difficulty} | "
          f"잡음 {'ON' if args.randomize else 'OFF'} | 모래비대칭 {soil_eff} | "
          f"감속권한 최대 -{cfg.d_wheel_frac * 100:.0f}%(감속전용) | "
          f"lookahead L={cfg.pp_l_min:.2f}~{cfg.pp_l_max:.2f}m", flush=True)
    if args.signal == "slip":
        print("  주의: --signal slip 은 **참 슬립**을 읽는다. 실기 배포 불가, "
              "상한 측정용이다.", flush=True)

    out = [[] for _ in arms]
    # 게이트를 켜면 보정이 전체 스텝의 일부에만 걸리므로 에피소드 평균이
    # 희석된다.  **보정이 실제로 작동한 구간**의 오차를 따로 모아야 효과를
    # 볼 수 있다.
    settled = [[] for _ in arms]
    zero = np.zeros(3 if mode == "body" else 4, dtype=np.float32)
    for i in range(args.episodes):
        seed = args.seed0 + i
        for j, (_, g) in enumerate(arms):
            env = envs[j]
            env.reset(seed=seed)
            done = False
            ep_settled = []
            while not done:
                phase = ((env._cmd_k - 1) % env.cmd_hold) / CTRL_HZ
                on = phase >= args.gate_settle_s
                if args.gate_align_deg > 0:
                    on = on and env.steer_err <= math.radians(args.gate_align_deg)
                if g is None or not on:
                    a = zero
                elif args.signal == "omega":
                    a = controller_omega(env, g)
                else:
                    a = controller(env, args.signal, g, args.slip_star)
                _, _, term, trunc, info = env.step(a)
                if on and env.task == "cmdvel":
                    ep_settled.append(env.vel_err_hist[-1])
                done = term or trunc
            if ep_settled:
                settled[j].append(np.mean(ep_settled, axis=0))
            out[j].append(info)
        if (i + 1) % 10 == 0 or i + 1 == args.episodes:
            print(f"  {i + 1}/{args.episodes} 짝 완료", flush=True)
    for env in envs:
        env.close()

    metrics = METRICS_CMDVEL if args.task == "cmdvel" else METRICS
    if args.dump:
        keys = [k for k, *_ in metrics if k in out[0][0]]
        np.savez(args.dump,
                 arms=np.array([lab for lab, _ in arms]),
                 gains=np.array([-1.0 if g is None else g for _, g in arms]),
                 keys=np.array(keys),
                 # [arm, episode, metric]
                 data=np.array([[[ep[k] for k in keys] for ep in arm_out]
                                for arm_out in out], dtype=float))
        print(f"raw 저장: {args.dump}", flush=True)
    rng = np.random.default_rng(0)
    resamples = rng.integers(0, args.episodes, size=(10000, args.episodes))

    hdr = f"{'지표':<17} {'순수 IK':>10}"
    for lab, _g in arms[1:]:
        hdr += f" {lab:>19}"
    print("\n" + hdr)
    print("-" * (28 + 20 * len(args.gains)))
    for key, label, scale, unit in metrics:
        if key not in out[0][0]:
            continue
        b = np.array([v[key] for v in out[0]], dtype=float) * scale
        row = f"{label:<17} {np.nanmean(b):10.2f}"
        for j in range(1, len(arms)):
            r = np.array([v[key] for v in out[j]], dtype=float) * scale
            valid = np.isfinite(b) & np.isfinite(r)
            if not valid.any():
                row += f" {'N/A':>19}"
                continue
            d = (r - b)[valid]
            idx = (resamples if valid.all() else
                   rng.integers(0, len(d), size=(10000, len(d))))
            lo, hi = np.percentile(d[idx].mean(axis=1), (2.5, 97.5))
            # `(lo>0)==(hi>0)` 로 쓰면 lo=hi=0 (모든 에피소드에서 차이가 정확히
            # 0, 예: 고착 0% vs 0%) 인 행에 별표가 찍힌다.  0 을 **포함하지
            # 않는지**를 직접 물어야 한다.
            sig = "*" if lo > 0 or hi < 0 else " "
            row += f" {np.nanmean(r):8.2f} {np.mean(d):+7.2f}{sig}"
        print(f"{row}  {unit}")
    # 보정이 실제로 켜진 구간만 따로.  게이트를 쓰면 이게 본 지표다.
    if settled[0] and args.task == "cmdvel":
        v, om = envs[0].v_max * 1000, np.degrees(envs[0].om_max)
        gl = []
        if args.gate_settle_s > 0:
            gl.append(f"명령 전환 후 {args.gate_settle_s:.2f}s 이후")
        if args.gate_align_deg > 0:
            gl.append(f"조향 정렬오차 <= {args.gate_align_deg:.0f}deg")
        print(f"\n[보정 작동 구간만] {' 그리고 '.join(gl) if gl else '전체'}")
        base_s = np.asarray(settled[0])
        for j in range(len(arms)):
            a = np.asarray(settled[j])
            row = f"  {arms[j][0]:<10} 병진 {a[:,0].mean()*v:6.1f} mm/s  요 {np.degrees(a[:,1].mean()*envs[0].om_max):5.2f} deg/s"
            if j:
                d = a - base_s
                lo, hi = np.percentile(
                    d[resamples[:, :len(d)], 0].mean(axis=1) * v, (2.5, 97.5))
                lo2, hi2 = np.percentile(
                    np.degrees(d[resamples[:, :len(d)], 1].mean(axis=1) * envs[0].om_max),
                    (2.5, 97.5))
                row += (f"  | 차이 병진 {d[:,0].mean()*v:+6.2f} [{lo:+.2f},{hi:+.2f}]"
                        f"  요 {np.degrees(d[:,1].mean()*envs[0].om_max):+5.2f} [{lo2:+.2f},{hi2:+.2f}]")
            print(row)

    print("\n* = 짝 bootstrap 95% CI 가 0 을 포함하지 않음 (10000 resample).")
    print("각 칸은 '해당 arm 평균 / 순수 IK 대비 차이' 다.")
    if args.task == "cmdvel" and args.signal != "omega":
        print("병진속도오차는 **악화되는 것이 정상**이다: 감속 전용이라 "
              "부족분을 메울 수 없다. 요오차·슬립·침하를 봐라.")
    if args.signal == "omega":
        print("차체 요 잔차는 **측정 자이로만** 쓴다 -- 실기 배포 가능한 숫자다.")


if __name__ == "__main__":
    main()
