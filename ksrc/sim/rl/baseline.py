#!/usr/bin/env python3
"""순수 IK(잔차 a=0) 기준선을 재는 도구.

`compare_policy.py` 는 매 실행에서 순수 IK 를 같은 시드로 다시 돌려 짝비교하므로
정책 판정에는 이 스크립트가 필요 없다.  정책이 없는 상태에서 **기준선 숫자 자체를
알고 싶을 때**(경로 생성기나 조향범위를 바꾼 뒤) 쓴다.

사용법:
    python3 baseline.py                                  # 기본 holdout 100 에피소드
    python3 baseline.py --episodes 60 --steer-limit-deg 135
    python3 baseline.py --drive-align-gate-deg 0
"""
import argparse

import numpy as np

from config import RoverCfg
from env import RoverEnv
from train import stage1_paths

METRICS = (
    ("success",    "완주율",       100.0, "%"),
    ("frac",       "경로 진행률",   100.0, "%"),
    ("e_y_m",      "횡오차 평균",  1000.0, "mm"),
    ("e_y_p90",    "횡오차 p90",   1000.0, "mm"),
    ("e_y_max",    "횡오차 최대",  1000.0, "mm"),
    ("e_psi_m",    "기수오차 평균",   1.0, "deg"),
    ("e_psi_p90",  "기수오차 p90",    1.0, "deg"),
    ("slip_m",     "슬립 평균",       1.0, ""),
    ("slip_p90",   "슬립 p90",        1.0, ""),
    # 침하는 terramech 의 zdot = alpha|i||v| - beta z 로 슬립에서 쌓이고
    # 구름저항 Rn = Fn(c_r + k_z z) 로 돌아온다 (k_z 8.0).  모래 로버에서
    # 결정적인 양인데 2026-10-04 까지 비교표에 **없었다** -- 슬립을 +26%
    # 쓰는 정책을 "개선" 으로 판정하면서 그 대가를 측정하지 않았다.
    ("sink_m",     "침하 평균",    1000.0, "mm"),
    ("t_elapsed",  "소요시간",        1.0, "s"),
    ("cmd_sat",    "차체명령 포화", 100.0, "%"),
    ("drive_gate", "조향 대기",     100.0, "%"),
    ("tip",        "전복",          100.0, "%"),
    ("stuck",      "고착",          100.0, "%"),
    ("oob",        "경계 이탈",     100.0, "%"),
)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--terrain", default="sand", choices=("sand", "rock"))
    p.add_argument("--path-seed", type=int, default=4321,
                   help="holdout 경로 seed. 학습 기본값은 1234 이므로 달라야 한다")
    p.add_argument("--n-paths", type=int, default=30)
    p.add_argument("--seed0", type=int, default=1000)
    p.add_argument("--episode-s", type=float, default=20.0)
    p.add_argument("--difficulty", type=float, default=0.8)
    p.add_argument("--drive-align-gate-deg", type=float, default=10.0)
    p.add_argument("--min-slope-deg", type=float, default=0.0,
                   help="경로가 지나는 평균 지형경사의 하한 [deg]. 0 이면 기존 동작(경사 무시). 2026-10-04 측정: 하한이 없으면 경로 경사 중앙값이 0도라 견인력 수요가 R/N=0.057 뿐이고, 바퀴별 mu 를 +-16%% 흔들어도 목적 현상이 발생하지 않는다. 8 이면 실제 평균 9.5도(수요 0.165)로 가용 mu(i) 0.17~0.39 와 같은 범위가 된다")
    p.add_argument("--path-d-ep", type=float, default=0.0,
                   help="make_path 의 급사면 선호 확률. min-slope-deg 와 함께 쓴다 (0.9 권장)")
    p.add_argument("--randomize", action="store_true",
                   help="센서·구동·측위 잡음 전부 켬 (전압/엔코더/IMU/서보지연/"
                        "제어지연/측위). 실기에 반드시 존재하는 외란이다")
    p.add_argument("--soil-amp", type=float, default=None,
                   help="모래 비대칭 장(바퀴별 mu 배율) 진폭. **이 RL 의 과제 본체**다 (terrain.soil_field 참고). --stage1 은 randomize 를 통째로 꺼서 이것까지 끄므로, Stage 1 에서 목적 현상을 쓰려면 이 값을 명시한다. 미지정이면 기존 동작(randomize 에 종속). 0.25 권장, 0.5 는 장이 음수가 된다")
    p.add_argument("--steer-limit-deg", type=float, default=None,
                   help="조향 가동범위 ±[deg] 덮어쓰기 (기본 config 의 ±100)")
    # **기준선 튜닝 노브 (2026-10-04 추가).**  s1~s18 캠페인 전체가 lookahead
    # 고정값(k_v 1.2, 0.18~0.45m)의 pure pursuit 를 상대로 측정됐고 스윕 로그가
    # 0건이다.  그런데 env.py:224 에 "참 위치를 쓰면 lookahead 를 짧게 잡는 게
    # 항상 유리하다 (e_y 중앙 38.8 -> 4.0mm)" 가 이미 적혀 있다 -- 노브 하나가
    # 34.8mm 를 움직이는데 RL 이 주장하는 개선은 1.88mm 다.  "잡음 ON 에서는
    # 짧은 lookahead 가 발진한다" 는 주장이었고 측정이 아니었으므로, 잡음을
    # 켠 상태에서 직접 재는 길을 열어둔다.
    p.add_argument("--d-rate-max", type=float, default=None,
                   help="잔차 변화율 제한 [스텝당 |da|]. 속도 액션은 급격한 변화를 요구할 수 있어 실기에서 깨진 사례가 있다 (arXiv:2504.02420). 0.25 = 80ms 에 -1 -> +1. 미지정이면 제한 없음(기존 동작)")
    p.add_argument("--pp-k-v", type=float, default=None,
                   help="pure pursuit lookahead 속도계수 (기본 1.2). "
                        "L = clip(k_v * v_cruise, l_min, l_max)")
    p.add_argument("--pp-l-min", type=float, default=None,
                   help="lookahead 하한 [m] (기본 0.18)")
    p.add_argument("--pp-l-max", type=float, default=None,
                   help="lookahead 상한 [m] (기본 0.45)")
    a = p.parse_args()
    if a.episodes < 1 or a.n_paths < 1:
        p.error("--episodes 와 --n-paths 는 양수여야 합니다")

    over = {}
    if a.steer_limit_deg is not None:
        if not (90.0 <= a.steer_limit_deg <= 180.0):
            p.error("--steer-limit-deg 는 90~180 사이여야 합니다")
        lim = float(a.steer_limit_deg)
        over.update(steer_lo_deg=(-lim,) * 4, steer_hi_deg=(lim,) * 4)
    for flag, key in (("pp_k_v", "pp_k_v"), ("pp_l_min", "pp_l_min"),
                      ("pp_l_max", "pp_l_max")):
        v = getattr(a, flag)
        if v is not None:
            if not np.isfinite(v) or v <= 0:
                p.error(f"--{flag.replace('_', '-')} 는 양의 유한한 값이어야 합니다")
            over[key] = float(v)
    if a.d_rate_max is not None:
        if not np.isfinite(a.d_rate_max) or not 0 < a.d_rate_max <= 2.0:
            p.error("--d-rate-max 는 0 초과 2.0 이하여야 합니다 (액션 범위가 [-1,1])")
        over["d_rate_max"] = float(a.d_rate_max)
    base_cfg = RoverCfg()
    l_min = over.get("pp_l_min", base_cfg.pp_l_min)
    l_max = over.get("pp_l_max", base_cfg.pp_l_max)
    if l_min > l_max:
        p.error("--pp-l-min 이 --pp-l-max 보다 큽니다")
    cfg = RoverCfg(**over) if over else None
    shown = a.steer_limit_deg if a.steer_limit_deg is not None else base_cfg.steer_hi_deg[0]

    # **라벨 버그 (2026-10-04 수정).**  `'없음' if not a.soil_amp` 은
    # soil_amp=None (미지정) 에서도 "없음" 을 찍었다.  그런데 미지정이면
    # env 가 randomize 에 종속시켜 cfg.soil_amp=0.25 를 적용한다.  그래서
    # b_noise.log(soil 0.25) 와 b_noise_nosoil.log(soil 0) 의 헤더가 완전히
    # 동일했고, "토양 비대칭은 e_y 를 +0.03mm 만 움직인다" 는 결론이 실제로는
    # 잡음 OFF 조건의 수치였다 (잡음 ON 에서는 +0.231mm, 7.7배).
    soil_eff = a.soil_amp if a.soil_amp is not None else (
        RoverCfg().soil_amp if a.randomize else 0.0)
    soil_lbl = ("없음" if soil_eff == 0 else f"{soil_eff}") + (
        "" if a.soil_amp is not None else "(randomize 기본값)")

    paths = stage1_paths(a.terrain, a.path_seed, a.n_paths,
                         min_slope_deg=a.min_slope_deg, d_ep=a.path_d_ep)
    env = RoverEnv(cfg=cfg, paths=paths, seed=a.seed0, difficulty=a.difficulty,
                   episode_s=a.episode_s, arena_eval=True, eval_kind=a.terrain,
                   randomize=a.randomize, drive_align_gate_deg=a.drive_align_gate_deg,
                   soil_amp=a.soil_amp)
    print(f"순수 IK 기준선 | {a.terrain} | holdout seed={a.path_seed} | "
          f"{a.episodes} 에피소드 | d={a.difficulty} | 조향 ±{shown:.0f}도 | "
          f"게이트 {a.drive_align_gate_deg:.0f}도 | "
          f"모래비대칭 {soil_lbl} | "
          f"잡음 {'ON' if a.randomize else 'OFF'} | "
          f"lookahead k_v={over.get('pp_k_v', base_cfg.pp_k_v)} "
          f"L={l_min:.2f}~{l_max:.2f}m", flush=True)

    zero = np.zeros(3, dtype=np.float32)
    out = []
    for i in range(a.episodes):
        env.reset(seed=a.seed0 + i)
        done = False
        while not done:
            _, _, term, trunc, info = env.step(zero)
            done = term or trunc
        out.append(info)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{a.episodes}", flush=True)
    env.close()

    rng = np.random.default_rng(0)
    rs = rng.integers(0, a.episodes, size=(10000, a.episodes))
    print(f"\n{'지표':<16}{'평균':>10}{'95% CI':>22}")
    for k, lab, sc, unit in METRICS:
        if k not in out[0]:
            continue
        v = np.array([x[k] for x in out], dtype=float) * sc
        lo, hi = np.percentile(v[rs].mean(axis=1), (2.5, 97.5))
        print(f"{lab:<16}{v.mean():10.3f}  [{lo:+.3f}, {hi:+.3f}] {unit}")


if __name__ == "__main__":
    main()
