#!/usr/bin/env python3
"""lookahead 를 **에피소드별·상황별**로 감사한다.

왜 필요한가: baseline.py 는 100 에피소드 평균 하나만 낸다.  평균이 좋아도
(ㄱ) 소수 에피소드에서 크게 벌고 다수에서 조금씩 잃은 것일 수 있고,
(ㄴ) 직선에서만 좋고 꼭짓점에서 나쁠 수 있고,
(ㄷ) L 을 고정하면 **빠른 에피소드에서만** 망가질 수 있다 (속도비례를 뺐으므로).
그래서 같은 seed/경로를 여러 L 로 짝지어 돌리고 스텝 단위로 기록한다.

  python3 pp_audit.py --L 0.16 0.264 --episodes 60 --randomize
"""
import argparse, math, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from config import RoverCfg
from env import RoverEnv
from train import stage1_paths


def vertex_context(path, s):
    """현재 호길이에서 (다음 꼭짓점까지 거리, 그 꼭짓점 꺾임각[deg])."""
    S, U = path.S, path.U
    i = int(np.clip(np.searchsorted(S, s, side="right") - 1, 0, len(U) - 1))
    d_next = float(S[i + 1] - s)
    if i + 1 < len(U):
        c = float(np.clip(np.dot(U[i], U[i + 1]), -1.0, 1.0))
        turn = math.degrees(math.acos(c))
    else:
        turn = 0.0                       # 마지막 구간 -> 꺾임 없음
    return d_next, turn


def run(L, a, paths, cfg_kw):
    """하나의 L 로 모든 에피소드를 돌리고 스텝/에피소드 기록을 반환."""
    cfg = RoverCfg(pp_l_min=L, pp_l_max=L, **cfg_kw)
    env = RoverEnv(cfg=cfg, paths=paths, seed=a.seed0, difficulty=a.difficulty,
                   episode_s=a.episode_s, arena_eval=True, eval_kind=a.terrain,
                   randomize=a.randomize, drive_align_gate_deg=a.drive_align_gate_deg,
                   soil_amp=a.soil_amp)
    zero = np.zeros(3, dtype=np.float32)
    steps, eps = [], []
    for i in range(a.episodes):
        env.reset(seed=a.seed0 + i)
        v_cruise, total = float(env.v_cruise), float(env.path.total)
        done, rec = False, []
        while not done:
            s_before = float(env.s_path)
            _, _, term, trunc, info = env.step(zero)
            d_next, turn = vertex_context(env.path, s_before)
            rec.append((s_before, info["e_y"], info["e_psi"], info["slip"],
                        info["sink"], d_next, turn))
            done = term or trunc
        r = np.asarray(rec, float)
        steps.append(r)
        eps.append(dict(ep=i, v_cruise=v_cruise, total=total,
                        n=len(r),
                        e_y_m=float(r[:, 1].mean()),
                        e_y_p90=float(np.percentile(r[:, 1], 90)),
                        e_y_max=float(r[:, 1].max()),
                        e_psi_m=float(r[:, 2].mean()),
                        slip_m=float(r[:, 3].mean()),
                        sink_m=float(r[:, 4].mean()),
                        turn_max=float(r[:, 6].max()),
                        success=float(info["success"])))
        if (i + 1) % 20 == 0:
            print(f"  L={L}  {i+1}/{a.episodes}", flush=True)
    env.close()
    return steps, eps


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--L", type=float, nargs="+", required=True,
                   help="비교할 lookahead 들 [m]. 첫 값이 기준")
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--terrain", default="sand", choices=("sand", "rock"))
    p.add_argument("--path-seed", type=int, default=4321)
    p.add_argument("--n-paths", type=int, default=30)
    p.add_argument("--seed0", type=int, default=1000)
    p.add_argument("--episode-s", type=float, default=20.0)
    p.add_argument("--difficulty", type=float, default=0.8)
    p.add_argument("--drive-align-gate-deg", type=float, default=10.0)
    p.add_argument("--randomize", action="store_true")
    p.add_argument("--soil-amp", type=float, default=None)
    p.add_argument("--min-slope-deg", type=float, default=0.0)
    p.add_argument("--path-d-ep", type=float, default=0.0)
    p.add_argument("--save", default=None, help="npz 저장 경로")
    a = p.parse_args()
    if a.episodes < 1 or len(a.L) < 2:
        p.error("--episodes 는 1 이상, --L 은 2개 이상이어야 합니다")

    paths = stage1_paths(a.terrain, a.path_seed, a.n_paths,
                         min_slope_deg=a.min_slope_deg, d_ep=a.path_d_ep)
    soil_eff = a.soil_amp if a.soil_amp is not None else (
        RoverCfg().soil_amp if a.randomize else 0.0)
    print(f"lookahead 감사 | {a.terrain} | seed={a.path_seed} | {a.episodes} 에피소드 | "
          f"잡음 {'ON' if a.randomize else 'OFF'} | soil {soil_eff} | "
          f"경사하한 {a.min_slope_deg}도 | L={a.L}", flush=True)

    D = {}
    for L in a.L:
        D[L] = run(L, a, paths, {})
    # 짝 검증: 같은 seed 면 v_cruise/경로길이가 같아야 한다 (L 은 RNG 를 안 쓴다)
    ref = a.L[0]
    for L in a.L[1:]:
        for e0, e1 in zip(D[ref][1], D[L][1]):
            assert abs(e0["v_cruise"] - e1["v_cruise"]) < 1e-9 and \
                   abs(e0["total"] - e1["total"]) < 1e-9, "짝이 깨졌습니다"
    print("짝 검증 통과: 같은 seed 에서 v_cruise/경로길이 동일\n")

    # ---------- 1) 에피소드별 승패 ----------
    print("=" * 92)
    print("[1] 에피소드별 승패 (기준 L=%.3f, 음수 = 후보가 좋음, 단위 mm)" % ref)
    print("=" * 92)
    E = {L: D[L][1] for L in a.L}
    print(f"{'비교':<16}{'승':>5}{'패':>5}{'승률':>7}"
          f"{'Δ평균':>9}{'Δp10':>8}{'Δp50':>8}{'Δp90':>8}{'최악ep':>9}{'최악Δ':>8}")
    for L in a.L[1:]:
        d = np.array([E[L][i]["e_y_m"] - E[ref][i]["e_y_m"] for i in range(a.episodes)]) * 1000
        w = int((d < 0).sum()); l = int((d > 0).sum())
        worst = int(np.argmax(d))
        print(f"{'L=%.3f' % L:<16}{w:>5}{l:>5}{w/len(d)*100:>6.0f}%"
              f"{d.mean():>9.2f}{np.percentile(d,10):>8.2f}{np.percentile(d,50):>8.2f}"
              f"{np.percentile(d,90):>8.2f}{worst:>9d}{d[worst]:>8.2f}")
    # 완주 불일치
    print()
    for L in a.L[1:]:
        only_ref = [i for i in range(a.episodes)
                    if E[ref][i]["success"] > E[L][i]["success"]]
        only_cand = [i for i in range(a.episodes)
                     if E[L][i]["success"] > E[ref][i]["success"]]
        print(f"  완주 불일치 L={L:.3f}: 기준만 성공 {len(only_ref)}건 {only_ref[:8]}, "
              f"후보만 성공 {len(only_cand)}건 {only_cand[:8]}")

    # ---------- 2) 속도 의존 ----------
    print("\n" + "=" * 92)
    print("[2] 속도 의존 -- L 고정이 빠른 에피소드를 망치는가 (v_cruise 3분위)")
    print("=" * 92)
    v = np.array([e["v_cruise"] for e in E[ref]])
    q = np.quantile(v, [1/3, 2/3])
    bins = [("느림 v<%.3f" % q[0], v < q[0]),
            ("중간", (v >= q[0]) & (v < q[1])),
            ("빠름 v>=%.3f" % q[1], v >= q[1])]
    print(f"{'구간':<20}{'n':>4}" + "".join(f"{'L=%.3f' % L:>11}" for L in a.L))
    for name, m in bins:
        row = f"{name:<20}{int(m.sum()):>4}"
        for L in a.L:
            row += f"{np.mean([E[L][i]['e_y_m'] for i in np.where(m)[0]])*1000:>11.2f}"
        print(row)
    print("  (숫자 = 횡오차 평균 mm)")

    # ---------- 3) 상황 의존: 꼭짓점까지 거리 x 꺾임각 ----------
    print("\n" + "=" * 92)
    print("[3] 상황 의존 -- 직선 구간 vs 꼭짓점 접근 (스텝 단위)")
    print("=" * 92)
    cat = {}
    for L in a.L:
        S = np.vstack(D[L][0])
        d_next, turn, e_y = S[:, 5], S[:, 6], S[:, 1] * 1000
        cat[L] = (d_next, turn, e_y)
    # L 마다 에피소드 길이가 달라 스텝 수가 다르므로 **각 L 의 자기 궤적에서**
    # 상황을 분류한다.  상황 구성비(스텝%)도 L 마다 달라지는데, 그 자체가
    # 정보다 -- 짧은 L 은 꼭짓점 근처에서 더 오래 머물 수 있다.
    def masks(d_next, turn):
        return [("직선 (꼭짓점 >0.5m)", d_next > 0.5),
                ("접근 0.25~0.5m, 꺾임<45도",
                 (d_next > 0.25) & (d_next <= 0.5) & (turn < 45)),
                ("접근 0.25~0.5m, 꺾임>=45도",
                 (d_next > 0.25) & (d_next <= 0.5) & (turn >= 45)),
                ("꼭짓점 <0.25m, 꺾임<45도", (d_next <= 0.25) & (turn < 45)),
                ("꼭짓점 <0.25m, 꺾임>=45도", (d_next <= 0.25) & (turn >= 45))]
    names = [n for n, _ in masks(*cat[ref][:2])]
    print(f"{'상황':<30}" + "".join(f"{'L=%.3f' % L:>19}" for L in a.L))
    print(f"{'':<30}" + "".join(f"{'스텝%':>8}{'횡오차':>11}" for _ in a.L))
    for j, name in enumerate(names):
        row = f"{name:<30}"
        for L in a.L:
            d_next, turn, e_y = cat[L]
            m = masks(d_next, turn)[j][1]
            row += (f"{m.mean()*100:>7.1f}%{e_y[m].mean():>11.2f}" if m.any()
                    else f"{'-':>8}{'-':>11}")
        print(row)
    print("  (횡오차 = 그 상황 스텝들의 평균 mm.  각 L 의 자기 궤적에서 분류)")

    # ---------- 4) 스텝 분위수 ----------
    print("\n" + "=" * 92)
    print("[4] 스텝 횡오차 분포 (평균 하나로 숨는 꼬리)")
    print("=" * 92)
    print(f"{'L':<10}{'p50':>9}{'p75':>9}{'p90':>9}{'p99':>9}{'최대':>9}{'평균':>9}")
    for L in a.L:
        e = cat[L][2]
        print(f"{'%.3f' % L:<10}{np.percentile(e,50):>9.2f}{np.percentile(e,75):>9.2f}"
              f"{np.percentile(e,90):>9.2f}{np.percentile(e,99):>9.2f}"
              f"{e.max():>9.2f}{e.mean():>9.2f}")

    if a.save:
        np.savez_compressed(a.save, **{f"steps_{L}": np.vstack(D[L][0]) for L in a.L},
                            **{f"eps_{L}": np.array([[e[k] for k in
                                ("ep","v_cruise","total","n","e_y_m","e_y_p90","e_y_max",
                                 "e_psi_m","slip_m","sink_m","turn_max","success")]
                                for e in D[L][1]]) for L in a.L})
        print(f"\n저장 -> {a.save}")


if __name__ == "__main__":
    main()
