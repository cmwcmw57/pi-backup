#!/usr/bin/env python3
"""순수 IK 가 **랜덤 바닥에서 cmd_vel 을 얼마나 못 추종하는지** 측정한다.

왜: 지금까지의 RL 은 "경로추종" 과제였는데, 그 보상이 경로 기하 + pure pursuit
튜닝 + 지형을 뒤섞어서 "RL 이 잘한 건가 lookahead 가 틀렸던 건가" 를 가릴 수
없었다 (실측: lookahead 를 0.264 -> 0.20 으로 맞추자 RL 의 횡오차 이득이
-1.88mm -> +0.18mm 로 소멸).  밑단 RL 로 쪼개면 과제가
    "명령한 차체속도를 변형지형에서 달성하라"
하나가 되고 보상이 ||v_실제 - v_명령|| 하나가 된다.  그 전에 **개선 여지의
상한**을 알아야 한다 = 순수 IK 의 추종오차.

경로는 env 내부 기계(종료판정 등)를 위해 남겨두되 명령에는 쓰지 않는다.

  python3 cmdvel_probe.py --episodes 40 --randomize
"""
import argparse, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from config import RoverCfg
from env import RoverEnv, CTRL_HZ
from train import stage1_paths


# env.RoverEnv(task="cmdvel") 가 명령 스케줄과 보상을 직접 갖는다.
# (이 파일이 먼저 서브클래스로 시제품을 만들었고, 검증 후 env 로 옮겼다.)


def ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    r = np.random.default_rng(seed)
    bs = x[r.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--episode-s", type=float, default=8.0)
    p.add_argument("--terrain", default="sand", choices=("sand", "rock"))
    p.add_argument("--path-seed", type=int, default=4321)
    p.add_argument("--n-paths", type=int, default=30)
    p.add_argument("--seed0", type=int, default=1000)
    p.add_argument("--difficulty", type=float, default=0.8)
    p.add_argument("--randomize", action="store_true")
    p.add_argument("--soil-amp", type=float, default=None)
    p.add_argument("--save", default=None)
    a = p.parse_args()

    paths = stage1_paths(a.terrain, a.path_seed, a.n_paths)
    env = RoverEnv(paths=paths, seed=a.seed0, difficulty=a.difficulty,
                   episode_s=a.episode_s, arena_eval=True, eval_kind=a.terrain,
                   randomize=a.randomize, soil_amp=a.soil_amp, task="cmdvel")
    soil_eff = a.soil_amp if a.soil_amp is not None else (
        RoverCfg().soil_amp if a.randomize else 0.0)
    print(f"cmd_vel 추종 측정 (순수 IK, a=0) | {a.terrain} | {a.episodes} 에피소드 x "
          f"{a.episode_s}s | 잡음 {'ON' if a.randomize else 'OFF'} | soil {soil_eff}",
          flush=True)
    print(f"  권한 참고: d_vx_max={env.cfg.d_vx_max} d_vy_max={env.cfg.d_vy_max} "
          f"d_om_max={env.cfg.d_om_max} | v_max={env.v_max:.3f} m/s", flush=True)

    zero = np.zeros(3, dtype=np.float32)
    rows = []
    for i in range(a.episodes):
        env.reset(seed=a.seed0 + i)
        done = False
        while not done:
            cmd = env.cmd_nom.copy()          # 이번 스텝에 인가되는 명령
            # 명령이 목표에 수렴한 뒤 몇 스텝 지났나.  과도 지연(대역폭 한계,
            # 잔차로 못 고침)과 정상상태 편향(슬립, 잔차로 고침)을 가르는 열쇠.
            settle = float(np.max(np.abs(env.cmd_cur - env.cmd_tgt)))
            _, _, term, trunc, _ = env.step(zero)
            tw = env.body_twist()
            rows.append(np.concatenate([cmd, tw, env.tm.slip, env.tm.z,
                                        [float(env.n_cmd_sat), settle,
                                         float(env._cmd_k % env.cmd_hold)]]))
            done = term or trunc
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{a.episodes}", flush=True)
    env.close()
    R = np.asarray(rows)
    cmd, tw, slip, sink = R[:, 0:3], R[:, 3:6], R[:, 6:10], R[:, 10:14]
    settle, since = R[:, 15], R[:, 16]
    err = tw - cmd
    # 정상상태: 명령이 목표의 2% 안으로 수렴했고 그 뒤 1초 이상 지난 스텝
    ss = (settle < 0.02 * max(env.v_max, 1e-6)) & (since > CTRL_HZ)
    names = ("vx [m/s]", "vy [m/s]", "omega [rad/s]")
    lim = (env.cfg.d_vx_max, env.cfg.d_vy_max, env.cfg.d_om_max)

    print(f"\n{'성분':<14}{'명령|평균|':>10}{'오차평균':>10}{'오차|평균|':>11}"
          f"{'|오차|p90':>10}{'|오차|p99':>10}{'잔차권한':>10}{'권한/p90':>9}")
    for j in range(3):
        e = err[:, j]
        print(f"{names[j]:<14}{np.abs(cmd[:,j]).mean():10.4f}{e.mean():+10.4f}"
              f"{np.abs(e).mean():11.4f}{np.percentile(np.abs(e),90):10.4f}"
              f"{np.percentile(np.abs(e),99):10.4f}{lim[j]:10.3f}"
              f"{lim[j]/max(np.percentile(np.abs(e),90),1e-9):9.2f}")
    print("\n  '권한/p90' < 1 이면 **잔차 권한이 오차의 90분위를 못 덮는다**")

    # ---- 과도 vs 정상상태 분리 ----
    print(f"\n=== 과도 지연 vs 정상상태 편향 ===")
    print(f"정상상태 스텝: {ss.sum()}/{len(ss)} ({ss.mean()*100:.1f}%)"
          f"   (명령 수렴 후 1초 경과)")
    print(f"{'성분':<14}{'전체|오차|':>11}{'과도|오차|':>11}{'정상|오차|':>11}"
          f"{'정상 평균':>11}{'정상 p90':>10}{'권한':>8}{'권한/정상p90':>13}")
    need = []
    for j in range(3):
        e = err[:, j]
        ep90 = np.percentile(np.abs(e[ss]), 90) if ss.any() else float('nan')
        need.append(ep90)
        print(f"{names[j]:<14}{np.abs(e).mean():11.4f}{np.abs(e[~ss]).mean():11.4f}"
              f"{np.abs(e[ss]).mean():11.4f}{e[ss].mean():+11.4f}{ep90:10.4f}"
              f"{lim[j]:8.3f}{lim[j]/max(ep90,1e-9):13.2f}")
    print(f"\n  정상상태 p90 을 덮는 권한 = "
          f"[{need[0]:.3f}, {need[1]:.3f}, {need[2]:.3f}]  (현재 [0.030, 0.030, 0.100])")
    # 정상상태에서 전진 중 속도 부족이 슬립으로 설명되는가
    m = ss & (cmd[:, 0] > 0.3 * cmd[:, 0].max())
    if m.sum() > 30:
        sm = np.abs(slip[m]).mean(axis=1)
        q = np.quantile(sm, [1/3, 2/3])
        print(f"\n  정상상태 전진 구간 (n={int(m.sum())}), 슬립 3분위별:")
        for nm, k in (("하위", sm < q[0]), ("중간", (sm >= q[0]) & (sm < q[1])),
                      ("상위", sm >= q[1])):
            print(f"    슬립{nm} ({sm[k].mean():.3f})  vx 오차 {err[m,0][k].mean():+.4f}"
                  f"   |omega 오차| {np.abs(err[m,2][k]).mean():.4f}")

    # 상대오차 (명령이 작을 때의 나눗셈 폭주를 피해 큰 명령만)
    print(f"\n{'성분':<14}{'큰명령 n':>10}{'상대오차 평균':>14}{'상대오차 p90':>14}")
    for j in range(3):
        big = np.abs(cmd[:, j]) > 0.3 * np.abs(cmd[:, j]).max()
        if big.sum() < 50:
            continue
        rel = np.abs(err[big, j]) / np.abs(cmd[big, j])
        print(f"{names[j]:<14}{int(big.sum()):10d}{rel.mean()*100:13.1f}%"
              f"{np.percentile(rel,90)*100:13.1f}%")

    # 전진속도 부족이 슬립으로 설명되는가
    fwd = cmd[:, 0] > 0.3 * cmd[:, 0].max()
    s_m = np.abs(slip[fwd]).mean(axis=1)
    print(f"\n전진 명령 구간 (n={int(fwd.sum())}):")
    print(f"  슬립 평균 {s_m.mean():.3f}  p90 {np.percentile(s_m,90):.3f}")
    print(f"  vx 오차 평균 {err[fwd,0].mean():+.4f} m/s  "
          f"(= 명령의 {err[fwd,0].mean()/cmd[fwd,0].mean()*100:+.1f}%)")
    q = np.quantile(s_m, [1/3, 2/3])
    for nm, m in (("슬립 하위1/3", s_m < q[0]), ("중간", (s_m >= q[0]) & (s_m < q[1])),
                  ("슬립 상위1/3", s_m >= q[1])):
        print(f"    {nm:<12} vx 오차 {err[fwd,0][m].mean():+.4f}  "
              f"|omega 오차| {np.abs(err[fwd,2][m]).mean():.4f}  침하 "
              f"{sink[fwd][m].mean()*1000:.1f}mm")
    print(f"\n차체명령 포화 스텝 비율: {np.diff(np.concatenate([[0],R[:,14]])).clip(0,1).mean()*100:.2f}%")
    if a.save:
        np.savez_compressed(a.save, cmd=cmd, twist=tw, slip=slip, sink=sink)
        print(f"저장 -> {a.save}")


if __name__ == "__main__":
    main()
