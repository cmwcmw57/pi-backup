#!/usr/bin/env python3
"""평가 -- 학습 지형이 아니라 **실제 경기장 STL** 에서 재는 게 핵심.

  python3 eval.py --model runs/ppo/final --episodes 30

비교군: 순수 PP+IK, 일률감속 x0.8/x0.6, 경사스케줄, RL 3-D 차체 잔차.
경로추종 성능의 직접 비교에는 compare_policy.py 의 짝비교를 쓴다.
"""
import argparse, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from env import RoverEnv, OBS_PER_FRAME, HISTORY

OBS_DIM = OBS_PER_FRAME * HISTORY

from traction import SlopeScheduledLimit
import unicodedata


def pad(t, w):
    """한글은 터미널에서 2칸을 먹는다. 표시폭 기준으로 맞춘다."""
    d = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in str(t))
    return str(t) + " " * max(w - d, 0)

def _scale_action(e, k):
    """nominal 명령을 **k 배로 스케일**하는 3D 차체 잔차.

    고전 제어가 하는 일은 속도 레벨 조절뿐이고 방향비(=기구학)는 건드리지 않는다.
    a = (k-1)*cmd_nom / lim 이면 cmd_applied = k*cmd_nom 이 된다.
    범위를 넘으면 클립되므로 극단적인 k 는 표현되지 않는다 (그건 잔차 구조의
    의도된 성질이다 -- 기본 컨트롤러에서 멀리 못 간다).
    """
    c = e.cfg
    lim = np.array([c.d_vx_max, c.d_vy_max, c.d_om_max])
    return np.clip((k - 1.0) * e.cmd_nom / lim, -1.0, 1.0)


def make_baselines():
    """(이름, 정책, 상태객체) 목록.  정책은 (obs, env) 를 받는다.

    3D 차체 잔차로 바뀌면서 비교군도 다시 정의했다.  예전 "풀스로틀/고정스로틀" 은
    바퀴 duty 를 직접 밀어붙이는 것이라 3D 구조로는 표현되지 않고, 애초에
    경로추종 비교군으로 적절하지도 않았다.
    """
    sched = SlopeScheduledLimit()

    def base(o, e):                      # 잔차 0 = 순수 Pure Pursuit + IK
        return np.zeros(3)

    def slow08(o, e):                    # 일률 20% 감속
        return _scale_action(e, 0.80)

    def slow06(o, e):                    # 일률 40% 감속
        return _scale_action(e, 0.60)

    def slope(o, e):                     # 피치로 속도 상한 스케줄 (고전 terrain-aware)
        pitch = float(e.hist[-1][16]) * (np.pi / 2)     # 관측의 피치
        k = float(np.abs(sched(np.ones(1), pitch))[0])
        return _scale_action(e, k)

    return [("PP+IK (잔차 0)", base, None),
            ("일률감속 x0.8", slow08, None),
            ("일률감속 x0.6", slow06, None),
            ("경사스케줄", slope, sched)]


def rollout(policy, n, arena, difficulty, episode_s, seed0=1000,
            stateful=None, kind="sand", privileged=False):
    out = []; causes = []
    for i in range(n):
        e = RoverEnv(difficulty=difficulty,
                     episode_s=episode_s, arena_eval=arena, eval_kind=kind,
                     privileged=privileged, seed=seed0 + i)
        o, _ = e.reset(seed=seed0 + i)
        if stateful is not None:
            stateful.reset()
        R = 0.0; term = trunc = False; info = {}
        while not (term or trunc):
            a = np.zeros(3) if policy is None else policy(o, e)
            o, r, term, trunc, info = e.step(a); R += r
        # 전 구간 집계를 쓴다 (종료 스텝 순간값 금지 -- slip 이 3.2배 틀렸다)
        # 성공률뿐 아니라 전 구간 경로 오차·잔차 사용량을 함께 본다.
        out.append((R, info["dist"], info["slip_m"], info["sink_m"],
                    float(info["success"]), float(info["tip"]), float(info["stuck"]),
                    info["f_belly"], info["f_lifted"], info["f_blocked"],
                    info["cmd_sat"], info["a_use"], info["a_p95"],
                    float(info["goals"]), info["e_y_m"], info["e_y_p90"],
                    info["e_psi_m"]))
        causes.append(info["cause"])
    return np.array(out), causes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None)
    p.add_argument("--vecnorm", default=None)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--difficulty", type=float, default=0.8)
    p.add_argument("--episode-s", type=float, default=20.0)
    p.add_argument("--arena", action="store_true", help="실제 경기장 STL 로 평가")
    p.add_argument("--terrain", default="sand", choices=["sand", "rock"],
                   help="실측 지형 종류. sand=규사 경사지형(주), rock=암석 착륙지")
    args = p.parse_args()

    policy = None
    # teacher 정책은 특권정보까지 붙은 관측(1193)을 받는다.  env 를 1150 으로
    # 만들면 정규화 단계에서 모양이 안 맞아 터진다.  모델에서 읽어 맞춘다.
    # 특권정보는 관측 뒤에 덧붙을 뿐 동역학을 바꾸지 않으므로, 비교군도 같은
    # env 로 돌려야 조건이 완전히 같다.
    priv = False
    if args.model:
        from stable_baselines3 import PPO
        model = PPO.load(args.model, device="cpu")
        priv = int(model.observation_space.shape[0]) > OBS_DIM
        if priv:
            print(f"[eval] teacher 모델 감지 (관측 {model.observation_space.shape[0]}) "
                  f"-> 특권 env 로 평가\n")
        norm = None
        if args.vecnorm:
            from stable_baselines3.common.vec_env import VecNormalize
            import pickle
            with open(args.vecnorm, "rb") as f:
                norm = pickle.load(f)

        def policy(o, e=None):
            if norm is not None:
                o = np.clip((o - norm.obs_rms.mean) / np.sqrt(norm.obs_rms.var + 1e-8),
                            -10, 10).astype(np.float32)
            return model.predict(o, deterministic=True)[0]

    tag = (("규사 경사지형 STL" if args.terrain == "sand" else "암석 착륙지 STL")
           if args.arena else f"랜덤 지형 d={args.difficulty}")
    print(f"평가: {tag},  {args.episodes} 에피소드\n")
    print(f"{pad('변형',20)} {'보상':>9s} {'목표수':>7s} {'이동 m':>8s} {'슬립':>7s} "
          f"{'침하%':>7s} {'성공%':>6s} {'전복%':>6s} {'고착%':>6s}")
    runs = list(make_baselines())
    if policy is not None:
        runs.append(("RL 3-D", policy, None))
    breakdown = {}
    for name, pol, stateful in runs:
        r, causes = rollout(pol, args.episodes, args.arena, args.difficulty,
                            args.episode_s, stateful=stateful,
                            kind=args.terrain, privileged=priv)
        breakdown[name] = (r, causes)
        print(f"{pad(name,20)} {r[:,0].mean():+9.1f} {r[:,13].mean():7.2f} "
              f"{r[:,1].mean():8.3f} {r[:,2].mean():7.3f} "
              f"{r[:,3].mean()*100:7.1f} {r[:,4].mean()*100:6.0f} {r[:,5].mean()*100:6.0f} "
              f"{r[:,6].mean()*100:6.0f}")

    # --- 왜 멈췄나 --------------------------------------------------------
    print(f"\n{pad('변형',20)}  --- 멈춘 원인 (에피소드 수) ---        "
          f"--- 에피소드 중 상태 비율 ---")
    print(f"{pad('',20)} {'하중상실':>9s} {'배접촉':>8s} {'막힘':>6s} {'슬립고착':>9s}   "
          f"{'배접촉':>8s} {'하중상실':>9s} {'막힘':>6s}")
    for name, (r, causes) in breakdown.items():
        cnt = {k: causes.count(k) for k in
               ("배걸림-하중상실", "배걸림-접촉", "막힘", "슬립고착")}
        print(f"{pad(name,20)} {cnt['배걸림-하중상실']:9d} {cnt['배걸림-접촉']:8d} "
              f"{cnt['막힘']:6d} {cnt['슬립고착']:9d}   "
              f"{r[:,7].mean()*100:7.1f}% {r[:,8].mean()*100:8.1f}% {r[:,9].mean()*100:5.1f}%")

    # --- 경로 추종 정확도 -------------------------------------------------
    print(f"\n{pad('변형',20)} {'횡오차 평균 mm':>16s} {'횡오차 p90 mm':>16s} "
          f"{'기수오차 평균 deg':>19s}")
    for name, (r, _) in breakdown.items():
        print(f"{pad(name,20)} {r[:,14].mean()*1000:16.1f} "
              f"{r[:,15].mean()*1000:16.1f} {r[:,16].mean():19.1f}")

    # --- 현재 3D 잔차의 실제 사용량 ----------------------------------------
    print(f"\n{pad('변형',20)} {'차체명령 포화':>14s} {'잔차 사용률':>13s} "
          f"{'잔차 크기 p95':>14s}")
    for name, (r, _) in breakdown.items():
        print(f"{pad(name,20)} {r[:,10].mean()*100:13.1f}% "
              f"{r[:,11].mean()*100:12.1f}% {r[:,12].mean()*100:13.1f}%")


if __name__ == "__main__":
    main()
