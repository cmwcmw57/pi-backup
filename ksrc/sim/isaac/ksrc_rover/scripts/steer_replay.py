"""조향 계단 재생: 실기 STEP 측정(pi/common/sysid_uart.py steer)과 같은 계단을 Isaac 에 넣고 비교.

서보 모델(mdp/actions.py _servo_step: 지연 → 데드밴드 → 사다리꼴 7.7 rad/s²) + PhysX PD 추종까지
포함한 조향각이 실측(2026-10-13, 서보 4개 평균, 바퀴 띄움)과 맞는지 본다.  기본은 몸체를 공중에
고정(바퀴 띄움과 같은 조건), --ground 면 바닥에 놓고 정지 조향(S4 비교용).
DR 1단계로 돌려 서보는 공칭값(지연 34 ms, 데드밴드 0.25°, 오프셋 0)이다.

예:
  bash run.sh /workspace/scripts/steer_replay.py --headless
  bash run.sh /workspace/scripts/steer_replay.py --headless --ground --out /workspace/logs/steer_replay_ground.npz
"""

import argparse

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--ground", action="store_true", help="바닥에 놓고 (기본: 몸체 공중 고정 = 바퀴 띄움)")
parser.add_argument("--hold", type=float, default=1.2, help="계단마다 유지 시간 [s]")
parser.add_argument("--out", default="/workspace/logs/steer_replay.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app

import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import ksrc_rover  # noqa: F401,E402
import ksrc_rover.rover_env_cfg as rcfg  # noqa: E402

ANGLES = (0, 90, 0, -90, 0, 45, 0, 15, 0, 5, 0, 3, 0)          # sysid_uart.py STEER_ANGLES
# 실측 (ACC 50, 서보 1–4 평균): 크기 -> (10→90% ms, 도달 ms = ±1° 에 들어가 100 ms 유지한 시작 시각)
REAL = {90: (502, 873), 45: (356, 606), 15: (206, 338), 5: (126, 184), 3: (96, 131)}

rcfg.DR_STAGE = 1
rcfg.TERRAIN_KIND = "plane"
rcfg.WHEEL_MU = "off"
if not args.ground:
    rcfg.SPAWN_Z = 0.5
cfg = rcfg.RoverRes8EnvCfg_PLAY()
cfg.scene.num_envs = 1
n_steps = int(len(ANGLES) * args.hold / 0.02) + 10
cfg.episode_length_s = n_steps * 0.02 + 1.0
if not args.ground:
    import isaaclab.sim as sim_utils
    ap = cfg.scene.robot.spawn.articulation_props or sim_utils.ArticulationRootPropertiesCfg()
    ap.fix_root_link = True
    cfg.scene.robot.spawn.articulation_props = ap
c = cfg.commands.base_velocity
c.rel_standing_envs = 1.0
c.resampling_time_range = (1e6, 1e6)

import traceback  # noqa: E402
import sys  # noqa: E402


def main():
    global log_t
    env = gym.make("Isaac-KSRC-Rover-Res8-Play-v0", cfg=cfg)
    u = env.unwrapped
    robot = u.scene["robot"]
    term = u.action_manager.get_term("rover")
    goal = torch.zeros(1, 4, device=u.device)


    def process_actions(actions):          # 정책·IK 대신 고정 조향 목표, 바퀴 0
        term._raw[:] = 0.0
        term._processed[:, :4] = goal
        term._processed[:, 4:] = 0.0
        term._limit_steering()


    log_t, log_q, log_g = [], [], []
    _apply = term.apply_actions


    def apply_actions():                   # 물리 스텝(5 ms)마다 기록
        log_t.append(len(log_t) * u.physics_dt)
        log_q.append(robot.data.joint_pos[0, term.steer_ids].cpu().numpy().copy())
        log_g.append(goal[0].cpu().numpy().copy())
        _apply()


    term.process_actions = process_actions
    term.apply_actions = apply_actions
    env.reset()
    log_t.clear(); log_q.clear(); log_g.clear()
    zero = torch.zeros(1, u.action_manager.total_action_dim, device=u.device)
    per = int(args.hold / 0.02)
    t_switch = []
    for ang in ANGLES:
        goal[:] = math.radians(ang)
        t_switch.append(len(log_t) * u.physics_dt)
        for _ in range(per):
            _, _, te, tr, _ = env.step(zero)
            if bool(te[0] | tr[0]):
                print("[warn] 재생 중 리셋 -- 결과 무효")

    t = np.asarray(log_t)
    q = np.degrees(np.asarray(log_q))           # (T, 4)
    g = np.degrees(np.asarray(log_g))
    np.savez(args.out, t=t, q_deg=q, goal_deg=g, t_switch=np.asarray(t_switch), angles=np.asarray(ANGLES),
             ground=args.ground)

    print(f"STEER_REPLAY {'ground' if args.ground else 'lifted'}  (sim = 바퀴 4개 평균, real = 서보 4개 평균)")
    print("  계단        10→90% ms sim/real      도달 ms sim/real     최종 오차 도")
    res = {}
    for i in range(1, len(ANGLES)):
        a0, a1 = ANGLES[i - 1], ANGLES[i]
        mag = abs(a1 - a0)
        sel = (t >= t_switch[i]) & (t < t_switch[i] + args.hold)
        tt = t[sel] - t_switch[i]
        out = []
        for w in range(4):
            x = q[sel, w]
            prog = (x - a0) / (a1 - a0)
            t10 = tt[np.argmax(prog >= 0.1)] if (prog >= 0.1).any() else np.nan
            t90 = tt[np.argmax(prog >= 0.9)] if (prog >= 0.9).any() else np.nan
            inb = np.abs(x - a1) <= 1.0
            arr = np.nan
            for k in np.flatnonzero(inb):            # 펌웨어와 같은 정의: ±1° 에 들어가 100 ms 유지한 시작
                j = k + int(round(0.1 / u.physics_dt))
                if j < len(inb) and inb[k:j + 1].all():
                    arr = tt[k]
                    break
            out.append(((t90 - t10) * 1e3, arr * 1e3, x[-1] - a1))
        m = np.nanmean(np.asarray(out), 0)
        res.setdefault(mag, []).append(m)
        r = REAL.get(mag, (np.nan, np.nan))
        print(f"  {a0:+4d}→{a1:+4d}   {m[0]:6.0f} / {r[0]:4.0f}           {m[1]:6.0f} / {r[1]:4.0f}        {m[2]:+.2f}")
    print("  크기별 평균 (sim/real, 차이):")
    for mag in sorted(res, reverse=True):
        m = np.nanmean(res[mag], 0)
        r = REAL[mag]
        print(f"    {mag:3d}°  10→90% {m[0]:5.0f}/{r[0]} ({m[0] - r[0]:+4.0f})   도달 {m[1]:5.0f}/{r[1]} ({m[1] - r[1]:+4.0f})")
    print(f"  저장 {args.out}")
    env.close()


try:
    main()
except Exception:
    traceback.print_exc()
# Kit 의 app.close() 는 파이썬 버퍼를 비우지 않고 프로세스를 끝낸다 (컨테이너 stdout 은 블록 버퍼)
sys.stdout.flush()
sys.stderr.flush()
app.close()
