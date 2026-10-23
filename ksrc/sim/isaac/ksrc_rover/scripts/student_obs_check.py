"""Student 관측(실기식 계산)이 참값과 얼마나 다른지 확인 (2026-10-13).

IK 주행(Res8 zero action)으로 env 를 돌리며 student 그룹의 마지막 프레임을 참값과 비교한다:
  gravity  (IMU + 보완 필터)   vs  projected_gravity       [deg]
  wheel    (엔코더 20 ms 차분)  vs  관절 속도               [rad/s]
  steering (서보 4096 격자)     vs  관절 각 − 서보 오프셋   [deg]
DR 2단계 (지연·잡음 포함) 그대로.

  bash run.sh /workspace/scripts/student_obs_check.py --headless --num_envs 64
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--dump", default="", help="관측 중력·참 중력·에피소드 스텝을 npz 로")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import traceback  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import ksrc_rover  # noqa: F401,E402
import ksrc_rover.rover_env_cfg as rcfg  # noqa: E402


def main():
    cfg = rcfg.RoverRes8EnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = gym.make("Isaac-KSRC-Rover-Res8-v0", cfg=cfg)
    u = env.unwrapped
    om = u.observation_manager
    names = om.active_terms["student"]
    dims = om.group_obs_term_dim["student"]
    hist = cfg.observations.student.history_length
    print("student terms:", list(zip(names, dims)), "history", hist, flush=True)
    robot = u.scene["robot"]
    term = u.action_manager.get_term("rover")
    env.reset()
    zero = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
    err = {"gravity": [], "grav_settled": [], "wheel": [], "steer": []}
    dump = {"g_obs": [], "g_true": [], "ep_step": [], "done": []}
    for k in range(args.steps):
        obs, _, te, tr, _ = env.step(zero)
        st = obs["student"]
        # 그룹 = 항목별로 [history × dim] 이 이어붙음 (Isaac Lab flatten_history_dim). 항목별 마지막 프레임
        off, last = 0, {}
        for n, d in zip(names, dims):
            w = d[-1] if isinstance(d, tuple) else d
            per = w // hist if w % hist == 0 and hist > 1 else w
            last[n] = st[:, off + w - per: off + w]
            off += w
        if args.dump:
            for key, v in (("g_obs", last["gravity"]), ("g_true", robot.data.projected_gravity_b),
                           ("ep_step", u.episode_length_buf), ("done", te | tr)):
                dump[key].append(v.cpu().numpy().copy())
        if k < 30:
            continue
        ok = ~(te | tr)
        g_true = robot.data.projected_gravity_b
        cos = (last["gravity"] * g_true).sum(-1).clamp(-1, 1)
        e_g = torch.rad2deg(torch.acos(cos))
        err["gravity"].append(e_g[ok])
        settled = ok & (u.episode_length_buf > 50)          # 리셋 1 s 뒤부터 (공중 생성 → 착지 제외)
        err["grav_settled"].append(e_g[settled])
        err["wheel"].append((last["wheel_rate"] - robot.data.joint_vel[:, term.wheel_ids])[ok].abs().flatten())
        q = robot.data.joint_pos[:, term.steer_ids] - term.servo.offset
        err["steer"].append(torch.rad2deg(last["steering"] - q)[ok].abs().flatten())
    for n, v in err.items():
        v = torch.cat(v)
        print(f"STUDENT_OBS {n:8s} 평균 {v.mean():.3f}  p95 {v.quantile(0.95):.3f}  최대 {v.max():.3f}  (n={v.numel()})",
              flush=True)
    if args.dump:
        import numpy as np
        np.savez_compressed(args.dump, **{k: np.stack(v) for k, v in dump.items()}, dt=u.step_dt)
        print("저장", args.dump, flush=True)
    env.close()


try:
    main()
except Exception:
    traceback.print_exc()
sys.stdout.flush()
sys.stderr.flush()
app.close()
