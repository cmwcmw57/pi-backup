"""Isaac IMU 기본 점검 (2026-10-13): 'PhysX 가속도가 튄다' 가 센서 설정 버그인지 진짜 물리인지.

Isaac Lab Imu 센서를 base_link 에 달고 물리 스텝(5 ms)마다 기록해서:
  1) 정지: 가속도 평균이 (0,0,+9.81) 이고 -f/|f| 가 참 중력과 맞는가 (좌표·부호 버그 검사)
  2) 센서 = 몸체 운동인가: IMU 가속도 vs 루트 속도 5 ms 차분 + g (몸체 좌표), IMU 자이로 vs 루트 각속도
  3) 진동 크기: 5 ms 순간값이 20 ms 평균에서 얼마나 벗어나는가 (자이로 rad/s, 가속도 m/s²),
     가속도계 단독 기울기 오차 (순간값 / 20 ms 평균)
DR 1단계 (센서 잡음 없음).  지형·주행은 인자로:
  bash run.sh /workspace/scripts/imu_sanity.py --headless --terrain plane --stand
  bash run.sh /workspace/scripts/imu_sanity.py --headless --terrain rough_slope
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--terrain", default="plane")
parser.add_argument("--stand", action="store_true", help="정지 명령만 (기본: 학습과 같은 명령 분포로 IK 주행)")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=300, help="정책 스텝 (20 ms)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import traceback  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as mu  # noqa: E402
from isaaclab.sensors import ImuCfg  # noqa: E402

import ksrc_rover  # noqa: F401,E402
import ksrc_rover.rover_env_cfg as rcfg  # noqa: E402


def stats(name, x):
    x = x.flatten()
    print(f"IMU_SANITY {name:34s} 평균 {x.mean():8.4f}  p95 {x.quantile(0.95):8.4f}  최대 {x.max():8.4f}", flush=True)


def main():
    rcfg.DR_STAGE = 1
    rcfg.TERRAIN_KIND = args.terrain
    cfg = rcfg.RoverRes8EnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.scene.imu = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/base_link", update_period=0.0, gravity_bias=(0.0, 0.0, 9.81))
    if args.stand:
        cfg.commands.base_velocity.rel_standing_envs = 1.0
    env = gym.make("Isaac-KSRC-Rover-Res8-v0", cfg=cfg)
    u = env.unwrapped
    robot, imu = u.scene["robot"], u.scene["imu"]
    term = u.action_manager.get_term("rover")
    dt = u.physics_dt
    log = {k: [] for k in ("acc", "gyro", "w_root", "acc_pose", "g_true", "ep")}
    prev_v = [None]
    _apply = term.apply_actions

    def apply_actions():                       # 물리 스텝마다 (직전 스텝 결과가 센서·데이터에 있음)
        d = robot.data
        v = d.root_lin_vel_w.clone()
        if prev_v[0] is not None:
            a_w = (v - prev_v[0]) / dt + torch.tensor([0.0, 0.0, 9.81], device=v.device)
            log["acc_pose"].append(mu.quat_apply_inverse(d.root_quat_w, a_w))
            log["acc"].append(imu.data.lin_acc_b.clone())
            log["gyro"].append(imu.data.ang_vel_b.clone())
            log["w_root"].append(d.root_ang_vel_b.clone())
            log["g_true"].append(d.projected_gravity_b.clone())
            log["ep"].append(u.episode_length_buf.clone())
        prev_v[0] = v
        _apply()

    term.apply_actions = apply_actions
    env.reset()
    zero = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
    for _ in range(args.steps):
        env.step(zero)
    L = {k: torch.stack(v) for k, v in log.items()}        # (T, N, 3)
    T = (L["acc"].shape[0] // 4) * 4
    m = (L["ep"][:T] > 50)                                 # 리셋 1 s 뒤부터, 리셋 스텝 차분 제외
    m = m & torch.roll(m, 1, 0)
    print(f"IMU_SANITY terrain={args.terrain} {'stand' if args.stand else 'drive(IK)'} envs={u.num_envs} "
          f"physics_steps={T}", flush=True)
    acc, gyro, g = L["acc"][:T], L["gyro"][:T], L["g_true"][:T]
    print("IMU_SANITY 가속도 평균 (몸체) " + str((acc[m]).mean(0).cpu().numpy().round(3)), flush=True)
    a_dir = -acc / acc.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    tilt = torch.rad2deg(torch.acos((a_dir * g).sum(-1).clamp(-1, 1)))
    stats("가속도계 단독 기울기 오차 [deg]", tilt[m])
    stats("|IMU 가속 - 속도차분 가속| [m/s2]", (acc - L["acc_pose"][:T]).norm(dim=-1)[m])
    stats("|IMU 자이로 - 루트 각속도| [rad/s]", (gyro - L["w_root"][:T]).norm(dim=-1)[m])
    stats("| |f| - 9.81 | [m/s2]", (acc.norm(dim=-1) - 9.81).abs()[m])
    # 20 ms (물리 4 스텝) 평균과 순간값 차이
    N = acc.shape[1]
    acc4, gyro4, g4 = acc.view(T // 4, 4, N, 3), gyro.view(T // 4, 4, N, 3), g.view(T // 4, 4, N, 3)
    m4 = m.view(T // 4, 4, N).all(1)
    stats("자이로 순간-20ms평균 [rad/s]", (gyro4[:, -1] - gyro4.mean(1)).norm(dim=-1)[m4])
    stats("|자이로| 순간값 [rad/s]", gyro4[:, -1].norm(dim=-1)[m4])
    stats("가속 순간-20ms평균 [m/s2]", (acc4[:, -1] - acc4.mean(1)).norm(dim=-1)[m4])
    am = acc4.mean(1)
    a_dir4 = -am / am.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    stats("20ms평균 가속도계 기울기 오차 [deg]",
          torch.rad2deg(torch.acos((a_dir4 * g4[:, -1]).sum(-1).clamp(-1, 1)))[m4])
    env.close()


try:
    main()
except Exception:
    traceback.print_exc()
sys.stdout.flush()
sys.stderr.flush()
app.close()
