"""바퀴 각속도 요동 진단: 고정 명령 + pure-IK(Res8 zero-action)에서 목표 vs 실제 바퀴속도.

IK 기준 eval 에서 명령이 일정한데도 바퀴 ω 가 스텝마다 ±수 rad/s 요동(연속 변화 부호반전 63%).
원인 후보(제어기 damping / 지형 접촉 / 조향 scrub / μ)를 조건 하나씩 바꿔 가른다.

예:
  bash run.sh /workspace/scripts/wheel_diag.py --headless --terrain plane --mu off --damping 0.2 --vx 0.15 --wz 0.0 --tag A
"""

import argparse

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--terrain", default="plane", help="rover_env_cfg TERRAIN_KIND (plane/rough_slope/...)")
parser.add_argument("--mu", default="off", help="WHEEL_MU (off/narrow/wide)")
parser.add_argument("--damping", type=float, default=None, help="drive 액추에이터 damping (None=rover_cfg 값)")
parser.add_argument("--vx", type=float, default=0.15)
parser.add_argument("--wz", type=float, default=0.0)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--warmup", type=int, default=50, help="조향 정렬·착지 구간 제외 스텝")
parser.add_argument("--tag", default="")
parser.add_argument("--mu-fixed", type=float, default=None, help="모든 env 바퀴(전 shape) μs 를 이 값으로 고정")
parser.add_argument("--mu-dyn-ratio", type=float, default=1.0, help="고정 μ 일 때 μd = μs × 이 값")
parser.add_argument("--actuator", default="dc", choices=["dc", "implicit"], help="바퀴 모터: DCMotor(명시적) / PhysX 내장 드라이브")
parser.add_argument("--pos-iters", type=int, default=None, help="articulation solver position iteration")
parser.add_argument("--vel-iters", type=int, default=None, help="articulation solver velocity iteration")
parser.add_argument("--solver-type", type=int, default=None, help="0=PGS 1=TGS")
parser.add_argument("--sim-dt", type=float, default=None, help="물리 dt [s] (제어 20ms 유지하도록 decimation 조정)")
parser.add_argument("--armature", type=float, default=None, help="바퀴 joint armature [kg m²] (감속기 반사 관성)")
parser.add_argument("--slope-deg", type=float, default=None, help="경사 고정(+x 방향 오르막), 로봇 yaw=0 고정")
parser.add_argument("--spawn-z", type=float, default=None)
parser.add_argument("--trace", action="store_true", help="env0 리셋 직후부터 시계열 출력")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

import ksrc_rover  # noqa: F401,E402
import ksrc_rover.rover_env_cfg as rcfg  # noqa: E402
from ksrc_rover.utils.contract import DRIVE_EFFORT_NM, MODULE_XY_M, WHEEL_RADIUS_M  # noqa: E402

rcfg.TERRAIN_KIND = args.terrain
rcfg.WHEEL_MU = args.mu
if args.slope_deg is not None:
    rcfg.TERRAIN_OVERRIDES = {"slope_deg": args.slope_deg, "slope_heading_deg": 0.0}
if args.spawn_z is not None:
    rcfg.SPAWN_Z = args.spawn_z
cfg = rcfg.RoverRes8EnvCfg_PLAY()
if args.slope_deg is not None:
    cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (-0.1, 0.1), "yaw": (0.0, 0.0)}
cfg.scene.num_envs = args.num_envs
cfg.episode_length_s = (args.steps + 10) * 0.02          # 진단 중 리셋 없음
c = cfg.commands.base_velocity
c.rel_standing_envs = 0.0
c.resampling_time_range = (1e6, 1e6)
c.ranges.lin_vel_x = (args.vx, args.vx)
c.ranges.lin_vel_y = (0.0, 0.0)
c.ranges.ang_vel_z = (args.wz, args.wz)
if args.damping is not None:
    cfg.scene.robot.actuators["drive"].damping = args.damping
if args.actuator == "implicit":
    from isaaclab.actuators import ImplicitActuatorCfg
    from ksrc_rover.utils.contract import WHEEL_MAX_RAD_S
    d = cfg.scene.robot.actuators["drive"]
    cfg.scene.robot.actuators["drive"] = ImplicitActuatorCfg(
        joint_names_expr=d.joint_names_expr, effort_limit_sim=DRIVE_EFFORT_NM,
        velocity_limit_sim=WHEEL_MAX_RAD_S, stiffness=0.0, damping=d.damping)
if args.armature is not None:
    cfg.scene.robot.actuators["drive"].armature = args.armature
damping = cfg.scene.robot.actuators["drive"].damping
if args.pos_iters is not None or args.vel_iters is not None:
    import isaaclab.sim as sim_utils
    cfg.scene.robot.spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        solver_position_iteration_count=args.pos_iters or 4, solver_velocity_iteration_count=args.vel_iters or 0)
if args.solver_type is not None:
    cfg.sim.physx.solver_type = args.solver_type
if args.sim_dt is not None:
    cfg.decimation = max(1, round(0.02 / args.sim_dt))
    cfg.sim.dt = 0.02 / cfg.decimation
    cfg.sim.render_interval = cfg.decimation

env = gym.make("Isaac-KSRC-Rover-Res8-Play-v0", cfg=cfg)
u = env.unwrapped
robot = u.scene["robot"]
term = u.action_manager.get_term("rover")
env.reset()
if args.mu_fixed is not None:
    mats = robot.root_physx_view.get_material_properties()
    mats[..., 0] = args.mu_fixed
    mats[..., 1] = args.mu_fixed * args.mu_dyn_ratio
    mats[..., 2] = 0.0
    robot.root_physx_view.set_material_properties(mats, torch.arange(args.num_envs, dtype=torch.int32))
zero = torch.zeros(args.num_envs, u.action_manager.total_action_dim, device=u.device)

tgt, act, tau, steer, vel, done = [], [], [], [], [], []
for _ in range(args.steps):
    _, _, te, tr, _ = env.step(zero)
    tgt.append(term.processed_actions[:, 4:].cpu().numpy())
    act.append(robot.data.joint_vel[:, term.wheel_ids].cpu().numpy())
    tau.append(robot.data.applied_torque[:, term.wheel_ids].cpu().numpy())
    steer.append(robot.data.joint_pos[:, term.steer_ids].cpu().numpy())
    v = robot.data.root_lin_vel_b
    vel.append(torch.stack((v[:, 0], v[:, 1], robot.data.root_ang_vel_b[:, 2]), 1).cpu().numpy())
    done.append((te | tr).cpu().numpy())

tgt, act, tau, steer, vel, done = map(np.asarray, (tgt, act, tau, steer, vel, done))
if args.trace:
    print("TRACE env0: step | vx | ω fl fr rl rr | 목표ω fl | τ fl fr rl rr")
    for t in list(range(0, 30)) + list(range(30, args.steps, 50)):
        print(f"  {t:3d} | {vel[t, 0, 0]:+.3f} | " + " ".join(f"{x:+.2f}" for x in act[t, 0])
              + f" | {tgt[t, 0, 0]:+.2f} | " + " ".join(f"{x:+.3f}" for x in tau[t, 0]))
if done.any():
    print(f"[warn] 진단 중 리셋 {int(done.sum())}건 — 해당 env 제외")
if args.slope_deg is not None:
    first = np.where(done.any(0), done.argmax(0), -1)
    print(f"SLOPE {args.slope_deg}° μ={args.mu_fixed}: 넘어짐/리셋 env {int(done.any(0).sum())}/{done.shape[1]}"
          + (f" (첫 리셋 스텝 중앙 {int(np.median(first[first >= 0]))})" if done.any() else ""))
ok = ~done.any(0)
w = slice(args.warmup, None)
tgt, act, tau, steer, vel = (a[w][:, ok] for a in (tgt, act, tau, steer, vel))

dw = np.diff(act, axis=0)
flip = np.mean(np.sign(dw[1:]) != np.sign(dw[:-1]))
xy = np.asarray(MODULE_XY_M)
vx, vy, wz = vel[..., 0:1], vel[..., 1:2], vel[..., 2:3]
v_long = (vx - wz * xy[:, 1]) * np.cos(steer) + (vy + wz * xy[:, 0]) * np.sin(steer)
surf = act * WHEEL_RADIUS_M
slip = (surf - v_long) / np.maximum(np.maximum(np.abs(surf), np.abs(v_long)), 0.05)

print(f"WHEEL_DIAG tag={args.tag} terrain={args.terrain} mu={args.mu} mu_fixed={args.mu_fixed}x{args.mu_dyn_ratio} "
      f"act={args.actuator} arm={args.armature} iters={args.pos_iters}/{args.vel_iters} solver={args.solver_type} dt={cfg.sim.dt} damping={damping} "
      f"cmd=({args.vx},{args.wz}) envs={int(ok.sum())}")
print(f"  목표ω 평균 {np.abs(tgt).mean():.2f} rad/s | 실제ω 평균 {np.abs(act).mean():.2f} | "
      f"|실제-목표| 평균 {np.abs(act - tgt).mean():.2f}")
print(f"  스텝당 |Δω| 중앙 {np.median(np.abs(dw)):.2f} p90 {np.percentile(np.abs(dw), 90):.2f} rad/s | "
      f"Δω 부호반전 {flip:.2f}")
print(f"  토크 포화(|τ|≥{0.98 * DRIVE_EFFORT_NM:.2f}) {np.mean(np.abs(tau) >= 0.98 * DRIVE_EFFORT_NM):.2f} | "
      f"|τ| 평균 {np.abs(tau).mean():.3f} Nm")
print(f"  몸체 vx 평균 {vel[..., 0].mean():.3f} (명령 {args.vx}) std {vel[..., 0].std():.3f} | "
      f"wz 평균 {vel[..., 2].mean():.3f} (명령 {args.wz}) | |slip| 평균 {np.abs(slip).mean():.3f}")
print(f"  env0 바퀴fl ω 20스텝: {np.round(act[:20, 0, 0], 2).tolist()}")
realized = getattr(u, "_wheel_mu_realized", None)
if realized is not None:
    print("  env별: μL(s/d) μR(s/d) | 실제ω/목표 좌 우 | |slip| 좌 우 | vx | |τ| 좌 우")
    ids = np.where(ok)[0]
    for k, i in enumerate(ids):
        r = realized[i]
        if r is None:
            continue
        ratio = np.abs(act[:, k]).mean(0) / np.maximum(np.abs(tgt[:, k]).mean(0), 1e-6)
        sl = np.abs(slip[:, k]).mean(0)
        tq = np.abs(tau[:, k]).mean(0)
        print(f"   {i:2d}: {r['left']['static']:.2f}/{r['left']['dynamic']:.2f} {r['right']['static']:.2f}/{r['right']['dynamic']:.2f}"
              f" | {ratio[[0, 2]].mean():.2f} {ratio[[1, 3]].mean():.2f} | {sl[[0, 2]].mean():.2f} {sl[[1, 3]].mean():.2f}"
              f" | {vel[:, k, 0].mean():.3f} | {tq[[0, 2]].mean():.3f} {tq[[1, 3]].mean():.3f}")
env.close()
app.close()
