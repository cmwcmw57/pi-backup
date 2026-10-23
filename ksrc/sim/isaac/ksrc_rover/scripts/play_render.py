"""정책 롤아웃을 replicator 카메라로 렌더해 PNG 프레임 덤프 (뷰포트/RecordVideo 회피).

이 서버의 RTX 뷰포트 하이드라 엔진은 헤드리스에서 createHydraEngine 세그폴트가 난다.
동료 minimal_rtx.py 가 입증한 경로 = replicator render_product + BasicWriter 는 됨.
그래서 env.render()(뷰포트) 대신 rep 카메라로 현재 상태를 렌더해 rgb PNG 로 떨군다.
프레임 → 로컬에서 ffmpeg 로 mp4.

예:
  bash run.sh /workspace/scripts/play_render.py --task Isaac-KSRC-Rover-Direct8-v0 \
    --checkpoint .../model_1650.pt --num_envs 1 --steps 60 --every 3 \
    --out /workspace/logs/render/B --headless --enable_cameras
"""
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-KSRC-Rover-Direct8-v0")
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--every", type=int, default=2, help="N스텝마다 1프레임 캡처")
parser.add_argument("--wheel-seed", type=int, default=777)
parser.add_argument("--out", default="/workspace/logs/render/out")
parser.add_argument("--cam", default="5,5,3.5", help="카메라 위치 x,y,z")
parser.add_argument("--look", default="0,0,0.2", help="바라보는 점 x,y,z")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True  # render_product 에 필요

app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import ksrc_rover  # noqa: F401,E402
import ksrc_rover.rover_env_cfg as rcfg  # noqa: E402

rcfg.WHEEL_SEED = args.wheel_seed

PLAY_TASK = {
    "Isaac-KSRC-Rover-Direct8-v0": "Isaac-KSRC-Rover-Direct8-Play-v0",
    "Isaac-KSRC-Rover-Res8-v0": "Isaac-KSRC-Rover-Res8-Play-v0",
    "Isaac-KSRC-Rover-Residual3-v0": "Isaac-KSRC-Rover-Residual3-Play-v0",
}
PLAY_CFG = {
    "Isaac-KSRC-Rover-Direct8-v0": rcfg.RoverDirect8EnvCfg_PLAY,
    "Isaac-KSRC-Rover-Res8-v0": rcfg.RoverRes8EnvCfg_PLAY,
    "Isaac-KSRC-Rover-Residual3-v0": rcfg.RoverResidual3EnvCfg_PLAY,
}
base_task = args.task.replace("-Play", "")
cfg = PLAY_CFG[base_task]()
cfg.scene.num_envs = args.num_envs

env = gym.make(PLAY_TASK[base_task], cfg=cfg)
unwrapped = env.unwrapped
device = unwrapped.device

# --- 정책 로드 (없으면 zero-action) ---
policy = None
step_env = env
if args.checkpoint:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    from ksrc_rover.agents.rsl_rl_ppo_cfg import RoverPPORunnerCfg

    step_env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(step_env, RoverPPORunnerCfg().to_dict(), log_dir=None, device=str(device))
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=device)
    reset_out = step_env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
else:
    obs, _ = env.reset()

action_dim = unwrapped.action_manager.total_action_dim
ne = args.num_envs

# --- replicator 카메라 (뷰포트 아님) ---
import omni.replicator.core as rep  # noqa: E402

cx = tuple(float(v) for v in args.cam.split(","))
lx = tuple(float(v) for v in args.look.split(","))
cam = rep.create.camera(position=cx, look_at=lx)
rp = rep.create.render_product(cam, (1280, 720))
writer = rep.WriterRegistry.get("BasicWriter")
os.makedirs(args.out, exist_ok=True)
writer.initialize(output_dir=args.out, rgb=True)
writer.attach([rp])
print(f"RENDER_START out={args.out} cam={cx} look={lx}", flush=True)

frames = 0
for t in range(args.steps):
    if policy is not None:
        with torch.inference_mode():
            action = policy(obs)
    else:
        action = torch.zeros(ne, action_dim, device=device)
    step_out = step_env.step(action)
    obs = step_out[0]
    if t % args.every == 0:
        rep.orchestrator.step(pause_timeline=True, delta_time=0.0)  # 물리 안 돌리고 현재상태만 렌더→PNG
        frames += 1

print(f"RENDER_DONE frames={frames} out={args.out}", flush=True)
env.close()
app.close()
