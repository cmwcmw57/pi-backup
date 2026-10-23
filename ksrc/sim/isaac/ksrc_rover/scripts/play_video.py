"""헤드리스 렌더 롤아웃을 mp4 로 녹화한다 (서버에 떨구고 scp 로 노트북에서 재생).

- `--checkpoint <pt>` 주면 그 정책(결정론 추론), 없으면 zero-action.
  Res8/Residual3 zero-action = 순수 IK 롤아웃, Direct8 zero-action = 정지(의미 약함).
- PLAY cfg 를 쓴다(num_envs 8, physics_material DR off, 지형+비대칭 μ+tilt 종료는 유지)
  → 학습 지형 그대로 보인다. `--num_envs` 로 덮어쓸 수 있다(영상은 2~4 가 깔끔).
- 핵심: 오프스크린 렌더라 **`--enable_cameras` 필요**. `--video` 주면 자동으로 켠다.

예:
  bash run.sh /workspace/scripts/play_video.py \
      --task Isaac-KSRC-Rover-Direct8-v0 \
      --checkpoint /workspace/logs/rsl_rl/ksrc_rover_direct8/<run>/model_2000.pt \
      --num_envs 4 --video --video_length 500 --headless --enable_cameras
  → mp4 가 <checkpoint 디렉토리>/videos/ (없으면 /workspace/logs/videos/) 에 생긴다.
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-KSRC-Rover-Direct8-v0")
parser.add_argument("--checkpoint", default=None, help="정책 .pt 경로. 없으면 zero-action")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--video", action="store_true", help="mp4 녹화(엔진 enable_cameras 자동 on)")
parser.add_argument("--video_length", type=int, default=500, help="녹화 스텝 수(제어스텝=0.02s)")
parser.add_argument("--wheel-seed", type=int, default=777, help="평가용 홀드아웃 μ seed")
parser.add_argument("--out-dir", default=None, help="mp4 저장 폴더(기본: 체크포인트 옆 videos/)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# 녹화하려면 오프스크린 렌더(카메라)가 반드시 켜져 있어야 한다.
if args.video:
    args.enable_cameras = True

app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402

import ksrc_rover  # noqa: F401,E402
import ksrc_rover.rover_env_cfg as rcfg  # noqa: E402

# 홀드아웃 μ seed 를 cfg 상수에 주입 (env 생성 전).
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

# 보기 좋은 카메라: env 0 를 비스듬히 내려다본다.
cfg.viewer.resolution = (1280, 720)
cfg.viewer.origin_type = "env"
cfg.viewer.env_index = 0
cfg.viewer.eye = (3.5, 3.5, 2.5)
cfg.viewer.lookat = (0.0, 0.0, 0.3)

# mp4 저장 위치: 체크포인트 옆 videos/ (없으면 기본 폴더).
if args.out_dir:
    video_dir = args.out_dir
elif args.checkpoint:
    video_dir = os.path.join(os.path.dirname(args.checkpoint), "videos")
else:
    video_dir = "/workspace/logs/videos"
os.makedirs(video_dir, exist_ok=True)

env = gym.make(PLAY_TASK[base_task], cfg=cfg, render_mode="rgb_array")

if args.video:
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_dir,
        step_trigger=lambda step: step == 0,   # 시작부터 한 클립
        video_length=args.video_length,
        name_prefix=f"{base_task.split('-')[-2]}",
        disable_logger=True,
    )
    print(f"RECORDING -> {video_dir}  length={args.video_length} steps", flush=True)

import torch  # noqa: E402

# --- 정책 로드 (없으면 zero-action) ---
policy = None
if args.checkpoint:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    from ksrc_rover.agents.rsl_rl_ppo_cfg import RoverPPORunnerCfg

    env = RslRlVecEnvWrapper(env)
    device = env.unwrapped.device
    runner = OnPolicyRunner(env, RoverPPORunnerCfg().to_dict(), log_dir=None, device=str(device))
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=device)
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
else:
    device = env.unwrapped.device
    obs, _ = env.reset()

action_dim = env.unwrapped.action_manager.total_action_dim
ne = args.num_envs

for _ in range(args.video_length):
    if policy is not None:
        with torch.inference_mode():
            action = policy(obs)
    else:
        action = torch.zeros(ne, action_dim, device=device)
    step_out = env.step(action)
    obs = step_out[0]

print("PLAY_VIDEO_DONE", flush=True)
env.close()
app.close()
