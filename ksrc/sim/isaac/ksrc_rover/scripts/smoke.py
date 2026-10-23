"""Create a task, reset once, and take a few zero-action steps. No training."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-KSRC-Rover-Direct8-v0")
parser.add_argument("--steps", type=int, default=2)
parser.add_argument("--drive-smoke", action="store_true", help="command forward wheel speed on Direct8")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import ksrc_rover  # noqa: F401,E402
from ksrc_rover.rover_env_cfg import RoverDirect8EnvCfg, RoverRes8EnvCfg, RoverResidual3EnvCfg  # noqa: E402

cfg_class = {
    "Isaac-KSRC-Rover-Direct8-v0": RoverDirect8EnvCfg,
    "Isaac-KSRC-Rover-Res8-v0": RoverRes8EnvCfg,
    "Isaac-KSRC-Rover-Residual3-v0": RoverResidual3EnvCfg,
}[args.task]
cfg = cfg_class()
cfg.scene.num_envs = 1
env = gym.make(args.task, cfg=cfg)
env.reset()
start_xy = env.unwrapped.scene["robot"].data.root_pos_w[0, :2].clone()
for _ in range(args.steps):
    action = torch.zeros((1, env.unwrapped.action_manager.total_action_dim), device=env.unwrapped.device)
    if args.drive_smoke:
        if args.task != "Isaac-KSRC-Rover-Direct8-v0":
            raise ValueError("--drive-smoke requires Direct8")
        action[:, 4:] = 0.5
    env.step(action)
displacement = (env.unwrapped.scene["robot"].data.root_pos_w[0, :2] - start_xy).norm().item()
wheel_ids = env.unwrapped.action_manager.get_term("rover").wheel_ids
wheel_speed = env.unwrapped.scene["robot"].data.joint_vel[0, wheel_ids].mean().item()
print(f"SMOKE_OK task={args.task} action_dim={env.unwrapped.action_manager.total_action_dim} "
      f"xy_displacement_m={displacement:.4f} mean_wheel_rad_s={wheel_speed:.3f}", flush=True)
env.close()
app.close()
