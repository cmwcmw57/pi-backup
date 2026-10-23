"""Log a zero-residual IK rollout in Isaac Lab; no training or policy loading."""

import argparse
import pathlib

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("/workspace/_verify/pure_ik_isaac.npz"))
parser.add_argument("--scenario", choices=("full", "arc_left", "arc_right"), default="full")
parser.add_argument("--rocker-k", type=float, default=20.0)
parser.add_argument("--rocker-d", type=float, default=0.3)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import ksrc_rover  # noqa: F401,E402
from ik_benchmark_common import DT, schedule  # noqa: E402
from ksrc_rover.rover_env_cfg import RoverResidual3EnvCfg  # noqa: E402

names, commands = schedule(args.scenario)
cfg = RoverResidual3EnvCfg()
cfg.scene.num_envs = 1
cfg.seed = 123
cfg.episode_length_s = (len(names) + 50) * DT
cfg.terminations.excessive_tilt = None
cfg.events.reset_base.params["pose_range"] = {
    "x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)
}
cfg.events.physics_material.params["static_friction_range"] = (1.0, 1.0)
cfg.events.physics_material.params["dynamic_friction_range"] = (1.0, 1.0)
cfg.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
cfg.commands.base_velocity.rel_standing_envs = 0.0
cfg.commands.base_velocity.rel_heading_envs = 0.0

env = gym.make("Isaac-KSRC-Rover-Residual3-v0", cfg=cfg)
raw = env.unwrapped
robot = raw.scene["robot"]
action = raw.action_manager.get_term("rover")
action.rocker_k = args.rocker_k
action.rocker_d = args.rocker_d
env.reset()
rows = {k: [] for k in ("body", "xy", "yaw", "steer_target", "steer_actual", "wheel_target", "wheel_actual", "rocker_actual")}
for command in commands:
    raw.command_manager.get_command("base_velocity")[:] = torch.as_tensor(command, device=raw.device)
    _, _, terminated, truncated, _ = env.step(torch.zeros((1, 3), device=raw.device))
    if bool(terminated[0]) or bool(truncated[0]):
        raise RuntimeError(f"Isaac rollout ended at step {len(rows['body'])}")
    lin = robot.data.root_lin_vel_b[0]
    ang = robot.data.root_ang_vel_b[0]
    quat = robot.data.root_quat_w[0]
    yaw = torch.atan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                      1 - 2 * (quat[2].square() + quat[3].square()))
    sample = {
        "body": torch.stack((lin[0], lin[1], ang[2])),
        "xy": robot.data.root_pos_w[0, :2],
        "yaw": yaw.reshape(1),
        "steer_target": action.processed_actions[0, :4],
        "steer_actual": robot.data.joint_pos[0, action.steer_ids],
        "wheel_target": action.processed_actions[0, 4:],
        "wheel_actual": robot.data.joint_vel[0, action.wheel_ids],
        "rocker_actual": robot.data.joint_pos[0, action.rocker_ids],
    }
    for key, value in sample.items():
        rows[key].append(value.detach().cpu().numpy().copy())

args.out.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(args.out, phase=names, command=commands, dt=DT,
                    rocker_k=args.rocker_k, rocker_d=args.rocker_d,
                    **{key: np.stack(value) for key, value in rows.items()})
print(f"IK_BENCHMARK_OK {args.out} steps={len(names)}", flush=True)
env.close()
app.close()
