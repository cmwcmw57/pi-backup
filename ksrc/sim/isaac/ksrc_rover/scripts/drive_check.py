"""Wheel drive check (2026-10-13): how well wheels follow the speed target, "isaac" vs "pi" drive.

IK driving (Res8 zero action) with the training command distribution, DR stage 2.  Per physics step:
  |w - w_target|        wheel speed vs the target the Nucleo got (after the command delay)  [rad/s]
  stuck                 |target| > 0.3 but |w| < 0.05 (stiction holding a wheel)
  duty sat              |duty| >= 799 ("pi" only)
and the body velocity error, so a drive model that tracks wheels worse shows up in the body.

  bash run.sh /workspace/scripts/drive_check.py --headless
  bash run.sh /workspace/scripts/drive_check.py --headless --set DRIVE_MODEL='"pi"'
"""

import argparse
import ast
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=500, help="policy steps (20 ms)")
parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import traceback  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import ksrc_rover  # noqa: F401,E402
import ksrc_rover.rover_env_cfg as rcfg  # noqa: E402


def stats(name, x):
    x = x.flatten().float()
    print(f"DRIVE_CHECK {name:28s} 평균 {x.mean():8.4f}  p95 {x.quantile(0.95):8.4f}  최대 {x.max():8.4f}", flush=True)


def main():
    for kv in args.set:
        k, v = kv.split("=", 1)
        setattr(rcfg, k, ast.literal_eval(v))
    cfg = rcfg.RoverRes8EnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = gym.make("Isaac-KSRC-Rover-Res8-v0", cfg=cfg)
    u = env.unwrapped
    robot = u.scene["robot"]
    term = u.action_manager.get_term("rover")
    print(f"DRIVE_CHECK drive={term.drive} envs={u.num_envs}", flush=True)
    log = {"err": [], "w_t": [], "w": [], "duty": [], "ep": [], "gate": []}
    _apply = term.apply_actions

    def apply_actions():
        _apply()
        log["w_t"].append(term._processed[:, 4:].clone())
        log["w"].append(robot.data.joint_vel[:, term.wheel_ids].clone())
        log["duty"].append(term.duty.clone())
        log["ep"].append(u.episode_length_buf.clone())
        log["gate"].append(term.gate_hold.clone())

    term.apply_actions = apply_actions
    env.reset()
    zero = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
    exy = []
    for _ in range(args.steps):
        env.step(zero)
        cmd = u.command_manager.get_command("base_velocity")
        v = robot.data.root_lin_vel_b[:, :2]
        exy.append(((cmd[:, :2] - v).norm(dim=-1))[u.episode_length_buf > 50])
    L = {k: torch.stack(v) for k, v in log.items() if v}
    m = (L["ep"] > 50).unsqueeze(-1).expand_as(L["w"])        # 1 s after reset
    wt, w = L["w_t"], L["w"]
    stats("|w - w_target| [rad/s]", (w - wt).abs()[m])
    big = m & (wt.abs() > 0.3)
    stats("|w - w_target| (|w_t|>0.3)", (w - wt).abs()[big])
    print(f"DRIVE_CHECK stuck (|w_t|>0.3, |w|<0.05)  {((w.abs() < 0.05) & big).sum() / big.sum():.4f}", flush=True)
    small = m & (wt.abs() > 0.02) & (wt.abs() < 0.3)
    if small.any():
        stats("|w - w_t| (0.02<|w_t|<0.3)", (w - wt).abs()[small])
    if term.drive == "pi":
        stats("|duty|", L["duty"].abs()[m])
        print(f"DRIVE_CHECK duty sat (>=799)  {(L['duty'].abs()[m] >= 799).float().mean():.4f}", flush=True)
    stats("body |v_xy err| [m/s]", torch.cat(exy))
    g = L["gate"][L["ep"] > 50].float()
    print(f"DRIVE_CHECK gate={term.gate_on} 보류 비율 {g.mean():.4f}", flush=True)
    env.close()


try:
    main()
except Exception:
    traceback.print_exc()
sys.stdout.flush()
sys.stderr.flush()
app.close()
