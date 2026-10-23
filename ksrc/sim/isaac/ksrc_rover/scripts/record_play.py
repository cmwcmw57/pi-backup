"""Record a policy rollout for viewing in Isaac Sim 6.1 (scripts/replay61.py).

Isaac Sim 5.1's RTX renderer crashes on this server (RTX 5090, driver 595), 6.1 renders
fine but has no Isaac Lab 2.3.2. So: run the policy headless here (exact training
physics), record every link's world pose plus the terrain mesh, and replay the poses
kinematically in 6.1 -- no physics there, only rendering.

Records K robots on distinct terrain tiles (Isaac envs overlap on the same tile).
Output npz: link_paths (B,) relative to the robot root, body_pos (T,K,B,3),
body_quat (T,K,B,4, w x y z), terrain_points (V,3), terrain_faces (F,3), cmd (T,K,3),
vel (T,K,3), mu4 (T,K,4), done (T,K), dt.

  KSRC_GPU=0 bash run.sh /workspace/scripts/record_play.py --task Isaac-KSRC-Rover-Teacher-Res8-v0 \
      --checkpoint /workspace/logs/rsl_rl/<exp>/<run>/model_XXXX.pt --steps 1500 \
      --out /workspace/logs/play/res8.npz --headless
  (no --checkpoint = zero action = pure IK with Res8/Residual3)
"""

import argparse
import functools
import os

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-KSRC-Rover-Res8-v0")
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--robots", type=int, default=8, help="기록할 로버 수 (서로 다른 타일)")
parser.add_argument("--steps", type=int, default=1500, help="정책 스텝 (50 Hz)")
parser.add_argument("--wheel-seed", type=int, default=777)
parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
parser.add_argument("--out", default="/workspace/logs/play/play.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
print = functools.partial(print, flush=True)  # noqa: A001  app.close() drops unflushed stdout

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

os.environ["KSRC_SET"] = "\n".join(args.set)   # before import (obs defaults built at import)
import ksrc_rover  # noqa: F401,E402
import ksrc_rover.rover_env_cfg as rcfg  # noqa: E402

rcfg.WHEEL_SEED = args.wheel_seed
TEACHER = "-Teacher-" in args.task
STUDENT = "MLP" if "-Student-MLP-" in args.task else "GRU" if "-Student-GRU-" in args.task else None
base = args.task.replace("-Play", "").replace("Teacher-", "").replace("Student-MLP-", "").replace("Student-GRU-", "")
mode = base.split("-")[-2]                       # Direct8 / Res8 / Residual3
cfg = getattr(rcfg, f"Rover{mode}EnvCfg_PLAY")()
cfg.scene.num_envs = args.num_envs
env = gym.make(f"Isaac-KSRC-Rover-{mode}-Play-v0", cfg=cfg)
u = env.unwrapped
dev = u.device

policy = None
if args.checkpoint:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import DistillationRunner, OnPolicyRunner

    from ksrc_rover.agents.rsl_rl_ppo_cfg import (RoverPPORunnerCfg, RoverStudentGRURunnerCfg,
                                                  RoverStudentMLPRunnerCfg, RoverTeacherPPORunnerCfg)
    wrapped = RslRlVecEnvWrapper(env)
    if STUDENT:
        rc = RoverStudentGRURunnerCfg() if STUDENT == "GRU" else RoverStudentMLPRunnerCfg()
        runner = DistillationRunner(wrapped, rc.to_dict(), log_dir=None, device=str(dev))
    else:
        rc = RoverTeacherPPORunnerCfg() if TEACHER else RoverPPORunnerCfg()
        runner = OnPolicyRunner(wrapped, rc.to_dict(), log_dir=None, device=str(dev))
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=dev)
    out = wrapped.reset()
    obs = out[0] if isinstance(out, tuple) else out
    step_env = wrapped
else:
    obs, _ = env.reset()
    step_env = env

robot = u.scene["robot"]
# K envs on distinct terrain tiles
terr = u.scene.terrain
tiles = (terr.terrain_levels * 1000 + terr.terrain_types).tolist() if terr.terrain_origins is not None \
    else list(range(u.num_envs))
sel, seen = [], set()
for i, t in enumerate(tiles):
    if t not in seen:
        seen.add(t)
        sel.append(i)
    if len(sel) >= args.robots:
        break
sel_t = torch.as_tensor(sel, device=dev)
print(f"[record] robots on envs {sel} (tiles {[tiles[i] for i in sel]})")

# link prim paths relative to the robot root (same order as body_pos_w)
lp = robot.root_physx_view.link_paths[0]
root = os.path.commonpath(lp) if len(lp) > 1 else os.path.dirname(lp[0])
link_paths = [p[len(root) + 1:] for p in lp]
print(f"[record] robot root {root}, {len(link_paths)} links: {link_paths}")

# terrain mesh straight from the stage (world coordinates)
from pxr import UsdGeom  # noqa: E402
import omni.usd  # noqa: E402

stage = omni.usd.get_context().get_stage()
pts_all, faces_all, off = [], [], 0
for prim in stage.Traverse():
    if prim.GetPath().pathString.startswith("/World/ground") and prim.IsA(UsdGeom.Mesh):
        m = UsdGeom.Mesh(prim)
        pts = np.asarray(m.GetPointsAttr().Get(), dtype=np.float64)
        idx = np.asarray(m.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        cnt = np.asarray(m.GetFaceVertexCountsAttr().Get())
        if not np.all(cnt == 3):
            print(f"[record] skip non-triangle mesh {prim.GetPath()}")
            continue
        xf = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).T   # USD row-vector -> column
        pts = (np.c_[pts, np.ones(len(pts))] @ xf.T)[:, :3]
        pts_all.append(pts)
        faces_all.append(idx.reshape(-1, 3) + off)
        off += len(pts)
terrain_points = np.concatenate(pts_all).astype(np.float32) if pts_all else np.zeros((0, 3), np.float32)
terrain_faces = np.concatenate(faces_all).astype(np.int32) if faces_all else np.zeros((0, 3), np.int32)
print(f"[record] terrain {len(terrain_points)} points, {len(terrain_faces)} triangles")

buf = {k: [] for k in ("body_pos", "body_quat", "cmd", "vel", "mu4", "done")}
nd = u.action_manager.total_action_dim
for k in range(args.steps):
    with torch.inference_mode():
        act = policy(obs) if policy is not None else torch.zeros(u.num_envs, nd, device=dev)
    out = step_env.step(act)
    if len(out) == 5:
        obs, _, term, trunc, _ = out
        done = term | trunc
    else:
        obs, _, done, _ = out
    if policy is not None and STUDENT == "GRU":
        with torch.inference_mode():
            runner.alg.policy.reset(done.bool())
    buf["body_pos"].append(robot.data.body_pos_w[sel_t].cpu().numpy())
    buf["body_quat"].append(robot.data.body_quat_w[sel_t].cpu().numpy())
    buf["cmd"].append(u.command_manager.get_command("base_velocity")[sel_t].cpu().numpy())
    v = torch.cat((robot.data.root_lin_vel_b[:, :2], robot.data.root_ang_vel_b[:, 2:3]), 1)
    buf["vel"].append(v[sel_t].cpu().numpy())
    mu = getattr(u, "_wheel_mu4", None)
    buf["mu4"].append(mu[sel].numpy() if mu is not None else np.full((len(sel), 4), np.nan))
    buf["done"].append(done[sel_t].cpu().numpy().astype(np.int8))
    if k % 250 == 0:
        print(f"[record] step {k}/{args.steps}")

os.makedirs(os.path.dirname(args.out), exist_ok=True)
np.savez_compressed(args.out, link_paths=np.asarray(link_paths), dt=u.step_dt, task=args.task,
                    checkpoint=str(args.checkpoint), terrain_points=terrain_points, terrain_faces=terrain_faces,
                    **{k: np.stack(v).astype(np.float32 if k != "done" else np.int8) for k, v in buf.items()})
print(f"RECORD_SAVED {args.out} T={args.steps} robots={len(sel)}")
env.close()
app.close()
