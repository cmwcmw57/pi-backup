"""Export a distilled student for the Pi: TorchScript + input-layout JSON.

The exported module contains the student's observation normalizer, so the Pi feeds raw
sensor values in the layout written to <out>.json:
  terms in order (command, gyro, gravity, steering, steering_rate, wheel_rate,
  applied_targets), each with STUDENT_HISTORY steps, oldest first (Isaac Lab
  flatten_history_dim), units as in simulation (m/s, rad, rad/s, body frame).
MLP:  forward(obs[1, D]) -> action[1, A]
GRU:  forward(obs[1, D], h[layers, 1, H]) -> (action[1, A], h_next); start with h = 0.

  bash run.sh /workspace/scripts/export_student.py --task Isaac-KSRC-Rover-Student-MLP-Res8-v0 \
      --checkpoint /workspace/logs/rsl_rl/ksrc_rover_teacher_res8/<run>/model_999.pt \
      --set STUDENT_HISTORY=5 --out /workspace/logs/export/res8_mlp_h5 --headless
"""

import argparse
import ast
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out", required=True, help="출력 경로(확장자 없이). <out>.pt, <out>.json")
parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

# --set must reach rover_env_cfg before import: obs-group defaults (STUDENT_HISTORY,
# DR_STAGE noise switch) are instantiated at import time. Same path as train.py.
import os  # noqa: E402
os.environ["KSRC_SET"] = "\n".join(args.set)
import ksrc_rover  # noqa: F401,E402
import ksrc_rover.rover_env_cfg as rcfg  # noqa: E402

for kv in args.set:
    k, v = kv.split("=", 1)
    setattr(rcfg, k, ast.literal_eval(v))

GRU = "-Student-GRU-" in args.task
base = args.task.replace("Student-MLP-", "").replace("Student-GRU-", "").replace("-v0", "")
cfg = getattr(rcfg, {"Isaac-KSRC-Rover-Direct8": "RoverDirect8EnvCfg_PLAY",
                     "Isaac-KSRC-Rover-Res8": "RoverRes8EnvCfg_PLAY",
                     "Isaac-KSRC-Rover-Residual3": "RoverResidual3EnvCfg_PLAY"}[base])()
cfg.scene.num_envs = 1
env = gym.make(base + "-Play-v0", cfg=cfg)

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import DistillationRunner  # noqa: E402

from ksrc_rover.agents.rsl_rl_ppo_cfg import RoverStudentGRURunnerCfg, RoverStudentMLPRunnerCfg  # noqa: E402

wrapped = RslRlVecEnvWrapper(env)
rc = RoverStudentGRURunnerCfg() if GRU else RoverStudentMLPRunnerCfg()
runner = DistillationRunner(wrapped, rc.to_dict(), log_dir=None, device="cpu")
runner.load(args.checkpoint, map_location="cpu")
pol = runner.alg.policy
pol.cpu()
pol.eval()   # rsl_rl StudentTeacher.train() returns None, so no chaining


class MLPExport(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.norm, self.student = p.student_obs_normalizer, p.student

    def forward(self, obs):
        return self.student(self.norm(obs))


class GRUExport(nn.Module):
    def __init__(self, p):
        super().__init__()
        mem = getattr(p, "memory_s", None)
        if mem is None or not hasattr(mem, "rnn"):
            raise SystemExit("StudentTeacherRecurrent 구조가 예상(memory_s.rnn)과 다름 — rsl_rl 버전 확인")
        self.norm, self.rnn, self.student = p.student_obs_normalizer, mem.rnn, p.student

    def forward(self, obs, h):
        y, h_next = self.rnn(self.norm(obs).unsqueeze(0), h)
        return self.student(y.squeeze(0)), h_next


os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
obs_mgr = env.unwrapped.observation_manager
names = obs_mgr.active_terms["student"]
dims = obs_mgr.group_obs_term_dim["student"]
d_total = sum(int(torch.tensor(d).prod()) for d in dims)
with torch.no_grad():
    if GRU:
        mod = GRUExport(pol)
        h0 = torch.zeros(mod.rnn.num_layers, 1, mod.rnn.hidden_size)
        traced = torch.jit.trace(mod, (torch.zeros(1, d_total), h0))
    else:
        mod = MLPExport(pol)
        traced = torch.jit.trace(mod, torch.zeros(1, d_total))
traced.save(args.out + ".pt")
meta = {
    "task": args.task, "checkpoint": args.checkpoint, "recurrent": GRU,
    "student_history": rcfg.STUDENT_HISTORY, "obs_dim": d_total,
    "terms": [{"name": n, "shape": list(d)} for n, d in zip(names, dims)],
    "history_order": "oldest first per term",
    "action_dim": env.unwrapped.action_manager.total_action_dim,
    "control_dt_s": cfg.sim.dt * cfg.decimation,
}
if GRU:
    meta["hidden_shape"] = [mod.rnn.num_layers, 1, mod.rnn.hidden_size]
json.dump(meta, open(args.out + ".json", "w"), indent=2, ensure_ascii=False,
          default=lambda o: o.item() if hasattr(o, "item") else str(o))   # numpy/torch ints
print(f"[export] {args.out}.pt / .json  obs_dim={d_total}", flush=True)
env.close()
app.close()
