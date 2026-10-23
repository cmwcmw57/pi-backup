"""노이즈 끈 결정론 평가 + 원시신호 로깅 (마찰 보정 분석용).

- `--checkpoint <pt>` 주면 그 정책(결정론 추론), 없으면 **zero-action = pure-IK 베이스라인**
  (Res8/Residual3 에서 잔차 0 = 펌웨어 IK 그대로). Direct8 zero-action 은 "정지" 라 베이스라인
  의미가 약하니, pure-IK 베이스라인은 Res8 또는 Residual3 로 돌린다.
- 비대칭 μ 는 CONFIG(WHEEL_MU) 를 따르고, 평가는 `--wheel-seed`(홀드아웃, 학습 4321 과 다르게)
  로 고정해 정책/베이스라인이 **같은 μ 열**을 보게 한다(짝비교 가능).
- 에피소드/스텝 원시신호만 npz 로 덤프한다. 슬립·오차·드리프트·보정량은 eval_analyze.py 에서 계산.

예:
  bash run.sh /workspace/scripts/eval_log.py --task Isaac-KSRC-Rover-Res8-v0 --num_envs 64 \
      --steps 2000 --wheel-seed 777 --out /workspace/logs/eval/res8_pureik.npz --headless
  (+ --checkpoint logs/rsl_rl/ksrc_rover_res8/<run>/model_XXXX.pt  로 정책 평가)
"""

import argparse
import os

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-KSRC-Rover-Res8-v0")
parser.add_argument("--checkpoint", default=None, help="정책 .pt 경로. 없으면 zero-action(pure IK) 베이스라인")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--wheel-seed", type=int, default=777, help="평가용 홀드아웃 μ seed")
parser.add_argument("--seed", type=int, default=0,
                    help="env seed (시작 위치·명령·DR·μ 재추첨 시각). 고정해야 IK/정책 비교가 같은 조건이 된다")
parser.add_argument("--out", default="/workspace/logs/eval/eval.npz")
parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="rover_env_cfg CONFIG 상수 덮어쓰기(파이썬 리터럴). 예: --set MU_PER='\"wheel\"' --set MU_RANGE='(0.1,0.6)'")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

# --set must reach rover_env_cfg before import: obs-group defaults (STUDENT_HISTORY,
# DR_STAGE noise switch) are instantiated at import time. Same path as train.py.
import os  # noqa: E402
os.environ["KSRC_SET"] = "\n".join(args.set)
import ksrc_rover  # noqa: F401,E402
import ksrc_rover.rover_env_cfg as rcfg  # noqa: E402
from ksrc_rover.utils.contract import MODULE_XY_M, WHEEL_RADIUS_M  # noqa: E402

# 홀드아웃 μ seed 를 cfg 상수에 주입 (env 생성 전).
rcfg.WHEEL_SEED = args.wheel_seed
import ast  # noqa: E402
for kv in args.set:
    key, val = kv.split("=", 1)
    if not hasattr(rcfg, key):
        raise SystemExit(f"--set: rover_env_cfg 에 {key} 없음")
    setattr(rcfg, key, ast.literal_eval(val))
    print(f"[set] {key} = {getattr(rcfg, key)!r}", flush=True)

PLAY_CFG = {
    "Isaac-KSRC-Rover-Direct8-v0": rcfg.RoverDirect8EnvCfg_PLAY,
    "Isaac-KSRC-Rover-Res8-v0": rcfg.RoverRes8EnvCfg_PLAY,
    "Isaac-KSRC-Rover-Residual3-v0": rcfg.RoverResidual3EnvCfg_PLAY,
}
PLAY_TASK = {
    "Isaac-KSRC-Rover-Direct8-v0": "Isaac-KSRC-Rover-Direct8-Play-v0",
    "Isaac-KSRC-Rover-Res8-v0": "Isaac-KSRC-Rover-Res8-Play-v0",
    "Isaac-KSRC-Rover-Residual3-v0": "Isaac-KSRC-Rover-Residual3-Play-v0",
}
base_task = args.task.replace("-Play", "")
# Teacher/Student tasks share the env; only the runner (obs groups, network) differs.
TEACHER = "-Teacher-" in base_task
STUDENT = "MLP" if "-Student-MLP-" in base_task else "GRU" if "-Student-GRU-" in base_task else None
base_task = base_task.replace("Teacher-", "").replace("Student-MLP-", "").replace("Student-GRU-", "")
cfg = PLAY_CFG[base_task]()
cfg.scene.num_envs = args.num_envs
cfg.seed = args.seed

env = gym.make(PLAY_TASK[base_task], cfg=cfg)  # teacher: same env, privileged group always present
unwrapped = env.unwrapped
device = unwrapped.device



def reseed():
    # Policy path consumes RNG (network init) and resets once more (wrapper __init__) than the
    # zero-action path. Equalize the reset count and reseed right before the final reset, so IK
    # and every checkpoint start from the same spawn / commands / DR / mu draws.
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


# --- 정책 로드 (없으면 zero-action) ---
policy = None
if args.checkpoint:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    from rsl_rl.runners import DistillationRunner

    from ksrc_rover.agents.rsl_rl_ppo_cfg import (RoverPPORunnerCfg, RoverStudentGRURunnerCfg,
                                                  RoverStudentMLPRunnerCfg, RoverTeacherPPORunnerCfg)

    wrapped = RslRlVecEnvWrapper(env)
    if STUDENT:
        # Student checkpoints hold student + teacher weights; inference runs the student only.
        runner_cfg = RoverStudentGRURunnerCfg() if STUDENT == "GRU" else RoverStudentMLPRunnerCfg()
        runner = DistillationRunner(wrapped, runner_cfg.to_dict(), log_dir=None, device=str(device))
    else:
        runner_cfg = RoverTeacherPPORunnerCfg() if TEACHER else RoverPPORunnerCfg()
        runner = OnPolicyRunner(wrapped, runner_cfg.to_dict(), log_dir=None, device=str(device))
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=device)
    reseed()
    reset_out = wrapped.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    step_env = wrapped
else:
    env.reset()          # same count as RslRlVecEnvWrapper.__init__ + wrapped.reset()
    reseed()
    obs, _ = env.reset()
    step_env = env

robot = unwrapped.scene["robot"]
action_term = unwrapped.action_manager.get_term("rover")
steer_ids = action_term.steer_ids
wheel_ids = action_term.wheel_ids
action_dim = unwrapped.action_manager.total_action_dim
ne = args.num_envs

buf = {k: [] for k in ("cmd", "vel", "steer", "wheel_w", "root_pos", "root_quat", "action", "ang", "mu_l", "mu_r", "mu4", "done", "duty", "pi_stall", "pi_enc_dir", "gate_hold")}


def wheel_mu_arrays():
    realized = getattr(unwrapped, "_wheel_mu_realized", None)
    left = np.full(ne, np.nan, dtype=np.float32)
    right = np.full(ne, np.nan, dtype=np.float32)
    if realized is not None:
        for i in range(ne):
            rec = realized[i] if i < len(realized) else None
            if rec is not None:
                left[i] = rec["left"]["static"]
                right[i] = rec["right"]["static"]
    return left, right


for _ in range(args.steps):
    if policy is not None:
        with torch.inference_mode():
            action = policy(obs)
    else:
        action = torch.zeros(ne, action_dim, device=device)

    cmd = unwrapped.command_manager.get_command("base_velocity").clone()
    lin = robot.data.root_lin_vel_b[:, :3].clone()
    ang = robot.data.root_ang_vel_b[:, 2:3].clone()
    vel = torch.cat((lin[:, :2], ang), dim=1)  # [vx, vy, wz] body
    ang_b = robot.data.root_ang_vel_b.clone()  # [wx, wy, wz] body (roll/pitch 흔들림 참고용)
    steer = robot.data.joint_pos[:, steer_ids].clone()
    wheel_w = robot.data.joint_vel[:, wheel_ids].clone()
    root_pos = robot.data.root_pos_w[:, :3].clone()
    root_quat = robot.data.root_quat_w.clone()

    step_out = step_env.step(action)
    if len(step_out) == 5:
        obs, _, term, trunc, _ = step_out
        done = (term | trunc)
    else:  # rsl_rl wrapper: obs, rew, done, info
        obs, _, done, _ = step_out
    if policy is not None and STUDENT == "GRU":
        with torch.inference_mode():   # hidden state was created in inference mode
            runner.alg.policy.reset(done.bool())   # clear GRU memory of envs that just reset

    ml, mr = wheel_mu_arrays()
    buf["cmd"].append(cmd.cpu().numpy())
    buf["vel"].append(vel.cpu().numpy())
    buf["steer"].append(steer.cpu().numpy())
    buf["wheel_w"].append(wheel_w.cpu().numpy())
    buf["root_pos"].append(root_pos.cpu().numpy())
    buf["root_quat"].append(root_quat.cpu().numpy())
    buf["action"].append(action.detach().cpu().numpy())
    buf["ang"].append(ang_b.cpu().numpy())
    buf["mu_l"].append(ml)
    buf["mu_r"].append(mr)
    mu4 = getattr(unwrapped, "_wheel_mu4", None)
    buf["mu4"].append(mu4.numpy().copy() if mu4 is not None else np.full((ne, 4), np.nan, dtype=np.float32))
    buf["done"].append(done.cpu().numpy().astype(np.int8))
    # wheel drive (2026-10-13): duty and PI protection state per wheel ("pi"/"openloop" drives)
    buf["duty"].append(action_term.duty.cpu().numpy().copy())
    buf["pi_stall"].append(action_term.pi.stall.cpu().numpy().copy())
    buf["pi_enc_dir"].append(action_term.pi.enc_dir.cpu().numpy().copy())
    buf["gate_hold"].append(action_term.gate_hold.cpu().numpy().copy())

os.makedirs(os.path.dirname(args.out), exist_ok=True)
np.savez_compressed(
    args.out,
    task=base_task,
    is_baseline=(policy is None),
    wheel_seed=args.wheel_seed,
    wheel_radius_m=WHEEL_RADIUS_M,
    drive=action_term.drive,
    module_xy=np.asarray(MODULE_XY_M),
    **{k: np.asarray(v) for k, v in buf.items()},
)
print(f"EVAL_SAVED {args.out}  steps={args.steps} envs={ne} baseline={policy is None}", flush=True)
env.close()
app.close()
