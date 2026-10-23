"""Pre-run checks for the stage-2 DR code (FINAL_TRAINING_PLAN: teacher-student 첫 실행 전).

1) Drive motor tensors: after randomize_drive_motor, is the torque the actuator writes
   to the sim limited by each env's own voltage-scaled stall torque / speed?
   Compares actuator.applied_effort (what Isaac Lab sends to PhysX) with the DC-motor
   formula evaluated on the per-env tensors, and checks the per-env limits really
   differ and match the privileged record (voltage, damping).
2) Observation delay: with noise off, is the sensor term in the policy obs exactly
   env._obs_delay policy steps old (also across the double compute() per step)?
3) Action delay: does the applied wheel target lag the commanded one by env._act_delay?

  bash run.sh /workspace/scripts/check_dr.py --headless
"""

import argparse
import functools

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=60)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
print = functools.partial(print, flush=True)  # app.close() drops unflushed stdout

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import os  # noqa: E402
os.environ["KSRC_SET"] = "DR_STAGE=2"      # before import: obs defaults are built at import
import ksrc_rover  # noqa: F401,E402
import ksrc_rover.rover_env_cfg as rcfg  # noqa: E402

cfg = rcfg.RoverDirect8EnvCfg_PLAY()
cfg.scene.num_envs = args.num_envs
cfg.observations.policy.enable_corruption = False     # exact values for the delay check
env = gym.make("Isaac-KSRC-Rover-Direct8-Play-v0", cfg=cfg)
u = env.unwrapped
env.reset()
robot = u.scene["robot"]
term = u.action_manager.get_term("rover")
drive = robot.actuators["drive"]
wid = term.wheel_ids
N = u.num_envs
ok = True

# ---- 1) drive motor --------------------------------------------------------------
# Direct8: wheels to +max speed, steering 0 -> large velocity error, torque near limits.
act = torch.zeros(N, 8, device=u.device)
act[:, 4:] = 1.0
priv = u._priv_dr
jidx = drive.joint_indices                   # actuator's joint order (not module order)
worst = sent_err = 0.0
for k in range(args.steps):
    env.step(act)
    # _clip_effort ran in the last substep with the velocity it stored in _joint_vel
    v = drive._joint_vel
    top = torch.clip(drive._saturation_effort * (1 - v / drive.velocity_limit), max=drive.effort_limit)
    bot = torch.clip(drive._saturation_effort * (-1 - v / drive.velocity_limit), min=-drive.effort_limit)
    worst = max(worst, (drive.applied_effort - torch.clip(drive.computed_effort, bot, top)).abs().max().item())
    sent_err = max(sent_err, (robot.data.applied_torque[:, jidx] - drive.applied_effort).abs().max().item())
    if k == 0:   # at rest, full target: limit = per-env stall torque, so different envs saturate differently
        at_rest = drive.applied_effort.clone()
sat = drive._saturation_effort
print(f"[1] stall torque per env/motor: min {sat.min():.3f} max {sat.max():.3f} N m "
      f"(nominal 1.96, voltage {priv[:, 0].min():.3f}-{priv[:, 0].max():.3f})")
ratio = sat / (1.96 * priv[:, 0:1])
print(f"[1] stall/(1.96*voltage) = motor spread: {ratio.min():.3f}-{ratio.max():.3f} (expect 0.95-1.05)")
print(f"[1] damping/0.20 vs privileged: max diff {(drive.damping[:, 0] / 0.20 - priv[:, 1]).abs().max():.2e}")
print(f"[1] applied vs DC-motor clip of computed (per-env tensors): max diff {worst:.2e} N m")
print(f"[1] torque written to sim vs actuator applied: max diff {sent_err:.2e} N m")
print(f"[1] first-step torque per env (target 7.96 rad/s from rest): {at_rest.min():.3f}-{at_rest.max():.3f} N m")
spread_ok = sat.max() - sat.min() > 0.1
c1 = worst < 1e-4 and sent_err < 1e-4 and spread_ok and 0.94 < ratio.min() and ratio.max() < 1.06
print(f"[1] {'PASS' if c1 else 'FAIL'}")
ok &= c1

# ---- 2) observation delay ------------------------------------------------------------
# policy obs layout: per term [t-1, t]; wheel_rate is term 6 (cmd3 gyro3 grav3 steer4 svel4 wheel4 tgt8)
sizes = [3, 3, 3, 4, 4, 4, 8]
off = sum(2 * s for s in sizes[:5])          # start of wheel_rate history block
cur = slice(off + 4, off + 8)                # its newest slot
delay = u._obs_delay.clone()
truth, seen = [], []
act2 = torch.zeros(N, 8, device=u.device)
for k in range(30):
    act2[:, 4:] = 0.8 * torch.sin(torch.tensor(k * 0.4)).item()   # varying wheel speed
    obs, *_ = env.step(act2)
    o = obs["policy"] if isinstance(obs, dict) else obs
    truth.append(robot.data.joint_vel[:, wid].clone())
    seen.append(o[:, cur].clone())
errs = {}
for d in delay.unique().tolist():
    m = delay == d
    e = max((seen[k][m] - truth[k - d][m]).abs().max().item() for k in range(5, 30))
    errs[d] = e
print(f"[2] obs delay steps present: {sorted(errs)}; max |obs - truth(t-d)| per delay: "
      + ", ".join(f"d={d}: {e:.2e}" for d, e in sorted(errs.items())))
c2 = all(e < 1e-5 for e in errs.values())
print(f"[2] {'PASS' if c2 else 'FAIL'}")
ok &= c2

# ---- 3) action delay -------------------------------------------------------------------
ad = u._act_delay.clone()
cmdh, apph = [], []
for k in range(12):
    act2[:, 4:] = (k % 4) / 4.0
    env.step(act2)
    cmdh.append(term.commanded_targets[:, 4:].clone())
    apph.append(term.processed_actions[:, 4:].clone())
errs = {}
for d in ad.unique().tolist():
    m = ad == d
    errs[d] = max((apph[k][m] - cmdh[k - d][m]).abs().max().item() for k in range(4, 12))
print("[3] act delay: " + ", ".join(f"d={d}: {e:.2e}" for d, e in sorted(errs.items())))
c3 = all(e < 1e-5 for e in errs.values())
print(f"[3] {'PASS' if c3 else 'FAIL'}")
ok &= c3

print("CHECK_DR_ALL_PASS" if ok else "CHECK_DR_FAIL")
env.close()
app.close()
