"""Export a trained rsl_rl actor and its observation normalizer to TorchScript."""

import argparse
import pathlib

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=("Residual3", "Res8", "Direct8"), required=True)
parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
parser.add_argument("--out", type=pathlib.Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit  # noqa: E402

import ksrc_rover  # noqa: F401,E402
from ksrc_rover.agents.rsl_rl_ppo_cfg import (  # noqa: E402
    RoverDirect8PPORunnerCfg, RoverRes8PPORunnerCfg, RoverResidual3PPORunnerCfg,
)
from ksrc_rover.rover_env_cfg import RoverDirect8EnvCfg, RoverRes8EnvCfg, RoverResidual3EnvCfg  # noqa: E402

classes = {"Residual3": RoverResidual3EnvCfg, "Res8": RoverRes8EnvCfg, "Direct8": RoverDirect8EnvCfg}
cfg = classes[args.task]()
cfg.scene.num_envs = 1
agent_cfg = {
    "Residual3": RoverResidual3PPORunnerCfg,
    "Res8": RoverRes8PPORunnerCfg,
    "Direct8": RoverDirect8PPORunnerCfg,
}[args.task]()
env = RslRlVecEnvWrapper(
    gym.make(f"Isaac-KSRC-Rover-{args.task}-v0", cfg=cfg), clip_actions=agent_cfg.clip_actions
)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(str(args.checkpoint.resolve()))
policy = runner.alg.policy
normalizer = getattr(policy, "actor_obs_normalizer", None)
out = args.out or args.checkpoint.resolve().parent / "exported" / "policy.pt"
out.parent.mkdir(parents=True, exist_ok=True)
export_policy_as_jit(policy, normalizer=normalizer, path=str(out.parent), filename=out.name)
print(f"EXPORT_OK {out}", flush=True)
env.close()
app.close()
