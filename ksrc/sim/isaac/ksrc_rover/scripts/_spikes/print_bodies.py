"""로버 USD 의 body/joint 이름을 찍는다 (좌우 wheel material 배선용)."""
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p); a = p.parse_args()
app = AppLauncher(a).app
import gymnasium as gym           # noqa: E402
import ksrc_rover                 # noqa: F401,E402
from ksrc_rover.rover_env_cfg import RoverDirect8EnvCfg  # noqa: E402
cfg = RoverDirect8EnvCfg(); cfg.scene.num_envs = 1
env = gym.make("Isaac-KSRC-Rover-Direct8-v0", cfg=cfg); env.reset()
r = env.unwrapped.scene["robot"]
print("BODY_NAMES", list(r.body_names), flush=True)
print("JOINT_NAMES", list(r.joint_names), flush=True)
env.close(); app.close()
