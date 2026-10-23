"""KSRC rover task registration. Keep imports here safe before AppLauncher."""

import gymnasium as gym

from ksrc_rover.agents import rsl_rl_patches  # noqa: F401,E402  (rsl-rl 3.1.2 GRU-student fix)

for _mode, _cfg in (
    ("Residual3", "RoverResidual3EnvCfg"),
    ("Res8", "RoverRes8EnvCfg"),
    ("Direct8", "RoverDirect8EnvCfg"),
):
    for _suffix, _class in (("", _cfg), ("-Play", f"{_cfg}_PLAY")):
        gym.register(
            id=f"Isaac-KSRC-Rover-{_mode}{_suffix}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"ksrc_rover.rover_env_cfg:{_class}",
                "rsl_rl_cfg_entry_point": f"ksrc_rover.agents.rsl_rl_ppo_cfg:Rover{_mode}PPORunnerCfg",
            },
        )

# Teachers: same env (it always exposes the "privileged" obs group); only the runner
# differs -- it feeds that group to the actor and critic.
for _mode, _cfg in (
    ("Residual3", "RoverResidual3EnvCfg"),
    ("Res8", "RoverRes8EnvCfg"),
    ("Direct8", "RoverDirect8EnvCfg"),
):
    for _suffix, _class in (("", _cfg), ("-Play", f"{_cfg}_PLAY")):
        gym.register(
            id=f"Isaac-KSRC-Rover-Teacher-{_mode}{_suffix}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"ksrc_rover.rover_env_cfg:{_class}",
                "rsl_rl_cfg_entry_point": f"ksrc_rover.agents.rsl_rl_ppo_cfg:RoverTeacher{_mode}PPORunnerCfg",
            },
        )

# Students: distilled from the same-mode teacher. Same env (it always exposes the
# "student" obs group); MLP uses STUDENT_HISTORY steps, GRU should run with 1.
for _mode, _cfg in (
    ("Residual3", "RoverResidual3EnvCfg"),
    ("Res8", "RoverRes8EnvCfg"),
    ("Direct8", "RoverDirect8EnvCfg"),
):
    for _arch in ("MLP", "GRU"):
        for _suffix, _class in (("", _cfg), ("-Play", f"{_cfg}_PLAY")):
            gym.register(
                id=f"Isaac-KSRC-Rover-Student-{_arch}-{_mode}{_suffix}-v0",
                entry_point="isaaclab.envs:ManagerBasedRLEnv",
                disable_env_checker=True,
                kwargs={
                    "env_cfg_entry_point": f"ksrc_rover.rover_env_cfg:{_class}",
                    "rsl_rl_cfg_entry_point": f"ksrc_rover.agents.rsl_rl_ppo_cfg:RoverStudent{_arch}{_mode}RunnerCfg",
                },
            )
