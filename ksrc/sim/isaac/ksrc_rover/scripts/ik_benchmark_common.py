"""Deterministic 50 Hz command schedule for Isaac↔MuJoCo pure-IK comparison."""

import numpy as np

DT = 0.02
PHASES = (
    ("settle", 50, (0.0, 0.0, 0.0)),
    ("forward", 100, (0.18, 0.0, 0.0)),
    ("strafe", 100, (0.0, 0.12, 0.0)),
    ("yaw", 100, (0.0, 0.0, 0.60)),
    ("arc", 100, (0.12, 0.04, 0.45)),
    ("reverse", 100, (-0.15, 0.0, 0.0)),
)


def schedule(scenario="full"):
    if scenario == "full":
        phases = PHASES
    elif scenario == "arc_left":
        phases = (("settle", 50, (0.0, 0.0, 0.0)),
                  ("arc", 200, (0.12, 0.04, 0.45)))
    elif scenario == "arc_right":
        phases = (("settle", 50, (0.0, 0.0, 0.0)),
                  ("arc", 200, (0.12, -0.04, -0.45)))
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    names, commands = [], []
    for name, count, command in phases:
        names.extend([name] * count)
        commands.extend([command] * count)
    return np.asarray(names), np.asarray(commands, dtype=np.float32)
