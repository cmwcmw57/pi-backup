"""Launch the version-matched Isaac Lab rsl_rl player with KSRC tasks registered."""

import runpy
import sys

STOCK = "/opt/IsaacLab/scripts/reinforcement_learning/rsl_rl"
sys.path.insert(0, STOCK)
import ksrc_rover  # noqa: F401,E402

runpy.run_path(f"{STOCK}/play.py", run_name="__main__")
