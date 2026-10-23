"""Launch the version-matched Isaac Lab rsl_rl trainer with KSRC tasks registered.

Extra option (stripped before the stock trainer parses argv):
  --set KEY=VALUE   override a rover_env_cfg CONFIG constant (python literal), repeatable.
                    e.g. --set STUDENT_HISTORY=1 --set DR_STAGE=2
Passed via the KSRC_SET env var; rover_env_cfg applies and echoes them at import.
"""

import os
import runpy
import sys

STOCK = "/opt/IsaacLab/scripts/reinforcement_learning/rsl_rl"
sys.path.insert(0, STOCK)  # stock train.py imports its adjacent cli_args.py

overrides, argv = [], [sys.argv[0]]
it = iter(sys.argv[1:])
for a in it:
    if a == "--set":
        overrides.append(next(it))
    elif a.startswith("--set="):
        overrides.append(a.split("=", 1)[1])
    else:
        argv.append(a)
sys.argv = argv

# rover_env_cfg imports Omniverse modules, so it cannot be imported before AppLauncher;
# it applies KSRC_SET itself when the stock trainer imports it (same process).
os.environ["KSRC_SET"] = "\n".join(overrides)
import ksrc_rover  # noqa: F401,E402

runpy.run_path(f"{STOCK}/train.py", run_name="__main__")
