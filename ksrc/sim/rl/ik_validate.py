"""Compare batched Isaac torch IK with the firmware C implementation.

Run locally where gcc and torch are available: python3 sim/rl/ik_validate.py
"""

import ctypes as ct
import pathlib
import random
import subprocess
import sys
import tempfile

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
ISAAC_SOURCE = ROOT / "sim/isaac/ksrc_rover/source"
sys.path.insert(0, str(ISAAC_SOURCE))
from ksrc_rover.utils.contract import MODULE_XY_M, STEER_LIMIT_RAD, WHEEL_MAX_M_S  # noqa: E402
from ksrc_rover.utils.ik_torch import BatchedSwerveIK  # noqa: E402


class ModulePos(ct.Structure):
    _fields_ = [("x", ct.c_float), ("y", ct.c_float)]


class State(ct.Structure):
    _fields_ = [("last_angle_rad", ct.c_float), ("last_cmd_rad", ct.c_float), ("initialized", ct.c_int)]


class Command(ct.Structure):
    _fields_ = [("angle_rad", ct.c_float), ("speed_mps", ct.c_float)]


def main():
    with tempfile.TemporaryDirectory() as tmp:
        lib_path = pathlib.Path(tmp) / "libswerve.so"
        subprocess.run(["gcc", "-shared", "-fPIC", "-O2", str(ROOT / "firmware/common/swerve_kinematics.c"),
                        "-lm", "-o", str(lib_path)], check=True)
        lib = ct.CDLL(str(lib_path))
        lib.swerve_ik_compute.argtypes = [ct.c_float, ct.c_float, ct.c_float,
            ct.POINTER(ModulePos), ct.c_float, ct.POINTER(State), ct.POINTER(Command)]
        lib.swerve_fold_to_limit.argtypes = [ct.c_float, ct.c_float, ct.c_float,
            ct.POINTER(State), ct.POINTER(Command)]
        positions = (ModulePos * 4)(*(ModulePos(*xy) for xy in MODULE_XY_M))
        states = (State * 4)()
        out = (Command * 4)()
        ik = BatchedSwerveIK(1, "cpu")
        rng = random.Random(20261008)
        worst = 0.0
        for index in range(1000):
            if index % 20 == 0:
                cmd = (0.0, 0.0, 0.0)
            else:
                cmd = (rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), rng.uniform(-1.5, 1.5))
            lib.swerve_ik_compute(*map(ct.c_float, cmd), positions, ct.c_float(WHEEL_MAX_M_S), states, out)
            lib.swerve_fold_to_limit(ct.c_float(STEER_LIMIT_RAD), ct.c_float(0), ct.c_float(0), states, out)
            th, speed = ik.compute(torch.tensor([cmd], dtype=torch.float32))
            for j in range(4):
                worst = max(worst, abs(th[0, j].item() - out[j].angle_rad),
                            abs(speed[0, j].item() - out[j].speed_mps))
        print(f"IK_PARITY_OK cases=1000 max_abs_error={worst:.8f}")
        if worst > 1e-4:
            raise AssertionError(f"IK parity failed: {worst}")


if __name__ == "__main__":
    main()
