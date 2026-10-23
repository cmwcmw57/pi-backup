"""Laptop checks for the Pi policy runtime (no SPI, no Isaac):

  python3 pi/policy/test_student_runtime.py

1) FirmwareIK (firmware C via ctypes) == Isaac BatchedSwerveIK + Res8 residual (torch) on random
   command sequences -- the policy was trained on the torch one, the rover runs the C one.
2) StudentObs layout: per-term history, oldest first, first frame repeated after reset.
3) run_policy.py --fake end to end with a random TorchScript policy.
"""

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "sim/isaac/ksrc_rover/source"))

from student_runtime import C, FirmwareIK, StudentObs, action_to_residual  # noqa: E402
from ksrc_rover.utils.ik_torch import BatchedSwerveIK  # noqa: E402

FAIL = []


def check(ok, msg):
    print(("ok   " if ok else "FAIL ") + msg)
    if not ok:
        FAIL.append(msg)


def test_ik_matches_sim():
    rng = np.random.default_rng(0)
    fw, sim = FirmwareIK(), BatchedSwerveIK(1, "cpu")
    worst_a, worst_w = 0.0, 0.0
    for k in range(3000):
        if k % 20 == 0:
            cmd = rng.uniform([-0.25, -0.2, -1.0], [0.25, 0.2, 1.0])
            if rng.random() < 0.1:
                cmd[:] = 0.0
            elif rng.random() < 0.15:          # 순수 옆방향·전후 (접기 경계, 10-14 버그)
                cmd[:] = rng.choice([[0.0, 0.1, 0.0], [0.0, -0.1, 0.0], [0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
        act = rng.uniform(-1, 1, 8)
        d, g = action_to_residual(act)
        goal, spd = fw.step(*cmd, d, g)
        st, sp = sim.compute(torch.as_tensor(np.asarray(cmd, np.float32)[None]))
        # Res8Action.process_actions
        t_steer = (st + torch.as_tensor(act[:4], dtype=torch.float32) * C.RES8_STEER_SCALE_RAD).clamp(
            -C.STEER_LIMIT_RAD, C.STEER_LIMIT_RAD)
        t_wheel = ((sp / C.WHEEL_RADIUS_M) * (1.0 + torch.as_tensor(act[4:], dtype=torch.float32) * C.RES8_WHEEL_SCALE)
                   ).clamp(-C.WHEEL_MAX_RAD_S, C.WHEEL_MAX_RAD_S)
        worst_a = max(worst_a, float(np.abs(goal - t_steer[0].numpy()).max()))
        worst_w = max(worst_w, float(np.abs(spd / C.WHEEL_RADIUS_M - t_wheel[0].numpy()).max()))
    # 바퀴: 펌웨어 최고속 0.378 m/s vs 시뮬 0.047503*7.96 = 0.37812 m/s (0.03%) 차이만 남음
    check(worst_a < 1e-4 and worst_w < 5e-3,
          f"펌웨어 IK+잔차 (C) == 시뮬 Res8 (torch): 3000 스텝 최대 차이 조향 {math.degrees(worst_a):.2e}°, 바퀴 {worst_w:.1e} rad/s")


def fake_meta(hist=5):
    dims = [("command", 3), ("gyro", 3), ("gravity", 3), ("steering", 4), ("steering_rate", 4),
            ("wheel_rate", 4), ("applied_targets", 8)]
    return {"student_history": hist, "obs_dim": 29 * hist, "action_dim": 8, "recurrent": False,
            "control_dt_s": 0.02, "terms": [{"name": n, "shape": [d * hist]} for n, d in dims]}


def test_obs_layout():
    ob = StudentObs(fake_meta(3))
    fr = lambda v: {n: np.full(d, v, np.float32) for n, d in ob.layout}
    o1 = ob.push(fr(1.0))[0]
    o2 = ob.push(fr(2.0))[0]
    cmd = o2[:9].reshape(3, 3)          # command term: 3 frames x 3
    check((o1 == 1.0).all() and (cmd[:2] == 1.0).all() and (cmd[2] == 2.0).all() and o2.size == 87,
          "관측: 항목별 이력, 오래된 것 먼저, 리셋 뒤 첫 프레임 반복 (Isaac CircularBuffer)")


def test_fake_run():
    with tempfile.TemporaryDirectory() as d:
        p = str(Path(d) / "pol")
        meta = fake_meta()
        net = torch.nn.Sequential(torch.nn.Linear(meta["obs_dim"], 8), torch.nn.Tanh())
        torch.jit.trace(net, torch.zeros(1, meta["obs_dim"])).save(p + ".pt")
        json.dump(meta, open(p + ".json", "w"))
        r = subprocess.run([sys.executable, str(HERE / "run_policy.py"), "--policy", p, "--fake", "--seconds", "1",
                            "--bias-s", "0.1", "--cmd", "0.1", "0", "0.3", "--log", p + "_log.npz"],
                           capture_output=True, text=True, timeout=60)
        ok = r.returncode == 0 and Path(p + "_log.npz").exists()
        n = len(np.load(p + "_log.npz")["obs"]) if ok else 0
        check(ok and n >= 45, f"run_policy.py --fake 1 s: {n} 스텝, rc={r.returncode} {r.stderr[-300:] if not ok else ''}")


if __name__ == "__main__":
    test_ik_matches_sim()
    test_obs_layout()
    test_fake_run()
    print("ALL PASS" if not FAIL else f"FAIL {len(FAIL)}")
    sys.exit(1 if FAIL else 0)
