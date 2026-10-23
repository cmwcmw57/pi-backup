# ================================================================
# test_run_real.py
# 설명: run_real --fake가 여의도 경로 전체를 완주하는지, q 정지·시간초과·reanchor·tilt 중단·텔레메트리 UDP 왕복을 검증한다.
# 사용법: python3 -m pytest -p no:cacheprovider tests/yeouido_real/test_run_real.py
# ================================================================
import json
import math
import socket

import pytest

from experiments.orbslam3_ros2.swerve import SwerveGeometry
from experiments.orbslam3_ros2.yeouido_route import RouteRunner
from experiments.yeouido_real.pose_estimator import GEOMETRY_PATH, PoseEstimator
from experiments.yeouido_real.run_real import PLAN, FakeRover, UdpTelemetry, main, run, telemetry_packet

GEO = SwerveGeometry.from_json(GEOMETRY_PATH)
SPEC = json.loads(PLAN.read_text(encoding="utf-8"))
X0, Y0 = SPEC["start"]["base_link_xyz_m"][:2]


def setup(**kw):
    """est, runner, FakeRover를 경로 시작 pose로 만든다."""
    rover = kw.get("rover") or FakeRover(GEO, X0, Y0, 0.0)
    return PoseEstimator(GEO, X0, Y0, 0.0), RouteRunner(SPEC), rover


def test_fake_full_route_completes(capsys, tmp_path):
    assert main(["--fake", "--log", str(tmp_path / "t.npz")]) == 0
    assert capsys.readouterr().out.startswith("complete: step=done")


def test_fake_true_pose_reaches_lane_c_top_with_clamped_speed():
    est, runner, rover = setup()
    peak = 0.0
    orig = rover.send

    def spy(vx, vy, w):
        nonlocal peak
        peak = max(peak, abs(vx), abs(vy))
        return orig(vx, vy, w)

    rover.send = spy
    assert run(runner, est, rover, rover) == "complete"
    assert math.hypot(rover.x - 2.2, rover.y - 2.06) < 0.06 and abs(rover.yaw) < math.radians(6)
    assert 0.0 < peak <= 0.20 and rover.cmd == (0.0, 0.0, 0.0)


def test_q_stops_and_sends_zero():
    est, runner, rover = setup()
    keys = iter([None] * 600 + ["q"])
    assert run(runner, est, rover, rover, key=lambda: next(keys)) == "quit"
    assert rover.cmd == (0.0, 0.0, 0.0) and runner.index > 0


def test_reanchor_key_overwrites_pose():
    est, runner, rover = setup()
    keys = iter([None] * 300 + ["c", "q"])
    run(runner, est, rover, rover, key=lambda: next(keys), ask_pose=lambda: (1.23, 0.45, 0.1))
    assert est.pose[0] == pytest.approx(1.23, abs=0.02)


def test_timeout_and_tilt_abort():
    est, runner, rover = setup()
    assert run(runner, est, rover, rover, max_s=5.0) == "timeout"
    est, runner, rover = setup()
    orig = rover.recv
    rover.recv = lambda: {**orig(), "acc_g": (-math.sin(0.9), 0.0, math.cos(0.9))}  # pitch ~52deg > 40deg
    assert run(runner, est, rover, rover) == "aborted" and rover.cmd == (0.0, 0.0, 0.0)


def test_udp_telemetry_roundtrip():
    src = UdpTelemetry(0, timeout_s=0.2)
    port = src.sock.getsockname()[1]
    assert src.recv() is None
    tlm = FakeRover(GEO, 0, 0, 0).recv() | {"extra": 1}
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(telemetry_packet(tlm), ("127.0.0.1", port))
    got = src.recv()
    assert got is not None and got["t_ms"] == 20 and "extra" not in got and len(got["wheel_mps"]) == 4
    PoseEstimator(GEO, bias_samples=0).update(got)  # JSON(list)로 와도 estimator가 받는다
    s.close()
    src.sock.close()
