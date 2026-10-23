# ================================================================
# test_pose_estimator.py
# 설명: 가짜 텔레메트리로 PoseEstimator의 직진·제자리 회전·옆걸음 적분, gyro bias, 중복 tick, 기울기, reanchor를 검증한다.
# 사용법: python3 -m pytest -p no:cacheprovider tests/yeouido_real/test_pose_estimator.py
# ================================================================
import math

import pytest

from experiments.orbslam3_ros2.swerve import SwerveGeometry
from experiments.yeouido_real.pose_estimator import GEOMETRY_PATH, PoseEstimator
from experiments.yeouido_real.run_real import FakeRover

GEO = SwerveGeometry.from_json(GEOMETRY_PATH)


def drive(est: PoseEstimator, cmd: tuple, seconds: float, rover: FakeRover | None = None) -> FakeRover:
    """FakeRover에 명령을 주며 20 ms 텔레메트리를 est에 넣는다."""
    rover = rover or FakeRover(GEO, 0.0, 0.0, 0.0)
    for _ in range(round(seconds / 0.02)):
        rover.send(*cmd)
        est.update(rover.recv())
    return rover


def test_straight_line_integrates_distance():
    est = PoseEstimator(GEO, bias_samples=0)
    drive(est, (0.1, 0.0, 0.0), 5.0)
    x, y, yaw = est.pose
    assert x == pytest.approx(0.5, abs=0.01) and abs(y) < 1e-6 and abs(yaw) < 1e-6


def test_rotate_in_place_integrates_yaw_without_translation():
    est = PoseEstimator(GEO, bias_samples=0)
    drive(est, (0.0, 0.0, 0.4), 5.0)
    x, y, yaw = est.pose
    assert yaw == pytest.approx(2.0, abs=0.02) and math.hypot(x, y) < 0.01


def test_crab_after_turn_moves_in_world_frame():
    est = PoseEstimator(GEO, x=1.0, y=1.0, yaw=math.pi / 2, bias_samples=0)
    drive(est, (0.0, 0.1, 0.0), 5.0)  # 차체 좌측 = 월드 -x
    x, y, _ = est.pose
    assert x == pytest.approx(0.5, abs=0.01) and y == pytest.approx(1.0, abs=0.01)


def test_arc_motion_matches_truth():
    est = PoseEstimator(GEO, bias_samples=0)
    rover = drive(est, (0.1, 0.0, 0.2), 5.0)
    assert est.pose == pytest.approx((rover.x, rover.y, rover.yaw), abs=0.01)


def test_gyro_bias_estimated_while_still_and_removed():
    est = PoseEstimator(GEO, bias_samples=50)
    rover = FakeRover(GEO, 0.0, 0.0, 0.0)
    for i in range(300):
        tlm = rover.recv()
        tlm["gyro_dps"] = (0.0, 0.0, 1.0)  # 일정 bias 1 dps
        est.update(tlm)
        assert est.ready == (i >= 49)
    assert est.pose == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


def test_duplicate_tick_is_ignored():
    est = PoseEstimator(GEO, bias_samples=0)
    rover = FakeRover(GEO, 0.0, 0.0, 0.0)
    rover.send(0.1, 0.0, 0.0)
    rover.recv()
    tlm = rover.recv()
    assert est.update(tlm) and not est.update(tlm)


def test_tilt_from_gravity_and_reanchor():
    est = PoseEstimator(GEO, bias_samples=0)
    p = math.radians(30)
    for k in range(200):
        est.update({"t_ms": 20 * k, "tick": k & 0xFF, "gyro_dps": (0, 0, 0),
                    "acc_g": (-math.sin(p), 0.0, math.cos(p)), "steer_meas_rad": (0,) * 4, "wheel_mps": (0,) * 4})
    assert est.pitch == pytest.approx(p, abs=1e-3) and abs(est.roll) < 1e-6
    est.reanchor(1.5, -0.2, 3 * math.pi / 2)
    assert est.pose == pytest.approx((1.5, -0.2, -math.pi / 2))
