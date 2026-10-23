# ================================================================
# test_orb_pose.py
# 설명: OrbAligner가 회전·기울어진 ORB 지도 좌표에서 경로 좌표를 복원하는지, 정렬 전·추적 실패·지도 변경·pose 튐에서 멈추는지, 합성 ORB pose로 run()이 경로를 완주하는지 검증한다.
# 사용법: python3 -m pytest -p no:cacheprovider tests/yeouido_real/test_orb_pose.py
# ================================================================
import json
import math

import numpy as np
import pytest

pytest.importorskip("scipy")
from scipy.spatial.transform import Rotation  # noqa: E402

from experiments.orbslam3_ros2.swerve import SwerveGeometry
from experiments.orbslam3_ros2.yeouido_route import RouteRunner
from experiments.yeouido_real.orb_pose import OrbAligner
from experiments.yeouido_real.pose_estimator import GEOMETRY_PATH
from experiments.yeouido_real.run_real import PLAN, FakeRover, run

GEO = SwerveGeometry.from_json(GEOMETRY_PATH)
SPEC = json.loads(PLAN.read_text(encoding="utf-8"))
X0, Y0 = SPEC["start"]["base_link_xyz_m"][:2]
# 경로 -> ORB 지도: yaw 37도 + 약간 기울어진 중력(지도 z가 위가 아님) + 이동
MAP_R = Rotation.from_euler("zyx", [math.radians(37), math.radians(6), math.radians(-4)])
MAP_T = np.array([5.0, -3.0, 1.0])


def sample(x: float, y: float, yaw: float, z: float = 0.0, pitch: float = 0.0, ok: bool = True,
           frame: str = "orb_map_1", t_ms: int = 0) -> dict:
    """경로 좌표 base pose를 MAP_R/MAP_T로 옮긴 ORB 샘플."""
    r_route = Rotation.from_euler("zyx", [yaw, pitch, 0.0])
    q = (MAP_R * r_route).as_quat()
    return {"t_ms": t_ms, "frame_id": frame, "pos": tuple(MAP_R.apply([x, y, z]) + MAP_T), "quat": tuple(q),
            "tracking_ok": ok}


def anchored() -> OrbAligner:
    """시작 pose에서 정렬된 OrbAligner."""
    est = OrbAligner()
    est.update(sample(X0, Y0, 0.0))
    est.reanchor(X0, Y0, 0.0)
    return est


def test_not_ready_before_anchor_then_recovers_route_pose():
    est = OrbAligner()
    est.update(sample(X0, Y0, 0.0))
    assert not est.ready
    est.reanchor(X0, Y0, 0.0)
    assert est.ready
    for x, y, yaw in [(1.0, 0.42, 0.0), (2.2, 0.42, math.pi), (0.6, 1.24, -math.pi / 2)]:
        est.update(sample(x, y, yaw))
        assert est.pose[:2] == pytest.approx((x, y), abs=1e-6)
        assert abs(math.atan2(math.sin(est.pose[2] - yaw), math.cos(est.pose[2] - yaw))) < 1e-6


def test_ramp_climb_reports_horizontal_xy_and_pitch():
    est = anchored()
    s = math.radians(30)
    est.update(sample(1.0, 1.24, 0.0, z=0.4, pitch=-s))  # 오르막: 코가 위 -> ROS pitch 음수
    assert est.pose[:2] == pytest.approx((1.0, 1.24), abs=1e-6)
    assert est.pitch == pytest.approx(-s, abs=1e-6) and abs(est.roll) < 1e-6


def test_tracking_loss_stops_then_resumes():
    est = anchored()
    est.update(sample(X0, Y0, 0.0, ok=False))
    assert not est.ready
    est.update(sample(X0, Y0, 0.0))
    assert est.ready


def test_map_change_requires_reanchor():
    est = anchored()
    est.update(sample(1.0, 0.4, 0.0, frame="orb_map_2"))
    assert not est.ready
    est.reanchor(1.0, 0.4, 0.0)
    assert est.ready and est.pose[:2] == pytest.approx((1.0, 0.4), abs=1e-6)
    est.update(sample(1.2, 0.4, 0.0, frame="orb_map_2"))
    assert est.pose[0] == pytest.approx(1.2, abs=1e-6)


def test_pose_jump_stops_until_reanchor():
    est = anchored()
    est.update(sample(X0 + 0.05, Y0, 0.0))
    assert est.ready
    est.update(sample(X0 + 1.5, Y0, 0.0))
    assert not est.ready and est.n_jump == 1


class FakeOrb(FakeRover):
    """FakeRover의 진짜 pose를 ORB 지도 좌표 샘플로 내보낸다 (stamp는 epoch처럼 큰 값)."""

    def recv(self) -> dict:
        super().recv()
        return sample(self.x, self.y, self.yaw, t_ms=1_760_000_000_000 + self.t_ms)


def test_full_route_with_orb_pose_completes():
    rover = FakeOrb(GEO, X0, Y0, 0.0)
    est = OrbAligner()
    keys = iter([None, "c"] + [None] * 10**6)
    result = run(RouteRunner(SPEC), est, rover, rover, key=lambda: next(keys), ask_pose=lambda: (X0, Y0, 0.0))
    assert result == "complete"
    assert math.hypot(rover.x - 2.2, rover.y - 2.06) < 0.06 and abs(rover.yaw) < math.radians(6)


def test_orb_route_robot_waits_until_anchored():
    rover = FakeOrb(GEO, X0, Y0, 0.0)
    est = OrbAligner()
    assert run(RouteRunner(SPEC), est, rover, rover, max_s=3.0) == "timeout"
    assert (rover.x, rover.y) == (X0, Y0)  # 정렬 전에는 한 발짝도 안 간다
