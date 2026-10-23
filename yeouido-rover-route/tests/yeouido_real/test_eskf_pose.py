# ================================================================
# test_eskf_pose.py
# 설명: 합성 IMU(50 Hz)+지연 ORB pose로 EskfPose가 여의도 경로를 끝까지 돌리는지, IMU 끊김·invalid·ORB 상실·reanchor 방어, UnixTelemetry와 trace npz를 검증한다. 환경변수 ROVER_MAPPING_ROOT(rover-mapping 체크아웃)와 scipy가 없으면 skip한다.
# 사용법: ROVER_MAPPING_ROOT=/path/to/rover-mapping python3 -m pytest -p no:cacheprovider tests/yeouido_real/test_eskf_pose.py
# ================================================================
import json
import math
import os
import socket
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
_ROOT = os.environ.get("ROVER_MAPPING_ROOT")
pytestmark = pytest.mark.skipif(not (_ROOT and (Path(_ROOT) / "experiments/localization/online.py").is_file()),
                                reason="ROVER_MAPPING_ROOT(rover-mapping 체크아웃)가 필요하다")

from experiments.orbslam3_ros2.swerve import SwerveGeometry  # noqa: E402
from experiments.orbslam3_ros2.yeouido_route import RouteRunner  # noqa: E402
from experiments.yeouido_real.eskf_pose import EskfPose  # noqa: E402
from experiments.yeouido_real.pose_estimator import GEOMETRY_PATH  # noqa: E402
from experiments.yeouido_real.run_real import (PLAN, TRACE_COLUMNS, FakeEskfRover, UnixTelemetry, main,  # noqa: E402
                                               run)

GEO = SwerveGeometry.from_json(GEOMETRY_PATH)
SPEC = json.loads(PLAN.read_text(encoding="utf-8"))
X0, Y0, Z0 = SPEC["start"]["base_link_xyz_m"][:3]


class Scripted(FakeEskfRover):
    """호출 번호(20 ms 단위) 구간별로 텔레메트리 끊김·IMU invalid·ORB 추적 상실·gyro bias를 주입하는 합성 로버."""

    def __init__(self, drop=(), invalid=(), orb_off=(), gyro_bias_dps=0.0) -> None:
        super().__init__(GEO, X0, Y0, 0.0, Z0)
        self.drop, self.invalid, self.orb_off, self.bias, self.calls = drop, invalid, orb_off, gyro_bias_dps, 0

    def recv(self):
        """구간 안이면 로버(시간)는 진행시키되 None 또는 변형된 sample을 돌려준다."""
        self.calls += 1
        inside = lambda spans: any(a <= self.calls < b for a, b in spans)  # noqa: E731
        self.orb_ok = not inside(self.orb_off)
        s = super().recv()
        s["tlm"]["gyro_dps"] = (0.0, 0.0, s["tlm"]["gyro_dps"][2] + self.bias)
        if inside(self.invalid):
            s["tlm"]["flags"] = 0
        return None if inside(self.drop) else s


def drive(rover: Scripted, **kw) -> tuple[str, EskfPose, list]:
    """EskfPose로 경로 전체를 돌리고 (결과, est, trace)를 반환한다."""
    est, trace = EskfPose(X0, Y0, Z0, 0.0), []
    return run(RouteRunner(SPEC), est, rover, rover, trace=trace, **kw), est, trace


def at_target(rover: Scripted) -> bool:
    """진짜 pose가 C 램프 끝 목표 근처인지."""
    return math.hypot(rover.x - 2.2, rover.y - 2.06) < 0.06 and abs(math.atan2(math.sin(rover.yaw), math.cos(rover.yaw))) < math.radians(6)


def err(est: EskfPose, rover: Scripted) -> float:
    """추정과 진짜 pose의 평면 거리 [m]."""
    return math.hypot(est.pose[0] - rover.x, est.pose[1] - rover.y)


def test_full_route_completes_and_estimate_tracks_truth():
    rover = Scripted(gyro_bias_dps=1.0)  # 시작 정지 구간 평균으로 bias를 잡는다
    result, est, trace = drive(rover)
    assert result == "complete" and at_target(rover) and err(est, rover) < 0.05
    assert np.linalg.norm(est._bias) == pytest.approx(math.radians(1.0), rel=0.05)
    assert not any(r[10] for r in trace[:100]) and trace[-1][10]  # bias 측정 2 s 동안은 정지
    assert all(r[7] == r[8] == r[9] == 0.0 for r in trace[:100])


def test_robot_waits_until_visual_update_accepted():
    rover = Scripted(orb_off=[(0, 10**6)])
    result, est, _ = drive(rover, max_s=10.0)
    assert result == "timeout" and not est.ready and (rover.x, rover.y) == (X0, Y0)


@pytest.mark.parametrize("span", [(500, 550), (500, 508)], ids=["1.0s", "0.16s"])
def test_imu_gap_stops_reinitializes_and_resumes(span):
    rover = Scripted(drop=[span])
    zero_in_gap, orig = [], rover.recv

    def recv():
        s = orig()
        if s is None:
            zero_in_gap.append(rover.cmd == (0.0, 0.0, 0.0) or rover.calls == span[0])
        return s

    rover.recv = recv
    runner, est, trace = RouteRunner(SPEC), EskfPose(X0, Y0, Z0, 0.0), []
    assert run(runner, est, rover, rover, trace=trace, max_s=24.0) == "timeout"
    assert est.n_gap == 1 and est.ready and err(est, rover) < 0.06
    assert all(zero_in_gap[1:]) and any(not r[10] for r in trace[span[0]:span[0] + 60])  # 끊긴 동안 명령 0, 정지
    gap_rows = [r for r in trace if r[0] > span[1] * 20 + 1000 and r[10]]
    assert gap_rows and runner.index > 0 and rover.x > X0 + 0.1  # 복귀 뒤 경로를 다시 진행


def test_imu_invalid_flag_counts_as_gap():
    rover = Scripted(invalid=[(500, 520)])
    result, est, _ = drive(rover, max_s=22.0)
    assert result == "timeout" and est.n_gap == 1 and est.ready and err(est, rover) < 0.06


def test_orb_loss_longer_than_invalid_window_stops_then_resumes():
    rover = Scripted(orb_off=[(500, 900)])  # 8 s
    runner, est, trace = RouteRunner(SPEC), EskfPose(X0, Y0, Z0, 0.0), []
    assert run(runner, est, rover, rover, trace=trace, max_s=26.0) == "timeout"
    stopped = [r for r in trace if not r[10] and r[0] > 12000]
    assert stopped and all(r[7] == r[8] == r[9] == 0.0 for r in stopped)
    assert est.ready and err(est, rover) < 0.06 and trace[-1][12] != "done"


def test_reanchor_resets_pose_and_keeps_running():
    rover = Scripted()
    est, runner = EskfPose(X0, Y0, Z0, 0.0), RouteRunner(SPEC)
    keys = iter([None] * 400 + ["c"] + [None] * 10**6)
    result = run(runner, est, rover, rover, key=lambda: next(keys), max_s=16.0,
                 ask_pose=lambda: (rover.x + 0.03, rover.y, rover.yaw))  # 줄자 값이 추정과 3 cm 다르다
    assert result == "timeout" and est.ready and err(est, rover) < 0.04


def test_unix_telemetry_unwraps_tee_payload(tmp_path):
    path = str(tmp_path / "t.sock")
    src = UnixTelemetry(path, timeout_s=0.2)
    assert src.recv() is None
    tlm = {k: 0 for k in ("t_ms", "tick", "flags", "gyro_dps", "acc_g", "steer_meas_rad", "wheel_mps")} | {"t_ms": 40}
    s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    s.sendto(json.dumps({"schema": 1, "host_receipt_ns": 1_760_000_000_500_000_000, "telemetry": tlm}).encode(), path)
    got = src.recv()
    assert got["t_ms"] == 40 and got["host_s"] == pytest.approx(1_760_000_000.5)
    s.sendto(b'{"telemetry": {"t_ms": 1}}', path)  # 필요한 키가 없으면 버린다
    assert src.recv() is None
    s.close()
    src.close()
    assert not (tmp_path / "t.sock").exists()


def test_trace_npz_has_documented_columns(tmp_path):
    log = tmp_path / "trace.npz"
    assert main(["--fake", "--pose", "eskf", "--log", str(log), "--max-s", "12"]) == 1  # 12 s에서 중단 -> timeout
    d = np.load(log)
    assert tuple(k for k in d.files if k != "meta") == TRACE_COLUMNS
    n = len(d["t_s"])
    assert n > 500 and all(len(d[k]) == n for k in TRACE_COLUMNS)
    assert json.loads(str(d["meta"]))["result"] == "timeout" and bool(d["ready"][-1])


def test_replays_recorded_sim_run_like_online_estimator():
    """시뮬 repeat1018a 기록(IMU 축·시각·ORB pose)을 텔레메트리 모양으로 바꿔 넣어도 기록된 추정과 같고 GT에 가깝다."""
    from scipy.spatial.transform import Rotation
    root = Path(_ROOT) / "outputs/site_sim/repeat1018a"
    if not (root / "estimator_events.jsonl").is_file():
        pytest.skip("repeat1018a 기록이 없다")
    cfg = json.loads((root / "estimator_calibration.json").read_text())
    start = np.asarray(cfg["start_world_from_base"])
    est = EskfPose(*start[:3, 3], math.atan2(start[1, 0], start[0, 0]), calibration=cfg["calibration"],
                   lag_s=cfg["render_lag_s"], imu_offset_s=0.0, bias_samples=40)
    errs, diffs, ready, n = [], [], 0, 0
    for k, line in enumerate(open(root / "estimator_events.jsonl")):
        e = json.loads(line)
        t = e["stamp_s"]
        if t > 130.0:
            break
        if e["kind"] == "imu":  # 필터의 IMU축 값을 로버 몸체(base)축 텔레메트리로 되돌린다
            r = est.rot_base_from_imu
            tlm = {"t_ms": t * 1000, "tick": k & 0xFF, "flags": 16, "gyro_dps": tuple(np.degrees(r @ e["gyro"])),
                   "acc_g": tuple(r @ np.asarray(e["force"]) / 9.81)}
            est.update({"host_s": t, "tlm": tlm, "orb": None})
        elif e["kind"] == "frame":
            pose = np.asarray(e["world_from_camera"]) @ est.cam_from_base
            orb = {"frame_id": f"orb_map_{e['map_id']}", "pos": tuple(pose[:3, 3]), "tracking_ok": e["orb_state"] == 2,
                   "quat": tuple(Rotation.from_matrix(pose[:3, :3]).as_quat())}
            est.update({"host_s": t, "tlm": None, "orb": orb})
        elif est._T is not None:
            n, ready = n + 1, ready + est.ready
            errs.append(math.hypot(est.pose[0] - e["position_m"][0], est.pose[1] - e["position_m"][1]))
            diffs.append(math.hypot(est.pose[0] - e["estimated_position_m"][0], est.pose[1] - e["estimated_position_m"][1]))
    assert n > 2000 and ready / n > 0.99 and est.n_gap == 0
    assert max(diffs) < 0.02 and math.sqrt(np.mean(np.square(errs))) < 0.05
