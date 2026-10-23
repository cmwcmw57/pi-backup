# ================================================================
# test_slip.py
# 설명: 합성 trace(알려진 슬립률)로 slip.py의 twist 역산·구간 슬립·무효 처리·그룹 표를 검증한다.
# 사용법:
#   python -m pytest -p no:cacheprovider tests/yeouido_analysis/test_slip.py -q
# ================================================================
import csv
import math
from pathlib import Path

import numpy as np
import pytest

from experiments.orbslam3_ros2.swerve import SwerveGeometry, body_twist_to_modules
from experiments.yeouido_analysis.common import Checkpoint, detect_stops
from experiments.yeouido_analysis.slip import (
    GEOMETRY_JSON,
    Trace,
    analyze,
    body_twists,
    summarize,
)

GEOMETRY = SwerveGeometry.from_json(GEOMETRY_JSON)
DT = 0.02


def make_trace(legs: list[tuple[float, float, float]], *, vy: float = 0.0, stop_s: float = 3.0) -> Trace:
    """(속도, 지속시간, 조향 yaw_rate) 구간 사이에 정지를 넣은 합성 trace를 만든다."""
    wheel, steer, speeds = [], [], []
    plan = [(0.0, stop_s, 0.0)]
    for speed, duration, yaw_rate in legs:
        plan += [(speed, duration, yaw_rate), (0.0, stop_s, 0.0)]
    for speed, duration, yaw_rate in plan:
        commands = body_twist_to_modules(GEOMETRY, vx=speed, vy=vy if speed else 0.0, yaw_rate=yaw_rate)
        for _ in range(int(duration / DT)):
            wheel.append([c.wheel_angular_velocity_rad_s for c in commands])
            steer.append([c.steering_angle_rad for c in commands])
    t = np.arange(len(wheel)) * DT
    return Trace(t_s=t, wheel_rad_s=np.array(wheel), steer_rad=np.array(steer), ref_pos_m=None)


def checkpoints(distances: list[float], ramp_deg: float | None = 10.0) -> list[Checkpoint]:
    """x축 위에 누적 거리로 놓인 체크포인트 목록을 만든다."""
    xs = np.concatenate([[0.0], np.cumsum(distances)])
    return [Checkpoint(str(i), float(x), 0.0, None, ramp_deg if i else None, None) for i, x in enumerate(xs)]


def test_body_twists_recovers_commanded_twist() -> None:
    trace = make_trace([(0.1, 1.0, 0.0)], vy=0.05)
    twist = body_twists(GEOMETRY, trace)
    moving = twist[(trace.t_s > 3.2) & (trace.t_s < 3.8)]
    assert np.allclose(moving[:, 0], 0.1, atol=1e-9)
    assert np.allclose(moving[:, 1], 0.05, atol=1e-9)


def test_body_twists_pure_rotation_yaw_rate() -> None:
    commands = body_twist_to_modules(GEOMETRY, vx=0.0, vy=0.0, yaw_rate=0.3)
    trace = Trace(t_s=np.array([0.0, 1.0]),
                  wheel_rad_s=np.array([[c.wheel_angular_velocity_rad_s for c in commands]] * 2),
                  steer_rad=np.array([[c.steering_angle_rad for c in commands]] * 2), ref_pos_m=None)
    assert np.allclose(body_twists(GEOMETRY, trace)[0], [0.0, 0.0, 0.3], atol=1e-9)


def test_detect_stops_requires_minimum_duration() -> None:
    t = np.arange(0, 10, 0.1)
    speed = np.where((t > 2) & (t < 6), 0.1, 0.0)
    assert len(detect_stops(t, speed, 1.5)) == 2
    assert detect_stops(t, np.where(t > 1, 0.1, 0.0), 1.5) == []


def test_known_slip_recovered_per_segment_and_wheel() -> None:
    kin_m = 0.5
    trace = make_trace([(0.1, kin_m / 0.1, 0.0)] * 2)
    rows = analyze(GEOMETRY, trace, checkpoints([kin_m * 0.8, kin_m * 0.8]))
    assert [r["valid"] for r in rows] == [True, True]
    for row in rows:
        assert row["kin_m"] == pytest.approx(kin_m, rel=2e-3)
        assert row["slip"] == pytest.approx(0.2, abs=2e-3)
        assert row["slip_fl"] == pytest.approx(0.2, abs=2e-3)
        assert row["speed_m_s"] == pytest.approx(0.1)
    (group,) = summarize(rows)
    assert group["ramp_deg"] == 10.0 and group["n"] == 2
    assert group["slip_pooled"] == pytest.approx(0.2, abs=2e-3)


def test_low_speed_and_short_segments_are_invalid() -> None:
    trace = make_trace([(0.03, 5.0, 0.0), (0.15, 0.4, 0.0)])
    rows = analyze(GEOMETRY, trace, checkpoints([0.15, 0.06]))
    assert not rows[0]["valid"] and "low_speed" in rows[0]["reason"]
    assert not rows[1]["valid"] and "short_distance" in rows[1]["reason"]
    assert summarize(rows) == []


def test_plan_frame_corrects_horizontal_projection() -> None:
    kin_m = 0.6
    trace = make_trace([(0.15, kin_m / 0.15, 0.0)])
    plan_distance = kin_m * math.cos(math.radians(30))
    rows = analyze(GEOMETRY, trace, checkpoints([plan_distance], ramp_deg=30.0), measured_frame="plan")
    assert rows[0]["slip"] == pytest.approx(0.0, abs=2e-3)
    rows = analyze(GEOMETRY, trace, checkpoints([plan_distance], ramp_deg=30.0), measured_frame="surface")
    assert rows[0]["slip"] == pytest.approx(1 - math.cos(math.radians(30)), abs=2e-3)


def test_stop_count_mismatch_raises_and_t_s_overrides() -> None:
    trace = make_trace([(0.15, 4.0, 0.0)] * 2)
    with pytest.raises(ValueError, match="t_s column"):
        analyze(GEOMETRY, trace, checkpoints([0.6, 0.6, 0.6]))
    with_t = [Checkpoint("0", 0.0, 0.0, 1.5, None, None), Checkpoint("1", 0.6, 0.0, 8.5, 10.0, None)]
    row = analyze(GEOMETRY, trace, with_t)[0]
    assert row["tape_m"] == pytest.approx(0.6) and row["kin_m"] == pytest.approx(0.6, rel=2e-3)


def test_in_place_rotation_is_not_a_stop() -> None:
    trace = make_trace([(0.0, 3.0, 0.4), (0.1, 5.0, 0.0)])
    rows = analyze(GEOMETRY, trace, checkpoints([0.45, 0.5]))
    assert rows[0]["segment"] == "0->1" and not rows[0]["valid"]
    assert rows[1]["segment"] == "1->2" and rows[1]["valid"]
    assert rows[1]["slip"] == pytest.approx(0.0, abs=2e-3)


def test_cli_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from experiments.yeouido_analysis.slip import main
    import sys

    trace = make_trace([(0.15, 4.0, 0.0)])
    tags = ["fl", "fr", "rl", "rr"]
    with (tmp_path / "trace.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["t_s"] + [f"wheel_{t}_rad_s" for t in tags] + [f"steer_{t}_rad" for t in tags])
        for t, w, s in zip(trace.t_s, trace.wheel_rad_s, trace.steer_rad):
            writer.writerow([t, *w, *s])
    (tmp_path / "cps.csv").write_text(
        "checkpoint_id,measured_x_m,measured_y_m,ramp_deg\nstart,0,0,\nend,0.54,0,10\n")
    sys.argv = ["slip", "--trace", str(tmp_path / "trace.csv"), "--checkpoints", str(tmp_path / "cps.csv"),
               "--source", "sim", "--out", str(tmp_path / "out.csv")]
    main()
    printed = capsys.readouterr().out
    assert "[source=sim]" in printed and "start->end" in printed
    assert (tmp_path / "out.csv").exists()
