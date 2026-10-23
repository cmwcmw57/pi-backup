# ================================================================
# test_orb_retention.py
# 설명: 합성 tracking 로그로 orb_retention.py의 유지율·손실 횟수·map 변경·위치 오차 계산을 검증한다.
# 사용법:
#   python -m pytest -p no:cacheprovider tests/yeouido_analysis/test_orb_retention.py -q
# ================================================================
import math
from pathlib import Path

import numpy as np
import pytest

from experiments.yeouido_analysis.common import Checkpoint
from experiments.yeouido_analysis.orb_retention import analyze, load_tracking, summarize

T0_NS = 1_792_000_000_000_000_000


def write_log(path: Path, *, time_column: str = "stamp_ns") -> None:
    """10 Hz 30 s 로그: 0.1 m/s로 이동(축은 임의 회전), ORB 거리 스케일 +5%, 15~16 s state 3, 22 s에 map 교체."""
    rotation = np.array([[0.0, -0.6, 0.8], [1.0, 0.0, 0.0], [0.0, 0.8, 0.6]])
    lines = [f"{time_column},state,map_id,m03,m13,m23"]
    for i in range(300):
        t = i * 0.1
        state = 3 if 15.0 <= t < 16.0 else 2
        map_id = -1 if state == 3 else (0 if t < 22.0 else 7)
        position = rotation @ np.array([0.1 * 1.05 * t, 0.0, 0.0]) + (0.0 if map_id != 7 else 50.0)
        shown = ["nan"] * 3 if state == 3 else [repr(float(v)) for v in position]
        lines.append(",".join([str(T0_NS + round(t * 1e9)), str(state), str(map_id), *shown]))
    path.write_text("\n".join(lines) + "\n")


def checkpoints() -> list[Checkpoint]:
    """0→10 s, 10→20 s, 20→29 s 세 구간(10 s 구간은 줄자 1.0 m)."""
    return [Checkpoint("0", 0.0, 0.0, 0.0, None, None),
            Checkpoint("1", 1.0, 0.0, 10.0, 10.0, 0.1),
            Checkpoint("2", 2.0, 0.0, 20.0, 10.0, 0.1),
            Checkpoint("3", 2.9, 0.0, 29.0, 15.0, 0.2)]


def test_retention_counts_and_position_error(tmp_path: Path) -> None:
    write_log(tmp_path / "log.csv")
    rows = analyze(load_tracking(tmp_path / "log.csv", relative_time=True), checkpoints())
    first, second, third = rows
    assert first["ok_fraction"] == 1.0 and first["loss_events"] == 0
    assert first["orb_m"] == pytest.approx(1.05, rel=1e-6) and first["err_pct"] == pytest.approx(5.0, rel=1e-6)
    assert second["ok_fraction"] == pytest.approx(1 - 10 / second["frames"])
    assert second["recent_lost_fraction"] == pytest.approx(10 / second["frames"])
    assert second["loss_events"] == 1 and second["map_changes"] == 0
    assert second["orb_m"] == pytest.approx(1.05, rel=1e-6)
    assert third["map_changes"] == 1 and "orb_m" not in third
    table = summarize(rows)
    assert [(g["ramp_deg"], g["speed_m_s"]) for g in table] == [(10.0, 0.1), (15.0, 0.2)]
    assert table[0]["loss_events"] == 1 and table[0]["abs_err_pct_mean"] == pytest.approx(5.0, rel=1e-6)
    assert table[1]["abs_err_pct_mean"] is None


def test_timestamp_ns_alias_and_missing_pose(tmp_path: Path) -> None:
    (tmp_path / "t.csv").write_text("timestamp_ns,state,imu_init_elapsed_s\n1000000000,2,0\n2000000000,3,0\n")
    log = load_tracking(tmp_path / "t.csv")
    assert log.position_m is None and log.t_s.tolist() == [1.0, 2.0]
    rows = analyze(log, [Checkpoint("a", 0, 0, 0.0, None, None), Checkpoint("b", 1, 0, 3.0, None, None)])
    assert rows[0]["ok_fraction"] == 0.5 and "err_pct" not in rows[0]


def test_missing_checkpoint_time_and_columns_raise(tmp_path: Path) -> None:
    write_log(tmp_path / "log.csv")
    log = load_tracking(tmp_path / "log.csv", relative_time=True)
    with pytest.raises(ValueError, match="t_s"):
        analyze(log, [Checkpoint("a", 0, 0, None, None, None), Checkpoint("b", 1, 0, 3.0, None, None)])
    (tmp_path / "bad.csv").write_text("foo,bar\n1,2\n")
    with pytest.raises(ValueError, match="state"):
        load_tracking(tmp_path / "bad.csv")


def write_stop_go_log(path: Path) -> None:
    """10 Hz 35 s: 정지 5 s → 0.1 m/s 이동 10 s → 정지 5 s → 이동 10 s → 정지 5 s, 정지 중 1e-4 m jitter."""
    lines = ["stamp_ns,state,map_id,m03,m13,m23"]
    for i in range(350):
        t = i * 0.1
        along = 0.1 * (min(max(t - 5, 0), 10) + min(max(t - 20, 0), 10))
        jitter = 1e-4 * math.sin(7 * t)
        lines.append(f"{T0_NS + round(t * 1e9)},2,0,{along + jitter!r},{jitter!r},0.0")
    path.write_text("\n".join(lines) + "\n")


def test_stops_detected_without_t_s_and_speed_label_from_orb(tmp_path: Path) -> None:
    write_stop_go_log(tmp_path / "log.csv")
    log = load_tracking(tmp_path / "log.csv", relative_time=True)
    points = [Checkpoint(str(i), float(i), 0.0, None, 10.0 if i else None, None) for i in range(3)]
    rows = analyze(log, points)
    assert [row["orb_m"] for row in rows] == [pytest.approx(1.0, abs=2e-3)] * 2
    assert [row["speed_m_s"] for row in rows] == [pytest.approx(0.1)] * 2
    four = points + [Checkpoint("3", 3.0, 0.0, None, 10.0, None)]
    with pytest.raises(ValueError, match="3 ORB stops"):
        analyze(log, four)
