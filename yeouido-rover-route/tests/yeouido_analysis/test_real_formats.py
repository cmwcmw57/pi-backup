# ================================================================
# test_real_formats.py
# 설명: Pi run_policy npz·Isaac trace.npz 합성 파일로 trace_io 로더, slip.py npz 입력, sim_vs_real.py 비교 지표를 검증한다.
# 사용법:
#   python -m pytest -p no:cacheprovider tests/yeouido_analysis/test_real_formats.py -q
# ================================================================
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from experiments.orbslam3_ros2.swerve import SwerveGeometry, body_twist_to_modules
from experiments.yeouido_analysis.common import Checkpoint
from experiments.yeouido_analysis.sim_vs_real import compare, label_stats, main as compare_main
from experiments.yeouido_analysis.slip import GEOMETRY_JSON, analyze, load_trace_npz
from experiments.yeouido_analysis.trace_io import load_run_trace, with_label_table

GEOMETRY = SwerveGeometry.from_json(GEOMETRY_JSON)
R = GEOMETRY.wheel_radius_m


def drive(plan: list[tuple[float, float]], dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(vx, 지속시간) 계획으로 시각·바퀴 속도(rad/s)·조향각(rad)을 만든다."""
    wheel, steer = [], []
    for vx, duration in plan:
        commands = body_twist_to_modules(GEOMETRY, vx=vx, vy=0.0, yaw_rate=0.0)
        for _ in range(round(duration / dt)):
            wheel.append([c.wheel_angular_velocity_rad_s for c in commands])
            steer.append([c.steering_angle_rad for c in commands])
    return np.arange(len(wheel)) * dt, np.array(wheel), np.array(steer)


def write_pi_npz(path: Path, plan: list[tuple[float, float]], *, tilt_rad: float = 0.2) -> None:
    """Pi 로그 키(host_ns, wheel_mps, steer_meas, acc_g, slip)를 가진 50 Hz 합성 npz를 쓴다."""
    t, wheel, steer = drive(plan, 0.02)
    np.savez(path, host_ns=(1_792_000_000_000_000_000 + t * 1e9).astype(np.int64), wheel_mps=wheel * R,
             steer_meas=steer, acc_g=np.tile([math.sin(tilt_rad), 0.0, math.cos(tilt_rad)], (len(t), 1)),
             slip=np.full((len(t), 4), 0.25, dtype=np.float32), cmd=np.zeros((len(t), 3)))


def write_sim_npz(path: Path, *, slide_m: float) -> None:
    """라벨 2개(오르기 5 s, 정지 3 s 동안 slide_m만큼 미끄러짐)를 가진 200 Hz 합성 trace.npz를 쓴다."""
    t, wheel, steer = drive([(0.1, 5.0), (0.0, 3.0)], 0.005)
    n_move = round(5.0 / 0.005)
    along = np.concatenate([0.08 * t[:n_move], 0.4 - slide_m * (t[n_move:] - t[n_move]) / 3.0])  # slip 0.2
    table = np.zeros((len(t), 18))
    table[:, 0] = t
    table[:, 1] = along * math.cos(math.radians(10))
    table[:, 3] = along * math.sin(math.radians(10))
    table[:, 8] = -math.radians(10)
    table[:, 10:14], table[:, 14:18] = wheel, steer
    names = ["t", "x", "y", "z", "qw", "qx", "qy", "qz", "pitch", "roll"] + \
        [f"wheel_vel_{i}" for i in range(4)] + [f"steer_{i}" for i in range(4)]
    labels = np.array(["오르기"] * n_move + ["정지"] * (len(t) - n_move))
    np.savez(path, trace=table, columns=np.array(names), labels=labels)


def test_pi_npz_loader_units_time_and_pitch(tmp_path: Path) -> None:
    write_pi_npz(tmp_path / "pi.npz", [(0.0, 1.0), (0.1, 2.0)])
    run = load_run_trace(tmp_path / "pi.npz")
    assert run.t_s[0] == 0.0 and run.t_s[-1] == pytest.approx(2.98, abs=1e-6)
    assert run.wheel_rad_s[-1, 0] == pytest.approx(0.1 / R, rel=1e-6)
    assert np.allclose(run.pitch_rad, -0.2, atol=1e-9) and run.xyz_m is None and run.labels is None
    assert np.allclose(run.log_slip, 0.25)


def test_unknown_npz_format_rejected(tmp_path: Path) -> None:
    np.savez(tmp_path / "x.npz", foo=np.zeros(3))
    with pytest.raises(ValueError, match="unknown trace npz format"):
        load_run_trace(tmp_path / "x.npz")


def test_slip_reads_pi_npz_without_checkpoint_times(tmp_path: Path) -> None:
    write_pi_npz(tmp_path / "pi.npz", [(0.0, 3.0), (0.1, 5.0), (0.0, 3.0)])
    cps = [Checkpoint("0", 0.0, 0.0, None, None, None), Checkpoint("1", 0.4, 0.0, None, 10.0, None)]
    row = analyze(GEOMETRY, load_trace_npz(tmp_path / "pi.npz"), cps)[0]
    assert row["slip"] == pytest.approx(0.2, abs=5e-3) and row["valid"]
    assert row["fw_slip_mean"] == pytest.approx(0.25)


def test_slip_uses_sim_xyz_as_reference(tmp_path: Path) -> None:
    write_sim_npz(tmp_path / "sim.npz", slide_m=0.0)
    cps = [Checkpoint("0", 0.0, 0.0, 0.0, None, None), Checkpoint("1", 0.4, 0.0, 5.0, 10.0, None)]
    row = analyze(GEOMETRY, load_trace_npz(tmp_path / "sim.npz"), cps)[0]
    assert row["slip_ref"] == pytest.approx(0.2, abs=5e-3)


def test_label_stats_time_distance_pitch_and_stop_slide(tmp_path: Path) -> None:
    write_sim_npz(tmp_path / "sim.npz", slide_m=0.1)
    stats = label_stats(load_run_trace(tmp_path / "sim.npz"), GEOMETRY)
    up, stop = stats["오르기"], stats["정지"]
    assert up["duration_s"] == pytest.approx(5.0, abs=0.01)
    assert up["dist_pose_m"] == pytest.approx(0.4, abs=2e-3) and up["dist_kin_m"] == pytest.approx(0.5, abs=3e-3)
    assert up["pitch_max_deg"] == pytest.approx(10.0)
    assert up["stop_slide_m"] == pytest.approx(0.0, abs=1e-6)
    assert stop["stop_slide_m"] == pytest.approx(0.1, abs=2e-3) and stop["dist_kin_m"] == pytest.approx(0.0, abs=1e-9)


def test_compare_with_label_table_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_sim_npz(tmp_path / "sim.npz", slide_m=0.1)
    write_pi_npz(tmp_path / "pi.npz", [(0.1, 5.0), (0.0, 3.0)])
    (tmp_path / "labels.csv").write_text("label,t_start_s,t_end_s\n오르기,0,5\n정지,5,8\n")
    real = with_label_table(load_run_trace(tmp_path / "pi.npz"), tmp_path / "labels.csv")
    rows = compare(label_stats(load_run_trace(tmp_path / "sim.npz"), GEOMETRY), label_stats(real, GEOMETRY))
    up = rows[0]
    assert up["label"] == "오르기" and up["duration_s_real"] == pytest.approx(5.0, abs=0.03)
    assert up["dist_kin_m_real"] == pytest.approx(0.5, abs=3e-3) and math.isnan(up["dist_pose_m_real"])
    assert up["pitch_max_deg_real"] == pytest.approx(11.46, abs=0.01)  # tilt 0.2 rad
    sys.argv = ["sim_vs_real", "--sim", str(tmp_path / "sim.npz"), "--real", str(tmp_path / "pi.npz"),
               "--real-labels", str(tmp_path / "labels.csv")]
    compare_main()
    printed = capsys.readouterr().out
    assert "0.25 m 반복 인공물" in printed and "오르기" in printed
