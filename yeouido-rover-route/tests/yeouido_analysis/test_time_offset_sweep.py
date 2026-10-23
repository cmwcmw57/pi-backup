# ================================================================
# test_time_offset_sweep.py
# 설명: time_offset_sweep.py의 명령 생성·summary 파싱·집계·dry-run·가짜 도구 실행을 검증한다(실제 ORB 재생은 사용하지 않음).
# 사용법:
#   python -m pytest -p no:cacheprovider tests/yeouido_analysis/test_time_offset_sweep.py -q
# ================================================================
import json
import stat
from pathlib import Path

import pytest

from experiments.yeouido_analysis.time_offset_sweep import (
    aggregate,
    build_command,
    parse_summary,
    run_name,
    run_sweep,
)

FAKE_TOOL = """#!/usr/bin/env bash
# 가짜 imu_fusion.sh: 요청 보정에 비례하는 summary.json만 만든다.
out=; corr=
while [ $# -gt 0 ]; do
  case "$1" in --output) out=$2; shift;; --correction-s) corr=$2; shift;; esac
  shift
done
mkdir -p "$out"
frac=$(awk -v c="$corr" 'BEGIN{printf "%.3f", 1+c*10}')
echo "{\\"status\\":\\"initialized\\",\\"imu_time_correction_s\\":$corr,\\"frames\\":100,\\"tracking_ok_fraction\\":0.9,\\"imu_initialized_fraction\\":$frac,\\"map_ids\\":[0,3],\\"final_export_has_inertial_map\\":true}" > "$out/summary.json"
"""


def fake_tool(tmp_path: Path) -> Path:
    """실행 권한이 있는 가짜 도구 스크립트를 만든다."""
    path = tmp_path / "fake_fusion.sh"
    path.write_text(FAKE_TOOL)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_build_command_sign_and_options() -> None:
    command = build_command(Path("tools/imu_fusion.sh"), Path("/r"), Path("/o"), -37.5, 1, True)
    assert command == ["bash", "tools/imu_fusion.sh", "/r", "--output", "/o", "--correction-s", "-0.037500",
                       "--segment", "1", "--allow-legacy"]
    assert "--auto" not in command and run_name(-35.0, 2) == "off-35ms_r2"


def test_parse_summary_variants(tmp_path: Path) -> None:
    assert parse_summary(tmp_path / "none", -35.0, 0)["status"] == "no_summary"
    (tmp_path / "ok").mkdir()
    (tmp_path / "ok/summary.json").write_text(json.dumps(
        {"status": "initialized", "imu_time_correction_s": -0.035, "frames": 10, "map_ids": [1, 2, 3],
         "tracking_ok_fraction": 0.5, "imu_initialized_fraction": 0.25, "final_export_has_inertial_map": True}))
    row = parse_summary(tmp_path / "ok", -35.0, 0)
    assert row["status"] == "initialized" and row["n_map_ids"] == 3 and row["imu_initialized_fraction"] == 0.25
    assert parse_summary(tmp_path / "ok", -45.0, 0)["status"] == "offset_mismatch"
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad/summary.json").write_text(json.dumps({"status": "failed", "error": "boom"}))
    assert parse_summary(tmp_path / "bad", -35.0, 0)["error"] == "boom"


def test_aggregate_median_over_repeats() -> None:
    rows = [{"offset_ms": -40.0, "repeat": i, "status": "initialized", "n_map_ids": n, "tracking_ok_fraction": 0.9,
             "imu_initialized_fraction": f, "final_inertial_map": ok}
            for i, (n, f, ok) in enumerate([(5, 0.5, True), (9, 0.7, True), (7, 0.1, False)])]
    rows.append({"offset_ms": -40.0, "repeat": 3, "status": "failed", "n_map_ids": None, "tracking_ok_fraction": None,
                 "imu_initialized_fraction": None, "final_inertial_map": None})
    (entry,) = aggregate(rows)
    assert entry["n_done"] == 3 and entry["n_total"] == 4 and entry["n_final_inertial"] == 2
    assert entry["n_map_ids"] == 7 and entry["imu_initialized_fraction"] == 0.5


def test_dry_run_runs_nothing_and_real_run_reuses(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    tool, root = fake_tool(tmp_path), tmp_path / "sweep"
    kwargs = dict(tool=tool, segment=1, allow_legacy=False)
    assert run_sweep(Path("/r"), root, [-30.0, -45.0], 1, dry_run=True, **kwargs) == []
    assert not root.exists() and capsys.readouterr().out.count("[dry-run]") == 2
    rows = run_sweep(Path("/r"), root, [-30.0, -45.0], 2, dry_run=False, **kwargs)
    assert [r["status"] for r in rows] == ["initialized"] * 4
    assert [r["imu_initialized_fraction"] for r in rows] == [0.7, 0.7, 0.55, 0.55]
    again = run_sweep(Path("/r"), root, [-30.0, -45.0], 2, dry_run=True, **kwargs)
    assert len(again) == 4 and "[reuse]" in capsys.readouterr().out
