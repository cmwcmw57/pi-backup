# ================================================================
# time_offset_sweep.py
# 설명: 같은 RUN에 IMU 시각 보정(--correction-s)을 −30~−55 ms로 바꿔 tools/imu_fusion.sh(offline ORB-SLAM3 관성 재생)를 돌리고, 관성 초기화 유지 비율·map ID 수를 비교한다.
# 사용법:
#   python -m experiments.yeouido_analysis.time_offset_sweep RUN --segment 1 --dry-run
#   python -m experiments.yeouido_analysis.time_offset_sweep RUN --segment 1 --repeats 3 --offsets-ms -30 -35 -37.5 -40 -45 -47 -50 -55
#   보정 부호: aligned_imu_time = fitted_imu_time + correction (−45 ms는 IMU 시각을 45 ms 앞당김)
# ================================================================
import argparse
import csv
import json
import statistics
import subprocess
import time
from pathlib import Path

from experiments.yeouido_analysis.common import SIM_CAVEAT, format_table

ROOT = Path(__file__).parents[2]
DEFAULT_OFFSETS_MS = [-30.0, -35.0, -40.0, -45.0, -50.0, -55.0]
FINISHED = ("initialized", "not_initialized")
RESULT_COLUMNS = ["offset_ms", "repeat", "status", "frames", "tracking_ok_fraction",
                  "imu_initialized_fraction", "n_map_ids", "final_inertial_map", "error"]


def run_name(offset_ms: float, repeat: int) -> str:
    """offset·반복 번호로 출력 폴더 이름(예: off-35ms_r0)을 만든다."""
    return f"off{offset_ms:+g}ms_r{repeat}"


def build_command(tool: Path, run: Path, out_dir: Path, offset_ms: float,
                  segment: int | None, allow_legacy: bool) -> list[str]:
    """tools/imu_fusion.sh를 수동 시각 보정으로 호출하는 명령을 만든다(--auto와는 함께 쓸 수 없음)."""
    command = ["bash", str(tool), str(run), "--output", str(out_dir),
               "--correction-s", f"{offset_ms / 1000:.6f}"]
    if segment is not None:
        command += ["--segment", str(segment)]
    if allow_legacy:
        command.append("--allow-legacy")
    return command


def parse_summary(out_dir: Path, offset_ms: float, repeat: int) -> dict:
    """재생 폴더의 summary.json을 한 행으로 읽고, 요청 offset과 실제 적용 보정이 같은지 확인한다."""
    path = out_dir / "summary.json"
    if not path.is_file():
        return {"offset_ms": offset_ms, "repeat": repeat, "status": "no_summary"}
    summary = json.loads(path.read_text(encoding="utf-8"))
    row = {"offset_ms": offset_ms, "repeat": repeat, "status": summary.get("status", "unknown"),
           "frames": summary.get("frames"),
           "tracking_ok_fraction": summary.get("tracking_ok_fraction"),
           "imu_initialized_fraction": summary.get("imu_initialized_fraction"),
           "n_map_ids": len(summary["map_ids"]) if "map_ids" in summary else None,
           "final_inertial_map": summary.get("final_export_has_inertial_map"),
           "error": summary.get("error")}
    applied = summary.get("imu_time_correction_s")
    if row["status"] in FINISHED and (applied is None or abs(applied * 1000 - offset_ms) > 1e-3):
        row.update(status="offset_mismatch", error=f"applied correction {applied} s != requested {offset_ms} ms")
    return row


def aggregate(rows: list[dict]) -> list[dict]:
    """offset별로 완료한 반복의 중앙값과 최종 관성 map 보유 횟수를 요약한다."""
    table = []
    for offset in sorted({row["offset_ms"] for row in rows}):
        done = [r for r in rows if r["offset_ms"] == offset and r["status"] in FINISHED]
        entry = {"offset_ms": offset, "n_done": len(done), "n_total": sum(r["offset_ms"] == offset for r in rows),
                 "n_final_inertial": sum(bool(r["final_inertial_map"]) for r in done)}
        for key in ("tracking_ok_fraction", "imu_initialized_fraction", "n_map_ids"):
            values = [r[key] for r in done if r[key] is not None]
            entry[key] = float(statistics.median(values)) if values else None
        table.append(entry)
    return table


def run_sweep(run: Path, out_root: Path, offsets_ms: list[float], repeats: int, *, tool: Path,
              segment: int | None, allow_legacy: bool, dry_run: bool, timeout_s: int = 3600) -> list[dict]:
    """offset×반복마다 재생을 실행(dry-run이면 명령만 출력)하고 완료된 폴더는 재사용해 결과 행을 모은다."""
    rows = []
    for offset in offsets_ms:
        for repeat in range(repeats):
            name = run_name(offset, repeat)
            out_dir = out_root / name
            if (out_dir / "summary.json").is_file() and parse_summary(out_dir, offset, repeat)["status"] in FINISHED:
                print(f"[reuse] {name}")
            else:
                command = build_command(tool, run, out_dir, offset, segment, allow_legacy)
                if dry_run:
                    print("[dry-run]", " ".join(command))
                    continue
                out_root.mkdir(parents=True, exist_ok=True)
                with (out_root / f"{name}.log").open("wb") as log:
                    started = time.monotonic()
                    result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=log, timeout=timeout_s)
                print(f"[run] {name} exit={result.returncode} {time.monotonic() - started:.0f}s", flush=True)
            rows.append(parse_summary(out_dir, offset, repeat))
    return rows


def main() -> None:
    """sweep을 실행(또는 dry-run)하고 offset별 비교표와 sweep_results.csv를 출력한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="종료된 handheld_mapping RUN 폴더(imu.jsonl·frames.jsonl 포함)")
    parser.add_argument("--offsets-ms", type=float, nargs="+", default=DEFAULT_OFFSETS_MS)
    parser.add_argument("--repeats", type=int, default=1,
                        help="비동기 재생이라 map ID·keyframe이 실행마다 달라지므로 offset당 반복 권장")
    parser.add_argument("--segment", type=int, help="MCU reset이 있는 RUN에서 사용할 IMU segment")
    parser.add_argument("--allow-legacy", action="store_true")
    parser.add_argument("--out-root", type=Path,
                        help="결과 루트(기본: outputs/yeouido_analysis/time_offset_sweep/<시각>)")
    parser.add_argument("--tool", type=Path, default=ROOT / "tools/imu_fusion.sh")
    parser.add_argument("--dry-run", action="store_true", help="명령만 출력하고 이미 끝난 폴더만 집계")
    args = parser.parse_args()
    out_root = args.out_root or ROOT / "outputs/yeouido_analysis/time_offset_sweep" / time.strftime("%Y%m%d_%H%M%S")
    rows = run_sweep(args.run.resolve(), out_root.resolve(), args.offsets_ms, args.repeats, tool=args.tool,
                     segment=args.segment, allow_legacy=args.allow_legacy, dry_run=args.dry_run)
    if not rows:
        return
    print(f"\ninput RUN (recorded data): {args.run}")
    print(SIM_CAVEAT + " 이 표는 실측 RUN 입력이며, 비동기 재생이라 1회 실행으로는 offset을 가르지 못한다.")
    print(format_table(aggregate(rows), ["offset_ms", "n_done", "n_total", "n_final_inertial",
                                         "imu_initialized_fraction", "tracking_ok_fraction", "n_map_ids"]))
    if not args.dry_run:
        with (out_root / "sweep_results.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=RESULT_COLUMNS)
            writer.writeheader()
            writer.writerows({key: row.get(key) for key in RESULT_COLUMNS} for row in rows)


if __name__ == "__main__":
    main()
