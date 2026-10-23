# ================================================================
# orb_retention.py
# 설명: ORB-SLAM3 tracking state 로그(2=OK, 3=RECENTLY_LOST, 4=LOST)에서 체크포인트 구간·속도별 추적 유지율과 손실 횟수를 구하고, 체크포인트 간 ORB 이동 거리를 줄자 실측과 비교한다.
# 사용법:
#   python -m experiments.yeouido_analysis.orb_retention --tracking orb_live/live_poses.csv --checkpoints cps.csv --source real
#   tracking 열: stamp_ns 또는 timestamp_ns, state (+선택 map_id, m03/m13/m23 = world_from_camera 위치)
#   체크포인트 열: checkpoint_id, measured_x_m, measured_y_m (+선택 t_s, ramp_deg, speed_m_s). t_s는 tracking 시각(초) 기준, 없으면 ORB 위치가 멎은 구간을 순서대로 대응
# ================================================================
import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiments.yeouido_analysis.common import (
    SIM_CAVEAT,
    Checkpoint,
    detect_stops,
    format_table,
    load_checkpoints,
    speed_bin,
    tape_distances,
)

STATE_OK = 2
POSE_COLUMNS = ("m03", "m13", "m23")
MAX_POSE_GAP_S = 0.5
MOVING_SPEED_M_S = 0.02


@dataclass(frozen=True)
class TrackingLog:
    """프레임별 시각(초), tracking state, map_id와 선택 3D 위치(없으면 None)를 보관한다."""

    t_s: np.ndarray
    state: np.ndarray
    map_id: np.ndarray
    position_m: np.ndarray | None


def load_tracking(path: Path, relative_time: bool = False) -> TrackingLog:
    """live_poses.csv·timings.csv·headless tracking.csv 형식의 로그를 읽는다."""
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames or []
        time_column = next((n for n in ("stamp_ns", "timestamp_ns") if n in names), None)
        if time_column is None or "state" not in names:
            raise ValueError("tracking CSV needs stamp_ns|timestamp_ns and state columns")
        rows = list(reader)
    if not rows:
        raise ValueError("tracking CSV is empty")
    stamp = np.array([int(row[time_column]) for row in rows], dtype=np.int64)
    if np.any(np.diff(stamp) <= 0):
        raise ValueError("tracking timestamps must be strictly increasing")
    t = (stamp - (stamp[0] if relative_time else 0)) * 1e-9
    map_id = np.array([int(row["map_id"]) if "map_id" in names else 0 for row in rows])
    position = None
    if all(name in names for name in POSE_COLUMNS):
        position = np.array([[float(row[name]) for name in POSE_COLUMNS] for row in rows])
    return TrackingLog(t_s=t, state=np.array([int(row["state"]) for row in rows]), map_id=map_id,
                       position_m=position)


def position_speed(log: TrackingLog) -> np.ndarray:
    """연속한 유효(state 2·유한·같은 map) 프레임 쌍의 ORB 위치 속도(m/s)를 구한다(유효 쌍이 없는 프레임은 inf)."""
    speed = np.full(len(log.t_s), np.inf)
    if log.position_m is None:
        return speed
    valid = (log.state == STATE_OK) & np.isfinite(log.position_m).all(axis=1)
    for i in np.flatnonzero(valid[:-1] & valid[1:] & (log.map_id[:-1] == log.map_id[1:])):
        v = np.linalg.norm(log.position_m[i + 1] - log.position_m[i]) / (log.t_s[i + 1] - log.t_s[i])
        speed[i], speed[i + 1] = min(speed[i], v), min(speed[i + 1], v)
    return speed


def checkpoint_times(log: TrackingLog, points: list[Checkpoint], min_stop_s: float) -> list[float]:
    """체크포인트 t_s 열이 있으면 그대로, 없으면 ORB 위치가 멎은 구간을 순서대로 대응시킨다."""
    if all(point.t_s is not None for point in points):
        return [float(point.t_s) for point in points]
    if log.position_m is None:
        raise ValueError("checkpoint t_s is missing and the tracking log has no m03/m13/m23 pose to detect stops")
    stops = detect_stops(log.t_s, position_speed(log), min_stop_s)
    if len(stops) != len(points):
        raise ValueError(f"detected {len(stops)} ORB stops (>= {min_stop_s}s) but {len(points)} checkpoints; "
                         "add a t_s column to the checkpoint CSV")
    return stops


def nearest_pose(log: TrackingLog, t_s: float) -> tuple[int, np.ndarray] | None:
    """t_s에 가장 가까운 state 2·유한 위치 프레임의 (map_id, 위치)를 반환하고, 0.5 s 안에 없으면 None."""
    if log.position_m is None:
        return None
    good = np.flatnonzero((log.state == STATE_OK) & np.isfinite(log.position_m).all(axis=1))
    if len(good) == 0:
        return None
    best = good[np.argmin(np.abs(log.t_s[good] - t_s))]
    if abs(log.t_s[best] - t_s) > MAX_POSE_GAP_S:
        return None
    return int(log.map_id[best]), log.position_m[best]


def analyze(log: TrackingLog, points: list[Checkpoint], *, measured_frame: str = "surface",
            min_stop_s: float = 1.5) -> list[dict]:
    """체크포인트 구간별 state 비율·손실 횟수·map 변경 수와 ORB 대 줄자 이동 거리를 계산한다."""
    times = checkpoint_times(log, points, min_stop_s)
    speed = position_speed(log)
    tape = tape_distances(points, measured_frame)
    rows = []
    for k in range(1, len(points)):
        start, end = times[k - 1], times[k]
        inside = (log.t_s >= start) & (log.t_s <= end)
        state, maps = log.state[inside], log.map_id[inside]
        row = {"segment": f"{points[k - 1].checkpoint_id}->{points[k].checkpoint_id}",
               "ramp_deg": points[k].ramp_deg, "speed_m_s": points[k].speed_m_s,
               "frames": int(inside.sum()), "tape_m": tape[k - 1]}
        if not inside.any():
            rows.append(row)
            continue
        row.update(
            ok_frames=int((state == STATE_OK).sum()),
            ok_fraction=float((state == STATE_OK).mean()),
            recent_lost_fraction=float((state == 3).mean()),
            lost_fraction=float((state == 4).mean()),
            loss_events=int(np.sum((state[:-1] == STATE_OK) & (state[1:] != STATE_OK))),
            map_changes=int(np.sum(maps[maps >= 0][1:] != maps[maps >= 0][:-1])),
        )
        first, last = nearest_pose(log, start), nearest_pose(log, end)
        if first and last and first[0] == last[0]:
            orb = float(np.linalg.norm(last[1] - first[1]))
            row.update(orb_m=orb, err_m=orb - tape[k - 1], err_pct=100 * (orb - tape[k - 1]) / tape[k - 1])
            moving_s = float(np.sum(np.diff(log.t_s[inside]) * (speed[inside][1:] > MOVING_SPEED_M_S)
                                    * np.isfinite(speed[inside][1:])))
            if points[k].speed_m_s is None and moving_s > 0:
                row["speed_m_s"] = speed_bin(orb / moving_s)
        rows.append(row)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """구간을 (ramp_deg, speed_m_s)별로 묶어 프레임 가중 유지율·손실 횟수·평균 절대 위치 오차(%)를 만든다."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        if row.get("frames"):
            groups.setdefault((row["ramp_deg"], row["speed_m_s"]), []).append(row)
    table = []
    for (ramp, speed), items in sorted(groups.items(), key=lambda g: tuple(math.inf if v is None else v for v in g[0])):
        frames = sum(r["frames"] for r in items)
        errors = [abs(r["err_pct"]) for r in items if "err_pct" in r]
        table.append({"ramp_deg": ramp, "speed_m_s": speed, "segments": len(items), "frames": frames,
                      "ok_fraction": sum(r["ok_frames"] for r in items) / frames,
                      "loss_events": sum(r["loss_events"] for r in items),
                      "map_changes": sum(r["map_changes"] for r in items),
                      "abs_err_pct_mean": float(np.mean(errors)) if errors else None,
                      "n_pos": len(errors)})
    return table


def main() -> None:
    """tracking 로그와 체크포인트 CSV를 읽어 구간별·그룹별 유지율 표를 출력한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--source", choices=["real", "sim"], required=True,
                        help="실측/시뮬 구분(표 헤더에 표시). 두 값을 한 표에 섞어 해석하지 않는다")
    parser.add_argument("--measured-frame", choices=["surface", "plan"], default="surface",
                        help="줄자를 경사면 따라 쟀으면 surface, 수평 투영이면 plan(ramp_deg로 1/cosθ 보정)")
    parser.add_argument("--relative-time", action="store_true",
                        help="tracking 시각을 첫 프레임 기준 초로 환산(체크포인트 t_s가 상대 시각일 때)")
    parser.add_argument("--min-stop-s", type=float, default=1.5,
                        help="t_s가 없을 때 ORB 정지로 인정할 최소 시간(초)")
    parser.add_argument("--out", type=Path, help="구간별 결과 CSV 저장 경로")
    args = parser.parse_args()
    rows = analyze(load_tracking(args.tracking, args.relative_time), load_checkpoints(args.checkpoints),
                   measured_frame=args.measured_frame, min_stop_s=args.min_stop_s)
    print(f"[source={args.source}] measured_frame={args.measured_frame} tracking={args.tracking}")
    if args.source == "sim":
        print(SIM_CAVEAT)
    segment_columns = ["segment", "ramp_deg", "speed_m_s", "frames", "ok_fraction", "recent_lost_fraction",
                       "lost_fraction", "loss_events", "map_changes", "tape_m", "orb_m", "err_m", "err_pct"]
    print(format_table(rows, segment_columns))
    print()
    print(format_table(summarize(rows), ["ramp_deg", "speed_m_s", "segments", "frames", "ok_fraction",
                                         "loss_events", "map_changes", "abs_err_pct_mean", "n_pos"]))
    if args.out:
        with args.out.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=segment_columns + ["ok_frames"], extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
