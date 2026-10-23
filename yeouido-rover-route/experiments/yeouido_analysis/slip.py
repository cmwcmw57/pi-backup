# ================================================================
# slip.py
# 설명: 바퀴 속도·조향각 trace에서 swerve 운동학 이동 거리를 구하고, 줄자 실측 체크포인트와 비교해 구간별 슬립률(1-실제/운동학)을 계산한다.
# 사용법:
#   python -m experiments.yeouido_analysis.slip --trace trace.csv|ik_1.npz --checkpoints cps.csv --source real
#   trace 열: t_s, wheel_{fl,fr,rl,rr}_rad_s, steer_{fl,fr,rl,rr}_rad (+선택 ref_x_m, ref_y_m)
#   체크포인트 열: checkpoint_id, measured_x_m, measured_y_m (+선택 t_s, ramp_deg, speed_m_s)
# ================================================================
import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiments.orbslam3_ros2.swerve import SwerveGeometry
from experiments.yeouido_analysis.trace_io import load_run_trace
from experiments.yeouido_analysis.common import (
    Checkpoint,
    SIM_CAVEAT,
    detect_stops,
    format_table,
    load_checkpoints,
    speed_bin,
    tape_distances,
)

GEOMETRY_JSON = (
    Path(__file__).parents[2] / "ros2_ws/src/rover_description/config/swerve_geometry.json"
)
MOVING_SPEED_M_S = 0.02
STRAIGHT_YAW_LIMIT_RAD = math.radians(5.0)
trapezoid = getattr(np, "trapezoid", None) or np.trapz


@dataclass(frozen=True)
class Trace:
    """시각, 바퀴 속도(rad/s) 4개, 조향각(rad) 4개와 선택 기준 위치(N×2|3)·펌웨어 slip(N×4)을 보관한다."""

    t_s: np.ndarray
    wheel_rad_s: np.ndarray
    steer_rad: np.ndarray
    ref_pos_m: np.ndarray | None
    log_slip: np.ndarray | None = None


def module_tags(geometry: SwerveGeometry) -> list[str]:
    """geometry module 순서대로 front_left→fl 같은 두 글자 접미사를 만든다."""
    return ["".join(word[0] for word in module.name.split("_")) for module in geometry.modules]


def load_trace(path: Path, geometry: SwerveGeometry) -> Trace:
    """trace CSV를 읽어 시각 단조 증가와 유한값을 검증한다."""
    tags = module_tags(geometry)
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames or []
        needed = ["t_s"] + [f"wheel_{t}_rad_s" for t in tags] + [f"steer_{t}_rad" for t in tags]
        missing = [name for name in needed if name not in names]
        if missing:
            raise ValueError(f"trace CSV missing columns: {missing}")
        has_ref = "ref_x_m" in names and "ref_y_m" in names
        rows = list(reader)
    if len(rows) < 2:
        raise ValueError("trace needs at least two samples")
    column = lambda name: np.array([float(row[name]) for row in rows])  # noqa: E731
    t = column("t_s")
    wheel = np.stack([column(f"wheel_{tag}_rad_s") for tag in tags], axis=1)
    steer = np.stack([column(f"steer_{tag}_rad") for tag in tags], axis=1)
    ref = np.stack([column("ref_x_m"), column("ref_y_m")], axis=1) if has_ref else None
    if not (np.isfinite(t).all() and np.isfinite(wheel).all() and np.isfinite(steer).all()):
        raise ValueError("trace contains non-finite values")
    if np.any(np.diff(t) <= 0):
        raise ValueError("trace t_s must be strictly increasing")
    return Trace(t_s=t, wheel_rad_s=wheel, steer_rad=steer, ref_pos_m=ref)


def load_trace_npz(path: Path) -> Trace:
    """Pi run_policy npz(wheel_mps, steer_meas, host_ns, slip) 또는 Isaac/run_real trace.npz를 읽는다."""
    run = load_run_trace(path)
    return Trace(t_s=run.t_s, wheel_rad_s=run.wheel_rad_s, steer_rad=run.steer_rad,
                 ref_pos_m=run.xyz_m, log_slip=run.log_slip)


def body_twists(geometry: SwerveGeometry, trace: Trace) -> np.ndarray:
    """네 module 속도 벡터에서 최소제곱으로 base_link (vx, vy, yaw_rate) 열을 구한다."""
    design = np.zeros((8, 3))
    for i, module in enumerate(geometry.modules):
        design[2 * i] = [1.0, 0.0, -module.y_m]
        design[2 * i + 1] = [0.0, 1.0, module.x_m]
    speed = geometry.wheel_radius_m * trace.wheel_rad_s
    observed = np.stack([speed * np.cos(trace.steer_rad), speed * np.sin(trace.steer_rad)], axis=2)
    return observed.reshape(len(speed), 8) @ np.linalg.pinv(design).T


def checkpoint_times(points: list[Checkpoint], t_s: np.ndarray, speed_m_s: np.ndarray,
                     min_stop_s: float) -> list[float]:
    """체크포인트 t_s 열이 있으면 그대로, 없으면 정지 구간(speed_m_s 기준)을 순서대로 대응시킨다."""
    if all(point.t_s is not None for point in points):
        return [float(point.t_s) for point in points]
    stops = detect_stops(t_s, speed_m_s, min_stop_s)
    if len(stops) != len(points):
        raise ValueError(
            f"detected {len(stops)} stops (>= {min_stop_s}s) but {len(points)} checkpoints; "
            "add a t_s column to the checkpoint CSV"
        )
    return stops


def analyze(geometry: SwerveGeometry, trace: Trace, points: list[Checkpoint], *,
            measured_frame: str = "surface", min_dist_m: float = 0.2,
            min_speed_m_s: float = 0.05, min_stop_s: float = 1.5) -> list[dict]:
    """체크포인트 구간마다 운동학 거리·줄자 거리·슬립률과 바퀴별 슬립을 계산한다."""
    twist = body_twists(geometry, trace)
    speed = np.hypot(twist[:, 0], twist[:, 1])
    # 정지는 바퀴 기준으로 판정한다(제자리 회전은 선속도가 0이어도 바퀴가 돈다).
    wheel_speed = geometry.wheel_radius_m * np.abs(trace.wheel_rad_s).max(axis=1)
    times = checkpoint_times(points, trace.t_s, wheel_speed, min_stop_s)
    tape = tape_distances(points, measured_frame)
    radius = geometry.wheel_radius_m
    rows = []
    for k in range(1, len(points)):
        inside = (trace.t_s >= times[k - 1]) & (trace.t_s <= times[k])
        t = trace.t_s[inside]
        row = {"segment": f"{points[k - 1].checkpoint_id}->{points[k].checkpoint_id}",
               "ramp_deg": points[k].ramp_deg, "tape_m": tape[k - 1]}
        if len(t) < 2:
            rows.append({**row, "valid": False, "reason": "no_samples"})
            continue
        kin = float(trapezoid(speed[inside], t))
        moving_s = float(np.sum(np.diff(t) * (speed[inside][1:] > MOVING_SPEED_M_S)))
        mean_speed = kin / moving_s if moving_s > 0 else 0.0
        yaw_change = float(trapezoid(twist[inside, 2], t))
        row.update(kin_m=kin, mean_speed_m_s=mean_speed, slip=1 - tape[k - 1] / kin if kin > 0 else math.nan)
        row["speed_m_s"] = (points[k].speed_m_s if points[k].speed_m_s is not None
                            else speed_bin(mean_speed))
        reasons = []
        if kin < min_dist_m:
            reasons.append("short_distance")
        if mean_speed < min_speed_m_s:
            reasons.append("low_speed")
        row.update(valid=not reasons, reason=",".join(reasons))
        if abs(yaw_change) < STRAIGHT_YAW_LIMIT_RAD and kin > 0:
            for i, tag in enumerate(module_tags(geometry)):
                rolled = float(trapezoid(radius * np.abs(trace.wheel_rad_s[inside, i]), t))
                row[f"slip_{tag}"] = 1 - tape[k - 1] / rolled if rolled > 0 else math.nan
        if trace.ref_pos_m is not None:
            ref = trace.ref_pos_m
            first, last = np.searchsorted(trace.t_s, [times[k - 1], times[k]])
            row["ref_m"] = float(np.linalg.norm(ref[min(last, len(ref) - 1)] - ref[min(first, len(ref) - 1)]))
            row["slip_ref"] = 1 - row["ref_m"] / kin if kin > 0 else math.nan
        if trace.log_slip is not None:
            # 펌웨어 'slip'은 VO 기준 헛돎 추정이라 줄자 슬립과 정의가 다르다. 참고 열로만 둔다.
            finite = trace.log_slip[inside][np.isfinite(trace.log_slip[inside])]
            row["fw_slip_mean"] = float(finite.mean()) if finite.size else math.nan
        rows.append(row)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """유효 구간을 (ramp_deg, speed_m_s)별로 묶어 평균·표준편차·최소·최대·합산 슬립을 만든다."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        if row.get("valid"):
            groups.setdefault((row["ramp_deg"], row["speed_m_s"]), []).append(row)
    table = []
    for (ramp, speed), items in sorted(groups.items(), key=lambda g: (g[0][0] is None, g[0])):
        slips = np.array([r["slip"] for r in items])
        pooled = 1 - sum(r["tape_m"] for r in items) / sum(r["kin_m"] for r in items)
        table.append({"ramp_deg": ramp, "speed_m_s": speed, "n": len(items),
                      "slip_mean": float(slips.mean()), "slip_std": float(slips.std()),
                      "slip_min": float(slips.min()), "slip_max": float(slips.max()),
                      "slip_pooled": pooled})
    return table


def main() -> None:
    """trace와 체크포인트 CSV를 읽어 구간별·그룹별 슬립 표를 출력한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True, help=".csv(제안 형식) 또는 .npz(Pi run_policy·Isaac trace)")
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--source", choices=["real", "sim"], required=True,
                        help="실측/시뮬 구분(표 헤더에 표시). 두 값을 한 표에 섞어 해석하지 않는다")
    parser.add_argument("--geometry", type=Path, default=GEOMETRY_JSON)
    parser.add_argument("--measured-frame", choices=["surface", "plan"], default="surface",
                        help="줄자를 경사면 따라 쟀으면 surface, 수평 투영이면 plan(ramp_deg로 1/cosθ 보정)")
    parser.add_argument("--min-dist-m", type=float, default=0.2)
    parser.add_argument("--min-speed-m-s", type=float, default=0.05)
    parser.add_argument("--min-stop-s", type=float, default=1.5)
    parser.add_argument("--out", type=Path, help="구간별 결과 CSV 저장 경로")
    args = parser.parse_args()
    geometry = SwerveGeometry.from_json(args.geometry)
    trace = load_trace_npz(args.trace) if args.trace.suffix == ".npz" else load_trace(args.trace, geometry)
    rows = analyze(geometry, trace, load_checkpoints(args.checkpoints),
                   measured_frame=args.measured_frame, min_dist_m=args.min_dist_m,
                   min_speed_m_s=args.min_speed_m_s, min_stop_s=args.min_stop_s)
    tags = module_tags(geometry)
    print(f"[source={args.source}] measured_frame={args.measured_frame} wheel_radius_m={geometry.wheel_radius_m}")
    if args.source == "sim":
        print(SIM_CAVEAT)
    segment_columns = (["segment", "ramp_deg", "speed_m_s", "tape_m", "kin_m", "slip"]
                       + [f"slip_{t}" for t in tags] + ["slip_ref", "fw_slip_mean", "valid", "reason"])
    print(format_table(rows, segment_columns))
    print()
    print(format_table(summarize(rows), ["ramp_deg", "speed_m_s", "n", "slip_mean", "slip_std",
                                         "slip_min", "slip_max", "slip_pooled"]))
    if args.out:
        with args.out.open("w", newline="", encoding="utf-8") as stream:
            names = sorted({key for row in rows for key in row}, key=lambda k: (k not in segment_columns, k))
            writer = csv.DictWriter(stream, fieldnames=names)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
