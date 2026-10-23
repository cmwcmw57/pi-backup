# ================================================================
# sim_vs_real.py
# 설명: 같은 경로의 Isaac 시뮬 trace.npz와 실기 trace(Pi run_policy npz 또는 run_real trace.npz)를 구간 라벨별 시간·이동거리·pitch 최대·정지 중 활주 거리로 나란히 비교한다.
# 사용법:
#   python -m experiments.yeouido_analysis.sim_vs_real --sim sim4/trace.npz --real run_real/trace.npz
#   Pi npz처럼 라벨이 없는 실기 로그: --real-labels labels.csv (열: label, t_start_s, t_end_s; sim 라벨 이름과 동일하게)
# ================================================================
import argparse
import math
from pathlib import Path

import numpy as np

from experiments.orbslam3_ros2.swerve import SwerveGeometry
from experiments.yeouido_analysis.common import SIM_CAVEAT, STOP_SPEED_M_S, format_table
from experiments.yeouido_analysis.slip import GEOMETRY_JSON, Trace, body_twists
from experiments.yeouido_analysis.trace_io import RunTrace, load_run_trace, with_label_table


def stationary_pairs(run: RunTrace, geometry: SwerveGeometry, min_stop_s: float) -> np.ndarray:
    """바퀴 속도가 STOP_SPEED_M_S 미만으로 min_stop_s 이상 이어진 표본 쌍(i, i+1) 마스크를 만든다."""
    still = geometry.wheel_radius_m * np.abs(run.wheel_rad_s).max(axis=1) < STOP_SPEED_M_S
    keep = np.zeros(len(still), dtype=bool)
    start = None
    for i, flag in enumerate(np.append(still, False)):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if run.t_s[i - 1] - run.t_s[start] >= min_stop_s:
                keep[start:i] = True
            start = None
    return keep[:-1] & keep[1:]


def label_stats(run: RunTrace, geometry: SwerveGeometry, min_stop_s: float = 0.5) -> dict[str, dict]:
    """라벨별 duration_s, dist_pose_m(3D 경로 길이), dist_kin_m(바퀴 운동학), pitch_max_deg, stop_slide_m."""
    labels = run.labels if run.labels is not None else np.full(len(run.t_s), "(all)")
    twist = body_twists(geometry, Trace(run.t_s, run.wheel_rad_s, run.steer_rad, None))
    speed = np.hypot(twist[:, 0], twist[:, 1])
    dt = np.diff(run.t_s)
    kin_step = 0.5 * (speed[:-1] + speed[1:]) * dt
    pose_step = np.linalg.norm(np.diff(run.xyz_m, axis=0), axis=1) if run.xyz_m is not None else None
    still = stationary_pairs(run, geometry, min_stop_s)
    stats = {}
    for label in dict.fromkeys(labels.tolist()):
        mask = labels[:-1] == label
        stats[label] = {
            "duration_s": float(dt[mask].sum()),
            "dist_pose_m": float(pose_step[mask].sum()) if pose_step is not None else math.nan,
            "dist_kin_m": float(kin_step[mask].sum()),
            "pitch_max_deg": (float(np.degrees(np.abs(run.pitch_rad[labels == label]).max()))
                              if run.pitch_rad is not None else math.nan),
            "stop_slide_m": float(pose_step[mask & still].sum()) if pose_step is not None else math.nan,
        }
    return stats


def compare(sim: dict[str, dict], real: dict[str, dict]) -> list[dict]:
    """라벨별 sim/real 지표를 한 행으로 합친다(한쪽에만 있는 라벨은 비워 둔다)."""
    rows = []
    for label in dict.fromkeys([*sim, *real]):
        row = {"label": label}
        for tag, stats in (("sim", sim), ("real", real)):
            for key in ("duration_s", "dist_pose_m", "dist_kin_m", "pitch_max_deg", "stop_slide_m"):
                row[f"{key}_{tag}"] = stats.get(label, {}).get(key)
        rows.append(row)
    return rows


def main() -> None:
    """시뮬·실기 trace를 읽어 구간 라벨별 비교표를 출력한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", type=Path, required=True)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--real-labels", type=Path, help="라벨 없는 실기 npz용 label,t_start_s,t_end_s CSV")
    parser.add_argument("--geometry", type=Path, default=GEOMETRY_JSON)
    parser.add_argument("--min-stop-s", type=float, default=0.5, help="정지로 인정할 최소 시간(초)")
    args = parser.parse_args()
    geometry = SwerveGeometry.from_json(args.geometry)
    real = load_run_trace(args.real)
    if args.real_labels:
        real = with_label_table(real, args.real_labels)
    rows = compare(label_stats(load_run_trace(args.sim), geometry, args.min_stop_s),
                   label_stats(real, geometry, args.min_stop_s))
    print(f"sim={args.sim}\nreal={args.real}\n{SIM_CAVEAT} 두 열을 나란히 보는 비교용이며 합치거나 보정하지 않는다.")
    print("dist_pose=위치 3D 경로 길이(없으면 -), dist_kin=바퀴 운동학 거리, pitch는 |max|(실기는 가속도 기반·동적 가속 포함), stop_slide=바퀴 정지 중 위치 이동")
    names = ["duration_s", "dist_pose_m", "dist_kin_m", "pitch_max_deg", "stop_slide_m"]
    print(format_table(rows, ["label"] + [f"{n}_{t}" for n in names for t in ("sim", "real")]))


if __name__ == "__main__":
    main()
