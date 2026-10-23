# ================================================================
# follower_metrics.py
# 설명: 시뮬 run의 telemetry.jsonl에서 추종기 비교 지표(GT 각속도, 시작 후 회전, 추정 오차, 게이트 정지, 추정기 degraded, 완주·벽 거리)를 뽑아 표로 보인다.
# 사용법: python -m experiments.site_sim.follower_metrics MISSION_FOLDER RUN_DIR [RUN_DIR ...]
#   또는: from experiments.site_sim.follower_metrics import summarize
# ================================================================
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiments.site_sim import clearance

DEGRADED_STATUS = ('imu_gap', 'recovering')
TABLE_COLUMNS = ('completed', 'duration_sim_s', 'min_clearance_m', 'contact_samples', 'omega_mean_rad_s', 'omega_max_rad_s',
                 'over_limit_fraction', 'moving_s', 'excluded_s', 'rotation_after_start_deg', 'max_estimate_error_m',
                 'estimator_degraded_s', 'gate_stop_s_total', 'gate_resume_max', 'realign_stop_s')


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """가장자리는 유효한 표본만으로 정규화한 이동평균."""
    kernel = np.ones(window)
    return np.convolve(values, kernel, 'same') / np.convolve(np.ones(len(values)), kernel, 'same')


def summarize(output: Path, mission_folder: Path | None = None, *, omega_limit: float = .11, smooth_s: float = .5,
              gap_exclude_deg: float = 60., moving_speed: float = .01) -> dict:
    """한 run의 추종기 비교 지표. 이동 중 = 적용 명령 hypot(vx, vy) > moving_speed, 각속도는 GT yaw 기준."""
    rows = [json.loads(line) for line in (Path(output) / 'telemetry.jsonl').open() if line.strip()]
    t = np.array([r['stamp_s'] for r in rows], float)
    dt = np.r_[np.diff(t), 0.] if len(t) else t
    w, x, y, z = np.array([r['quaternion_wxyz'] for r in rows], float).T
    yaw_raw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    yaw = np.unwrap(yaw_raw)
    omega = np.gradient(yaw, t) if len(t) > 1 else np.zeros(len(t))
    omega = np.abs(smooth(omega, max(1, int(round(smooth_s / np.median(np.diff(t))))))) if len(t) > 1 else omega
    speed = np.array([math.hypot(r['applied_command']['vx_m_s'], r['applied_command'].get('vy_m_s', 0.)) for r in rows])
    moving = speed > moving_speed
    follower = [r['mission'].get('follower') for r in rows]
    gap = np.array([f['heading_gap_rad'] if f else 0. for f in follower])
    excluded = moving & (gap > math.radians(gap_exclude_deg))
    counted = moving & ~excluded
    first = int(np.argmax(moving)) if moving.any() else None
    estimated = np.array([[i, np.linalg.norm(np.asarray(r['estimated_position_m'][:2]) - np.asarray(r['position_m'][:2]))]
                          for i, r in enumerate(rows) if 'estimated_position_m' in r])
    degraded = np.array([r.get('estimator_status') in DEGRADED_STATUS or r.get('estimator_quality') == 'degraded' for r in rows])
    stop_s, resumes, segments, seen_before = {}, {}, [], {}
    previous, last_gated = None, {}
    for i, r in enumerate(rows):   # 연속한 같은 (state, route)를 한 구간으로, 같은 이름이 다시 나오면 #1, #2
        pair = (r['mission']['state'], r['mission']['route'])
        if pair != previous:
            seen_before[pair] = seen_before.get(pair, -1) + 1
            previous = pair
        key = f'{pair[0]}:{pair[1]}#{seen_before[pair]}'
        segments.append(key)
        if follower[i] is None:continue
        stop_s.setdefault(key, 0.);resumes.setdefault(key, 0)
        gated = bool(follower[i]['gated'])
        if gated:stop_s[key] += float(dt[i])
        if last_gated.get(key) and not gated:resumes[key] += 1
        last_gated[key] = gated
    summary = {'omega_mean_rad_s': float(omega[moving].mean()) if moving.any() else 0.,
               'omega_max_rad_s': float(omega[moving].max()) if moving.any() else 0.,
               'over_limit_fraction': float((omega[counted] > omega_limit).mean()) if counted.any() else 0.,
               'moving_s': float(dt[moving].sum()), 'excluded_s': float(dt[excluded].sum()),
               'rotation_after_start_deg': 0. if first is None else float(np.degrees(np.abs(
                   np.arctan2(np.sin(yaw_raw[first:] - yaw_raw[first]), np.cos(yaw_raw[first:] - yaw_raw[first]))).max())),
               'max_estimate_error_m': float(estimated[:, 1].max()) if len(estimated) else None,
               'estimator_degraded_s': float(dt[degraded].sum()),
               'realign_stop_s': float(dt[np.array([bool(f and f.get('realigning')) for f in follower])].sum()),
               'gate_stop_s_by_phase': stop_s, 'gate_resume_by_phase': resumes, 'gate_stop_s_total': float(sum(stop_s.values()))}
    if mission_folder is not None:
        merged = clearance.evaluate(Path(output), Path(mission_folder))
        summary.update({k: merged[k] for k in ('completed', 'min_clearance_m', 'contact_samples', 'duration_sim_s')})
    return summary


def format_table(summaries: dict[str, dict]) -> str:
    """run 이름 → 지표 dict를 열 정렬한 비교표로 만든다(없는 열은 생략)."""
    columns = [c for c in TABLE_COLUMNS if c == 'gate_resume_max' or any(c in s for s in summaries.values())]
    def cell(summary, column):
        value = max(summary['gate_resume_by_phase'].values(), default=0) if column == 'gate_resume_max' else summary.get(column)
        return '-' if value is None else f'{value:.4g}' if isinstance(value, float) else str(value)
    table = [['run', *columns]] + [[name, *(cell(s, c) for c in columns)] for name, s in summaries.items()]
    widths = [max(len(row[i]) for row in table) for i in range(len(table[0]))]
    return '\n'.join('  '.join(text.ljust(width) for text, width in zip(row, widths)).rstrip() for row in table)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='추종기 비교 지표 표')
    parser.add_argument('mission_folder', type=Path)
    parser.add_argument('runs', type=Path, nargs='+')
    args = parser.parse_args(argv)
    print(format_table({run.name: summarize(run, args.mission_folder) for run in args.runs}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
