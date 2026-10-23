# ================================================================
# clearance.py
# 설명: 한 번의 시뮬 run이 임무를 끝냈는지와 정답(GT) 위치가 벽·단차에 얼마나 가까웠는지를 요약한다. 위치 추정 정확도가 아니라 "완주와 벽 비충돌"을 평가한다.
# 사용법: python -m experiments.site_sim.clearance RUN/output MISSION_FOLDER   (JSON 한 줄 출력)
# ================================================================
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

BODY_RADIUS_M = 0.195      # 실제 로버 284×263 mm의 외접원 반지름: 이보다 가까우면 접촉으로 본다
PLANNING_RADIUS_M = 0.23   # 임무 계획이 쓰는 로버 여유 반경(simulation.json footprint_radius_m)


def clearance_map(blocked: np.ndarray, resolution_m: float) -> np.ndarray:
    """각 칸에서 가장 가까운 막힌 칸(벽·단차)까지의 거리 [m]. 막힌 칸이 없으면 무한대."""
    if not blocked.any():
        return np.full(blocked.shape, np.inf)
    return distance_transform_edt(~blocked)*resolution_m


def evaluate(output: Path, mission_folder: Path) -> dict:
    """telemetry.jsonl(GT 위치)와 summary.json으로 완주·벽 거리·정지 이유를 돌려준다."""
    from experiments.handheld_mapping.site_mission import SiteMission
    site = SiteMission.load(mission_folder, load_mesh=False).site
    distance = clearance_map(site.blocked, site.resolution)
    rows = [json.loads(line) for line in (Path(output)/'telemetry.jsonl').open() if line.strip()]
    cells = site.cells(np.array([r['position_m'][:2] for r in rows]))
    inside = (cells[:, 0] >= 0) & (cells[:, 0] < distance.shape[1]) & (cells[:, 1] >= 0) & (cells[:, 1] < distance.shape[0])
    values = np.zeros(len(rows))                                         # 지도 밖은 거리 0(충돌)으로 둔다
    values[inside] = distance[cells[inside, 1], cells[inside, 0]]
    summary = json.loads((Path(output)/'summary.json').read_text())
    state = summary.get('mission', {})
    return {'completed': summary.get('status') == 'complete' and state.get('state') == 'complete',
            'status': summary.get('status'), 'reason': summary.get('reason'), 'mission_state': state.get('state'),
            'route_status': state.get('route_status'), 'duration_sim_s': summary.get('duration_sim_s'),
            'stuck_events': summary.get('stuck_events'), 'samples': len(rows),
            'min_clearance_m': float(values.min()) if len(values) else None,
            'contact_samples': int((values < BODY_RADIUS_M).sum()), 'margin_violation_samples': int((values < PLANNING_RADIUS_M).sum())}


if __name__ == '__main__':
    print(json.dumps(evaluate(Path(sys.argv[1]), Path(sys.argv[2]))))
