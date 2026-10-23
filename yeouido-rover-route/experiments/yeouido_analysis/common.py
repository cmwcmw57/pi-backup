# ================================================================
# common.py
# 설명: 체크포인트 CSV(줄자 실측) 읽기, 구간별 줄자 거리 계산, 텍스트 표 출력을 slip·orb_retention이 공유한다.
# 사용법:
#   from experiments.yeouido_analysis.common import load_checkpoints, tape_distances, format_table
# ================================================================
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STOP_SPEED_M_S = 0.01
SPEED_BIN_M_S = 0.05
SIM_CAVEAT = "주의: 시뮬 수치는 모래 텍스처 0.25 m 반복 인공물 때문에 실측으로 해석하지 않는다."


@dataclass(frozen=True)
class Checkpoint:
    """줄자로 잰 체크포인트 한 점과 그 점에 도착하는 구간의 선택 라벨을 보관한다."""

    checkpoint_id: str
    x_m: float
    y_m: float
    t_s: float | None
    ramp_deg: float | None
    speed_m_s: float | None


def _optional(row: dict[str, str], key: str) -> float | None:
    """CSV 셀이 비어 있거나 열이 없으면 None, 아니면 float를 반환한다."""
    text = (row.get(key) or "").strip()
    return float(text) if text else None


def load_checkpoints(path: Path) -> list[Checkpoint]:
    """checkpoint_id, measured_x_m, measured_y_m (+선택 t_s, ramp_deg, speed_m_s) CSV를 읽는다."""
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = {"checkpoint_id", "measured_x_m", "measured_y_m"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"checkpoint CSV missing columns: {sorted(missing)}")
        points = [
            Checkpoint(
                checkpoint_id=row["checkpoint_id"].strip(),
                x_m=float(row["measured_x_m"]),
                y_m=float(row["measured_y_m"]),
                t_s=_optional(row, "t_s"),
                ramp_deg=_optional(row, "ramp_deg"),
                speed_m_s=_optional(row, "speed_m_s"),
            )
            for row in reader
        ]
    if len(points) < 2:
        raise ValueError("at least two checkpoints are required")
    return points


def tape_distances(points: list[Checkpoint], measured_frame: str) -> list[float]:
    """연속 체크포인트 사이의 줄자 거리를 경사면 기준으로 반환한다(plan이면 1/cosθ 보정)."""
    if measured_frame not in ("surface", "plan"):
        raise ValueError("measured_frame must be 'surface' or 'plan'")
    distances = []
    for previous, current in zip(points[:-1], points[1:]):
        chord = math.hypot(current.x_m - previous.x_m, current.y_m - previous.y_m)
        if measured_frame == "plan" and current.ramp_deg:
            chord /= math.cos(math.radians(current.ramp_deg))
        distances.append(chord)
    return distances


def detect_stops(t_s: np.ndarray, speed_m_s: np.ndarray, min_stop_s: float) -> list[float]:
    """속도가 STOP_SPEED_M_S 미만으로 min_stop_s 이상 이어진 구간의 중앙 시각을 순서대로 반환한다."""
    stops, start = [], None
    for i, still in enumerate(np.append(speed_m_s < STOP_SPEED_M_S, False)):
        if still and start is None:
            start = i
        elif not still and start is not None:
            end = i - 1
            if t_s[end] - t_s[start] >= min_stop_s:
                stops.append(float(0.5 * (t_s[start] + t_s[end])))
            start = None
    return stops


def speed_bin(speed_m_s: float) -> float:
    """속도를 SPEED_BIN_M_S 단위로 반올림(반은 올림)한 그룹 라벨을 반환한다."""
    return math.floor(speed_m_s / SPEED_BIN_M_S + 0.5) * SPEED_BIN_M_S


def format_table(rows: list[dict], columns: list[str]) -> str:
    """dict 행을 열 순서대로 정렬한 텍스트 표로 만든다(float는 소수 3자리, None/NaN은 '-')."""
    def cell(value: object) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "-"
        return f"{value:.3f}" if isinstance(value, float) else str(value)

    body = [[cell(row.get(name)) for name in columns] for row in rows]
    widths = [max(len(name), *(len(line[i]) for line in body)) if body else len(name)
              for i, name in enumerate(columns)]
    lines = ["  ".join(name.rjust(w) for name, w in zip(columns, widths))]
    lines += ["  ".join(text.rjust(w) for text, w in zip(line, widths)) for line in body]
    return "\n".join(lines)
