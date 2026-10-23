# ================================================================
# sand_boundary_candidates.py
# 설명: IR 밝기 기반 모래 후보 마스크의 연속된 바깥 윤곽을 직선 구간으로 분할한다.
#       오프라인 검토용이며 지형 종류·STL 대응·측정 신뢰도를 확정하지 않는다.
# 사용법: from experiments.localization.sand_boundary_candidates import extract_boundaries
# ================================================================
from __future__ import annotations

import cv2
import numpy as np

from experiments.localization.sand_edges import sand_mask


def extract_boundaries(gray: np.ndarray, min_span_px: float = 60.) -> tuple[np.ndarray, list[dict]]:
    """마스크와 순서가 보존된 윤곽선 후보를 반환한다. coverage는 확률이 아닌 픽셀 지지율이다."""
    if gray.dtype != np.uint8 or gray.ndim != 2 or min(gray.shape) < 12:
        raise ValueError('expected nonempty uint8 grayscale image of at least 12x12 pixels')
    if not np.isfinite(min_span_px) or min_span_px <= 0:
        raise ValueError('min_span_px must be finite and positive')
    mask = sand_mask(gray)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return mask, []
    contour = max(contours, key=len)
    vertices = cv2.approxPolyDP(contour, 3., True)[:, 0, :]
    boundary = np.zeros_like(gray)
    cv2.drawContours(boundary, [contour], -1, 255, 1)
    distance = cv2.distanceTransform(255-boundary, cv2.DIST_L2, 5)
    edges = []
    for index, (a, b) in enumerate(zip(vertices, np.roll(vertices, -1, axis=0))):
        visible, a, b = cv2.clipLine((5, 5, gray.shape[1]-10, gray.shape[0]-10),
                                    tuple(map(int, a)), tuple(map(int, b)))
        if not visible:
            continue
        a, b = np.array(a, float), np.array(b, float)
        span = float(np.linalg.norm(b-a))
        if span < min_span_px:
            continue
        points = np.rint(np.linspace(a, b, int(np.ceil(span))+1)).astype(int)
        supported = distance[points[:, 1], points[:, 0]] <= 3.
        longest = current = 0
        for good in supported:
            current = 0 if good else current+1
            longest = max(longest, current)
        coverage = float(supported.mean())
        if coverage >= .75 and longest <= 20:
            edges.append(dict(contour_segment=index, p0=a, p1=b, span_px=span,
                              coverage=coverage, max_gap_px=longest))
    return mask, edges
