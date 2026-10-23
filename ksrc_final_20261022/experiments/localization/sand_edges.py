# ================================================================
# sand_edges.py
# 설명: IR 영상의 모래 면 윤곽 직선에서 롤·피치를 분리한 수평 방향 ψ와 지면 수직거리(좌·우 가장자리, 폭, 중심 횡위치)를 계산한다.
#       10-18 여의도 오프라인 검증용 실험 코드이며 주행 경로에는 연결하지 않았다(검증 범위는 outputs/pi_1018/rlc_feasibility 보고 참고).
# 사용법:
#   from experiments.localization.sand_edges import edge_signal
#   sig = edge_signal(gray_uint8, up_cam, h=0.16)   # up_cam: 카메라 광학 좌표의 위쪽 단위벡터
# ================================================================
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

FX, CX, CY = 377.186, 320.54, 241.73
K = np.array([[FX, 0, CX], [0, FX, CY], [0, 0, 1.]])


@dataclass
class EdgeSignal:
    """좌·우 가장자리의 지면 수직거리(m, 오른쪽 +), 방향 ψ(도, 오른쪽 +), 폭, 길 중심 대비 카메라 횡위치(오른쪽 +)."""
    off_right: float
    off_left: float
    psi_right: float
    psi_left: float
    axis_deg: float
    width: float
    center_m: float


def wrap180(x: float | np.ndarray) -> float | np.ndarray:
    """각도 차를 [-90,90)로 접는다(직선 방향은 180° 주기)."""
    return (np.asarray(x)+90.) % 180.-90.


def sand_mask(gray: np.ndarray) -> np.ndarray:
    """블러 + Otsu 밝기 분할 후 최대 연결성분의 구멍을 메운 모래 마스크(uint8 0/1)."""
    b = cv2.GaussianBlur(gray, (0, 0), 3)
    th, _ = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    m = cv2.morphologyEx((b > th).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return m*0
    m = (lab == 1+np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    # 모서리 픽셀이 모래여도 flood-fill 시작점이 항상 외부 배경에 있도록 패딩한다.
    inv = np.pad((1-m).astype(np.uint8), 1, constant_values=1)
    cv2.floodFill(inv, np.zeros((inv.shape[0]+2, inv.shape[1]+2), np.uint8), (0, 0), 2)
    m[inv[1:-1, 1:-1] == 1] = 1
    return m


def contour_lines(mask: np.ndarray, n_lines: int = 3, tol: float = 2., min_pts: int = 70, min_span: float = 100.) -> list[tuple[np.ndarray, np.ndarray]]:
    """마스크 윤곽(영상 테두리 4 px 안 제외)에서 RANSAC으로 직선 끝점 (p0,p1)을 최대 n_lines개 뽑는다."""
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return []
    p = max(cs, key=len)[:, 0, :].astype(float)
    h, w = mask.shape
    p = p[(p[:, 0] > 4) & (p[:, 0] < w-5) & (p[:, 1] > 4) & (p[:, 1] < h-5)]
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n_lines):
        if len(p) < min_pts:
            break
        best = None
        for _ in range(250):
            i, j = rng.choice(len(p), 2, replace=False)
            d = p[j]-p[i]
            if np.hypot(*d) < 20:
                continue
            r = np.abs((p-p[i])@(np.array([-d[1], d[0]])/np.hypot(*d)))
            if best is None or (r < tol).sum() > best.sum():
                best = r < tol
        if best is None or best.sum() < min_pts:
            break
        q = p[best]
        c0 = q.mean(0)
        d = np.linalg.svd(q-c0, full_matrices=False)[2][0]
        t = (q-c0)@d
        if t.max()-t.min() >= min_span:
            out.append((c0+d*t.min(), c0+d*t.max()))
        p = p[~best]
    return out


def line_heading_deg(p0: np.ndarray, p1: np.ndarray, up_cam: np.ndarray) -> float:
    """이미지 직선이 가리키는 수평 지면 직선의 방향을 카메라 앞 수평축 기준 [0,180)°로 준다(오른쪽 +)."""
    n_l = K.T@np.cross([*p0, 1.], [*p1, 1.])
    d = np.cross(n_l, up_cam)
    d /= np.linalg.norm(d)
    f = np.array([0, 0, 1.])-up_cam*up_cam[2]
    f /= np.linalg.norm(f)
    return float(np.degrees(np.arctan2(d@np.cross(f, up_cam), d@f)) % 180.)


def heading_sensitivity(p0: np.ndarray, p1: np.ndarray, up_cam: np.ndarray) -> float:
    """직선을 중점 둘레로 영상에서 1° 돌렸을 때 ψ가 변하는 크기(°/°). 클수록 지평선에 가까워 방향이 불안정하다."""
    mid, d = (p0+p1)/2, (p1-p0)/2
    c, s = np.cos(np.radians(1.)), np.sin(np.radians(1.))
    d2 = np.array([c*d[0]-s*d[1], s*d[0]+c*d[1]])
    return abs(float(wrap180(line_heading_deg(mid-d2, mid+d2, up_cam)-line_heading_deg(p0, p1, up_cam))))


def line_ground_offset(p0: np.ndarray, p1: np.ndarray, up_cam: np.ndarray, h: float) -> float:
    """지면 위 수평선이라 가정할 때 카메라 발밑에서 그 선까지의 수직 거리(m, 오른쪽 +). h는 카메라 지면 높이."""
    n = K.T@np.cross([*p0, 1.], [*p1, 1.])
    n /= np.linalg.norm(n)
    if n@up_cam < 0:
        n = -n
    d = np.cross(n, up_cam)
    d /= np.linalg.norm(d)
    a, b = np.linalg.solve(np.array([[1., n@up_cam], [n@up_cam, 1.]]), [0., -h])
    v = a*n+b*up_cam+h*up_cam
    f = np.array([0, 0, 1.])-up_cam*up_cam[2]
    f /= np.linalg.norm(f)
    perp = v-(v@d)*d
    return float(np.sign(perp@np.cross(f, up_cam))*np.linalg.norm(perp))


def edge_signal(gray: np.ndarray, up_cam: np.ndarray, h: float, max_off: float = 1., max_sens: float = 3., max_dpsi: float = 30., max_axis: float = 45.) -> EdgeSignal | None:
    """영상에서 좌우 모래 가장자리를 찾아 EdgeSignal을 준다. 조건(거리<=max_off, 민감도<=max_sens, 두 방향 차<=max_dpsi, |축 방향|<=max_axis)을 못 채우면 None.
    검증 범위: ψ(axis_deg)만 846cdc 고정 직선 트랙으로 부분 검증, off/center/width는 미검증(h는 depth 평면 추정값 의존). 끝벽 프레임은 max_axis 로 거른다."""
    cand = []
    for p0, p1 in contour_lines(sand_mask(gray)):
        if heading_sensitivity(p0, p1, up_cam) > max_sens:
            continue
        o = line_ground_offset(p0, p1, up_cam, h)
        if 0.05 < abs(o) <= max_off:
            cand.append((o, line_heading_deg(p0, p1, up_cam)))
    pairs = [(abs(float(wrap180(r[1]-l[1]))), r, l) for r in cand if r[0] > 0 for l in cand if l[0] < 0]
    pairs = [x for x in pairs if x[0] <= max_dpsi]
    if not pairs:
        return None
    _, r, l = min(pairs, key=lambda x: x[0])
    axis = float(wrap180(r[1]+float(wrap180(l[1]-r[1]))/2))
    if abs(axis) > max_axis:
        return None
    return EdgeSignal(r[0], l[0], r[1], l[1], axis, r[0]-l[0], -(r[0]+l[0])/2)
