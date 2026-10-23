# ================================================================
# sand_lane.py
# 설명: 좌 영상의 밝기 군집으로 모래 바닥을 가르고 바닥-벽 경계를 지면 평면(차체 z=-h)에 투영해 카메라 기준 좌/우/전방 여유 거리를 낸다(위치·지도 무관).
# 사용법: lane = SandLane(stereo_calibration_dict); obs = lane.observe(rgb_uint8, stamp_s)  # dict(stamp_s,left,right,front,n)
# ================================================================
from __future__ import annotations

import cv2
import numpy as np


def pixel_ground_mask(gray: np.ndarray) -> np.ndarray:
    """밝기 histogram을 세 군집으로 나누고 영상 아래 중앙(가까운 바닥)과 같은 군집을 모래로 고른다."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.)
    hist = np.bincount(blurred.ravel(), minlength=256).astype(float)
    values = np.arange(256.)
    centers = np.percentile(blurred, [5, 50, 95]).astype(float)
    for _ in range(12):
        labels = np.argmin(abs(values[:, None]-centers[None, :]), axis=1)
        updated = centers.copy()
        for k in range(3):
            weights = hist*(labels == k)
            if weights.sum():updated[k] = (values@weights)/weights.sum()
        if np.max(abs(updated-centers)) < .1:
            centers = updated;break
        centers = updated
    h, w = gray.shape
    seed = float(np.median(blurred[int(h*.9):, int(w*.35):int(w*.65)]))
    if seed < 25:return np.zeros_like(gray)
    selected = int(np.argmin(abs(centers-seed)))
    lookup = np.argmin(abs(values[:, None]-centers[None, :]), axis=1)
    return cv2.morphologyEx((lookup[blurred] == selected).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


class SandLane:
    """모래 바닥 마스크의 위쪽 경계를 지면에 투영해 좌/우/전방 여유를 계산한다."""

    def __init__(self, calibration: dict, base_height_m: float = .165) -> None:
        self.fx, self.fy, self.cx, self.cy = (float(calibration[k]) for k in ('fx', 'fy', 'cx', 'cy'))
        transform = np.asarray(calibration['T_base_left_optical'], float)
        self.rotation, self.translation, self.height = transform[:3, :3], transform[:3, 3], float(base_height_m)

    def points(self, rgb: np.ndarray) -> np.ndarray:
        """열별로 아래에서 위로 바닥이 끊기는 지점을 지면 평면에 투영한 차체 좌표(x 전방, y 좌)를 반환한다."""
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
        mask = pixel_ground_mask(gray)
        rows, cols = mask.shape
        out = []
        for u in range(8, cols-8, 8):
            column = mask[:, u]
            v = rows-1
            while v > 5 and (column[v] or column[max(0, v-3):v+1].any()):v -= 1
            if v <= 5 or v >= 440:continue
            ray = self.rotation@np.array([(u-self.cx)/self.fx, (v-self.cy)/self.fy, 1.])
            if ray[2] >= -1e-3:continue
            scale = (-self.height-self.translation[2])/ray[2]
            if scale <= 0:continue
            point = self.translation+scale*ray
            if .1 < point[0] < 2.5 and abs(point[1]) < 1.5:out.append(point[:2])
        return np.asarray(out).reshape(-1, 2)

    def observe(self, rgb: np.ndarray, stamp_s: float, x_lo: float = .30, x_hi: float = 1.0, half: float = .20) -> dict:
        """좌/우 측방 여유(전방 x_lo~x_hi 구간 경계의 20분위)와 전방 여유(중앙 |y|<half의 최소 x)를 반환한다. 안 보이면 None."""
        pts = self.points(np.asarray(rgb))
        result = dict(stamp_s=float(stamp_s), left=None, right=None, front=None, n=int(len(pts)))
        if not len(pts):return result
        x, y = pts[:, 0], pts[:, 1]
        side = (x >= x_lo) & (x <= x_hi)
        left, right, ahead = y[side & (y > .05)], -y[side & (y < -.05)], x[(np.abs(y) < half) & (x >= .35)]
        if len(left) >= 3:result['left'] = float(np.percentile(left, 20))
        if len(right) >= 3:result['right'] = float(np.percentile(right, 20))
        if len(ahead) >= 2:result['front'] = float(ahead.min())
        return result
