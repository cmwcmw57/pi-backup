# ================================================================
# wall_track.py
# 설명: 주행 중 영상 밝기 경계의 좌·우 옆벽 수직거리로 좁은 복도의 가운데 오차를, 앞벽 수직거리로 진행방향 위치 오차를 구한다. 가운데 오차는 vy 보정 명령에, 진행방향 오차는 추정 위치의 작은 보정에 쓴다(위치 추정에는 횡방향을 반영하지 않는다).
# 사용법: wt = WallTrack(stereo_calibration_dict, site); m = wt.measure(rgb, est_xy, est_yaw, roll, pitch, yaw_rate)  # dict(lateral_err, front_delta, ...)
# ================================================================
from __future__ import annotations

import math

import numpy as np

import cv2

import ksrc_runtime.site_sim.sand_lane as _sand_lane
from ksrc_runtime.site_sim.sand_lane import SandLane


def sand_mask_seed(gray: np.ndarray, low: float = .5, high: float = 1.35) -> np.ndarray:
    """영상 하단 중앙 밝기 s를 시드로 low*s~high*s 밝기이면서 시드와 이어진 영역(구멍 메움)을 모래로 잡는다(시뮬·실기 영상 공통)."""
    b = cv2.GaussianBlur(gray, (0, 0), 3).astype(np.float32)
    h, w = gray.shape
    seed_value = float(np.median(b[int(h*.88):, int(w*.35):int(w*.65)]))
    if seed_value < 25:
        return np.zeros_like(gray, np.uint8)
    close = ((b >= low*seed_value) & (b <= high*seed_value)).astype(np.uint8)
    close = cv2.morphologyEx(close, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, _, _ = cv2.connectedComponentsWithStats(close)
    seed_label = lab[int(h*.94), w//2]
    if seed_label == 0:
        counts = np.bincount(lab[int(h*.88):, int(w*.35):int(w*.65)].ravel(), minlength=n)
        counts[0] = 0
        if counts.max() == 0:
            return np.zeros_like(gray, np.uint8)
        seed_label = int(counts.argmax())
    m = (lab == seed_label).astype(np.uint8)
    inv = np.pad((1-m).astype(np.uint8), 1, constant_values=1)
    cv2.floodFill(inv, np.zeros((inv.shape[0]+2, inv.shape[1]+2), np.uint8), (0, 0), 2)
    m[inv[1:-1, 1:-1] == 1] = 1
    return m


class _SeedLane(SandLane):
    """SandLane.points를 시드 밝기 대역 마스크로 계산한다(모듈 전역 마스크는 호출 동안만 바꾼다)."""

    def points(self, rgb: np.ndarray) -> np.ndarray:
        original = _sand_lane.pixel_ground_mask
        _sand_lane.pixel_ground_mask = sand_mask_seed
        try:
            return super().points(rgb)
        finally:
            _sand_lane.pixel_ground_mask = original


def wrap_half(angle: float) -> float:
    """직선 방향 차이를 (-pi/2, pi/2]로 접는다."""
    return (angle+math.pi/2) % math.pi-math.pi/2


class WallTrack:
    """복도축에 정렬된 평지 구간에서만 벽 수직거리로 가운데·진행방향 오차를 측정한다."""

    def __init__(self, calibration: dict, site, max_roll_deg: float = 4., max_pitch_deg: float = 8., max_rate_rad_s: float = .15, align_deg: float = 10.,
                 max_residual_m: float = .03, corridor_width_m: tuple = (.6, 1.0), front_range_m: tuple = (.35, .8)) -> None:
        self.lane, self.site = _SeedLane(calibration), site
        self.max_roll, self.max_pitch, self.max_rate, self.align = math.radians(max_roll_deg), math.radians(max_pitch_deg), max_rate_rad_s, math.radians(align_deg)
        self.max_residual, self.width, self.front_range = max_residual_m, corridor_width_m, front_range_m

    def _ray(self, origin: np.ndarray, direction: np.ndarray, max_m: float = 4.) -> float | None:
        """origin에서 direction으로 처음 만나는 blocked 셀까지의 거리(1 cm 간격)."""
        s = self.site
        for d in np.arange(0., max_m, .01):
            c = s.cells([origin+direction*d])[0]
            if not (0 <= c[0] < s.blocked.shape[1] and 0 <= c[1] < s.blocked.shape[0]):
                return None
            if s.blocked[c[1], c[0]]:
                return float(d)
        return None

    def planar_ahead(self, xy: np.ndarray, yaw: float, length_m: float, max_range_deg: float = 3., max_slope_deg: float = 8.) -> bool:
        """지도상 로버 앞 0~length_m·좌우 0.3 m의 경사가 한 평면처럼(경사 변화 3° 이하, 8° 이하) 이어지는지 확인한다. 로버가 경사 위에 있으면 지면 평면이 몸체 기준 같다."""
        c, s = math.cos(yaw), math.sin(yaw)
        pts = np.array([[xy[0]+a*c-l*s, xy[1]+a*s+l*c] for a in np.linspace(0., length_m, 8) for l in (-.3, 0., .3)])
        cells = self.site.cells(pts)
        rows, cols = self.site.slope_deg.shape
        inside = (cells[:, 0] >= 0) & (cells[:, 0] < cols) & (cells[:, 1] >= 0) & (cells[:, 1] < rows)
        if not inside.any():
            return False
        cc = cells[inside]
        free = ~self.site.blocked[cc[:, 1], cc[:, 0]]      # 벽 셀의 급경사 값은 제외한다
        if not free.any():
            return False
        slope = self.site.slope_deg[cc[free, 1], cc[free, 0]]
        return bool(np.nanmax(slope) <= max_slope_deg and np.nanmax(slope)-np.nanmin(slope) <= max_range_deg)

    @staticmethod
    def _fit(points: np.ndarray):
        """TLS 직선의 (원점에서 선까지 수직거리, 잔차, 길이, 방향각)."""
        centre = points.mean(axis=0)
        _, _, vt = np.linalg.svd(points-centre)
        direction = vt[0]
        normal = np.array([-direction[1], direction[0]])
        if normal @ centre < 0:
            normal = -normal
        return float(normal @ centre), float(np.median(np.abs((points-centre) @ vt[1]))), float(np.ptp(points @ direction)), math.atan2(direction[1], direction[0]) % math.pi

    def measure(self, rgb: np.ndarray, est_xy: np.ndarray, est_yaw: float, roll: float, pitch: float, yaw_rate: float) -> dict:
        """lateral_err: 왼쪽이면 +(m), lateral_delta: 추정→실제 왼쪽 이동량(m), front_delta: 진행축으로 진짜 위치가 추정보다 앞서면 +(m). 없는 항목은 None. reason에 건너뛴 사유를 남긴다."""
        out = dict(reason='', lateral_err=None, lateral_delta=None, width=None, front_delta=None, front_d=None, axis=None, side_n=0)
        if abs(roll) > self.max_roll or abs(pitch) > self.max_pitch:
            out['reason'] = 'tilted';return out
        if abs(yaw_rate) > self.max_rate:
            out['reason'] = 'turning';return out
        axis = round(est_yaw/(math.pi/2))*(math.pi/2)
        out['axis'] = axis
        if abs((est_yaw-axis+math.pi) % (2*math.pi)-math.pi) > self.align:
            out['reason'] = 'diagonal';return out
        lateral_ok, front_ok = self.planar_ahead(est_xy, est_yaw, 1.2), self.planar_ahead(est_xy, est_yaw, 1.0)
        if not (lateral_ok or front_ok):
            out['reason'] = 'not_planar';return out
        pts = self.lane.points(np.asarray(rgb))
        if len(pts) < 10:
            out['reason'] = 'no_floor';return out
        heading = np.array([math.cos(axis), math.sin(axis)])
        left = np.array([-heading[1], heading[0]])
        tl, tr = self._ray(est_xy, left), self._ray(est_xy, -left)
        if lateral_ok and tl is not None and tr is not None and self.width[0] <= tl+tr <= self.width[1]:
            width = tl+tr
            out['width'] = width
            errors = []
            for sign in (1., -1.):
                sel = pts[(pts[:, 1]*sign > .08) & (pts[:, 0] > .3) & (pts[:, 0] < 1.5)]
                if len(sel) < 8:
                    continue
                distance, residual, extent, angle = self._fit(sel if sign > 0 else sel*np.array([1., -1.]))
                if residual <= self.max_residual and extent >= .3 and abs(wrap_half(angle)) <= math.radians(15.):
                    errors.append(width/2-distance if sign > 0 else distance-width/2)   # 왼쪽 벽이 가까우면 왼쪽(+)
            if errors:
                out['lateral_err'] = float(np.mean(errors));out['side_n'] = len(errors)
                out['lateral_delta'] = float(out['lateral_err']-(tr-tl)/2)   # 추정 위치에서 실제 위치까지 왼쪽(+)으로 필요한 이동량
        front = pts[(np.abs(pts[:, 1]) < .7) & (pts[:, 0] > .3) & (pts[:, 0] < 2.2)]
        if front_ok and len(front) >= 10:
            distance, residual, extent, angle = self._fit(front)
            if (residual <= self.max_residual and extent >= .4 and abs(abs(wrap_half(angle))-math.pi/2) <= math.radians(15.)
                    and self.front_range[0] <= distance <= self.front_range[1]):
                hit = self._ray(est_xy, heading)
                if hit is not None:
                    out['front_delta'] = float(hit-distance);out['front_d'] = float(distance)
        if out['lateral_delta'] is None and out['front_delta'] is None:
            out['reason'] = 'no_wall'
        return out

    def front_only(self, rgb: np.ndarray, est_yaw: float, roll: float, pitch: float, yaw_rate: float) -> dict:
        """모서리 전용 앞벽 수직거리(m). 경사 게이트 없이 롤·피치·회전 속도·정렬(10°)과 선 품질만 보고, 0.30~0.80 m 범위만 채택한다."""
        out = dict(reason='', d=None)
        if abs(roll) > self.max_roll or abs(pitch) > self.max_pitch:
            out['reason'] = 'tilted';return out
        if abs(yaw_rate) > self.max_rate:
            out['reason'] = 'turning';return out
        axis = round(est_yaw/(math.pi/2))*(math.pi/2)
        if abs((est_yaw-axis+math.pi) % (2*math.pi)-math.pi) > self.align:
            out['reason'] = 'diagonal';return out
        pts = self.lane.points(np.asarray(rgb))
        front = pts[(np.abs(pts[:, 1]) < .7) & (pts[:, 0] > .28) & (pts[:, 0] < 1.0)] if len(pts) else pts
        if len(front) < 10:
            out['reason'] = 'no_front';return out
        distance, residual, extent, angle = self._fit(front)
        if residual > self.max_residual or extent < .4 or abs(abs(wrap_half(angle))-math.pi/2) > math.radians(15.):
            out['reason'] = 'poor_line';return out
        if not .30 <= distance <= .80:
            out['reason'] = 'out_of_range';return out
        out['d'] = float(distance)
        return out
