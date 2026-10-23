# ================================================================
# wall_yaw.py
# 설명: 좌 영상의 밝기 경계(모래-벽)에서 앞 벽선을 맞춰 지도 복도축(0/90/180/-90°) 기준 절대 yaw를 측정한다. 램프·회전 중에는 건너뛰고 연속 두 측정이 일치할 때만 채택한다.
# 사용법: wy = WallYaw(stereo_calibration_dict); m = wy.measure(rgb, stamp_s, yaw_est, roll, pitch, yaw_rate)  # dict(valid, reason, yaw, ...)
# ================================================================
from __future__ import annotations

import math

import numpy as np

from ksrc_runtime.site_sim.sand_lane import SandLane


def wrap_half(angle: float) -> float:
    """직선 방향 차이를 (-pi/2, pi/2]로 접는다."""
    return (angle+math.pi/2) % math.pi-math.pi/2


class WallYaw:
    """앞 벽선 방향으로 base yaw(site 기준)를 측정한다."""

    def __init__(self, calibration: dict, site=None, sigma_deg: float = 5., max_tilt_deg: float = 4., max_rate_rad_s: float = .15,
                 max_residual_m: float = .03, min_extent_m: float = .4, agree_deg: float = 3., min_interval_s: float = .5) -> None:
        self.lane = SandLane(calibration)
        self.site, self.max_slope_deg = site, 4.
        self.sigma = math.radians(sigma_deg)
        self.max_tilt, self.max_rate = math.radians(max_tilt_deg), max_rate_rad_s
        self.max_roll, self.max_pitch = math.radians(4.), math.radians(8.)   # 경사 위 주행을 허용하려고 피치만 넓힌다
        self.max_residual, self.min_extent, self.agree = max_residual_m, min_extent_m, math.radians(agree_deg)
        self.min_interval = min_interval_s
        self.candidate = None            # (stamp_s, yaw) 직전 후보
        self.last_applied = -math.inf

    def flat_ahead(self, xy: np.ndarray, yaw: float) -> bool:
        """지도상 로버 앞 0~2.2 m·좌우 0.3 m가 평지(경사 4° 이하)인지 확인한다. 경사 지면은 지면 평면 투영이 틀려 yaw가 치우친다."""
        along = np.linspace(0., 2.2, 12)
        c, s = math.cos(yaw), math.sin(yaw)
        pts = np.array([[xy[0]+a*c-l*s, xy[1]+a*s+l*c] for a in along for l in (-.3, 0., .3)])
        cells = self.site.cells(pts)
        rows, cols = self.site.slope_deg.shape
        inside = (cells[:, 0] >= 0) & (cells[:, 0] < cols) & (cells[:, 1] >= 0) & (cells[:, 1] < rows)
        if not inside.any():
            return False
        slope = self.site.slope_deg[cells[inside, 1], cells[inside, 0]]
        return bool(np.nanmax(slope) <= self.max_slope_deg)

    def planar_ahead(self, xy: np.ndarray, yaw: float, length_m: float = 1.2, max_range_deg: float = 3., max_slope_deg: float = 8.) -> bool:
        """지도상 로버 앞 0~length_m·좌우 0.3 m의 경사가 한 평면처럼(변화 3° 이하, 8° 이하) 이어지는지 확인한다. 벽 셀은 제외한다."""
        c, s = math.cos(yaw), math.sin(yaw)
        pts = np.array([[xy[0]+a*c-l*s, xy[1]+a*s+l*c] for a in np.linspace(0., length_m, 8) for l in (-.3, 0., .3)])
        cells = self.site.cells(pts)
        rows, cols = self.site.slope_deg.shape
        inside = (cells[:, 0] >= 0) & (cells[:, 0] < cols) & (cells[:, 1] >= 0) & (cells[:, 1] < rows)
        if not inside.any():
            return False
        cc = cells[inside]
        free = ~self.site.blocked[cc[:, 1], cc[:, 0]]
        if not free.any():
            return False
        slope = self.site.slope_deg[cc[free, 1], cc[free, 0]]
        return bool(np.nanmax(slope) <= max_slope_deg and np.nanmax(slope)-np.nanmin(slope) <= max_range_deg)

    def measure(self, rgb: np.ndarray, stamp_s: float, yaw_est: float, roll: float, pitch: float, yaw_rate: float, xy_est: np.ndarray | None = None) -> dict:
        """채택 가능한 yaw 측정이면 valid=True와 yaw를 돌려준다. 사유는 reason에 남긴다."""
        out = dict(valid=False, reason='', yaw=None, n=0, residual_m=None, extent_m=None)
        if abs(roll) > self.max_roll or abs(pitch) > self.max_pitch:
            out['reason'] = 'tilted';return out
        if abs(yaw_rate) > self.max_rate:
            out['reason'] = 'turning';return out
        if self.site is not None and xy_est is not None and not self.planar_ahead(np.asarray(xy_est, float), yaw_est):
            out['reason'] = 'slope_ahead';return out
        pts = self.lane.points(np.asarray(rgb))
        front = pts[(np.abs(pts[:, 1]) < .7) & (pts[:, 0] > .35) & (pts[:, 0] < (1.5 if abs(pitch) > math.radians(2.) else 2.2))] if len(pts) else pts
        out['n'] = int(len(front))
        if len(front) < 10:
            out['reason'] = 'no_front';return out
        centre = front.mean(axis=0)
        _, _, vt = np.linalg.svd(front-centre)
        direction = vt[0]
        residual = float(np.median(np.abs((front-centre) @ vt[1])))
        extent = float(np.ptp(front @ direction))
        out.update(residual_m=residual, extent_m=extent)
        angle = math.atan2(direction[1], direction[0]) % math.pi
        if residual > self.max_residual or extent < self.min_extent:
            out['reason'] = 'poor_line';return out
        if abs(abs(wrap_half(angle))-math.pi/2) > math.radians(35.):
            out['reason'] = 'not_front_wall';return out
        axis = round(yaw_est/(math.pi/2))*(math.pi/2)
        yaw = axis-wrap_half(angle-math.pi/2)
        yaw = yaw_est+(yaw-yaw_est+math.pi) % (2*math.pi)-math.pi
        out['yaw'] = float(yaw)
        if abs(yaw-yaw_est) > math.radians(12.):
            out['reason'] = 'large_innovation';self.candidate = None;return out
        previous, self.candidate = self.candidate, (stamp_s, yaw)
        if previous is None or stamp_s-previous[0] > .6 or abs(yaw-previous[1]) > self.agree:
            out['reason'] = 'await_confirmation';return out
        if stamp_s-self.last_applied < self.min_interval:
            out['reason'] = 'rate_limited';return out
        self.last_applied = stamp_s
        out.update(valid=True, reason='ok')
        return out
