# ================================================================
# region_observation.py
# 설명: 시뮬 stereo 깊이와 추정 pose로 알려진 지면의 일관성·가시 구역·장애물 후보를 반환한다.
#       pose 보정, 지도 free-space 갱신, 주행 중단 결정은 수행하지 않는다.
# 사용법: from experiments.handheld_mapping.region_observation import RegionObservation
# ================================================================
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import maximum_filter

from experiments.handheld_mapping.site_map import SiteMap
from experiments.localization.wall_range import Camera, stereo_depth
from experiments.site_sim.wall_replay import sgbm


class RegionObservation:
    """ESKF pose를 그대로 사용해 현재 stereo의 비학습 지면 관측을 요약한다."""

    def __init__(self, site: SiteMap, calibration_dict: dict,
                 regions: list[dict] | None = None) -> None:
        self.site = site
        # 추정 pose 오차(.3 m 이내)로 경사 지면이 낮은 격자에 놓여도 장애물로 보지 않도록 주변 최고 지면을 쓴다.
        radius = int(round(.3/site.resolution))
        self.local_ground_max = maximum_filter(np.where(np.isfinite(site.height_m), site.height_m, -np.inf),
                                               size=2*radius+1)
        self.camera = Camera.from_sim(calibration_dict)
        self.matcher = sgbm()
        self.set_regions([] if regions is None else regions)

    def set_regions(self, regions: list[dict]) -> None:
        """전방에서 관측한 지면을 표시할 명시적인 지도 polygon을 설정한다."""
        self.regions = {int(r['id']): np.asarray(r['polygon_xy_m'], np.float32)
                        for r in regions}

    def update(self, left_rgb: np.ndarray | None, right_rgb: np.ndarray | None,
               world_from_base: np.ndarray, stamp_s: float) -> dict:
        """JSON-safe 관측 품질과 최대 200개 장애물 XY 후보를 반환한다."""
        stamp = float(stamp_s)
        result = dict(stamp_s=stamp if np.isfinite(stamp) else None, valid=False,
                      reason='missing_stereo', visible_regions=[], obstacles_xy=[],
                      ground_points=0, consistent_fraction=0.)
        if left_rgb is None or right_rgb is None:
            return result
        pose = np.asarray(world_from_base, float)
        if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.isfinite(stamp):
            result['reason'] = 'invalid_pose_or_stamp'
            return result
        left, right = np.asarray(left_rgb), np.asarray(right_rgb)
        if left.shape != right.shape or left.ndim not in (2, 3):
            result['reason'] = 'invalid_stereo_shape'
            return result
        try:
            if left.ndim == 3:
                left = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
                right = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
            depth = stereo_depth(self.matcher, left, right, self.camera)
        except cv2.error:
            result['reason'] = 'stereo_unavailable'
            return result
        # SGBM의 대비/uniqueness 조건을 통과한 깊이만 사용한다. 결측은 free가 아니다.
        v, u = np.mgrid[0:depth.shape[0]:3, 0:depth.shape[1]:3]
        z = depth[v, u].ravel()
        keep = np.isfinite(z) & (z >= .25) & (z <= 3.)
        if not keep.any():
            result['reason'] = 'no_depth_support'
            return result
        c = self.camera
        rays = np.column_stack(((u.ravel()[keep]-c.cx)/c.fx,
                                (v.ravel()[keep]-c.cy)/c.fy, np.ones(keep.sum())))
        transform = pose @ c.base_from_camera
        world_rays = rays @ transform[:3, :3].T
        points = world_rays*z[keep, None]+transform[:3, 3]
        cells = self.site.cells(points[:, :2])
        rows, cols = self.site.height_m.shape
        inside = ((cells >= 0).all(axis=1) & (cells[:, 0] < cols) & (cells[:, 1] < rows))
        points, cells, world_rays = points[inside], cells[inside], world_rays[inside]
        known = (self.site.observed[cells[:, 1], cells[:, 0]] &
                 ~self.site.blocked[cells[:, 1], cells[:, 0]])
        points, cells, world_rays = points[known], cells[known], world_rays[known]
        residual = points[:, 2]-self.site.height_m[cells[:, 1], cells[:, 0]]
        # 벽·장애물의 높이를 바닥 잔차 분모에 넣지 않는다. 경사 격자 오차에 여유를 둔다.
        floor = (world_rays[:, 2] < -.03) & (np.abs(residual) <= .22)
        consistent = floor & (np.abs(residual) <= .12)
        count = int(floor.sum())
        result['ground_points'] = int(consistent.sum())
        result['consistent_fraction'] = float(consistent.sum()/count) if count else 0.
        if result['ground_points'] < 12:
            result['reason'] = 'insufficient_ground_support'
            return result
        if result['consistent_fraction'] < .35:
            result['reason'] = 'ground_inconsistent'
            return result
        result.update(valid=True, reason='ground_supported')
        ground_xy = points[consistent, :2]
        for region_id, polygon in sorted(self.regions.items()):
            if sum(cv2.pointPolygonTest(polygon, tuple(map(float, xy)), False) >= 0
                   for xy in ground_xy) >= 3:
                result['visible_regions'].append(region_id)
        # 같은 지도 cell에 여러 픽셀이 비슷한 높이를 지지해야 장애물로 제안한다.
        relative = points[:, 2]-self.local_ground_max[cells[:, 1], cells[:, 0]]
        raised = np.flatnonzero((relative > .16) & (residual < 1.2))
        if len(raised):
            unique, inverse, counts = np.unique(cells[raised], axis=0, return_inverse=True,
                                                return_counts=True)
            proposals = []
            for group in np.flatnonzero(counts >= 3):
                indices = raised[inverse == group]
                if np.ptp(residual[indices]) <= .12:
                    proposals.append(np.median(points[indices, :2], axis=0))
            proposals.sort(key=lambda xy: float(np.linalg.norm(xy-pose[:2, 3])))
            result['obstacles_xy'] = [xy.tolist() for xy in proposals[:200]]
        return result
