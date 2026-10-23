# ================================================================
# pixel_region_observation.py
# 설명: 단일 영상의 명암 마스크와 알려진 지형의 예상 투영을 비교해 가시 구역을 확인한다.
#       관측 구역은 추정 pose·지도에 조건부이며 독립 위치 추정이나 미지 장애물 거리 측정이 아니다.
# 사용법: observer = PixelRegionObservation(site, calibration, regions); observer.update(left, None, pose, stamp)
# ================================================================
from __future__ import annotations

import cv2
import numpy as np
import time

from experiments.localization.wall_range import Camera


def pixel_ground_mask(gray: np.ndarray) -> np.ndarray:
    """밝기 histogram을 세 군집으로 나누고 가까운 하단 중앙 면과 같은 군집을 고른다."""
    blurred=cv2.GaussianBlur(gray,(5,5),1.)
    histogram=np.bincount(blurred.ravel(),minlength=256).astype(float)
    values=np.arange(256.)
    centers=np.percentile(blurred,[5,50,95]).astype(float)
    for _ in range(12):
        labels=np.argmin(abs(values[:,None]-centers[None,:]),axis=1)
        updated=centers.copy()
        for k in range(3):
            weights=histogram*(labels==k)
            if weights.sum():updated[k]=(values@weights)/weights.sum()
        if np.max(abs(updated-centers))<.1:
            centers=updated;break
        centers=updated
    h,w=gray.shape
    seed=float(np.median(blurred[int(h*.9):,int(w*.35):int(w*.65)]))
    if seed<25:return np.zeros_like(gray)
    selected=int(np.argmin(abs(centers-seed)))
    lookup=np.argmin(abs(values[:,None]-centers[None,:]),axis=1)
    mask=(lookup[blurred]==selected).astype(np.uint8)
    return cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))


class PixelRegionObservation:
    """지도 기반 예상 지면 투영에 실제 모래 픽셀 지지가 있는 구역만 반환한다."""

    def __init__(self, site, calibration: dict, regions: list[dict]) -> None:
        self.site, self.camera = site, Camera.from_sim(calibration)
        self.regions = {int(r['id']): np.asarray(r['polygon_xy_m'], np.float32) for r in regions}
        self.labels = np.zeros(site.height_m.shape, np.int16)
        # 구역 polygon은 경계를 공유한다. 내부 가시 구역의 영상 확인에만 사용한다.
        for region, polygon in self.regions.items():
            cells = np.round((polygon-site.origin)/site.resolution-.5).astype(np.int32)
            cv2.fillPoly(self.labels, [cells], int(region))

    def predicted_labels(self, shape: tuple[int, int], pose: np.ndarray, stride: int = 8) -> tuple:
        """지도 높이에 대한 광선 첫 교차다. 센서 depth를 계산하거나 읽지 않는다."""
        h, w = shape
        v, u = np.mgrid[stride//2:h:stride, stride//2:w:stride]
        c = self.camera
        rays = np.column_stack(((u.ravel()-c.cx)/c.fx, (v.ravel()-c.cy)/c.fy, np.ones(u.size)))
        camera = pose@c.base_from_camera
        rays = rays@camera[:3, :3].T
        labels, active = np.zeros(u.size, np.int16), np.ones(u.size, bool)
        rows, cols = self.labels.shape
        # 광선별 XY 진행이 반 셀 이하여야 얇은 벽을 뛰어넘지 않는다.
        step = min(.05, self.site.resolution*.5/max(1., np.linalg.norm(rays[:, :2], axis=1).max()))
        for z in np.arange(.25, 3.001, step):
            ids = np.flatnonzero(active)
            if not len(ids):
                break
            points = camera[:3, 3]+rays[ids]*z
            cells = self.site.cells(points[:, :2])
            inside = ((cells >= 0).all(axis=1) & (cells[:, 0] < cols) & (cells[:, 1] < rows))
            ids, points, cells = ids[inside], points[inside], cells[inside]
            if not len(ids):
                continue
            x, y = cells.T
            hit = self.site.observed[y, x] & (points[:, 2] <= self.site.height_m[y, x])
            hit_ids, x, y = ids[hit], x[hit], y[hit]
            labels[hit_ids] = np.where(self.site.blocked[y, x], 0, self.labels[y, x])
            active[hit_ids] = False
        return u, v, labels.reshape(u.shape)

    def update(self, left: np.ndarray | None, right: np.ndarray | None,
               world_from_base: np.ndarray, stamp_s: float) -> dict:
        """실측 픽셀 지지와 지도 예상 구역을 구분한다. 오른쪽 영상은 사용하지 않는다."""
        result = dict(stamp_s=float(stamp_s), valid=False, reason='missing_pixels', source='pixel_map_conditioned',
                      visible_regions=[], predicted_regions=[], obstacles_xy=[], ground_points=0,
                      consistent_fraction=0., region_support={}, boundary_points=0)
        if left is None:
            return result
        pose = np.asarray(world_from_base, float)
        image = np.asarray(left)
        if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.isfinite(stamp_s):
            result.update(reason='invalid_pose_or_stamp', stamp_s=None)
            return result
        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if image.ndim != 2 or image.dtype != np.uint8 or min(image.shape) < 12:
            result['reason'] = 'invalid_pixels'
            return result
        if int(image.max())-int(image.min()) < 10:
            result['reason'] = 'no_pixel_contrast'
            return result
        begin=time.perf_counter()
        mask = pixel_ground_mask(image)
        boundaries = cv2.Canny(mask*255, 50, 100)
        boundaries[:5] = 0; boundaries[-5:] = 0; boundaries[:, :5] = 0; boundaries[:, -5:] = 0
        result['boundary_points'] = int(np.count_nonzero(boundaries))
        if result['boundary_points'] < 12:
            result['reason'] = 'no_pixel_boundary'
            return result
        mask_ms=(time.perf_counter()-begin)*1000
        begin=time.perf_counter()
        u, v, labels = self.predicted_labels(image.shape, pose)
        result['timings_ms']=dict(pixel_mask=mask_ms,map_projection=(time.perf_counter()-begin)*1000)
        pixel_ground = (mask[v, u] > 0) & (image[v, u] > 25)
        known = labels > 0
        result['predicted_regions'] = sorted(int(k) for k in np.unique(labels[known]))
        result['ground_points'] = int(np.count_nonzero(known & pixel_ground))
        result['consistent_fraction'] = float(np.count_nonzero(known & pixel_ground)/max(1, known.sum()))
        for region in result['predicted_regions']:
            expected = labels == region
            supported = int(np.count_nonzero(expected & pixel_ground))
            fraction = float(supported/max(1, expected.sum()))
            result['region_support'][str(region)] = dict(pixels=supported, fraction=fraction)
            if supported >= 12 and fraction >= .35:
                result['visible_regions'].append(region)
        result['valid'] = bool(result['visible_regions'])
        result['reason'] = 'pixel_map_supported' if result['valid'] else 'no_pixel_map_support'
        return result
