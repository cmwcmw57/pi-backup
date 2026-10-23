# ================================================================
# terrain.py
# 설명: 주기별 3D 관측을 고정 크기 격자 통계로 누적하고 지형 정보 층을 계산한다.
# 사용법: from ksrc_runtime.handheld_mapping.terrain import GridSpec, TerrainGrid
# ================================================================
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    """중력 정렬된 XY 범위와 격자/갱신 크기 상한을 정의한다."""

    bounds: tuple[float, float, float, float]
    resolution: float
    max_cells: int = 250_000
    max_points: int = 100_000

    @property
    def shape(self) -> tuple[int, int]:
        if (len(self.bounds) != 4 or not all(math.isfinite(v) for v in (*self.bounds, self.resolution))
                or self.resolution <= 0 or not isinstance(self.max_cells, int) or self.max_cells <= 0
                or not isinstance(self.max_points, int) or self.max_points <= 0):
            raise ValueError('invalid grid limits')
        xmin, xmax, ymin, ymax = self.bounds
        if xmax <= xmin or ymax <= ymin:
            raise ValueError('invalid grid bounds')
        nx, ny = math.ceil((xmax-xmin)/self.resolution), math.ceil((ymax-ymin)/self.resolution)
        if nx*ny > self.max_cells:
            raise ValueError('grid exceeds max_cells before allocation')
        return ny, nx


def shifted(array: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """지도 밖에서 NaN을 반환하는 이웃 셀 배열을 만든다."""
    result = np.roll(array, (-dy, -dx), axis=(0, 1))
    if dy > 0:
        result[-dy:] = np.nan
    elif dy < 0:
        result[:-dy] = np.nan
    if dx > 0:
        result[:, -dx:] = np.nan
    elif dx < 0:
        result[:, :-dx] = np.nan
    return result


class TerrainGrid:
    """프레임을 보관하지 않고 셀별 높이 통계와 관측 이력만 유지한다."""

    def __init__(self, spec: GridSpec) -> None:
        self.spec = spec
        shape = spec.shape
        self.height = np.full(shape, np.nan, np.float32)
        self.m2 = np.zeros(shape, np.float32)
        self.low = np.full(shape, np.inf, np.float32)
        self.high = np.full(shape, -np.inf, np.float32)
        self.count = np.zeros(shape, np.uint32)
        self.last = np.zeros(shape, np.uint64)
        self.stamp_ns = -1
        self.stats = {'updates': 0, 'outside_points': 0, 'invalid_points': 0, 'accepted_points': 0}

    @property
    def persistent_bytes(self) -> int:
        return sum(value.nbytes for value in (self.height, self.m2, self.low, self.high, self.count, self.last))

    def update(self, points: np.ndarray, stamp_ns: int) -> dict:
        if not isinstance(stamp_ns, int) or not self.stamp_ns < stamp_ns < 2**64:
            raise ValueError('non-increasing/invalid stamp_ns')
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) > self.spec.max_points:
            raise ValueError('invalid/excessive points per update')
        finite = np.isfinite(points).all(axis=1)
        self.stats['invalid_points'] += int((~finite).sum())
        points = points[finite]
        xmin, xmax, ymin, ymax = self.spec.bounds
        inside = ((points[:, 0] >= xmin) & (points[:, 0] < xmax)
                  & (points[:, 1] >= ymin) & (points[:, 1] < ymax))
        self.stats['outside_points'] += int((~inside).sum())
        points = points[inside]
        self.stats['accepted_points'] += len(points)
        self.stats['updates'] += 1
        self.stamp_ns = stamp_ns
        if not len(points):
            return dict(self.stats)
        xy = np.floor((points[:, :2]-[xmin, ymin])/self.spec.resolution).astype(np.int64)
        indices = xy[:, 1]*self.height.shape[1]+xy[:, 0]
        unique, inverse, hits = np.unique(indices, return_inverse=True, return_counts=True)
        values = (np.bincount(inverse, weights=points[:, 2])/hits).astype(np.float32)
        count = self.count.ravel()[unique].astype(np.float64)
        old = np.where(count > 0, self.height.ravel()[unique], 0.)
        updated = old+(values-old)/(count+1)
        self.m2.ravel()[unique] += (values-old)*(values-updated)
        self.height.ravel()[unique] = updated
        self.count.ravel()[unique] = np.minimum(count+1, np.iinfo(np.uint32).max).astype(np.uint32)
        self.last.ravel()[unique] = stamp_ns
        np.minimum.at(self.low.ravel(), indices, points[:, 2])
        np.maximum.at(self.high.ravel(), indices, points[:, 2])
        return dict(self.stats)

    def layers(self, min_observations: int = 2) -> dict[str, np.ndarray]:
        if not isinstance(min_observations, int) or min_observations < 1:
            raise ValueError('invalid minimum observations')
        observed = self.count >= min_observations
        height = np.where(observed, self.height, np.nan)
        left, right = shifted(height, 0, -1), shifted(height, 0, 1)
        down, up = shifted(height, -1, 0), shifted(height, 1, 0)
        gx, gy = (right-left)/(2*self.spec.resolution), (up-down)/(2*self.spec.resolution)
        valid = observed & np.isfinite(gx) & np.isfinite(gy)
        error2 = np.zeros(height.shape, np.float32)
        neighbours = np.zeros(height.shape, np.uint8)
        step = np.zeros(height.shape, np.float32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if not dx and not dy:
                    continue
                other = shifted(height, dy, dx)
                available = valid & np.isfinite(other)
                residual = other-height-(gx*dx+gy*dy)*self.spec.resolution
                error2 += np.where(available, residual*residual, 0.)
                neighbours += available
                step = np.fmax(step, np.where(available, np.abs(other-height), 0.))
        valid &= neighbours >= 6
        normal_scale = np.sqrt(1+gx*gx+gy*gy)
        variance = np.full(height.shape, np.nan, np.float32)
        np.divide(self.m2, np.maximum(self.count.astype(np.float64)-1, 1), out=variance, where=self.count > 1)
        return {'height_m': self.height.copy(), 'observations': self.count.copy(),
                'last_stamp_ns': self.last.copy(), 'observed': observed,
                'variance_m2': variance,
                'vertical_span_m': np.where(self.count > 0, self.high-self.low, np.nan),
                'slope_deg': np.where(valid, np.degrees(np.arctan(np.hypot(gx, gy))), np.nan),
                'heading_deg': np.where(valid, np.degrees(np.arctan2(gy, gx)), np.nan),
                'normal_x': np.where(valid, -gx/normal_scale, np.nan),
                'normal_y': np.where(valid, -gy/normal_scale, np.nan),
                'normal_z': np.where(valid, 1/normal_scale, np.nan),
                'roughness_m': np.where(valid, np.sqrt(error2/np.maximum(neighbours, 1)), np.nan),
                'step_m': np.where(valid, step, np.nan), 'valid': valid}
