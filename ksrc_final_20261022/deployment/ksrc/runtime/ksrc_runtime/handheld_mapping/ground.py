"""중력 좌표에서 반복된 아래쪽 표면과 지면 위 장애물을 따로 누적한다."""
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

from ksrc_runtime.handheld_mapping.terrain import TerrainGrid


@dataclass(frozen=True)
class GroundConfig:
    surface_band_m: float = .04
    max_connect_slope_deg: float = 55.
    connect_step_m: float = .035
    seed_radius_m: float = .8
    seed_below_camera_m: float = .05
    seed_max_drop_m: float = 3.
    obstacle_lookup_radius_m: float = .25
    minimum_mode_fraction: float = .2

    def __post_init__(self):
        values = tuple(asdict(self).values())
        if (not all(np.isfinite(values)) or any(v <= 0 for v in values)
                or not self.seed_below_camera_m < self.seed_max_drop_m
                or not 0 < self.max_connect_slope_deg < 80 or self.minimum_mode_fraction > 1):
            raise ValueError('invalid ground separation settings')


class GroundGrid(TerrainGrid):
    """셀당 최대 세 높이군을 보존하며 카메라 주변 지면에서 연결된 표면만 지면으로 쓴다.

    전역 평면을 빼지 않는다. 최저 반복 높이군이 지면 후보이며 단독 물체 윗면은
    지면 종자와 연결되지 않으면 unknown이다. seed/연결 기준은 실기 검증 전이다.
    """

    def __init__(self, spec, config: GroundConfig | None = None):
        super().__init__(spec)
        self.config = config or GroundConfig()
        shape = self.height.shape
        self.mode_height = np.zeros((3, *shape), np.float32)
        self.mode_count = np.zeros((3, *shape), np.uint32)
        self.mode_m2 = np.zeros((3, *shape), np.float32)
        self.mode_low = np.full((3, *shape), np.inf, np.float32)
        self.mode_high = np.full((3, *shape), -np.inf, np.float32)
        self.mode_stamp = np.zeros((3, *shape), np.uint64)
        self.seed_count = np.zeros(shape, np.uint32)
        self.seed_height = np.zeros(shape, np.float32)
        self.upper_height = np.zeros(shape, np.float32)
        self.surface_count = np.zeros(shape, np.uint32)
        self.stats.update({'surface_mode_overflow': 0, 'seed_frames': 0})

    @property
    def persistent_bytes(self):
        return super().persistent_bytes+sum(v.nbytes for v in
            (self.mode_height, self.mode_count, self.mode_m2, self.mode_low, self.mode_high,
             self.mode_stamp, self.seed_count, self.seed_height, self.upper_height, self.surface_count))

    def update(self, points, stamp_ns, camera_position=None):
        if not isinstance(stamp_ns, int) or not self.stamp_ns < stamp_ns < 2**64:
            raise ValueError('non-increasing/invalid stamp_ns')
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) > self.spec.max_points:
            raise ValueError('invalid/excessive points per update')
        if camera_position is not None:
            camera_position = np.asarray(camera_position, float)
            if camera_position.shape != (3,) or not np.isfinite(camera_position).all():
                raise ValueError('invalid camera position for ground seed')
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
        keys = xy[:, 1]*self.height.shape[1]+xy[:, 0]
        order = np.lexsort((points[:, 2], keys))
        keys, z = keys[order], points[order, 2]
        unique, starts, hits = np.unique(keys, return_index=True, return_counts=True)
        inverse = np.repeat(np.arange(len(unique)), hits)
        # A lower band, not the average of floor + wall/table points in a cell.
        lower = z[starts+np.floor((hits-1)*.1).astype(int)]
        selected = np.abs(z-lower[inverse]) <= self.config.surface_band_m/2
        values = np.bincount(inverse[selected], weights=z[selected], minlength=len(unique))
        values /= np.bincount(inverse[selected], minlength=len(unique))
        upper = z[starts+hits-1]
        counts = self.surface_count.ravel()[unique].astype(np.float64)
        self.upper_height.ravel()[unique] += (upper-self.upper_height.ravel()[unique])/(counts+1)
        self.surface_count.ravel()[unique] = np.minimum(counts+1, np.iinfo(np.uint32).max)
        mode_count = self.mode_count.reshape(3, -1)[:, unique]
        delta = np.abs(self.mode_height.reshape(3, -1)[:, unique]-values)
        eligible = (mode_count > 0) & (delta <= self.config.surface_band_m)
        match = np.argmin(np.where(eligible, delta, np.inf), axis=0)
        empty = mode_count == 0
        match = np.where(eligible.any(axis=0), match, np.argmax(empty, axis=0))
        keep = eligible.any(axis=0) | empty.any(axis=0)
        self.stats['surface_mode_overflow'] += int((~keep).sum())
        indices, slots, values = unique[keep], match[keep], values[keep]
        c = self.mode_count.reshape(3, -1)[slots, indices].astype(np.float64)
        old = self.mode_height.reshape(3, -1)[slots, indices]
        updated = old+(values-old)/(c+1)
        self.mode_m2.reshape(3, -1)[slots, indices] += (values-old)*(values-updated)
        self.mode_height.reshape(3, -1)[slots, indices] = updated
        self.mode_count.reshape(3, -1)[slots, indices] = np.minimum(c+1, np.iinfo(np.uint32).max)
        self.mode_low.reshape(3, -1)[slots, indices] = np.minimum(
            self.mode_low.reshape(3, -1)[slots, indices], values)
        self.mode_high.reshape(3, -1)[slots, indices] = np.maximum(
            self.mode_high.reshape(3, -1)[slots, indices], values)
        self.mode_stamp.reshape(3, -1)[slots, indices] = stamp_ns
        if camera_position is not None:
            self._seed(unique, lower, camera_position)
        return dict(self.stats)

    def _seed(self, indices, values, camera):
        """주변 아래쪽 국소 면의 기울기·잔차를 검사해 프레임당 종자 한 개를 남긴다."""
        ny, nx = self.height.shape
        xy = np.c_[indices % nx, indices // nx]
        metric_xy = (xy+.5)*self.spec.resolution+[self.spec.bounds[0], self.spec.bounds[2]]
        distance = np.linalg.norm(metric_xy-camera[:2], axis=1)
        eligible = ((distance <= self.config.seed_radius_m)
                    & (values < camera[2]-self.config.seed_below_camera_m)
                    & (values > camera[2]-self.config.seed_max_drop_m))
        for centre in np.argsort(np.where(eligible, distance, np.inf))[:30]:
            if not eligible[centre]:
                break
            nearby = np.max(np.abs(xy-xy[centre]), axis=1) <= 2
            patch = metric_xy[nearby]-metric_xy[centre]
            if len(patch) < 6 or np.linalg.matrix_rank(np.c_[patch, np.ones(len(patch))]) < 3:
                continue
            design = np.c_[patch, np.ones(len(patch))]
            fit, *_ = np.linalg.lstsq(design, values[nearby], rcond=None)
            if (np.degrees(np.arctan(np.hypot(*fit[:2]))) > self.config.max_connect_slope_deg
                    or np.percentile(np.abs(design@fit-values[nearby]), 90) > .02):
                continue
            index = indices[centre]
            count = int(self.seed_count.ravel()[index])
            self.seed_height.ravel()[index] += (values[centre]-self.seed_height.ravel()[index])/(count+1)
            self.seed_count.ravel()[index] = min(count+1, np.iinfo(np.uint32).max)
            self.stats['seed_frames'] += 1
            break

    def layers(self, min_observations=2):
        if not isinstance(min_observations, int) or min_observations < 1:
            raise ValueError('invalid minimum observations')
        eligible = ((self.mode_count >= min_observations)
                    & (self.mode_count >= self.surface_count*self.config.minimum_mode_fraction))
        slot = np.argmin(np.where(eligible, self.mode_height, np.inf), axis=0)
        index = slot[None, ...]
        candidate = eligible.any(axis=0)
        for target, source in ((self.height, self.mode_height), (self.count, self.mode_count),
                               (self.m2, self.mode_m2), (self.low, self.mode_low),
                               (self.high, self.mode_high), (self.last, self.mode_stamp)):
            target[:] = np.take_along_axis(source, index, axis=0)[0]
        self.height[~candidate] = np.nan
        local = super().layers(min_observations)
        # A known vertical/very steep local surface is not a ground connector.
        candidate &= (~np.isfinite(local['slope_deg'])
                      | (local['slope_deg'] <= self.config.max_connect_slope_deg))
        self.height[~candidate] = np.nan
        seeds = candidate & (self.seed_count > 0) & (np.abs(self.height-self.seed_height) <= self.config.surface_band_m)
        ground = seeds.copy()
        queue = deque(zip(*np.nonzero(seeds)))
        ny, nx = ground.shape
        tangent = np.tan(np.deg2rad(self.config.max_connect_slope_deg))
        neighbours = [(dy, dx, self.config.connect_step_m+tangent*self.spec.resolution*np.hypot(dx, dy))
                      for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dx or dy]
        while queue:
            y, x = queue.popleft()
            for dy, dx, tolerance in neighbours:
                yy, xx = y+dy, x+dx
                if (0 <= yy < ny and 0 <= xx < nx and candidate[yy, xx] and not ground[yy, xx]
                        and abs(self.height[yy, xx]-self.height[y, x]) <= tolerance):
                    ground[yy, xx] = True
                    queue.append((yy, xx))
        self.height[~ground] = np.nan
        self.count[~ground] = 0
        layers = super().layers(min_observations)
        surface_observed = self.surface_count >= min_observations
        obstacle_height = np.full(ground.shape, np.nan, np.float32)
        obstacle_observed = np.zeros(ground.shape, bool)
        reference = np.full(ground.shape, np.nan, np.float32)
        gradient_x = np.full(ground.shape, np.nan, np.float32)
        gradient_y = np.full(ground.shape, np.nan, np.float32)
        if ground.any():
            distance, nearest = distance_transform_edt(~ground, sampling=self.spec.resolution, return_indices=True)
            yy, xx = nearest
            y, x = np.indices(ground.shape)
            # Extrapolate only a short distance using the local ground normal, never a global floor plane.
            gx = np.nan_to_num(-layers['normal_x'][yy, xx]/layers['normal_z'][yy, xx])
            gy = np.nan_to_num(-layers['normal_y'][yy, xx]/layers['normal_z'][yy, xx])
            expected = self.height[yy, xx]+self.spec.resolution*(gx*(x-xx)+gy*(y-yy))
            obstacle_observed = surface_observed & (distance <= self.config.obstacle_lookup_radius_m)
            obstacle_height[obstacle_observed] = np.maximum(0., self.upper_height[obstacle_observed]-expected[obstacle_observed])
            reference[obstacle_observed] = expected[obstacle_observed]
            gradient_x[obstacle_observed] = gx[obstacle_observed]
            gradient_y[obstacle_observed] = gy[obstacle_observed]
        layers.update({'ground': ground, 'ground_candidate': candidate, 'surface_observed': surface_observed,
                       'surface_observations': self.surface_count.copy(),
                       'obstacle_height_m': obstacle_height, 'obstacle_observed': obstacle_observed,
                       'reference_ground_height_m': reference,
                       'reference_ground_gradient_x': gradient_x, 'reference_ground_gradient_y': gradient_y,
                       'unclassified_surface': surface_observed & ~ground & ~obstacle_observed})
        return layers

    def metadata(self):
        return {'method': 'repeated_lower_surface_modes_and_seeded_local_connectivity_v1',
                **asdict(self.config), 'max_modes_per_cell': 3, 'global_plane_removed': False,
                'seed_source': 'local low surface below and near camera; heuristic, not wheel contact measurement',
                'limitations': 'occluded ground and isolated surfaces remain unknown; seed and connection thresholds need real validation'}
