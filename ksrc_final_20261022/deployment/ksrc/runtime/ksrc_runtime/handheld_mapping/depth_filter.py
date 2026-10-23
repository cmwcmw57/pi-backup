"""원본을 보존하는 depth 지지점 필터와 최종 자세 기반 다중 시점 확인."""
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class DepthFilter:
    min_depth_m: float = .25
    max_depth_m: float = 3.
    absolute_tolerance_m: float = .02
    quadratic_tolerance: float = .015
    min_neighbours: int = 3
    temporal_neighbours: int = 2
    max_reference_age_s: float = 3.

    def __post_init__(self):
        values = (self.min_depth_m, self.max_depth_m, self.absolute_tolerance_m,
                  self.quadratic_tolerance, self.max_reference_age_s)
        if (not all(np.isfinite(values)) or not 0 < self.min_depth_m < self.max_depth_m
                or self.absolute_tolerance_m <= 0 or self.quadratic_tolerance < 0
                or self.max_reference_age_s <= 0 or type(self.min_neighbours) is not int
                or not 1 <= self.min_neighbours <= 8 or type(self.temporal_neighbours) is not int
                or not 0 <= self.temporal_neighbours <= 4):
            raise ValueError('invalid depth filter settings')

    def metadata(self):
        return {'method': 'spatial_support_and_final_pose_multiview_v1', **asdict(self),
                'fills_missing_depth': False, 'changes_raw': False,
                'single_frame_fallback': 'spatial only when no nearby reference frame exists',
                'limitations': 'repeatable stereo errors can survive; pose error can reject real surfaces'}


def spatial_filter(depth: np.ndarray, scale: float, config: DepthFilter) -> tuple[np.ndarray, dict]:
    """빈 영역을 채우거나 깊이를 평균내지 않고 고립값과 큰 깊이 경계를 무효화한다."""
    if depth.ndim != 2 or depth.dtype != np.uint16 or not np.isfinite(scale) or not 0 < scale < .1:
        raise ValueError('invalid metric Z16 depth')
    z = depth.astype(np.float32)*scale
    valid = (z >= config.min_depth_m) & (z <= config.max_depth_m)
    tolerance = config.absolute_tolerance_m+config.quadratic_tolerance*z*z
    support = np.zeros(depth.shape, np.uint8)
    discontinuity = np.zeros(depth.shape, bool)
    padded = np.pad(z, 1, constant_values=np.nan)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == dy == 0:
                continue
            other = padded[1+dy:1+dy+depth.shape[0], 1+dx:1+dx+depth.shape[1]]
            available = (other >= config.min_depth_m) & (other <= config.max_depth_m)
            difference = np.abs(other-z)
            support += available & (difference <= tolerance)
            discontinuity |= available & (difference > tolerance*2)
    keep = valid & (support >= config.min_neighbours) & ~discontinuity
    result = np.where(keep, depth, 0).astype(np.uint16)
    return result, {'range_valid_pixels': int(valid.sum()), 'spatial_kept_pixels': int(keep.sum())}


def multiview_filter(depth: np.ndarray, info: dict, world_depth: np.ndarray,
                     references: list, config: DepthFilter) -> tuple[np.ndarray, dict]:
    """주변의 다른 프레임에서 실측 depth가 일치한 픽셀만 확인된 관측으로 남긴다."""
    if not references:
        return depth.copy(), {'reference_frames': 0, 'multiview_kept_pixels': int((depth > 0).sum())}
    scale = float(info['depth_scale_m'])
    k = np.asarray(info['k'], float).reshape(3, 3)
    v, u = np.nonzero(depth)
    z = depth[v, u].astype(np.float64)*scale
    points = np.c_[(u-k[0, 2])*z/k[0, 0], (v-k[1, 2])*z/k[1, 1], z]
    confirmed = np.zeros(len(z), bool)
    height, width = depth.shape
    for reference_depth, reference_pose in references:
        relative = np.linalg.inv(reference_pose) @ world_depth
        projected = points @ relative[:3, :3].T+relative[:3, 3]
        pz = projected[:, 2]
        visible = pz > 0
        divisor = np.where(visible, pz, 1.)
        x = np.rint(projected[:, 0]*k[0, 0]/divisor+k[0, 2]).astype(np.int64)
        y = np.rint(projected[:, 1]*k[1, 1]/divisor+k[1, 2]).astype(np.int64)
        visible &= (x >= 0) & (x < width) & (y >= 0) & (y < height)
        indices = np.flatnonzero(visible)
        measured = reference_depth[y[indices], x[indices]].astype(np.float64)*scale
        tolerance = config.absolute_tolerance_m+config.quadratic_tolerance*pz[indices]**2
        confirmed[indices] |= (measured > 0) & (np.abs(measured-pz[indices]) <= tolerance)
    result = np.zeros_like(depth)
    result[v[confirmed], u[confirmed]] = depth[v[confirmed], u[confirmed]]
    return result, {'reference_frames': len(references), 'multiview_kept_pixels': int(confirmed.sum())}


def filtered_depths(run: Path, keyfile: Path, map_id: int, info: dict,
                    world_from_slam: np.ndarray, config: DepthFilter, stats: dict):
    """최대 9장의 depth만 유지하며 TSDF와 지형에 같은 필터를 적용한다."""
    from ksrc_runtime.handheld_mapping.recording import matched_depths
    from ksrc_runtime.handheld_mapping.reconstruct import pose_matrix, transform
    left_depth = transform(info['T_left_depth'])
    height, width = int(info['height']), int(info['width'])
    scale = float(info['depth_scale_m'])
    stats.update({'keyframes': 0, 'matched_frames': 0, 'range_valid_pixels': 0,
                  'spatial_kept_pixels': 0, 'multiview_kept_pixels': 0,
                  'frames_without_reference': 0})

    def load():
        for row, frame, gap in matched_depths(run, keyfile, map_id, info['depth_frame']):
            stats['keyframes'] += 1
            if frame is None:
                yield row, frame, gap, None, None
                continue
            path = (run/frame['path']).resolve()
            if not path.is_relative_to((run/'raw').resolve()):
                raise ValueError('invalid depth path')
            depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if depth is None or depth.dtype != np.uint16 or depth.shape != (height, width):
                raise ValueError('depth file/profile mismatch')
            filtered, counts = spatial_filter(depth, scale, config)
            stats['matched_frames'] += 1
            for key, value in counts.items():
                stats[key] += value
            yield row, frame, gap, filtered, world_from_slam @ pose_matrix(row) @ left_depth

    # Unmatched keyframes do not consume reference slots.
    source = iter(load())
    future, previous = deque(), deque(maxlen=config.temporal_neighbours)
    exhausted = False
    while True:
        while not exhausted and sum(entry[3] is not None for entry in future) < config.temporal_neighbours+1:
            entry = next(source, None)
            if entry is None:
                exhausted = True
            else:
                future.append(entry)
        if not future:
            break
        row, frame, gap, depth, pose = future.popleft()
        if depth is None:
            yield row, frame, gap, None
            continue
        references = [(entry[3], entry[4]) for entry in [*previous, *future]
                      if entry[3] is not None and
                      abs(entry[1]['stamp_ns']-frame['stamp_ns']) <= config.max_reference_age_s*1e9]
        filtered, counts = multiview_filter(depth, info, pose, references, config)
        stats['multiview_kept_pixels'] += counts['multiview_kept_pixels']
        stats['frames_without_reference'] += not references
        yield row, frame, gap, filtered
        if config.temporal_neighbours:
            previous.append((row, frame, gap, depth, pose))
