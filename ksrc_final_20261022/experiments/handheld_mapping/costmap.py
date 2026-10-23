# ================================================================
# costmap.py
# 설명: 지형 정보 층과 로버 허용 조건으로 unknown·장애물·위험 비용을 계산한다.
# 사용법: from experiments.handheld_mapping.costmap import RoverLimits, build_costmap
# ================================================================
from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True)
class RoverLimits:
    """실기에서 조정할 지형 한계와 원형 장애물 여유를 정의한다."""

    max_slope_deg: float = 20.
    max_step_m: float = .06
    max_roughness_m: float = .025
    safety_radius_m: float = .15

    def __post_init__(self) -> None:
        values = (self.max_slope_deg, self.max_step_m, self.max_roughness_m, self.safety_radius_m)
        if (not all(math.isfinite(v) for v in values) or not 0 < self.max_slope_deg < 90
                or self.max_step_m <= 0 or self.max_roughness_m <= 0 or self.safety_radius_m < 0):
            raise ValueError('invalid rover limits')


def build_costmap(layers: dict[str, np.ndarray], resolution: float, limits: RoverLimits) -> np.ndarray:
    """unknown=255, blocked=254, 유효 위험 비용=0..252인 배열을 반환한다."""
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError('invalid costmap resolution')
    shape = layers['height_m'].shape
    if len(shape) != 2 or any(value.shape != shape for value in layers.values()):
        raise ValueError('inconsistent terrain layer dimensions')
    valid = layers['valid'] & layers['observed']
    metrics = (('slope_deg', limits.max_slope_deg, .4),
               ('roughness_m', limits.max_roughness_m, .3), ('step_m', limits.max_step_m, .3))
    risk = np.zeros(shape, np.float32)
    # Ground-mode span excludes upper objects. Obstacles have their own evidence layer.
    obstacle = layers.get('obstacle_observed', np.zeros(shape, bool))
    blocked = layers['observed'] & (layers['vertical_span_m'] >= limits.max_step_m)
    if 'obstacle_height_m' in layers:
        blocked |= obstacle & (layers['obstacle_height_m'] >= limits.max_step_m)
    for name, maximum, weight in metrics:
        value = layers[name]
        valid &= np.isfinite(value) & (value >= 0)
        blocked |= layers['observed'] & (value >= maximum)
        risk += weight*np.nan_to_num(np.clip(value/maximum, 0., 1.), nan=0.)
    if blocked.any() and limits.safety_radius_m > 0:
        blocked = distance_transform_edt(~blocked, sampling=resolution) <= limits.safety_radius_m
    cost = np.full(shape, 255, np.uint8)
    cost[valid] = np.rint(np.clip(risk[valid], 0., 1.)*252).astype(np.uint8)
    cost[blocked & layers.get('surface_observed', layers['observed'])] = 254
    return cost
