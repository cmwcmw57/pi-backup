# ================================================================
# start_noise.py
# 설명: 실기에서 줄자로 놓은 시작 자세의 오차를 흉내 낸다. 로버는 (dx, dy, dyaw)만큼 틀어 놓고 추정기는 이상적인 시작 자세를 믿는다.
# 사용법: dx, dy, dyaw = start_offset(seed, sigma_xy_m=0.02, sigma_yaw_deg=2.0)
# ================================================================
import math

import numpy as np


def start_offset(seed: int, sigma_xy_m: float, sigma_yaw_deg: float) -> tuple[float, float, float]:
    """(dx [m], dy [m], dyaw [rad]) 정규분포 표본. 시그마가 0이면 (0, 0, 0)이고 값은 ±3σ로 자른다."""
    if sigma_xy_m < 0 or sigma_yaw_deg < 0:
        raise ValueError('sigmas must be non-negative')
    rng = np.random.default_rng(seed)
    dx, dy = np.clip(rng.normal(0., sigma_xy_m, 2), -3*sigma_xy_m, 3*sigma_xy_m) if sigma_xy_m > 0 else (0., 0.)
    dyaw = float(np.clip(rng.normal(0., sigma_yaw_deg), -3*sigma_yaw_deg, 3*sigma_yaw_deg)) if sigma_yaw_deg > 0 else 0.
    return float(dx), float(dy), math.radians(dyaw)
