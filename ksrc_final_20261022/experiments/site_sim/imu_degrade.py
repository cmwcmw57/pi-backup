# ================================================================
# imu_degrade.py
# 설명: 시뮬 이상적 200Hz IMU를 실기 특성(20ms 평균, 실측 잡음, 시각 지연, 끊김)으로 바꾼 EuRoC IMU 표를 만든다.
# 사용법: from experiments.site_sim.imu_degrade import degrade
# ================================================================
import numpy as np

GYRO_DENSITY, ACC_DENSITY = 8.2e-5, 1.0e-3   # rad/s/rtHz, m/s^2/rtHz (2026-10-16 정지 구간 실측)
WINDOW_S = .02                                  # MCU 평균 구간(제어 주기)


def degrade(times_s: np.ndarray, signals: np.ndarray, *, average: bool=True, noise: bool=True,
            delay_s: float=0., gaps_s: tuple=(), seed: int=0) -> tuple[np.ndarray, np.ndarray]:
    """(시각 s, 6열 신호)를 반환한다. 평균은 5ms 간격 4표본 창이며 시각은 창 중심이다.

    delay_s는 보정하지 않은 도장 지연(창 중심 대비 늦은 양)이다. 보정했다면 0으로 둔다.
    """
    times_s, signals = np.asarray(times_s, float), np.asarray(signals, float)
    if signals.ndim != 2 or signals.shape != (len(times_s), 6) or np.any(np.diff(times_s) <= 0):
        raise ValueError('times must increase and signals must have shape (n, 6)')
    rng = np.random.default_rng(seed)
    if average:
        step = max(1, int(round(WINDOW_S/np.median(np.diff(times_s)))))
        starts = np.arange(0, len(times_s)-step+1, step)
        stamp = np.array([times_s[s:s+step].mean() for s in starts])
        values = np.array([signals[s:s+step].mean(axis=0) for s in starts])
        window = WINDOW_S
    else:
        stamp, values, window = times_s.copy(), signals.copy(), float(np.median(np.diff(times_s)))
    if noise:
        sigma = np.r_[np.full(3, GYRO_DENSITY), np.full(3, ACC_DENSITY)]/np.sqrt(window)
        values = values+rng.normal(0., 1., values.shape)*sigma
    stamp = stamp+delay_s
    keep = np.ones(len(stamp), bool)
    for low, high in gaps_s:
        keep &= ~((stamp >= low) & (stamp < high))
    return stamp[keep], values[keep]
