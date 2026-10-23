# ================================================================
# imu_from_poses.py
# 설명: 시뮬 GT pose 시계열을 평활 spline으로 미분해 이상적인 body-frame IMU(자이로·비력)를 만든다.
# 사용법: from experiments.site_sim.imu_from_poses import ideal_imu_from_poses
# ================================================================
import numpy as np
from scipy.interpolate import make_smoothing_spline
from scipy.spatial.transform import Rotation, RotationSpline

GRAVITY = np.array([0., 0., 9.81])


def ideal_imu_from_poses(times: np.ndarray, positions: np.ndarray, quaternions_wxyz: np.ndarray,
                         imu_times: np.ndarray, smoothing: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    """GT pose 표본에서 imu_times의 [gyro(body), specific force(body)] 6열 신호를 반환한다.

    위치는 평활 spline의 2계 도함수, 자세는 RotationSpline의 body 각속도를 쓴다. 접촉 진동은 평활로 줄어든다.
    """
    times, positions = np.asarray(times, float), np.asarray(positions, float)
    imu_times, quaternions = np.asarray(imu_times, float), np.asarray(quaternions_wxyz, float)
    if np.any(np.diff(times) <= 0) or positions.shape != (len(times), 3) or quaternions.shape != (len(times), 4):
        raise ValueError('times must increase and poses must be (n,3)/(n,4)')
    if imu_times.min() < times[0] or imu_times.max() > times[-1]:
        raise ValueError('IMU times must lie inside the pose time range')
    acceleration = np.column_stack([make_smoothing_spline(times, positions[:, axis], lam=smoothing).derivative(2)(imu_times)
                                    for axis in range(3)])
    rotation = Rotation.from_quat(np.column_stack([quaternions[:, 1:], quaternions[:, :1]]))
    spline = RotationSpline(times, rotation)
    body = spline(imu_times)
    gyro = spline(imu_times, 1)
    force = body.inv().apply(acceleration+GRAVITY)
    return imu_times, np.column_stack([gyro, force])
