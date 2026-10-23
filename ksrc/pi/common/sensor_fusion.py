"""Student 관측을 SPI 텔레메트리에서 만드는 계산 (Pi, numpy).

시뮬 학습과 **같은 식**이어야 한다: sim/isaac/ksrc_rover/source/ksrc_rover/utils/sensors_torch.py
(torch 판).  둘이 같은지는 sim/rl/test_actuators.py 가 같은 입력열로 비교한다.

  gravity   GravityFilter: 자이로 + 가속도 보완 필터 → 몸체 좌표 중력 방향 (정자세 (0,0,-1))
  wheel     텔레메트리 wheel_mps (펌웨어가 엔코더 20 ms 차분으로 계산) / 바퀴 반경 → rad/s
  steering  텔레메트리 steer_meas_rad (서보 위치 레지스터, 4096 격자) 그대로

IMU 단위: nucleo_link.decode_spi_telemetry 의 gyro_dps / acc_g (로버 좌표계) →
  gyro rad/s = dps · π/180 − 부팅 정지 바이어스,  acc m/s² = g · 9.81
"""

import math

import numpy as np

GRAVITY = 9.81
GRAV_TAU_S = 0.5
GRAV_GATE = 1.5
WHEEL_RADIUS_M = 0.047503


class GravityFilter:
    """sensors_torch.GravityFilter 의 바퀴 하나짜리(N=1) numpy 판. 식 설명은 그쪽."""

    def __init__(self, dt=0.02, tau=GRAV_TAU_S, gate=GRAV_GATE):
        self.dt, self.alpha, self.gate = dt, dt / tau, gate
        self.g = np.array([0.0, 0.0, -1.0], np.float32)
        self.need_init = True

    def reset(self):
        self.need_init = True
        self.g = np.array([0.0, 0.0, -1.0], np.float32)

    def step(self, gyro, acc):
        """리셋 뒤에는 가속도 크기가 9.81±gate 안인 첫 샘플로 초기화 (그 전엔 (0,0,-1) 유지)."""
        gyro = np.asarray(gyro, np.float32)
        acc = np.asarray(acc, np.float32)
        fn = max(float(np.linalg.norm(acc)), 1e-6)
        a_dir = -acc / np.float32(fn)
        ok = abs(fn - GRAVITY) < self.gate
        if self.need_init:
            if ok:
                self.g = a_dir.astype(np.float32)
                self.need_init = False
            return self.g
        pred = self.g - np.float32(self.dt) * np.cross(gyro, self.g)
        a = np.float32(self.alpha) if ok else np.float32(0.0)
        g = (np.float32(1.0) - a) * pred + a * a_dir
        self.g = (g / np.float32(max(float(np.linalg.norm(g)), 1e-6))).astype(np.float32)
        return self.g


class GyroBias:
    """부팅 직후 정지 상태 평균으로 자이로 바이어스를 잡는다 (n 샘플)."""

    def __init__(self, n=100):
        self.n, self.buf, self.bias = n, [], None

    def update(self, gyro_rad):
        if self.bias is None:
            self.buf.append(np.asarray(gyro_rad, float))
            if len(self.buf) >= self.n:
                self.bias = np.mean(self.buf, axis=0)
        return self.bias


def imu_si(tlm, gyro_bias=None):
    """decode_spi_telemetry 결과 → (gyro rad/s, acc m/s²), 로버 좌표계."""
    gyro = np.asarray(tlm["gyro_dps"], float) * math.pi / 180.0
    if gyro_bias is not None:
        gyro = gyro - gyro_bias
    return gyro, np.asarray(tlm["acc_g"], float) * GRAVITY


def wheel_rad_s(tlm):
    """펌웨어가 엔코더 틱 20 ms 차분으로 낸 바퀴 선속도 → 각속도 (모듈 순서 FL, FR, RL, RR)."""
    return np.asarray(tlm["wheel_mps"], float) / WHEEL_RADIUS_M
