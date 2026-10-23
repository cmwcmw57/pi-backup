# ================================================================
# imu_realism.py
# 설명: 시뮬의 이상적 IMU 구간 평균에 실기에서 잰 자이로 바이어스·정지 잡음(nominal: 맑은 정지 30 s, stress: 이전 모래 구간 값)·이동 중 진동(모래 15° 램프 student v3 12:04~12:08)을 더한다. 기본은 꺼짐이다.
# 사용법:
#   from experiments.localization.imu_realism import ImuRealism
#   OnlineEstimator(..., imu_realism=ImuRealism(seed=1))   # run.py는 --imu-realism
# ================================================================
from dataclasses import asdict, dataclass

import numpy as np

G = 9.81


@dataclass
class ImuRealism:
    """IMU 구간 (gyro, 비력)에 body 축 바이어스·백색 잡음·이동 진동을 더하고 IMU 축으로 되돌린다."""

    gyro_bias_dps: tuple = (-0.637, 0.485, 0.486)     # 맑은 정지 30 s(10-18 밤, 타일, 바퀴 정지 확인) 평균. 이전 로그 세 개와 소수점 둘째 자리까지 같음
    gyro_density: float = 7.2e-5                       # rad/s/√Hz, nominal = 같은 정지 30 s 실측 6.3~8.1e-5의 평균(σ = 밀도/√dt)
    acc_density: float = 6e-4                          # m/s²/√Hz, nominal = 같은 정지 30 s 실측 0.0005~0.0007의 평균
    gyro_vibration_dps: tuple = (1.44, 2.51, 2.87)     # 전진(오르기) 중 1 Hz 고역통과 표준편차, body 축
    acc_vibration_g: tuple = (0.034, 0.022, 0.026)
    motion_speed_m_s: float = 0.08                     # 이 속도 이상이면 이동 진동 100% (제 가정)
    motion_yaw_rate_rad_s: float = 0.3                 # 제자리 회전도 이동으로 본다 (제 가정)
    vibration_scale: float = 1.0                       # 진동 배율 (정지 잡음·바이어스에는 적용 안 함)
    seed: int = 0
    drop_probability: float = 0.0                      # IMU 패킷 한 개가 누락될 확률. 실측: 40 ms 이상 간격이 0.8%
    max_hold_s: float = 0.06                           # 누락으로 합쳐지는 구간의 상한(필터 공백 기준 0.1 s보다 짧게)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    @classmethod
    def stress(cls, **kwargs) -> "ImuRealism":
        """stress 프로필: 정지 백색 잡음을 모래 구간에서 뽑았던 이전 값(자이로 1.1e-4, 가속도 0.025)으로 둔다. nominal보다 가속도 약 40배 크다."""
        return cls(**{'gyro_density': 1.1e-4, 'acc_density': 0.025, **kwargs})

    def parameters(self) -> dict:
        """기록용 파라미터(난수 상태 제외)."""
        return asdict(self)

    def should_drop(self, held_s: float, dt: float) -> bool:
        """이번 패킷을 누락시킬지. 누락 구간이 max_hold_s를 넘기게 되면 누락시키지 않는다."""
        return self.drop_probability > 0 and held_s+dt <= self.max_hold_s and bool(self.rng.random() < self.drop_probability)

    def motion_factor(self, speed: float, yaw_rate: float) -> float:
        """0(정지)~1(이동 진동 최대)."""
        return float(min(1.0, max(speed/self.motion_speed_m_s, abs(yaw_rate)/self.motion_yaw_rate_rad_s)))

    def apply(self, dt: float, speed: float, base_from_imu: np.ndarray, gyro: np.ndarray, force: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray]:
        """IMU 축 (gyro rad/s, 비력 m/s²)에 잡음을 더해 반환한다. 측정값이 body 축이라 body에서 더한 뒤 되돌린다."""
        if not dt > 0:
            raise ValueError('dt must be positive')
        r = np.asarray(base_from_imu, float)[:3, :3]               # IMU 축 → body 축
        gyro_b, force_b = r @ np.asarray(gyro, float), r @ np.asarray(force, float)
        m = self.motion_factor(speed, float(np.linalg.norm(gyro_b)))
        white_g, white_a = self.gyro_density/np.sqrt(dt), self.acc_density/np.sqrt(dt)
        vib_g = np.radians(np.asarray(self.gyro_vibration_dps, float))*self.vibration_scale
        vib_a = np.asarray(self.acc_vibration_g, float)*G*self.vibration_scale
        sigma_g = (1-m)*white_g+m*np.maximum(white_g, vib_g)
        sigma_a = (1-m)*white_a+m*np.maximum(white_a, vib_a)
        gyro_b = gyro_b+np.radians(self.gyro_bias_dps)+self.rng.standard_normal(3)*sigma_g
        force_b = force_b+self.rng.standard_normal(3)*sigma_a
        return r.T @ gyro_b, r.T @ force_b
