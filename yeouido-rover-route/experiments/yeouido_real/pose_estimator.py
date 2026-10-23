# ================================================================
# pose_estimator.py
# 설명: Nucleo 텔레메트리(wheel_mps, steer_meas_rad, gyro_dps, acc_g)로 (x, y, yaw)를 dead-reckoning하고 pitch/roll을 acc 중력 방향으로 구한다. 체크포인트에서 reanchor()로 줄자 실측 pose를 덮어쓴다.
# 사용법:
#   from experiments.yeouido_real.pose_estimator import PoseEstimator
#   est = PoseEstimator(x=0.29, y=0.32, yaw=0.0)
#   if est.update(tlm) and est.ready: x, y, yaw = est.pose   # tlm = decode_spi_telemetry() dict
# ================================================================
import math
from pathlib import Path

from experiments.orbslam3_ros2.swerve import SwerveGeometry

GEOMETRY_PATH = Path(__file__).resolve().parents[2] / "ros2_ws/src/rover_description/config/swerve_geometry.json"
MAX_DT_S = 0.1


class PoseEstimator:
    """바퀴 선속도·조향각으로 몸체 평면 속도를, gyro z로 yaw를 적분하고 정지 평균 gyro bias를 뺀다."""

    def __init__(self, geometry: SwerveGeometry | None = None, x: float = 0.0, y: float = 0.0,
                 yaw: float = 0.0, bias_samples: int = 100, acc_tau_s: float = 0.5) -> None:
        self.modules = (geometry or SwerveGeometry.from_json(GEOMETRY_PATH)).modules  # FL, FR, RL, RR
        self.x, self.y, self.yaw = x, y, yaw
        self.pitch = self.roll = 0.0
        self.bias_n, self.acc_tau = bias_samples, acc_tau_s
        self._bias_buf: list[float] = []
        self._bias = 0.0
        self._acc: tuple[float, float, float] | None = None
        self._tick: int | None = None
        self._t_ms: int | None = None

    @property
    def ready(self) -> bool:
        """정지 상태 gyro bias 측정이 끝났는지 반환한다 (그 전에는 pose를 적분하지 않는다)."""
        return len(self._bias_buf) >= self.bias_n

    @property
    def pose(self) -> tuple[float, float, float]:
        """(x [m], y [m], yaw [rad])를 반환한다."""
        return self.x, self.y, self.yaw

    def reanchor(self, x: float, y: float, yaw: float) -> None:
        """줄자로 잰 pose로 덮어쓴다."""
        self.x, self.y, self.yaw = x, y, math.atan2(math.sin(yaw), math.cos(yaw))

    def update(self, tlm: dict) -> bool:
        """텔레메트리 한 프레임을 반영한다. 같은 tick 중복이면 False."""
        tick, t_ms = tlm["tick"], tlm["t_ms"]
        if tick == self._tick:
            return False
        dt = 0.0 if self._t_ms is None else min(max((t_ms - self._t_ms) / 1000.0, 0.0), MAX_DT_S)
        self._tick, self._t_ms = tick, t_ms
        self._update_tilt(tlm["acc_g"], dt)
        gz = math.radians(tlm["gyro_dps"][2])
        if not self.ready:
            self._bias_buf.append(gz)
            if self.ready:
                self._bias = sum(self._bias_buf) / len(self._bias_buf)
            return True
        w = gz - self._bias
        vx = vy = 0.0
        for m, v, d in zip(self.modules, tlm["wheel_mps"], tlm["steer_meas_rad"]):
            vx += (v * math.cos(d) + w * m.y_m) / 4.0
            vy += (v * math.sin(d) - w * m.x_m) / 4.0
        yaw_mid = self.yaw + 0.5 * w * dt
        c, s = math.cos(yaw_mid), math.sin(yaw_mid)
        self.x += (c * vx - s * vy) * dt
        self.y += (s * vx + c * vy) * dt
        self.reanchor(self.x, self.y, self.yaw + w * dt)
        return True

    def _update_tilt(self, acc_g: tuple, dt: float) -> None:
        """acc를 저역통과해 중력 방향에서 pitch(+전방 하향)/roll을 구한다."""
        a = float(dt) / (self.acc_tau + dt)
        self._acc = tuple(acc_g) if self._acc is None else tuple(o + a * (n - o) for o, n in zip(self._acc, acc_g))
        ax, ay, az = self._acc
        self.pitch = math.atan2(-ax, math.hypot(ay, az))
        self.roll = math.atan2(ay, az)
