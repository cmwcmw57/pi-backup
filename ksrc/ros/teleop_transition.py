"""수동 텔레옵의 큰 방향 전환에서 먼저 차체를 감속시킨다."""
import math

import numpy as np


class TeleopTransitionGuard:
    """현재 이동 방향과 크게 다른 새 병진 명령이면 정지 후 전달한다.

    조향이 맞기 전 구동 차단은 RoverEnv 의 drive_align_gate_deg 가 맡는다.
    둘을 함께 써야 이전 방향 관성과 잘못 정렬된 바퀴의 구동을 모두 줄인다.
    """

    def __init__(self, trigger_speed=0.05, brake_speed=0.02,
                 min_command=0.04, turn_deg=60.0,
                 max_brake_seconds=0.5, control_hz=50):
        self.trigger_speed = float(trigger_speed)
        self.brake_speed = float(brake_speed)
        self.min_command = float(min_command)
        self.turn_cos = math.cos(math.radians(turn_deg))
        self.max_brake_ticks = max(1, int(round(max_brake_seconds * control_hz)))
        self.braking = False
        self.brake_ticks = 0
        self.released = False

    def reset(self):
        self.braking = False
        self.brake_ticks = 0
        self.released = False

    def apply(self, requested, velocity_body):
        requested = np.asarray(requested, dtype=float)
        velocity_body = np.asarray(velocity_body, dtype=float)
        target_speed = float(np.linalg.norm(requested[:2]))
        actual_speed = float(np.linalg.norm(velocity_body[:2]))
        if target_speed < self.min_command:
            self.reset()
            return requested.copy()

        cosine = (float(np.dot(requested[:2], velocity_body[:2]) /
                        (target_speed * actual_speed))
                  if actual_speed > 1e-9 else 1.0)
        if self.braking:
            self.brake_ticks += 1
            # 경사에서 계속 미끄러질 때도 텔레옵이 영구 정지하지 않게 한다.
            if actual_speed > self.brake_speed and self.brake_ticks < self.max_brake_ticks:
                return np.zeros(3)
            self.braking = False
            self.released = True
        elif self.released:
            # 한 번 브레이크를 마친 동일 명령에 다시 진입해 덜컥거리지 않는다.
            if cosine >= self.turn_cos:
                self.released = False
        elif actual_speed > self.trigger_speed and cosine < self.turn_cos:
            self.braking = True
            self.brake_ticks = 0
            return np.zeros(3)
        return requested.copy()
