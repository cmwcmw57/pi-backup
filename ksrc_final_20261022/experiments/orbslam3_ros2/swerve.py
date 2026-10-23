# ================================================================
# swerve.py
# 설명: 4륜 독립 조향 로버의 평면 운동학과 자동 폐루프 경로 controller를 제공한다.
# 사용법:
#   from experiments.orbslam3_ros2.swerve import SwerveGeometry, LoopController
# ================================================================

import json
import numpy as np
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModuleGeometry:
    """한 swerve module의 joint 이름과 base_link 기준 조향축 XY를 보관한다."""

    name: str
    steering_joint: str
    wheel_joint: str
    x_m: float
    y_m: float
    steering_sign: float = 1.0
    rolling_heading_rad: float = 0.0


@dataclass(frozen=True)
class SwerveGeometry:
    """네 swerve module과 바퀴 반지름을 보관한다."""

    modules: tuple[ModuleGeometry, ...]
    wheel_radius_m: float = 0.0475

    @classmethod
    def from_json(cls, path: Path) -> "SwerveGeometry":
        """검증된 rover geometry JSON에서 평면 운동학 항목만 읽는다."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        modules = tuple(
            ModuleGeometry(
                name=item["name"],
                steering_joint=item["steering_joint"],
                wheel_joint=item["wheel_joint"],
                x_m=float(item["steering_pivot_base_link_m"][0]),
                y_m=float(item["steering_pivot_base_link_m"][1]),
                steering_sign=float(item.get("steering_sign", 1.0)),
                rolling_heading_rad=float(item.get("rolling_heading_at_zero_rad", 0.0)),
            )
            for item in data["modules"]
        )
        if len(modules) != 4:
            raise ValueError(f"swerve geometry requires four modules, got {len(modules)}")
        return cls(modules=modules, wheel_radius_m=float(data.get("wheel_radius_m", 0.0475)))


@dataclass(frozen=True)
class ModuleCommand:
    """한 swerve module의 steering position과 wheel velocity 명령을 보관한다."""

    steering_joint: str
    wheel_joint: str
    steering_angle_rad: float
    wheel_angular_velocity_rad_s: float


@dataclass(frozen=True)
class BodyCommand:
    """base_link 기준 평면 속도 명령을 보관한다."""

    vx: float
    vy: float
    yaw_rate: float


@dataclass(frozen=True)
class LoopRoute:
    """폐루프 waypoint, 반복 횟수와 속도·시간 제한을 보관한다."""

    waypoints: tuple[tuple[float, float], ...]
    laps: int
    max_speed_m_s: float = 0.08
    wait_s: float = 3.0
    ramp_s: float = 2.0
    final_hold_s: float = 2.0
    tolerance_m: float = 0.04

    @classmethod
    def default(cls) -> "LoopRoute":
        """settled-sand 평지 영역 안의 0.4×0.8 m 경로를 두 바퀴로 만든다."""
        return cls(
            waypoints=(
                (3.5, -0.4),
                (3.9, -0.4),
                (3.9, 0.4),
                (3.5, 0.4),
                (3.5, -0.4),
            ),
            laps=2,
        )


def fold_steering_angle(angle_rad: float) -> tuple[float, float]:
    """조향을 ±90°로 접고 같은 속도를 만드는 wheel 부호를 반환한다."""
    angle = math.atan2(math.sin(angle_rad), math.cos(angle_rad))
    sign = 1.0
    if angle > math.pi / 2:
        angle -= math.pi
        sign = -1.0
    elif angle < -math.pi / 2:
        angle += math.pi
        sign = -1.0
    return angle, sign


def body_twist_to_modules(
    geometry: SwerveGeometry,
    *,
    vx: float,
    vy: float,
    yaw_rate: float,
) -> tuple[ModuleCommand, ...]:
    """base_link twist를 네 steering position과 wheel velocity로 변환한다."""
    if not all(math.isfinite(value) for value in (vx, vy, yaw_rate)):
        raise ValueError("body twist must be finite")
    commands = []
    for module in geometry.modules:
        module_vx = vx - yaw_rate * module.y_m
        module_vy = vy + yaw_rate * module.x_m
        # 조향 joint 각도 = 부호 * (목표 heading - 0 rad 조향에서 +wheel 속도의 구름 방향)
        relative, sign = fold_steering_angle(
            math.atan2(module_vy, module_vx) - module.rolling_heading_rad
        )
        steering = module.steering_sign * relative
        wheel_speed = sign * math.hypot(module_vx, module_vy) / geometry.wheel_radius_m
        commands.append(
            ModuleCommand(
                steering_joint=module.steering_joint,
                wheel_joint=module.wheel_joint,
                steering_angle_rad=steering,
                wheel_angular_velocity_rad_s=wheel_speed,
            )
        )
    return tuple(commands)


class LoopController:
    """현재 world pose에서 다음 waypoint를 향하는 body twist를 순서대로 만든다."""

    def __init__(self, route: LoopRoute) -> None:
        if route.laps < 1 or len(route.waypoints) < 2:
            raise ValueError("loop route requires at least one lap and two waypoints")
        if route.waypoints[0] != route.waypoints[-1]:
            raise ValueError("loop route must be closed")
        self.route = route
        self._targets = tuple(route.waypoints[1:]) * route.laps
        self._target_index = 0
        self._finished_at_s: float | None = None

    @property
    def waypoints_reached(self) -> int:
        """완료한 segment endpoint 개수를 반환한다."""
        return self._target_index

    @property
    def complete(self) -> bool:
        """모든 lap의 waypoint가 완료됐는지 반환한다."""
        return self._target_index >= len(self._targets)

    def held_after_finish(self, elapsed_s: float) -> bool:
        """경로 완료 후 final hold 시간까지 지났는지 반환한다."""
        return (
            self._finished_at_s is not None
            and elapsed_s - self._finished_at_s >= self.route.final_hold_s
        )

    def command(self, *, elapsed_s: float, x: float, y: float, yaw_rad: float) -> BodyCommand:
        """현재 pose와 경과 시간으로 다음 body-frame twist를 계산한다."""
        if elapsed_s < self.route.wait_s or self.complete:
            return BodyCommand(0.0, 0.0, 0.0)

        while self._target_index < len(self._targets):
            target_x, target_y = self._targets[self._target_index]
            dx = target_x - x
            dy = target_y - y
            distance = math.hypot(dx, dy)
            if distance > self.route.tolerance_m:
                break
            self._target_index += 1
        if self.complete:
            self._finished_at_s = elapsed_s
            return BodyCommand(0.0, 0.0, 0.0)

        speed_fraction = min(1.0, max(0.0, (elapsed_s - self.route.wait_s) / self.route.ramp_s))
        speed = min(self.route.max_speed_m_s * speed_fraction, distance)
        world_vx = speed * dx / distance
        world_vy = speed * dy / distance
        cosine = math.cos(yaw_rad)
        sine = math.sin(yaw_rad)
        body_vx = cosine * world_vx + sine * world_vy
        body_vy = -sine * world_vx + cosine * world_vy
        yaw_error = math.atan2(math.sin(yaw_rad), math.cos(yaw_rad))
        yaw_rate = max(-0.4, min(0.4, -1.5 * yaw_error))
        return BodyCommand(body_vx, body_vy, yaw_rate)


def body_twist_from_modules(
    geometry: SwerveGeometry,
    steering_angles: list[float],
    wheel_velocities: list[float],
) -> tuple[float, float, float, float]:
    """측정한 네 조향각·바퀴 각속도에서 최소제곱으로 (vx, vy, yaw_rate)와 잔차 RMS(m/s)를 구한다.

    body_twist_to_modules의 역변환이며 인자 순서는 geometry.modules와 같다.
    """
    if len(steering_angles) != 4 or len(wheel_velocities) != 4:
        raise ValueError("four steering angles and wheel velocities required")
    rows, rhs = [], []
    for module, angle, omega in zip(geometry.modules, steering_angles, wheel_velocities):
        heading = module.rolling_heading_rad + module.steering_sign * angle
        speed = omega * geometry.wheel_radius_m
        rows += [[1., 0., -module.y_m], [0., 1., module.x_m]]
        rhs += [speed * math.cos(heading), speed * math.sin(heading)]
    matrix, vector = np.asarray(rows), np.asarray(rhs)
    solution = np.linalg.lstsq(matrix, vector, rcond=None)[0]
    residual = float(np.sqrt(np.mean((matrix @ solution - vector) ** 2)))
    return float(solution[0]), float(solution[1]), float(solution[2]), residual
