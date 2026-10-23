# ================================================================
# drive_runtime.py
# 설명: Isaac articulation joint drive, ROS joint/ground-truth telemetry와 주행 증거를 수집한다.
# 사용법:
#   from experiments.orbslam3_ros2.drive_runtime import DriveSession, DriveTelemetry
# ================================================================

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from experiments.orbslam3_ros2.swerve import (
    LoopController,
    LoopRoute,
    SwerveGeometry,
    body_twist_to_modules,
)


def write_tum_rows(path: Path, rows: Sequence[Sequence[float]]) -> None:
    """검증된 8열 TUM pose 행을 임시 파일에서 원자적으로 공개한다."""
    values = np.asarray(rows, dtype=float)
    if values.ndim != 2 or values.shape[1] != 8 or values.shape[0] == 0:
        raise ValueError("TUM trajectory requires a nonempty Nx8 array")
    if not np.isfinite(values).all():
        raise ValueError("TUM trajectory contains non-finite values")
    if np.any(np.diff(values[:, 0]) <= 0.0):
        raise ValueError("TUM timestamps must be strictly increasing")
    quaternion_norms = np.linalg.norm(values[:, 4:8], axis=1)
    if np.any(quaternion_norms < 1e-12):
        raise ValueError("TUM trajectory contains invalid quaternions")
    normalized = values.copy()
    normalized[:, 4:8] /= quaternion_norms[:, None]

    output = Path(path)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in normalized:
            stream.write(" ".join(f"{value:.9f}" for value in row) + "\n")
    temporary.replace(output)


def yaw_from_wxyz(quaternion: np.ndarray) -> float:
    """scalar-first quaternion에서 Z축 yaw를 계산한다."""
    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _ros_time(seconds: float) -> Any:
    """simulation seconds를 builtin_interfaces/Time으로 변환한다."""
    from builtin_interfaces.msg import Time

    sec = math.floor(seconds)
    nanosec = int(round((seconds - sec) * 1_000_000_000))
    if nanosec == 1_000_000_000:
        sec += 1
        nanosec = 0
    return Time(sec=int(sec), nanosec=nanosec)


class DriveTelemetry:
    """joint state와 physical base pose를 발행하고 실제 loopback 수신 수를 센다."""

    def __init__(self, node: Any) -> None:
        from geometry_msgs.msg import PoseStamped
        from rclpy.qos import QoSProfile
        from sensor_msgs.msg import JointState

        qos = QoSProfile(depth=100)
        self._node = node
        self._joint_type = JointState
        self._pose_type = PoseStamped
        self._joint_publisher = node.create_publisher(
            JointState, "/rover/joint_states", qos
        )
        self._pose_publisher = node.create_publisher(
            PoseStamped, "/rover/ground_truth/pose", qos
        )
        self.received_joint_states = 0
        self.received_ground_truth_poses = 0
        self.ground_truth_rows: list[tuple[float, ...]] = []
        node.create_subscription(
            JointState,
            "/rover/joint_states",
            self._on_joint_state,
            qos,
        )
        node.create_subscription(
            PoseStamped,
            "/rover/ground_truth/pose",
            self._on_ground_truth,
            qos,
        )
        self._next_joint_s = 0.0
        self._next_pose_s = 0.0

    def _on_joint_state(self, _message: Any) -> None:
        self.received_joint_states += 1

    def _on_ground_truth(self, _message: Any) -> None:
        self.received_ground_truth_poses += 1

    def publish_if_due(self, simulation_time_s: float, robot: Any) -> None:
        """50 Hz joint state와 30 Hz ground-truth pose를 simulation timestamp로 발행한다."""
        if simulation_time_s + 1e-12 >= self._next_joint_s:
            message = self._joint_type()
            message.header.stamp = _ros_time(simulation_time_s)
            message.header.frame_id = "base_link"
            message.name = list(robot.dof_names)
            message.position = [float(value) for value in robot.get_joint_positions()]
            message.velocity = [float(value) for value in robot.get_joint_velocities()]
            self._joint_publisher.publish(message)
            while self._next_joint_s <= simulation_time_s + 1e-12:
                self._next_joint_s += 1.0 / 50.0

        if simulation_time_s + 1e-12 >= self._next_pose_s:
            position, orientation = robot.get_world_pose()
            message = self._pose_type()
            message.header.stamp = _ros_time(simulation_time_s)
            message.header.frame_id = "sim_world"
            message.pose.position.x = float(position[0])
            message.pose.position.y = float(position[1])
            message.pose.position.z = float(position[2])
            message.pose.orientation.w = float(orientation[0])
            message.pose.orientation.x = float(orientation[1])
            message.pose.orientation.y = float(orientation[2])
            message.pose.orientation.z = float(orientation[3])
            self._pose_publisher.publish(message)
            self.ground_truth_rows.append(
                (
                    simulation_time_s,
                    float(position[0]),
                    float(position[1]),
                    float(position[2]),
                    float(orientation[1]),
                    float(orientation[2]),
                    float(orientation[3]),
                    float(orientation[0]),
                )
            )
            while self._next_pose_s <= simulation_time_s + 1e-12:
                self._next_pose_s += 1.0 / 30.0

    def write_ground_truth_tum(self, path: Path) -> None:
        """발행한 physical base pose를 TUM trajectory로 원자 저장한다."""
        write_tum_rows(path, self.ground_truth_rows)


class DriveSession:
    """네 steering position과 네 wheel velocity drive만으로 loop 경로를 추종한다."""

    def __init__(self, robot: Any, geometry_path: Path) -> None:
        from isaacsim.core.utils.types import ArticulationAction

        self._action_type = ArticulationAction
        self.robot = robot
        self.geometry = SwerveGeometry.from_json(geometry_path)
        self.loop = LoopController(LoopRoute.default())
        names = list(robot.dof_names)
        self.steering_indices = np.asarray(
            [robot.get_dof_index(module.steering_joint) for module in self.geometry.modules],
            dtype=np.int32,
        )
        self.wheel_indices = np.asarray(
            [robot.get_dof_index(module.wheel_joint) for module in self.geometry.modules],
            dtype=np.int32,
        )
        self.resolved_steering_joints = [names[index] for index in self.steering_indices]
        self.resolved_wheel_joints = [names[index] for index in self.wheel_indices]
        if len(set(self.steering_indices) | set(self.wheel_indices)) != 8:
            raise RuntimeError("drive requires eight unique steering/wheel DOFs")

        self.controller = robot.get_articulation_controller()
        kps, kds = (np.asarray(values, dtype=float) for values in self.controller.get_gains())
        for index in self.steering_indices:
            kps[index] = 10.0
            kds[index] = 0.3
            self.controller.switch_dof_control_mode(int(index), "position")
        for index in self.wheel_indices:
            kps[index] = 0.0
            kds[index] = 0.2
            self.controller.switch_dof_control_mode(int(index), "velocity")
        self.controller.set_gains(kps=kps, kds=kds)
        self.controller.set_max_efforts(
            np.full(4, 2.942, dtype=float), self.steering_indices
        )
        self.controller.set_max_efforts(
            np.full(4, 1.0, dtype=float), self.wheel_indices
        )

        position, _ = robot.get_world_pose()
        self.initial_position = np.asarray(position, dtype=float)
        self.final_position = self.initial_position.copy()
        self._previous_position = self.initial_position.copy()
        self.path_length_m = 0.0
        self.max_abs_wheel_velocity_rad_s = 0.0
        self.max_abs_steering_position_rad = 0.0

    def apply(self, elapsed_s: float) -> None:
        """현재 physical pose에서 계산한 두 종류의 joint action을 적용한다."""
        position, orientation = self.robot.get_world_pose()
        body = self.loop.command(
            elapsed_s=elapsed_s,
            x=float(position[0]),
            y=float(position[1]),
            yaw_rad=yaw_from_wxyz(np.asarray(orientation, dtype=float)),
        )
        modules = body_twist_to_modules(
            self.geometry,
            vx=body.vx,
            vy=body.vy,
            yaw_rate=body.yaw_rate,
        )
        self.controller.apply_action(
            self._action_type(
                joint_positions=np.asarray(
                    [command.steering_angle_rad for command in modules], dtype=float
                ),
                joint_indices=self.steering_indices,
            )
        )
        self.controller.apply_action(
            self._action_type(
                joint_velocities=np.asarray(
                    [command.wheel_angular_velocity_rad_s for command in modules],
                    dtype=float,
                ),
                joint_indices=self.wheel_indices,
            )
        )

    def observe(self) -> None:
        """물리 step 이후 실제 joint와 base 이동량을 누적한다."""
        positions = np.asarray(self.robot.get_joint_positions(), dtype=float)
        velocities = np.asarray(self.robot.get_joint_velocities(), dtype=float)
        self.max_abs_steering_position_rad = max(
            self.max_abs_steering_position_rad,
            float(np.max(np.abs(positions[self.steering_indices]))),
        )
        self.max_abs_wheel_velocity_rad_s = max(
            self.max_abs_wheel_velocity_rad_s,
            float(np.max(np.abs(velocities[self.wheel_indices]))),
        )
        position, _ = self.robot.get_world_pose()
        self.final_position = np.asarray(position, dtype=float)
        self.path_length_m += float(
            np.linalg.norm(self.final_position[:2] - self._previous_position[:2])
        )
        self._previous_position = self.final_position.copy()

    def stop(self) -> None:
        """flush 구간 전에 네 wheel velocity target을 0으로 만든다."""
        self.controller.apply_action(
            self._action_type(
                joint_velocities=np.zeros(4, dtype=float),
                joint_indices=self.wheel_indices,
            )
        )

    def report(self, telemetry: DriveTelemetry, duration_s: float) -> dict[str, Any]:
        """GPU smoke test가 소비할 drive proof manifest를 만든다."""
        if not math.isfinite(duration_s) or duration_s < 0.0:
            raise ValueError(f"drive duration must be finite and non-negative: {duration_s}")
        return {
            "status": "complete",
            "resolved_steering_joints": self.resolved_steering_joints,
            "resolved_wheel_joints": self.resolved_wheel_joints,
            "max_abs_wheel_velocity_rad_s": self.max_abs_wheel_velocity_rad_s,
            "max_abs_steering_position_rad": self.max_abs_steering_position_rad,
            "base_displacement_m": float(
                np.linalg.norm(self.final_position[:2] - self.initial_position[:2])
            ),
            "base_path_length_m": self.path_length_m,
            "initial_base_position_m": self.initial_position.tolist(),
            "final_base_position_m": self.final_position.tolist(),
            "waypoints_reached": self.loop.waypoints_reached,
            "route_complete": self.loop.complete,
            "duration_s": duration_s,
            "received_joint_states": telemetry.received_joint_states,
            "received_ground_truth_poses": telemetry.received_ground_truth_poses,
            "authored_base_pose_writes_after_start": 0,
            "control": "steering_position_and_wheel_velocity_only",
        }
