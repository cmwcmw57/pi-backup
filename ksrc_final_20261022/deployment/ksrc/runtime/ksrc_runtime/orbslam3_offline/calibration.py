# ================================================================
# calibration.py
# 설명: portable URDF의 fixed-joint 변환으로 ORB-SLAM3 stereo/IMU 보정을 만든다.
# 사용법:
#   from experiments.orbslam3_offline.calibration import build_orb_calibration, write_orb_yaml
# ================================================================

import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """ORB 설정에 기록할 pinhole camera 내부 파라미터를 보관한다."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


_FLOAT_KEYS = {
    "Stereo.ThDepth",
    "IMU.NoiseGyro", "IMU.NoiseAcc", "IMU.GyroWalk", "IMU.AccWalk", "IMU.Frequency",
    "ORBextractor.scaleFactor",
    "Viewer.KeyFrameSize", "Viewer.KeyFrameLineWidth", "Viewer.GraphLineWidth",
    "Viewer.PointSize", "Viewer.CameraSize", "Viewer.CameraLineWidth",
    "Viewer.ViewpointX", "Viewer.ViewpointY", "Viewer.ViewpointZ",
    "Viewer.ViewpointF", "Viewer.imageViewScale",
}
_FLOAT_KEYS.update(
    f"{camera}.{name}"
    for camera in ("Camera1", "Camera2")
    for name in ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2")
)

_MATRIX_KEYS = {
    "Stereo.T_c1_c2",
    "IMU.T_b_c1",
    "ROS.T_c1_base",
}


def _origin_transform(origin: ElementTree.Element | None) -> np.ndarray:
    xyz = [float(value) for value in origin.get("xyz", "0 0 0").split()] if origin is not None else [0.0] * 3
    roll, pitch, yaw = (
        [float(value) for value in origin.get("rpy", "0 0 0").split()]
        if origin is not None
        else [0.0] * 3
    )
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = xyz
    return transform


def load_frame_transform(urdf: Path, reference_frame: str, frame: str) -> np.ndarray:
    """URDF의 fixed 경로에서 frame 좌표를 reference_frame 좌표로 옮긴다."""
    root = ElementTree.parse(urdf).getroot()
    graph: dict[str, list[tuple[str, np.ndarray, str, str]]] = {
        link.attrib["name"]: [] for link in root.findall("link")
    }
    for joint in root.findall("joint"):
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        transform = _origin_transform(joint.find("origin"))
        name = joint.attrib["name"]
        joint_type = joint.attrib["type"]
        graph[parent].append((child, transform, name, joint_type))
        graph[child].append((parent, np.linalg.inv(transform), name, joint_type))

    for name in (reference_frame, frame):
        if name not in graph:
            raise ValueError(f"missing frame: {name}")
    if reference_frame == frame:
        return np.eye(4)

    queue = deque([reference_frame])
    previous: dict[str, tuple[str, np.ndarray, str, str]] = {}
    seen = {reference_frame}
    while queue and frame not in seen:
        current = queue.popleft()
        for neighbor, transform, joint_name, joint_type in graph[current]:
            if neighbor not in seen:
                seen.add(neighbor)
                previous[neighbor] = (current, transform, joint_name, joint_type)
                queue.append(neighbor)
    if frame not in seen:
        raise ValueError(f"no joint path from {reference_frame} to {frame}")

    edges: list[tuple[np.ndarray, str, str]] = []
    current = frame
    while current != reference_frame:
        parent, transform, joint_name, joint_type = previous[current]
        edges.append((transform, joint_name, joint_type))
        current = parent
    result = np.eye(4)
    for transform, joint_name, joint_type in reversed(edges):
        if joint_type != "fixed":
            raise ValueError(f"non-fixed joint on path: {joint_name} ({joint_type})")
        result = result @ transform
    return result


def build_orb_calibration(
    urdf: Path,
    intrinsics: CameraIntrinsics,
    camera_hz: float = 30.0,
    imu_hz: float = 200.0,
) -> dict[str, object]:
    """렌더된 내부 파라미터와 URDF 외부 파라미터를 ORB 설정 키에 담는다."""
    defaults_path = Path(__file__).parent / "config/orb_defaults.json"
    calibration: dict[str, object] = json.loads(defaults_path.read_text())
    for camera in ("Camera1", "Camera2"):
        for name in ("fx", "fy", "cx", "cy"):
            calibration[f"{camera}.{name}"] = getattr(intrinsics, name)
        for name in ("k1", "k2", "p1", "p2"):
            calibration[f"{camera}.{name}"] = 0.0
    calibration["Camera.width"] = intrinsics.width
    calibration["Camera.height"] = intrinsics.height
    calibration["Camera.fps"] = round(camera_hz)
    calibration["IMU.Frequency"] = imu_hz
    calibration["Stereo.T_c1_c2"] = load_frame_transform(
        urdf, "camera_infra1_optical_frame", "camera_infra2_optical_frame"
    ).tolist()
    calibration["IMU.T_b_c1"] = load_frame_transform(
        urdf, "imu_sensor_frame", "camera_infra1_optical_frame"
    ).tolist()
    calibration["ROS.T_c1_base"] = load_frame_transform(
        urdf, "camera_infra1_optical_frame", "base_link"
    ).tolist()
    return calibration


def write_orb_yaml(path: Path, calibration: dict[str, object]) -> None:
    """OpenCV FileStorage 형식의 ORB 설정 YAML을 기록한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["%YAML:1.0", "---"]
    for key, value in calibration.items():
        if key in _MATRIX_KEYS:
            matrix = np.asarray(value, dtype=np.float32)
            if matrix.shape != (4, 4):
                raise ValueError(f"{key} must be a 4x4 matrix")
            data = ", ".join(format(float(number), ".9g") for number in matrix.flat)
            lines.extend([f"{key}: !!opencv-matrix", "  rows: 4", "  cols: 4", "  dt: f", f"  data: [ {data} ]"])
        elif isinstance(value, str):
            lines.append(f"{key}: {json.dumps(value)}")
        elif key in _FLOAT_KEYS:
            lines.append(f"{key}: {float(value)}")
        else:
            lines.append(f"{key}: {value}")
    path.write_text("\n".join(lines) + "\n")
