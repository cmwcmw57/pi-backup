# ================================================================
# orb_pose.py
# 설명: ORB-SLAM3 /orbslam3/base_pose(PoseStamped)를 경로 좌표 (x, y, yaw)와 pitch/roll로 바꾼다. 줄자 실측으로 정렬(reanchor)하기 전·추적 실패·지도 번호 변경·pose 튐에서는 ready=False로 로버를 세운다. ROS 구독(OrbSource)은 rclpy가 있는 라파/노트북에서만 쓴다.
# 사용법:
#   est = OrbAligner(); est.update(sample); est.reanchor(0.293, 0.320, 0.0)   # 시작 위치에서 정렬
#   source = OrbSource()   # rclpy 필요. source.recv() -> sample dict | None
# ================================================================
import math
import queue
import threading

import numpy as np

TRACK_OK = 2  # ORB-SLAM3 tracking_state: 2 = OK
MAX_JUMP_M = 0.3  # 한 샘플 사이에 이보다 크게 움직이면 ORB 튐으로 본다 (0.2 m/s * 수십 ms 의 몇 배)


def _rot(q: tuple[float, float, float, float]) -> np.ndarray:
    """쿼터니언 (x, y, z, w)을 회전행렬로 바꾼다."""
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _align_up(up: np.ndarray) -> np.ndarray:
    """up 벡터를 +z로 보내는 최소 회전 (Rodrigues)."""
    up = up / np.linalg.norm(up)
    v, c = np.cross(up, [0.0, 0.0, 1.0]), float(up[2])
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1.0 + c) if c > -1.0 + 1e-9 else np.diag([1.0, -1.0, -1.0])


class OrbAligner:
    """ORB 지도 좌표의 base pose를 reanchor()로 잡은 yaw·평행이동으로 경로 좌표에 옮긴다. PoseEstimator와 같은 인터페이스."""

    def __init__(self, max_jump_m: float = MAX_JUMP_M) -> None:
        self.max_jump = max_jump_m
        self.pitch = self.roll = 0.0
        self._tilt: np.ndarray | None = None  # 지도 -> 중력정렬 회전. 첫 정렬 때 로버가 수평이라는 가정으로 정한다
        self._theta, self._t = 0.0, np.zeros(2)
        self._anchored = False
        self._map: str | None = None
        self._last: tuple[np.ndarray, np.ndarray] | None = None  # 지도 좌표 (위치, 회전)
        self._ok, self.n_jump = False, 0

    @property
    def ready(self) -> bool:
        """정렬됐고 추적 OK이며 pose가 튀지 않았는지 반환한다."""
        return self._anchored and self._ok and self._last is not None

    @property
    def pose(self) -> tuple[float, float, float]:
        """경로 좌표 (x [m], y [m], yaw [rad]). 마지막 샘플 기준, 샘플이 없으면 NaN."""
        if self._last is None:
            return math.nan, math.nan, math.nan
        p, r = self._to_level(*self._last)
        c, s = math.cos(self._theta), math.sin(self._theta)
        yaw = math.atan2(r[1, 0], r[0, 0]) + self._theta
        return (c * p[0] - s * p[1] + self._t[0], s * p[0] + c * p[1] + self._t[1],
                math.atan2(math.sin(yaw), math.cos(yaw)))

    def update(self, sample: dict) -> bool:
        """sample = {frame_id, pos, quat, tracking_ok}. 새 샘플을 반영하면 True."""
        pos, rot = np.asarray(sample["pos"], float), _rot(sample["quat"])
        self._ok = bool(sample["tracking_ok"])
        if sample["frame_id"] != self._map:
            if self._map is not None:
                self._anchored = False  # 지도가 바뀌면 좌표가 달라진다 -> 줄자로 다시 정렬
            self._map, self._last = sample["frame_id"], None
        elif self._last is not None and self._anchored and np.linalg.norm(pos - self._last[0]) > self.max_jump:
            self._anchored = False
            self.n_jump += 1
        self._last = (pos, rot)
        if self._tilt is not None:
            _, r = self._to_level(pos, rot)
            self.pitch, self.roll = math.atan2(-r[2, 0], math.hypot(r[2, 1], r[2, 2])), math.atan2(r[2, 1], r[2, 2])
        return True

    def reanchor(self, x: float, y: float, yaw: float) -> None:
        """줄자로 잰 pose에 현재 ORB pose를 맞춘다. 첫 정렬이면 로버가 수평이라고 보고 중력 방향도 잡는다."""
        if self._last is None:
            return
        if self._tilt is None:
            self._tilt = _align_up(self._last[1][:, 2])
        self._theta, self._t, self._anchored = 0.0, np.zeros(2), True
        px, py, pyaw = self.pose
        self._theta = yaw - pyaw
        c, s = math.cos(self._theta), math.sin(self._theta)
        self._t = np.array([x - (c * (px) - s * py), y - (s * px + c * py)])

    def _to_level(self, pos: np.ndarray, rot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """지도 좌표 pose를 중력 정렬 좌표로 옮긴다 (정렬 전에는 그대로)."""
        t = np.eye(3) if self._tilt is None else self._tilt
        return t @ pos, t @ rot


class OrbSource:
    """rclpy로 base_pose와 tracking_state를 구독해 최신 샘플을 sample dict로 내준다."""

    def __init__(self, pose_topic: str = "/orbslam3/base_pose", state_topic: str = "/orbslam3/tracking_state",
                 timeout_s: float = 0.5) -> None:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from std_msgs.msg import Int32
        self.timeout_s, self.track, self.q = timeout_s, TRACK_OK + 1, queue.Queue(maxsize=1)
        rclpy.init()
        self.node = rclpy.create_node("yeouido_orb_pose")
        self.node.create_subscription(PoseStamped, pose_topic, self._on_pose, 10)
        self.node.create_subscription(Int32, state_topic, lambda m: setattr(self, "track", m.data), 10)
        self._rclpy = rclpy
        threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True).start()

    def _on_pose(self, m) -> None:
        p, o = m.pose.position, m.pose.orientation
        s = {"t_ms": round((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9) * 1000), "frame_id": m.header.frame_id,
             "pos": (p.x, p.y, p.z), "quat": (o.x, o.y, o.z, o.w), "tracking_ok": self.track == TRACK_OK}
        if self.q.full():
            self.q.get_nowait()
        self.q.put(s)

    def poll(self) -> dict | None:
        """대기 없이 최신 pose 샘플, 없으면 None."""
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None

    def recv(self) -> dict | None:
        """다음 pose 샘플, timeout_s 안에 없으면 None."""
        try:
            return self.q.get(timeout=self.timeout_s)
        except queue.Empty:
            return None

    def close(self) -> None:
        """ROS를 종료한다."""
        self._rclpy.try_shutdown()
