# ================================================================
# eskf_pose.py
# 설명: 실기 SPI 텔레메트리 IMU(50 Hz, 20 ms 평균)와 ORB-SLAM3 base_pose를 rover-mapping의 OnlineEstimator(ESKF)에 먹여 경로 좌표 pose를 낸다. IMU 끊김(0.1 s 초과·IMU invalid)이면 정지(ready=False) 후 정상 IMU가 1 s 이어지면 마지막 유효 pose로 필터를 재초기화한다.
# 사용법:
#   est = EskfPose(0.293, 0.320, 0.463, 0.0); est.update({"t_ms", "host_s", "tlm": dict|None, "orb": dict|None})
#   if est.ready: x, y, yaw = est.pose        # 환경변수 ROVER_MAPPING_ROOT=rover-mapping 체크아웃 경로 필수 (읽기 전용 import, scipy 필요)
# ================================================================
import importlib
import json
import math
import os
import sys
import time
import types
from collections import deque
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

CALIBRATION_PATH = Path(__file__).with_name("eskf_calibration.json")
GRAVITY = 9.81
FLAG_IMU_VALID = 1 << 4


def _online_estimator():
    """rover-mapping의 OnlineEstimator를 이 repo의 experiments 패키지와 충돌 없이 불러온다 (그 repo는 수정하지 않는다)."""
    if "experiments.localization" not in sys.modules:
        root = os.environ.get("ROVER_MAPPING_ROOT")
        online = Path(root or ".") / "experiments" / "localization" / "online.py"
        if not root or not online.is_file():
            raise RuntimeError("환경변수 ROVER_MAPPING_ROOT에 rover-mapping 체크아웃 경로를 지정해야 한다 (experiments/localization/online.py 필요)")
        pkg = types.ModuleType("experiments.localization")
        pkg.__path__ = [str(online.parent)]
        sys.modules["experiments.localization"] = pkg
    return importlib.import_module("experiments.localization.online").OnlineEstimator


class EskfPose:
    """OnlineEstimator 래퍼. update(sample)로 IMU·ORB를 넣고 ready/pose/pitch/roll을 읽는다 (PoseEstimator와 같은 인터페이스)."""

    def __init__(self, x: float, y: float, z: float = 0.0, yaw: float = 0.0, *, calibration: dict | None = None,
                 lag_s: float = 0.25, imu_offset_s: float = -0.047, bias_samples: int = 100, gap_s: float = 0.1,
                 restart_s: float = 1.0, adaptive_visual_noise: bool = True) -> None:
        self.cal = calibration or json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
        self.cam_from_base = np.asarray(self.cal["ROS.T_c1_base"])
        self.rot_base_from_imu = (np.linalg.inv(self.cam_from_base) @ np.linalg.inv(np.asarray(self.cal["IMU.T_b_c1"])))[:3, :3]
        self.lag_s, self.imu_offset_s, self.bias_n, self.gap_s, self.restart_s = lag_s, imu_offset_s, bias_samples, gap_s, restart_s
        self.adaptive = adaptive_visual_noise
        self._start = (x, y, z, yaw)
        self._est = None
        self._buf: list[tuple[np.ndarray, np.ndarray]] = []
        self._bias = np.zeros(3)
        self._tick: int | None = None
        self._t: float | None = None
        self._offs: deque = deque(maxlen=500)
        self._hold, self._healthy, self.n_gap, self._imu_host, self._end = False, 0.0, 0, 0.0, -math.inf
        self._T: np.ndarray | None = None
        self._last_good: np.ndarray | None = None
        self._valid = False
        self.status, self.pitch, self.roll = "init", 0.0, 0.0

    @property
    def ready(self) -> bool:
        """필터가 돌고 있고 IMU 끊김 정지 중이 아니며 영상 보정이 최근 받아들여졌는지 반환한다."""
        return self._est is not None and not self._hold and self._valid

    @property
    def pose(self) -> tuple[float, float, float]:
        """경로 좌표 (x [m], y [m], yaw [rad]). 필터가 아직 없으면 시작 pose."""
        if self._T is None:
            return self._start[0], self._start[1], self._start[3]
        return float(self._T[0, 3]), float(self._T[1, 3]), math.atan2(self._T[1, 0], self._T[0, 0])

    def update(self, s: dict) -> bool:
        """sample = {host_s, tlm: 텔레메트리 dict|None, orb: ORB 샘플|None}을 반영한다."""
        host = s["host_s"]
        if s.get("tlm") is not None:
            self._imu(s["tlm"], host)
        if s.get("orb") is not None and self._est is not None and not self._hold:
            self._frame(s["orb"], host)
        if self._est is not None and not self._hold and host - self._imu_host > self.gap_s:
            self._stop_for_gap()  # 유효 IMU가 0.1 s 넘게 없다 (invalid 프레임이 이어지는 경우)
        if self._est is not None and not self._hold:
            self._T, self._valid, _, self.status = self._est.pose(host)
            if self._valid:
                self._last_good = self._T.copy()
            r = self._T[:3, :3]
            self.pitch, self.roll = math.atan2(-r[2, 0], math.hypot(r[2, 1], r[2, 2])), math.atan2(r[2, 1], r[2, 2])
        return True

    def reanchor(self, x: float, y: float, yaw: float) -> None:
        """줄자 실측 pose로 필터를 재초기화한다 (기울기·gyro bias는 유지, 속도는 0)."""
        base = self._last_good if self._last_good is not None else self._T
        if base is None:
            self._start = (x, y, self._start[2], yaw)
            return
        t = base.copy()
        t[:3, :3] = Rotation.from_euler("z", yaw - math.atan2(t[1, 0], t[0, 0])).as_matrix() @ t[:3, :3]
        t[0, 3], t[1, 3] = x, y
        self._last_good = t
        if not self._hold:
            self._build(t)

    def _build(self, start: np.ndarray) -> None:
        """start(base pose)에서 OnlineEstimator를 새로 만든다. gyro bias는 정지 평균값으로 시작한다."""
        self._est = _online_estimator()(start, self.cal, lag_s=self.lag_s, adaptive_visual_noise=self.adaptive)
        self._est.filter.gyro_bias = self._bias.copy()
        self._valid, self._T = False, start

    def _imu(self, tlm: dict, host: float) -> None:
        """텔레메트리 IMU 한 프레임: 중복 tick 제거, 끊김 감지, bias 측정, 필터 전진."""
        if tlm["tick"] == self._tick:
            return
        self._tick = tlm["tick"]
        if not tlm.get("flags", FLAG_IMU_VALID) & FLAG_IMU_VALID:
            return  # IMU invalid: 건너뛰면 다음 유효 프레임에서 dt가 커져 끊김으로 처리된다
        t, prev, self._t = tlm["t_ms"] / 1000.0, self._t, tlm["t_ms"] / 1000.0
        self._offs.append(host - t)
        self._imu_host = host
        if prev is None or t <= prev:  # 중복·역순 시각은 걸러낸다 (on_imu는 단조 증가를 검사하지 않는다)
            return
        dt = t - prev
        gyro_b, force_b = np.radians(tlm["gyro_dps"]), np.asarray(tlm["acc_g"], float) * GRAVITY
        if dt > self.gap_s:
            self._stop_for_gap()
            return
        if self._est is None:
            self._buf.append((gyro_b, force_b))
            if len(self._buf) >= self.bias_n:
                self._start_filter()
            return
        if self._hold:
            self._healthy += dt
            if self._healthy >= self.restart_s:
                self._hold = False
                self._build(self._last_good if self._last_good is not None else self._T)
            return
        r = self.rot_base_from_imu.T
        end = t + min(self._offs) + self.imu_offset_s
        if end > self._end:
            self._end = end
            self._est.on_imu(end, dt, r @ gyro_b, r @ force_b)

    def _stop_for_gap(self) -> None:
        """IMU 끊김: 정지(ready=False)하고 정상 IMU가 restart_s 이어질 때까지 필터를 전진시키지 않는다."""
        if not self._hold:
            self._hold, self.n_gap, self.status = True, self.n_gap + 1, "imu_gap"
        self._healthy = 0.0

    def _start_filter(self) -> None:
        """정지 구간 평균으로 gyro bias와 시작 roll/pitch(중력)를 잡고 시작 pose에서 필터를 만든다."""
        gyro = np.mean([g for g, _ in self._buf], axis=0)
        force = np.mean([f for _, f in self._buf], axis=0)
        self._bias = self.rot_base_from_imu.T @ gyro
        x, y, z, yaw = self._start
        roll, pitch = math.atan2(force[1], force[2]), math.atan2(-force[0], math.hypot(force[1], force[2]))
        start = np.eye(4)
        start[:3, :3] = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
        start[:3, 3] = (x, y, z)
        self._last_good = start
        self._build(start)

    def _frame(self, orb: dict, host: float) -> None:
        """ORB base pose를 world_from_camera로 바꿔 필터에 넣는다 (카메라 extrinsic은 상쇄되어 base pose만 의미가 있다)."""
        t = np.eye(4)
        t[:3, :3], t[:3, 3] = Rotation.from_quat(orb["quat"]).as_matrix(), orb["pos"]
        try:
            map_id = int(str(orb["frame_id"]).rsplit("_", 1)[-1])
        except ValueError:
            map_id = -1
        self._est.on_frame(host, 2 if orb["tracking_ok"] else 0, map_id, t @ np.linalg.inv(self.cam_from_base))


class EskfSource:
    """텔레메트리 소스(recv)에 ORB 소스의 최신 샘플(poll)을 붙여 EskfPose.update가 받는 sample을 만든다."""

    def __init__(self, telemetry, orb) -> None:
        self.telemetry, self.orb = telemetry, orb

    def recv(self) -> dict | None:
        """텔레메트리 한 프레임(없으면 None)과 그 사이 도착한 ORB 샘플."""
        tlm = self.telemetry.recv()
        if tlm is None:
            return None
        return {"t_ms": tlm["t_ms"], "host_s": tlm.pop("host_s", None) or time.time(), "tlm": tlm, "orb": self.orb.poll()}
