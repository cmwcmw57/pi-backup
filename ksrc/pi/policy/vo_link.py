"""VO 몸체 속도: ORB 자세 -> 속도 추정, 그리고 릴레이 -> run_policy UDP 패킷 (2026-10-16).

student v3 의 관측 `vo` = [vx, vy, valid, age] (sim/isaac/ksrc_rover/.../mdp/vo.py).  학습에서는
"delay 만큼 전의 몸체 속도"를 5-17 Hz 로 받았다.  여기서는 ORB-SLAM3 base 자세(지도 좌표) 몇 개를
짧은 창(기본 0.2 s)에서 직선 맞춤해 속도를 구하고, 창의 가운데 시각을 측정 시각으로 둔다.
-> age = (지금 - 그 시각) 은 카메라·ORB 지연 + 창 절반 + 전송이 모두 들어간 실제 나이.

몸체 속도는 지도 좌표계를 어떻게 잡았든 같다 (v_body = R_wb^T v_w).  ORB 지도의 중력 정렬이 없어도 된다.
지도 번호가 바뀌면(재위치·새 지도) 이력을 비운다 -> 지도 전환 순간의 튐을 내보내지 않는다.

패킷 (리틀엔디언 28 B): magic b"KVO1", seq u32, t_meas_ns i64 (time.time_ns 시계), vx f32, vy f32, flags u32
  flags bit0 = valid.  ORB 를 Pi 에서 돌리면 카메라 stamp 와 Pi time.time_ns 가 같은 시계다.
  노트북에서 돌리면 두 시계를 chrony/NTP 로 맞추거나 relay 의 --clock-offset-ms 로 보정한다.
"""

from __future__ import annotations

import math
import socket
import struct
from collections import deque

import numpy as np

MAGIC = b"KVO1"
FMT = "<4sIqffI"
SIZE = struct.calcsize(FMT)
PORT = 5600
V_MAX = 1.0           # [m/s] 로버 최고 0.38 m/s. 이보다 크면 ORB 튐으로 보고 버린다


def pack(seq: int, t_meas_ns: int, vx: float, vy: float, valid: bool = True) -> bytes:
    return struct.pack(FMT, MAGIC, seq & 0xFFFFFFFF, int(t_meas_ns), float(vx), float(vy), 1 if valid else 0)


def unpack(buf: bytes):
    """-> (seq, t_meas_ns, vx, vy, valid) 또는 None (크기·magic 불일치)."""
    if len(buf) != SIZE:
        return None
    magic, seq, t, vx, vy, fl = struct.unpack(FMT, buf)
    if magic != MAGIC or not (math.isfinite(vx) and math.isfinite(vy)):
        return None
    return seq, t, vx, vy, bool(fl & 1)


def quat_to_rot(qx, qy, qz, qw) -> np.ndarray:
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) or 1.0
    x, y, z, w = qx / n, qy / n, qz / n, qw / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class PoseToVelocity:
    """ORB base 자세 열 -> 몸체 속도 (vx, vy) 와 측정 시각.  ROS 없이 시험할 수 있게 분리."""

    def __init__(self, window_s: float = 0.2, min_span_s: float = 0.08, min_n: int = 3, v_max: float = V_MAX):
        self.window_s, self.min_span_s, self.min_n, self.v_max = window_s, min_span_s, min_n, v_max
        self.map_id = None
        self.hist = deque()           # (t [s], p_w (3,), R_wb (3,3))
        self.n_reject = 0

    def reset(self):
        self.hist.clear()

    def add(self, t: float, p_w, q_wb, map_id=None):
        """새 자세 하나.  -> (t_meas [s], vx, vy) 또는 None (아직 부족·튐)."""
        if map_id != self.map_id:
            self.map_id = map_id
            self.reset()
        if self.hist and t <= self.hist[-1][0]:
            return None                       # 같은/거꾸로 된 stamp
        self.hist.append((t, np.asarray(p_w, float), quat_to_rot(*q_wb)))
        while self.hist and t - self.hist[0][0] > self.window_s:
            self.hist.popleft()
        if len(self.hist) < self.min_n or t - self.hist[0][0] < self.min_span_s:
            return None
        ts = np.array([h[0] for h in self.hist])
        ps = np.stack([h[1] for h in self.hist])
        tc = ts.mean()
        dt = ts - tc
        v_w = (dt[:, None] * (ps - ps.mean(0))).sum(0) / (dt * dt).sum()      # 최소제곱 기울기
        r_mid = self.hist[int(np.argmin(np.abs(dt)))][2]
        v_b = r_mid.T @ v_w
        if not np.all(np.isfinite(v_b)) or math.hypot(v_b[0], v_b[1]) > self.v_max:
            self.n_reject += 1
            self.reset()
            return None
        return tc, float(v_b[0]), float(v_b[1])


class VoReceiver:
    """run_policy 쪽: 논블로킹 UDP, 가장 최근 측정만 유지."""

    def __init__(self, port: int = PORT, host: str = "0.0.0.0", clock=None):
        import time
        self.clock = clock or time.time_ns
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.setblocking(False)
        self.last = None              # (t_meas_ns, vx, vy, valid)
        self.n_rx = self.n_bad = self.n_valid = 0
        self.last_seq = None

    def poll(self):
        """소켓을 비우고 -> (vx, vy, age [s]) 또는 None (받은 적 없음 / 마지막이 무효)."""
        while True:
            try:
                buf = self.sock.recv(64)
            except (BlockingIOError, InterruptedError):
                break
            m = unpack(buf)
            if m is None:
                self.n_bad += 1
                continue
            seq, t, vx, vy, valid = m
            self.n_rx += 1
            self.n_valid += valid
            if self.last is None or t >= self.last[0]:
                self.last, self.last_seq = (t, vx, vy, valid), seq
        if self.last is None or not self.last[3]:
            return None
        return self.last[1], self.last[2], max(0.0, (self.clock() - self.last[0]) * 1e-9)

    def close(self):
        self.sock.close()
