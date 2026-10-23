# ================================================================
# command_sender.py
# 설명: RouteRunner의 body twist (vx, vy, w)를 UDP로 보내는 송신기와, 최신 값 하나만 읽는 수신 헬퍼. 패킷은 Pi run_policy --cmd-udp / udp_spi_bridge와 같은 KSRC 형식(b"KSRC", seq, vx, vy, w). 최대값 클램프, 0.3 s 끊기면 0 반환. 모터·SPI는 건드리지 않는다.
# 사용법:
#   송신(노트북/경로 쪽): CommandSender("raspberrypi.local", 5005).send(vx, vy, w)
#   수신(run_policy 없이 쓸 때): rx = CommandReceiver(5005); vx, vy, w = rx.latest()
# ================================================================
import math
import socket
import struct
import time

MAGIC = b"KSRC"
PACKET = struct.Struct("<4sIfff")  # run_policy.CmdReceiver / udp_spi_bridge와 같은 20 바이트
MAX_V, MAX_W, TIMEOUT_S = 0.20, 0.8, 0.3


def clamp_cmd(vx: float, vy: float, w: float) -> tuple[float, float, float]:
    """|vx|,|vy| <= 0.20 m/s, |w| <= 0.8 rad/s로 자른다. 유한하지 않은 값은 0."""
    def c(v: float, lim: float) -> float:
        return max(-lim, min(lim, v)) if math.isfinite(v) else 0.0
    return c(vx, MAX_V), c(vy, MAX_V), c(w, MAX_W)


class CommandSender:
    """클램프한 명령을 UDP로 보낸다. max_hz보다 빠른 호출은 버려 최신값만 주기에 맞춰 내보낸다."""

    def __init__(self, host: str, port: int, max_hz: float = 50.0) -> None:
        self.addr, self.min_dt, self.seq, self._last = (host, port), 0.9 / max_hz, 0, -math.inf
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, vx: float, vy: float, w: float) -> bool:
        """보냈으면 True, 주기 제한으로 건너뛰었으면 False."""
        now = time.monotonic()
        if now - self._last < self.min_dt:
            return False
        self._last = now
        self.seq = (self.seq + 1) & 0xFFFFFFFF
        self.sock.sendto(PACKET.pack(MAGIC, self.seq, *clamp_cmd(vx, vy, w)), self.addr)
        return True

    def close(self) -> None:
        """소켓을 닫는다."""
        self.sock.close()


class CommandReceiver:
    """UDP 소켓을 비워 가장 최신 명령 하나만 유지하고, 0.3 s 넘게 끊기면 0을 돌려준다."""

    def __init__(self, port: int, host: str = "0.0.0.0", timeout_s: float = TIMEOUT_S) -> None:
        self.timeout_s, self.cmd, self.seq, self._rx = timeout_s, (0.0, 0.0, 0.0), None, -math.inf
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.setblocking(False)
        self.port = self.sock.getsockname()[1]

    def latest(self) -> tuple[float, float, float]:
        """최신 (vx, vy, w). 마지막 수신이 timeout_s보다 오래됐으면 (0, 0, 0)."""
        while True:
            try:
                data, _ = self.sock.recvfrom(256)
            except BlockingIOError:
                break
            if len(data) != PACKET.size:
                continue
            magic, seq, vx, vy, w = PACKET.unpack(data)
            if magic != MAGIC:
                continue
            if self.seq is None or (seq - self.seq) & 0xFFFFFFFF < 0x80000000:  # 늦게 온 옛 패킷 무시
                self.cmd, self.seq, self._rx = clamp_cmd(vx, vy, w), seq, time.monotonic()
        return self.cmd if time.monotonic() - self._rx < self.timeout_s else (0.0, 0.0, 0.0)

    def close(self) -> None:
        """소켓을 닫는다."""
        self.sock.close()
