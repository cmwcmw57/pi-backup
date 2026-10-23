# ================================================================
# udp_tcp_bridge.py
# 설명: UDP 데이터그램에 2바이트 길이를 붙여 TCP로 실어 나른다. SSH 터널로 WebRTC 영상(UDP)을
#       통과시키기 위한 양끝 브리지이며, 10초마다 방향별 패킷·바이트 수를 출력한다.
# 사용법:
#   서버: python3 udp_tcp_bridge.py server 127.0.0.1:47999 127.0.0.1:47998
#   노트북: python3 udp_tcp_bridge.py client 127.0.0.1:47998 127.0.0.1:47999
# ================================================================
import socket
import struct
import sys
import threading
import time
from typing import Callable

STATS = {"tcp_to_udp": [0, 0], "udp_to_tcp": [0, 0], "dropped_no_peer": [0, 0]}


def _addr(text: str) -> tuple[str, int]:
    host, port = text.rsplit(":", 1)
    return host, int(port)


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("tcp closed")
        buf += chunk
    return buf


def _tcp_to_udp(conn: socket.socket, udp: socket.socket, dest: Callable[[], tuple | None]) -> None:
    while True:
        data = _recv_exact(conn, struct.unpack("!H", _recv_exact(conn, 2))[0])
        target = dest()
        key = "tcp_to_udp" if target else "dropped_no_peer"
        if target:
            udp.sendto(data, target)
        STATS[key][0] += 1
        STATS[key][1] += len(data)


def _udp_to_tcp(udp: socket.socket, conn: socket.socket, on_peer: Callable[[tuple], None] | None = None) -> None:
    while True:
        data, peer = udp.recvfrom(65535)
        if on_peer:
            on_peer(peer)
        conn.sendall(struct.pack("!H", len(data)) + data)
        STATS["udp_to_tcp"][0] += 1
        STATS["udp_to_tcp"][1] += len(data)


def _report() -> None:
    while True:
        time.sleep(10)
        print("stats", STATS, flush=True)


def server(listen: str, udp_target: str) -> None:
    """TCP 연결마다 대상 UDP 포트로 중계한다(서버 쪽)."""
    srv = socket.create_server(_addr(listen))
    target = _addr(udp_target)
    while True:
        conn, _ = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        threading.Thread(target=_tcp_to_udp, args=(conn, udp, lambda: target), daemon=True).start()
        threading.Thread(target=_udp_to_tcp, args=(udp, conn), daemon=True).start()
        print("server: tunnel connected", flush=True)


def client(udp_listen: str, tcp_target: str) -> None:
    """로컬 UDP 포트의 패킷을 TCP 터널로 보내고, 응답은 마지막 송신자에게 돌려준다(노트북 쪽)."""
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind(_addr(udp_listen))
    conn = socket.create_connection(_addr(tcp_target))
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    peer: dict[str, tuple] = {}
    threading.Thread(target=_tcp_to_udp, args=(conn, udp, lambda: peer.get("addr")), daemon=True).start()
    print("client: bridge up", flush=True)
    _udp_to_tcp(udp, conn, on_peer=lambda p: peer.__setitem__("addr", p))


if __name__ == "__main__":
    threading.Thread(target=_report, daemon=True).start()
    {"server": server, "client": client}[sys.argv[1]](sys.argv[2], sys.argv[3])
