# ================================================================
# test_command_sender.py
# 설명: CommandSender/CommandReceiver의 UDP 왕복, 최신값 유지, 0.3 s 타임아웃 정지, 클램프, 전송 주기 제한을 검증한다.
# 사용법: python3 -m pytest -p no:cacheprovider tests/yeouido_real/test_command_sender.py
# ================================================================
import socket
import struct
import time

import pytest

from experiments.yeouido_real.command_sender import PACKET, CommandReceiver, CommandSender, clamp_cmd


@pytest.fixture
def link():
    rx = CommandReceiver(0)
    tx = CommandSender("127.0.0.1", rx.port)
    yield tx, rx
    tx.close()
    rx.close()


def wait_latest(rx: CommandReceiver, want: tuple, timeout: float = 1.0) -> tuple:
    """loopback 지연을 감안해 want가 보일 때까지 latest()를 반복 호출한다."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        got = rx.latest()
        if got == pytest.approx(want):
            return got
        time.sleep(0.005)
    return rx.latest()


def test_roundtrip_and_latest_wins(link):
    tx, rx = link
    tx.min_dt = 0.0
    for v in (0.05, 0.08, 0.10):
        tx.send(v, -0.02, 0.3)
    assert wait_latest(rx, (0.10, -0.02, 0.3)) == pytest.approx((0.10, -0.02, 0.3))
    assert rx.seq == 3


def test_timeout_returns_zero_then_recovers(link):
    tx, rx = link
    tx.send(0.1, 0.0, 0.0)
    assert wait_latest(rx, (0.1, 0.0, 0.0)) == pytest.approx((0.1, 0.0, 0.0))
    time.sleep(0.35)
    assert rx.latest() == (0.0, 0.0, 0.0)
    tx.send(0.1, 0.0, 0.0)
    assert wait_latest(rx, (0.1, 0.0, 0.0)) == pytest.approx((0.1, 0.0, 0.0))


def test_no_packet_yet_is_zero(link):
    assert link[1].latest() == (0.0, 0.0, 0.0)


def test_clamp_limits_and_nonfinite():
    assert clamp_cmd(1.0, -1.0, 5.0) == (0.20, -0.20, 0.8)
    assert clamp_cmd(-0.5, 0.1, -5.0) == (-0.20, 0.1, -0.8)
    assert clamp_cmd(float("nan"), float("inf"), 0.0) == (0.0, 0.0, 0.0)


def test_clamp_applied_on_wire(link):
    tx, rx = link
    tx.send(9.0, -9.0, 9.0)
    assert wait_latest(rx, (0.2, -0.2, 0.8)) == pytest.approx((0.2, -0.2, 0.8))


def test_send_rate_limited_to_50hz(link):
    tx, _ = link
    t0 = time.monotonic()
    sent = sum(tx.send(0.0, 0.0, 0.0) for _ in range(100000) if time.monotonic() - t0 < 0.5)
    assert sent <= 0.5 * 50 * 1.12 + 2  # 0.9/50 간격 한도


def test_wire_format_matches_pi_ksrc_packet(link):
    tx, rx = link
    raw = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    raw.sendto(b"XXXX" + bytes(16), ("127.0.0.1", rx.port))  # 매직 불일치는 버린다
    raw.close()
    tx.send(0.1, 0.0, 0.0)
    assert wait_latest(rx, (0.1, 0.0, 0.0)) == pytest.approx((0.1, 0.0, 0.0))
    assert PACKET.size == 20 and struct.unpack("<4sIfff", PACKET.pack(b"KSRC", 7, 0.1, 0.0, 0.0))[:2] == (b"KSRC", 7)
