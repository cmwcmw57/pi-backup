# ================================================================
# control_gateway.py
# 설명: Pi에서 조이스틱(텔레옵)과 MPPI(자동)의 속도 명령 중 현재 모드의 것만 골라 run_policy(UDP 5005)로 전달하는 중계.
#       모드 전환·시작 조건·비상 정지를 처리하고 보내는 순서번호를 하나로 이어 붙인다. 모터·SPI는 열지 않는다.
# 사용법:
#   python -m experiments.handheld_mapping.control_gateway --policy 127.0.0.1:5005 --runner 127.0.0.1:5010
#   또는: from experiments.handheld_mapping.control_gateway import Gateway
# ================================================================
from __future__ import annotations

import argparse
import json
import math
import select
import socket
import struct
import threading
import time
from typing import Callable

CMD_FORMAT = '<4sIfff'      # run_policy 수신 형식과 같다: b'KSRC', 순서번호, vx, vy, ω
CTL_FORMAT = '<4sIB3f'      # 제어: b'KCTL', 순서번호, 동작, 값 3개
HEARTBEAT, POSE, START, ESTOP = 1, 2, 3, 4
RESTART_GAP = 1000          # 순서번호가 이만큼 이상 뒤로 가면 보내는 쪽이 다시 시작한 것으로 본다
ZERO = (0., 0., 0.)


def pack_cmd(seq: int, vx: float, vy: float, w: float) -> bytes:
    return struct.pack(CMD_FORMAT, b'KSRC', seq & 0xFFFFFFFF, vx, vy, w)


def parse_cmd(data: bytes) -> tuple[int, tuple[float, float, float]] | None:
    """속도 명령 패킷을 (순서번호, (vx, vy, ω))로 읽는다. 형식·유한값이 틀리면 None."""
    if len(data) != struct.calcsize(CMD_FORMAT):
        return None
    magic, seq, vx, vy, w = struct.unpack(CMD_FORMAT, data)
    if magic != b'KSRC' or not all(math.isfinite(v) for v in (vx, vy, w)):
        return None
    return seq, (vx, vy, w)


def pack_ctl(seq: int, op: int, a: float = 0., b: float = 0., c: float = 0.) -> bytes:
    return struct.pack(CTL_FORMAT, b'KCTL', seq & 0xFFFFFFFF, op, a, b, c)


def parse_ctl(data: bytes) -> tuple[int, int, tuple[float, float, float]] | None:
    """제어 패킷을 (순서번호, 동작, 값 3개)로 읽는다. 형식·유한값이 틀리면 None."""
    if len(data) != struct.calcsize(CTL_FORMAT):
        return None
    magic, seq, op, a, b, c = struct.unpack(CTL_FORMAT, data)
    if magic != b'KCTL' or not all(math.isfinite(v) for v in (a, b, c)):
        return None
    return seq, op, (a, b, c)


class Gateway:
    """모드(TELEOP/AUTO)에 따라 명령 출처를 고르고 run_policy로 보낼 명령을 정한다. 소켓과 시계는 모른다."""

    def __init__(self, send: Callable[[float, float, float], None], on_start: Callable[[int], None] | None = None,
                 on_pose: Callable[..., None] | None = None, on_stop: Callable[[], None] | None = None,
                 stale_s: float = .3, keepalive_s: float = .02, estop_hold_s: float = .5,
                 ready_stationary_s: float = 3., status_max_age_s: float = 1., link_timeout_s: float = 1.) -> None:
        self.send, self.on_start, self.on_pose, self.on_stop = send, on_start, on_pose, on_stop
        self.stale_s, self.keepalive_s, self.estop_hold_s = stale_s, keepalive_s, estop_hold_s
        self.ready_stationary_s, self.status_max_age_s, self.link_timeout_s = ready_stationary_s, status_max_age_s, link_timeout_s
        self.mode, self.run_id = 'TELEOP', 0
        self.latest = {'TELEOP': (ZERO, -math.inf), 'AUTO': (ZERO, -math.inf)}   # 출처별 마지막 명령과 받은 시각
        self.sent_at, self.zero_until = -math.inf, -math.inf
        self.status, self.status_at, self.heartbeat_at, self.last_reject = {}, -math.inf, -math.inf, ''

    def _emit(self, command: tuple[float, float, float], now: float) -> None:
        self.send(*command)
        self.sent_at = now

    def teleop(self, command: tuple[float, float, float], now: float) -> None:
        """조이스틱 명령. TELEOP이고 비상 정지 유지 시간이 아니면 받는 즉시 전달한다."""
        self.latest['TELEOP'] = (command, now)
        if self.mode == 'TELEOP' and now >= self.zero_until:
            self._emit(command, now)

    def auto(self, command: tuple[float, float, float], now: float) -> None:
        """MPPI 명령. AUTO일 때만 받는 즉시 전달한다."""
        self.latest['AUTO'] = (command, now)
        if self.mode == 'AUTO':
            self._emit(command, now)

    def runner_status(self, status: dict, now: float) -> None:
        """MPPI 실행기가 알려 주는 준비 상태. 임무가 끝났다고 하면 TELEOP으로 돌아가 정지한다."""
        self.status, self.status_at = status, now
        if status.get('done') and self.mode == 'AUTO':
            self.mode = 'TELEOP'
            self._emit(ZERO, now)

    def missing(self, now: float) -> list[str]:
        """START에 모자란 조건 이름 목록(비어 있으면 시작 가능). 시작 pose는 임무의 값을 그대로 쓰므로 조건이 아니다."""
        if now-self.status_at > self.status_max_age_s:
            return ['status']
        status = self.status
        checks = {'stationary_s': status.get('stationary_s', 0.) >= self.ready_stationary_s, 'orb_ok': bool(status.get('orb_ok')),
                  'in_region1': bool(status.get('in_region1'))}
        return [name for name, ok in checks.items() if not ok]

    def control(self, op: int, args: tuple[float, float, float], now: float) -> bool:
        """제어 메시지를 처리하고 받아들였는지 돌려준다."""
        if op == HEARTBEAT:
            self.heartbeat_at = now
            return True
        if op == POSE:
            if self.on_pose:
                self.on_pose(*args)
            return True
        if op == ESTOP:
            self.mode, self.zero_until = 'TELEOP', now+self.estop_hold_s
            self._emit(ZERO, now)
            if self.on_stop:
                self.on_stop()
            return True
        if op == START:
            if self.mode != 'TELEOP' or now < self.zero_until:
                self.last_reject = 'mode'
                return False
            missing = self.missing(now)
            if missing:
                self.last_reject = ','.join(missing)
                return False
            self.mode, self.run_id, self.last_reject = 'AUTO', int(args[0]), ''
            if self.on_start:
                self.on_start(self.run_id)
            return True
        return False

    def tick(self, now: float) -> None:
        """주기 호출. 비상 정지 유지 중이거나 출처가 조용하면 0을, 아니면 마지막 명령을 20 ms마다 다시 보낸다."""
        if now-self.sent_at < self.keepalive_s:
            return
        command, at = self.latest[self.mode]
        if now < self.zero_until or now-at >= self.stale_s:
            self._emit(ZERO, now)
        else:
            self._emit(command, now)

    def snapshot(self, now: float) -> dict:
        """조이스틱 화면에 보여 줄 상태."""
        return dict(mode=self.mode, run_id=self.run_id, missing=self.missing(now), last_reject=self.last_reject,
                    link_ok=now-self.heartbeat_at <= self.link_timeout_s,
                    teleop_age_s=now-self.latest['TELEOP'][1], auto_age_s=now-self.latest['AUTO'][1])


def open_sockets(host: str, ports: tuple[int, int, int]) -> tuple[socket.socket, socket.socket, socket.socket]:
    """(조이스틱 명령·제어, MPPI 명령, 실행기 상태) 수신 소켓 세 개를 연다. 포트 0이면 빈 포트를 고른다."""
    sockets = []
    for port in ports:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((host, port))
        sockets.append(sock)
    return tuple(sockets)  # type: ignore[return-value]


def newer(seq: int, last: int) -> bool:
    """순서번호가 마지막보다 새것이거나 보내는 쪽이 다시 시작한 것이면 True."""
    return seq > last or last-seq > RESTART_GAP


def serve(sockets, policy_addr: tuple[str, int], stop: threading.Event, runner_addr: tuple[str, int] | None = None,
          status_hz: float = 5.) -> None:
    """소켓을 읽어 Gateway에 넣고, 나가는 명령에 순서번호를 이어 붙여 policy_addr로 보낸다. stop이 켜질 때까지 돈다."""
    teleop_sock, auto_sock, status_sock = sockets
    out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    counter = [0]

    def send(vx: float, vy: float, w: float) -> None:
        counter[0] += 1
        out.sendto(pack_cmd(counter[0], vx, vy, w), policy_addr)

    def to_runner(message: dict) -> None:
        if runner_addr:
            out.sendto(json.dumps(message).encode(), runner_addr)

    gateway = Gateway(send, on_start=lambda run: to_runner(dict(op='START', run=run)),
                      on_pose=lambda x, y, yaw: to_runner(dict(op='POSE', x=x, y=y, yaw=yaw)),
                      on_stop=lambda: to_runner(dict(op='STOP')))
    last = dict(teleop=-1, auto=-1, ctl=-1)
    gui, next_status = None, 0.
    while not stop.is_set():
        ready, _, _ = select.select([teleop_sock, auto_sock, status_sock], [], [], .005)
        for sock in ready:
            data, address = sock.recvfrom(256)
            now = time.monotonic()
            if sock is status_sock:
                try:
                    gateway.runner_status(json.loads(data), now)
                except ValueError:
                    pass
            elif sock is auto_sock:
                parsed = parse_cmd(data)
                if parsed and newer(parsed[0], last['auto']):
                    last['auto'] = parsed[0]
                    gateway.auto(parsed[1], now)
            else:
                gui = address
                parsed = parse_cmd(data)
                if parsed and newer(parsed[0], last['teleop']):
                    last['teleop'] = parsed[0]
                    gateway.teleop(parsed[1], now)
                control = parse_ctl(data)
                if control and newer(control[0], last['ctl']):
                    last['ctl'] = control[0]
                    gateway.control(control[1], control[2], now)
        now = time.monotonic()
        gateway.tick(now)
        if gui and now >= next_status:
            next_status = now+1./status_hz
            out.sendto(json.dumps(gateway.snapshot(now)).encode(), gui)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--ports', type=int, nargs=3, default=(5006, 5007, 5011), metavar=('TELEOP', 'AUTO', 'STATUS'))
    parser.add_argument('--policy', default='127.0.0.1:5005', help='run_policy 수신 주소 host:port')
    parser.add_argument('--runner', default=None, help='MPPI 실행기 제어 주소 host:port')
    args = parser.parse_args()
    addr = lambda text: (text.rsplit(':', 1)[0], int(text.rsplit(':', 1)[1]))
    serve(open_sockets(args.host, tuple(args.ports)), addr(args.policy), threading.Event(),
          addr(args.runner) if args.runner else None)


if __name__ == '__main__':
    main()
