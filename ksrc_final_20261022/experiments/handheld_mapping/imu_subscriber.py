# ================================================================
# imu_subscriber.py
# 설명: Pi 기존 SPI owner의 UNIX datagram만 받아 JSONL stdout으로 전달한다.
# 사용법: python3 -u -m experiments.handheld_mapping.imu_subscriber --socket /dev/shm/rover-mapping-imu.sock
# ================================================================
import argparse
import importlib.util
import json
import os
from pathlib import Path
import select
import signal
import socket
import sys
import threading


def actual_chip_to_body() -> list[list[int]]:
    """현재 Pi decoder 모듈의 축 메타데이터만 읽고 SpiLink는 생성하지 않는다."""
    source = Path('/home/pi/ksrc/pi/common/nucleo_link.py')
    spec = importlib.util.spec_from_file_location('mapping_imu_decoder_metadata', source)
    if spec is None or spec.loader is None:
        raise RuntimeError('nucleo_link metadata module unavailable')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [list(row) for row in module.IMU_CHIP_TO_BODY]


def subscribe(socket_path: str, chip_to_body: list[list[int]]) -> None:
    """독점 소켓을 소유하며 stdin·signal로 종료하고 자신이 만든 소켓만 제거한다."""
    target = Path(socket_path)
    if os.path.lexists(target):
        raise FileExistsError(target)
    receiver = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    owned = None
    stop = threading.Event()
    previous_handlers = {}

    def emit(data: dict) -> None:
        print(json.dumps(data, separators=(',', ':'), allow_nan=False), flush=True)

    def request_stop(signum: int, frame: object) -> None:
        stop.set()

    try:
        receiver.bind(str(target))
        owned = target.stat()
        receiver.settimeout(.2)
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                previous_handlers[signum] = signal.signal(signum, request_stop)
        emit({'kind': 'ready', 'chip_to_body': chip_to_body})
        malformed = 0
        while not stop.is_set():
            if select.select([sys.stdin], [], [], 0)[0]:
                # 부분 입력에서 readline으로 막히지 않아 signal 처리도 계속 가능하다.
                chunk = os.read(sys.stdin.fileno(), 4096)
                if not chunk or b'\n' in chunk:
                    break
            try:
                payload, _, flags, _ = receiver.recvmsg(65536)
            except socket.timeout:
                continue
            try:
                if flags & socket.MSG_TRUNC:
                    raise ValueError('oversized datagram')
                def reject_constant(token: str) -> None:
                    raise ValueError(f'nonfinite JSON: {token}')
                data = json.loads(payload, parse_constant=reject_constant)
                if not isinstance(data, dict):
                    raise ValueError('JSON object required')
                data['kind'] = 'telemetry'
                emit(data)
            except (ValueError, UnicodeError) as error:
                malformed += 1
                # 개수만 누적하며 메시지와 입력 payload를 메모리에 저장하지 않는다.
                emit({'kind': 'malformed', 'count': malformed, 'error': str(error)[:200]})
    finally:
        receiver.close()
        if owned is not None:
            try:
                current = target.lstat()
                if (current.st_dev, current.st_ino) == (owned.st_dev, owned.st_ino):
                    target.unlink()
            except FileNotFoundError:
                pass
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def main() -> None:
    """Pi native Python에서 구독기를 시작한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--socket', default='/dev/shm/rover-mapping-imu.sock')
    args = parser.parse_args()
    subscribe(args.socket, actual_chip_to_body())


if __name__ == '__main__':
    main()
