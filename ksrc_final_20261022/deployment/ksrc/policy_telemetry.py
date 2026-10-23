# ================================================================
# policy_telemetry.py
# 설명: 기존 run_policy를 동일 프로세스에서 실행하며 이미 읽은 IMU telemetry만 UDP로 분기한다.
#       SPI 소유자는 원래 정책 하나뿐이다. 기본은 --fake이며 --real을 명시해야 실기를 연다.
# 사용법: python policy_telemetry.py --policy-script /home/pi/ksrc/pi/policy/run_policy.py -- --policy PREFIX --seconds 10
# ================================================================
from __future__ import annotations
import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
import socket
import signal
import sys
import time

import numpy as np


class Telemetry:
    """MCU 고유 시각·보정·축 변환을 적용하고 기존 제어로 즉시 복귀한다."""

    def __init__(self, config: dict, port: int) -> None:
        self.config, self.port = config, port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setblocking(False)
        self.last_mcu = self.last_receipt = self.stamp = None
        self.dropped = 0
        cal = config['calibration']
        base_imu = np.linalg.inv(np.asarray(cal['ROS.T_c1_base'])) @ np.linalg.inv(np.asarray(cal['IMU.T_b_c1']))
        self.imu_from_base_rotation = base_imu[:3, :3].T

    def emit(self, tlm: dict | None) -> None:
        """중복 tick은 버리고 MCU reset은 새로운 구간으로 처리한다."""
        if not tlm or not int(tlm.get('flags', 0)) & (1 << 4):
            return
        mcu, now = int(tlm['t_ms']), time.monotonic()
        if self.last_mcu == mcu:
            return
        delta = None if self.last_mcu is None else (mcu - self.last_mcu) % 2**32 / 1000.
        if delta is None or delta > .1 or now - self.last_receipt > .2:
            dt = .02
            self.stamp = now - .02
        else:
            dt = delta
            # MCU 간격 보존. 호스트 clock보다 앞서지 않게 제한하고 장시간 clock drift를 억제한다.
            self.stamp = min(self.stamp + dt, now)
            if now - self.stamp > .1:
                self.stamp = now - .02
        self.last_mcu, self.last_receipt = mcu, now
        imu = self.config['imu']
        gyro = np.radians(np.asarray(tlm['gyro_dps']) - imu['gyro_bias_dps'])
        force = (np.asarray(tlm['acc_g']) - imu['acc_offset_g']) * imu['acc_scale'] * 9.81
        data = dict(kind='imu', monotonic_s=self.stamp, dt_s=dt,
                    gyro=(self.imu_from_base_rotation @ gyro).tolist(),
                    force=(self.imu_from_base_rotation @ force).tolist(),
                    wheel_speed=float(np.max(np.abs(tlm.get('wheel_mps', [0.])))), mcu_t_ms=mcu)
        try:
            self.socket.sendto(json.dumps(data, allow_nan=False).encode(), ('127.0.0.1', self.port))
        except (OSError, ValueError):
            self.dropped += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy-script', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--sensor-port', type=int, default=5012)
    parser.add_argument('--real', action='store_true')
    parser.add_argument('policy_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    sys.path.insert(0, str(args.policy_script.resolve().parent))
    spec = importlib.util.spec_from_file_location('ksrc_navigation_policy', args.policy_script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = json.loads(args.config.read_text())
    spi_lock = None
    if args.real:
        spi_lock = Path('/dev/shm/ksrc-spi-owner.lock').open('a')
        fcntl.flock(spi_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if [list(v) for v in module.nl.IMU_CHIP_TO_BODY] != config['imu']['chip_to_body']:
            raise ValueError('deployed decoder axis mapping differs from calibration')
        for fd_dir in Path('/proc').glob('[0-9]*/fd'):
            try:
                for fd in fd_dir.iterdir():
                    try:
                        if str(fd.resolve()).startswith('/dev/spidev'):
                            raise RuntimeError('existing SPI owner: ' + str(fd_dir.parent))
                    except (FileNotFoundError, PermissionError):
                        pass
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                pass
    tee = Telemetry(config, args.sensor_port)

    def wrap(original):
        def exchange(self, *values, **keywords):
            result = original(self, *values, **keywords)
            tee.emit(result)
            return result
        return exchange

    owner = module.nl.SpiLink if args.real else module.FakeLink
    for name in ('exchange', 'exchange_res'):
        setattr(owner, name, wrap(getattr(owner, name)))
    forwarded = args.policy_args[1:] if args.policy_args[:1] == ['--'] else args.policy_args
    if not args.real and '--fake' not in forwarded:
        forwarded = ['--fake', *forwarded]
    sys.argv = [str(args.policy_script), *forwarded]
    def terminate(signum: int, frame: object) -> None:
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    try:
        module.main()
    finally:
        tee.socket.close()
        if spi_lock:
            spi_lock.close()


if __name__ == '__main__':
    main()
