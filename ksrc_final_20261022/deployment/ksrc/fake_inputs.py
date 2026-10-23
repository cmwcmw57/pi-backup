# ================================================================
# fake_inputs.py
# 설명: SPI·ROS 없이 명령을 따르는 점 로버의 합성 pose·IMU를 실행기에 전달한다.
#       실제 물리 검증이 아닌 CLI·UDP·상태 전환 연결 시험 전용이다.
# 사용법: python fake_inputs.py --state-dir state
# ================================================================
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import socket
import time

import numpy as np
from scipy.spatial.transform import Rotation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--state-dir', type=Path, required=True)
    args = parser.parse_args()
    output = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    xy, yaw, i, command = np.array([3.6, 2.82]), -math.pi / 2, 0, [0., 0., 0.]
    next_tick = time.monotonic()
    while True:
        now = time.monotonic()
        try:
            status = json.loads((args.state_dir / 'status.json').read_text())
            command = status['command']
        except (OSError, ValueError):
            pass
        vx, vy, rate = command
        yaw += rate * .02
        xy += np.array([vx * math.cos(yaw) - vy * math.sin(yaw), vx * math.sin(yaw) + vy * math.cos(yaw)]) * .02
        event = dict(kind='imu', monotonic_s=now, dt_s=.02, gyro=[0., 0., rate],
                     force=[0., 0., 9.81], wheel_speed=math.hypot(vx, vy))
        output.sendto(json.dumps(event).encode(), ('127.0.0.1', 5012))
        if i % 3 == 0:
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_euler('z', yaw).as_matrix()
            pose[:3, 3] = [*xy, .455]
            output.sendto(json.dumps(dict(kind='frame', monotonic_s=now, camera_pose=pose.tolist(),
                                         map_id='FAKE_0', state=2)).encode(), ('127.0.0.1', 5012))
        i += 1
        next_tick += .02
        if next_tick < now - .02:
            next_tick = now + .02
        time.sleep(max(0., next_tick - time.monotonic()))


if __name__ == '__main__':
    main()
