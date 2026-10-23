# ================================================================
# ros_inputs.py
# 설명: Pi ROS 2의 ORB base pose·추적 상태·좌 IR 영상을 실행기로 전달한다.
#       camera stamp를 같은 Pi의 monotonic clock으로 옮긴다. SPI를 열지 않는다.
# 사용법: python ros_inputs.py --config local/real.json
# ================================================================
from __future__ import annotations
import argparse
import base64
import json
from pathlib import Path
import socket
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--port', type=int, default=5012)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import Int32
    from sensor_msgs.msg import Image
    from rclpy.qos import qos_profile_sensor_data
    rclpy.init()
    node = rclpy.create_node('ksrc_navigation_inputs')
    output = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    output.setblocking(False)
    state = {'tracking': 0, 'tracking_at': -1e9, 'image_at': -1e9}
    c1_base = np.asarray(config['calibration']['ROS.T_c1_base'])

    def send(event: dict) -> None:
        try:
            payload = json.dumps(event, allow_nan=False).encode()
            if len(payload) <= 65000:
                output.sendto(payload, ('127.0.0.1', args.port))
        except (OSError, ValueError):
            pass

    def mono_stamp(header) -> float:
        # 각 수신 시점의 offset 사용: Pi realtime clock 보정이 누적 offset 오류를 만들지 않는다.
        capture = header.stamp.sec + header.stamp.nanosec * 1e-9
        return capture - time.time() + time.monotonic()

    def tracking(msg) -> None:
        now = time.monotonic()
        state.update(tracking=int(msg.data), tracking_at=now)
        send(dict(kind='tracking', monotonic_s=now, state=int(msg.data)))

    def pose(msg) -> None:
        p, q = msg.pose.position, msg.pose.orientation
        try:
            base = np.eye(4)
            base[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            base[:3, 3] = [p.x, p.y, p.z]
        except ValueError:
            return
        now = time.monotonic()
        track = state['tracking'] if now - state['tracking_at'] < 1. else 0
        send(dict(kind='frame', monotonic_s=mono_stamp(msg.header),
                  camera_pose=(base @ np.linalg.inv(c1_base)).tolist(),
                  map_id=msg.header.frame_id, state=track))

    def image(msg) -> None:
        now = time.monotonic()
        if now - state['image_at'] < .18 or msg.encoding not in ('mono8', '8UC1'):
            return
        state['image_at'] = now
        if msg.height != 480 or msg.width != 640 or msg.step < msg.width:
            return
        gray = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.step)[:, :msg.width]
        ok, encoded = cv2.imencode('.jpg', gray, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            send(dict(kind='image', monotonic_s=mono_stamp(msg.header), jpg=base64.b64encode(encoded).decode()))

    node.create_subscription(Int32, '/orbslam3/tracking_state', tracking, 10)
    node.create_subscription(PoseStamped, '/orbslam3/base_pose', pose, qos_profile_sensor_data)
    node.create_subscription(Image, '/camera/camera/infra1/image_rect_raw', image, qos_profile_sensor_data)
    print('ROS inputs ready: base_pose + tracking_state + IR1; depth OFF', flush=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        send(dict(kind='tracking', monotonic_s=time.monotonic(), state=0))
        output.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
