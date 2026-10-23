# ================================================================
# demo.py
# 설명: 카메라 없이 실제 전송·종료·TSDF 경로를 시험할 합성 평면/단차를 보낸다.
# 사용법: .venv-mapping/bin/python -m experiments.handheld_mapping.demo
# ================================================================
import hashlib
import json
import signal
import sys
import threading
import zlib

import numpy as np

from experiments.handheld_mapping.protocol import write_packet


def main() -> None:
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    threading.Thread(target=lambda: (sys.stdin.readline(), stop.set()), daemon=True).start()
    width, height, focal = 160, 120, 140.
    k = [focal, 0, 79.5, 0, focal, 59.5, 0, 0, 1]
    info = {"width": width, "height": height, "k": k, "d": [0.]*5,
            "depth_scale_m": .001, "T_left_depth": np.eye(4).tolist(),
            "depth_frame": "demo_depth_optical", "left_frame": "demo_left_optical"}
    stereo = {"left": {"width": width, "height": height, "k": k, "d": [0.]*5},
              "right": {"width": width, "height": height, "k": k, "d": [0.]*5},
              "T_c1_c2": [[1,0,0,.05], [0,1,0,0], [0,0,1,0], [0,0,0,1]],
              "camera_profile": "160x120x15:Y8", "pose_output": "camera"}
    output = sys.stdout.buffer
    write_packet(output, {"kind": "calibration", "source": "synthetic_demo", "depth": info, "stereo": stereo})
    write_packet(output, {"kind": "ready", "source": "synthetic_demo", "message": "합성 입력 촬영 중 — 카메라/SLAM 성능 시험 아님"})
    rows = ["map_id,keyframe_id,source_stamp,x,y,z,qx,qy,qz,qw\n"]
    yy, xx = np.indices((height, width))
    count = 0
    while not stop.is_set() and count < 900:
        stamp = 10_000_000_000 + count * 66_666_667
        x_camera = .005 * count
        depth = np.where((xx-79.5)/focal*.8+x_camera < 0, 800, 650).astype(np.uint16)
        left = ((xx//8 + yy//8 + count//3) % 2 * 120 + 60).astype(np.uint8)
        right = np.roll(left, -4, axis=1)
        for name, image, frame in (("left", left, "demo_left_optical"),
                                   ("right", right, "demo_right_optical"),
                                   ("depth", depth, "demo_depth_optical")):
            write_packet(output, {"kind": "image", "stream": name, "stamp_ns": stamp,
                                 "width": width, "height": height, "step": image.strides[0],
                                 "encoding": "16UC1" if name == "depth" else "mono8",
                                 "frame_id": frame, "compression": "zlib"}, zlib.compress(image.tobytes(), 1))
        rows.append(f"0,{count},{stamp/1e9:.9f},{x_camera},0,0,0,0,0,1\n")
        write_packet(output, {"kind":"pose", "stamp_ns":stamp, "map_id":0,
                              "position":[x_camera,0,0], "quaternion_xyzw":[0,0,0,1],
                              "source":"synthetic_known_pose"})
        write_packet(output, {"kind": "status", "tracking_state": 2, "source_frames": count+1, "dropped": 0})
        count += 1
        stop.wait(1/15)
    for name, payload in (("map_0_keyframes.csv", "".join(rows).encode()),
                          ("tracking.json", json.dumps({"status": "saved" if count else "incomplete", "source": "synthetic_demo"}).encode())):
        write_packet(output, {"kind": "artifact", "name": name, "sha256": hashlib.sha256(payload).hexdigest()}, payload)
    write_packet(output, {"kind": "finished", "status": "saved" if count else "incomplete",
                          "source": "synthetic_demo", "dropped": 0, "frames": count})


if __name__ == "__main__":
    main()
