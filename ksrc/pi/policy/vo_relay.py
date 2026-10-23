"""ORB-SLAM3 base 자세 -> 몸체 속도 -> run_policy (UDP) 릴레이 (2026-10-16).

동료 ORB 노드 (`~/ksrc/ORB` = lee-robot-lab/rover-mapping, `orbslam3_stereo_inertial`) 를 **pose_output:=base**
로 띄우면 `/orbslam3/base_pose` (PoseStamped, frame_id = orb_map_<n>) 가 나온다.  이 노드는 그걸 받아
vo_link.PoseToVelocity 로 몸체 속도를 구하고 run_policy 의 --vo-port 로 보낸다.

  # Pi (ORB 와 같은 기계 권장: 카메라 stamp 와 Pi 시계가 같아 age 가 정확)
  source /opt/ros/jazzy/setup.bash
  python3 pi/policy/vo_relay.py                         # -> 127.0.0.1:5600
  python3 pi/policy/vo_relay.py --to 192.168.0.10       # 다른 기계의 run_policy 로
  python3 pi/policy/vo_relay.py --fake --rate 15        # ORB 없이: 0.15 m/s 직진 가짜 자세 (연결 시험)

추적 상태 (/orbslam3/tracking_state, ORB-SLAM3: 2 = OK) 가 OK 가 아니면 이력을 비우고 valid=0 을 보낸다.
2초마다 받은 자세 Hz, 보낸 속도, 나이, ORB 지연(받은 시각 - 이미지 stamp, 평균/최대), 버린 튐 수를 찍는다.
나이 = ORB 지연 + 창 절반(기본 100 ms) + 전송.  ORB 지연이 크면 Pi 부하, 나이만 크면 창 길이.
"""

from __future__ import annotations

import argparse
import socket
import time

from vo_link import PORT, PoseToVelocity, pack

TRACK_OK = 2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--to", default="127.0.0.1", help="run_policy 가 도는 기계")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--pose-topic", default="/orbslam3/base_pose")
    ap.add_argument("--state-topic", default="/orbslam3/tracking_state")
    ap.add_argument("--window", type=float, default=0.2, help="속도 맞춤 창 [s] (길면 잡음 적고 지연 큼)")
    ap.add_argument("--clock-offset-ms", type=float, default=0.0,
                    help="자세 stamp 를 run_policy 시계로 옮길 때 더할 값 (노트북 ORB 일 때만)")
    ap.add_argument("--fake", action="store_true", help="ORB/ROS 없이 가짜 자세 (0.15 m/s 직진)")
    ap.add_argument("--rate", type=float, default=15.0, help="--fake 자세 Hz")
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = (args.to, args.port)
    est = PoseToVelocity(window_s=args.window)
    off_ns = int(args.clock_offset_ms * 1e6)
    st = {"seq": 0, "n_pose": 0, "n_sent": 0, "last": None, "t_print": time.monotonic(), "track": TRACK_OK,
          "lat": []}         # 자세를 받은 시각 - 이미지 stamp [s] = 카메라 노출~ORB 출력 지연 (창 절반 제외)

    def send(t_meas_s, vx, vy, valid):
        sock.sendto(pack(st["seq"], int(t_meas_s * 1e9) + off_ns, vx, vy, valid), dest)
        st["seq"] += 1
        st["n_sent"] += valid

    def on_pose(t, p, q, map_id):
        st["n_pose"] += 1
        st["lat"].append(time.time() - t)
        if st["track"] != TRACK_OK:
            return
        r = est.add(t, p, q, map_id)
        if r is not None:
            send(*r, True)
            st["last"] = r

    def report():
        now = time.monotonic()
        if now - st["t_print"] < 2.0:
            return
        dt = now - st["t_print"]
        last = st["last"]
        age = (time.time_ns() * 1e-9 - last[0]) if last else float("nan")
        lat = st["lat"]
        lat_s = (f"  ORB 지연 평균 {1000 * sum(lat) / len(lat):.0f} / 최대 {1000 * max(lat):.0f} ms" if lat else "")
        print(f"[vo_relay] 자세 {st['n_pose'] / dt:.1f} Hz  보냄 {st['n_sent'] / dt:.1f} Hz  "
              f"추적 {'OK' if st['track'] == TRACK_OK else st['track']}  "
              f"v=({last[1]:+.3f}, {last[2]:+.3f}) m/s  나이 {age * 1000:.0f} ms{lat_s}  튐 버림 {est.n_reject}" if last else
              f"[vo_relay] 자세 {st['n_pose'] / dt:.1f} Hz  아직 속도 없음  추적 {st['track']}{lat_s}", flush=True)
        st["n_pose"] = st["n_sent"] = 0
        st["lat"] = []
        st["t_print"] = now

    print(f"[vo_relay] -> {dest[0]}:{dest[1]}  창 {args.window} s  {'가짜 자세' if args.fake else args.pose_topic}", flush=True)
    if args.fake:
        t0 = time.time()
        try:
            while True:
                t = time.time()
                x = 0.15 * (t - t0)
                on_pose(t, (x, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 0)
                report()
                time.sleep(1.0 / args.rate)
        except KeyboardInterrupt:
            send(time.time(), 0.0, 0.0, False)
        return

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.qos import qos_profile_sensor_data
    from std_msgs.msg import Int32

    rclpy.init()
    node = rclpy.create_node("vo_relay")

    def cb_pose(m):
        s = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        p, q = m.pose.position, m.pose.orientation
        on_pose(s, (p.x, p.y, p.z), (q.x, q.y, q.z, q.w), m.header.frame_id)

    def cb_state(m):
        if m.data != TRACK_OK and st["track"] == TRACK_OK:
            est.reset()
            send(time.time(), 0.0, 0.0, False)
        st["track"] = m.data

    node.create_subscription(PoseStamped, args.pose_topic, cb_pose, 10)
    node.create_subscription(Int32, args.state_topic, cb_state, 10)
    node.create_timer(0.5, report)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        send(time.time(), 0.0, 0.0, False)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
