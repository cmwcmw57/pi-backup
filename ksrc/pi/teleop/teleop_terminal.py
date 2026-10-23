#!/usr/bin/env python3
"""
숫자 직접 입력식 텔레옵. "vx vy omega" 를 치면 그대로 Nucleo 에 보낸다
(개발 중에는 가상 Nucleo 시뮬에도 동일 프로토콜로 붙는다 -- --port 만 바꾸면 됨).

패드 UI 를 완전히 건너뛰고 저수준 명령 경로를 직접 때려보기 위한 도구다.
"기구학/프로토콜/액추에이터 매핑이 맞았나" 와 "마우스를 제대로 끌었나" 를
분리해서 확인할 때 쓴다.

다른 텔레옵과 마찬가지로 마지막 명령을 백그라운드에서 일정 주기로 계속
보낸다 (다음에 칠 값을 생각하며 가만히 있어도 Nucleo 워치독이 안 물리도록).

사용법:
  python3 teleop_terminal.py --port /dev/pts/3
  vx vy omega> 0.3 0 0        # 0.3 m/s 전진
  vx vy omega> 0 0 1.5        # 제자리 반시계 1.5 rad/s
  vx vy omega> 0 0 0          # (또는 그냥: stop)
  vx vy omega> q
"""
import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # pi/ 를 경로에 넣어 common.* 를 import
from common.nucleo_link import encode_velocity_cmd


def sender_loop(ser, get_cmd, period_s, stop_event):
    next_send = time.monotonic()
    while not stop_event.is_set():
        now = time.monotonic()
        if now >= next_send:
            vx, vy, omega = get_cmd()
            ser.write(encode_velocity_cmd(vx, vy, omega))
            next_send = now + period_s
        time.sleep(0.005)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True, help="serial device (real Nucleo or sim pty)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--rate", type=float, default=50.0, help="background command send rate, Hz")
    args = ap.parse_args()

    import serial
    ser = serial.Serial(args.port, args.baud, timeout=0)

    state = {"cmd": (0.0, 0.0, 0.0)}
    lock = threading.Lock()

    def get_cmd():
        with lock:
            return state["cmd"]

    stop_event = threading.Event()
    thread = threading.Thread(target=sender_loop, args=(ser, get_cmd, 1.0 / args.rate, stop_event), daemon=True)
    thread.start()

    print("vx vy omega, space-separated (m/s, m/s, rad/s). 'stop' = 0 0 0. 'q' to quit.")
    try:
        while True:
            try:
                line = input("vx vy omega> ").strip()
            except EOFError:
                break
            if not line:
                continue
            if line.lower() in ("q", "quit", "exit"):
                break
            if line.lower() == "stop":
                with lock:
                    state["cmd"] = (0.0, 0.0, 0.0)
                print("  -> vx=0.00 vy=0.00 omega=0.00")
                continue
            parts = line.split()
            if len(parts) != 3:
                print(f"  need exactly 3 numbers (vx vy omega), got {len(parts)}")
                continue
            try:
                vx, vy, omega = (float(p) for p in parts)
            except ValueError:
                print(f"  couldn't parse '{line}' as three floats")
                continue
            with lock:
                state["cmd"] = (vx, vy, omega)
            print(f"  -> vx={vx:+.3f} vy={vy:+.3f} omega={omega:+.3f}  (sending @ {args.rate:.0f}Hz until changed)")
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        thread.join(timeout=1.0)
        zero = encode_velocity_cmd(0.0, 0.0, 0.0)
        for _ in range(5):
            ser.write(zero)
            time.sleep(0.02)
        ser.close()
        print("\nsent zero + closed port")


if __name__ == "__main__":
    main()
