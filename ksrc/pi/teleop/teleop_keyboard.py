#!/usr/bin/env python3
"""
키보드 텔레오퍼레이션. 터미널에서 WASD/QE 를 읽어 차체 속도 명령
(vx, vy, omega) 을 UART 로 Nucleo 에 보낸다.

조작 (ROS 의 teleop_twist_keyboard 처럼 **바꾸기 전까지 유지**된다.
"누르고 있는 동안만 이동" 이 아니다):
  w/s       : 전진 속도 vx 증감
  a/d       : 좌우 게걸음 vy 증감  [스워브는 홀로노믹이라 차체를 안 돌려도 됨]
  q/e       : 반시계/시계 회전 omega 증감
  x / space : 전부 0
  +/-       : 한 번에 바뀌는 속도 폭 조절
  ESC/Ctrl-C: 종료 (0 프레임을 몇 번 보낸 뒤 포트를 닫는다)

Usage:
  python3 teleop_keyboard.py --port /dev/ttyAMA0 --baud 115200
"""
import argparse
import curses
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # pi/ 를 경로에 넣어 common.* 를 import
from common.nucleo_link import encode_velocity_cmd


class TeleopState:
    def __init__(self, max_lin_mps: float, max_ang_rps: float):
        self.vx = 0.0
        self.vy = 0.0
        self.omega = 0.0
        self.lin_step = 0.10  # 키 한 번당 m/s
        self.ang_step = 0.30  # 키 한 번당 rad/s
        self.max_lin = max_lin_mps
        self.max_ang = max_ang_rps

    def clamp(self):
        self.vx = max(-self.max_lin, min(self.max_lin, self.vx))
        self.vy = max(-self.max_lin, min(self.max_lin, self.vy))
        self.omega = max(-self.max_ang, min(self.max_ang, self.omega))

    def apply_key(self, key: int) -> bool:
        """종료 요청이면 False 를 돌려준다."""
        if key in (ord('w'), ord('W')):
            self.vx += self.lin_step
        elif key in (ord('s'), ord('S')):
            self.vx -= self.lin_step
        elif key in (ord('a'), ord('A')):
            self.vy += self.lin_step
        elif key in (ord('d'), ord('D')):
            self.vy -= self.lin_step
        elif key in (ord('q'), ord('Q')):
            self.omega += self.ang_step
        elif key in (ord('e'), ord('E')):
            self.omega -= self.ang_step
        elif key in (ord('x'), ord('X'), ord(' ')):
            self.vx = self.vy = self.omega = 0.0
        elif key in (ord('+'), ord('=')):
            self.lin_step *= 1.25
            self.ang_step *= 1.25
        elif key in (ord('-'), ord('_')):
            self.lin_step /= 1.25
            self.ang_step /= 1.25
        elif key == 27:  # ESC
            return False
        self.clamp()
        return True


def run(stdscr, args):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(0)

    import serial  # pyserial 없이도 --hexdump 가 동작하도록 지역 import
    ser = serial.Serial(args.port, args.baud, timeout=0)

    state = TeleopState(args.max_lin, args.max_ang)
    period = 1.0 / args.rate
    next_send = time.monotonic()

    try:
        while True:
            key = stdscr.getch()
            while key != -1:
                if not state.apply_key(key):
                    raise KeyboardInterrupt
                key = stdscr.getch()

            now = time.monotonic()
            if now >= next_send:
                frame = encode_velocity_cmd(state.vx, state.vy, state.omega)
                ser.write(frame)
                next_send = now + period

                stdscr.erase()
                stdscr.addstr(0, 0, "KSRC swerve teleop -- w/a/s/d/q/e, x/space=stop, +/-=step, ESC=quit")
                stdscr.addstr(2, 0, f"vx={state.vx:+.2f} m/s  vy={state.vy:+.2f} m/s  omega={state.omega:+.2f} rad/s")
                stdscr.addstr(3, 0, f"step: lin={state.lin_step:.2f} ang={state.ang_step:.2f}  rate={args.rate}Hz")
                stdscr.refresh()

            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        # 안전장치: 닫기 전에 0 프레임을 몇 번 명시적으로 보낸다. 안 그러면
        # 로버가 마지막 명령대로 계속 굴러간다. Nucleo 쪽 워치독도 링크 끊김을
        # 잡아주긴 하지만 이중으로 막는다.
        zero = encode_velocity_cmd(0.0, 0.0, 0.0)
        for _ in range(5):
            ser.write(zero)
            time.sleep(0.02)
        ser.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyAMA0", help="serial device to the Nucleo")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--rate", type=float, default=50.0, help="command send rate, Hz")
    ap.add_argument("--max-lin", type=float, default=0.5, help="max |vx|,|vy| in m/s")
    ap.add_argument("--max-ang", type=float, default=2.0, help="max |omega| in rad/s")
    ap.add_argument("--hexdump", action="store_true",
                     help="print the encoded frame for vx=1.5 vy=-2.25 omega=3.0 and exit "
                          "(cross-check against `test_teleop_protocol --hexdump`)")
    args = ap.parse_args()

    if args.hexdump:
        print(" ".join(f"{b:02x}" for b in encode_velocity_cmd(1.5, -2.25, 3.0)))
        return

    curses.wrapper(run, args)


if __name__ == "__main__":
    main()
