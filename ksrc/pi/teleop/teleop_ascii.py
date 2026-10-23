#!/usr/bin/env python3
"""아스키 `V` 텔레옵 — 펌웨어 USART2 터미널에 직접 붙는다.

`pi/common/nucleo_link.py` 의 바이너리 프레임(0xAA 0x55 + CRC8)은
`firmware/common/teleop_protocol.c` 가 **STM32 프로젝트에 포함돼 있지 않아서**
실기에서 파싱되지 않는다 (가상 Nucleo 전용). 실기에서 지금 동작하는 유일한
명령 경로는 `main.c process_command()` 의 아스키 줄 명령이다.

  PC/라파 --USB-- ST-Link --UART(PA2/PA3)-- STM32 USART2 @115200

**벤치·측정 전용이다.** 실주행은 SPI2(`pi_spi.c`+`spi_link.c`)로 간다.
아스키는 CRC 가 없어서 깨진 바이트를 거부할 수 없다. 단일 비트로 도달 가능한
위험 전이는 `V`(0x56) -> `T`(0x54) 하나뿐이고, 그 경우 서보 3개가 한 프레임
동안 ~0도로 갔다가 다음 프레임이 덮어쓴다 (10Hz 에서 100ms 과도현상).
`OFF`/`ZERO`/`SCAN` 은 토큰 전체가 일치해야 하므로 단일 비트로는 안 나온다.

전송률 10Hz 를 고정한 이유:
  - 펌웨어 워치독이 500ms (`TELEOP_TIMEOUT_MS`) -> 5프레임 마진
  - 제어주기가 20ms 라 두 주기마다 갱신이면 텔레옵에 충분
  - 50Hz 대비 깨질 기회 1/5, 링크 점유 2%
  - 파싱 실패 시 펌웨어가 사용법을 printf 하는데 115200 에서 ~4.3ms 를 막는다

사용법:
  python3 teleop_ascii.py                       # /dev/ttyACM0
  python3 teleop_ascii.py --port /dev/ttyUSB0
  python3 teleop_ascii.py --hz 2 --log          # A4 무부하속도 측정용

  vx vy om> 0.05 0 0      전진 0.05 m/s
  vx vy om> 0 0.05 0      좌 게걸음
  vx vy om> 0 0 0.3       제자리 반시계
  vx vy om> stop          전부 0 (= 0 0 0)
  vx vy om> !T 0 0 0 0    ! 로 시작하면 임의 아스키 명령을 그대로 보낸다
  vx vy om> !LOG          주기 로그 토글
  vx vy om> q             종료 (0 을 몇 번 보낸 뒤 포트 닫음)

`[CTRL] deg a b c d duty ...` 가 펌웨어의 5Hz 로그다. `deg` 는 **접기 뒤
목표 조향각**이라 ±90도 기구학 검증의 관측값이다. |deg| > 90 이 보이면
`swerve_fold_to_range` 가 한계를 못 지킨 것이다 (joint.c 가 조용히 클램프한다).
`GATE` 는 구동 정렬 게이트가 보류 중이라는 뜻이다.
"""
import argparse
import re
import sys
import threading
import time

try:
    import serial
except ImportError:
    sys.exit("pyserial 이 필요합니다: pip3 install pyserial")

PROMPT = "vx vy om> "
# 펌웨어 main.c 의 5Hz 한 줄 로그
CTRL_RE = re.compile(r"\[CTRL\]\s+deg\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+"
                     r"duty\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)(\s+GATE)?")
STEER_LIMIT_DEG = 90.0      # joint.c joint_config[].min_deg/max_deg 와 같은 값


class Link:
    """마지막 속도 명령을 고정 주기로 재전송한다 (워치독 먹이기)."""

    def __init__(self, ser, hz):
        self.ser = ser
        self.period = 1.0 / hz
        self.cmd = (0.0, 0.0, 0.0)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.sent = 0
        self.extra = []          # 사용자가 ! 로 넣은 아스키 명령 큐

    def set(self, vx, vy, om):
        with self.lock:
            self.cmd = (float(vx), float(vy), float(om))

    def raw(self, line):
        with self.lock:
            self.extra.append(line)

    def _writer(self):
        while not self.stop.is_set():
            with self.lock:
                vx, vy, om = self.cmd
                pending, self.extra = self.extra, []
            try:
                # 임의 명령을 속도 프레임보다 먼저 보낸다. 같은 주기에 둘을
                # 섞어도 펌웨어 RX 는 줄 단위라 안전하다.
                for line in pending:
                    self.ser.write((line + "\r\n").encode())
                self.ser.write(f"V {vx:.3f} {vy:.3f} {om:.3f}\r\n".encode())
                self.sent += 1
            except serial.SerialException as exc:
                print(f"\n[링크] 송신 실패: {exc}", file=sys.stderr)
                self.stop.set()
                return
            time.sleep(self.period)

    def _reader(self, show_log, warn):
        buf = b""
        while not self.stop.is_set():
            try:
                n = self.ser.in_waiting
                buf += self.ser.read(n if n else 1)
            except serial.SerialException:
                return
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                text = line.decode("utf-8", "replace").rstrip("\r")
                if not text:
                    continue
                m = CTRL_RE.search(text)
                if m and warn:
                    degs = [int(m.group(i)) for i in range(1, 5)]
                    over = [d for d in degs if abs(d) > STEER_LIMIT_DEG]
                    if over:
                        print(f"\n[!! 한계초과] 목표 조향각 {degs} 도 — "
                              f"±{STEER_LIMIT_DEG:.0f}도 밖. fold 가 한계를 "
                              f"못 지켰고 joint.c 가 클램프함\n{PROMPT}",
                              end="", flush=True)
                        continue
                if show_log or not m:
                    print(f"\n{text}\n{PROMPT}", end="", flush=True)

    def start(self, show_log, warn):
        for fn, a in ((self._writer, ()), (self._reader, (show_log, warn))):
            threading.Thread(target=fn, args=a, daemon=True).start()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--baud", type=int, default=115200,
                   help="펌웨어 USART2 와 같아야 한다 (main.c: 115200)")
    p.add_argument("--hz", type=float, default=10.0,
                   help="재전송 주기 [Hz]. 워치독 500ms 보다 빨라야 한다")
    p.add_argument("--log", action="store_true",
                   help="펌웨어 [CTRL] 5Hz 로그를 그대로 출력 (A4 측정용)")
    p.add_argument("--no-limit-warn", action="store_true",
                   help="목표 조향각이 ±90도를 넘을 때 경고하지 않는다")
    a = p.parse_args()
    if not 1.0 <= a.hz <= 50.0:
        p.error("--hz 는 1~50 사이여야 한다 (워치독 500ms, 제어주기 20ms)")

    try:
        ser = serial.Serial(a.port, a.baud, timeout=0.1)
    except serial.SerialException as exc:
        sys.exit(f"포트를 열 수 없습니다: {exc}\n"
                 f"  ls -l /dev/ttyACM*          장치 확인\n"
                 f"  sudo usermod -aG dialout $USER   권한 (재로그인 필요)")

    link = Link(ser, a.hz)
    link.start(a.log, not a.no_limit_warn)
    print(f"{a.port} @ {a.baud}  재전송 {a.hz:g}Hz  "
          f"(워치독 500ms, 제어주기 20ms)")
    print("첫 V 로 펌웨어가 텔레옵 모드로 전환되며 에코·로그를 끈다.")
    if a.log:
        # 펌웨어가 첫 V 에서 로그를 끄므로(main.c:511) 다시 켠다.
        time.sleep(0.4)
        link.raw("LOG")
        print("LOG 재활성화 요청 보냄 ([CTRL] 줄이 5Hz 로 올라와야 함)")
    print("숫자 3개 입력, stop, !<아스키명령>, q 로 종료\n")

    try:
        while not link.stop.is_set():
            try:
                s = input(PROMPT).strip()
            except EOFError:
                break
            if not s:
                continue
            if s in ("q", "quit", "exit"):
                break
            if s.startswith("!"):
                cmd = s[1:].strip()
                if cmd:
                    link.raw(cmd)
                continue
            if s in ("stop", "x", "0"):
                link.set(0, 0, 0)
                continue
            parts = s.split()
            if len(parts) != 3:
                print("  숫자 3개(vx vy omega) 또는 stop / !<명령> / q")
                continue
            try:
                vx, vy, om = (float(x) for x in parts)
            except ValueError:
                print("  숫자로 해석할 수 없음")
                continue
            link.set(vx, vy, om)
    except KeyboardInterrupt:
        pass
    finally:
        # 워치독에 맡기지 않고 명시적으로 0 을 보낸다 (최대 500ms 관성 제거).
        link.set(0, 0, 0)
        time.sleep(max(0.3, 3.0 / a.hz))
        link.stop.set()
        time.sleep(0.1)
        try:
            ser.write(b"V 0 0 0\r\n")
            ser.flush()
        except serial.SerialException:
            pass
        ser.close()
        print(f"\n정지 후 포트 닫음 (프레임 {link.sent}개 전송)")


if __name__ == "__main__":
    main()
