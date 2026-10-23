"""PC 조이스틱(UDP) -> 라파 -> Nucleo (SPI) 중계. **라파에서 실행한다.**

    python3 pi/common/udp_spi_bridge.py              # UDP 5005 대기, SPI 50Hz

PC 쪽:
    python3 pi/teleop/teleop_joystick.py --udp raspberrypi.local --max-lin 0.15 --max-ang 0.8 --monitor

왜 이렇게 나눴나: 라파는 화면이 없어서 조이스틱 창(pygame)을 ssh X 포워딩으로 띄워야 하는데,
SDL 2.30 이 원격 X 에서 MIT-SHM 요청을 거부당하고 죽는다 (-X, -Y 모두). 그래서 창은 PC 에서
그리고 라파는 받은 vx, vy, ω 를 SPI 로 넘기는 일만 한다. Nucleo 와의 통신은 여전히 라파가 맡는다.

안전:
  - 명령이 STALE_S(0.3s) 넘게 안 오면 0 명령을 보낸다 (PC 창 닫힘, 와이파이 끊김).
    그것까지 끊기면 STM 워치독(500ms)이 다시 한 번 막는다.
  - 패킷은 매직 + seq + CRC 없음 대신 길이·매직·유한값 검사. 순서가 뒤집힌 오래된 패킷은 버린다.

패킷 (PC -> 라파, 20 바이트): b"KSRC" + <I seq> + <f vx> <f vy> <f omega>
응답 (라파 -> PC, 5Hz): JSON 한 줄 {"cmd":[...],"deg":[...],"duty":[...],"flags":n,"crc_bad":n}
"""
import argparse
import json
import math
import socket
import struct
import time

from nucleo_link import SpiLink, TLM_FLAG_GATE_PENDING, TLM_FLAG_WATCHDOG

MAGIC = b"KSRC"
PKT = struct.Struct("<4sIfff")
STALE_S = 0.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--bus", type=int, default=0)
    ap.add_argument("--dev", type=int, default=0)
    ap.add_argument("--speed", type=int, default=1_000_000)
    ap.add_argument("--max-lin", type=float, default=0.32, help="받은 값 상한 m/s (안전 클램프)")
    ap.add_argument("--max-ang", type=float, default=2.0, help="받은 값 상한 rad/s")
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.port))
    sock.setblocking(False)
    link = SpiLink(args.bus, args.dev, args.speed)

    cmd = (0.0, 0.0, 0.0)
    last_rx = 0.0
    last_seq = None
    peer = None
    tlm = None
    n_rx = n_drop = n_bad = 0
    period = 1.0 / args.hz
    next_reply = next_print = time.monotonic()
    print(f"UDP {args.port} 대기 중 -> SPI {args.hz:.0f}Hz. Ctrl+C 종료")

    def clamp(v, lim):
        return max(-lim, min(lim, v))

    try:
        while True:
            t0 = time.monotonic()
            # 쌓인 패킷 중 가장 최신 것만 쓴다
            while True:
                try:
                    data, addr = sock.recvfrom(64)
                except BlockingIOError:
                    break
                if len(data) != PKT.size:
                    n_drop += 1
                    continue
                magic, seq, vx, vy, w = PKT.unpack(data)
                if magic != MAGIC or not all(map(math.isfinite, (vx, vy, w))):
                    n_drop += 1
                    continue
                # 오래된 패킷(순서 뒤집힘) 버림. PC 재시작(seq 가 작게 다시 시작)은 허용
                if last_seq is not None and addr == peer and 0 < (last_seq - seq) % 2**32 < 2**31:
                    n_drop += 1
                    continue
                last_seq, peer = seq, addr
                cmd = (clamp(vx, args.max_lin), clamp(vy, args.max_lin), clamp(w, args.max_ang))
                last_rx = t0
                n_rx += 1

            stale = (t0 - last_rx) > STALE_S
            send = (0.0, 0.0, 0.0) if stale else cmd
            if peer is None:
                r = link.exchange(0.0, 0.0, 0.0, poll_only=True)   # PC 가 오기 전엔 명령 안 보냄
            else:
                r = link.exchange(*send)
            if r is None:
                n_bad += 1
            else:
                tlm = r

            now = time.monotonic()
            if peer is not None and tlm is not None and now >= next_reply:
                reply = {
                    "cmd": [round(x, 3) for x in tlm["cmd"]],
                    "deg": [round(math.degrees(a), 1) for a in tlm["steer_cmd_rad"]],
                    "duty": list(tlm["duty"]),
                    "flags": tlm["flags"],
                    "crc_bad": n_bad,
                    "stale": stale,
                }
                try:
                    sock.sendto(json.dumps(reply).encode(), peer)
                except OSError:
                    pass
                next_reply = now + 0.2
            if now >= next_print:
                f = tlm["flags"] if tlm else 0
                tags = ("W" if f & TLM_FLAG_WATCHDOG else "-") + ("G" if f & TLM_FLAG_GATE_PENDING else "-")
                src = "대기(PC 없음)" if peer is None else ("끊김->0" if stale else f"{peer[0]}")
                print(f"[{src}] 보냄 v {send[0] + 0.0:+.3f} {send[1] + 0.0:+.3f} {send[2] + 0.0:+.3f} "
                      f"[{tags}] rx={n_rx} drop={n_drop} crc_bad={n_bad}")
                next_print = now + 0.5
            time.sleep(max(0.0, period - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        for _ in range(5):
            link.exchange(0.0, 0.0, 0.0)
            time.sleep(0.02)
        link.close()
        sock.close()


if __name__ == "__main__":
    main()
