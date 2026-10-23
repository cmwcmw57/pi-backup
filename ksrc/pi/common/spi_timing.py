"""Pi <-> Nucleo SPI 링크 주기·지연 측정. **라파에서 실행** (명령은 보내지 않는다: 폴링만).

    python3 pi/common/spi_timing.py              # 50Hz 로 10초
    python3 pi/common/spi_timing.py --hz 100 --sec 20

재는 것:
  1. Pi 교환 주기   : Pi 가 실제로 SPI 를 주고받은 간격 (리눅스 스케줄링 흔들림 포함)
  2. 교환 1회 시간  : xfer2 호출 하나에 걸린 시간 (144바이트 + 드라이버 오버헤드)
  3. STM 제어 주기  : 텔레메트리 t_ms (Nucleo 시계) 를 tick 별로 차분 -> 20ms 여야 함
  4. 중복 / 건너뜀  : 같은 tick 을 또 받음 (Pi 가 더 빠름) / tick 이 2 이상 뛰어 놓침
  5. 왕복 지연      : seq 를 보낸 교환부터 그 seq 가 텔레메트리 seq_echo 로 돌아올 때까지
                      (Pi 시계 기준. STM 이 다음 제어주기에 텔레메트리를 다시 만들어야 돌아오므로
                       0~20ms + 교환 주기 정도가 정상)
"""
import argparse
import statistics
import time

from nucleo_link import SpiLink


def summary(name, xs, unit="ms"):
    if not xs:
        print(f"{name:<16} (표본 없음)")
        return
    xs_sorted = sorted(xs)
    p99 = xs_sorted[min(len(xs) - 1, int(len(xs) * 0.99))]
    print(f"{name:<16} 평균 {statistics.mean(xs):7.2f}  표준편차 {statistics.pstdev(xs):6.2f}  "
          f"최소 {xs_sorted[0]:7.2f}  최대 {xs_sorted[-1]:7.2f}  p99 {p99:7.2f} {unit}  (n={len(xs)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hz", type=float, default=50.0, help="Pi 교환 목표 주기")
    ap.add_argument("--sec", type=float, default=10.0)
    ap.add_argument("--bus", type=int, default=0)
    ap.add_argument("--dev", type=int, default=0)
    ap.add_argument("--speed", type=int, default=1_000_000)
    args = ap.parse_args()

    link = SpiLink(args.bus, args.dev, args.speed)
    period = 1.0 / args.hz

    pi_dt, xfer_ms, stm_dt, rtt = [], [], [], []
    n = n_bad = n_dup = n_skip = 0
    last_t0 = None
    last_tick = last_tms = None
    sent_at = {}                    # seq -> 보낸 시각
    seen_echo = None

    print(f"측정 중: {args.hz:.0f}Hz, {args.sec:.0f}초, SPI {args.speed / 1e6:.2f}MHz (폴링만)")
    t_end = time.monotonic() + args.sec
    next_t = time.monotonic()
    try:
        while time.monotonic() < t_end:
            t0 = time.monotonic()
            if last_t0 is not None:
                pi_dt.append((t0 - last_t0) * 1000)
            last_t0 = t0

            seq = (link._seq + 1) & 0xFF        # exchange() 가 붙일 seq
            sent_at[seq] = t0
            tlm = link.exchange(0.0, 0.0, 0.0, poll_only=True)
            t1 = time.monotonic()
            xfer_ms.append((t1 - t0) * 1000)
            n += 1

            if tlm is None:
                n_bad += 1
            else:
                echo = tlm["seq_echo"]
                if echo != seen_echo and echo in sent_at:
                    rtt.append((t1 - sent_at[echo]) * 1000)
                    seen_echo = echo
                tick, tms = tlm["tick"], tlm["t_ms"]
                if tick == last_tick:
                    n_dup += 1
                else:
                    if last_tick is not None:
                        step = (tick - last_tick) & 0xFF
                        if step > 1:
                            n_skip += step - 1
                        stm_dt.append((tms - last_tms) / step)
                    last_tick, last_tms = tick, tms

            next_t += period
            time.sleep(max(0.0, next_t - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        link.close()

    n_ok = n - n_bad
    print()
    summary("Pi 교환 주기", pi_dt)
    summary("교환 1회 시간", xfer_ms)
    summary("STM 제어 주기", stm_dt)
    summary("왕복 지연", rtt)
    print(f"{'프레임':<16} 총 {n}, CRC 실패 {n_bad}, 같은 tick 중복 {n_dup} "
          f"({100 * n_dup / max(n_ok, 1):.0f}%), 놓친 tick {n_skip}")
    print(f"{'실효 갱신율':<16} {(n_ok - n_dup) / args.sec:.1f} Hz (새 텔레메트리를 받은 횟수/초)")


if __name__ == "__main__":
    main()
