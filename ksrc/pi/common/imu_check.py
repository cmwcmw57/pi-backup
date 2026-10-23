"""IMU(ISM330DHCX) 동작 확인: Nucleo SPI 텔레메트리로 받은 값을 사람이 보기 쉽게 찍는다.

    python3 pi/common/imu_check.py            # 2초 정지 보정 후 10Hz 표시
    python3 pi/common/imu_check.py --no-cal   # 보정 없이 (자이로 원시값 그대로)

처음 2초는 **가만히** 두세요. 그동안 자이로 바이어스(정지 상태에서도 나오는 작은 값)를 재서 뺀다.
그 뒤 줄마다:
    acc  x y z [g]   |acc|   기울기 roll pitch [deg]   gyro x y z [deg/s]   누적각 x y z [deg]

확인할 것 (값은 **로버 좌표계**: x 전진, y 좌측, z 위. nucleo_link.IMU_CHIP_TO_BODY 로 변환):
  1. 정지: |acc| ~ 1.00, gyro ~ 0, 누적각이 거의 안 변함 (1분에 수 도 이내)
  2. 한 축을 위로 세워 보기: 그 축 acc 가 +1g, 뒤집으면 -1g
  3. 수평으로 90도 돌리기: 수직축 누적각이 ~ +-90 (반시계면 +)
  4. 원래 자리로 돌려놓기: 누적각이 ~0 으로 돌아옴
Enter 를 누르면 누적각을 0 으로 리셋. Ctrl+C 종료.
"""
import argparse
import math
import select
import sys
import time

from nucleo_link import SpiLink, TLM_FLAG_IMU_VALID


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bus", type=int, default=0)
    ap.add_argument("--dev", type=int, default=0)
    ap.add_argument("--speed", type=int, default=1_000_000)
    ap.add_argument("--cal", type=float, default=2.0, help="정지 보정 시간 s")
    ap.add_argument("--no-cal", action="store_true")
    args = ap.parse_args()

    link = SpiLink(args.bus, args.dev, args.speed)
    period = 0.02
    last_tick = None
    last_t_ms = None
    bias = [0.0, 0.0, 0.0]
    ang = [0.0, 0.0, 0.0]
    n_invalid = n_bad = 0

    def samples():
        """중복 tick 을 거른 (t_ms, gyro, acc) 를 50Hz 로 내준다. IMU 무효면 None."""
        nonlocal last_tick, n_invalid, n_bad
        while True:
            t0 = time.monotonic()
            tlm = link.exchange(0.0, 0.0, 0.0, poll_only=True)
            if tlm is None:
                n_bad += 1
            elif tlm["tick"] != last_tick:
                last_tick = tlm["tick"]
                if tlm["flags"] & TLM_FLAG_IMU_VALID:
                    yield tlm["t_ms"], tlm["gyro_dps"], tlm["acc_g"]
                else:
                    n_invalid += 1
                    yield tlm["t_ms"], None, None
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    src = samples()

    if not args.no_cal:
        print(f"보정 중 ({args.cal:.0f}초) -- 움직이지 마세요")
        acc_sum = [0.0] * 3
        n = 0
        t_end = time.monotonic() + args.cal
        while time.monotonic() < t_end:
            _, g, a = next(src)
            if g is None:
                continue
            for i in range(3):
                bias[i] += g[i]
                acc_sum[i] += a[i]
            n += 1
        if n == 0:
            print("IMU 값이 하나도 유효하지 않음 -- Nucleo 시리얼 부팅 메시지의 IMU 줄 확인")
            return
        bias = [b / n for b in bias]
        mag = math.sqrt(sum((s / n) ** 2 for s in acc_sum))
        print(f"자이로 바이어스 {bias[0]:+.2f} {bias[1]:+.2f} {bias[2]:+.2f} deg/s, "
              f"정지 |acc| {mag:.3f} g ({n} 샘플)")
        print("Enter = 누적각 0 으로 리셋, Ctrl+C = 종료\n")

    k = 0
    stdin_open = sys.stdin.isatty()
    try:
        for t_ms, g, a in src:
            if stdin_open and select.select([sys.stdin], [], [], 0)[0]:
                if sys.stdin.readline() == "":
                    stdin_open = False       # EOF (파이프 등)
                else:
                    ang = [0.0, 0.0, 0.0]
                    print("-- 누적각 리셋 --")
            if g is None:
                last_t_ms = None
                print(f"t={t_ms:8d} IMU --- (무효, 누적 {n_invalid})")
                continue
            gc = [g[i] - bias[i] for i in range(3)]
            if last_t_ms is not None:
                dt = (t_ms - last_t_ms) / 1000.0
                if 0.0 < dt < 0.2:
                    for i in range(3):
                        ang[i] += gc[i] * dt
            last_t_ms = t_ms
            k += 1
            if k % 5:
                continue                     # 화면은 10Hz
            mag = math.sqrt(sum(x * x for x in a))
            roll = math.degrees(math.atan2(a[1], a[2]))
            pitch = math.degrees(math.atan2(-a[0], math.hypot(a[1], a[2])))
            print(f"acc {a[0]:+.2f} {a[1]:+.2f} {a[2]:+.2f} |{mag:.2f}| "
                  f"roll {roll:+6.1f} pitch {pitch:+6.1f} | "
                  f"gyro {gc[0]:+7.1f} {gc[1]:+7.1f} {gc[2]:+7.1f} | "
                  f"누적 {ang[0]:+7.1f} {ang[1]:+7.1f} {ang[2]:+7.1f}")
    except KeyboardInterrupt:
        pass
    finally:
        link.close()
        print(f"\n무효 {n_invalid}, CRC 실패 {n_bad}")


if __name__ == "__main__":
    main()
