"""Nucleo SPI 링크 확인용: 50Hz 로 교환하며 텔레메트리를 한 줄씩 찍는다.

    python3 pi/common/spi_monitor.py                 # 폴링만 (명령을 보내지 않음)
    python3 pi/common/spi_monitor.py --vx 0.05       # 0.05 m/s 전진 명령을 계속 보냄
    python3 pi/common/spi_monitor.py --vy 0.05       # 게걸음 (안 준 축은 0)
    python3 pi/common/spi_monitor.py --w 0.5         # 제자리 회전

폴링 모드는 속도 명령을 보내지 않을 뿐이다. 펌웨어 모터 자가시험(MTEST)이 켜져
있었다면 그건 폴링으로 멈추지 않는다 (기본 OFF). 같은 제어주기(tick)를 두 번 받으면
화면에 찍지 않는다.

Ctrl+C 로 종료. 명령을 보내던 중이면 종료 직전에 정지 명령을 한 번 보낸다.
"""
import argparse
import math
import time

from nucleo_link import (SpiLink, TLM_FLAG_GATE_PENDING, TLM_FLAG_GATE_TIMEOUT,
                         TLM_FLAG_IMU_VALID, TLM_FLAG_WATCHDOG)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bus", type=int, default=0)
    ap.add_argument("--dev", type=int, default=0)
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--speed", type=int, default=1_000_000, help="SPI 클럭 Hz")
    ap.add_argument("--vx", type=float, default=None)
    ap.add_argument("--vy", type=float, default=None)
    ap.add_argument("--w", type=float, default=None)
    args = ap.parse_args()

    link = SpiLink(args.bus, args.dev, args.speed)
    # 셋 중 하나라도 주면 명령 모드 (안 준 축은 0)
    poll = args.vx is None and args.vy is None and args.w is None
    vx, vy, w = (0.0 if v is None else v for v in (args.vx, args.vy, args.w))
    period = 1.0 / args.hz
    n_ok = n_bad = n_dup = 0
    last_tick = None
    try:
        while True:
            t0 = time.monotonic()
            tlm = link.exchange(vx, vy, w, poll_only=poll)
            if tlm is None:
                n_bad += 1
                print(f"[BAD] CRC/동기 실패 (누적 {n_bad})")
            elif tlm["tick"] == last_tick:
                n_dup += 1          # 같은 제어주기 프레임을 다시 받음 (비동기라 정상)
            else:
                last_tick = tlm["tick"]
                n_ok += 1
                if n_ok % max(1, int(args.hz / 5)) == 0:   # 화면은 5Hz
                    f = tlm["flags"]
                    steer = " ".join(f"{math.degrees(a):6.1f}" if ok else "   ---"
                                     for a, ok in zip(tlm["steer_meas_rad"], tlm["servo_valid"]))
                    wheel = " ".join(f"{v:+.3f}" for v in tlm["wheel_mps"])
                    enc = " ".join(f"{e:+7d}" for e in tlm["enc_ticks"])
                    imu = ("gyro " + " ".join(f"{g:+6.1f}" for g in tlm["gyro_dps"]) +
                           " acc " + " ".join(f"{a:+.2f}" for a in tlm["acc_g"])
                           if f & TLM_FLAG_IMU_VALID else "IMU ---")
                    tags = ("W" if f & TLM_FLAG_WATCHDOG else "-") + \
                           ("G" if f & TLM_FLAG_GATE_PENDING else "-") + \
                           ("T" if f & TLM_FLAG_GATE_TIMEOUT else "-")
                    print(f"t={tlm['t_ms']:>8} tick={tlm['tick']:>3} [{tags}] "
                          f"steer[{steer}] wheel[{wheel}] enc[{enc}] {imu} rx_err={tlm['rx_err']} dup={n_dup}")
            time.sleep(max(0.0, period - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        if not poll:
            link.exchange(0.0, 0.0, 0.0)
        link.close()


if __name__ == "__main__":
    main()
