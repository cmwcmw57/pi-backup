"""실물 로버 SPI 점검·측정 묶음 (2026-10-14).  **라파에서 실행**, 새 펌웨어(10-13 이후) 기준.

    python3 pi/common/bringup.py link                 # 0. 링크 건강 (명령 없음)        ~10 s
    python3 pi/common/bringup.py latency              # 1. 명령이 적용되기까지 (띄움)   ~20 s
    python3 pi/common/bringup.py encdir               # 2. 바퀴별 구동·엔코더 부호 (띄움, WPI OFF)
    python3 pi/common/bringup.py res                  # 3. RL 잔차 경로 0x02 (띄움)     ~15 s
    python3 pi/common/bringup.py gate                 # 4. 구동 정렬 게이트 (띄움)      ~15 s
    python3 pi/common/bringup.py duty --vx 0.15       # 5. 지면 직진: 바퀴끼리 미는지
    python3 pi/common/bringup.py log --vx 0.1 --sec 8 --tag sand_wpi_on   # 6. 모래·경사 기록
    python3 pi/common/bringup.py wpi --wpi off            # 7. 바퀴 PI 끄기/켜기·corr (SPI, USB 불필요)
    python3 pi/common/bringup.py sand --vx 0.1 --sec 8    # 8. 모래 A/B (조건 자동 전환, 줄자 입력)
    ... --fake                                        # 노트북: 가짜 Nucleo (fake_nucleo.py)

각 항목은 PASS / 확인 필요 를 찍고, --out 디렉터리(기본 bringup_logs/<시각>_<항목>.npz)에
교환마다의 텔레메트리 + Pi 시각을 저장한다.  모든 항목은 끝에 속도 0 명령을 보낸다.
WPI ON/OFF 는 UART 명령이다 (USB 터미널에서 `WPI ON`).  SPI 로는 바꾸지 않는다.

시뮬과 맞출 값:
  latency  -> rover_env_cfg.DR2 action_delay (0, 1) [정책 스텝], obs_delay (1, 2)
  res      -> 펌웨어 잔차 = 시뮬 Res8Action (조향 += d, 속도 *= g, 100 ms 타임아웃)
  gate     -> DRIVE_GATE (60° 트리거, 10° 해제, 1.5 s 타임아웃)
"""

from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np

import nucleo_link as nl

DT = 0.020
F = nl  # flags


class Rec:
    """교환 기록기: 매 교환의 Pi 시각 + 텔레메트리 (None 이면 CRC 실패)."""

    def __init__(self, link):
        self.link, self.rows, self.bad = link, [], 0
        self.next_t = time.monotonic()

    def _wait(self):
        self.next_t += DT
        time.sleep(max(0.0, self.next_t - time.monotonic()))

    def _keep(self, t_send, tlm, kind, cmd):
        if tlm is None:
            self.bad += 1
        self.rows.append((t_send, time.monotonic(), kind, cmd, tlm))
        return tlm

    def poll(self):
        t = time.monotonic()
        r = self._keep(t, self.link.exchange(0.0, 0.0, 0.0, poll_only=True), "poll", (0, 0, 0))
        self._wait()
        return r

    def vel(self, vx, vy, w):
        t = time.monotonic()
        r = self._keep(t, self.link.exchange(vx, vy, w), "vel", (vx, vy, w))
        self._wait()
        return r

    def res(self, vx, vy, w, d, g):
        t = time.monotonic()
        r = self._keep(t, self.link.exchange_res(vx, vy, w, list(d), list(g)), "res", (vx, vy, w))
        self._wait()
        return r

    def hold(self, sec, fn, *a):
        out = []
        for _ in range(max(1, int(round(sec / DT)))):
            out.append(fn(*a))
        return [x for x in out if x is not None]

    def stop(self):
        for _ in range(5):
            self.vel(0.0, 0.0, 0.0)

    def save(self, path):
        keys = ("t_ms", "tick", "flags", "seq_echo", "imu_n", "rx_err")
        arr = {k: [] for k in keys + ("t_send", "t_recv", "kind", "cmd", "wheel_mps", "steer_cmd", "steer_meas",
                                       "duty", "gyro_dps", "acc_g", "cmd_applied", "ok")}
        for ts, tr, kind, cmd, tlm in self.rows:
            arr["t_send"].append(ts); arr["t_recv"].append(tr); arr["kind"].append(kind); arr["cmd"].append(cmd)
            arr["ok"].append(tlm is not None)
            z4 = (np.nan,) * 4
            for k in keys:
                arr[k].append(tlm[k] if tlm else -1)
            for k, src in (("wheel_mps", "wheel_mps"), ("steer_cmd", "steer_cmd_rad"), ("steer_meas", "steer_meas_rad"),
                           ("duty", "duty")):
                arr[k].append(tlm[src] if tlm else z4)
            arr["gyro_dps"].append(tlm["gyro_dps"] if tlm else (np.nan,) * 3)
            arr["acc_g"].append(tlm["acc_g"] if tlm else (np.nan,) * 3)
            arr["cmd_applied"].append(tlm["cmd"] if tlm else (np.nan,) * 3)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(path, **{k: np.asarray(v) for k, v in arr.items()})
        print(f"  저장 {path}")


def verdict(ok, msg):
    print(("  PASS        " if ok else "  확인 필요   ") + msg)
    return ok


def flags_str(f):
    names = [("TELEOP", F.TLM_FLAG_TELEOP_ACTIVE), ("WATCHDOG", F.TLM_FLAG_WATCHDOG), ("GATE", F.TLM_FLAG_GATE_PENDING),
             ("SPI", F.TLM_FLAG_SRC_SPI), ("IMU", F.TLM_FLAG_IMU_VALID), ("SERVO_ERR", F.TLM_FLAG_SERVO_READ_ERR),
             ("CLAMP", F.TLM_FLAG_STEER_CLAMPED), ("GATE_TO", F.TLM_FLAG_GATE_TIMEOUT), ("WPI", F.TLM_FLAG_WHEEL_PI),
             ("STALL", F.TLM_FLAG_WHEEL_STALL), ("ENC_DIR", F.TLM_FLAG_ENC_DIR), ("RES", F.TLM_FLAG_RES_ACTIVE)]
    return " ".join(n for n, b in names if f & b) or "-"


# ---------------------------------------------------------------- 0. link
def t_link(rec, a):
    """정지 명령(속도 0) 상태로: CRC 실패, 틱 중복·건너뜀, Nucleo 주기, imu_n, 서보, 플래그.
    폴링만 하면 텔레옵 전이라 펌웨어 10 Hz UART 로그가 켜져 있어 0.1 s 마다 40 ms 주기가 섞인다
    (10-14 실측).  속도 0 명령으로 텔레옵에 넘겨 로그를 끄고, 처음 0.5 s 는 버린다."""
    rec.hold(0.5, rec.vel, 0.0, 0.0, 0.0)
    n0, bad0 = len(rec.rows), rec.bad
    tl = rec.hold(a.sec, rec.vel, 0.0, 0.0, 0.0)
    rec.rows, rec.bad = rec.rows[n0:], rec.bad - bad0
    n = len(rec.rows)
    ticks = [t["tick"] for t in tl]
    tms = [t["t_ms"] for t in tl]
    uniq, per, skip = 0, [], 0
    for i in range(1, len(tl)):
        st = (ticks[i] - ticks[i - 1]) & 0xFF
        if st:
            uniq += 1
            per.append((tms[i] - tms[i - 1]) / st)
            skip += st - 1
    imu_n = np.array([t["imu_n"] for t in tl])
    servo = np.array([all(t["servo_valid"]) for t in tl])
    acc = np.array([np.linalg.norm(t["acc_g"]) for t in tl])
    print(f"  교환 {n}, CRC 실패 {rec.bad}, 고유 틱 {uniq}, 건너뛴 틱 {skip}, Nucleo 주기 {np.mean(per):.2f} ms "
          f"(최소 {np.min(per):.0f} 최대 {np.max(per):.0f}), 마지막 플래그 [{flags_str(tl[-1]['flags'])}]")
    print(f"  imu_n 분포 {dict(zip(*np.unique(imu_n, return_counts=True)))}, |acc| {acc.mean():.3f} g, "
          f"서보 4개 다 읽힘 {servo.mean():.1%}, rx_err {tl[-1]['rx_err']}")
    ok = verdict(rec.bad / max(n, 1) < 0.001, f"CRC 실패율 {rec.bad / max(n, 1):.2%} (< 0.1%)")
    ok &= verdict(abs(np.mean(per) - 20.0) < 0.5 and np.max(per) < 30, "Nucleo 제어주기 20 ms")
    ok &= verdict(np.mean((imu_n >= 3) & (imu_n <= 5)) > 0.95, "imu_n 3–5 (IMU 20 ms 평균 펌웨어가 들어감)")
    ok &= verdict(abs(acc.mean() - 1.0) < 0.05, "|acc| ≈ 1 g (정지)")
    ok &= verdict(servo.mean() > 0.99, "서보 4개 피드백")
    return ok


# ---------------------------------------------------------------- 1. latency
def t_latency(rec, a):
    """vx 0 <-> 0.05 를 번갈아 보내고, Nucleo 가 그 명령을 처음 IK 에 넣은 제어 틱의 시각까지 [ms].
    Nucleo 시계(t_ms)는 HSI 라 Pi 보다 0.3 % 쯤 빠르다 (10-14 실측) -> 두 시계 차이를 직선으로 맞추고
    가장 신선한 프레임(아래 1 %) 을 나이 0 으로 둔다.  첫 0.5 s 의 묵은 프레임은 버린다."""
    rec.hold(0.5, rec.vel, 0.0, 0.0, 0.0)
    n0 = len(rec.rows)
    events = []                       # (보낸 Pi 시각, 값)
    for v in [0.05, 0.0] * int(a.reps):
        events.append((time.monotonic(), v))
        rec.hold(0.4, rec.vel, v, 0.0, 0.0)
    rows = [r for r in rec.rows[n0:] if r[4] is not None]
    tr = np.array([r[1] for r in rows]) * 1000
    tms = np.array([r[4]["t_ms"] for r in rows], float)
    app = np.array([r[4]["cmd"][0] for r in rows])
    k = np.polyfit(tr, tr - tms, 1)                     # Pi 시각 - Nucleo 시각 = 직선 (시계 표류)
    resid = (tr - tms) - np.polyval(k, tr)
    k[1] += np.quantile(resid, 0.01)
    ages = (tr - tms) - np.polyval(k, tr)
    delays = []
    for t_send, v in events:
        j = np.nonzero((tr >= t_send * 1000) & (np.abs(app - v) < 1e-6))[0]
        if len(j):
            delays.append(tms[j[0]] + np.polyval(k, t_send * 1000) - t_send * 1000)
    d = np.array(delays)
    print(f"  Nucleo 시계 표류 {k[0] * 1e6:+.0f} ppm")
    print(f"  명령 → 적용: 평균 {d.mean():.1f} ms, 최소 {d.min():.1f}, 최대 {d.max():.1f} "
          f"= 정책 스텝 {d.min() / 20:.1f}–{d.max() / 20:.1f} (평균 {d.mean() / 20:.2f}), n={len(d)}")
    print(f"  관측 나이 (프레임 생성 → 라파 수신): 평균 {ages.mean():.1f} ms, p95 {np.quantile(ages, .95):.1f}, "
          f"최대 {ages.max():.1f}")
    ok = verdict(d.max() < 60, "명령은 3 스텝 안에 적용 (시뮬 rover_env_cfg DR2 action_delay 와 비교)")
    ok &= verdict(np.quantile(ages, .95) < 30, "관측 나이 < 30 ms (시뮬 obs_delay 와 비교)")
    return ok


# ---------------------------------------------------------------- 2. encdir
def t_encdir(rec, a):
    """**바퀴 띄우고, WPI OFF**.  vx +v / -v 로 각 모듈의 엔코더 속도·duty 부호를 본다."""
    ok = True
    for v in (a.v, -a.v):
        rec.hold(1.0, rec.vel, v, 0.0, 0.0)
        tl = rec.hold(1.0, rec.vel, v, 0.0, 0.0)
        w = np.mean([t["wheel_mps"] for t in tl], 0)
        d = np.mean([t["duty"] for t in tl], 0)
        print(f"  vx {v:+.2f}: 엔코더 m/s {np.round(w, 3)}  duty {np.round(d).astype(int)}")
        for m, name in enumerate(("FL", "FR", "RL", "RR")):
            if np.sign(w[m]) != np.sign(v) or abs(w[m]) < 0.3 * abs(v):
                ok &= verdict(False, f"{name}: 명령 {v:+.2f} 인데 엔코더 {w[m]:+.3f} (부호·배선·MOTOR_DRIVE_SIGN·ENCODER_SIGN)")
            if np.sign(d[m]) != np.sign(v):
                ok &= verdict(False, f"{name}: duty 부호 {d[m]:+.0f} (모터 맵·MOTOR_DRIVE_SIGN)")
    rec.stop()
    verdict(ok, "네 바퀴 모두 명령 방향으로 돌고 엔코더도 같은 부호 (WPI 켜도 됨)" if ok else "부호 문제 -- WPI 켜지 말 것")
    return ok


# ---------------------------------------------------------------- 3. res
def t_res(rec, a):
    """RL 잔차 경로 (타입 0x02).  띄운 상태 권장.  vx a.v 로 굴리면서:
    (1) 0x01 과 잔차 0 인 0x02 가 같은 조향목표  (2) 모듈 0 조향 +10°  (3) 모듈 0 속도 x1.5
    (4) 0x02 를 끊고 폴링만 -> 100 ms 쯤에 RES 꺼짐, 속도 명령은 500 ms 까지 살아 있음
    (5) NaN 잔차 -> 잔차만 버림   (6) 0x01 이 오면 즉시 잔차 꺼짐"""
    v, ok = a.v, True
    z, one = [0.0] * 4, [1.0] * 4
    b = rec.hold(1.0, rec.vel, v, 0.0, 0.0)[-10:]
    r0 = rec.hold(0.5, rec.res, v, 0.0, 0.0, z, one)[-10:]
    s_b, s_r = np.mean([t["steer_cmd_rad"] for t in b], 0), np.mean([t["steer_cmd_rad"] for t in r0], 0)
    ok &= verdict(np.abs(s_b - s_r).max() < 1e-4 and all(t["flags"] & F.TLM_FLAG_RES_ACTIVE for t in r0[-5:]),
                  f"(1) 잔차 0: 조향목표 0x01 과 같음 (차이 {np.degrees(np.abs(s_b - s_r).max()):.3f}°), RES 비트 켜짐")
    d10 = [math.radians(10), 0, 0, 0]
    r1 = rec.hold(0.8, rec.res, v, 0.0, 0.0, d10, one)[-10:]
    s1 = np.mean([t["steer_cmd_rad"] for t in r1], 0)
    m1 = np.mean([t["steer_meas_rad"] for t in r1], 0)
    ok &= verdict(abs(math.degrees(s1[0] - s_r[0]) - 10) < 0.2 and np.abs(s1[1:] - s_r[1:]).max() < 1e-4,
                  f"(2) 모듈 0 조향 +10°: 목표 {math.degrees(s1[0] - s_r[0]):+.2f}° (실측 {math.degrees(m1[0] - s_r[0]):+.1f}°), 나머지 그대로")
    r2 = rec.hold(1.0, rec.res, v, 0.0, 0.0, z, [1.5, 1, 1, 1])[-15:]
    w2 = np.mean([t["wheel_mps"] for t in r2], 0)
    ratio = w2[0] / np.mean(w2[1:])
    ok &= verdict(abs(ratio - 1.5) < 0.15, f"(3) 모듈 0 속도 x1.5: 엔코더 비 {ratio:.2f} (띄운 바퀴 기준)")
    rec.res(v, 0.0, 0.0, z, one)
    t_cut = time.monotonic()
    off_at, alive = None, True
    for _ in range(int(0.4 / DT)):
        t = rec.poll()
        if t is None:
            continue
        if off_at is None and not t["flags"] & F.TLM_FLAG_RES_ACTIVE:
            off_at = (time.monotonic() - t_cut) * 1000
        alive &= abs(t["cmd"][0] - v) < 1e-6
    ok &= verdict(off_at is not None and 80 <= off_at <= 160 and alive,
                  f"(4) 0x02 끊김: RES 꺼짐 {off_at if off_at is None else round(off_at)} ms (100 ms + 틱), 속도 명령은 유지 {alive}")
    rec.hold(0.3, rec.res, v, 0.0, 0.0, z, one)
    r5 = rec.hold(0.3, rec.res, v, 0.0, 0.0, [float("nan")] + [0.0] * 3, one)[-5:]
    ok &= verdict(all(not t["flags"] & F.TLM_FLAG_RES_ACTIVE for t in r5[-3:]) and all(abs(t["cmd"][0] - v) < 1e-6 for t in r5[-3:]),
                  "(5) NaN 잔차: 잔차만 꺼지고 속도 명령은 그대로")
    rec.hold(0.3, rec.res, v, 0.0, 0.0, z, one)
    r6 = [rec.vel(v, 0.0, 0.0) for _ in range(4)]
    ok &= verdict(r6[-1] is not None and not r6[-1]["flags"] & F.TLM_FLAG_RES_ACTIVE,
                  "(6) 텔레옵(0x01) 이 오면 잔차 즉시 꺼짐")
    rec.stop()
    return ok


# ---------------------------------------------------------------- 4. gate
def t_gate(rec, a):
    """구동 정렬 게이트.  띄운 상태.  (1) vx -> vy 로 한 번에 90° 전환: GATE 대기, duty 0, 정렬되면 해제
    (2) 방향을 20 ms 마다 40° 씩 0 -> 120° 로 돌림 (한 번에 60° 안 넘음): 실측 트리거로 대기 (10-13 수정)"""
    v, ok = a.v, True
    rec.hold(1.5, rec.vel, v, 0.0, 0.0)
    t0, hold_ticks, duty0 = None, 0, True
    seen = []
    for i in range(int(2.0 / DT)):
        t = rec.vel(0.0, v, 0.0)
        if t is None:
            continue
        seen.append(t)
        if t["flags"] & F.TLM_FLAG_GATE_PENDING:
            hold_ticks += 1
            duty0 &= all(d == 0 for d in t["duty"])
    rel = next((i for i, t in enumerate(seen) if i > 2 and not t["flags"] & F.TLM_FLAG_GATE_PENDING), None)
    err = max(abs(math.degrees(t_ - m_)) for t_, m_ in zip(seen[-1]["steer_cmd_rad"], seen[-1]["steer_meas_rad"]))
    ok &= verdict(hold_ticks > 3 and duty0 and rel is not None,
                  f"(1) 90° 전환: 대기 {hold_ticks * 20} ms 동안 duty 0, 해제 뒤 조향 오차 {err:.1f}°")
    rec.hold(1.5, rec.vel, v, 0.0, 0.0)
    pend = 0
    for k in range(4):
        th = math.radians(40 * k)
        t = rec.vel(v * math.cos(th), v * math.sin(th), 0.0)
    for _ in range(int(1.5 / DT)):
        t = rec.vel(v * math.cos(th), v * math.sin(th), 0.0)
        if t is not None and t["flags"] & F.TLM_FLAG_GATE_PENDING:
            pend += 1
    ok &= verdict(pend > 0, f"(2) 40° 씩 세 번 (120°): 실측 트리거로 대기 {pend * 20} ms "
                            f"(0 이면 10-13 이전 펌웨어 -- 게이트 수정 안 들어감)")
    rec.stop()
    return ok


# ---------------------------------------------------------------- 5. duty (ground)
def t_duty(rec, a):
    """**지면** 직진 a.sec 초: 바퀴별 duty·속도.  네 바퀴가 같은 속도를 내는데 duty 가 크게 벌어지거나
    부호가 갈리면 서로 밀고 있다는 뜻 (WPI ON 이면 특히).  STALL / ENC_DIR 플래그도 본다."""
    rec.hold(1.0, rec.vel, a.vx, 0.0, 0.0)
    tl = rec.hold(a.sec, rec.vel, a.vx, 0.0, 0.0)
    rec.stop()
    d = np.array([t["duty"] for t in tl], float)
    w = np.array([t["wheel_mps"] for t in tl], float)
    wpi = bool(tl[-1]["flags"] & F.TLM_FLAG_WHEEL_PI)
    fl = np.bitwise_or.reduce([t["flags"] for t in tl])
    print(f"  WPI {'ON' if wpi else 'OFF'}, 바퀴 m/s 평균 {np.round(w.mean(0), 3)}, duty 평균 {np.round(d.mean(0)).astype(int)} "
          f"표준편차 {np.round(d.std(0)).astype(int)}")
    spread = (d.mean(0).max() - d.mean(0).min()) / max(abs(d.mean()), 1)
    ok = verdict(np.all(np.sign(d.mean(0)) == np.sign(a.vx)), "네 바퀴 duty 가 모두 전진 방향")
    ok &= verdict(spread < 0.5, f"바퀴 간 duty 차이 {spread:.0%} (평균 대비, < 50%: 서로 미는 징후 없음)")
    ok &= verdict(not fl & (F.TLM_FLAG_WHEEL_STALL | F.TLM_FLAG_ENC_DIR), f"보호 안 걸림 (플래그 [{flags_str(fl)}])")
    return ok


# ---------------------------------------------------------------- 6. log
def t_log(rec, a):
    """명령 하나를 a.sec 초 보내며 전부 기록 (모래 A/B, 경사 정지 등). 진행 거리는 줄자로 따로."""
    rec.hold(0.5, rec.poll)
    tl, prev, t0 = [], 0, time.monotonic()
    watch = ((F.TLM_FLAG_WHEEL_STALL, "STALL (어느 바퀴가 보정 상한에서 0.5 s 넘게 멈춤 -> FF 만)"),
             (F.TLM_FLAG_ENC_DIR, "ENC_DIR (엔코더 방향 불일치로 잠김)"), (F.TLM_FLAG_GATE_PENDING, "GATE (정렬 대기)"))
    for _ in range(max(1, int(round(a.sec / DT)))):
        t = rec.vel(a.vx, a.vy, a.w)
        if t is None:
            continue
        tl.append(t)
        for bit, name in watch:      # 걸리고 풀리는 순간을 바로 찍는다
            if (t["flags"] ^ prev) & bit:
                print(f"  {time.monotonic() - t0:5.2f}s  {'켜짐' if t['flags'] & bit else '꺼짐'}  {name}  "
                      f"duty {list(t['duty'])}  바퀴 m/s {[round(x, 3) for x in t['wheel_mps']]}", flush=True)
        prev = t["flags"]
    rec.stop()
    w = np.array([t["wheel_mps"] for t in tl], float)
    d = np.array([t["duty"] for t in tl], float)
    fl = np.bitwise_or.reduce([t["flags"] for t in tl])
    print(f"  {len(tl)} 프레임, 바퀴 m/s 평균 {np.round(w.mean(0), 3)} → 엔코더 거리 {np.round(w.sum(0) * DT, 3)} m, "
          f"|duty| 평균 {np.round(np.abs(d).mean(0)).astype(int)}, 포화 {np.mean(np.abs(d) >= 799):.1%}, 플래그 [{flags_str(fl)}]")
    print("  → 실제 이동 거리(줄자)와 엔코더 거리 비 = 진행률. 차이가 헛돎.")
    return True


# ---------------------------------------------------------------- 7. wpi (설정)
def _set_wpi(rec, on, corr):
    """SPI 타입 0x03 으로 바퀴 PI 설정 후, 텔레메트리로 적용 확인. (on, corr) 실제값을 돌려준다."""
    rec.link.config(wpi_on=on, corr=corr)
    rec.next_t = time.monotonic()        # 앞선 주행으로 밀린 일정 초기화 (안 하면 대기 없이 연달아 읽어 옛 프레임을 봄)
    for _ in range(5):
        t = rec.poll()
    if t is None:
        return None, None
    return bool(t["flags"] & F.TLM_FLAG_WHEEL_PI), t["wpi_corr"]


def t_wpi(rec, a):
    """바퀴 PI 를 라파에서 SPI 로 켜고 끄기 (--wpi on|off|keep, --corr N).  USB 없이 됨.  재부팅하면 펌웨어 기본값."""
    on = {"on": True, "off": False, "keep": None}[a.wpi]
    got_on, got_corr = _set_wpi(rec, on, a.corr)
    print(f"  지금 바퀴 PI {'ON' if got_on else 'OFF'}, 보정량 상한 corr {got_corr} duty")
    ok = verdict(got_on is not None and (on is None or got_on == on) and (a.corr is None or got_corr == round(a.corr)),
                 "설정이 펌웨어에 들어감 (텔레메트리로 확인)")
    return ok


# ---------------------------------------------------------------- 8. sand (모래 A/B)
SAND_CONDS = {"off": (False, None), "on": (True, 250.0), "c100": (True, 100.0), "c50": (True, 50.0)}


def t_sand(rec, a):
    """모래 A/B: 조건(--conds off,on,c100)마다 같은 코스를 --vx 로 --sec 초.  조건은 SPI 로 바꾸고(USB 불필요),
    매 회 출발선 정렬 Enter -> 주행 -> 줄자 거리 입력.  회마다 npz, 끝에 요약 CSV·표.  진행률 = 줄자 ÷ 엔코더 거리."""
    conds = [c.strip() for c in a.conds.split(",") if c.strip()]
    for c in conds:
        if c not in SAND_CONDS:
            raise SystemExit(f"모르는 조건 {c} (가능: {', '.join(SAND_CONDS)})")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    rows = []
    order = [(r, c) for r in range(a.sand_reps) for c in conds]
    for i, (r, c) in enumerate(order):
        on, corr = SAND_CONDS[c]
        got_on, got_corr = _set_wpi(rec, on, corr)
        print(f"\n[{i + 1}/{len(order)}] 조건 {c}: 바퀴 PI {'ON' if got_on else 'OFF'}, corr {got_corr}  (반복 {r + 1})")
        if not a.no_prompt:
            input("  로버를 출발선에 맞추고 Enter (주행 시작) ")
        run = Rec(rec.link)
        run.hold(0.3, run.vel, 0.0, 0.0, 0.0)
        tl, fl_all = [], 0
        for _ in range(max(1, int(round(a.sec / DT)))):
            t = run.vel(a.vx, a.vy, a.w)
            if t is not None:
                tl.append(t)
                fl_all |= t["flags"]
        run.stop()
        w = np.array([t["wheel_mps"] for t in tl], float)
        d = np.array([t["duty"] for t in tl], float)
        enc = float((w.sum(0) * DT).mean())
        tape = None
        if not a.no_prompt:
            s = input(f"  엔코더 거리 {enc:.3f} m.  줄자로 잰 실제 이동 거리 [m] (모르면 Enter): ").strip()
            tape = float(s) if s else None
        prog = tape / enc if tape and enc > 1e-3 else float("nan")
        row = dict(i=i, cond=c, rep=r, wpi=int(bool(got_on)), corr=got_corr, enc_m=round(enc, 4),
                   tape_m=tape if tape is not None else float("nan"), progress=round(prog, 3),
                   duty_abs=round(float(np.abs(d).mean()), 1), sat=round(float(np.mean(np.abs(d) >= 799)), 4),
                   stall=int(bool(fl_all & F.TLM_FLAG_WHEEL_STALL)), enc_dir=int(bool(fl_all & F.TLM_FLAG_ENC_DIR)))
        rows.append(row)
        print(f"  진행률 {prog:.2f}, |duty| {row['duty_abs']}, 포화 {row['sat']:.1%}, 스톨 {row['stall']}")
        run.save(os.path.join(a.out, f"{stamp}_sand_{i:02d}_{c}_r{r}{'_' + a.tag if a.tag else ''}.npz"))
    _set_wpi(rec, True, 250.0)                       # 펌웨어 기본으로 돌려놓음
    import csv
    path = os.path.join(a.out, f"{stamp}_sand_summary{'_' + a.tag if a.tag else ''}.csv")
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\n  요약 {path}")
    print(f"  {'조건':>6} {'회':>3} {'엔코더 m':>9} {'줄자 m':>8} {'진행률':>7} {'|duty|':>7} {'포화':>6} {'스톨':>4}")
    for c in conds:
        rr = [x for x in rows if x["cond"] == c]
        mean = lambda k: np.nanmean([x[k] for x in rr]) if rr else float("nan")
        print(f"  {c:>6} {len(rr):3d} {mean('enc_m'):9.3f} {mean('tape_m'):8.3f} {mean('progress'):7.2f} "
              f"{mean('duty_abs'):7.0f} {mean('sat'):6.1%} {sum(x['stall'] for x in rr):4d}")
    return True


TESTS = {"link": t_link, "latency": t_latency, "encdir": t_encdir, "res": t_res, "gate": t_gate,
         "duty": t_duty, "log": t_log, "wpi": t_wpi, "sand": t_sand}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("test", choices=list(TESTS) + ["all-lifted"])
    ap.add_argument("--fake", action="store_true", help="가짜 Nucleo (노트북)")
    ap.add_argument("--sec", type=float, default=None)
    ap.add_argument("--v", type=float, default=0.05, help="띄운 시험의 속도 [m/s]")
    ap.add_argument("--vx", type=float, default=0.1)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--w", type=float, default=0.0)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="bringup_logs")
    ap.add_argument("--enc-sign", type=int, nargs=4, default=(1, 1, 1, 1), help="--fake 에서 엔코더 부호 실수 흉내")
    ap.add_argument("--wpi", choices=("on", "off", "keep"), default="keep", help="wpi: 바퀴 PI 켜기/끄기")
    ap.add_argument("--corr", type=float, default=None, help="wpi: 보정량 상한 [duty] (기본 250)")
    ap.add_argument("--conds", default="off,on,c100", help="sand: 조건 목록 (off, on=corr250, c100, c50)")
    ap.add_argument("--sand-reps", type=int, default=2, help="sand: 조건마다 반복 횟수")
    ap.add_argument("--no-prompt", action="store_true", help="sand: 출발선·줄자 입력 없이 (시험용)")
    a = ap.parse_args()
    names = ["link", "latency", "encdir", "res", "gate"] if a.test == "all-lifted" else [a.test]
    results = {}
    for name in names:
        if a.fake:
            from fake_nucleo import FakeSpi
            link = nl.SpiLink(spi=FakeSpi(enc_sign=tuple(a.enc_sign)))
        else:
            link = nl.SpiLink()
        rec = Rec(link)
        a.sec = a.sec if a.sec is not None else {"link": 10.0, "duty": 5.0, "log": 8.0}.get(name, 5.0)
        print(f"[{name}] {TESTS[name].__doc__.strip().splitlines()[0]}")
        try:
            results[name] = TESTS[name](rec, a)
        finally:
            if name not in ("link", "wpi"):
                rec.stop()
            stamp = time.strftime("%Y%m%d_%H%M%S")
            if name not in ("wpi", "sand"):              # sand 는 회마다 따로 저장
                rec.save(os.path.join(a.out, f"{stamp}_{name}{'_' + a.tag if a.tag else ''}.npz"))
            link.close()
        a.sec = None if a.test == "all-lifted" else a.sec
    print("결과: " + ", ".join(f"{k} {'PASS' if v else '확인 필요'}" for k, v in results.items()))


if __name__ == "__main__":
    main()
