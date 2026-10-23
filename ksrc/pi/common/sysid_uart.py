#!/usr/bin/env python3
"""모터·서보 모델링 측정 (firmware/MOTOR_MODELING_PLAN.md §0) — USB 터미널로 직접.

  노트북 --USB-- ST-Link --UART(PA2/PA3)-- STM32 USART2 @115200

SPI 없이 펌웨어 아스키 명령만 쓴다:
  CAP  구동모터 1 kHz 캡처 (main.c run_capture): 계단·램프 duty 를 넣고 엔코더 틱을 RAM 에 기록 후 덤프
  STEP 조향 서보 계단 응답 (main.c, 펌웨어 안에서 ~0.7 ms 로 읽음) -- 출력 텍스트를 그대로 저장
  ACC  서보 가속도 레지스터,  S  서보 최고속 제한

** 바퀴를 띄운 상태에서 쓴다.  텔레옵(V) 을 한 번이라도 보냈으면 리셋 후 쓴다. **

믿을 수 있는 값인지 확인하려고 drive·steer 는 같은 시험을 --reps 번(기본 5) 반복한다.
  - 반복 순서는 섞는다 (배터리가 서서히 처져도 특정 duty 에만 몰리지 않게).
  - drive 는 기준 시험 REF (duty 400 정상속도) 를 처음·10개마다·끝에 끼워 넣어 시간에 따른 표류를 잰다.
  - 판정·피팅·모델 검증은 sysid_analyze.py 가 한다 (측정과 분리 -- 같은 데이터를 다시 분석할 수 있게).

사용법 (첫 연결 날 순서는 MOTOR_MODELING_PLAN.md §10):
  python3 sysid_uart.py raw HELP                       # 포트 확인: 아무 명령 한 줄, 출력이 멎을 때까지 표시
  python3 sysid_uart.py cap ALL 0 400 100 600          # CAP 한 번 → npz + 요약 (연결 확인용)
  python3 sysid_uart.py steer --servo 1                # S1 큰 계단 × 5회, ACC 50 과 0, 1 ms 궤적 (~8분)
  python3 sysid_uart.py steer-small --servo 1          # S2·S3 작은 계단·히스테리시스 × 5회 (~5분)
  python3 sysid_uart.py drive                          # D1·D2·D3 × 5회, 모터 4개 동시 (~12분)
  python3 sysid_uart.py steer --servo 1 --tag ground   # S4: 바퀴 땅에 놓고 같은 시험 → 띄운 것과 비교
  python3 sysid_uart.py pi                             # 바퀴 속도 PI (WCAP) 계단·저속·역전 × 5회 (~2분)
  python3 sysid_uart.py wcap ALL 0 2 100 600           # WCAP 한 번 (PI 0→2 rad/s 계단)
  python3 sysid_uart.py imu --ref-deg 10 --tag tilt10   # IMU 정지 기울기 시험 (경사계 값과 비교)
  python3 sysid_uart.py imu --tag bumps                # IMU 주행 중 진동 (기록 중 Pi 로 몰고, 끝나면 정지)
  python3 sysid_analyze.py sysid_logs/<시각>_<종류>    # 분석·판정 → report.txt, summary.json
  python3 test_sysid.py                                # 분석 코드 자체 검증 (가짜 펌웨어, ~2분)
  공통 옵션은 하위 명령 앞에: --port /dev/ttyACM0  --out sysid_logs
  하위 명령별 옵션: python3 sysid_uart.py <명령> -h

저장 (<out>/<시각>_<종류>[_tag]/ 디렉터리 하나에):
  drive              CAP 마다 npz (t_s, pos_ticks(N,4), duty(N), 헤더 값, group, rep, order) + manifest.json
  steer_servoN       steer.txt -- 펌웨어 출력 그대로, '### ACC a STEP n ang rep r' 줄로 구분, STEP 마다 1 ms 궤적
  steersmall_servoN  steer.txt -- 위와 같고 표식 줄 끝에 role=.. amp=..
CAP 의 pos_ticks 는 부호 보정 적용(+ = 바퀴 전진), 열 순서 = MOTOR1..4 (모듈 순서 아님).
바퀴 속도 [rad/s] = d(pos_ticks)/dt × 2π / ticks_per_rev.
"""
import argparse
import json
import os
import random
import re
import sys
import time
import warnings

import numpy as np

BAUD = 115200

D1_DUTIES = (40, 80, 120, 160, 200, 300, 400, 500, 600, 700, 799,
             -40, -80, -120, -160, -200, -400, -799)
D2_SPECS = ("0 200 100 700", "0 400 100 700", "0 799 100 700", "799 0 100 1200",
            "400 -400 100 1200", "0 -799 100 700")
D3_SPECS = ("0 150 2000 2000 RAMP", "150 0 2000 2000 RAMP", "0 -150 2000 2000 RAMP", "-150 0 2000 2000 RAMP")
REF_SPEC = "400 400 0 1000 DEC 5"
REF_EVERY = 10
STEER_ANGLES = (0, 90, 0, -90, 0, 45, 0, 15, 0, 5, 0, 3, 0)
SMALL_AMPS = (0.1, 0.2, 0.3, 0.5, 1.0, 1.5)     # 도.  1 스텝 = 0.088도
SMALL_MS = 600
STEP_DONE = ("읽기 ", "이미 목표 근처", "읽기 실패", "사용법", "텔레옵 중")   # STEP 출력의 마지막 줄 표식


class Link:
    """port 가 문자열이면 시리얼을 열고, 아니면 그 객체(시험용 가짜 펌웨어)를 그대로 쓴다."""

    def __init__(self, port, pause=1.0):
        if isinstance(port, str):
            try:
                import serial
            except ImportError:
                sys.exit("pyserial 필요: pip install pyserial")
            self.s = serial.Serial(port, BAUD, timeout=0.2)
            time.sleep(0.3)
        else:
            self.s = port
        self.s.reset_input_buffer()
        self.pause = pause          # 시험 사이 쉬는 시간 배율 (가짜 펌웨어 시험에서 0)
        if isinstance(port, str):
            self.quiet()

    def quiet(self):
        """측정 전 터미널 정리.
        - 빈 줄(\r) 하나: 포트를 열 때 들어온 쓰레기 바이트(0x00 등)를 명령 버퍼에서 밀어낸다.
          안 그러면 첫 명령이 '\0HELP' 가 돼 "알 수 없는 명령: ''" 로 버려진다.
        - 주기 엔코더 로그(LOG, 부팅 기본 ON, 10 Hz) 끄기: 켜져 있으면 CAP·STEP 출력 사이에 섞이고
          raw 가 출력이 멎기를 영영 못 기다린다.  LOG 는 토글이라 응답을 보고 ON 이면 한 번 더 보낸다."""
        self.send("")
        list(self.lines(0.3))
        for _ in range(2):
            out = self.collect("LOG", 1.0, until=("[LOG]",))
            st = [ln for ln in out if "[LOG]" in ln]
            if st and "OFF" in st[-1]:
                break
        else:
            print("  경고: LOG 끄기 확인 못 함 -- 터미널에서 LOG 를 직접 보내 OFF 로 만들 것", file=sys.stderr)
        time.sleep(0.15)
        self.s.reset_input_buffer()

    def rest(self, sec):
        time.sleep(sec * self.pause)

    def send(self, line):
        self.s.write((line + "\r").encode("ascii"))

    def lines(self, timeout, idle=None):
        """timeout 초 동안 받은 줄을 하나씩 돌려준다 (펌웨어 출력은 UTF-8 한글 섞임).
        idle 을 주면 무언가 받은 뒤 idle 초 동안 새 바이트가 없을 때 일찍 끝낸다."""
        end, buf, last = time.time() + timeout, b"", None
        while time.time() < end:
            chunk = self.s.read(4096)
            now = time.time()
            if chunk:
                last = now
            elif idle is not None and last is not None and now - last > idle:
                break
            buf += chunk
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                yield raw.decode("utf-8", "replace").strip("\r ")
        if buf:
            yield buf.decode("utf-8", "replace").strip("\r ")

    def collect(self, line, timeout=2.0, until=()):
        """명령 한 줄을 보내고 출력을 모은다. until 의 표식이 들어간 줄을 받으면 바로 끝낸다."""
        self.send(line)
        out = []
        for ln in self.lines(timeout):
            out.append(ln)
            if any(u in ln for u in until):
                break
        return out


def capture(link, spec, timeout=60.0, cmd="CAP"):
    """CAP 한 번. spec 예: 'ALL 0 400 100 600' / '1 0 150 2000 2000 RAMP'.
    cmd="WCAP" 면 바퀴 속도 PI 계단 (spec 의 w0 w1 은 rad/s): duty 대신 PI 주기마다 건 duty 를
    pi_t_ms (M,), pi_duty (M,4) 로 돌려준다."""
    tag = f"[{cmd}]"
    link.send(f"{cmd} {spec}")
    hdr, rows, drows, end = None, [], [], None
    for ln in link.lines(timeout):
        if ln.startswith(f"{tag} 사용법") or "텔레옵 중" in ln:
            raise RuntimeError(f"{cmd} 거부: {ln}  (명령: {cmd} {spec})")
        if ln.startswith(f"{tag} BEGIN"):
            hdr = dict(kv.split("=") for kv in ln.split()[2:])
            continue
        if hdr is not None and ln.startswith(f"{tag} END"):
            end = dict(kv.split("=") for kv in ln.split()[2:])
            break
        if hdr is not None and re.fullmatch(r"\d+(,-?\d+){4}", ln):
            rows.append([int(x) for x in ln.split(",")])
        elif hdr is not None and re.fullmatch(r"D,\d+(,-?\d+){4}", ln):
            drows.append([int(x) for x in ln.split(",")[1:]])
    if hdr is None or end is None:
        raise RuntimeError(f"{cmd} 응답 불완전 (BEGIN={hdr is not None}, 줄 {len(rows)})")
    h = {k: int(v) for k, v in hdr.items()}
    a = np.array(rows, dtype=np.int64)
    if len(a) != h["n"]:
        raise RuntimeError(f"{cmd} 줄 수 {len(a)} != n {h['n']} (UART 손실?)")
    t_s = a[:, 0] * 1e-6
    pos = np.cumsum(a[:, 1:], axis=0)
    if cmd == "WCAP":
        d = np.array(drows, dtype=np.int64).reshape(-1, 5)
        return dict(t_s=t_s, pos_ticks=pos, pi_t_ms=d[:, 0], pi_duty=d[:, 1:], late_max_us=int(end["late_max_us"]),
                    vs_dv=int(end.get("vs_dv", -1)), spec=spec, w0=h["w0_mrad"] / 1000, w1=h["w1_mrad"] / 1000, **h)
    dec_ms = h["dt_us"] // 1000
    duty = np.array([h["d0"]] + [cap_duty_at(h, k * dec_ms) for k in range(1, h["n"])])   # 샘플 k 직전에 걸려 있던 duty
    return dict(t_s=t_s, pos_ticks=pos, duty=duty, late_max_us=int(end["late_max_us"]),
                vs_dv=int(end.get("vs_dv", -1)), spec=spec, **h)


def cap_duty_at(h, t_ms):
    """main.c run_capture 와 같은 식 (정수 나눗셈 포함). t_ms 시각부터 걸리는 duty."""
    if t_ms >= h["t_sw"]:
        return h["d1"]
    return h["d0"] + int((h["d1"] - h["d0"]) * t_ms / h["t_sw"]) if h["ramp"] else h["d0"]


def wheel_speed(c, i0=0, i1=None):
    """[i0, i1) 구간 평균 바퀴 속도 rad/s (모터별)."""
    i1 = len(c["t_s"]) if i1 is None else i1
    dp = c["pos_ticks"][i1 - 1] - c["pos_ticks"][i0]
    dt = c["t_s"][i1 - 1] - c["t_s"][i0]
    return dp / max(dt, 1e-9) * 2 * np.pi / c["ticks_per_rev"]


def stamp_dir(out, name):
    path = os.path.join(out, time.strftime("%Y%m%d_%H%M%S_") + name)
    os.makedirs(path, exist_ok=True)
    return path


def save_single(out, name, **arrays):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, time.strftime("%Y%m%d_%H%M%S_") + name)
    np.savez(path + ".npz", **arrays)
    print(f"  저장 {path}.npz")


def run_cap(link, args):
    c = capture(link, " ".join(args.spec))
    save_single(args.out, "cap", **c)
    n = len(c["t_s"])
    print(f"  샘플 {n}, 늦음 최대 {c['late_max_us']} us, 서보 전압 {c['vs_dv'] / 10:.1f} V")
    print(f"  앞 10%  속도 rad/s: {np.round(wheel_speed(c, 0, max(n // 10, 2)), 3)}")
    print(f"  뒤 20%  속도 rad/s: {np.round(wheel_speed(c, n - max(n // 5, 2)), 3)}")


def drive_plan(reps, seed):
    """(group, spec, rep) 목록. 그룹 안에서 반복을 섞고, REF 를 처음·REF_EVERY 개마다·끝에 끼운다."""
    rng = random.Random(seed)
    groups = [("D1", [f"{d} {d} 0 1000 DEC 5" for d in D1_DUTIES]), ("D2", list(D2_SPECS)), ("D3", list(D3_SPECS))]
    tests = []
    for g, specs in groups:
        block = [(g, s, r) for r in range(reps) for s in specs]
        rng.shuffle(block)
        tests += block
    plan, ref_i = [("REF", REF_SPEC, 0)], 1
    for i, t in enumerate(tests):
        plan.append(t)
        if (i + 1) % REF_EVERY == 0:
            plan.append(("REF", REF_SPEC, ref_i))
            ref_i += 1
    plan.append(("REF", REF_SPEC, ref_i))
    return plan


def run_drive(link, args):
    """D1 duty 별 정상속도 / D2 계단·제동 / D3 저속 램프 를 각각 reps 번. 모터 4개 동시."""
    m = args.motors
    plan = drive_plan(args.reps, args.seed)
    d = stamp_dir(args.out, "drive" + ("" if m == "ALL" else f"_m{m}") + (f"_{args.tag}" if args.tag else ""))
    manifest = dict(kind="drive", motors=m, reps=args.reps, seed=args.seed, started=time.strftime("%F %T"), files=[])
    print(f"[drive] 시험 {len(plan)}개 (반복 {args.reps}, REF {sum(p[0] == 'REF' for p in plan)}개) → {d}")
    if args.warmup > 0:
        # 데워지는 동안 마찰이 줄어 속도가 1~2% 오른다 (10-13 첫 측정 REF +1.65%) → 기록 없이 duty 400 으로 먼저 돌린다
        n = int(np.ceil(args.warmup / 2.8))
        print(f"  예열 {args.warmup:.0f} s (duty 400, 기록 안 함)")
        for _ in range(n):
            capture(link, f"{m} 400 400 0 2000 DEC 50")
        manifest["warmup_s"] = args.warmup
    t0 = time.time()
    for order, (g, spec, rep) in enumerate(plan):
        c = capture(link, f"{m} {spec}")
        fn = f"{order:03d}_{g}_{spec.replace(' ', '_')}_r{rep}.npz"
        np.savez(os.path.join(d, fn), group=g, rep=rep, order=order, wall_s=time.time() - t0, **c)
        manifest["files"].append(fn)
        w = wheel_speed(c, len(c["t_s"]) // 2) if g in ("D1", "REF") else wheel_speed(c, len(c["t_s"]) - 100)
        print(f"  [{order + 1:3d}/{len(plan)}] {g:3s} r{rep} {spec:24s} 끝속도 "
              + " ".join(f"{x:+6.2f}" for x in w) + f"  늦음 {c['late_max_us']}us  {c['vs_dv'] / 10:.1f}V")
        link.rest(0.3 if g == "D1" or g == "REF" else 0.5)
    manifest["finished"] = time.strftime("%F %T")
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"완료 ({(time.time() - t0) / 60:.1f}분).  분석: python3 pi/common/sysid_analyze.py {d}")
    print("D5(하중)·D6(배터리)는 수동: D6 은 배터리 상태마다 drive 를 다시 돌려 디렉터리끼리 비교한다.")
    return d


# 바퀴 속도 PI 벤치 시험 (WCAP, rad/s).  설계 시뮬 기대값 (공칭 플랜트, firmware/MOTOR_MODELING_PLAN.md §8):
#   0->2 계단 90% 37 ms / 오버슈트 0.5% / 정상오차 <0.01,  0->0.3 저속 기동 29 ms
PI_SPECS = ("0 2 100 700", "0 0.3 100 1500", "0 5 100 700", "2 -2 100 900", "0 1 1000 1600 RAMP",
            "0 0.15 100 2000", "5 0 100 600")
PI_EXPECT = {"0 2 100 700": (37, 0.5), "0 0.3 100 1500": (29, None)}


def wcap_speed(c, win_ms=10):
    """1 ms 기록 → 가운데 정렬 win_ms 창 평균 속도 (N,4) rad/s.  1 ms 한 틱 = 0.75 rad/s 라 창 필요."""
    pos = c["pos_ticks"].astype(float)
    dec = c["dt_us"] / 1000
    k = max(1, int(round(win_ms / dec)))
    w = np.full_like(pos, np.nan)
    h = k // 2
    w[h:len(pos) - (k - h)] = (pos[k:] - pos[:-k]) / (k * dec * 1e-3) * 2 * np.pi / c["ticks_per_rev"]
    return w


def pi_metrics(c):
    """모터별 (90% 도달 ms, 오버슈트 %, 정상오차 rad/s, 정상 rms rad/s).  계단 크기 = w1 - w0."""
    w0, w1, t_sw = c["w0"], c["w1"], c["t_sw"]
    step = w1 - w0
    # 창 안 1틱(2π/8384/창)이 계단의 5% 이하가 되게: 2 rad/s 10 ms, 0.3 rad/s 50 ms (상한).
    # 저속 계단의 오버슈트·rms 는 창 양자화가 섞인 값이다.
    w = wcap_speed(c, win_ms=float(np.clip(15.0 / max(abs(step), 1e-3), 10, 50)))
    t_ms = c["t_s"] * 1e3
    after = t_ms >= t_sw
    steady = t_ms >= t_sw + 0.6 * (t_ms[-1] - t_sw)
    out = []
    for m in range(4):
        if not (c["motors"] >> m) & 1:
            out.append((np.nan,) * 4)
            continue
        x = w[:, m]
        prog = (x - w0) / step if step else np.zeros_like(x)
        hit = np.flatnonzero(after & (prog >= 0.9))
        t90 = (t_ms[hit[0]] - t_sw) if len(hit) and not c["ramp"] else np.nan
        ovs = (np.nanmax(prog[after]) - 1) * 100 if step and not c["ramp"] else np.nan
        e = w1 - x[steady]
        out.append((t90, ovs, np.nanmean(e), np.sqrt(np.nanmean(e ** 2))))
    return np.array(out)


def show_pi(c, label):
    m = pi_metrics(c)
    ex = PI_EXPECT.get(c["spec"].split(" ", 1)[1] if " " in c["spec"] else "", None)
    cols = " | ".join(f"M{i + 1} {r[0]:4.0f}ms {r[1]:+5.1f}% e{r[2]:+.3f} rms{r[3]:.3f}" for i, r in enumerate(m)
                      if not np.isnan(r[2]))
    exs = f"  (설계 {ex[0]} ms" + (f", {ex[1]}%" if ex[1] is not None else "") + ")" if ex else ""
    print(f"  {label:28s} {cols}{exs}", flush=True)


def run_wcap(link, args):
    c = capture(link, " ".join(args.spec), cmd="WCAP")
    save_single(args.out, "wcap", **c)
    show_pi(c, "WCAP " + " ".join(args.spec))


def run_pi(link, args):
    """바퀴 속도 PI 벤치 검증: PI_SPECS 를 reps 번 (순서 섞음), 모터 --motors.  바퀴 띄우고."""
    m = args.motors
    rng = random.Random(args.seed)
    plan = [(s, r) for r in range(args.reps) for s in PI_SPECS]
    rng.shuffle(plan)
    d = stamp_dir(args.out, "pi" + ("" if m == "ALL" else f"_m{m}") + (f"_{args.tag}" if args.tag else ""))
    manifest = dict(kind="pi", motors=m, reps=args.reps, seed=args.seed, started=time.strftime("%F %T"), files=[],
                    wpi=link.collect("WPI", 1.0, until=("[WPI]",)))
    print(f"[pi] 시험 {len(plan)}개 → {d}\n  {manifest['wpi'][-1] if manifest['wpi'] else '(WPI 응답 없음)'}")
    for order, (spec, rep) in enumerate(plan):
        c = capture(link, f"{m} {spec}", cmd="WCAP")
        fn = f"{order:03d}_{spec.replace(' ', '_')}_r{rep}.npz"
        np.savez(os.path.join(d, fn), rep=rep, order=order, **c)
        manifest["files"].append(fn)
        show_pi(c, f"[{order + 1:2d}/{len(plan)}] {spec} r{rep}")
        link.rest(0.5)
    manifest["finished"] = time.strftime("%F %T")
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print("요약 (시험별 모터·반복 평균):")
    for spec in PI_SPECS:
        ms = [pi_metrics(dict(np.load(os.path.join(d, fn)))) for fn in manifest["files"]
              if fn.split("_", 1)[1].rsplit("_r", 1)[0] == spec.replace(" ", "_")]
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)      # 램프는 90%·오버슈트가 nan
            a = np.nanmean(np.array(ms), axis=(0, 1)) if ms else [np.nan] * 4
        ex = PI_EXPECT.get(spec)
        print(f"  {spec:20s} 90% {a[0]:5.0f} ms  오버슈트 {a[1]:+5.1f}%  정상오차 {a[2]:+.3f}  rms {a[3]:.3f}"
              + (f"   설계 {ex[0]} ms" + (f" / {ex[1]}%" if ex[1] is not None else "") if ex else ""))
    return d


def icap_read(link, timeout=60.0):
    """ICAP DUMP 한 번 → dict(t_s, gyro (N,3) rad/s, acc (N,3) m/s², 로버 좌표계, gaps) 또는 거부 줄."""
    link.send("ICAP DUMP")
    hdr, rows, end = None, [], None
    for ln in link.lines(timeout):
        if ln.startswith("[ICAP]") and ("아직" in ln or "주행 명령" in ln):
            return ln
        if ln.startswith("[ICAP] BEGIN"):
            hdr = dict(kv.split("=") for kv in ln.split()[2:])
            continue
        if hdr is not None and ln.startswith("[ICAP] END"):
            end = dict(kv.split("=") for kv in ln.split()[2:])
            break
        if hdr is not None and re.fullmatch(r"\d+(,-?\d+){6}", ln):
            rows.append([int(x) for x in ln.split(",")])
    if hdr is None or end is None or len(rows) != int(hdr["n"]):
        raise RuntimeError(f"ICAP 덤프 불완전 (BEGIN={hdr is not None}, 줄 {len(rows)}/{hdr and hdr['n']})")
    import nucleo_link as nl
    a = np.array(rows, dtype=np.int64).reshape(-1, 7)
    body = lambda xyz: np.stack([sg * xyz[:, i] for i, sg in nl.IMU_CHIP_TO_BODY], 1)
    gyro = body(a[:, 1:4] * float(hdr["gyro_dps_lsb"])) * np.pi / 180
    acc = body(a[:, 4:7] * float(hdr["acc_g_lsb"])) * 9.81
    return dict(t_s=a[:, 0] * 1e-6, gyro=gyro, acc=acc, gaps=int(end["gaps"]))


def imu_report(c, ref_deg=None, still_s=1.0):
    """진동·필터 요약.  앞 still_s 초는 정지라고 보고 자이로 바이어스·기준 기울기를 잡는다."""
    import sensor_fusion as sf
    t, g, f = c["t_s"], c["gyro"], c["acc"]
    n = len(t)
    print(f"  샘플 {n}, {n / max(t[-1] - t[0], 1e-9):.0f} Hz, 7 ms 넘는 빈틈 {c['gaps']}")
    st = t - t[0] < still_s
    bias = g[st].mean(0)
    tilt = lambda a: np.degrees(np.arccos(np.clip(a[:, 2] / np.linalg.norm(a, axis=1), -1, 1)))   # a = 비력 (정지 시 위)
    print(f"  정지 {still_s:.0f} s: 자이로 바이어스 {np.round(bias, 4)} rad/s (σ {np.round(g[st].std(0), 4)}), "
          f"|f| {np.linalg.norm(f[st], axis=1).mean():.3f} m/s², 기울기 {tilt(f[st]).mean():.2f}°"
          + (f"  (기준 {ref_deg}°, 차이 {tilt(f[st]).mean() - ref_deg:+.2f}°)" if ref_deg is not None else ""))
    # 20 ms 칸으로 나눠 순간값(칸 마지막 샘플) vs 칸 평균
    k = np.floor((t - t[0]) / 0.02).astype(int)
    idx = [np.flatnonzero(k == j) for j in range(k.max() + 1)]
    idx = [i for i in idx if len(i)]
    gp, gm = np.array([g[i[-1]] for i in idx]) - bias, np.array([g[i].mean(0) for i in idx]) - bias
    fp, fm = np.array([f[i[-1]] for i in idx]), np.array([f[i].mean(0) for i in idx])
    cnt = np.array([len(i) for i in idx])
    q = lambda x: f"평균 {x.mean():.3f} p95 {np.quantile(x, .95):.3f} 최대 {x.max():.3f}"
    print(f"  20 ms 칸 샘플 수 {cnt.mean():.2f} (펌웨어 imu_n 과 같아야 함)")
    print(f"  자이로 순간−평균 [rad/s]   {q(np.linalg.norm(gp - gm, axis=1))}")
    print(f"  가속 순간−평균 [m/s²]      {q(np.linalg.norm(fp - fm, axis=1))}")
    print(f"  | |f|−9.81 | 순간 [m/s²]   {q(np.abs(np.linalg.norm(fp, axis=1) - 9.81))}")
    # 보완 필터: 순간값 입력 vs 칸 평균 입력 (참값이 없으니 둘의 차이와, 정지 구간 기준 대비)
    outs = {}
    for name, (gg, ff) in (("순간값", (gp, fp)), ("20ms 평균", (gm, fm))):
        flt = sf.GravityFilter(0.02)
        outs[name] = np.array([flt.step(gg[j], ff[j]) for j in range(len(gg))])
    d = np.degrees(np.arccos(np.clip((outs["순간값"] * outs["20ms 평균"]).sum(1), -1, 1)))
    print(f"  필터 출력 차이 (순간값 vs 평균 입력) [deg] {q(d)}")
    for name, o in outs.items():
        print(f"  필터({name}) 기울기 [deg]   {q(tilt(-o))}")
    return outs


def run_imu(link, args):
    """IMU 208 Hz 원시 기록 (ICAP).  정지 기울기 시험(--ref-deg) 또는 주행 중 진동.
    기록하는 동안 로버는 Pi(SPI)·조이스틱으로 몬다. 끝나면 멈춰야 덤프된다."""
    d = stamp_dir(args.out, "imu" + (f"_{args.tag}" if args.tag else ""))
    print(f"[imu] {args.ms} ms 기록 시작 → {d}" + ("" if args.ref_deg is not None else "   (지금 주행: 처음 1 s 는 정지)"))
    print("  " + " / ".join(link.collect(f"ICAP {args.ms}", 1.0, until=("[ICAP]",))))
    link.rest(args.ms / 1000 + 0.5)
    t0 = time.time()
    while True:
        c = icap_read(link)
        if not isinstance(c, str):
            break
        if time.time() - t0 > 120:
            raise RuntimeError(f"ICAP 덤프 못 받음: {c}")
        print(f"  {c} -- 2 s 뒤 다시", flush=True)
        link.rest(2)
    np.savez(os.path.join(d, "imu.npz"), ref_deg=np.nan if args.ref_deg is None else args.ref_deg, **c)
    imu_report(c, args.ref_deg)
    print(f"  저장 {d}/imu.npz")
    return d


def run_steer(link, args):
    """S1 큰·중간 계단 (서보 최고속 0 = 최대) 을 reps 번. 가속도 50(부팅 기본)과 0 을 ACC 로 바꿔 가며.
    ACC 순서는 반복마다 번갈아 (50,0) / (0,50) -- 서보 발열·배터리 표류가 한쪽에 몰리지 않게.
    2도 미만 작은 계단(S2 데드밴드·S3 히스테리시스)은 steer-small 이 따로 잰다."""
    n = args.servo
    d = stamp_dir(args.out, f"steer_servo{n}" + (f"_{args.tag}" if args.tag else ""))
    text = [f"### META servo={n} reps={args.reps} accs={args.accs} started={time.strftime('%F %T')}"]
    text += link.collect("S 0", 0.5)
    t0 = time.time()
    for rep in range(args.reps):
        for acc in (args.accs if rep % 2 == 0 else args.accs[::-1]):
            text += [f"### ACC {acc}"] + link.collect(f"ACC {acc}", 1.0, until=(f"ID={4} ",))
            for ang in STEER_ANGLES:
                out = link.collect(f"STEP {n} {ang} RAW", 15.0, until=STEP_DONE)
                text += [f"### ACC {acc} STEP {n} {ang} rep {rep}"] + out
                link.rest(0.2)
        print(f"  반복 {rep + 1}/{args.reps} 끝 ({time.time() - t0:.0f} s)")
    text += link.collect("ACC 50", 1.0, until=(f"ID={4} ",))     # 기본으로 되돌림
    with open(os.path.join(d, "steer.txt"), "w") as f:
        f.write("\n".join(text) + "\n")
    print(f"완료.  분석: python3 pi/common/sysid_analyze.py {d}")
    return d


def show_small(out, target, rep, role):
    """steer-small 진행 표시: 명령마다 한 줄 (시작 → 최종, 움직였는지). 시작각은 STEPRAW 의 0.01도 값."""
    t = "\n".join(out)
    m0 = re.search(r"start_cdeg=(-?\d+)", t)
    m1 = re.search(r"최종 ([+-]?\d+\.\d+)도", t)
    if not m1:
        print(f"  r{rep} {role:8s} 명령 {target:+6.2f}°  ?? 응답 없음: {out[-1] if out else ''}")
        return
    fin = float(m1[1])
    st = f"시작 {int(m0[1]) / 100:+6.2f} → " if m0 else ""
    mv = f"(이동 {fin - int(m0[1]) / 100:+5.2f}°)" if m0 else ""
    flag = "안 움직임" if "안 움직임" in t else ("이미 목표 근처" if "이미 목표 근처" in t else "")
    print(f"  r{rep} {role:8s} 명령 {target:+6.2f}°  {st}최종 {fin:+6.2f}°  {mv}  {flag}", flush=True)


def run_steer_small(link, args):
    """S2 데드밴드: 0 에서 ±A (SMALL_AMPS) 로 갔다 오며 움직였는지·어디서 멈췄는지 (RAW, SMALL_MS 고정 기록).
    S3 히스테리시스: 0 에 -5 쪽과 +5 쪽에서 다가갔을 때 멈춘 위치 차이 (서보 데드존 / 기어 놀음 중 서보 출력축에서 보이는 것).
    서보 엔코더는 출력축에 있으므로 서보 바깥(커플링·조향 링크)의 놀음은 여기서 안 보인다.
    가속도는 지금 운용값(--acc, 기본 50) 그대로."""
    n = args.servo
    d = stamp_dir(args.out, f"steersmall_servo{n}" + (f"_{args.tag}" if args.tag else ""))
    text = [f"### META servo={n} reps={args.reps} kind=small started={time.strftime('%F %T')}"]
    text += link.collect("S 0", 0.5)
    text += link.collect(f"ACC {args.acc}", 1.0, until=(f"ID={4} ",))
    rng = random.Random(args.seed)
    t0 = time.time()

    def step(target, rep, role, amp, raw=True):
        cmd = f"STEP {n} {target:g}" + (f" RAW MS {SMALL_MS}" if raw else "")
        out = link.collect(cmd, 15.0, until=STEP_DONE)
        show_small(out, target, rep, role)
        return [f"### ACC {args.acc} STEP {n} {target:g} rep {rep} role={role} amp={amp:g}"] + out

    for rep in range(args.reps):
        for side, role in ((-5, "hyst_lo"), (5, "hyst_hi")) if rep % 2 == 0 else ((5, "hyst_hi"), (-5, "hyst_lo")):
            text += step(side, rep, "pre", 5, raw=False)
            text += step(0, rep, role, 5)
        amps = list(SMALL_AMPS)
        rng.shuffle(amps)
        for a in amps:
            for target, role in ((a, "out"), (0, "back"), (-a, "out"), (0, "back")):
                text += step(target, rep, role, a)
            link.rest(0.1)
        print(f"  반복 {rep + 1}/{args.reps} 끝 ({time.time() - t0:.0f} s)")
    text += link.collect("ACC 50", 1.0, until=(f"ID={4} ",))
    with open(os.path.join(d, "steer.txt"), "w") as f:
        f.write("\n".join(text) + "\n")
    print(f"완료.  분석: python3 pi/common/sysid_analyze.py {d}")
    return d


def run_raw(link, args):
    """한 줄 보내고 출력이 --idle 초 멎을 때까지 (최대 --wait 초) 받는 대로 찍는다. STEP RAW 덤프는 ~4 s 걸린다."""
    link.send(" ".join(args.line))
    for ln in link.lines(args.wait, idle=args.idle):
        print(ln, flush=True)


def motor_set(s):
    """--motors: ALL 또는 1~4 숫자를 붙여 쓴 것 (펌웨어 CAP 과 같은 형식)."""
    s = s.upper()
    if s == "ALL" or (s and set(s) <= set("1234") and len(set(s)) == len(s)):
        return s if s == "ALL" else "".join(sorted(s))
    raise argparse.ArgumentTypeError(f"ALL 또는 1~4 숫자 (예: 234): {s}")


def main(argv=None, port=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyACM0", help="ST-Link 가상 COM (기본 %(default)s)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sysid_logs"),
                    help="저장 위치 (기본 저장소 최상위 sysid_logs/ -- 어느 폴더에서 실행해도 같은 곳)")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="{drive,steer,steer-small,cap,pi,wcap,imu,raw}")
    rep = argparse.ArgumentParser(add_help=False)         # drive / steer / steer-small 공통
    rep.add_argument("--reps", type=int, default=5, help="같은 시험 반복 횟수 (기본 %(default)s, 신뢰구간용 최소 3)")
    rep.add_argument("--tag", default="", help="디렉터리 이름 꼬리표 (예: ground, batt_low)")
    servo = argparse.ArgumentParser(add_help=False)
    servo.add_argument("--servo", type=int, default=1, choices=(1, 2, 3, 4), help="조향 서보 번호 (기본 %(default)s)")

    p = sub.add_parser("drive", parents=[rep], help="구동모터 D1 정상속도 / D2 계단·제동 / D3 저속 램프 (~12분)")
    p.add_argument("--motors", default="ALL", type=motor_set,
                   help="ALL = 4개 동시, 숫자를 붙여 쓰면 그 모터들만 (예: 234, 1). 디렉터리 이름에 _m234. 기본 %(default)s")
    p.add_argument("--seed", type=int, default=0, help="반복 순서 섞기 시드")
    p.add_argument("--warmup", type=float, default=60, help="측정 전 예열 초 (duty 400, 기록 안 함). 0 = 안 함. 기본 %(default)s")
    p.set_defaults(fn=run_drive)
    p = sub.add_parser("steer", parents=[rep, servo], help="조향 S1 큰 계단, ACC 50 과 0, 1 ms 궤적 (~8분)")
    p.add_argument("--accs", default="50,0", type=lambda x: [int(v) for v in x.split(",")],
                   help="번갈아 잴 가속도 레지스터 값 (쉼표, 0-254). 기본 %(default)s")
    p.set_defaults(fn=run_steer)
    p = sub.add_parser("steer-small", parents=[rep, servo], help="조향 S2 데드밴드 / S3 히스테리시스 (~5분)")
    p.add_argument("--acc", type=int, default=50, help="서보 가속도 레지스터 (기본 %(default)s = 운용값)")
    p.add_argument("--seed", type=int, default=0, help="계단 크기 순서 섞기 시드")
    p.set_defaults(fn=run_steer_small)
    p = sub.add_parser("cap", help="CAP 한 번 → npz + 요약.  예: cap ALL 0 400 100 600")
    p.add_argument("spec", nargs="+", help="<1-4|ALL> <d0> <d1> <t_sw> <t_end> [RAMP] [DEC n]  (firmware/CLI.md)")
    p.set_defaults(fn=run_cap)
    p = sub.add_parser("pi", parents=[rep], help="바퀴 속도 PI 벤치 검증 (WCAP 계단·램프·저속, ~2분)")
    p.add_argument("--motors", default="ALL", type=motor_set, help="ALL 또는 숫자 조합 (기본 %(default)s)")
    p.add_argument("--seed", type=int, default=0, help="순서 섞기 시드")
    p.set_defaults(fn=run_pi)
    p = sub.add_parser("wcap", help="WCAP 한 번 → npz + 요약.  예: wcap ALL 0 2 100 600")
    p.add_argument("spec", nargs="+", help="<1-4|ALL> <w0> <w1> <t_sw> <t_end> [RAMP] [DEC n]  rad/s (firmware/CLI.md)")
    p.set_defaults(fn=run_wcap)
    p = sub.add_parser("imu", help="IMU 208 Hz 원시 기록 (ICAP) → 진동·필터 요약.  정지 기울기 / 주행 중")
    p.add_argument("--ms", type=int, default=9000, help="기록 길이 (최대 ~9600, 기본 %(default)s)")
    p.add_argument("--ref-deg", type=float, default=None, help="정지 기울기 시험: 경사계로 잰 기울기 [deg]")
    p.add_argument("--tag", default="", help="디렉터리 꼬리표 (예: flat, tilt10, bumps)")
    p.set_defaults(fn=run_imu)
    p = sub.add_parser("raw", help="펌웨어 명령 한 줄 보내고 출력 표시.  예: raw HELP / raw STEP 1 30 RAW")
    p.add_argument("line", nargs="+", help="보낼 명령 (공백으로 이어 붙인다)")
    p.add_argument("--idle", type=float, default=2.0, help="출력이 이만큼(초) 멎으면 끝 (기본 %(default)s)")
    p.add_argument("--wait", type=float, default=30.0, help="최대 대기 초 (기본 %(default)s)")
    p.set_defaults(fn=run_raw)
    args = ap.parse_args(argv)
    link = Link(port if port is not None else args.port, pause=0.0 if port is not None else 1.0)
    try:
        return args.fn(link, args)
    finally:
        link.send("CAP ALL 0 0 0 1")   # 혹시 모를 구동을 0 으로 (짧은 캡처, 출력은 버림)
        link.s.close()


if __name__ == "__main__":
    main()
