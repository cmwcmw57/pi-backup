"""sysid_uart.py(측정) + sysid_analyze.py(분석·판정) 를 가짜 펌웨어로 끝까지 돌려 검증한다.

가짜 펌웨어 FakeRover 는 main.c 의 CAP·STEP·ACC 출력 형식과 duty 전환 시각을 그대로 흉내 내고,
안쪽은 정답 파라미터를 아는 모터(1차 + 지연 + 데드밴드, 1틱 양자화) · 서보(사다리꼴) 모델이다.
  A. 정상: 분석이 정답 파라미터를 허용오차 안에서 되찾고 모든 확인이 PASS 인가
  B. 배터리 표류 5%: 표류 확인이 FAIL 로 잡는가
  C. 제동 시정수 3배 + 정지마찰: 모델 검증이 제동·저속 램프에서만 FAIL 로 잡는가 (거짓 PASS 방지)
  조향: 사다리꼴·시작 지연을 되찾는가 / D. 작은 계단: 데드존 0 과 0.25도를 구별하는가

    python3 pi/common/test_sysid.py        (~2분, 하드웨어 없이)
"""
import io
import math
import re
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sim"))   # ksrc_common (wheel_pi.c, sim/build.sh)
import sysid_analyze as sa  # noqa: E402
import sysid_uart as su  # noqa: E402

TPR = 8384
PWM_MAX = 799


class FakeRover:
    def __init__(self, seed=0, w0=(7.90, 8.15, 7.70, 8.05), rev=0.98, dead=(32, 38, 45, 35), tau=0.035,
                 tau_brake=None, lag=0.004, stiction=0, drift=0.0, servo_v=330.0, servo_lag=0.005, servo_db=0.0, servo_ring=0.0):
        self.rng = np.random.default_rng(seed)
        self.w0, self.rev, self.dead = np.array(w0), rev, np.array(dead, float)
        self.tau, self.tau_brake, self.lag = tau, tau_brake or tau, lag
        self.stiction, self.drift = stiction, drift
        self.servo_v, self.servo_lag, self.servo_db = servo_v, servo_lag, servo_db
        self.servo_ring = servo_ring             # 도착 후 감쇠 진동 진폭 (도) -- 사다리꼴 모델에 없는 항
        self.acc_reg, self.servo_ang = 50, np.zeros(4)
        self.n_caps, self.inbuf, self.outbuf = 0, b"", b""

    # ---- 시리얼 흉내
    def reset_input_buffer(self):
        self.outbuf = b""

    def close(self):
        pass

    def read(self, n):
        out, self.outbuf = self.outbuf[:n], self.outbuf[n:]
        return out

    def write(self, data):
        self.inbuf += data
        while b"\r" in self.inbuf:
            line, self.inbuf = self.inbuf.split(b"\r", 1)
            for s in self.handle(line.decode().strip()):
                self.outbuf += (s + "\r\n").encode()

    def handle(self, cmd):
        if cmd.startswith("WCAP"):
            return self.wcap(cmd[4:].split())
        if cmd.startswith("ICAP"):
            return self.icap(cmd[4:].strip())
        if cmd.startswith("WPI"):
            return ["[WPI] OFF (개루프 duty)  kp 82.5 ki 2062 (가짜)"]
        if cmd.startswith("CAP"):
            return self.cap(cmd[3:].split())
        if cmd.startswith("STEP"):
            a = cmd.split()
            ms = int(a[a.index("MS") + 1]) if "MS" in a else 0
            return self.step(int(a[1]) - 1, float(a[2]), "RAW" in a, ms)
        if cmd.startswith("ACC"):
            self.acc_reg = int(cmd.split()[1])
            return [f"[ACC] ID={i} 가속도 {self.acc_reg} OK" for i in range(1, 5)]
        return []

    # ---- 구동모터
    def w_ss(self, d, gain):
        ad = np.abs(d)
        w = np.where(ad > self.dead, (ad - self.dead) / (PWM_MAX - self.dead), 0.0) * self.w0 * gain
        return np.where(d < 0, -w * self.rev, w)

    def cap(self, a):
        mask = 0b1111 if a[0] == "ALL" else sum(1 << (int(c) - 1) for c in set(a[0]))
        d0, d1, t_sw, t_end = (int(x) for x in a[1:5])
        ramp = "RAMP" in a
        dec = int(a[a.index("DEC") + 1]) if "DEC" in a else 1
        n = t_end // dec
        h = dict(d0=d0, d1=d1, t_sw=t_sw, ramp=int(ramp))
        gain = (1 - self.drift * min(self.n_caps / 90, 1)) * (1 + 0.001 * self.rng.standard_normal())
        self.n_caps += 1
        sel = np.array([(mask >> i) & 1 for i in range(4)], bool)
        hist = [(-1e9, 0)]                     # (적용 시각 s, duty)
        st = dict(w=np.zeros(4), th=np.zeros(4), moving=np.zeros(4, bool), t=-0.8 if d0 else 0.0)
        if d0:
            hist.append((-0.8, d0))

        def run_to(t_end_s):
            dt = 2.5e-4
            while st["t"] < t_end_s - 1e-12:
                h_ = min(dt, t_end_s - st["t"])
                te = st["t"] + h_ / 2 - self.lag
                d = next(dd for tt, dd in reversed(hist) if tt <= te)
                dv = np.where(sel, d, 0).astype(float)
                ws = self.w_ss(dv, gain)
                start = np.abs(dv) > self.dead + self.stiction
                st["moving"] |= start
                tau = self.tau_brake if d == 0 else self.tau
                w = st["w"] + (ws - st["w"]) * (1 - math.exp(-h_ / tau))
                w = np.where(st["moving"], w, 0.0)
                stop = st["moving"] & (ws == 0) & (np.abs(w) < 0.05)
                st["moving"] &= ~stop
                w = np.where(stop, 0.0, w)
                st["th"] += 0.5 * (st["w"] + w) * h_ * (1 + 0.01 * math.sin(st["t"] * 37))   # 기어 리플
                st["w"] = w
                st["t"] += h_

        run_to(0.0)
        ticks = lambda: np.floor(st["th"] * TPR / (2 * np.pi)).astype(int)
        last, rows, late, cur = ticks(), [], 0, d0
        for k in range(n):
            jit = int(self.rng.integers(0, 6))
            late = max(late, jit)
            t_s = k * dec / 1000 + jit * 1e-6
            run_to(t_s)
            c = ticks()
            dp = np.where(sel, c - last, 0)
            last = c
            rows.append(f"{int(round(t_s * 1e6))}," + ",".join(str(int(x)) for x in dp))
            want = su.cap_duty_at(h, (k + 1) * dec)
            if want != cur:
                hist.append((st["t"], want))
                cur = want
        vs = int(round(118 * gain))
        return ([f"[CAP] BEGIN motors={mask} d0={d0} d1={d1} t_sw={t_sw} t_end={t_end} ramp={int(ramp)} "
                 f"dt_us={dec * 1000} n={n} ticks_per_rev={TPR}"] + rows + [f"[CAP] END late_max_us={late} vs_dv={vs}"])

    # ---- 바퀴 속도 PI 캡처 (main.c run_wcap).  PI 는 sim/libksrc_common.so 의 진짜 wheel_pi.c,
    # 플랜트는 실측 모델 (test_wheel_pi.c 와 같은 식: 1차 τ 40 ms + Stribeck 마찰 + 정지마찰 47 duty).
    def wcap(self, a):
        from ksrc_common import WheelPi
        mask = 0b1111 if a[0] == "ALL" else sum(1 << (int(c) - 1) for c in set(a[0]))
        w0, w1 = float(a[1]), float(a[2])
        t_sw, t_end = int(a[3]), int(a[4])
        ramp = "RAMP" in a
        dec = int(a[a.index("DEC") + 1]) if "DEC" in a else 1
        n, k_ = t_end // dec, 0.0097
        sel = np.array([(mask >> i) & 1 for i in range(4)], bool)
        pis = [WheelPi() for _ in range(4)]
        st = dict(w=np.zeros(4), th=np.zeros(4), stuck=np.ones(4, bool), duty=np.zeros(4))
        last_pi = np.zeros(4, int)
        ticks = lambda: np.floor(st["th"] * TPR / (2 * np.pi)).astype(int)

        def plant_ms():
            for _ in range(4):                    # 0.25 ms 적분
                dt = 2.5e-4
                net = st["duty"]
                move = ~st["stuck"] | (np.abs(net) > 47)
                st["stuck"] &= ~move
                sg = np.where(st["w"] != 0, np.sign(st["w"]), np.sign(net))
                fr = 5 + 41 * np.exp(-np.abs(st["w"]) / 0.13)
                wn = st["w"] + (k_ * (net - sg * fr) - st["w"]) / 0.040 * dt
                cross = (st["w"] != 0) & (np.sign(wn) != np.sign(st["w"])) | (st["w"] == 0) & (np.sign(wn) != sg)
                wn = np.where(cross, 0.0, wn)
                st["stuck"] |= cross & (np.abs(net) <= 47)
                st["w"] = np.where(move, wn, 0.0)
                st["th"] += st["w"] * dt

        def pi_tick(w_ref):
            c = ticks()
            wm = (c - last_pi) / TPR * 2 * np.pi / 0.020
            last_pi[:] = c
            st["duty"] = np.array([pis[i].step(w_ref, wm[i]) if sel[i] else 0 for i in range(4)], float)

        if w0:
            for k in range(800):
                if k % 20 == 0:
                    pi_tick(w0)
                plant_ms()
        last, rows, drows = ticks(), [], []
        for k in range(n):
            c = ticks()
            rows.append(f"{k * dec * 1000}," + ",".join(str(int(x)) for x in np.where(sel, c - last, 0)))
            last = c
            t_ms = k * dec
            if t_ms % 20 == 0:
                w = w1 if t_ms >= t_sw else (w0 + (w1 - w0) * t_ms / t_sw if ramp else w0)
                pi_tick(w)
                drows.append(f"D,{t_ms}," + ",".join(str(int(x)) for x in st["duty"]))
            for _ in range(dec):
                plant_ms()
        return ([f"[WCAP] BEGIN motors={mask} w0_mrad={round(w0 * 1000)} w1_mrad={round(w1 * 1000)} t_sw={t_sw} "
                 f"t_end={t_end} ramp={int(ramp)} dt_us={dec * 1000} n={n} ticks_per_rev={TPR} pi_ms=20"]
                + rows + drows + ["[WCAP] END late_max_us=0 vs_dv=118"])

    # ---- IMU 고속 기록 (main.c run_icap). 정지 기울기 icap_tilt_deg (roll), 1 s 뒤부터 진동 icap_vib m/s²
    icap_tilt_deg, icap_vib, icap_ms = 10.0, 1.0, 0

    def icap(self, arg):
        if arg != "DUMP":
            self.icap_ms = int(arg or 9000)
            return [f"[ICAP] 기록 시작 {self.icap_ms} ms"]
        n = min(int(self.icap_ms * 0.208), 2000)
        t = np.arange(n) / 208.0
        r = math.radians(self.icap_tilt_deg)
        acc_b = np.tile([0.0, 9.81 * math.sin(r), 9.81 * math.cos(r)], (n, 1))      # roll: 몸체 y 쪽으로 중력
        vib = (t >= 1.0)[:, None] * self.rng.normal(0, self.icap_vib, (n, 3))
        acc_b = acc_b + vib
        gyro_b = 0.002 + self.rng.normal(0, 0.003, (n, 3)) + (t >= 1.0)[:, None] * self.rng.normal(0, 0.1, (n, 3))
        chip = lambda b: np.stack([-b[:, 0], -b[:, 1], b[:, 2]], 1)              # IMU_CHIP_TO_BODY 의 역
        ga = np.round(chip(gyro_b) * 180 / math.pi / 0.0175).astype(int)
        aa = np.round(chip(acc_b) / 9.81 / 0.000122).astype(int)
        rows = [f"{int(round(t[k] * 1e6))}," + ",".join(str(x) for x in (*ga[k], *aa[k])) for k in range(n)]
        return ([f"[ICAP] BEGIN n={n} odr_hz=208 gyro_dps_lsb=0.0175 acc_g_lsb=0.000122"] + rows + ["[ICAP] END gaps=0"])

    # ---- 조향 서보 (main.c steer_step_response 의 지표 계산을 그대로)
    def step(self, i, target, raw=False, ms=0):
        """main.c steer_step_response 흉내.  서보 모델: 지연 servo_lag 뒤 사다리꼴(가속 = ACC 레지스터, 최고속 servo_v),
        목표에서 servo_db 안에 들어오면 멈춤 (데드존 → 작은 명령 무시, 다가온 방향 쪽으로 덜 감)."""
        q = 360 / 4096
        start = float(self.servo_ang[i])
        delta = target - start
        small = abs(delta) < 2.0
        if (small and not raw) or abs(delta) < 0.08:
            return [f"[STEP] 이미 목표 근처 (현재 {start:+.1f}도). 다른 각도로 시험" + ("" if raw else " (2도 미만은 RAW 로)")]
        a = 1e5 if self.acc_reg == 0 else self.acc_reg * 100 * 360 / 4096
        v = self.servo_v * (1 + 0.01 * self.rng.standard_normal())
        th = max(abs(delta) - self.servo_db, 0.0)        # 데드존: 목표 db 안이면 멈춤
        sg = 1 if delta > 0 else -1

        def prof(t):                           # 시작 후 t 초의 (이동량, 속도), 지연 포함
            s_ = t - self.servo_lag
            if s_ <= 0 or th == 0:
                return 0.0, 0.0
            ta = v / a
            if a * ta * ta >= th:
                tp = math.sqrt(th / a)
                if s_ < tp:
                    return 0.5 * a * s_ * s_, a * s_
                if s_ < 2 * tp:
                    r = 2 * tp - s_
                    return th - 0.5 * a * r * r, a * r
                return th, 0.0
            xa = 0.5 * a * ta * ta
            T = 2 * ta + (th - 2 * xa) / v
            if s_ < ta:
                return 0.5 * a * s_ * s_, a * s_
            if s_ < T - ta:
                return xa + v * (s_ - ta), v
            if s_ < T:
                r = T - s_
                return th - 0.5 * a * r * r, a * r
            return th, 0.0

        fixed = raw and (small or ms > 0)
        ta_ = v / a
        t_end = (2 * math.sqrt(th / a) if a * ta_ * ta_ >= th else 2 * ta_ + (th - a * ta_ * ta_) / v) + self.servo_lag
        prof0 = prof

        def prof(t):
            x, vel = prof0(t)
            if self.servo_ring and th > 2 and t > t_end:
                u = t - t_end
                x += self.servo_ring * math.exp(-u / 0.08) * math.sin(2 * math.pi * 6 * u)
            return x, vel

        t_limit = (ms or 800) if fixed else 3000
        move_th = min(0.5, 0.5 * abs(delta))
        t10 = t90 = t_settle = band_since = t_move = -1.0
        peak_reg = peak_pos = overshoot = 0.0
        d_t, d_a, last, n, t = 0.0, start, start, 0, 0.0
        rawrows = []
        while True:
            t += 0.7 + 0.05 * self.rng.random()
            if t > t_limit:
                break
            x, vel = prof(t / 1000)
            ang = round((start + sg * x) / q) * q
            n += 1
            last = ang
            if raw and len(rawrows) < 2000 and t >= len(rawrows):
                rawrows.append(f"{int(t * 1000)},{int(round(ang * 100))},{int(round(sg * vel))},{int(sg * 30)}")
            prog = (ang - start) / delta
            if t_move < 0 and abs(ang - start) >= move_th:
                t_move = t
            if t10 < 0 and prog >= 0.1:
                t10 = t
            if t90 < 0 and prog >= 0.9:
                t90 = t
            peak_reg = max(peak_reg, round(vel / q) * q)
            if t - d_t >= 10:
                peak_pos = max(peak_pos, abs(ang - d_a) / ((t - d_t) / 1000))
                d_t, d_a = t, ang
            overshoot = max(overshoot, (prog - 1) * abs(delta))
            if fixed:
                continue
            if abs(ang - target) <= 1.0:
                if band_since < 0:
                    band_since = t
                elif t - band_since >= 100:
                    t_settle = band_since
                    break
            else:
                band_since = -1.0
        self.servo_ang[i] = last
        out = [f"[STEP] 서보{i + 1} {start:+.1f} -> {target:+.1f}도 (변화 {delta:+.1f}), 속도설정 S 0 (최대)"]
        if t_move >= 0:
            out.append(f"       시작 지연({move_th:.2f}도 움직일 때까지): {t_move:.1f}ms")
        else:
            out.append(f"       안 움직임 ({t_limit:.0f}ms 동안 {move_th:.2f}도 미만). 최종 {last:+.2f}도")
        if fixed:
            out.append(f"       고정 기록 {t_limit:.0f}ms: 최종 {last:+.2f}도 (목표 대비 {last - target:+.2f})")
        if not small:
            out += [f"       10->90%: {t90 - t10:.0f}ms -> 평균 {0.8 * abs(delta) / ((t90 - t10) / 1000):.0f}도/s",
                    f"       최고속도: 위치미분 {peak_pos:.0f}도/s, 서보 보고값 {peak_reg:.0f}도/s"]
            if t_settle >= 0:
                out.append(f"       도달(±1도 100ms 유지): {t_settle:.0f}ms, 오버슈트 {overshoot:.1f}도, 최종 {last:+.1f}도")
            elif not fixed:
                out.append(f"       ±1도 안에 안정 못 함 (3초). 최종 {last:+.1f}도, 오버슈트 {overshoot:.1f}도")
        if raw:
            dz = round(self.servo_db / q)
            out.append(f"[STEPRAW] BEGIN servo={i + 1} start_cdeg={round(start * 100)} target_cdeg={round(target * 100)} "
                       f"n={len(rawrows)} acc={self.acc_reg} dz_cw={dz} dz_ccw={dz}")
            out += rawrows + ["[STEPRAW] END"]
        out.append(f"       읽기 {n}회 (평균 {t / n:.2f}ms 간격), 실패 0")
        return out


def run(fake, kind, reps, out):
    buf = io.StringIO()
    with redirect_stdout(buf):
        extra = ["--warmup", "0"] if kind == "drive" else []   # 가짜 펌웨어는 예열 효과가 없고, 예열 CAP 이 난수열·표류 계수를 바꾼다
        d = su.main(["--out", out, kind, "--reps", str(reps)] + extra, port=fake)
        res = sa.analyze(d)
    return res, buf.getvalue()


def failed(res):
    return [c["name"] for c in res["checks"] if not c["ok"]]


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    return bool(cond)


def main():
    ok = True
    tmp = tempfile.mkdtemp(prefix="sysid_test_")
    print("A. 정상 (정답: ω0 7.90/8.15/7.70/8.05, 데드밴드 32/38/45/35, τ 35 ms, 지연 4 ms)")
    fk = FakeRover(seed=1)
    r, log = run(fk, "drive", 3, tmp)
    ok &= check(not failed(r), f"모든 확인 PASS (FAIL: {failed(r)})")
    for m in range(4):
        f = r["motor"][m + 1]["fwd"]
        ok &= check(abs(f["w0"]["mean"] / fk.w0[m] - 1) < 0.015, f"M{m + 1} ω0 {f['w0']['mean']:.3f} (정답 {fk.w0[m]})")
        ok &= check(abs(f["dead"]["mean"] - fk.dead[m]) < 6, f"M{m + 1} 데드밴드 {f['dead']['mean']:.1f} (정답 {fk.dead[m]:.0f})")
        b = r["motor"][m + 1]["rev"]
        ok &= check(abs(b["w0"]["mean"] / (fk.w0[m] * fk.rev) - 1) < 0.015, f"M{m + 1} ω0 후진 {b['w0']['mean']:.3f}")
        t = r["d2"][sa.FIT_SPEC][m + 1]
        ok &= check(abs(t["tau_ms"]["mean"] - 35) < 3.5, f"M{m + 1} τ {t['tau_ms']['mean']:.1f} ms (정답 35)")
        ok &= check(abs(t["lag_ms"]["mean"] - 4) < 1.5, f"M{m + 1} 지연 {t['lag_ms']['mean']:.1f} ms (정답 4)")
    ok &= check(abs(r["tau_ratio"]["799 0 100 1200"] - 1) < 0.15, f"제동 τ 비 {r['tau_ratio']['799 0 100 1200']:.2f} (정답 1)")
    ok &= check(not any(r["stiction"][m][k]["sig"] for m in range(1, 5) for k in ("fwd", "rev")),
                "정지마찰 없음으로 판정: " + " ".join(f"{r['stiction'][m]['fwd']['gap']:+.1f}" for m in range(1, 5)))

    fs = FakeRover(seed=2)
    r, log = run(fs, "steer", 3, tmp)
    ok &= check(not failed(r), f"조향 모든 확인 PASS (FAIL: {failed(r)})")
    a50, a0 = r["acc"][50]["fit"], r["acc"][0]["fit"]
    exp_a = 50 * 100 * 360 / 4096
    ok &= check(abs(a50["a"] / exp_a - 1) < 0.2, f"ACC50 가속 {a50['a']:.0f} 도/s² (정답 {exp_a:.0f})")
    ok &= check(a50["v_lower"], f"ACC50 최고속은 하한({a50['v']:.0f} 도/s)으로만 보고 (큰 계단도 삼각형 프로파일)")
    ok &= check(abs(a0["v"] / 330 - 1) < 0.1, f"ACC0 최고속 {a0['v']:.0f} 도/s (정답 330)")
    ok &= check(a0["a"] > 3000, f"ACC0 가속 {a0['a']:.0f} 도/s² (정답 제한 없음)")
    ok &= check(r["acc50_vs_0_t1090_ms"] > 100, f"ACC50−0 큰 계단 10→90% 차이 {r['acc50_vs_0_t1090_ms']:.0f} ms")

    st5 = r["acc"][50]["lag_ms"]
    ok &= check(abs(st5["mean"] - 5) < 1.5, f"ACC50 시작 지연 {st5['mean']:.1f} ms (정답 5)")
    ok &= check(abs(r["acc"][0]["lag_ms"]["mean"] - 5) < 1.5, f"ACC0 시작 지연 {r['acc'][0]['lag_ms']['mean']:.1f} ms (정답 5)")
    ok &= check(Path(r["dir"], "steer_traj.npz").exists(), "궤적 steer_traj.npz 저장")

    r, log = run(FakeRover(seed=7, servo_ring=4.0), "steer", 3, tmp)
    ok &= check(any("궤적 모델 적합" in f for f in failed(r)),
                f"도착 후 4도 진동하는 서보: 궤적 모델 FAIL 로 잡힘 ({[f for f in failed(r) if '궤적' in f]})")

    print("D. 조향 작은 계단: 데드존 없음 / 0.25도")
    r, log = run(FakeRover(seed=5), "steer-small", 3, tmp)
    ok &= check(not failed(r), f"데드존 0: 모든 확인 PASS (FAIL: {failed(r)})")
    ok &= check(abs(r["hysteresis_deg"]["mean"]) < 0.1, f"데드존 0: 히스테리시스 {r['hysteresis_deg']['mean']:+.3f}도 (정답 0)")
    ok &= check(r["threshold_deg"]["mean"] < 0.15, f"데드존 0: 최소 명령 {r['threshold_deg']['mean']:.2f}도 (정답 가장 작은 A)")
    r, log = run(FakeRover(seed=6, servo_db=0.25), "steer-small", 3, tmp)
    ok &= check(not failed(r), f"데드존 0.25: 모든 확인 PASS (FAIL: {failed(r)})")
    ok &= check(abs(r["hysteresis_deg"]["mean"] - 0.5) < 0.1, f"데드존 0.25: 히스테리시스 {r['hysteresis_deg']['mean']:+.3f}도 (정답 0.5)")
    ok &= check(r["threshold_deg"]["mean"] == 0.3, f"데드존 0.25: 최소 명령 {r['threshold_deg']['mean']:.2f}도 (정답 0.3 = 0.25 위 첫 격자)")

    print("B. 배터리 표류 5%")
    r, log = run(FakeRover(seed=3, drift=0.05), "drive", 3, tmp)
    ok &= check("배터리·온도 표류 (REF)" in failed(r), f"표류 FAIL 로 잡힘 (범위 {r['ref_drift'] * 100:.1f}%)")

    print("C. 제동 시정수 3배 + 정지마찰 40 duty (1차 모델에 없는 항)")
    r, log = run(FakeRover(seed=4, tau_brake=0.105, stiction=40), "drive", 3, tmp)
    fl = failed(r)
    ok &= check("모델 검증 D2 799 0 100 1200" in fl, "제동 시험 검증 FAIL")
    ok &= check("모델 검증 D3 0 150 2000 2000 RAMP" in fl, "저속 램프(기동) 검증 FAIL")
    ok &= check("모델 검증 D2 0 799 100 700" not in fl and "모델 검증 D2 0 200 100 700" not in fl,
                "해당 없는 계단(0→200, 0→799)은 PASS")
    ok &= check(r["tau_ratio"]["799 0 100 1200"] > 2.5, f"제동 τ 비 {r['tau_ratio']['799 0 100 1200']:.2f} (정답 3)")
    for m in range(1, 5):
        g = r["stiction"][m]["fwd"]
        ok &= check(g["sig"] and abs(g["gap"] - 40) < 10, f"M{m} 정지마찰 {g['gap']:.1f} duty (정답 40)")
    print("E. 바퀴 속도 PI (WCAP, 진짜 wheel_pi.c + 실측 플랜트 모델) -- 설계 시뮬과 같은 값이 나오는가")
    buf = io.StringIO()
    with redirect_stdout(buf):
        d = su.main(["--out", tmp, "pi", "--reps", "1"], port=FakeRover(seed=8))
    mets = {}
    for fn in sorted(Path(d).glob("*.npz")):
        c = dict(np.load(fn))
        mets[" ".join(str(c["spec"]).split()[1:])] = np.nanmean(su.pi_metrics(c), axis=0)
    m2, m03 = mets["0 2 100 700"], mets["0 0.3 100 1500"]
    ok &= check(30 <= m2[0] <= 50, f"0→2 rad/s 90% {m2[0]:.0f} ms (설계 37)")
    ok &= check(m2[1] < 3, f"0→2 오버슈트 {m2[1]:+.1f}% (설계 0.5)")
    ok &= check(abs(m2[2]) < 0.02, f"0→2 정상오차 {m2[2]:+.3f} rad/s")
    ok &= check(abs(m03[2]) < 0.02 and m03[0] < 60, f"0→0.3 저속 기동 {m03[0]:.0f} ms, 정상오차 {m03[2]:+.3f}")
    ok &= check(abs(mets["2 -2 100 900"][2]) < 0.02, f"2→−2 역전 정상오차 {mets['2 -2 100 900'][2]:+.3f}")
    ok &= check(abs(mets["5 0 100 600"][2]) < 1e-9, "5→0 정지 (duty 0 = 브레이크)")
    ok &= check("요약" in buf.getvalue(), "요약 출력")
    print("F. IMU 고속 기록 (ICAP): 정지 기울기 10°, 1 s 뒤 진동")
    buf = io.StringIO()
    with redirect_stdout(buf):
        d = su.main(["--out", tmp, "imu", "--ms", "5000", "--ref-deg", "10", "--tag", "t"], port=FakeRover(seed=9))
    c = dict(np.load(Path(d) / "imu.npz"))
    log = buf.getvalue()
    m = re.search(r"기울기 ([\d.]+)°  \(기준 10.0°", log)
    ok &= check(len(c["t_s"]) == 1040, f"샘플 {len(c['t_s'])} (5 s × 208 Hz)")
    ok &= check(m is not None and abs(float(m[1]) - 10) < 0.3, f"정지 기울기 {m and m[1]}° (정답 10, 칩→몸체 좌표 변환 포함)")
    ok &= check("필터 출력 차이" in log and "20 ms 칸 샘플 수 4.1" in log, "진동·필터 요약 출력 (20 ms 칸 ≈ 4.16 샘플)")
    print("전체:", "PASS" if ok else "FAIL", f"  (로그 {tmp})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
