#!/usr/bin/env python3
"""sysid_uart.py 측정 결과 분석·판정 (firmware/MOTOR_MODELING_PLAN.md §0).  numpy 만 쓴다.

    python3 pi/common/sysid_analyze.py sysid_logs/<시각>_drive
    python3 pi/common/sysid_analyze.py sysid_logs/<시각>_steer_servo1
    python3 pi/common/sysid_analyze.py sysid_logs/<시각>_steersmall_servo1

결과는 화면과 같은 디렉터리의 report.txt / summary.json (시뮬 반영용 값) 에 남는다.

믿을 수 있는지 세 단계로 본다.
  1. 반복성: 같은 시험 반복(--reps)마다 따로 피팅해서 평균, 표준편차, 95% 신뢰구간(t 분포)을 낸다.
     신뢰구간이 허용폭(TOL)보다 넓으면 그 값은 "반복 부족/불안정" 으로 FAIL.
     반복 중 혼자 튀는 값(중앙값에서 4 MAD 이상)은 따로 적는다 -- 배선 접촉·UART 손실 의심.
  2. 측정 조건: REF 시험(duty 400)이 측정 내내 2% 안에서 같은지 (배터리 처짐), 샘플 타이밍(late_max_us).
  3. 모델 검증 (증명): D1 의 duty→속도 표 + D2 "0→400" 의 시정수·지연만으로 만든 1차 모델이
     **피팅에 안 쓴** 시험(다른 크기 계단, 제동, 역전, 저속 램프)의 속도 궤적을 맞히는지.
     오차를 반복 간 잡음 바닥과 비교해서 PASS/FAIL. FAIL 이 나는 시험의 종류가 모델에 빠진 항을 알려준다
     (제동만 → 제동 시정수 따로, 램프만 → 정지마찰, 역전만 → 백래시, 큰 계단만 → 전류 제한).
조향은 STEP 의 10→90% 시간을 계단 크기별로 모아 사다리꼴(가속 a, 최고속 v) 프로파일을 피팅하고,
ACC 50 과 0 의 차이가 신뢰구간 밖인지 본다.  RAW 궤적이 있으면 궤적 전체에 (공통 가속도 + 계단마다 지연) 을 맞춰
시작 지연(#15)을 내고, 궤적 RMS 로 사다리꼴 모델이 맞는지 판정한다 (steer_traj.npz 는 시뮬 비교용).
작은 계단(steersmall)은 0 에 ±5° 쪽에서 다가가 멈춘 위치 차이(히스테리시스)와 움직이기 시작하는 최소 명령(데드밴드)을 낸다.
"""
import json
import os
import re
import sys

import numpy as np

PWM_MAX = 799
TOL = dict(w0_rel=0.01, dead_duty=5.0, tau_rel=0.10, lag_ms=2.0, break_duty=6.0, steer_rel=0.10, small_deg=0.1)
REF_DRIFT_MAX = 0.02
LATE_MAX_US = 100
MOVE_RAD_S = 0.3          # 이보다 빠르면 "돈다" (D1 직선 피팅, D3 기동 판정)
FIT_SPEC = "0 400 100 700"
D3_MOVE_RAD_S = 0.06     # D3 (50 ms 창, 1틱 양자화 0.015 rad/s)
STICTION_MIN_DUTY = 10.0   # 첫 틱이 쌓이는 시간 때문에 정지마찰 없어도 +5~7 로 나옴 (가짜 펌웨어 시험) → 10 미만은 구별 못 함
_T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145,
         2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045]


# ---------------------------------------------------------------- 공통 통계
def stats(x):
    """평균, 표준편차, 95% 신뢰구간 반폭, n."""
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    n = len(x)
    if n == 0:
        return dict(n=0, mean=np.nan, std=np.nan, ci=np.nan)
    if n == 1:
        return dict(n=1, mean=float(x[0]), std=np.nan, ci=np.inf)
    sd = float(x.std(ddof=1))
    t = _T975[n - 2] if n - 2 < len(_T975) else 1.96
    return dict(n=n, mean=float(x.mean()), std=sd, ci=t * sd / np.sqrt(n))


def outliers(x, rel=0.02, absf=0.05):
    """중앙값에서 4·MAD(정규 환산) 와 (rel·|중앙값| + absf) 둘 다 넘게 벗어난 원소의 위치."""
    x = np.asarray(x, float)
    if len(x) < 3:
        return []
    med = np.median(x)
    mad = 1.4826 * np.median(np.abs(x - med))
    lim = max(4 * mad, rel * abs(med) + absf)
    return [i for i, v in enumerate(x) if abs(v - med) > lim]


def fmt(s, unit="", nd=3):
    if s["n"] == 0:
        return "없음"
    ci = "∞" if not np.isfinite(s["ci"]) else f"{s['ci']:.{nd}f}"
    return f"{s['mean']:.{nd}f} ± {ci}{unit} (n={s['n']})"


class Report:
    def __init__(self):
        self.lines, self.checks = [], []

    def __call__(self, s=""):
        print(s)
        self.lines.append(s)

    def check(self, name, ok, detail):
        self.checks.append(dict(name=name, ok=bool(ok), detail=detail))
        self(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


# ---------------------------------------------------------------- 구동
def load_drive(d):
    with open(os.path.join(d, "manifest.json")) as f:
        man = json.load(f)
    caps = []
    for fn in man["files"]:
        z = np.load(os.path.join(d, fn), allow_pickle=False)
        c = {k: z[k] for k in z.files}
        for k in ("group", "spec"):
            c[k] = str(c[k])
        for k in ("rep", "order", "d0", "d1", "t_sw", "t_end", "ramp", "dt_us", "n", "ticks_per_rev",
                  "late_max_us", "vs_dv", "motors"):
            if k in c:
                c[k] = int(c[k])
        c["key"] = " ".join(c["spec"].split()[1:])     # spec 앞의 모터 선택(ALL/1-4) 제외
        c["pos_rad"] = c["pos_ticks"] * 2 * np.pi / c["ticks_per_rev"]
        caps.append(c)
    return man, caps


def speed_trace(t, pos, h):
    """중앙 차분 속도 (창 2h 샘플). 반환 (t_center, v[N,4])."""
    v = (pos[2 * h:] - pos[:-2 * h]) / (t[2 * h:] - t[:-2 * h])[:, None]
    return t[h:-h], v


def switch_index(c):
    """duty 가 d1 로 바뀐 직후의 첫 샘플 번호. 펌웨어는 샘플 k 직후 t_next=(k+1)·dec 의 duty 를 걸므로
    실제 전환 시각은 t_s[i_sw - 1] (= t_sw - dec)."""
    return max(int(c["t_sw"] // (c["dt_us"] // 1000)), 1)


def mean_speed(c, i0, i1):
    return (c["pos_rad"][i1 - 1] - c["pos_rad"][i0]) / (c["t_s"][i1 - 1] - c["t_s"][i0])


def step_model_pos(u, wi, wf, tau, lag):
    """1차 계단 응답의 위치 (u = 전환 후 시간). u<lag 는 wi 등속."""
    s = np.maximum(u - lag, 0.0)
    return wi * np.minimum(u, lag) + wf * s + (wi - wf) * tau * (1 - np.exp(-s / tau))


TAU_GRID = np.exp(np.linspace(np.log(0.002), np.log(0.6), 160))
LAG_GRID = np.arange(0.0, 0.0305, 0.0005)


def fit_step(c, m):
    """모터 m 의 계단에서 (tau, lag, wi, wf, 속도 잔차 RMS). 위치 궤적에 격자 탐색 -- 속도 미분 잡음을 피한다."""
    i_sw = switch_index(c)
    t, p = c["t_s"], c["pos_rad"][:, m]
    ts = t[i_sw - 1]
    n = len(t)
    wi = 0.0 if c["d0"] == 0 else float(mean_speed(c, 0, i_sw)[m])
    wf = float(mean_speed(c, n - max(n // 4, 20), n)[m])
    u = t[i_sw - 1:] - ts
    y = p[i_sw - 1:] - p[i_sw - 1]
    best = (np.inf, None, None)
    for lag in LAG_GRID:
        pred = step_model_pos(u[None, :], wi, wf, TAU_GRID[:, None], lag)
        sse = ((pred - y[None, :]) ** 2).sum(1)
        j = int(np.argmin(sse))
        if sse[j] < best[0]:
            best = (sse[j], TAU_GRID[j], lag)
    _, tau, lag = best
    h = 5
    tc, v = speed_trace(u, y[:, None], h)
    vm = np.gradient(step_model_pos(u, wi, wf, tau, lag), u)[h:-h]
    rms = float(np.sqrt(np.mean((v[:, 0] - vm) ** 2)))
    edge = tau <= TAU_GRID[0] * 1.01 or tau >= TAU_GRID[-1] * 0.99
    return dict(tau=float(tau), lag=float(lag), wi=wi, wf=wf, rms=rms, edge=edge)


def simulate(c, table_d, table_w, tau, lag, sub=4):
    """duty 시퀀스를 1차 모델에 넣어 위치 궤적 (샘플 시각). 정상속도 = D1 표 보간, d=0 근처도 표 그대로."""
    t, duty = c["t_s"], c["duty"]
    w = float(np.interp(duty[0], table_d, table_w))
    pos = np.zeros(len(t))
    for k in range(1, len(t)):
        dt = (t[k] - t[k - 1]) / sub
        p = pos[k - 1]
        for j in range(sub):
            tm = t[k - 1] + (j + 0.5) * dt - lag
            i = min(int(np.searchsorted(t, tm, side="left")), len(t) - 1)
            w_ss = float(np.interp(duty[i], table_d, table_w))
            a = 1 - np.exp(-dt / tau)
            w_new = w + (w_ss - w) * a
            p += 0.5 * (w + w_new) * dt
            w = w_new
        pos[k] = p
    return pos


def analyze_drive(d, rep=None):
    rep = rep or Report()
    man, caps = load_drive(d)
    by = {}
    for c in caps:
        by.setdefault((c["group"], c["key"]), []).append(c)
    motors = [m for m in range(4) if any(np.any(c["pos_ticks"][:, m] != 0) for c in caps)]
    out = dict(kind="drive", dir=d, motors=[m + 1 for m in motors], reps=man["reps"])
    rep(f"== 구동 분석: {d}")
    rep(f"   시험 {len(caps)}개, 반복 {man['reps']}, 움직인 모터 {[m + 1 for m in motors]}, {man['started']} ~ {man.get('finished', '?')}")

    # ---- 측정 조건
    rep("\n[측정 조건]")
    late = max(c["late_max_us"] for c in caps)
    rep.check("샘플 타이밍", late <= LATE_MAX_US, f"late_max {late} us (허용 {LATE_MAX_US})")
    refs = sorted(by.get(("REF", "400 400 0 1000 DEC 5"), []), key=lambda c: c["order"])
    if refs:
        rw = np.array([mean_speed(c, 0, len(c["t_s"]))[motors].mean() for c in refs])
        dev = rw / rw[0] - 1
        vs = [c["vs_dv"] / 10 for c in refs if c["vs_dv"] >= 0]
        rep("  REF(duty 400) 평균속도 순서대로: " + " ".join(f"{x:.3f}" for x in rw) + " rad/s")
        if vs:
            rep("  그때 서보 전압: " + " ".join(f"{x:.1f}" for x in vs) + " V")
        rng = float(dev.max() - dev.min())
        rep.check("배터리·온도 표류 (REF)", rng <= REF_DRIFT_MAX,
                  f"처음 대비 {dev[-1] * 100:+.2f}%, 범위 {rng * 100:.2f}% (허용 {REF_DRIFT_MAX * 100:.0f}%)")
        out["ref_drift"] = rng

    # ---- D1
    rep("\n[D1] duty → 정상속도 (각 반복 따로 직선 피팅: ω = k·(duty − 데드밴드))")
    d1 = {}
    for (g, key), cs in by.items():
        if g == "D1":
            d1[int(key.split()[0])] = sorted(cs, key=lambda c: c["rep"])
    duties = sorted(d1)
    nrep = min(len(v) for v in d1.values()) if d1 else 0
    W = np.full((len(duties), nrep, 4), np.nan)        # duty × rep × motor
    for i, du in enumerate(duties):
        for r, c in enumerate(d1[du][:nrep]):
            W[i, r] = mean_speed(c, 0, len(c["t_s"]))
    da = np.array(duties, float)
    if nrep:
        rep("  duty  " + "  ".join(f"     M{m + 1} (평균±표준편차)" for m in motors))
        for i, du in enumerate(duties):
            rep(f"  {du:+4d}  " + "  ".join(f"{np.nanmean(W[i, :, m]):+8.3f} ± {np.nanstd(W[i, :, m], ddof=1):.3f}    "
                                             for m in motors))
            for m in motors:
                for j in outliers(W[i, :, m]):
                    rep(f"    ! 튀는 반복: duty {du} M{m + 1} 반복 {j}: {W[i, j, m]:+.3f} (나머지 중앙값 {np.median(W[i, :, m]):+.3f})")
    out["d1_table"] = dict(duty=duties, w_mean=np.nanmean(W, 1).tolist() if nrep else [])
    out["motor"] = {}
    for m in motors:
        res = {}
        for sgn, name in ((1, "fwd"), (-1, "rev")):
            sel = (np.sign(da) == sgn)
            w0s, dds, lin, ks = [], [], [], []
            for r in range(nrep):
                x, y = da[sel], W[sel, r, m]
                ok = np.abs(y) > MOVE_RAD_S
                if ok.sum() < 3:
                    continue
                k, b = np.polyfit(x[ok], y[ok], 1)
                w0s.append(abs(k * sgn * PWM_MAX + b))
                dds.append(abs(-b / k))
                ks.append(abs(k))
                lin.append(np.max(np.abs(y[ok] - (k * x[ok] + b))) / abs(k * sgn * PWM_MAX + b))
            res[name] = dict(w0=stats(w0s), dead=stats(dds), nonlin=float(np.max(lin)) if lin else np.nan,
                             slope=float(np.mean(ks)) if ks else np.nan)
        out["motor"][m + 1] = res
    w0f = np.array([out["motor"][m + 1]["fwd"]["w0"]["mean"] for m in motors])
    rep("  모터  ω0 전진 (duty 799) rad/s     ω0 후진               데드밴드 전진 duty   데드밴드 후진    비선형(최대)")
    for m in motors:
        f, b = out["motor"][m + 1]["fwd"], out["motor"][m + 1]["rev"]
        rep(f"  M{m + 1}   {fmt(f['w0'])}   {fmt(b['w0'])}   {fmt(f['dead'], nd=1)}   {fmt(b['dead'], nd=1)}"
            f"   {f['nonlin'] * 100:.1f}% / {b['nonlin'] * 100:.1f}%")
    if len(w0f):
        rep(f"  ω0 평균 {w0f.mean():.3f} rad/s = 바퀴 {w0f.mean() * 0.047503:.3f} m/s (반경 0.047503).  "
            f"모터 개체차 {', '.join(f'M{m + 1} {x / w0f.mean() - 1:+.1%}' for m, x in zip(motors, w0f))}")
        out["w0_mean"] = float(w0f.mean())
        out["motor_spread"] = float(np.max(np.abs(w0f / w0f.mean() - 1)))
    for m in motors:
        for name in ("fwd", "rev"):
            s = out["motor"][m + 1][name]
            rep.check(f"M{m + 1} ω0 {name} 반복성", s["w0"]["n"] >= 2 and s["w0"]["ci"] <= TOL["w0_rel"] * s["w0"]["mean"],
                      f"95%CI ±{s['w0']['ci']:.3f} (허용 ±{TOL['w0_rel'] * s['w0']['mean']:.3f})")
            rep.check(f"M{m + 1} 데드밴드 {name} 반복성", s["dead"]["n"] >= 2 and s["dead"]["ci"] <= TOL["dead_duty"],
                      f"95%CI ±{s['dead']['ci']:.1f} duty (허용 ±{TOL['dead_duty']:.0f})")
    # #9 엔코더 속도 잡음 @ 50 Hz
    noise = []
    for du in duties:
        if abs(du) < 200:
            continue
        for c in d1[du]:
            dec = c["dt_us"] // 1000
            h = max(20 // dec // 2, 1)
            _, v = speed_trace(c["t_s"], c["pos_rad"], h)
            noise += [float(np.std(v[:, m])) for m in motors]
    if noise:
        q = 2 * np.pi / caps[0]["ticks_per_rev"] / 0.02
        out["enc_noise_50hz"] = float(np.median(noise))
        rep(f"  #9 정상상태 속도 잡음 (20 ms 창): 중앙값 {np.median(noise):.4f} rad/s  (1틱 양자화만이면 ~{q / np.sqrt(6):.4f}, 최대 {q:.4f})")

    # ---- D2
    rep("\n[D2] 계단 응답 (위치 궤적에 1차 + 지연 피팅, 반복·모터별)")
    fits = {}
    for (g, key), cs in sorted(by.items()):
        if g != "D2":
            continue
        for c in cs:
            for m in motors:
                fits.setdefault(key, {}).setdefault(m, []).append(fit_step(c, m))
    out["d2"] = {}
    for key in sorted(fits, key=lambda k: D2_ORDER.index(k) if k in D2_ORDER else 99):
        rep(f"  {key:18s}")
        for m in motors:
            fl = fits[key][m]
            st, sl = stats([f["tau"] * 1e3 for f in fl]), stats([f["lag"] * 1e3 for f in fl])
            rms = np.mean([f["rms"] for f in fl])
            edge = sum(f["edge"] for f in fl)
            rep(f"     M{m + 1}  τ {fmt(st, ' ms', 1)}   지연 {fmt(sl, ' ms', 1)}   끝속도 {np.mean([f['wf'] for f in fl]):+.3f}   "
                f"피팅잔차 {rms:.3f} rad/s" + (f"   ! τ 가 격자 끝 {edge}회" if edge else ""))
            for j in outliers([f["tau"] for f in fl], rel=0.05, absf=0.002):
                rep(f"       ! 튀는 반복 {j}: τ {fl[j]['tau'] * 1e3:.1f} ms")
            out["d2"].setdefault(key, {})[m + 1] = dict(tau_ms=st, lag_ms=sl, fit_rms=float(rms))
    fit_set = out["d2"].get(FIT_SPEC, {})
    for m in motors:
        if m + 1 not in fit_set:
            continue
        st, sl = fit_set[m + 1]["tau_ms"], fit_set[m + 1]["lag_ms"]
        rep.check(f"M{m + 1} τ ({FIT_SPEC}) 반복성", st["n"] >= 2 and st["ci"] <= TOL["tau_rel"] * st["mean"],
                  f"{st['mean']:.1f} ms, 95%CI ±{st['ci']:.1f} (허용 ±{TOL['tau_rel'] * st['mean']:.1f})")
        rep.check(f"M{m + 1} 지연 반복성", sl["n"] >= 2 and sl["ci"] <= TOL["lag_ms"],
                  f"{sl['mean']:.1f} ms, 95%CI ±{sl['ci']:.1f} (허용 ±{TOL['lag_ms']:.0f})")
    # 모양 확인: τ 가 크기·방향·제동에 따라 같은가 (같으면 선형 1차 모델로 충분, J = τ·D)
    rep("  τ 비교 (모터 평균, 0→400 대비):")
    base = np.mean([fit_set[m + 1]["tau_ms"]["mean"] for m in motors if m + 1 in fit_set]) if fit_set else np.nan
    out["tau_ms"] = float(base)
    out["tau_ratio"] = {}
    for key in out["d2"]:
        v = np.mean([out["d2"][key][m + 1]["tau_ms"]["mean"] for m in motors])
        out["tau_ratio"][key] = float(v / base)
        tag = {"799 0 100 1200": "  ← 제동 (MDD3A 브레이크)", "400 -400 100 1200": "  ← 역전"}.get(key, "")
        rep(f"     {key:18s} τ {v:6.1f} ms  ×{v / base:.2f}{tag}")
    if np.isfinite(base):
        rep(f"  → 반사관성 J = τ·D: D(토크-속도 기울기)를 D5 에서 재면 J = {base * 1e-3:.4f}·D  "
            f"(D=0.246 이면 {base * 1e-3 * 0.246:.2e} kg·m², 지금 시뮬 armature 1.5e-3)")
    # D1 정상속도 vs D2 끝속도 일치
    rep("  D1 정상속도 vs D2 끝속도 (같은 duty, 다른 시험):")
    for key in ("0 200 100 700", "0 400 100 700", "0 799 100 700", "0 -799 100 700"):
        if key not in out["d2"]:
            continue
        du = int(key.split()[1])
        if du not in d1:
            continue
        a = np.nanmean(W[duties.index(du)][:, motors], 0)
        b = np.array([np.mean([f["wf"] for f in fits[key][m]]) for m in motors])
        dmax = float(np.max(np.abs(b / a - 1)))
        rep.check(f"정상속도 일치 duty {du:+d}", dmax <= 0.02, f"최대 차이 {dmax * 100:.1f}% (허용 2%)")

    # ---- D3
    rep("\n[D3] 저속 램프: 기동 duty(정지→회전) / 정지 duty(회전→정지)")
    rep("  기동 = 확실히 돈 뒤(50 ms 창 속도 > 문턱이 50 ms) 거슬러 올라가 엔코더가 처음 움직인 샘플의 duty − 램프기울기×지연")
    rep("  정지 = 엔코더가 50 ms 동안 안 움직이기 시작한 샘플의 duty + 램프기울기×(지연+τ).  기동을 D1 데드밴드(운동마찰)와 비교한다.")
    out["d3"] = {}
    for (g, key), cs in sorted(by.items()):
        if g != "D3":
            continue
        d0, d1v = int(key.split()[0]), int(key.split()[1])
        up = abs(d1v) > abs(d0)
        name = "fwd" if (d0 + d1v) > 0 else "rev"
        vals = {m: [] for m in motors}
        for c in cs:
            h = 25
            _, v = speed_trace(c["t_s"], c["pos_rad"], h)
            rate = abs(d1v - d0) / (c["t_sw"] * 1e-3)            # duty/s
            for m in motors:
                f2 = fit_set.get(m + 1)
                tau = f2["tau_ms"]["mean"] * 1e-3 if f2 else 0.0
                lag = f2["lag_ms"]["mean"] * 1e-3 if f2 else 0.0
                p = c["pos_ticks"][:, m]
                mv = np.abs(v[:, m]) > D3_MOVE_RAD_S
                hold, still = 50, 20
                if up:   # 확실히 돈 시점에서 거슬러 올라가 마지막으로 20 ms 동안 틱이 안 변한 샘플 = 움직이기 시작
                    idx = next((i for i in range(len(mv) - hold) if mv[i:i + hold].all()), None)
                    if idx is not None:
                        j = idx + h
                        while j > still and p[j] != p[j - still]:
                            j -= 1
                        vals[m].append(abs(float(c["duty"][j])) - rate * lag)
                else:    # 확실히 선 시점 = 그 뒤 50 ms 동안 틱이 안 변하는 첫 샘플.  τ 만큼 늦게 서므로 그만큼 되돌린다
                    j = next((i for i in range(len(p) - hold) if p[i + hold] == p[i]), None)
                    if j is not None:
                        vals[m].append(abs(float(c["duty"][j])) + rate * (lag + tau))
                if (up and idx is None) or (not up and j is None):
                    vals[m].append(np.nan)
        lab = "기동" if up else "정지"
        out["d3"][key] = {m + 1: stats(vals[m]) for m in motors}
        rep(f"  {key:22s} {lab} duty: " + "   ".join(f"M{m + 1} {fmt(out['d3'][key][m + 1], '', 1)}" for m in motors))
        if up:
            for m in motors:
                s_ = out["d3"][key][m + 1]
                rep.check(f"M{m + 1} 기동 duty ({key.split()[1]}) 반복성", s_["n"] >= 2 and s_["ci"] <= TOL["break_duty"],
                          f"95%CI ±{s_['ci']:.1f} (허용 ±{TOL['break_duty']:.0f})")
    out["stiction"] = {}
    for m in motors:
        for name, a in (("fwd", "0 150 2000 2000 RAMP"), ("rev", "0 -150 2000 2000 RAMP")):
            if a not in out["d3"]:
                continue
            s0, dk = out["d3"][a][m + 1], out["motor"][m + 1][name]["dead"]
            gap = s0["mean"] - dk["mean"]
            sig = gap > max(np.hypot(s0["ci"], dk["ci"]), STICTION_MIN_DUTY)
            out["stiction"].setdefault(m + 1, {})[name] = dict(gap=float(gap), sig=bool(sig))
            rep(f"  M{m + 1} {'전진' if name == 'fwd' else '후진'}: 기동(정지마찰) {s0['mean']:.1f} − D1 데드밴드(운동마찰) {dk['mean']:.1f}"
                f" = {gap:+.1f} duty " + ("→ 정지마찰 있음 (시뮬 FF·마찰에 반영 검토)" if sig else "→ 정지마찰 구별 안 됨 (10 duty 미만은 이 방법으로 못 가림)"))

    # ---- 모델 검증
    rep(f"\n[검증] D1 표 + '{FIT_SPEC}' 의 τ·지연만으로 만든 1차 모델이 피팅에 안 쓴 시험을 맞히는가")
    rep("  오차 = 모델 vs 반복 평균 궤적 RMS (10 ms 창 속도).  잡음 바닥 = 반복끼리의 RMS 편차.")
    rep("  PASS 조건: 오차 ≤ max(2 × 잡음 바닥, 최고속도의 3%)")
    out["validation"] = {}
    if fit_set and nrep:
        for (g, key), cs in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            if g not in ("D2", "D3") or key == FIT_SPEC:
                continue
            cs = sorted(cs, key=lambda c: c["rep"])
            nmin = min(len(c["t_s"]) for c in cs)
            line = []
            for m in motors:
                td = np.concatenate([da, [0.0]])
                tw = np.concatenate([np.nanmean(W[:, :, m], 1), [0.0]])
                o = np.argsort(td)
                tau = fit_set[m + 1]["tau_ms"]["mean"] * 1e-3
                lag = fit_set[m + 1]["lag_ms"]["mean"] * 1e-3
                h = 5
                vs = []
                for c in cs:
                    _, v = speed_trace(c["t_s"][:nmin], c["pos_rad"][:nmin, m:m + 1], h)
                    vs.append(v[:, 0])
                vs = np.array(vs)
                vmean = vs.mean(0)
                floor = float(np.sqrt(np.mean((vs - vmean) ** 2) * len(vs) / max(len(vs) - 1, 1))) if len(vs) > 1 else np.nan
                c0 = cs[0]
                pm = simulate(dict(t_s=c0["t_s"][:nmin], duty=c0["duty"][:nmin]), td[o], tw[o], tau, lag)
                _, vm = speed_trace(c0["t_s"][:nmin], pm[:, None], h)
                err = float(np.sqrt(np.mean((vm[:, 0] - vmean) ** 2)))
                bias = float(np.mean(vm[:, 0] - vmean))
                lim = max(2 * floor, 0.03 * float(np.max(np.abs(vmean))))
                ok = err <= lim
                out["validation"].setdefault(key, {})[m + 1] = dict(err=err, floor=floor, lim=lim, bias=bias, ok=ok)
                line.append(f"M{m + 1} {err:.3f}/{lim:.3f}{'' if ok else '✗'}")
            allok = all(out["validation"][key][m + 1]["ok"] for m in motors)
            rep.check(f"모델 검증 {g} {key}", allok, "오차/허용 rad/s: " + "  ".join(line))
    rep("")
    return out, rep


D2_ORDER = ["0 200 100 700", "0 400 100 700", "0 799 100 700", "799 0 100 1200", "400 -400 100 1200", "0 -799 100 700"]


# ---------------------------------------------------------------- 조향
STEP_RE = dict(
    head=re.compile(r"\[STEP\] 서보(\d) ([+-][\d.]+) -> ([+-][\d.]+)도 \(변화 ([+-][\d.]+)\)"),
    t1090=re.compile(r"10->90%: (\d+)ms"),
    peak=re.compile(r"위치미분 (\d+)도/s, 서보 보고값 (\d+)도/s"),
    settle=re.compile(r"도달\(±1도 100ms 유지\): (\d+)ms, 오버슈트 ([\d.]+)도"),
    move=re.compile(r"시작 지연\([\d.]+도 움직일 때까지\): ([\d.]+)ms"),
)
MAGS = (90, 80, 45, 15, 5, 3)   # 큰 계단은 ±80 (10-13 부터) 또는 예전 ±90


def parse_steer(path):
    """steer.txt -> STEP 별 dict.  RAW 블록이 있으면 traj = (t s, 각도 도, 속도 도/s, 부하) 배열."""
    rows, cur, raw = [], None, None
    for ln in open(path, encoding="utf-8"):
        ln = ln.rstrip("\n")
        m = re.match(r"### ACC (\d+) STEP (\d) (-?[\d.]+) rep (\d+)(.*)", ln)
        if m:
            cur = dict(acc=int(m[1]), target=float(m[3]), rep=int(m[4]))
            cur.update(dict(kv.split("=") for kv in m[5].split()))
            if "amp" in cur:
                cur["amp"] = float(cur["amp"])
            rows.append(cur)
            continue
        if cur is None:
            continue
        if ln.startswith("[STEPRAW] BEGIN"):
            cur["raw_hdr"] = {k: int(v) for k, v in (kv.split("=") for kv in ln.split()[2:])}
            raw = []
            continue
        if raw is not None:
            if ln.startswith("[STEPRAW] END"):
                cur["traj"] = np.array(raw, float).reshape(-1, 4) * [1e-6, 1e-2, 1, 1]
                raw = None
            elif re.fullmatch(r"\d+(,-?\d+){3}", ln):
                raw.append([int(x) for x in ln.split(",")])
            continue
        if (m := STEP_RE["head"].search(ln)):
            cur.update(start=float(m[2]), delta=float(m[4]))
        elif (m := STEP_RE["t1090"].search(ln)):
            cur["t1090"] = float(m[1])
        elif (m := STEP_RE["peak"].search(ln)):
            cur.update(peak_pos=float(m[1]), peak_reg=float(m[2]))
        elif (m := STEP_RE["settle"].search(ln)):
            cur.update(t_settle=float(m[1]), overshoot=float(m[2]))
        elif (m := STEP_RE["move"].search(ln)):
            cur["t_move"] = float(m[1])
    for r in rows:
        if "traj" in r and "raw_hdr" in r:
            r["start"] = r["raw_hdr"]["start_cdeg"] / 100     # 0.01도 해상도
            r["delta"] = r["raw_hdr"]["target_cdeg"] / 100 - r["start"]
    return [r for r in rows if "delta" in r]


def trap_pos(t, theta, a, v):
    """사다리꼴 프로파일 위치 (t<0 은 0, 끝나면 theta).  t 는 배열."""
    t = np.asarray(t, float)
    ta = v / a
    if a * ta * ta >= theta:
        tp = np.sqrt(theta / a)
        x = np.where(t < tp, 0.5 * a * t ** 2, theta - 0.5 * a * np.maximum(2 * tp - t, 0) ** 2)
    else:
        xa = 0.5 * a * ta * ta
        T = 2 * ta + (theta - 2 * xa) / v
        x = np.where(t < ta, 0.5 * a * t ** 2,
                     np.where(t < T - ta, xa + v * (t - ta), theta - 0.5 * a * np.maximum(T - t, 0) ** 2))
    return np.where(t <= 0, 0.0, x)


LAG_GRID_S = np.arange(0.0, 0.0402, 0.0002)
A_TRAJ = np.exp(np.linspace(np.log(30), np.log(1e6), 90))          # deg/s²  (궤적 시작 모양은 큰 가속도도 구별)


def _traj_xy(r):
    tr = r["traj"]
    sg = 1.0 if r["delta"] > 0 else -1.0
    x = (tr[:, 1] - r["start"]) * sg
    tail = tr[:, 0] >= tr[-1, 0] - 0.05
    return tr[:, 0], x, float(np.mean(x[tail]))


def fit_traj(rs, v, a_hint):
    """같은 ACC 의 큰 계단 궤적들에 공통 가속도 a + 계단마다 지연 L 을 맞춘다 (최고속 v 는 고정).
    가속도는 a_hint 근처(1/30~30배) 만 본다.  반환 (a, [L], [RMS 도])."""
    data = [_traj_xy(r) for r in rs]
    data = [(t, x, th) for t, x, th in data if th > 0.2]
    best = (np.inf, None, None, None)
    for a in A_TRAJ[(A_TRAJ >= a_hint / 30) & (A_TRAJ <= a_hint * 30)]:
        tot, Ls, rms = 0.0, [], []
        for t, x, th in data:
            pred = trap_pos(t[None, :] - LAG_GRID_S[:, None], th, a, v)
            sse = ((pred - x[None, :]) ** 2).sum(1)
            j = int(np.argmin(sse))
            tot += sse[j]
            Ls.append(float(LAG_GRID_S[j]))
            rms.append(float(np.sqrt(sse[j] / len(t))))
        if tot < best[0]:
            best = (tot, float(a), Ls, rms)
    return best[1], best[2], best[3], [th for _, _, th in data]


def trap_t1090(theta, a, v):
    """사다리꼴(가속 a, 최고속 v, 감속 a) 프로파일이 θ 의 10% → 90% 를 지나는 데 걸리는 시간 [s]."""
    ta = v / a
    if a * ta * ta >= theta:          # 삼각형 (최고속 못 감)
        tp = np.sqrt(theta / a)
        T = 2 * tp
        def x_to_t(x):
            return np.sqrt(2 * x / a) if x <= theta / 2 else T - np.sqrt(2 * (theta - x) / a)
    else:
        xa = 0.5 * a * ta * ta
        T = 2 * ta + (theta - 2 * xa) / v
        def x_to_t(x):
            if x <= xa:
                return np.sqrt(2 * x / a)
            if x <= theta - xa:
                return ta + (x - xa) / v
            return T - np.sqrt(2 * (theta - x) / a)
    return x_to_t(0.9 * theta) - x_to_t(0.1 * theta)


A_GRID = np.exp(np.linspace(np.log(30), np.log(30000), 140))     # deg/s²
V_GRID = np.linspace(40, 1200, 117)                                 # deg/s


def fit_trap(mags, t_meas):
    best = (np.inf, None, None)
    for a in A_GRID:
        for v in V_GRID:
            r = np.array([trap_t1090(th, a, v) for th in mags]) - t_meas
            sse = float((r ** 2).sum())
            if sse < best[0]:
                best = (sse, a, v)
    sse, a, v = best
    # 가장 큰 계단도 삼각형(최고속 못 감)이면 v 는 그 꼭짓점 속도 이상이라는 것만 안다
    v_lower = v >= 0.97 * np.sqrt(a * max(mags))
    return dict(a=float(a), v=float(v), rms_ms=float(np.sqrt(sse / len(mags)) * 1e3),
                a_edge=a >= A_GRID[-1] * 0.99, v_edge=v >= V_GRID[-1] * 0.99, v_lower=bool(v_lower))


def analyze_steer(d, rep=None):
    rep = rep or Report()
    rows = parse_steer(os.path.join(d, "steer.txt"))
    out = dict(kind="steer", dir=d, acc={})
    rep(f"== 조향 분석: {d}   (STEP {len(rows)}개)")
    accs = sorted({r["acc"] for r in rows})            # 0 먼저 (ACC 50 궤적 피팅이 ACC 0 최고속을 씀)
    for acc in accs:
        rep(f"\n[ACC {acc}]  계단크기   10→90% ms            최고속(위치미분) 도/s    도달 ms               오버슈트 도")
        res = {}
        for mag in MAGS:
            g = [r for r in rows if r["acc"] == acc and round(abs(r["delta"])) == mag]
            if not g:
                continue
            s = {k: stats([r.get(k, np.nan) for r in g]) for k in ("t1090", "peak_pos", "t_settle", "overshoot")}
            miss = sum("t1090" not in r for r in g)
            res[mag] = s
            rep(f"         {mag:3d}°   {fmt(s['t1090'], '', 0):18s}  {fmt(s['peak_pos'], '', 0):22s}  "
                f"{fmt(s['t_settle'], '', 0):18s}  {fmt(s['overshoot'], '', 2)}" + (f"  ! 90% 못 감 {miss}회" if miss else ""))
            for j in outliers([r.get("t1090", np.nan) for r in g], rel=0.1, absf=5):
                rep(f"           ! 튀는 반복: {g[j]}")
        mags = [mg for mg in res if res[mg]["t1090"]["n"] > 0]
        fit = fit_trap(np.array(mags, float), np.array([res[mg]["t1090"]["mean"] for mg in mags]) * 1e-3) if len(mags) >= 3 else None
        out["acc"][acc] = dict(steps={mg: res[mg] for mg in res}, fit=fit)
        if fit:
            exp_a = acc * 100 * 360 / 4096 if acc else None
            rep(f"  사다리꼴 피팅: 가속 {fit['a']:.0f} 도/s² = {np.radians(fit['a']):.1f} rad/s²"
                + (" (격자 끝 = 제한 없음과 구별 안 됨)" if fit["a_edge"] else "")
                + (f", 최고속 ≥ {fit['v']:.0f} 도/s (가속 제한 때문에 가장 큰 계단도 최고속에 못 감 → ACC 0 값을 쓴다)"
                   if fit["v_lower"] else f", 최고속 {fit['v']:.0f} 도/s = {np.radians(fit['v']):.2f} rad/s")
                + f", 잔차 {fit['rms_ms']:.1f} ms"
                + (f"   (Feetech 단위 100 step/s² 가정 시 ACC {acc} = {exp_a:.0f} 도/s²)" if exp_a else ""))
        if res:
            big = max(res)
            s = res[big]["t1090"]
            rep.check(f"ACC {acc} {big}° 10→90% 반복성", s["n"] >= 2 and s["ci"] <= TOL["steer_rel"] * s["mean"],
                      f"{s['mean']:.0f} ms, 95%CI ±{s['ci']:.0f} (허용 ±{TOL['steer_rel'] * s['mean']:.0f})")
        if fit:
            rep.check(f"ACC {acc} 사다리꼴 모델 적합", fit["rms_ms"] <= max(15.0, 0.1 * res[mags[0]]["t1090"]["mean"]),
                      f"계단 크기 {len(mags)}개에 잔차 {fit['rms_ms']:.1f} ms")
        # 궤적(RAW) 단위 검증: 위 (a, v) 로 각 계단 궤적 전체를 맞히는가 + 시작 지연 L (#15)
        tr_rows = [r for r in rows if r["acc"] == acc and "traj" in r and round(abs(r["delta"])) in MAGS]
        if fit and tr_rows:
            a_use = fit["a"]
            v_use = fit["v"]
            if fit["v_lower"] and 0 in out["acc"] and out["acc"][0].get("fit"):
                v_use = out["acc"][0]["fit"]["v"]         # ACC 0 에서 잰 최고속
            elif fit["v_lower"]:
                v_use = V_GRID[-1]
            a_tr, Ls, rmss, ths = fit_traj(tr_rows, v_use, a_use)
            mags_r = [round(abs(r["delta"])) for r in tr_rows if _traj_xy(r)[2] > 0.2]
            lags, line, traj_ok = Ls, [], True
            out["acc"][acc]["a_traj"] = a_tr
            rep(f"  궤적 피팅 가속도 {a_tr:.0f} 도/s² = {np.radians(a_tr):.1f} rad/s²"
                + (" (상한 = 제한 없음과 구별 안 됨)" if a_tr >= A_TRAJ[-1] * 0.99 else "")
                + f", 최고속 {v_use:.0f} 도/s 고정")
            for mag in MAGS:
                idx = [i for i, mg in enumerate(mags_r) if mg == mag]
                if not idx:
                    continue
                rms = float(np.mean([rmss[i] for i in idx]))
                lim = max(0.5, 0.03 * mag)
                traj_ok &= rms <= lim
                line.append(f"{mag}° {rms:.2f}/{lim:.2f}{'' if rms <= lim else '✗'}")
                out["acc"][acc]["steps"][mag]["traj_rms_deg"] = rms
                out["acc"][acc]["steps"][mag]["lag_ms"] = stats(np.array([Ls[i] for i in idx]) * 1e3)
            sl = stats(np.array(lags) * 1e3)
            out["acc"][acc]["lag_ms"] = sl
            rep(f"  시작 지연 L (궤적 피팅, 명령 송신 → 움직임 시작): {fmt(sl, ' ms', 1)}"
                + "   (펌웨어 '시작 지연' 줄은 0.5도 도달 시각이라 가속 제한이 섞임)")
            rep.check(f"ACC {acc} 궤적 모델 적합 (RAW)", traj_ok, "RMS/허용 도: " + "  ".join(line))
            rep.check(f"ACC {acc} 시작 지연 반복성", sl["n"] >= 2 and sl["ci"] <= TOL["lag_ms"],
                      f"95%CI ±{sl['ci']:.1f} ms (허용 ±{TOL['lag_ms']:.0f})")
    arrs = {}
    for r in rows:
        if "traj" in r and round(abs(r["delta"])) in MAGS:
            k = f"acc{r['acc']}_mag{round(abs(r['delta']))}"
            arrs.setdefault(k, []).append(r["traj"])
    if arrs:
        np.savez(os.path.join(d, "steer_traj.npz"),
                 **{f"{k}_r{i}": tr for k, lst in arrs.items() for i, tr in enumerate(lst)})
        rep(f"  궤적 {sum(len(v) for v in arrs.values())}개 → steer_traj.npz (시뮬에 같은 계단 넣어 비교용: t s, 각도 도, 속도, 부하)")
    both = sorted(set(out["acc"].get(50, {}).get("steps", {})) & set(out["acc"].get(0, {}).get("steps", {})))
    if both:
        big = both[-1]
        out["big_deg"] = big
        rep("\n[ACC 50 vs 0]  (시뮬: 최고속 5.45 rad/s = 312 도/s, 가속 제한 없음)")
        for k, lab in (("t1090", "10→90%"), ("t_settle", "도달")):
            s50, s0 = out["acc"][50]["steps"][big][k], out["acc"][0]["steps"][big][k]
            diff = s50["mean"] - s0["mean"]
            ci = np.hypot(s50["ci"], s0["ci"])
            rep(f"  {big}° {lab}: ACC50 {s50['mean']:.0f} ms, ACC0 {s0['mean']:.0f} ms, 차이 {diff:+.0f} ± {ci:.0f} ms "
                + ("→ 가속 제한 영향 확실" if abs(diff) > ci else "→ 차이 구별 안 됨"))
        out["acc50_vs_0_t1090_ms"] = out["acc"][50]["steps"][big]["t1090"]["mean"] - out["acc"][0]["steps"][big]["t1090"]["mean"]
    rep("")
    return out, rep


STEP_DEG = 360 / 4096


def small_final(r):
    """RAW 고정 기록의 마지막 100 ms 평균 각도."""
    tr = r["traj"]
    tail = tr[:, 0] >= tr[-1, 0] - 0.1
    return float(np.mean(tr[tail, 1]))


def analyze_steer_small(d, rep=None):
    rep = rep or Report()
    rows = [r for r in parse_steer(os.path.join(d, "steer.txt")) if "traj" in r and r.get("role") in
            ("hyst_lo", "hyst_hi", "out", "back")]
    out = dict(kind="steer_small", dir=d)
    rep(f"== 조향 작은 계단 분석: {d}   (RAW STEP {len(rows)}개, 1 스텝 = {STEP_DEG:.3f}도)")
    hdr = next((r["raw_hdr"] for r in rows), {})
    rep(f"  서보 레지스터: 가속도 {hdr.get('acc')}, 데드존 CW {hdr.get('dz_cw')} / CCW {hdr.get('dz_ccw')} 스텝"
        f" (= {hdr.get('dz_cw', 0) * STEP_DEG:.3f} / {hdr.get('dz_ccw', 0) * STEP_DEG:.3f}도)")
    for r in rows:
        r["final"] = small_final(r)
        r["moved"] = abs(r["final"] - r["start"]) >= 0.9 * STEP_DEG
        r["err"] = r["final"] - r["target"]

    # S3 히스테리시스: 0 에 아래(-5)·위(+5)에서 다가가 멈춘 위치
    rep("\n[S3] 히스테리시스: 0도에 -5 쪽 / +5 쪽에서 다가가 멈춘 위치 (서보 출력축 엔코더 기준)")
    reps_ = sorted({r["rep"] for r in rows})
    hs = []
    for k in reps_:
        lo = [r["final"] for r in rows if r["rep"] == k and r["role"] == "hyst_lo"]
        hi = [r["final"] for r in rows if r["rep"] == k and r["role"] == "hyst_hi"]
        if lo and hi:
            hs.append(hi[0] - lo[0])
            rep(f"  반복 {k}: 아래에서 {lo[0]:+.3f}, 위에서 {hi[0]:+.3f}  → 차이 {hi[0] - lo[0]:+.3f}도")
    sh = stats(hs)
    out["hysteresis_deg"] = sh
    sig = sh["n"] >= 2 and sh["mean"] > max(sh["ci"], 1.5 * STEP_DEG)
    rep(f"  히스테리시스 {fmt(sh, '도', 3)} → " + (f"있음: 데드존 ≈ ±{sh['mean'] / 2:.2f}도" if sig
                                                else "1.5 스텝 미만 → 구별 안 됨"))
    rep.check("히스테리시스 반복성", sh["n"] >= 2 and sh["ci"] <= TOL["small_deg"],
              f"95%CI ±{sh['ci']:.3f}도 (허용 ±{TOL['small_deg']})")

    # S2 데드밴드: 0 에서 ±A 명령 → 움직였나, 어디서 멈췄나
    rep("\n[S2] 작은 계단: 0 에서 ±A 로 명령 (out 만).  움직임 = 최종이 시작에서 1 스텝 이상")
    rep("   A 도     움직인 비율     |최종−목표| 중앙값 도 (움직인 것만)")
    amps = sorted({r["amp"] for r in rows if r["role"] == "out"})
    out["small"] = {}
    for a in amps:
        g = [r for r in rows if r["role"] == "out" and r["amp"] == a]
        mv = [r for r in g if r["moved"]]
        e = float(np.median([abs(r["err"]) for r in mv])) if mv else np.nan
        out["small"][a] = dict(moved=len(mv) / len(g), n=len(g), err_med=e)
        rep(f"   {a:4.2f}    {len(mv):2d}/{len(g):2d} ({len(mv) / len(g):4.0%})     {e:.3f}")
    ths = []
    for k in reps_:
        ok = {a: all(r["moved"] for r in rows if r["rep"] == k and r["role"] == "out" and r["amp"] == a) for a in amps}
        th = next((a for i, a in enumerate(amps) if all(ok[b] for b in amps[i:])), np.nan)
        ths.append(th)
    st = stats(ths)
    out["threshold_deg"] = st
    rep(f"  움직이기 시작하는 최소 명령 (반복마다, 그 이상 전부 움직인 가장 작은 A): {fmt(st, '도', 2)}")
    rep(f"  → 시뮬 조향에 명령 데드밴드 ≈ {st['mean'] / 2:.2f}~{st['mean']:.2f}도 (A 격자 사이에 있음)"
        if np.isfinite(st["mean"]) and st["mean"] > amps[0] else "  → 가장 작은 A 부터 움직임: 데드밴드 무시 가능")
    rep.check("최소 명령 반복성", st["n"] >= 2 and st["ci"] <= TOL["small_deg"] * 2,
              f"95%CI ±{st['ci']:.2f}도 (허용 ±{TOL['small_deg'] * 2})")
    rep("  주의: 서보 엔코더는 출력축에 있어 서보 바깥(혼·커플링·조향 링크) 놀음은 여기 안 보인다.")
    rep("        바퀴 쪽 놀음은 손으로 조향축을 좌우로 밀며 P 명령 각도가 안 변하는 범위를 따로 본다.")
    rep("")
    return out, rep


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer, np.bool_)):
        return o.item()
    return o


def analyze(d):
    if os.path.exists(os.path.join(d, "manifest.json")):
        out, rep = analyze_drive(d)
    elif os.path.exists(os.path.join(d, "steer.txt")):
        with open(os.path.join(d, "steer.txt"), encoding="utf-8") as f:
            small = "kind=small" in f.readline()
        out, rep = analyze_steer_small(d) if small else analyze_steer(d)
    else:
        sys.exit(f"{d}: manifest.json(drive) 도 steer.txt 도 없음")
    nf = [c for c in rep.checks if not c["ok"]]
    rep(f"== 종합: 확인 {len(rep.checks)}개 중 FAIL {len(nf)}개")
    for c in nf:
        rep(f"   FAIL {c['name']}: {c['detail']}")
    if not nf:
        rep("   전부 PASS: 반복성·측정조건·모델 검증 통과.  summary.json 값을 시뮬에 반영해도 된다.")
    out["checks"] = rep.checks
    with open(os.path.join(d, "report.txt"), "w") as f:
        f.write("\n".join(rep.lines) + "\n")
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump(_jsonable(out), f, ensure_ascii=False, indent=1)
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", metavar="DIR",
                    help="sysid_uart.py 가 만든 디렉터리 (<시각>_drive / _steer_servoN / _steersmall_servoN). 여러 개 가능")
    args = ap.parse_args(argv)
    n_fail = 0
    for d in args.dirs:
        out = analyze(d.rstrip("/"))
        n_fail += sum(not c["ok"] for c in out["checks"])
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
