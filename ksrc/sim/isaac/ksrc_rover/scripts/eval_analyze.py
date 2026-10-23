"""eval_log.py 가 덤프한 npz 를 읽어 마찰-보정 지표를 계산한다 (순수 numpy, Isaac 불필요).

단일 파일: 슬립·추종오차·헤딩드리프트·비대칭(Δμ) 요약 + 용량-반응(|Δμ| 구간별 오차).
구간 분리: 명령이 바뀐(또는 리셋된) 뒤 TRANSIENT_S 이내 = 과도, 이후 = 정상. 과도 구간 오차는
  조향 회전·가속 시간이라 정책과 거의 무관(IK·Res8 둘 다 0.117)해서 평균에 섞으면 차이가 묽어진다.
  정상 구간은 미끄러운 바퀴(μ<SLICK_MU) 수별로 다시 나눈다 (mu4 있는 npz 만).
두 파일(baseline, policy): 같은 seed·wheel-seed 로 찍었다는 전제 아래 짝비교 —
  - 추종오차/슬립불균형 baseline vs policy (보정량)
  - |Δμ| 구간별 두 곡선 (비대칭 커질수록 벌어지나)
  - residual(=action) 크기 ↔ Δμ 상관 (정책이 비대칭을 되받아치나)

사용:
  python3 eval_analyze.py res8_pureik.npz                      # 단일
  python3 eval_analyze.py res8_pureik.npz res8_policy.npz      # 짝비교(baseline, policy)
판정은 reward 가 아니라 아래 error_vel / slip 으로 본다.
"""
import sys

import numpy as np

EPS = 0.05  # m/s, 저속 슬립 분모 폭주 방지
DT = 0.02   # s, 정책 스텝 (contract.CONTROL_DT_S, decimation 4 × 5 ms)
TRANSIENT_S = 1.0  # s, 명령 변경 후 과도 구간 길이
SLICK_MU = 0.2     # 이보다 낮은 μ 의 바퀴 = 미끄러운 바퀴 (slick patch 범위 0.1–0.2)


def load(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def valid_mask(done, settle=8):
    """리셋 직후 settle 스텝 + done 스텝을 제외한 유효 마스크 (T,N)."""
    T, N = done.shape
    m = np.ones((T, N), dtype=bool)
    m[done.astype(bool)] = False
    # 리셋(이전 스텝 done) 직후 settle 스텝 제외
    for dt in range(1, settle + 1):
        shifted = np.zeros_like(done, dtype=bool)
        shifted[dt:] = done[:-dt].astype(bool)
        m[shifted] = False
    return m


def compute(d):
    cmd, vel = d["cmd"], d["vel"]            # (T,N,3) [vx,vy,wz]
    steer, ww = d["steer"], d["wheel_w"]      # (T,N,4)
    r = float(d["wheel_radius_m"])
    xy = np.asarray(d["module_xy"], float)    # (4,2) x,y per wheel
    mu_l, mu_r = d["mu_l"], d["mu_r"]          # (T,N)
    m = valid_mask(d["done"])

    err_xy = np.linalg.norm(cmd[..., :2] - vel[..., :2], axis=-1)   # (T,N)
    err_yaw = np.abs(cmd[..., 2] - vel[..., 2])                      # (T,N)

    # per-wheel 종방향 슬립
    th = steer                                   # (T,N,4) 조향각
    x = xy[:, 0][None, None, :]                  # (1,1,4)
    y = xy[:, 1][None, None, :]
    vx = vel[..., 0:1]; vy = vel[..., 1:2]; wz = vel[..., 2:3]      # (T,N,1)
    vwx = vx - wz * y                            # 바퀴 위치 병진속도 (T,N,4)
    vwy = vy + wz * x
    v_long = vwx * np.cos(th) + vwy * np.sin(th)  # 굴림방향 투영
    surf = ww * r                                # 바퀴 표면속도
    denom = np.maximum.reduce([np.abs(surf), np.abs(v_long), np.full_like(surf, EPS)])
    slip = (surf - v_long) / denom               # (T,N,4)
    # 좌(0,2 fl,rl) / 우(1,3 fr,rr) 슬립 불균형
    slip_l = slip[..., [0, 2]].mean(-1)
    slip_r = slip[..., [1, 3]].mean(-1)
    slip_imb = np.abs(slip_l - slip_r)           # (T,N)

    tot = mu_l + mu_r
    dmu = np.where(tot > 0, (mu_r - mu_l) / np.maximum(tot, 1e-6), 0.0)  # (T,N) 비대칭

    act = d["action"]
    act_mag = np.linalg.norm(act, axis=-1)       # (T,N) residual 크기

    return dict(err_xy=err_xy, err_yaw=err_yaw, slip_imb=slip_imb, dmu=dmu,
                act_mag=act_mag, mask=m, smooth=smoothness(d, m),
                t_cmd=time_since_cmd_change(cmd, d["done"]),
                n_slick=(d["mu4"] < SLICK_MU).sum(-1) if "mu4" in d else None)


def time_since_cmd_change(cmd, done, dt=DT):
    """명령이 바뀌거나 리셋된 뒤 지난 시간 [s] (T,N). 첫 스텝은 0."""
    T, N = cmd.shape[:2]
    event = np.ones((T, N), dtype=bool)
    event[1:] = np.any(cmd[1:] != cmd[:-1], axis=-1) | done[:-1].astype(bool)
    t = np.zeros((T, N))
    for k in range(1, T):
        t[k] = np.where(event[k], 0.0, t[k - 1] + dt)
    return t


def phase_masks(c):
    """[(이름, 마스크)] — 전체 / 과도 / 정상 / 정상·미끄러운 바퀴 k개."""
    m, steady = c["mask"], c["t_cmd"] >= TRANSIENT_S
    out = [("전체", m), (f"과도(<{TRANSIENT_S:g}s)", m & ~steady), ("정상", m & steady)]
    if c["n_slick"] is not None:
        out += [(f"정상·미끄러운 바퀴 {k}", m & steady & (c["n_slick"] == k)) for k in range(5)]
    return out


def phase_table(c):
    print(f"  구간별 error_vel_xy / yaw  (과도 = 명령 변경·리셋 후 {TRANSIENT_S:g}s 이내)")
    for name, mk in phase_masks(c):
        if mk.any():
            print(f"    {name:16s} xy {c['err_xy'][mk].mean():.4f}  yaw {c['err_yaw'][mk].mean():.4f}  "
                  f"(비중 {mk.sum() / c['mask'].sum():.0%})")


def smoothness(d, m, dt=DT, flip_rate=0.5):
    """측정 신호 기반 매끄러움 지표 (액션공간이 달라도 비교 가능).

    차분은 양 끝 스텝이 모두 유효할 때만 센다(리셋 경계 제외).
      yaw_acc   : 몸체 yaw 각가속도 RMS [rad/s²]  (SLAM 영상 흐름의 주성분)
      steer_rate: 조향 각속도 RMS [rad/s]
      steer_jerk: 조향각 2차 차분 RMS [rad/s²]  (등속 회전은 0, 떨림만)
      steer_flip: 조향 방향 반전 [회/s/바퀴]  (|각속도|>flip_rate 인 연속 두 스텝의 부호 반대)
      wheel_acc : 바퀴 각가속도 RMS [rad/s²]
      tilt_rate : roll/pitch 각속도 RMS [rad/s]  (참고용 — 지형 영향 큼, 보상 대상 아님)
      side_switch: 반대편 자세 전환 [회/분/바퀴] — 조향이 +60° 이상 ↔ −60° 이하를 오감
                   (θ,v ≡ θ±180°,−v 이중표현. ±90° 가동범위라 한계 근처 80–90° 에서만 가능)
      dual_zone : 조향 |θ|≥80° (한계 근처, 이중표현 가능 구간) 체류 비율
    """
    m1 = m[1:] & m[:-1]
    m2 = m1[1:] & m1[:-1]
    wz = d["vel"][..., 2]
    steer, ww = d["steer"], d["wheel_w"]
    yaw_acc = np.diff(wz, axis=0) / dt
    s_rate = np.diff(steer, axis=0) / dt                 # (T-1,N,4)
    s_jerk = np.diff(steer, n=2, axis=0) / dt**2         # (T-2,N,4)
    w_acc = np.diff(ww, axis=0) / dt
    big = np.abs(s_rate) > flip_rate
    flip = (np.sign(s_rate[1:]) != np.sign(s_rate[:-1])) & big[1:] & big[:-1]   # (T-2,N,4)

    def rms(a, mk):
        return float(np.sqrt(np.mean(a[mk] ** 2)))

    out = dict(
        yaw_acc=rms(yaw_acc, m1),
        steer_rate=rms(s_rate, m1),
        steer_jerk=rms(s_jerk, m2),
        steer_flip=float(flip[m2].mean() / dt),
        wheel_acc=rms(w_acc, m1),
        tilt_rate=float("nan"),
    )
    deg = np.degrees(steer)
    side = np.where(deg > 60, 1, np.where(deg < -60, -1, 0))          # (T,N,4)
    switches, last = 0, np.zeros(side.shape[1:], int)
    for t in range(side.shape[0]):
        last[~m[t]] = 0
        s_t = side[t]
        hit = (s_t != 0) & m[t][:, None]
        switches += int(np.sum(hit & (last != 0) & (s_t != last)))
        last = np.where(hit, s_t, last)
    out["side_switch"] = float(switches / max(m.sum() * dt / 60.0, 1e-9) / 4)
    mm = np.broadcast_to(m[..., None], deg.shape)
    out["dual_zone"] = float(np.mean((np.abs(deg) >= 80)[mm]))
    if "ang" in d:                                       # eval_log 신버전만 roll/pitch 각속도 기록
        out["tilt_rate"] = rms(np.linalg.norm(d["ang"][..., :2], axis=-1), m)
    return out


def summ(name, c):
    m = c["mask"]
    def mean(a):
        return float(np.nanmean(a[m]))
    print(f"\n[{name}]  (유효 스텝 {int(m.sum())})")
    print(f"  error_vel_xy  평균 {mean(c['err_xy']):.4f} m/s")
    print(f"  error_vel_yaw 평균 {mean(c['err_yaw']):.4f} rad/s")
    print(f"  slip 불균형|L-R| 평균 {mean(c['slip_imb']):.4f}")
    print(f"  |Δμ| 평균 {mean(np.abs(c['dmu'])):.4f} (0=대칭)")
    print(f"  residual 크기 평균 {mean(c['act_mag']):.4f}")
    s = c["smooth"]
    print(f"  매끄러움: yaw각가속 {s['yaw_acc']:.3f} rad/s²  조향속도 {s['steer_rate']:.3f} rad/s  "
          f"조향떨림 {s['steer_jerk']:.1f} rad/s²  조향반전 {s['steer_flip']:.3f} /s/바퀴  "
          f"바퀴각가속 {s['wheel_acc']:.2f} rad/s²  roll/pitch속도 {s['tilt_rate']:.3f} rad/s")
    print(f"  이중표현: 반대편 자세 전환 {s['side_switch']:.3f} 회/분/바퀴  |θ|≥80° 체류 {100 * s['dual_zone']:.0f}%")


def dose_response(c, label, nb=4):
    m = c["mask"]
    admu = np.abs(c["dmu"])[m]; exy = c["err_xy"][m]; eyaw = c["err_yaw"][m]
    qs = np.quantile(admu, np.linspace(0, 1, nb + 1))
    print(f"  {label}: |Δμ| 구간별 error_vel_xy / yaw")
    for i in range(nb):
        sel = (admu >= qs[i]) & (admu <= qs[i + 1] if i == nb - 1 else admu < qs[i + 1])
        if sel.sum() == 0:
            continue
        print(f"    |Δμ|∈[{qs[i]:.3f},{qs[i+1]:.3f}]  xy {exy[sel].mean():.4f}  yaw {eyaw[sel].mean():.4f}  (n={int(sel.sum())})")


def dose_mu_min(d, c, nb=4):
    """바퀴 4개 중 최소 μ 구간별 오차 (custom μ 로 찍은 npz 만 mu4 가 있음)."""
    if "mu4" not in d or np.isnan(d["mu4"]).all():
        return
    m = c["mask"]
    mumin = np.nanmin(d["mu4"], axis=-1)[m]
    exy, eyaw, sl = c["err_xy"][m], c["err_yaw"][m], c["slip_imb"][m]
    qs = np.quantile(mumin, np.linspace(0, 1, nb + 1))
    print("  최소 μ 구간별 error_vel_xy / yaw / slip불균형")
    for i in range(nb):
        sel = (mumin >= qs[i]) & (mumin <= qs[i + 1] if i == nb - 1 else mumin < qs[i + 1])
        if sel.sum():
            print(f"    μmin∈[{qs[i]:.2f},{qs[i+1]:.2f}]  xy {exy[sel].mean():.4f}  yaw {eyaw[sel].mean():.4f}  slip {sl[sel].mean():.3f}")


def main(paths):
    ds = [load(p) for p in paths]
    cs = [(p.split("/")[-1], compute(d)) for p, d in zip(paths, ds)]
    for d, (name, c) in zip(ds, cs):
        summ(name, c)
        phase_table(c)
        dose_response(c, name)
        dose_mu_min(d, c)

    if len(cs) == 2:
        (bn, bc), (pn, pc) = cs
        print(f"\n=== 짝비교: {bn}(baseline) vs {pn}(policy) ===")
        mb, mp = bc["mask"], pc["mask"]
        bx, px = np.nanmean(bc["err_xy"][mb]), np.nanmean(pc["err_xy"][mp])
        by, py = np.nanmean(bc["err_yaw"][mb]), np.nanmean(pc["err_yaw"][mp])
        bs, ps = np.nanmean(bc["slip_imb"][mb]), np.nanmean(pc["slip_imb"][mp])
        print(f"  error_vel_xy : {bx:.4f} -> {px:.4f}  (보정 {100*(bx-px)/max(bx,1e-6):+.1f}%)")
        print(f"  error_vel_yaw: {by:.4f} -> {py:.4f}  (보정 {100*(by-py)/max(by,1e-6):+.1f}%)")
        print("  구간별 (baseline -> policy, 보정%):")
        for (name, kb), (_, kp) in zip(phase_masks(bc), phase_masks(pc)):
            if not (kb.any() and kp.any()):
                continue
            row = []
            for k in ("err_xy", "err_yaw"):
                b, p_ = bc[k][kb].mean(), pc[k][kp].mean()
                row.append(f"{k[4:]} {b:.4f} -> {p_:.4f} ({100*(b-p_)/max(b,1e-6):+.0f}%)")
            print(f"    {name:16s} " + "   ".join(row))
        print(f"  slip 불균형   : {bs:.4f} -> {ps:.4f}  (감소 {100*(bs-ps)/max(bs,1e-6):+.1f}%)")
        for k, lab in (("yaw_acc", "yaw각가속"), ("steer_jerk", "조향떨림"),
                       ("steer_flip", "조향반전"), ("wheel_acc", "바퀴각가속"),
                       ("side_switch", "반대편전환")):
            b, p = bc["smooth"][k], pc["smooth"][k]
            rel = f"감소 {100*(b-p)/b:+.1f}%" if b > 1e-2 else "기준≈0, % 생략"
            print(f"  {lab:8s}: {b:.3f} -> {p:.3f}  ({rel})")
        # 정책 residual ↔ Δμ 상관 (절대 비대칭이 클수록 더 개입하나)
        admu = np.abs(pc["dmu"])[mp]; am = pc["act_mag"][mp]
        if admu.std() > 1e-6 and am.std() > 1e-6:
            r = np.corrcoef(admu, am)[0, 1]
            print(f"  residual크기 ↔ |Δμ| 상관 {r:+.3f}  (양수=비대칭 클수록 더 보정)")
        print("  판독: 비대칭 구간에서 baseline xy/yaw 가 크고 policy 가 낮으면 = 마찰 보정됨.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
