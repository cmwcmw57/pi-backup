"""IMU 실측 분석 (2026-10-15): 시뮬 student 관측의 IMU 잡음·진동·바이어스·장착 기울기를 정하기 위한 것.

입력: run_policy.py --log 가 저장한 npz (원시 텔레메트리 포함, 10-15 이후 형식) 여러 개.
  각 파일을 "정지" 구간(명령 0, 바퀴 거의 0)과 "주행" 구간으로 나눠
  - imu_n 분포 (펌웨어 20 ms 평균이 동작하는지: 3–4 정상, 0 = 순간값)
  - 자이로: 바이어스(정지 평균), 잡음 std (정지 / 주행)            → 시뮬 STUDENT_NOISE imu_gyro (0.01 rad/s), 바이어스 DR ±0.01
  - 가속도: 중력을 뺀 몸체 가속도 std (정지 / 주행), 크기 |a|-g  → 시뮬 STUDENT_NOISE imu_acc (0.05 m/s²), v2a 가속도 관측 잡음
  - 주행 중 가속도 스펙트럼 최대 주파수 vs 그라우저 14개 주파수 (바퀴 회전수 × 14) → 진동 모델
  - 정지 자세의 중력 방향 기울기 (장착·바닥 기울기)               → 시뮬 GRAV_ERR (3°)
  - 루프 주기 (host_ns 간격) p50/p99/최대                            → 50 Hz 유지 여부
을 내고, 시뮬에 넣을 값 제안을 JSON 으로 저장한다.

  python3 pi/common/imu_noise.py logs/run_*.npz --out imu_noise_summary.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sensor_fusion import GRAVITY, WHEEL_RADIUS_M, GravityFilter  # noqa: E402

GROUSERS = 14
SIM = dict(imu_gyro=0.01, imu_acc=0.05, gyro_bias=0.01, grav_err_deg=3.0)


def load(path):
    d = np.load(path, allow_pickle=False)
    need = ("acc_g", "gyro_dps", "wheel_mps", "cmd", "host_ns")
    miss = [k for k in need if k not in d.files]
    if miss:
        raise SystemExit(f"{path}: {miss} 없음 -- 10-15 이후 run_policy.py --log 로 다시 기록하세요")
    return {k: d[k] for k in d.files if k != "meta"}


def analyze(d, dt=0.02, still_s=1.0):
    acc = d["acc_g"].astype(float) * GRAVITY                    # (T,3) 로버 좌표, 정지 시 +9.81 z
    gyro = np.radians(d["gyro_dps"].astype(float))
    wheel = np.abs(d["wheel_mps"].astype(float)).mean(1)        # (T,) 평균 바퀴 표면 속도
    cmd = np.linalg.norm(d["cmd"].astype(float), axis=1)
    still = (cmd < 1e-3) & (wheel < 0.005)
    moving = wheel > 0.03
    n_still0 = int(still_s / dt)
    bias = gyro[still][:n_still0].mean(0) if still.sum() >= 10 else np.zeros(3)
    gf = GravityFilter(dt=dt)
    g = np.array([gf.step(w - bias, a) for w, a in zip(gyro, acc)])   # 몸체 좌표 중력 방향 (정자세 (0,0,-1))
    lin = acc + GRAVITY * g                                          # 중력을 뺀 몸체 가속도 [m/s²]
    out = {"T": int(len(acc)), "still_steps": int(still.sum()), "moving_steps": int(moving.sum())}
    if "imu_n" in d:
        n = d["imu_n"].astype(int)
        out["imu_n"] = {str(k): int((n == k).sum()) for k in np.unique(n)}
    out["gyro_bias_rad_s"] = bias.round(5).tolist()
    for name, m in (("still", still), ("moving", moving)):
        if m.sum() < 25:
            continue
        out[f"gyro_std_{name}"] = gyro[m].std(0).round(5).tolist()
        out[f"lin_acc_std_{name}"] = lin[m].std(0).round(4).tolist()
        out[f"acc_norm_minus_g_{name}"] = float(np.abs(np.linalg.norm(acc[m], axis=1) - GRAVITY).mean())
    if still.sum() >= 25:
        gs = g[still].mean(0)
        out["tilt_still_deg"] = float(np.degrees(np.arccos(np.clip(-gs[2] / np.linalg.norm(gs), -1, 1))))
    if moving.sum() >= 64:
        x = lin[moving] - lin[moving].mean(0)
        f = np.fft.rfftfreq(len(x), dt)
        p = (np.abs(np.fft.rfft(x, axis=0)) ** 2).sum(1)
        p[f < 1.0] = 0
        out["moving_acc_peak_hz"] = float(f[np.argmax(p)])
        w_rev = float(np.median(wheel[moving])) / (2 * math.pi * WHEEL_RADIUS_M)
        out["grouser_hz_expected"] = round(w_rev * GROUSERS, 2)
        out["nyquist_hz"] = 1 / (2 * dt)
    if "host_ns" in d and len(d["host_ns"]) > 2:
        per = np.diff(d["host_ns"].astype(np.int64))[1:] / 1e6
        out["loop_ms"] = {"p50": float(np.median(per)), "p99": float(np.quantile(per, .99)), "max": float(per.max()),
                          "over25": int((per > 25).sum())}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", default="imu_noise_summary.json")
    a = ap.parse_args()
    res = {}
    for p in a.logs:
        r = analyze(load(p)); res[Path(p).name] = r
        print(f"\n== {p}")
        for k, v in r.items():
            print(f"  {k}: {v}")
    mv = [r for r in res.values() if "lin_acc_std_moving" in r]
    st = [r for r in res.values() if "gyro_std_still" in r]
    sug = {}
    if st:
        sug["imu_gyro_noise_rad_s"] = float(np.max([np.max(r["gyro_std_still"]) for r in st]))
        sug["gyro_bias_rad_s"] = float(np.max([np.max(np.abs(r["gyro_bias_rad_s"])) for r in st]))
        sug["tilt_deg"] = float(np.max([r.get("tilt_still_deg", 0) for r in st]))
    if mv:
        sug["lin_acc_noise_moving_m_s2"] = float(np.max([np.max(r["lin_acc_std_moving"]) for r in mv]))
        sug["gyro_noise_moving_rad_s"] = float(np.max([np.max(r["gyro_std_moving"]) for r in mv]))
    print("\n== 시뮬 비교 (현재 시뮬 값 -> 실측)")
    for k, s in (("imu_gyro_noise_rad_s", SIM["imu_gyro"]), ("gyro_bias_rad_s", SIM["gyro_bias"]),
                 ("tilt_deg", SIM["grav_err_deg"]), ("lin_acc_noise_moving_m_s2", SIM["imu_acc"])):
        if k in sug:
            print(f"  {k}: 시뮬 {s}  실측 {sug[k]:.4f}  {'(시뮬이 더 큼: OK)' if s >= sug[k] else '(실측이 더 큼: 시뮬 키울 것)'}")
    json.dump({"files": res, "suggest": sug}, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n저장 {a.out}")


if __name__ == "__main__":
    main()
