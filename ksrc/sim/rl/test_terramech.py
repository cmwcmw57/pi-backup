"""슬립모델 검증: (1) 견인력 곡선이 실제로 나오나 (2) 고착이 재현되나."""
import numpy as np, mujoco
from config import RoverCfg
from mjcf import build
from terramech import Terramechanics, PRESETS

C = RoverCfg()


def run(terrain, throttle, slope_deg=0.0, T=8.0, model=None):
    m = model or mujoco.MjModel.from_xml_string(build(C))
    d = mujoco.MjData(m)
    g = 9.81; th = np.radians(slope_deg)
    m.opt.gravity[:] = [-g*np.sin(th), 0.0, -g*np.cos(th)]   # 경사면 = 중력 기울이기
    tm = Terramechanics(m, C.wheel_r, PRESETS[terrain])
    for _ in range(600):                      # 정착
        tm.apply(d, m.opt.timestep); mujoco.mj_step(m, d)
    tm.reset()
    for w in "fl rl fr rr".split():
        d.ctrl[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"a_wh_{w}")] = throttle
    x0 = d.qpos[0]; acc = []
    for _ in range(int(T/m.opt.timestep)):
        tm.apply(d, m.opt.timestep); mujoco.mj_step(m, d)
        acc.append((tm.slip.mean(), tm.z.mean(), tm.drawbar.mean()))
    a = np.array(acc)[len(acc)//2:]          # 후반부 평균 = 정상상태
    return dict(v=(d.qpos[0]-x0)/T, slip=a[:,0].mean(), z=a[:,1].mean()*1000,
                dp=a[:,2].mean(), dist=d.qpos[0]-x0)

print("=== (1) 평지, 스로틀을 올리면 어떻게 되나 ===")
print(f"{'지형':6s} {'throttle':>8s} {'속도 m/s':>9s} {'슬립':>7s} {'침하mm':>7s} {'DP/N':>7s}")
for terr in ("rock", "sand", "loose"):
    for thr in (0.15, 0.30, 0.60, 1.00):
        r = run(terr, thr)
        print(f"{terr:6s} {thr:8.2f} {r['v']:9.3f} {r['slip']:7.3f} {r['z']:7.1f} {r['dp']:7.3f}")
    print()

print("=== (2) 경사 등판: 스로틀이 셀수록 잘 가나? ===")
for terr, slope in (("rock", 25), ("firm", 20), ("sand", 15), ("sand", 20)):
    print(f"\n--- {terr}, {slope}도 (이론 최대등판: "
          f"rock 41.3 / firm 31.6 / sand 19.3도) ---")
    print(f"{'throttle':>8s} {'8초 이동 m':>10s} {'속도 m/s':>9s} {'슬립':>7s} {'침하mm':>7s}")
    for thr in (0.25, 0.40, 0.60, 0.85, 1.00):
        r = run(terr, thr, slope_deg=slope)
        tag = "  <- 고착" if r['v'] < 0.02 and r['slip'] > 0.5 else ""
        print(f"{thr:8.2f} {r['dist']:10.3f} {r['v']:9.3f} {r['slip']:7.3f} {r['z']:7.1f}{tag}")
