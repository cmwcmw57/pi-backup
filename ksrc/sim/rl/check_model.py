"""모델 물리 검증: 정착 / 디퍼렌셜 / 구동 / 비틀림접지 / **횡경사 유지**.

[5] 횡경사는 나중에 추가된 항목인데, 이유가 있다. 앞의 네 검사는 전부
종방향만 본다 -- 그래서 "슬립모델이 횡방향 접지력을 통째로 날려서 로버가
10도 횡경사에서 8초에 70m 미끄러지는" 버그가 **네 검사를 모두 통과했다**.
견인력 곡선도, 고착 재현도, 학습 커리큘럼도 다 정상으로 보였다.
한 축만 보는 검증은 그 축 밖의 버그를 절대 못 잡는다.
"""
import numpy as np, mujoco
from config import RoverCfg
from mjcf import build
from terramech import Terramechanics, PRESETS

C = RoverCfg()
WHEELS = ["fl", "rl", "fr", "rr"]


def wheel_normal_forces(m, d):
    """각 바퀴 지오메트리가 받는 접촉 수직력."""
    gid = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"wheel_{w}"): w for w in WHEELS}
    # geom 이름이 없으면 body 로 역추적
    out = dict.fromkeys(WHEELS, 0.0)
    f6 = np.zeros(6)
    for i in range(d.ncon):
        con = d.contact[i]
        mujoco.mj_contactForce(m, d, i, f6)
        for g in (con.geom1, con.geom2):
            bid = m.geom_bodyid[g]
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if nm.startswith("wheel_"):
                out[nm[6:]] += abs(f6[0])
    return out


def settle(m, d, t=2.5):
    for _ in range(int(t / m.opt.timestep)):
        mujoco.mj_step(m, d)


def jid(m, n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)

xml = build(C)
m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
print(f"총질량 {m.body_mass.sum():.3f} kg   (cfg {C.total_mass:.3f})")

# 1) 평지 정착
mujoco.mj_resetData(m, d); settle(m, d)
qa = {n: d.qpos[m.jnt_qposadr[jid(m, n)]] for n in ("rock_l", "rock_r")}
F = wheel_normal_forces(m, d)
print(f"\n[1] 평지 정착   차체높이 {d.qpos[2]*1000:6.1f} mm   "
      f"roll/pitch {np.degrees(np.arctan2(2*(d.qpos[3]*d.qpos[4]+d.qpos[5]*d.qpos[6]),1-2*(d.qpos[4]**2+d.qpos[5]**2))):+.2f} / "
      f"{np.degrees(np.arcsin(np.clip(2*(d.qpos[3]*d.qpos[5]-d.qpos[6]*d.qpos[4]),-1,1))):+.2f} deg")
print(f"    로커각 L{np.degrees(qa['rock_l']):+6.2f}  R{np.degrees(qa['rock_r']):+6.2f} deg")
print(f"    접촉력 " + "  ".join(f"{w}={F[w]:5.2f}N" for w in WHEELS) +
      f"   합 {sum(F.values()):5.2f}N  (무게 {m.body_mass.sum()*9.81:.2f}N)")

# 2) 디퍼렌셜: 로커를 강제로 꺾고 반대쪽이 따라오는지
mujoco.mj_resetData(m, d)
d.qpos[m.jnt_qposadr[jid(m, "rock_l")]] = np.radians(15)
mujoco.mj_forward(m, d)
for _ in range(400): mujoco.mj_step(m, d)
L = np.degrees(d.qpos[m.jnt_qposadr[jid(m, 'rock_l')]])
R = np.degrees(d.qpos[m.jnt_qposadr[jid(m, 'rock_r')]])
print(f"\n[2] 디퍼렌셜   L{L:+7.3f}  R{R:+7.3f} deg   L+R={L+R:+.4f} (0 이어야 정상)")

# 3) 전진 구동: 풀 스로틀 정상속도
mujoco.mj_resetData(m, d); settle(m, d, 1.0)
for w in WHEELS:
    d.ctrl[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"a_wh_{w}")] = 1.0
x0, t0 = d.qpos[0], d.time
settle(m, d, 4.0)
v = (d.qpos[0] - x0) / (d.time - t0)
wv = np.mean([d.qvel[m.jnt_dofadr[jid(m, f"wh_{w}")]] for w in WHEELS])
print(f"\n[3] 풀스로틀   차체속도 {v:.3f} m/s   바퀴각속도 {wv:.2f} rad/s "
      f"({wv*60/(2*np.pi):.0f} rpm)   이론 무부하 {C.mot_w_noload:.2f} rad/s / {C.mot_w_noload*C.wheel_r:.3f} m/s")

# 4) 비틀림: 앞왼쪽 바퀴 밑에 블록 -> 4륜 접지 유지되나
#
# 예전엔 40mm 를 썼는데 그건 **기구 한계를 넘는다**.  다리/모터 하우징은 바퀴와
# 같은 좌우 위치(73.5~126.7mm vs 바퀴 100.9~135.9mm)에 지상고 9.5mm 로 있어서,
# 바퀴 밑 40mm 턱은 반드시 하우징에 부딪힌다 (실측 접촉력 3271N = 무게의 147배).
# 그러면 기구가 잠겨 디퍼렌셜 자체를 검증할 수 없다.
#
# 턱 높이를 쓸어보면 경계가 **14~16mm** 다: 14mm 까지는 로커가 ±2.19도 꺾이며
# 4륜 접지를 유지하고, 16mm 부터 하우징이 박힌다(접촉력 2767N).  지상고 9.5mm
# 로 14mm 를 넘는 것은 로커 articulation 이 차체를 들어주기 때문이고, 그게 이
# 서스펜션의 존재 이유다.  검증은 그 한계 안쪽(12mm)에서 한다.
BUMP_H = 0.006   # 반높이 -> 12mm 턱
xml2 = xml.replace('<body name="chassis"',
   '<geom name="bump" type="box" pos="%.4f %.4f %.4f" size="0.05 0.05 %.4f" '
   'rgba="0.7 0.2 0.2 1"/>\n    <body name="chassis"'
   % (C.axle_x, C.track/2, BUMP_H, BUMP_H))
m2 = mujoco.MjModel.from_xml_string(xml2); d2 = mujoco.MjData(m2)
settle(m2, d2, 3.0)
F2 = wheel_normal_forces(m2, d2)
L2 = np.degrees(d2.qpos[m2.jnt_qposadr[jid(m2,'rock_l')]]); R2 = np.degrees(d2.qpos[m2.jnt_qposadr[jid(m2,'rock_r')]])
print(f"\n[4] 비틀림(앞왼쪽 {BUMP_H*2000:.0f}mm 턱, 기구 한계 14~16mm)  "
      f"로커 L{L2:+6.2f} R{R2:+6.2f} deg")
print(f"    접촉력 " + "  ".join(f"{w}={F2[w]:5.2f}N" for w in WHEELS))
lifted = [w for w in WHEELS if F2[w] < 0.5]
print("    => " + ("모든 바퀴 접지 유지 OK" if not lifted else f"뜬 바퀴: {lifted}  (접지 실패)"))


# 5) 횡경사 유지 -- 중력을 옆으로 기울이고 속도명령 0 에서 버티는지
#    슬립모델을 켠 상태로 본다. 버그가 있던 곳이 바로 거기다.
print("\n[5] 횡경사 유지 (슬립모델 ON, 속도명령 0, 8초)")
print(f"    {'지형':6s} {'경사':>5s} {'횡변위':>10s} {'mu':>6s} {'이론한계':>8s}  판정")
ok_all = True
for terr in ("rock", "sand"):
    prm = PRESETS[terr]
    lim = np.degrees(np.arctan(prm.mu_lat * prm.mu_max))   # 정지 시 유지 가능한 최대 횡경사
    for roll in (10, 20, 30):
        m5 = mujoco.MjModel.from_xml_string(xml)
        d5 = mujoco.MjData(m5)
        g = 9.81; th = np.radians(roll)
        m5.opt.gravity[:] = [0.0, -g*np.sin(th), -g*np.cos(th)]
        tm5 = Terramechanics(m5, C.wheel_r, prm)
        for _ in range(600):
            tm5.apply(d5, m5.opt.timestep); mujoco.mj_step(m5, d5)
        y0 = d5.qpos[1]
        for _ in range(int(8.0/m5.opt.timestep)):
            tm5.apply(d5, m5.opt.timestep); mujoco.mj_step(m5, d5)
        slid = abs(d5.qpos[1] - y0) * 1000.0
        expect_hold = roll < lim
        held = slid < 50.0                       # 8초에 5cm 미만이면 "버텼다"
        ok = (held == expect_hold)
        ok_all &= ok
        tag = ("유지 OK" if held else "미끄러짐") + ("" if ok else "   <- 예상과 다름")
        print(f"    {terr:6s} {roll:4d}도 {slid:9.1f}mm {m5.geom_friction[tm5.wg[0],0]:6.3f} "
              f"{lim:7.1f}도  {tag}")
print("    => " + ("횡방향 접지 정상" if ok_all else "!! 횡방향 접지 이상 !!"))


# ---------------------------------------------------------------------------
# 아래는 "전진만 본다" 는 사각지대를 메우려고 추가한 검사들이다.
# [5] 를 만든 교훈과 같다: 한 번도 명령해보지 않은 방향에는 버그가 숨는다.
# 여기서는 실제 펌웨어 C 기구학(ksrc_common)을 거쳐 구동한다 -- 파이썬으로
# 각도를 직접 써넣으면 IK 경로 자체를 검증하지 못한다.
# ---------------------------------------------------------------------------
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import ksrc_common as kc

MJ_W = ("fl", "rl", "fr", "rr")
C_ORD = ("fl", "fr", "rl", "rr")
C2MJ = [MJ_W.index(w) for w in C_ORD]


def drive(cmd, terrain="rock", T=6.0, settle=1.2, yaw_torque=0.0):
    """펌웨어 C IK 로 (vx,vy,omega) 를 구동하고 달성된 차체 속도를 돌려준다."""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    tm = Terramechanics(m, C.wheel_r, PRESETS[terrain])
    mods = kc._make_modules(half_w=C.track / 2, half_l=C.axle_x)
    st = (kc.SwerveModuleState * 4)()
    vmax = C.mot_w_noload * C.wheel_r
    aid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    j_st = [m.jnt_qposadr[jid(f"st_{w}")] for w in MJ_W]
    chassis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "chassis")

    for _ in range(int(settle / m.opt.timestep)):
        tm.apply(d, m.opt.timestep); mujoco.mj_step(m, d)

    out = kc.swerve_ik_compute(cmd[0], cmd[1], cmd[2], mods, vmax, st)
    for ci, mi in enumerate(C2MJ):
        d.ctrl[aid(f"a_st_{MJ_W[mi]}")] = out[ci].angle_rad
        d.ctrl[aid(f"a_wh_{MJ_W[mi]}")] = np.clip(out[ci].speed_mps / vmax, -1, 1)

    # 명령을 준 뒤 조향이 자리잡을 때까지(약 0.5s) 기다렸다가 측정을 시작한다.
    # 안 그러면 조향이 0 -> 90도로 움직이는 동안의 과도현상이 그대로 섞인다.
    for _ in range(int(0.5 / m.opt.timestep)):
        tm.apply(d, m.opt.timestep); mujoco.mj_step(m, d)

    def _yaw():
        q = d.qpos[3:7]
        return np.arctan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]**2+q[3]**2))

    p0 = d.qpos[:3].copy(); yaw0 = _yaw(); t0 = d.time
    # yaw 는 **증분을 누적**한다. 시작/끝만 비교하면 한 바퀴(2pi)를 넘는 순간
    # atan2 가 접어버려서 부호까지 뒤집힌 값이 나온다 (실제로 겪음).
    yaw_acc = 0.0; yprev = yaw0
    for _ in range(int(T / m.opt.timestep)):
        tm.apply(d, m.opt.timestep)
        if yaw_torque: d.xfrc_applied[chassis, 5] += yaw_torque
        mujoco.mj_step(m, d)
        yc = _yaw(); dy = yc - yprev
        yaw_acc += np.arctan2(np.sin(dy), np.cos(dy)); yprev = yc
    dt = d.time - t0
    dp = d.qpos[:3] - p0
    c, s_ = np.cos(yaw0), np.sin(yaw0)
    return dict(vx=(c*dp[0]+s_*dp[1])/dt, vy=(-s_*dp[0]+c*dp[1])/dt,
                om=yaw_acc/dt, steer=np.degrees([d.qpos[j] for j in j_st]))


VMAX = C.mot_w_noload * C.wheel_r
print(f"\n[6] 방향별 명령 추종 (rock, 6초).  최대 바퀴속도 {VMAX:.3f} m/s")
print(f"    {'명령':22s} {'달성 vx':>8s} {'vy':>8s} {'omega':>8s}   {'조향각 4개':>26s}  판정")
CASES = [("전진   vx=+0.20", (0.20, 0.0, 0.0)),
         ("후진   vx=-0.20", (-0.20, 0.0, 0.0)),
         ("게걸음 vy=+0.20", (0.0, 0.20, 0.0)),
         ("게걸음 vy=-0.20", (0.0, -0.20, 0.0)),
         ("대각   vx=vy=0.14", (0.14, 0.14, 0.0)),
         ("제자리 ω=+1.0", (0.0, 0.0, 1.0))]
ok6 = True
for lab, cmd in CASES:
    r = drive(cmd, "rock")
    # 병진은 절대오차(m/s), 회전은 상대오차로 본다. 예전엔 omega 에 0.05 를
    # 곱해서 사실상 검사를 안 하고 있었다 (명령 +1.0 에 달성 -0.17 이 OK 로 통과).
    e_lin = max(abs(r["vx"]-cmd[0]), abs(r["vy"]-cmd[1]))
    e_rot = abs(r["om"]-cmd[2]) / max(abs(cmd[2]), 1e-6) if abs(cmd[2]) > 1e-6 \
            else abs(r["om"])
    good = e_lin < 0.04 and e_rot < 0.25
    err = f"lin {e_lin:.3f} / rot {e_rot:.2f}"
    ok6 &= good
    print(f"    {lab:22s} {r['vx']:+8.3f} {r['vy']:+8.3f} {r['om']:+8.3f}   "
          f"{np.array2string(r['steer'], precision=0, floatmode='fixed'):>26s}  "
          f"{'OK' if good else err}")
print("    => " + ("전 방향 추종 OK" if ok6 else "!! 일부 방향 추종 실패 !!"))

print("\n[7] 요 외란 저항 (속도명령 0, 외부 요 토크 0.5 N·m, 6초)")
r0 = drive((0, 0, 0), "rock", yaw_torque=0.0)
r1 = drive((0, 0, 0), "rock", yaw_torque=0.5)
print(f"    토크 없음  회전 {np.degrees(r0['om'])*6:+7.2f}도")
print(f"    토크 인가  회전 {np.degrees(r1['om'])*6:+7.2f}도")
print("    => " + ("요 외란에 저항함" if abs(np.degrees(r1['om'])*6) < 25
                   else "!! 자유회전 -- 횡방향 접지 의심 !!"))

print("\n[8] 조향 스텝 응답 (0 -> 60도, 무부하)")
m8 = mujoco.MjModel.from_xml_string(xml); d8 = mujoco.MjData(m8)
tm8 = Terramechanics(m8, C.wheel_r, PRESETS["rock"])
for _ in range(600): tm8.apply(d8, m8.opt.timestep); mujoco.mj_step(m8, d8)
tgt = np.radians(60.0)
for w in WHEELS: d8.ctrl[mujoco.mj_name2id(m8, mujoco.mjtObj.mjOBJ_ACTUATOR, f"a_st_{w}")] = tgt
ja = m8.jnt_qposadr[mujoco.mj_name2id(m8, mujoco.mjtObj.mjOBJ_JOINT, "st_fl")]
t_rise = None; t0 = d8.time
for _ in range(int(3.0/m8.opt.timestep)):
    tm8.apply(d8, m8.opt.timestep); mujoco.mj_step(m8, d8)
    if t_rise is None and d8.qpos[ja] > 0.9*tgt: t_rise = d8.time - t0
err_ss = np.degrees(tgt - d8.qpos[ja])
print(f"    90% 도달 {('%.3f s' % t_rise) if t_rise else '3초 내 미달'}   정상상태 오차 {err_ss:+.2f}도")
print("    => " + ("조향 응답 OK" if (t_rise and t_rise < 1.0 and abs(err_ss) < 3.0)
                   else "!! 조향이 느리거나 오차가 큼 -- kp/forcerange 확인 !!"))


# 9) 누적 조향 드리프트 -- "시간축" 검사
#    [6] 방향별 추종은 매번 새 state 에서 명령을 **하나만** 준다. 그래서 IK 의
#    last_angle_rad 가 누적될 기회가 없고, 실제로 이 검사들을 전부 통과한 채로
#    "긴 주행에서 조향각이 +-90도를 벗어나 잘린다" 는 버그가 살아 있었다
#    (실제 경기장 지형에서 명령의 67%, 평균 오차 76.5도).
#    공간축([5] 횡경사)과 마찬가지로, 보지 않는 축에는 버그가 숨는다.
_lo9 = [C.steer_range_rad(w)[0] for w in C.C_STEER_ORDER]
_hi9 = [C.steer_range_rad(w)[1] for w in C.C_STEER_ORDER]
print(f"\n[9] 누적 조향 드리프트 (연속 명령 2000스텝, 바퀴별 가동범위 "
      f"{list(C.steer_lo_deg)} ~ {list(C.steer_hi_deg)}도)")
_mods = kc._make_modules(half_w=C.track / 2, half_l=C.axle_x)
_st = (kc.SwerveModuleState * 4)()
_vmax = C.mot_w_noload * C.wheel_r
_rng = np.random.default_rng(0)
raw, folded = [], []
_dir_err = 0.0
_pos = np.zeros(2); _yaw = 0.0
_goal = np.array([2.0, 1.0])
for _k in range(2000):
    to = _goal - _pos; dist = float(np.linalg.norm(to))
    if dist < 0.25:
        _goal = np.array([_rng.uniform(-3, 3), _rng.uniform(-3, 3)])
        to = _goal - _pos; dist = float(np.linalg.norm(to))
    cs, sn = np.cos(_yaw), np.sin(_yaw)
    vb = np.array([cs*to[0] + sn*to[1], -sn*to[0] + cs*to[1]]) / max(dist, 1e-6) \
         * min(0.7*_vmax, 1.5*dist)
    om = float(np.clip(1.2*np.arctan2(vb[1], vb[0] + 1e-6), -0.8, 0.8))
    o9 = kc.swerve_ik_compute(vb[0], vb[1], om, _mods, _vmax, _st)
    a_raw = np.degrees([o9[i].angle_rad for i in range(4)])
    v_raw = [(o9[i].speed_mps * np.cos(o9[i].angle_rad),
              o9[i].speed_mps * np.sin(o9[i].angle_rad)) for i in range(4)]
    raw.append(a_raw)
    # 펌웨어/env 와 **같은 C 함수**로 바퀴별 범위 안에 접는다
    kc.swerve_fold_to_range(_lo9, _hi9, _st, o9)
    folded.append(np.degrees([o9[i].angle_rad for i in range(4)]))
    # 접어도 바닥 속도벡터(방향·크기)는 같아야 한다 (각 +180 이면 속도 부호 반전)
    for i in range(4):
        vx_f = o9[i].speed_mps * np.cos(o9[i].angle_rad)
        vy_f = o9[i].speed_mps * np.sin(o9[i].angle_rad)
        _dir_err = max(_dir_err, abs(vx_f - v_raw[i][0]), abs(vy_f - v_raw[i][1]))
    _pos += np.array([cs*vb[0] - sn*vb[1], sn*vb[0] + cs*vb[1]]) * 0.02
    _yaw += om * 0.02
raw = np.array(raw); folded = np.array(folded)
_lo9d = np.array(C.steer_lo_deg); _hi9d = np.array(C.steer_hi_deg)
out_raw = 100.0 * np.mean((raw < _lo9d) | (raw > _hi9d))
clip_err = float(np.mean(np.abs(raw - np.clip(raw, _lo9d, _hi9d))))
ok9 = bool(((folded >= _lo9d - 1e-3) & (folded <= _hi9d + 1e-3)).all() and _dir_err < 1e-4)
print(f"    IK 원시각 범위 {raw.min():8.1f} ~ {raw.max():8.1f}도   범위 밖 {out_raw:5.1f}%")
print(f"    그냥 잘랐을 때 평균 오차 {clip_err:6.1f}도   (이게 0 이 아니면 클립은 쓰면 안 됨)")
print(f"    180도 등가 접기 후 범위 {folded.min():7.1f} ~ {folded.max():7.1f}도   "
      f"바닥 속도벡터 최대 오차 {_dir_err*1000:.3f} mm/s")
print("    => " + ("접기 후 전부 서보 범위 안 OK" if ok9 else "!! 접기 실패 !!"))
