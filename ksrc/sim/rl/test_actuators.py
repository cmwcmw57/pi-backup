"""Isaac 액추에이터 모델(ksrc_rover/utils/actuators_torch.py)이 기준과 같은지 (노트북, GPU 불필요).

  1. SteerServo == mdp/actions.py 의 _servo_step (10-13 서보 모델, 리팩터 전 코드를 아래에 그대로 둠)
  2. WheelPI (torch) == firmware/common/wheel_pi.c (sim/libksrc_common.so) -- 같은 입력열에 같은 duty
  3. encoder_speed: 20 ms 창 1틱 = 0.0375 rad/s

    bash sim/build.sh && python3 sim/rl/test_actuators.py
"""

import math
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sim/isaac/ksrc_rover/source"))
sys.path.insert(0, str(ROOT / "sim"))

from ksrc_common import WheelPi  # noqa: E402
from ksrc_rover.utils.actuators_torch import (  # noqa: E402
    DriveGate, SteerServo, WheelPI, drive_torque, encoder_speed, encoder_ticks, wheel_friction)
from ksrc_rover.utils.sensors_torch import EncoderSpeed, GravityFilter, servo_angle  # noqa: E402
sys.path.insert(0, str(ROOT / "pi/common"))
import sensor_fusion as sf  # noqa: E402
from ksrc_rover.utils.contract import (  # noqa: E402
    STEER_ACCEL_RAD_S2, STEER_DEADBAND_RAD, STEER_DELAY_S, STEER_MAX_RAD_S)

FAIL = []


def check(ok, msg):
    print(("ok   " if ok else "FAIL ") + msg)
    if not ok:
        FAIL.append(msg)


class OldServo:
    """mdp/actions.py RoverAction._servo_step as of 2026-10-13 (before moving to SteerServo), verbatim."""

    def __init__(self, n, dt):
        self.num_envs, self._phys_dt = n, dt
        self._steer_applied = torch.zeros(n, 4)
        self._steer_vel = torch.zeros(n, 4)
        self._goal_hist = torch.zeros(int(round(0.06 / dt)) + 1, n, 4)
        self.steer_delay_ticks = torch.full((n,), int(round(STEER_DELAY_S / dt)), dtype=torch.long)
        self.steer_deadband = torch.full((n, 1), STEER_DEADBAND_RAD)
        self.steer_rate_scale = torch.ones(n, 1)
        self._processed = torch.zeros(n, 8)

    def _servo_step(self):
        dt = self._phys_dt
        self._goal_hist = torch.roll(self._goal_hist, 1, dims=0)
        self._goal_hist[0] = self._processed[:, :4]
        d = self.steer_delay_ticks.clamp(max=self._goal_hist.shape[0] - 1)
        goal = self._goal_hist[d, torch.arange(self.num_envs)]
        err = goal - self._steer_applied
        err = torch.where(err.abs() <= self.steer_deadband, torch.zeros_like(err), err)
        a = STEER_ACCEL_RAD_S2 * self.steer_rate_scale
        v_des = torch.sign(err) * torch.minimum(torch.sqrt(2.0 * a * err.abs()), STEER_MAX_RAD_S * self.steer_rate_scale)
        self._steer_vel += torch.clamp(v_des - self._steer_vel, -a * dt, a * dt)
        pos = self._steer_applied + self._steer_vel * dt
        crossed = (err != 0) & ((goal - pos) * err <= 0)
        self._steer_applied[:] = torch.where(crossed, goal, pos)
        self._steer_vel[:] = torch.where(crossed, torch.zeros_like(pos), self._steer_vel)


def test_servo():
    g = torch.Generator().manual_seed(0)
    n, dt = 64, 0.005
    old, new = OldServo(n, dt), SteerServo(n, dt)
    rate = 0.85 + 0.2 * torch.rand(n, 1, generator=g)
    db = math.radians(0.15) + math.radians(0.2) * torch.rand(n, 1, generator=g)
    delay = torch.randint(6, 9, (n,), generator=g)
    old.steer_rate_scale[:], new.rate_scale[:] = rate, rate
    old.steer_deadband[:], new.deadband[:] = db, db
    old.steer_delay_ticks[:], new.delay_ticks[:] = delay, delay
    goal = torch.zeros(n, 4)
    worst = 0.0
    for k in range(3000):
        if k % 4 == 0 and torch.rand(1, generator=g) < 0.3:       # policy step: sometimes a new goal
            goal = (torch.rand(n, 4, generator=g) * 2 - 1) * math.radians(90) * (torch.rand(n, 1, generator=g) < 0.7)
        old._processed[:, :4] = goal
        old._servo_step()
        out = new.step(goal)
        worst = max(worst, (out - old._steer_applied).abs().max().item())
    check(worst == 0.0, f"SteerServo == 예전 _servo_step (64 env, 3000 스텝, DR 값 섞음): 최대 차이 {worst:g}")

    s = SteerServo(1, dt)                       # 90° 계단: 실측 도달(±1°) 0.87 s.  데드밴드 안에서 멈추므로 ±1° 로 판정
    t, last = None, None
    for k in range(400):
        q = s.step(torch.full((1, 4), math.radians(90)))
        if t is None and abs(q[0, 0].item() - math.radians(90)) < math.radians(1.0):
            t = (k + 1) * dt
        last = q[0, 0].item()
    check(t is not None and abs(t - 0.873) < 0.03, f"90° 계단 ±1° 도달 {t:.3f} s (실측 서보 4개 평균 0.873)")
    check(abs(last - math.radians(90)) <= STEER_DEADBAND_RAD + 1e-6,
          f"최종 오차 {math.degrees(last - math.radians(90)):+.3f}° (데드밴드 ±{math.degrees(STEER_DEADBAND_RAD):.2f}° 안)")


def test_wheel_pi():
    rng = np.random.default_rng(1)
    n = 16
    tpi = WheelPI(n)
    cpi = [[WheelPi() for _ in range(4)] for _ in range(n)]
    diff, total = 0, 0
    maxd = 0
    trips = np.zeros(2, int)
    state_ok = True
    for k in range(3000):
        # 목표: 가끔 바뀌는 계단 + 0 (정지) + 최고속 초과 (포화).
        # 측정: 바퀴별 상태가 50 주기씩 유지 -- 따라감 / 막힘(0) / 반대로 돎 / 느림 / 멈춤 섞임
        if k % 25 == 0:
            ref = rng.choice([0.0, 0.0005, 0.3, 2.0, -2.0, 5.0, 9.0, -9.0], size=(n, 4)) * rng.uniform(0.8, 1.2, (n, 4))
        if k % 50 == 0:
            mode = rng.choice(5, size=(n, 4), p=[0.45, 0.2, 0.1, 0.15, 0.1])
        follow = ref * rng.uniform(0.0, 1.3, (n, 4)) + rng.normal(0, 0.05, (n, 4))
        meas = np.select([mode == 0, mode == 1, mode == 2, mode == 3],
                         [follow, 0.0 * ref, -0.6 * np.abs(ref) * np.sign(ref) - 0.2 * np.sign(ref), 0.3 * ref],
                         np.where(rng.random((n, 4)) < 0.5, 0.0, follow))
        meas = np.round(meas / 0.0375) * 0.0375                     # 엔코더 1틱 격자
        if k % 600 == 599:                                           # 가끔 WPI ON (고장 해제)
            tpi.clear_fault()
            for row in cpi:
                for p in row:
                    p.clear_fault()
        ut = tpi.step(torch.as_tensor(ref, dtype=torch.float32), torch.as_tensor(meas, dtype=torch.float32)).numpy()
        uc = np.array([[cpi[i][j].step(float(np.float32(ref[i, j])), float(np.float32(meas[i, j]))) for j in range(4)]
                       for i in range(n)])
        d = np.abs(ut - uc)
        diff += int((d > 0).sum())
        total += d.size
        maxd = max(maxd, int(d.max()))
        cs = np.array([[cpi[i][j].state.stall for j in range(4)] for i in range(n)], bool)
        ce = np.array([[cpi[i][j].state.enc_dir for j in range(4)] for i in range(n)], bool)
        ct = np.array([[(cpi[i][j].state.enc_trips, cpi[i][j].state.enc_hold, cpi[i][j].state.enc_win) for j in range(4)]
                       for i in range(n)], float)
        tt = torch.stack((tpi.enc_trips, tpi.enc_hold, tpi.enc_win), -1).numpy()
        state_ok &= bool((cs == tpi.stall.numpy()).all() and (ce == tpi.enc_dir.numpy()).all() and (ct == tt).all())
        trips += [int(cs.sum()), int(ce.sum())]
    check(maxd <= 1 and diff / total < 1e-3,
          f"WheelPI(torch) == wheel_pi.c: {total} 샘플 중 다름 {diff}, 최대 차이 {maxd} duty")
    check(state_ok and trips.min() > 1000,
          f"보호 상태(스톨·방향 재시도·잠금·창) torch == C, 매 주기 (스톨 {trips[0]}, 잠금 {trips[1]} 바퀴·주기)")
    st = np.array([[cpi[i][j].state.i for j in range(4)] for i in range(n)])
    check(np.abs(tpi.i.numpy() - st).max() < 1e-2, f"적분 상태 차이 {np.abs(tpi.i.numpy() - st).max():.2e} duty")


def test_encoder():
    th = torch.tensor([0.0, 0.0])
    t0 = encoder_ticks(th)
    t1 = encoder_ticks(th + torch.tensor([2 * math.pi / 8384 * 1.5, 2.0 * 0.02]))
    w = encoder_speed(t1, t0, 0.02)
    check(abs(w[0].item() - 0.0375) < 1e-4, f"1.5틱 이동 → 1틱 = {w[0].item():.4f} rad/s")
    check(abs(w[1].item() - 2.0) < 0.04, f"2 rad/s × 20 ms → {w[1].item():.4f} rad/s (양자화 ±0.0375)")


def test_student_sensors():
    """sensors_torch (Isaac student) == pi/common/sensor_fusion (Pi), and the filter tracks a real tilt."""
    rng = np.random.default_rng(3)
    dt = 0.02
    # 몸체가 roll 로 천천히 흔들리고 (진폭 15°, 0.3 Hz), 진동·가속(±1 m/s²) 과 자이로 바이어스가 섞인 IMU
    T = 1500
    t = np.arange(T) * dt
    roll = np.radians(15) * np.sin(2 * np.pi * 0.3 * t)
    droll = np.radians(15) * 2 * np.pi * 0.3 * np.cos(2 * np.pi * 0.3 * t)
    g_true = np.stack([np.zeros(T), -np.sin(roll), -np.cos(roll)], 1)      # projected gravity, roll only
    gyro = np.stack([droll, np.zeros(T), np.zeros(T)], 1) + 0.01 + rng.normal(0, 0.01, (T, 3))
    lin = rng.normal(0, 0.4, (T, 3)) + np.where((t % 3) < 0.3, 1.0, 0.0)[:, None] * np.array([1.0, 0, 0])
    acc = -9.81 * g_true + lin
    tf = GravityFilter(1, dt)
    nf = sf.GravityFilter(dt)
    gt, gn = [], []
    for k in range(T):
        gt.append(tf.step(torch.as_tensor(gyro[k:k + 1], dtype=torch.float32),
                          torch.as_tensor(acc[k:k + 1], dtype=torch.float32))[0].numpy())
        gn.append(nf.step(gyro[k], acc[k]))
    gt, gn = np.array(gt), np.array(gn)
    check(np.abs(gt - gn).max() < 1e-5, f"중력 필터 torch == numpy(Pi): 최대 차이 {np.abs(gt - gn).max():.1e}")
    ang = lambda a, b: np.degrees(np.arccos(np.clip((a * b).sum(1), -1, 1)))
    e_f = ang(gt[100:], g_true[100:])
    e_a = ang(-acc[100:] / np.linalg.norm(acc[100:], axis=1, keepdims=True), g_true[100:])
    check(e_f.mean() < 1.5 and e_f.mean() < 0.5 * e_a.mean(),
          f"중력 오차 평균 {e_f.mean():.2f}° / 최대 {e_f.max():.2f}° (가속도계만 쓰면 {e_a.mean():.2f}° / {e_a.max():.2f}°)")

    enc = EncoderSpeed(2, dt)
    pos = torch.zeros(2, 4)
    enc.step(pos)
    w = torch.tensor([[2.0] * 4, [-7.9] * 4])
    outs = []
    for k in range(200):
        pos = torch.remainder(pos + w * dt + math.pi, 2 * math.pi) - math.pi     # 물리 엔진처럼 각도가 감김
        outs.append(enc.step(pos))
    o = torch.stack(outs)
    check(((o - w).abs() <= 0.0376).all().item(), f"엔코더 속도: 각도 감김에도 오차 ≤ 1틱 (최대 {(o - w).abs().max():.4f})")
    enc.reset(torch.tensor([1]))
    check(enc.step(pos + 0.1)[1].abs().max().item() == 0.0, "리셋한 env 의 첫 샘플 = 0")
    q = servo_angle(torch.tensor([math.radians(10.04)]))
    check(abs(math.degrees(q.item()) - 10.020) < 1e-3, f"서보 격자 10.04° → {math.degrees(q.item()):.3f}° (0.088° 격자)")


def _wheel_run(duty_fn, t_end, w0=0.0, scale=1.0, load=0.0):
    """One lifted wheel at 5 ms like the Isaac "pi" drive: motor line + friction, armature only.
    duty_fn(k, w_enc) is called every 20 ms (4 steps) with the encoder speed of the last 20 ms."""
    from ksrc_rover.utils.contract import DRIVE_ARMATURE
    dt, J = 0.005, DRIVE_ARMATURE
    w = torch.tensor([w0]); th = torch.zeros(1); duty = torch.zeros(1)
    t_prev = encoder_ticks(th); ws = []
    for k in range(int(round(t_end / dt))):
        if k % 4 == 0:
            t_now = encoder_ticks(th)
            duty = duty_fn(k // 4, encoder_speed(t_now, t_prev, 0.02) if k else torch.zeros(1))
            t_prev = t_now
        tau = drive_torque(duty, w) - load
        tau = tau + wheel_friction(tau, w, J, dt, scale)
        w = w + tau / J * dt
        th = th + w * dt
        ws.append(float(w))
    return np.array(ws)


def test_drive_pi():
    """Isaac "pi" drive pieces against the bench (MOTOR_MODELING_PLAN §8, wheel_pi_design.py)."""
    c = lambda d: (lambda k, w: torch.tensor([float(d)]))
    w40, w60 = _wheel_run(c(40), 0.5), _wheel_run(c(60), 0.5)
    check(np.abs(w40).max() == 0.0 and w60[-1] > 0.05,
          f"정지마찰: 개루프 40 duty 안 돎 ({w40[-1]:.3f}), 60 duty 돎 ({w60[-1]:.3f} rad/s)")
    w400 = _wheel_run(c(400), 1.0)
    check(abs(w400[-1] - 400 * 0.0097) < 0.25, f"개루프 400 duty 정상속도 {w400[-1]:.2f} (벤치 기울기 3.88)")
    stop = _wheel_run(c(0), 0.3, w0=1.0)
    check(stop[-1] == 0.0, f"duty 0 → 멈춤 ({stop[-1]:.3f})")
    for w_ref, t90_max in ((2.0, 0.08), (0.3, 0.08)):
        pi = WheelPI(1)
        w = _wheel_run(lambda k, we: pi.step(torch.full((1, 4), w_ref), we.expand(1, 4))[:, 0], 0.6)
        t = np.arange(len(w)) * 0.005
        t90 = t[np.argmax(w >= 0.9 * w_ref)] if (w >= 0.9 * w_ref).any() else 9.9
        ss = w[t > 0.4].mean()
        check(t90 <= t90_max and abs(ss - w_ref) < 0.05 * w_ref + 0.02,
              f"PI 0→{w_ref} rad/s: 90% {t90 * 1000:.0f} ms (설계 시뮬 2→37, 0.3→28 ms), 정상 {ss:.3f}")
        check(not pi.stall.any() and not pi.enc_dir.any() and not (pi.enc_trips > 0).any(), f"PI 0→{w_ref}: 보호 안 걸림")
    # 막힌 바퀴 (돌에 걸림): 마찰 ×20 (정지마찰 940 duty) 이라 앞뒤 어느 쪽으로도 안 돎 -> 보정 상한 + 스톨로 duty 가 FF 로 내려옴
    pi = WheelPI(1)
    duties = []

    def blocked(k, we):
        u = pi.step(torch.full((1, 4), 2.0), we.expand(1, 4))[:, 0]
        duties.append(float(u))
        return u
    _wheel_run(blocked, 1.5, scale=20.0)
    check(max(duties) <= 211 + 40 + 250 and duties[-1] == 211 and bool(pi.stall[0, 0]),
          f"막힌 바퀴: duty 최대 {max(duties):.0f} (≤ FF+킥+250), 1.5 s 뒤 {duties[-1]:.0f} (FF 211, 스톨)")


def test_drive_gate():
    """DriveGate(torch) == firmware/common/drive_align_gate.c, main.c 처럼 update_goal -> allow 매 주기."""
    import ctypes
    lib = ctypes.CDLL(str(pathlib.Path(__file__).resolve().parents[1] / "libksrc_common.so"))

    class G(ctypes.Structure):
        _fields_ = [("trigger_rad", ctypes.c_float), ("release_rad", ctypes.c_float),
                    ("prev_goal", ctypes.c_float * 4), ("has_prev", ctypes.c_int), ("pending", ctypes.c_int),
                    ("driving", ctypes.c_int), ("max_hold_ticks", ctypes.c_uint), ("hold_ticks", ctypes.c_uint),
                    ("timed_out", ctypes.c_int)]
    F4 = ctypes.c_float * 4
    lib.drive_gate_init.argtypes = [ctypes.POINTER(G), ctypes.c_float, ctypes.c_float, ctypes.c_uint]
    n = 32
    tg = DriveGate(n)
    gs = [G() for _ in range(n)]
    for g in gs:
        lib.drive_gate_init(ctypes.byref(g), ctypes.c_float(float(tg.trigger)), ctypes.c_float(float(tg.release)), 75)
    rng = np.random.default_rng(3)
    goal = np.zeros((n, 4), np.float32); act = np.zeros((n, 4), np.float32)
    same, held, total = True, 0, 0
    for k in range(4000):
        if k % 40 == 0:     # 새 목표: 작은 변화 / 큰 반전 / 40도씩 계단
            goal = goal + rng.choice([0.1, 1.2, -1.2, 0.7], size=(n, 1)).astype(np.float32) * rng.uniform(0.5, 1, (n, 4)).astype(np.float32)
            goal = np.clip(goal, -1.57, 1.57).astype(np.float32)
        spd = np.where(rng.random((n, 1)) < 0.9, 0.2, 0.0) * np.ones((n, 4))
        stuck = rng.random((n, 4)) < 0.02             # 가끔 서보가 안 따라감 (타임아웃 경로)
        act = np.where(stuck, act, act + np.clip(goal - act, -0.08, 0.08)).astype(np.float32)
        at = tg.step(torch.as_tensor(goal), torch.as_tensor(spd, dtype=torch.float32), torch.as_tensor(act)).numpy()
        ac = np.array([lib.drive_gate_update_goal(ctypes.byref(gs[i]), F4(*goal[i]), F4(*spd[i].astype(np.float32))) * 0
                       + lib.drive_gate_allow(ctypes.byref(gs[i]), F4(*goal[i]), F4(*act[i]), 1) for i in range(n)], bool)
        same &= bool((at == ac).all())
        held += int((~ac).sum()); total += n
    check(same and 0.02 < held / total < 0.6, f"DriveGate(torch) == drive_align_gate.c: {total} 주기 일치 (보류 {held / total:.1%})")
    tg.reset(torch.tensor([0]))
    allow = tg.step(torch.zeros(n, 4), torch.full((n, 4), 0.2), torch.full((n, 4), 0.5))
    check(not bool(allow[0]), "리셋한 env 의 첫 구동은 정렬될 때까지 보류 (전원 투입 직후처럼)")


if __name__ == "__main__":
    test_servo()
    test_wheel_pi()
    test_encoder()
    test_student_sensors()
    test_drive_pi()
    test_drive_gate()
    print("ALL PASS" if not FAIL else f"FAIL {len(FAIL)}")
    sys.exit(1 if FAIL else 0)
