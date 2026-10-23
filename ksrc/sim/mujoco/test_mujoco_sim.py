#!/usr/bin/env python3
"""MuJoCo 시뮬 통합 테스트. 헤드리스로 돈다 (뷰어/디스플레이 불필요,
mj_step 은 OpenGL 을 안 건드린다). 실행: python3 test_mujoco_sim.py"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "pi"))
from common.nucleo_link import encode_velocity_cmd

from mujoco_sim import MujocoRover, WHEEL_RADIUS_M

g_fail = False


def expect(label, cond, detail=""):
    global g_fail
    if not cond:
        print(f"FAIL {label} {detail}")
        g_fail = True
    else:
        print(f"ok   {label} {detail}")


rover = MujocoRover()
rover.settle()
x0, y0, z0 = rover.base_pos
expect("settled_on_ground", 0.10 < z0 < 0.20, f"(z0={z0:.4f})")

t_ms = 1000
frame = encode_velocity_cmd(0.5, 0.0, 0.0)
rover.feed_serial_bytes(frame, now_ms=t_ms)

n_steps = int(0.5 / rover.model.opt.timestep)  # 0.5s of forward driving
resend_every_ms = 20  # mimic a real client's continuous send, keeps the watchdog happy
next_resend = t_ms + resend_every_ms
for _ in range(n_steps):
    t_ms += int(rover.model.opt.timestep * 1000)
    if t_ms >= next_resend:
        rover.feed_serial_bytes(frame, now_ms=t_ms)
        next_resend = t_ms + resend_every_ms
    vx, vy, omega = rover.step(t_ms)

x1, y1, z1 = rover.base_pos
expect("forward.vx_applied", abs(vx - 0.5) < 1e-6, f"(vx={vx})")
expect("forward.moved_forward", x1 - x0 > 0.5 * 0.5 * 0.5, f"(dx={x1 - x0:.4f}, expect > half of ideal 0.25m)")
expect("forward.roughly_straight", abs(y1 - y0) < 0.03, f"(dy={y1 - y0:.4f})")
expect("still_upright", z1 > 0.05, f"(z1={z1:.4f})")
print(f"  -> dx over 0.5s @ vx=0.5 m/s: {x1 - x0:.4f} m (ideal, zero slip: 0.25 m)")

# 워치독: 송신을 멈추면 결국 0 이 아닌 vx 인가가 중단되는지 확인
for _ in range(300):  # 300 * timestep, plenty past the 300ms watchdog
    t_ms += int(rover.model.opt.timestep * 1000)
    vx, vy, omega = rover.step(t_ms)
expect("watchdog.stopped", vx == 0.0 and vy == 0.0 and omega == 0.0, f"(vx={vx},vy={vy},omega={omega})")

# 제자리 회전
rover2 = MujocoRover()
rover2.settle()
t_ms = 5000
rot_frame = encode_velocity_cmd(0.0, 0.0, 1.0)
rover2.feed_serial_bytes(rot_frame, now_ms=t_ms)
n_steps = int(0.3 / rover2.model.opt.timestep)
next_resend = t_ms + 20
for _ in range(n_steps):
    t_ms += int(rover2.model.opt.timestep * 1000)
    if t_ms >= next_resend:
        rover2.feed_serial_bytes(rot_frame, now_ms=t_ms)
        next_resend = t_ms + 20
    rover2.step(t_ms)
qw = rover2.data.qpos[3]  # free joint quaternion (w,x,y,z), w<1 means it rotated
expect("rotate.body_yawed", qw < 0.999, f"(qw={qw:.5f})")

if g_fail:
    print("\nSOME TESTS FAILED")
    raise SystemExit(1)
print("\nALL TESTS PASSED")
