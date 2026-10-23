#!/usr/bin/env python3
"""가상 Nucleo 순수 로직 테스트. pygame/pty 불필요. 실행: python3 test_virtual_nucleo.py"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pi"))  # common.nucleo_link 를 import 하기 위해
from common.nucleo_link import encode_velocity_cmd
from virtual_nucleo import VirtualNucleo

g_fail = False


def expect_near(label, got, want, tol=1e-3):
    global g_fail
    if abs(got - want) > tol:
        print(f"FAIL {label}: got {got:.5f} want {want:.5f}")
        g_fail = True
    else:
        print(f"ok   {label}: {got:.5f}")


# 1) 직진 주행. 자세 적분이 맞는지, tick() 이 보고하는 인가값과 일치하는지 확인
sim = VirtualNucleo()
t_ms = 1000
sim.feed_serial_bytes(encode_velocity_cmd(1.0, 0.0, 0.0), now_ms=t_ms)

dt = 0.02
n_ticks = 10  # 200ms 전진
for i in range(n_ticks):
    t_ms += int(dt * 1000)
    vx, vy, omega = sim.tick(t_ms, dt)
    expect_near(f"forward.tick{i}.vx_applied", vx, 1.0, 1e-6)

expect_near("forward.x_after_200ms", sim.x, 1.0 * n_ticks * dt, 1e-3)
expect_near("forward.y_after_200ms", sim.y, 0.0, 1e-3)

# 2) 프레임 공급 중단 -> 워치독은 **마지막으로 실제 수신한** 프레임(t_ms=1000)
#    부터 센다. 따라서 그 프레임 기준 300ms 이내인 동안은 vx=1.0 을 계속 인가하다가
#    그 뒤에 stale 이 된다. 충분히 지난 뒤 자세가 멈췄는지 확인
#    (tick() 이 0,0,0 을 보고하고 x 가 그대로인지).
for _ in range(30):  # t_ms 가 약 1200+600=1800 까지 간다. 마지막 프레임(t=1000)
    t_ms += 20        # 으로부터 800ms 경과 -> 300ms 상한을 한참 넘김
    vx, vy, omega = sim.tick(t_ms, dt)
expect_near("watchdog.vx_forced_zero", vx, 0.0, 1e-6)
x_after_stale = sim.x
for _ in range(5):
    t_ms += 20
    sim.tick(t_ms, dt)
expect_near("watchdog.x_stopped_advancing", sim.x, x_after_stale, 1e-6)

# 3) 제자리 회전: theta 가 omega rad/s 로 증가
sim2 = VirtualNucleo()
t_ms = 2000
sim2.feed_serial_bytes(encode_velocity_cmd(0.0, 0.0, 1.0), now_ms=t_ms)
for i in range(10):
    t_ms += 20
    sim2.tick(t_ms, 0.02)
expect_near("rotate.theta_after_200ms", sim2.theta, 1.0 * 10 * 0.02, 1e-3)
expect_near("rotate.x_stays_at_origin", sim2.x, 0.0, 1e-3)

# 4) 정상성 확인: 주행 중 모듈 명령이 실제로 생성됐는지(None 이 아닌지)
assert sim.module_cmds is not None
assert len(list(sim.module_cmds)) == 4
print("ok   module_cmds populated (4 modules)")

if g_fail:
    print("\nSOME TESTS FAILED")
    raise SystemExit(1)
print("\nALL TESTS PASSED")
