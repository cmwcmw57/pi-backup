#!/usr/bin/env python3
"""
KSRC 저수준 텔레옵 파이프라인의 진짜 end-to-end 검증.

**실제 클라이언트 스크립트**(pi/teleop/teleop_terminal.py,
pi/teleop/teleop_joystick.py)를 진짜 서브프로세스로 띄우고, 가상 Nucleo
(ctypes 로 부른 실제 firmware/common C 코드. ksrc_common.py + virtual_nucleo.py
참고)를 가리키게 한 뒤, 시뮬 안의 차체 자세를 확인해서 반대편으로 나온 명령이
맞았는지 검증한다. "이렇게 돼야 한다"를 다시 유도해서 비교하는 게 아니라,
실제로 출하될 코드를 처음부터 끝까지 그대로 돌려서 확인한다.

타이밍 주의: 서브프로세스/pygame 시작 지연은 머신 부하에 따라 달라 예측할 수
없다. 그래서 이 파일은 "명령이 반영됐어야 할 고정 시간"을 절대 가정하지 않고,
관측 가능한 조건이 만족될 때까지 폴링한다(wait_until).
"""
import os
import subprocess
import sys
import time
from pathlib import Path

from virtual_nucleo import start_headless

REPO_ROOT = Path(__file__).resolve().parent.parent
PI_TELEOP = REPO_ROOT / "pi" / "teleop"

g_fail = False


def expect_near(label, got, want, tol):
    global g_fail
    if abs(got - want) > tol:
        print(f"FAIL {label}: got {got:.4f} want {want:.4f} (tol {tol})")
        g_fail = True
    else:
        print(f"ok   {label}: {got:.4f}")


def wait_until(handle, predicate, timeout_s=5.0, step_s=0.05):
    """Polls handle.snapshot() (ticking the sim forward step_s at a time)
    until predicate(snapshot) is true. Raises TimeoutError otherwise.
    Used instead of a fixed sleep so this test doesn't depend on guessing
    subprocess/pygame startup latency."""
    start = time.monotonic()
    snap = handle.snapshot()
    while time.monotonic() - start < timeout_s:
        if predicate(snap):
            return snap
        snap = handle.run_realtime(duration_s=step_s)
    raise TimeoutError(f"condition not met within {timeout_s}s, last snapshot={snap}")


def test_terminal_client():
    print("\n=== subgoal 2: terminal numeric client -> virtual Nucleo ===")
    handle = start_headless()
    try:
        proc = subprocess.Popen(
            [sys.executable, str(PI_TELEOP / "teleop_terminal.py"), "--port", handle.slave_name, "--rate", "50"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        assert proc.stdin is not None
        proc.stdin.write("0.4 0 0\n")
        proc.stdin.flush()

        x0, _, _, vx, _, _, stale = wait_until(handle, lambda s: abs(s[3] - 0.4) < 1e-3)
        expect_near("terminal.vx_seen", vx, 0.4, 1e-6)
        assert not stale, "link went stale while terminal client was actively sending"
        print("ok   terminal.link_not_stale")

        t0 = time.monotonic()
        x1, y1, theta1, _, _, _, _ = handle.run_realtime(duration_s=0.3)  # clean window, command already stable
        elapsed = time.monotonic() - t0
        expect_near("terminal.x_displacement", x1 - x0, 0.4 * elapsed, 0.015)

        proc.stdin.write("0 0 -1.0\n")
        proc.stdin.flush()
        _, _, theta_a, _, _, omega2, _ = wait_until(handle, lambda s: abs(s[5] - (-1.0)) < 1e-3)
        expect_near("terminal.omega_seen", omega2, -1.0, 1e-6)

        t0 = time.monotonic()
        _, _, theta_b, _, _, _, _ = handle.run_realtime(duration_s=0.3)
        elapsed = time.monotonic() - t0
        expect_near("terminal.theta_delta", theta_b - theta_a, -1.0 * elapsed, 0.02)

        proc.stdin.write("q\n")
        proc.stdin.flush()
        proc.wait(timeout=3)
        snap = wait_until(handle, lambda s: s[3] == 0.0 and s[5] == 0.0, timeout_s=2.0)
        expect_near("terminal.final_vx_zero", snap[3], 0.0, 1e-6)
    finally:
        handle.close()


def test_pad_client():
    print("\n=== subgoal 1+3: pad (mouse joystick) client -> virtual Nucleo ===")
    handle = start_headless()
    env = dict(os.environ)
    env["SDL_VIDEODRIVER"] = "dummy"
    env["SDL_AUDIODRIVER"] = "dummy"
    try:
        driver_script = f"""
import sys, time, threading
sys.path.insert(0, {str(PI_TELEOP)!r})
import pygame
import teleop_joystick as tj

class Args:
    port = {handle.slave_name!r}
    baud = 115200
    rate = 50.0
    max_lin = 0.5
    max_ang = 2.0

def injector():
    time.sleep(0.2)
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {{"pos": tj.LEFT_CENTER, "button": 1}}))
    time.sleep(0.05)
    up_pos = (tj.LEFT_CENTER[0], tj.LEFT_CENTER[1] - 100)  # full deflection -> +max_lin
    pygame.event.post(pygame.event.Event(pygame.MOUSEMOTION, {{"pos": up_pos}}))
    time.sleep(3.0)  # hold generously long -- the test polls for when this
                      # 실제로 반영될 때까지 폴링한다 (타이밍을 가정하지 않음)
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, {{"pos": up_pos, "button": 1}}))
    time.sleep(0.1)
    pygame.event.post(pygame.event.Event(pygame.QUIT))

threading.Thread(target=injector, daemon=True).start()
tj.run(Args())
"""
        proc = subprocess.Popen([sys.executable, "-c", driver_script], env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        x0, _, _, vx, _, _, stale = wait_until(handle, lambda s: s[3] > 0.45, timeout_s=6.0)
        expect_near("pad.vx_near_max_lin_while_held", vx, 0.5, 0.05)
        assert not stale, "link went stale while pad was actively held"

        t0 = time.monotonic()
        x1, _, _, vx_mid, _, _, _ = handle.run_realtime(duration_s=0.5)  # still well within the 3.0s hold
        elapsed = time.monotonic() - t0
        expect_near("pad.vx_still_max_lin_mid_hold", vx_mid, 0.5, 0.05)
        expect_near("pad.x_displacement_while_held", x1 - x0, 0.5 * elapsed, 0.03)
        print(f"ok   pad.x_progressed: {x1:.4f}")

        proc.wait(timeout=8)
        out = proc.stdout.read()
        if proc.returncode != 0:
            print(out)
        assert proc.returncode == 0, f"pad client subprocess failed, exit {proc.returncode}"

        snap_after = wait_until(handle, lambda s: s[3] == 0.0, timeout_s=2.0)
        expect_near("pad.stopped_after_release", snap_after[3], 0.0, 1e-6)
    finally:
        handle.close()


if __name__ == "__main__":
    test_terminal_client()
    test_pad_client()
    if g_fail:
        print("\nSOME E2E CHECKS FAILED")
        raise SystemExit(1)
    print("\nALL E2E CHECKS PASSED")
