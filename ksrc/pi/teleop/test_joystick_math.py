#!/usr/bin/env python3
"""조이스틱 수학 유닛테스트. 실행: python3 test_joystick_math.py"""
import math

from joystick_math import (apply_deadzone, clamp_to_unit_circle, left_pad_to_linear,
                           lock_low_speed_direction, right_pad_to_angular, snap_to_axes)

g_fail = False


def expect_near(label, got, want, tol=1e-6):
    global g_fail
    if abs(got - want) > tol:
        print(f"FAIL {label}: got {got:.4f} want {want:.4f}")
        g_fail = True
    else:
        print(f"ok   {label}: {got:.4f}")


# 1) 원 안쪽: 클램프 없음
nx, ny = clamp_to_unit_circle(50, 0, 100)
expect_near("inside.nx", nx, 0.5)
expect_near("inside.ny", ny, 0.0)

# 2) 축 방향으로 가장자리 밖까지 끌기: 정확히 1.0 으로 클램프
nx, ny = clamp_to_unit_circle(150, 0, 100)
expect_near("clamp_straight.nx", nx, 1.0)
expect_near("clamp_straight.ny", ny, 0.0)

# 3) 패드 사각형 모서리까지 대각선으로 끌기: 방향은 유지, 크기만 1 로 클램프
nx, ny = clamp_to_unit_circle(100, 100, 100)
expect_near("clamp_diag.nx", nx, math.sqrt(2) / 2)
expect_near("clamp_diag.ny", ny, math.sqrt(2) / 2)
expect_near("clamp_diag.mag", math.hypot(nx, ny), 1.0)

# 4) 반경이 0 인 퇴화 케이스
nx, ny = clamp_to_unit_circle(10, 10, 0)
expect_near("zero_radius.nx", nx, 0.0)
expect_near("zero_radius.ny", ny, 0.0)

# 5) 왼쪽 패드 -> 병진 속도, 상하좌우 네 방향
vx, vy = left_pad_to_linear(0.0, -1.0, 2.0)  # 끝까지 위
expect_near("left.up.vx", vx, 2.0)
expect_near("left.up.vy", vy, 0.0)

vx, vy = left_pad_to_linear(0.0, 1.0, 2.0)  # 끝까지 아래
expect_near("left.down.vx", vx, -2.0)

vx, vy = left_pad_to_linear(-1.0, 0.0, 2.0)  # 끝까지 왼쪽
expect_near("left.left.vy", vy, 2.0)
expect_near("left.left.vx", vx, 0.0)

vx, vy = left_pad_to_linear(1.0, 0.0, 2.0)  # 끝까지 오른쪽
expect_near("left.right.vy", vy, -2.0)

# 6) 오른쪽 패드 -> 각속도, 가로축만
expect_near("right.center", right_pad_to_angular(0.0, 3.0), 0.0)
expect_near("right.full_right_is_cw", right_pad_to_angular(1.0, 3.0), -3.0)
expect_near("right.full_left_is_ccw", right_pad_to_angular(-1.0, 3.0), 3.0)

# 7) 데드존: 안쪽은 0, 경계는 0 에서 연속, 끝은 1, 방향 유지
nx, ny = apply_deadzone(0.05, 0.05, 0.1)
expect_near("dz.inside", math.hypot(nx, ny), 0.0)
nx, ny = apply_deadzone(0.1001, 0.0, 0.1)
expect_near("dz.edge_continuous", nx, 0.0, tol=1e-3)
nx, ny = apply_deadzone(0.0, -1.0, 0.1)
expect_near("dz.full", ny, -1.0)
nx, ny = apply_deadzone(0.3, 0.4, 0.1)            # 크기 0.5 -> (0.5-0.1)/0.9
expect_near("dz.rescaled_mag", math.hypot(nx, ny), 0.4 / 0.9)
expect_near("dz.direction_kept", math.atan2(ny, nx), math.atan2(0.4, 0.3))
nx, ny = apply_deadzone(0.3, 0.4, 0.0)
expect_near("dz.off", nx, 0.3)

# 8) 축 스냅: 10도 이내면 축으로, 크기 유지. 밖이면 그대로
nx, ny = snap_to_axes(1.0, math.tan(math.radians(8)), 10)
expect_near("snap.to_axis_y0", ny, 0.0)
expect_near("snap.mag_kept", nx, math.hypot(1.0, math.tan(math.radians(8))))
nx, ny = snap_to_axes(0.1, -1.0, 10)                 # 거의 위쪽
expect_near("snap.to_up_x0", nx, 0.0)
nx, ny = snap_to_axes(1.0, 1.0, 10)                  # 45도는 그대로
expect_near("snap.diag_kept", ny, 1.0)

# 9) 저속 방향 고정
R = (1.0, 0.0)                                       # 직전: 패드 오른쪽 (우 횡걸음)
nx, ny, d = lock_low_speed_direction(0.25, -0.15, R, 0.35, 20)   # 중앙 위를 스침 (31도)
expect_near("lock.blocks_offaxis_lowspeed", math.hypot(nx, ny), 0.0)
expect_near("lock.keeps_last_dir", d[0], 1.0)
nx, ny, d = lock_low_speed_direction(-0.2, 0.0, R, 0.35, 20)     # 정반대(좌)는 저속도 통과
expect_near("lock.opposite_passes", nx, -0.2)
nx, ny, d = lock_low_speed_direction(0.0, -0.5, R, 0.35, 20)     # 크게 밀면 새 방향 허용
expect_near("lock.high_mag_passes", ny, -0.5)
expect_near("lock.updates_last_dir", d[1], -0.5)
nx, ny, d = lock_low_speed_direction(0.2, -0.05, None, 0.35, 20) # 처음은 무조건 통과
expect_near("lock.first_passes", nx, 0.2)
nx, ny, d = lock_low_speed_direction(0.0, 0.0, R, 0.35, 20)      # 0 은 0, 방향 기억 유지
expect_near("lock.zero_keeps_dir", d[0], 1.0)

if g_fail:
    print("\nSOME TESTS FAILED")
    raise SystemExit(1)
print("\nALL TESTS PASSED")
