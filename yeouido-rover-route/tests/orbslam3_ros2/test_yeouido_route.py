# ================================================================
# test_yeouido_route.py
# 설명: RouteRunner를 이상적 평면 운동학으로 적분해 여의도 경로의 순서·정지·중단 동작을 검증한다.
# 사용법: python3 -m pytest tests/orbslam3_ros2/test_yeouido_route.py
# ================================================================
import math
from pathlib import Path

from experiments.orbslam3_ros2.yeouido_route import RouteRunner

PLAN = Path(__file__).resolve().parents[2] / "docs" / "yeouido_route_plan.json"
DT = 0.05


def drive(runner: RouteRunner, max_s: float = 3000.0, tilt_at: float | None = None):
    """이상적 추종으로 적분하고 (t, x, y, yaw, label) 궤적을 반환한다."""
    x, y, yaw, t, trace = 0.29327678, 0.32020833, 0.0, 0.0, []
    while t < max_s and not runner.complete and not runner.aborted:
        pitch = math.radians(50) if tilt_at is not None and t >= tilt_at else 0.0
        cmd = runner.command(elapsed_s=t, x=x, y=y, yaw_rad=yaw, pitch_rad=pitch)
        x += (math.cos(yaw) * cmd.vx - math.sin(yaw) * cmd.vy) * DT
        y += (math.sin(yaw) * cmd.vx + math.cos(yaw) * cmd.vy) * DT
        yaw += cmd.yaw_rate * DT
        t += DT
        trace.append((t, x, y, yaw, runner.label))
    return trace


def test_full_route_completes_at_lane_c_top():
    trace = drive(RouteRunner.from_json(PLAN))
    t, x, y, yaw, label = trace[-1]
    assert label == "done"
    assert math.hypot(x - 2.2, y - 2.06) < 0.06 and abs(yaw) < math.radians(6)


def test_visits_a_top_with_u_turn_then_b_partial_then_c():
    trace = drive(RouteRunner.from_json(PLAN))
    a_top = [p for p in trace if abs(p[2] - 0.42) < 0.1 and p[1] > 2.15]
    assert a_top and any(abs(abs(p[3]) - math.pi) < math.radians(8) for p in a_top)
    lane_b = [p for p in trace if 1.14 < p[2] < 1.34]
    assert 1.15 < max(p[1] for p in lane_b) < 1.26  # B는 x=1.20까지만
    assert all(p[1] < 2.3 for p in trace)  # 끝 벽(2.42) 앞에서 정지


def test_failure_hold_is_one_and_half_seconds_at_b():
    runner = RouteRunner.from_json(PLAN)
    waits = [s for s in runner.steps if s["kind"] == "wait" and s["s"] != 3.0]
    assert [w["s"] for w in waits] == [1.5]


def test_tilt_limit_aborts_and_stops():
    runner = RouteRunner.from_json(PLAN)
    trace = drive(runner, tilt_at=20.0)
    assert runner.aborted and trace[-1][0] < 21.0
    cmd = runner.command(elapsed_s=999.0, x=1.0, y=1.0, yaw_rad=0.0)
    assert (cmd.vx, cmd.vy, cmd.yaw_rate) == (0.0, 0.0, 0.0)


def test_rl_segment_uses_injected_policy():
    from experiments.orbslam3_ros2.swerve import BodyCommand

    seen = []
    runner = RouteRunner.from_json(PLAN, policy=lambda obs: seen.append(obs) or BodyCommand(0.1, 0.0, 0.0))
    cmd = runner.command(elapsed_s=0.0, x=0.29327678, y=0.32020833, yaw_rad=0.0)
    assert seen and cmd.vx == 0.1 and seen[0]["max_speed"] == 0.15


def test_stalled_b_climb_skips_to_wait_and_retreat():
    """B 구간에서 8 s 동안 진전이 없으면 올라가기를 포기하고 대기 후 후퇴한다."""
    runner = RouteRunner.from_json(PLAN)
    t, x, y, yaw = 0.0, 0.29327678, 0.32020833, 0.0
    stuck = False
    while t < 400.0 and not runner.complete:
        if runner.label.startswith("B") and x > 0.80:
            stuck = True  # 램프 입구에서 물리적으로 막힌 상태를 흉내
        cmd = runner.command(elapsed_s=t, x=x, y=y, yaw_rad=yaw)
        if stuck and runner.label.startswith("B"):
            cmd = type(cmd)(0.0, 0.0, 0.0) if cmd.vx > 0 else cmd  # 전진은 막고 후진은 허용
        x += (math.cos(yaw) * cmd.vx - math.sin(yaw) * cmd.vy) * DT
        y += (math.sin(yaw) * cmd.vx + math.cos(yaw) * cmd.vy) * DT
        yaw += cmd.yaw_rate * DT
        t += DT
    assert runner.complete and math.hypot(x - 2.2, y - 2.06) < 0.06
