"""공통 구동 게이트가 조향 전 주행을 막고 정렬 후 풀리는지 검사."""
import numpy as np

from env import RoverEnv


def test_drive_waits_for_steering():
    env = RoverEnv(arena_eval=True, eval_kind="sand", randomize=False,
                   seed=0, spawn_max_slope_deg=3.0)
    try:
        env.reset(seed=0)
        assert np.allclose(env.wheel_ref_applied, 0)
        env.cmd_nom = np.array([0.0, 0.25, 0.0])
        env.step(np.zeros(3))
        assert np.allclose(env.wheel_ref_applied, 0)
        assert np.allclose(env.d.ctrl[env.a_wh], 0)
        for _ in range(100):
            env.cmd_nom = np.array([0.0, 0.25, 0.0])
            env.step(np.zeros(3))
            if np.any(np.abs(env.wheel_ref_applied) > 0):
                break
        else:
            raise AssertionError("조향 정렬 후에도 구동이 재개되지 않음")
    finally:
        env.close()


def test_gate_can_be_disabled_for_comparison():
    env = RoverEnv(arena_eval=True, eval_kind="sand", randomize=False,
                   seed=0, spawn_max_slope_deg=3.0,
                   drive_align_gate_deg=0.0)
    try:
        env.reset(seed=0)
        env.cmd_nom = np.array([0.0, 0.25, 0.0])
        env.step(np.zeros(3))
        assert np.any(np.abs(env.wheel_ref_applied) > 0)
    finally:
        env.close()


def test_small_steering_change_does_not_stop_path_following():
    env = RoverEnv(arena_eval=True, eval_kind="sand", randomize=False,
                   seed=0, spawn_max_slope_deg=3.0)
    try:
        env.reset(seed=0)
        env.cmd_nom = np.array([0.22, 0.10, 0.0])
        env.step(np.zeros(3))
        assert np.any(np.abs(env.wheel_ref_applied) > 0)
        assert not env.drive_alignment_pending
    finally:
        env.close()


if __name__ == "__main__":
    test_drive_waits_for_steering()
    test_gate_can_be_disabled_for_comparison()
    test_small_steering_change_does_not_stop_path_following()
    print("drive alignment gate: OK")
