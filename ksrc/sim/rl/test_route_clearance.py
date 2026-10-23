"""중간 스폰의 경로 상태와 로버 크기를 반영한 경로 생성 회귀 검사."""
import numpy as np

import path as pth
import terrain as terr
from config import RoverCfg
from env import RoverEnv
from train import stage1_paths


def test_mid_path_projection_starts_at_spawn():
    path = pth.RefPath([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    spawn = path.at(1.5)
    path.seed(1.5, spawn)
    s, e_y, _ = path.project(spawn)
    assert np.isclose(s, 1.5)
    assert np.isclose(e_y, 0.0)
    assert path.i == 1


def test_full_route_distance_cannot_jump_faster_than_position():
    path = pth.RefPath([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    points = [np.array(p) for p in ((0.8, 0.1), (0.9, 0.1),
                                     (1.05, 0.1), (1.1, 0.3), (0.8, 0.3))]
    for a in points:
        for b in points:
            assert abs(path.distance(a) - path.distance(b)) <= np.linalg.norm(a - b) + 1e-12


def test_line_clear_detects_one_cell_barrier():
    mask = np.ones((100, 100), dtype=bool)
    mask[:, 50] = False
    assert not terr.line_clear(mask, 1.0, 1.0, (-0.2, 0.0), (0.2, 0.0))
    route = terr.center_clearance_mask(mask, 1.0, 1.0, 0.03)
    assert not route[50, 48]
    assert route[50, 40]


def test_stage1_reset_and_latest_action_observation():
    env = RoverEnv(paths=stage1_paths("sand", 1234, 10),
                   arena_eval=True, eval_kind="sand", randomize=False)
    try:
        for seed in range(1000, 1020):
            obs, _ = env.reset(seed=seed)
            settle = np.linalg.norm(env.d.qpos[:2] - env.path.at(env.s0))
            assert env.s_path >= env.s0
            assert env.s_path - env.s0 <= settle + 1e-3
            assert abs(env.s_est - env.s_path) < 1e-6
            assert abs(env.e_y) <= settle + 1e-3
            assert np.allclose(obs[-3:], 0.0)
        a = np.array([0.3, -0.4, 0.2])
        obs, *_ = env.step(a)
        assert np.allclose(obs[-3:], a)
        assert np.allclose(env.a_prev, a)
        b = np.array([-0.2, 0.1, -0.3])
        obs, *_ = env.step(b)
        assert np.allclose(obs[-3:], b)
        assert np.allclose(env.a_prev2, a)
    finally:
        env.close()


def test_stage1_segments_keep_wheel_clearance():
    cfg = RoverCfg()
    z = terr.load_arena("sand")
    ex, ey = terr.eval_extent("sand")
    drive, _ = terr.drivable_mask(z, ex, ey, max_slope_deg=cfg.max_slope_deg)
    route = terr.center_clearance_mask(drive, ex, ey, cfg.route_clearance)
    assert route.sum() < drive.sum()
    for path in stage1_paths("sand", 1234, 10):
        for i in range(len(path.L)):
            assert terr.line_clear(route, ex, ey, path.P[i], path.P[i + 1])


def test_randomized_path_errors_use_controller_pose_estimate():
    env = RoverEnv(paths=stage1_paths("sand", 1234, 10),
                   arena_eval=True, eval_kind="sand", randomize=True, seed=2)
    try:
        obs, _ = env.reset(seed=2)
        frame = obs[-env.hist[-1].size:]
        assert np.isclose(frame[-9], np.clip(env.e_y_est / 0.3, -3, 3))
        assert np.isclose(frame[-8], np.cos(env.e_psi_est))
        assert np.isclose(frame[-7], np.sin(env.e_psi_est))
        expected_L = pth.pursuit_distance(
            env.v_cruise, env.cfg.pp_k_v, env.cfg.pp_l_min, env.cfg.pp_l_max)
        assert np.isclose(frame[-4], expected_L)
        assert np.allclose(frame[-6:-4], env.target_b)
        assert abs(env.e_y_est - env.e_y) > 1e-4 or abs(env.e_psi_est - env.e_psi) > 1e-4
    finally:
        env.close()


def test_policy_target_is_controller_pursuit_point():
    env = RoverEnv(paths=stage1_paths("sand", 1234, 10),
                   arena_eval=True, eval_kind="sand", randomize=False, seed=3)
    try:
        obs, _ = env.reset(seed=3)
        xy = env.d.qpos[:2]
        yaw = env._yaw()
        c, s = np.cos(yaw), np.sin(yaw)
        target = env.path_c.at(env.s_est + env.target_L) - xy
        expected = np.array([c * target[0] + s * target[1],
                             -s * target[0] + c * target[1]])
        assert np.allclose(obs[-6:-4], expected)
        assert np.isclose(obs[-4], env.target_L)
    finally:
        env.close()


def test_rock_reset_finds_safe_short_route():
    env = RoverEnv(arena_eval=True, eval_kind="rock", randomize=False, seed=0)
    try:
        env.reset(seed=0)
        assert env.path.total > 0
        for i in range(len(env.path.L)):
            assert terr.line_clear(env.route_drive, env.hf_ex, env.hf_ey,
                                   env.path.P[i], env.path.P[i + 1])
    finally:
        env.close()


if __name__ == "__main__":
    test_mid_path_projection_starts_at_spawn()
    test_full_route_distance_cannot_jump_faster_than_position()
    test_line_clear_detects_one_cell_barrier()
    test_stage1_reset_and_latest_action_observation()
    test_stage1_segments_keep_wheel_clearance()
    test_randomized_path_errors_use_controller_pose_estimate()
    test_rock_reset_finds_safe_short_route()
    print("route clearance: OK")
