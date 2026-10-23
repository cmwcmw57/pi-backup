"""Speed-tracking task contracts: action size, sensor input, and terrain scale."""
import numpy as np
import pytest

import terrain
from env import HISTORY, OBS_PER_FRAME, RoverEnv
from reward import VelRewardCfg, velocity_terms


def test_tail_cost_preserves_default_and_penalizes_large_errors():
    a = np.zeros(3)
    ordinary = velocity_terms(0.30, 0.20, a, a, a, VelRewardCfg())
    assert ordinary["tail"] == 0.0
    cfg = VelRewardCfg(w_tail=0.1)
    small = velocity_terms(0.10, 0.05, a, a, a, cfg)
    large = velocity_terms(0.30, 0.20, a, a, a, cfg)
    assert large["track_v"] == ordinary["track_v"]
    assert large["tail"] < small["tail"] < 0.0


def test_cmdvel_action_modes_match_observation_space():
    for mode, n_action in (("body", 3), ("wheel", 4), ("body+wheel", 7)):
        env = RoverEnv(task="cmdvel", action_mode=mode, randomize=False,
                       episode_s=0.1, seed=4)
        try:
            obs, _ = env.reset(seed=4)
            assert env.action_space.shape == (n_action,)
            assert env.observation_space.shape == ((OBS_PER_FRAME - 3 + n_action) * HISTORY,)
            assert env.observation_space.contains(obs)
            action = np.linspace(-0.2, 0.2, n_action, dtype=np.float32)
            obs, _, _, _, _ = env.step(action)
            assert env.observation_space.contains(obs)
            assert np.allclose(obs[-n_action:], action)
        finally:
            env.close()


def test_velocity_observation_is_explicit_simulation_oracle():
    env = RoverEnv(task="cmdvel", velocity_obs=True, randomize=False,
                   episode_s=0.1, seed=5)
    try:
        obs, _ = env.reset(seed=5)
        assert env.observation_space.shape == ((OBS_PER_FRAME + 3) * HISTORY,)
        scale = np.array([env.v_max, env.v_max, env.om_max])
        assert np.allclose(obs[-3:], env.body_twist() / scale, atol=1e-6)
        obs, _, _, _, _ = env.step(np.zeros(3, dtype=np.float32))
        assert np.allclose(obs[-3:], env.body_twist() / scale, atol=1e-6)
    finally:
        env.close()


def test_short_history_has_matching_space_and_latest_frame():
    env = RoverEnv(task="cmdvel", velocity_obs=True, history_frames=5,
                   randomize=False, episode_s=0.1, seed=7)
    try:
        obs, _ = env.reset(seed=7)
        assert env.observation_space.shape == ((OBS_PER_FRAME + 3) * 5,)
        assert env.observation_space.contains(obs)
        obs, _, _, _, _ = env.step(np.zeros(3, dtype=np.float32))
        assert env.observation_space.contains(obs)
        assert np.allclose(obs[-3:], env.body_twist() /
                           np.array([env.v_max, env.v_max, env.om_max]), atol=1e-6)
    finally:
        env.close()


def test_history_length_does_not_change_noise_or_physics_sequence():
    short = RoverEnv(task="cmdvel", history_frames=5, randomize=True,
                     episode_s=0.1, seed=8)
    long = RoverEnv(task="cmdvel", history_frames=HISTORY, randomize=True,
                    episode_s=0.1, seed=8)
    try:
        a, _ = short.reset(seed=8)
        b, _ = long.reset(seed=8)
        assert np.array_equal(a[-OBS_PER_FRAME:], b[-OBS_PER_FRAME:])
        assert np.array_equal(a.reshape(5, OBS_PER_FRAME),
                              np.repeat(a[-OBS_PER_FRAME:][None], 5, axis=0))
        a, _, _, _, _ = short.step(np.zeros(3, dtype=np.float32))
        b, _, _, _, _ = long.step(np.zeros(3, dtype=np.float32))
        assert np.array_equal(a[-OBS_PER_FRAME:], b[-OBS_PER_FRAME:])
        assert np.array_equal(short.d.qpos, long.d.qpos)
    finally:
        short.close()
        long.close()


def test_external_push_changes_motion_without_changing_command():
    base = RoverEnv(task="cmdvel", randomize=False, episode_s=1.0, seed=9)
    pushed = RoverEnv(task="cmdvel", randomize=False, episode_s=1.0, seed=9,
                      push_force_max=1.0)
    try:
        base.reset(seed=9)
        pushed.reset(seed=9)
        assert np.allclose(base.d.qpos, pushed.d.qpos)
        assert np.allclose(base.cmd_nom, pushed.cmd_nom)
        base.step(np.zeros(3, dtype=np.float32))
        _, _, _, _, info = pushed.step(np.zeros(3, dtype=np.float32))
        assert np.linalg.norm(base.d.qvel[:2] - pushed.d.qvel[:2]) > 1e-5
        assert np.isnan(info["push_v_err_m"])
        for _ in range(16):
            _, _, term, trunc, info = pushed.step(np.zeros(3, dtype=np.float32))
            assert not (term or trunc)
        assert np.isfinite(info["push_v_err_m"])
    finally:
        base.close()
        pushed.close()


def test_cmdvel_arena_evaluation_keeps_original_map_dimensions():
    env = RoverEnv(task="cmdvel", arena_eval=True, eval_kind="sand",
                   randomize=False, episode_s=0.1, seed=6)
    try:
        env.reset(seed=6)
        assert np.allclose((env.hf_ex, env.hf_ey), terrain.eval_extent("sand"))
        assert np.allclose(env.m.hfield_size[0, :2],
                           np.asarray(terrain.eval_extent("sand")) / 2)
    finally:
        env.close()


def test_cmd_filter_can_match_ros_without_changing_prior_training_default():
    old = RoverEnv(task="cmdvel", cmd_filter_tau=0.28)
    ros = RoverEnv(task="cmdvel", cmd_filter_tau=0.10)
    try:
        assert np.isclose(old.cmd_a, 1.0 / (0.3 * 50))
        assert np.isclose(ros.cmd_a, 0.02 / (0.10 + 0.02))
        assert ros.cmd_a > old.cmd_a
    finally:
        old.close()
        ros.close()
    with pytest.raises(ValueError, match="cmd_filter_tau"):
        RoverEnv(task="cmdvel", cmd_filter_tau=-0.1)


def test_difficulty_floor_can_select_only_hardest_terrain():
    default = RoverEnv(task="cmdvel", difficulty=1.0, randomize=False,
                       episode_s=0.1, seed=17)
    hardest = RoverEnv(task="cmdvel", difficulty=1.0,
                       difficulty_min_frac=1.0, randomize=False,
                       episode_s=0.1, seed=17)
    try:
        default.reset(seed=17)
        hardest.reset(seed=17)
        assert 0.6 <= default.d_ep <= 1.0
        assert hardest.d_ep == 1.0
        assert default.d_ep < hardest.d_ep
    finally:
        default.close()
        hardest.close()
    with pytest.raises(ValueError, match="difficulty_min_frac"):
        RoverEnv(task="cmdvel", difficulty_min_frac=1.1)
