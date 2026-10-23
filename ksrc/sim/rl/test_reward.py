"""경로추종 보상이 스워브의 게걸음·후진과 횡오차를 올바르게 평가하는지 검사."""
import math

import numpy as np

from reward import RewardCfg, tracking_terms, recovery_potential


ZERO = np.zeros(3)


def score(cfg, *, progress=1.0, cross_track=0.0, heading=0.0,
          course_error=0.0, movement=0.0, lateral_recovery=0.0,
          action=ZERO, previous=ZERO, previous2=ZERO):
    gate, terms = tracking_terms(progress, cross_track, heading, action,
                                 previous, previous2, cfg,
                                 course_error=course_error, movement=movement,
                                 lateral_recovery=lateral_recovery)
    return gate, sum(terms.values()), terms


def test_paper_heading_gate_suppresses_sideways_and_reverse_alignment():
    cfg = RewardCfg()
    center = [score(cfg, heading=yaw, movement=1.0)[1] for yaw in
              (0.0, math.pi / 2, math.pi)]
    # 기대값은 가중치에서 유도한다.  예전엔 [1.0, 0.16, 0.16] 으로 박아뒀는데
    # w_head 가 보상에 들어오면서 옆·뒤 방향이 음수가 됐고, 숫자만 틀렸다.
    forward = cfg.w_prog + cfg.w_align          # 게이트 1.0
    # 게이트 0 + 헤딩 이차벌점 상한(min(..., 4)) 에 걸린 값
    sideways = cfg.w_prog - cfg.w_head * 4.0
    assert np.allclose(center, [forward, sideways, sideways]), center
    # 핵심 계약: 전방 정렬이 최대이고, 옆·뒤는 **구분 없이** 같은 바닥값이다.
    assert center[0] > center[1]
    assert math.isclose(center[1], center[2], abs_tol=1e-12)
    assert math.isclose(score(cfg, heading=math.pi / 2)[0], 0.0, abs_tol=1e-12)
    assert score(cfg, heading=math.pi)[0] == 0.0


def test_cross_track_gate_dominates_progress():
    cfg = RewardCfg()
    _, center, _ = score(cfg)
    gate, seven_cm, terms_seven = score(cfg, cross_track=0.07, movement=1.0)
    _, far, terms = score(cfg, cross_track=1.0)
    assert math.isclose(gate, math.exp(-(0.07 / cfg.sigma_y) ** 2), rel_tol=1e-12)
    # 기대값은 가중치에서 유도한다.  예전엔 `0.5 < seven_cm < 0.6` 으로 박아뒀는데
    # 그 범위는 sigma_y = 0.08 에 묶인 값이었고 0.05 로 줄이자 깨졌다 (2026-10-04).
    # 계약은 "7cm 이탈에서 보상이 중앙값보다 크게 깎이지만 여전히 양수" 다.
    expected_seven = (cfg.w_prog + cfg.w_align * gate
                      - cfg.w_lat * min((0.07 / cfg.sigma_y) ** 2, 4.0))
    assert math.isclose(seven_cm, expected_seven, rel_tol=1e-9)
    assert 0.0 < seven_cm < 0.7 * center
    assert center == 1.0
    assert terms_seven["lat"] < 0.0
    assert math.isclose(far, cfg.w_prog, rel_tol=1e-12)
    assert terms["align"] < 1e-12
    assert score(cfg, progress=0.0)[1] == 0.0


def test_constant_large_residual_is_penalized():
    cfg = RewardCfg()
    full = np.ones(3)
    _, _, terms = score(cfg, progress=0.0, action=full,
                        previous=full, previous2=full)
    assert terms["smooth"] == 0.0
    assert math.isclose(terms["resid"], -cfg.w_resid)


def test_small_lateral_improvement_beats_four_degree_heading_improvement():
    """횡오차 5mm 가 기수 4도보다 가치 있어야 한다.

    `w_head = 0.10` 이던 동안 이 계약이 깨져 있었고(centered 0.7740 <
    farther 0.8056), `s9` 정책이 5M 스텝 동안 정확히 그 틈을 최적화해
    holdout 에서 횡오차가 18.15 -> 21.6~22.0mm 로 역행했다.
    `w_head` 를 0.02 로 내려 복원했다 (2026-10-03).  그동안은 strict xfail 로
    표시해 두어, 가중치가 바뀌면 스위트가 알려주게 했다.
    """
    cfg = RewardCfg()
    # 이전 정책은 20→25mm 이탈을 허용하고 헤딩만 약 4도 개선했다.
    # 같은 진행량이라면 새 보상은 경로에 더 가까운 상태를 선호해야 한다.
    _, centered, _ = score(cfg, cross_track=0.020,
                           heading=math.radians(22), movement=1.0)
    _, farther, _ = score(cfg, cross_track=0.025,
                          heading=math.radians(18), movement=1.0)
    assert centered > farther


def test_diagonal_recovery_does_not_add_course_penalty():
    cfg = RewardCfg()
    straight_gate, straight, _ = score(cfg, progress=0.7, cross_track=0.10,
                                       movement=1.0)
    diagonal_gate, diagonal, terms = score(cfg, progress=0.7, cross_track=0.10,
                                           course_error=math.pi / 4,
                                           movement=1.0)
    assert math.isclose(straight_gate, diagonal_gate)
    assert math.isclose(straight, diagonal)
    assert terms["course"] == 0.0


def test_large_lateral_error_still_rewards_zero_residual_progress():
    cfg = RewardCfg()
    _, moving, _ = score(cfg, progress=1.0, cross_track=0.30, movement=1.0)
    _, stopped, _ = score(cfg, progress=0.0, cross_track=0.30, movement=0.0)
    assert moving > stopped == 0.0


def test_sideways_recovery_beats_stopping_when_heading_aligned():
    cfg = RewardCfg()
    full = np.ones(3)
    _, inward, terms = score(cfg, progress=0.0, cross_track=0.30,
                             heading=0.0, movement=1.0,
                             lateral_recovery=1.0, action=full,
                             previous=full, previous2=full)
    _, outward, _ = score(cfg, progress=0.0, cross_track=0.30,
                          heading=0.0, movement=1.0,
                          lateral_recovery=-1.0, action=full,
                          previous=full, previous2=full)
    _, stopped, _ = score(cfg, progress=0.0, cross_track=0.30,
                          heading=0.0, movement=0.0)
    assert inward > stopped == 0.0 > outward
    assert terms["recover"] == cfg.w_recover


def test_recovery_round_trip_has_no_discounted_advantage_over_staying():
    den, gamma, d0 = 0.0044, 0.998, 0.10
    # 바깥쪽 한 스텝 후 안쪽 두 스텝: 끝 위치와 걸린 시간이 정지와 같다.
    ds = (d0, d0 + 0.006, d0 + 0.002, d0)
    cyc = sum(gamma ** i * recovery_potential(ds[i], ds[i+1], den,
                                               gamma, terminal=i == 2)
              for i in range(3))
    stay = sum(gamma ** i * recovery_potential(d0, d0, den, gamma,
                                                terminal=i == 2)
               for i in range(3))
    assert math.isclose(cyc, stay, abs_tol=1e-12)


def test_environment_uses_paper_heading_gate():
    from env import RoverEnv
    from path import heading_at

    env = RoverEnv(arena_eval=True, eval_kind="sand", randomize=False, seed=0)
    try:
        env.reset(seed=0)
        rewards = []
        for yaw_error in (0.0, math.pi):
            env.s_prev = env.s_path - 0.001
            tangent_angle = heading_at(env.path, env.s_path - 0.0005)
            tangent = np.array([math.cos(tangent_angle), math.sin(tangent_angle)])
            env.last_xy = env.d.qpos[:2].copy() - 0.002 * tangent
            env.e_y = 0.07
            env.e_psi = yaw_error
            reward, _, info = env._reward(ZERO, np.zeros(4))
            expected = math.exp(-(0.07 / env.rew.sigma_y) ** 2) if yaw_error == 0.0 else 0.0
            assert math.isclose(info["gate"], expected, rel_tol=1e-6, abs_tol=1e-12)
            assert info["course_error_deg"] < 0.1
            rewards.append(reward)
        # 정렬된 쪽이 더 높다.  역방향(pi)은 게이트 0 + 헤딩 벌점으로 음수가
        # 되므로 "둘 다 양수" 를 요구하지 않는다 (w_head 도입 전 기대값이었다).
        assert rewards[0] > rewards[1], rewards
        assert rewards[0] > 0.0, rewards
    finally:
        env.close()


def test_environment_zeroes_recovery_potential_on_timeout():
    from env import RoverEnv, CTRL_HZ
    from train import stage1_paths

    env = RoverEnv(paths=stage1_paths("sand", 1234, 10), arena_eval=True,
                   eval_kind="sand", randomize=False, episode_s=1 / CTRL_HZ)
    try:
        env.reset(seed=0)
        prev_dist = env.path.distance(env.d.qpos[:2])
        den = env.v_cruise / CTRL_HZ
        _, _, _, truncated, info = env.step(ZERO)
        assert truncated
        assert math.isclose(info["rterm_recover"],
                            env.rew.w_recover * prev_dist / den,
                            rel_tol=1e-6, abs_tol=1e-9)
    finally:
        env.close()


if __name__ == "__main__":
    test_paper_heading_gate_suppresses_sideways_and_reverse_alignment()
    test_cross_track_gate_dominates_progress()
    test_constant_large_residual_is_penalized()
    test_small_lateral_improvement_beats_four_degree_heading_improvement()
    test_diagonal_recovery_does_not_add_course_penalty()
    test_large_lateral_error_still_rewards_zero_residual_progress()
    test_sideways_recovery_beats_stopping_when_heading_aligned()
    test_recovery_round_trip_has_no_discounted_advantage_over_staying()
    test_environment_uses_paper_heading_gate()
    test_environment_zeroes_recovery_potential_on_timeout()
    print("path reward: OK")
