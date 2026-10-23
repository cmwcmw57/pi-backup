"""Nav2 경로 관측이 학습 환경의 정책 입력 배치와 일치하는지 검사."""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "sim" / "rl"))

import path as pth
from policy_observation import PolicyPath, set_route_fields


def test_target_point_and_distance_match_pursuit_route_fields():
    plan = PolicyPath([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    xy = np.array([0.8, -0.1])
    lookahead = pth.pursuit_distance(0.22, 1.2, 0.18, 0.45)
    features = plan.features(xy, 0.0, lookahead)
    expected = pth.lookahead_body(plan.ref, 0.8, xy, 0.0,
                                   dists=(lookahead,))[0]
    assert np.allclose(features[3], expected)
    assert np.isclose(features[4], 0.264)

    frame = np.zeros(36, dtype=np.float32)
    frame[-3:] = [0.1, 0.2, 0.3]
    set_route_fields([frame], [0.22, 0.0, 0.1], 0.318, features)
    assert np.allclose(frame[30:32], expected)
    assert np.isclose(frame[32], lookahead)
    assert np.allclose(frame[-3:], [0.1, 0.2, 0.3])
