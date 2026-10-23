"""수동 텔레옵 방향 전환 시 정지·재출발 조건 검사."""
import numpy as np

from teleop_transition import TeleopTransitionGuard


def test_turn_brakes_once_then_releases():
    guard = TeleopTransitionGuard()
    side = np.array([0.0, 0.25, 0.0])
    assert np.allclose(guard.apply(side, [0.25, 0.0]), 0.0)
    assert np.allclose(guard.apply(side, [0.10, 0.0]), 0.0)
    assert np.allclose(guard.apply(side, [0.01, 0.0]), side)
    assert not guard.braking


def test_slide_cannot_hold_brake_forever():
    guard = TeleopTransitionGuard(max_brake_seconds=0.06, control_hz=50)
    side = np.array([0.0, 0.25, 0.0])
    assert np.allclose(guard.apply(side, [0.25, 0.0]), 0.0)
    assert np.allclose(guard.apply(side, [0.25, 0.0]), 0.0)
    assert np.allclose(guard.apply(side, [0.25, 0.0]), 0.0)
    assert np.allclose(guard.apply(side, [0.25, 0.0]), side)
    assert np.allclose(guard.apply(side, [0.25, 0.0]), side)
    guard.apply([0.0, 0.0, 0.0], [0.25, 0.0])
    assert np.allclose(guard.apply(side, [0.25, 0.0]), 0.0)


if __name__ == "__main__":
    test_turn_brakes_once_then_releases()
    test_slide_cannot_hold_brake_forever()
    print("teleop transition: OK")
