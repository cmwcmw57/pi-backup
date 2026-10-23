"""The wheel selector must distinguish steering direction from traction loss."""
from types import SimpleNamespace

import numpy as np

from train_wheel_residual import wheel_asymmetry


def test_opposite_wheel_directions_are_not_a_traction_asymmetry():
    core = SimpleNamespace(
        tm=SimpleNamespace(slip=np.array([0.3, -0.3, 0.3, -0.3]),
                           Fn=np.ones(4)),
        wheel_ref_applied=np.array([1.0, -1.0, 1.0, -1.0]),
        v_max=1.0,
    )
    assert wheel_asymmetry(core, "raw") == 0.3
    assert np.isclose(wheel_asymmetry(core, "traction"), 0.0)

    core.tm.slip[0] = 0.8
    assert wheel_asymmetry(core, "traction") > 0.15

    core.tm.Fn[0] = 0.0
    assert np.isclose(wheel_asymmetry(core, "traction"), 0.0)
