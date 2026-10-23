"""Episode termination predicates."""

import math


def excessive_tilt(env, max_tilt_rad: float = math.radians(80.0)):
    # High threshold for the initial smoke task; base_link pitch offset needs calibration.
    return -env.scene["robot"].data.projected_gravity_b[:, 2] < math.cos(max_tilt_rad)
