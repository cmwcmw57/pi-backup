"""Provisional converted rover USD with measured motor limits."""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from ksrc_rover.utils.contract import (
    DRIVE_ARMATURE, DRIVE_EFFORT_NM, STEER_EFFORT_NM, STEER_MAX_RAD_S, WHEEL_MAX_RAD_S,
)


ROVER_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(usd_path=os.environ.get("KSRC_ROVER_USD", "/workspace/assets/usd/rover.usd")),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.15)),
    actuators={
        "steering": ImplicitActuatorCfg(
            joint_names_expr=[".*_steering_joint"],
            effort_limit_sim=STEER_EFFORT_NM,
            velocity_limit_sim=STEER_MAX_RAD_S,
            stiffness=12.0,
            damping=0.30,
        ),
        "drive": DCMotorCfg(
            joint_names_expr=[".*_wheel_joint"],
            saturation_effort=DRIVE_EFFORT_NM,
            effort_limit=DRIVE_EFFORT_NM,
            velocity_limit=WHEEL_MAX_RAD_S,
            stiffness=0.0,
            # Open-loop duty model: tau = tau_stall*u - D*w  ==  D*(u*w0 - w) with
            # D = tau_stall / w_noload = 1.96 / 7.96 = 0.246 (same as MuJoCo
            # config.wheel_damping).  Explicit loop at 5 ms: D*dt/I = 0.246*0.005/0.00156
            # = 0.79 < 2, stable (1.0 gave a zero-command limit cycle).  0.20 until 10-12.
            damping=DRIVE_EFFORT_NM / WHEEL_MAX_RAD_S,
            # Reflected rotor inertia of the SPG30E-GR131 gearbox.  Measured 2026-10-13
            # (pi/common/sysid_*, duty steps, wheels lifted): mechanical time constant
            # tau = J/D = 40 ms on all 4 motors, the same with and without the wheel
            # (motor 1 had none) -> J = 0.040 * 0.246 = 9.8e-3 kg m^2, almost all rotor.
            # Was 1.5e-3 (tau 6 ms).  The USD wheel alone is 5.5e-5 kg m^2; explicit loop
            # D*dt/I = 0.246*0.005/0.0098 = 0.13 < 2, stable.
            armature=DRIVE_ARMATURE,
        ),
        "rocker": ImplicitActuatorCfg(
            joint_names_expr=[".*_rocker_joint"],
            effort_limit_sim=1.0,
            velocity_limit_sim=10.0,
            stiffness=0.0,
            damping=0.1,
        ),
    },
)
