"""Provisional Isaac-side physical constants; compare against firmware before training."""

import math
import os

MODULE_NAMES = ("fl", "fr", "rl", "rr")
# CAD rover_description: steering-axis positions in the base frame.  The
# rocker pivot is at +/-0.085 m and each steering joint adds +/-0.0475 m.
MODULE_XY_M = (
    (0.102431, 0.1325), (0.102431, -0.1325),
    (-0.102431, 0.1325), (-0.102431, -0.1325),
)
WHEEL_RADIUS_M = 0.047503
WHEEL_MAX_RAD_S = 7.96          # no-load at 12 V. Measured 2026-10-13: 7.68 @ 11.65 V, 7.53 @ 11.25 V
                                # (4-motor mean, ±2.4 % spread) -> 7.9 at 12 V; DR voltage_scale covers the battery
WHEEL_MAX_M_S = WHEEL_RADIUS_M * WHEEL_MAX_RAD_S
# ±90°: URDF/USD steering joint limit (rover.urdf lower/upper ±1.5708, rover.usd ±90).
# Was ±100° until 2026-10-11 -- commands in 90–100° then hit the joint stop in Isaac.
STEER_LIMIT_RAD = math.radians(90.0)
STEER_MAX_RAD_S = 5.45        # velocity cap of the servo profile (datasheet; never reached within ±90°)
# STS3215 12 V (C018) as measured on the rover 2026-10-13 (pi/common/sysid_*, wheels lifted, 4 servos):
# trapezoid profile at 7.7 rad/s² (= 439 °/s²; the ACC register has no effect: 0/50/100/254 identical),
# 90° in 0.87 s, motion starts 34 ms after the goal is written, servo spread < 1 %.
# Small steps: goals within ~0.25° of the current angle are ignored (min. command 0.3–0.5°,
# approach hysteresis 0.37°).
STEER_ACCEL_RAD_S2 = 7.7
STEER_DELAY_S = 0.034
STEER_DEADBAND_RAD = math.radians(0.25)
STEER_EFFORT_NM = 2.94        # 12 V C018 stall 30 kg·cm (was the 7.4 V C001 value 1.912)
DRIVE_EFFORT_NM = 1.96
DRIVE_ARMATURE = 9.8e-3       # measured 2026-10-13, see assets/rover_cfg.py
CONTROL_DT_S = 0.02
RESIDUAL3_SCALE = (0.11, 0.085, 0.45)
RESIDUAL_RATE_LIMIT = 0.10
# Res8 residual range. 2026-10-11: ±4°/±10% -> ±10°/±30%. With exploration noise 0.3 the
# old range nudged a wheel by only ~1°/3%, lost in μ changes, latency and noise, so the
# teacher (μ visible) stayed at pure IK (iter 150: xy -9.6% vs IK, |a| 0.17, 0% at the limit).
# The earlier cut from ±8°/±20% was judged in the buggy env (spawn/armature), not evidence.
# 2026-10-12: ±10°/±30% -> ±30°/±60% ("wide Res8" = Direct8 authority starting from IK).
# Direct8 never learned linear tracking from scratch (3 runs); the hard cases (all wheels
# slick, one side slick) may need more than ±10°/±30%. Smoothness penalties stay at ×30.
# KSRC_RES8_STEER_DEG / KSRC_RES8_WHEEL override (evaluating checkpoints trained at another range)
RES8_STEER_SCALE_RAD = math.radians(float(os.environ.get("KSRC_RES8_STEER_DEG", "30.0")))
RES8_WHEEL_SCALE = float(os.environ.get("KSRC_RES8_WHEEL", "0.60"))
