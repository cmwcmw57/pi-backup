"""Compare fixed-command pure IK rollouts from Isaac Lab and MuJoCo.

The acceptance check is on steady commanded velocity components (last 1 s of
each segment).  Target parity is checked separately so a controller or
geometry mismatch cannot be mistaken for a physics-engine difference.
"""

import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac", required=True)
    parser.add_argument("--mujoco", required=True)
    args = parser.parse_args()
    a = np.load(args.isaac)
    b = np.load(args.mujoco)
    if not (np.array_equal(a["phase"], b["phase"]) and np.allclose(a["command"], b["command"])):
        raise ValueError("Command schedules differ")
    if len(a["body"]) != len(b["body"]):
        raise ValueError("Rollout lengths differ")

    steer_diff = float(np.max(np.abs(a["steer_target"] - b["steer_target"])))
    wheel_diff = float(np.max(np.abs(a["wheel_target"] - b["wheel_target"])))
    print(f"IK target parity: steering max Δ={steer_diff:.6f} rad; wheel max Δ={wheel_diff:.6f} rad/s")
    print("phase       cmd     steady Isaac/MJ       steady Δ    integral Isaac/MJ      integral Δ")
    columns = (
        ("forward", 0), ("strafe", 1), ("yaw", 2),
        ("arc_vx", 0), ("arc_vy", 1), ("arc_om", 2), ("reverse", 0),
    )
    passes = []
    for label, axis in columns:
        phase = "arc" if label.startswith("arc_") else label
        all_indices = np.where(a["phase"] == phase)[0]
        if not len(all_indices):
            continue
        indices = all_indices[-50:]
        target = float(a["command"][indices, axis].mean())
        av = float(a["body"][indices, axis].mean())
        bv = float(b["body"][indices, axis].mean())
        relative = abs(av - bv) / abs(target)
        wa = float(np.abs(a["wheel_actual"][indices] - a["wheel_target"][indices]).mean())
        wb = float(np.abs(b["wheel_actual"][indices] - b["wheel_target"][indices]).mean())
        ai = float(a["body"][all_indices, axis].sum() * a["dt"])
        bi = float(b["body"][all_indices, axis].sum() * b["dt"])
        integrated = abs(ai - bi) / (abs(target) * len(all_indices) * float(a["dt"]))
        print(f"{label:<10} {target:+.3f}   {av:+.3f}/{bv:+.3f}        {relative*100:5.1f}%    "
              f"{ai:+.3f}/{bi:+.3f}           {integrated*100:5.1f}%")
        print(f"  wheel speed MAE Isaac/MuJoCo: {wa:.4f}/{wb:.4f} rad/s")
        passes.append(relative <= 0.20 and integrated <= 0.20)
    target_match = steer_diff < 1e-4 and wheel_diff < 1e-3
    final_xy_delta = float(np.linalg.norm(a["xy"][-1] - b["xy"][-1]))
    final_yaw_delta = float(np.abs(np.unwrap(a["yaw"][:, 0])[-1] - np.unwrap(b["yaw"][:, 0])[-1]))
    print(f"final pose Δ: xy={final_xy_delta:.4f} m, yaw={np.degrees(final_yaw_delta):.2f} deg")
    if "rocker_actual" in a and "rocker_actual" in b:
        for name, data in (("Isaac", a), ("MuJoCo", b)):
            print(f"{name} rocker sum mean |L+R|: "
                  f"{np.degrees(np.abs(data['rocker_actual'].sum(axis=1)).mean()):.3f} deg")
    print(f"RESULT {'PASS' if target_match and all(passes) else 'FAIL'} "
          f"target_parity={target_match} steady_and_integral_20pct={all(passes)}")


if __name__ == "__main__":
    main()
