# Pure IK: Isaac Lab ↔ MuJoCo, 2026-10-09

This is a **flat-ground actuator and IK baseline**, not a sand or learned-policy result. The Isaac `Residual3` action is always zero, so both engines receive the same body command through the same Torch IK. MuJoCo uses the current CAD MJCF with terramech forces disabled and an explicit wheel velocity servo matching the Isaac `DCMotorCfg`; this is not the default duty-controlled `RoverEnv` rollout. Steering targets are limited to 5.45 rad/s in both engines. The Isaac rocker soft constraint uses K=20 N·m/rad and D=0.3 N·m·s/rad; MuJoCo retains its hard differential constraint.

Physics step: 0.005 s. Control step: 0.020 s. One deterministic rover, initial yaw 0, plane friction 1.0, noise and domain randomization off. `full` has a 1 s settle and 2 s each of forward `(0.18,0,0)`, strafe `(0,0.12,0)`, yaw `(0,0,0.60)`, arc `(0.12,0.04,0.45)`, and reverse `(-0.15,0,0)`. `arc_left` and `arc_right` begin separately from rest, then run their mirrored arc for 4 s. Linear commands are m/s; yaw commands are rad/s.

The comparison uses the last 1 s of each commanded phase for **steady** body velocity, and integrates each component over the entire phase. Percentage differences are normalized by the magnitude of the corresponding command, not by the observed value. The predeclared acceptance limit is 20% for both measures on each nonzero commanded component. The command schedule and IK wheel/steering targets match to the comparator's displayed precision in every run.

| Scenario / component | Steady Isaac / MuJoCo | Steady difference | Integrated difference | Result |
|---|---:|---:|---:|---|
| Full, forward vx | 0.180 / 0.180 m/s | 0.0% | 1.0% | Pass |
| Full, strafe vy | 0.120 / 0.120 m/s | 0.0% | 4.2% | Pass |
| Full, yaw ω | 0.501 / 0.525 rad/s | 4.0% | 11.8% | Pass |
| Full, arc vx | 0.112 / 0.124 m/s | 9.5% | 10.7% | Pass |
| **Full, arc vy** | **0.042 / 0.031 m/s** | **29.9%** | **18.7%** | **Fail: steady** |
| Full, arc ω | 0.420 / 0.455 rad/s | 7.8% | 8.2% | Pass |
| Full, reverse vx | −0.150 / −0.150 m/s | 0.0% | 3.1% | Pass |
| Arc left, vx / vy / ω | — | 0.8% / 5.6% / 1.5% | 2.4% / 6.0% / 1.2% | Pass |
| Arc right, vx / vy / ω | — | 1.2% / 2.5% / 2.6% | 0.8% / 7.0% / 2.1% | Pass |

Final planar pose differences are 0.0370 m and 8.65° in the full 11 s sequence, 0.0135 m and 1.44° for the independent left arc, and 0.0122 m and 1.11° for the independent right arc. Mean `|left rocker + right rocker|` is 0.181° in Isaac and 0.002° in MuJoCo over the full run. An earlier Isaac K=2,D=0.1 ablation produced about 1.76° in each independent arc, so the stronger soft constraint improves rocker parity without making the two constraints identical.

Agreement between engines is separate from tracking accuracy: in the full yaw phase, both undertrack the 0.600 rad/s request (Isaac 0.501, MuJoCo 0.525 rad/s). The 20% check compares engines; it does not certify that pure IK is an adequate controller for every commanded maneuver.

**Conclusion:** pure IK targets and the uncomplicated translation response match. Independent left and right arcs meet the 20% criterion. The sequential full run **does not pass** because arc lateral velocity differs after prior forward, strafe, and yaw maneuvers. That points to a state/history-dependent plant or contact mismatch, but this test does not isolate its cause. It must not be described as an IK numerical error or as evidence that the Isaac sand task is validated. These rollouts do not test terrain reset, granular sinkage, sensor noise, RL, or sim-to-real behavior.

The six `.npz` files in this directory are the raw 50 Hz series (`command`, `body`, `xy`, `yaw`, steering/wheel targets and actuals, rocker angles). Reproduce the comparisons from the repository root:

```bash
python3 sim/rl/compare_ik_engines.py \
  --isaac sim/isaac/ksrc_rover/benchmarks/2026-10-09_pure_ik/pure_ik_isaac_full_k20.npz \
  --mujoco sim/isaac/ksrc_rover/benchmarks/2026-10-09_pure_ik/pure_ik_mujoco_full_new.npz
python3 sim/rl/compare_ik_engines.py \
  --isaac sim/isaac/ksrc_rover/benchmarks/2026-10-09_pure_ik/pure_ik_isaac_arc_left_k20.npz \
  --mujoco sim/isaac/ksrc_rover/benchmarks/2026-10-09_pure_ik/pure_ik_mujoco_arc_left.npz
python3 sim/rl/compare_ik_engines.py \
  --isaac sim/isaac/ksrc_rover/benchmarks/2026-10-09_pure_ik/pure_ik_isaac_arc_right_k20.npz \
  --mujoco sim/isaac/ksrc_rover/benchmarks/2026-10-09_pure_ik/pure_ik_mujoco_arc_right.npz
```

To regenerate on the server after `bash sim/isaac/ksrc_rover/deploy.sh`, check for a free GPU, then run `bash run.sh /workspace/scripts/ik_benchmark_isaac.py --scenario full --out /workspace/_verify/pure_ik_isaac_full_k20.npz --headless` from `~/gyusa/ksrc_rover`. Substitute `arc_left` or `arc_right` for the other scenarios. On a local MuJoCo/PyTorch installation, run `python3 sim/rl/ik_benchmark_mujoco.py --scenario full --out /tmp/pure_ik_mujoco_full.npz` from the repository root, substituting the scenario as needed.
