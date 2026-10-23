# KSRC Isaac Lab task

This directory mirrors `~/gyusa/ksrc_rover` on `hpi-lab`. Its `source/ksrc_rover/` package registers six Gymnasium task IDs (three action modes and their `-Play` variants). `scripts/train.py` and `scripts/play.py` run the version-matched Isaac Lab rsl_rl scripts from the server image. The image already contains `rsl_rl`; no extra pip install is needed.

From the repository root, `bash sim/isaac/ksrc_rover/deploy.sh` synchronizes the tracked task files without replacing the server's `assets/` or legacy diagnostic scripts. `scripts/_spikes/` is reserved for those diagnostics; they remain at their original paths on the server to preserve existing CLI references.

The default is a **flat-ground bring-up task**, not a validated sand-training environment. It uses the converted `/workspace/assets/usd/rover.usd` provisionally, 5 ms physics steps, decimation 4 (20 ms control), an 8 s episode, and friction randomization. `run.sh` mounts this directory at `/workspace`, mounts the existing terrain package read-only at `/terrain`, and keeps logs under `/workspace/logs/rsl_rl`.

On the server, after checking GPU availability:

```bash
cd ~/gyusa/ksrc_rover
bash run.sh /workspace/scripts/smoke.py --task Isaac-KSRC-Rover-Direct8-v0 --headless
KSRC_TERRAIN_KIND=rough_slope KSRC_TERRAIN_MODE=narrow bash run.sh /workspace/scripts/smoke.py --task Isaac-KSRC-Rover-Direct8-v0 --headless
bash run.sh /workspace/scripts/train.py --task Isaac-KSRC-Rover-Direct8-v0 --headless --num_envs 64 --max_iterations 10
```

The training command is an example only; do not use this task for comparative training until the asset, rocker coupling, actuator response, observation contract, terrain reset, and Isaac↔MuJoCo pure-IK baseline have been validated. `Residual3` and `Res8` use `utils/ik_torch.py`; `python3 sim/rl/ik_validate.py` checks its parity with firmware C. The actor currently has a common 29D single-frame observation (command, gyro, gravity, steering positions/rates, wheel rates, applied 8D targets); the older migration note's 26D is not implemented or frozen. The critic is not yet privileged and teacher–student distillation is not wired in.

The [2026-10-09 pure-IK benchmark](benchmarks/2026-10-09_pure_ik/RESULTS.md) records the matched flat-ground Isaac/MuJoCo rollouts and six raw `.npz` series. Use `scripts/ik_benchmark_isaac.py` on the server, `sim/rl/ik_benchmark_mujoco.py` locally, and `sim/rl/compare_ik_engines.py` to compare. Both use 5 ms physics and 20 ms command steps. Current flat-ground calibration is a 0.20 wheel velocity-servo gain, a 5.45 rad/s steering-target slew limit, and an Isaac rocker soft constraint K=20,D=0.3. Straight travel and independent left/right arcs meet the 20% comparison criterion, but the full sequential run fails on arc lateral speed (29.9% difference); the Phase 3 baseline remains incomplete.

Each mode has its own rsl_rl experiment directory: `ksrc_rover_residual3`, `ksrc_rover_res8`, or `ksrc_rover_direct8`. When a trained checkpoint exists, `scripts/export_jit.py --task Direct8 --checkpoint /workspace/logs/rsl_rl/ksrc_rover_direct8/<run>/model_XXX.pt --headless` exports the actor and its normalizer to `exported/policy.pt` beside the checkpoint. The export path has not yet been exercised because no Isaac policy has been trained.
