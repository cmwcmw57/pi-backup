# ================================================================
# local_mujoco_probe.py
# 설명: 팀원 ArenaRover와 stv3를 그대로 호출해 기존 모래 맵의 15° 등판을 로컬에서 기록한다.
# 사용법:
#   source /opt/ros/humble/setup.bash
#   LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 OMP_NUM_THREADS=1 python -m experiments.site_sim.local_mujoco_probe --repo PATH --checkpoint PATH --output PATH
# ================================================================
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def main() -> None:
    """원본 물리/정책 코드를 수정하지 않고 고정 직진 명령의 결과를 보존한다."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--friction-duty', type=float, default=0.)
    ap.add_argument('--soil-preset', choices=('isaac-legacy','bridge-random'), default='isaac-legacy')
    ap.add_argument('--seconds', type=float, default=120.)
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()
    args.repo = args.repo.resolve();args.checkpoint = args.checkpoint.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(args.repo/'sim/rl'), str(args.repo/'ros')]
    import mujoco
    import torch
    import sim2sim_isaac as s2s
    from isaac_bridge import ArenaRover

    torch.set_num_threads(1);torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    spawn = (-1.2, .7, -np.pi/2)
    rover = ArenaRover('sand', 'sand', rng, False, spawn, friction_duty=args.friction_duty)
    rover.reset()
    if args.soil_preset == 'isaac-legacy':
        from experiments.site_sim.ksrc_soil import SoilParameters
        rover.tm.p = type(rover.tm.p)(mu_max=.4, **asdict(SoilParameters()))
        rover.tm.reset()
    rover.set_mu(np.full(4, .4))
    env, agent = s2s.load_run_cfg(str(args.checkpoint))
    policy = s2s.Policy(str(args.checkpoint), agent)
    obs = s2s.ObsBuilder(env, agent['obs_groups']['policy'], 1, True, rng, vo='dr')
    action_cfg = env['actions']['rover']
    amap = s2s.ActionMap('Res8', 1, action_cfg.get('rate_limit', 0.), True)
    order = [0, 2, 1, 3]
    files = [args.repo/'sim/rl'/n for n in ('sim2sim_isaac.py','terramech.py','mjcf.py','terrain.py')]
    files += [args.repo/'ros/isaac_bridge.py',args.repo/'sim/rl/assets/sand_hfield.npz',args.checkpoint]
    files += list((args.checkpoint.parent/'params').glob('*.yaml'))
    files += [Path(__file__).resolve(), Path(__file__).with_name('ksrc_soil.py').resolve()]
    meta = dict(repo_commit=subprocess.check_output(['git','-C',str(args.repo),'rev-parse','HEAD'],text=True).strip(),
                python=sys.version, mujoco=mujoco.__version__, torch=torch.__version__, seed=args.seed,
                spawn=spawn, goal_y=-1.55, corridor_x=[-1.36,-1.04], command=[.15,0.,0.],
                mu_requested=.4, mu_effective=rover.mu_effective().tolist(), soil=asdict(rover.tm.p),soil_preset=args.soil_preset,
                friction_duty=args.friction_duty, wheel_joint_friction=rover.m.dof_frictionloss[rover.wheel_v].tolist(),
                drive='pi', sensors='real', obs_delay=1, act_delay=1, noise=True, vo='dr',
                mass_kg=float(rover.m.body_mass.sum()), hardware_commands_enabled=False,
                exact_teammate_run=False,
                source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (args.output/'manifest.json').write_text(json.dumps(meta,indent=2))
    rows = {k:[] for k in ('t','pos','quat','lin_b','pitch','duty','wheel_w','steer','slip','z','Fn','R','mu','action','qpos')}
    start = time.monotonic();reason = 'duration_elapsed'
    for k in range(int(args.seconds/s2s.CONTROL_DT_S)):
        state = rover.state();cmd = np.array([.15,0.,0.])
        observation = obs(dict(cmd=cmd,s=state,amap=amap,mu_eff=rover.mu_effective(),act_delay=1,obs_delay=1))
        action = policy(observation)
        rover.step(amap.process(action,cmd));state = rover.state()
        pitch = float(np.degrees(np.arctan2(-state['grav_b'][0],-state['grav_b'][2])))
        values = dict(t=(k+1)*s2s.CONTROL_DT_S,pos=state['pos'],quat=state['quat'],lin_b=state['lin_b'],pitch=pitch,
                      duty=rover.duty,wheel_w=state['wheel_w'],steer=state['steer'],slip=s2s.wheel_slip(state),
                      z=rover.tm.z[order],Fn=rover.tm.Fn[order],R=rover.tm.R[order],
                      mu=rover.m.geom_friction[rover.tm.wg,0][order],action=action,qpos=rover.d.qpos)
        for key in rows:rows[key].append(np.asarray(values[key]).copy())
        if k%500 == 0:print(f't={(k+1)*.02:.2f} pos={np.round(state["pos"],3)} pitch={pitch:.1f}',flush=True)
        if not np.isfinite(state['pos']).all():reason='invalid_state';break
        if abs(state['grav_b'][2]) < np.cos(np.deg2rad(40)):reason='tilt_limit';break
        if not -1.36 <= state['pos'][0] <= -1.04:reason='corridor_limit';break
        if state['pos'][1] <= -1.55:reason='ascent_reached';break
    data = {k:np.asarray(v) for k,v in rows.items()}
    np.savez_compressed(args.output/'telemetry.npz',**data)
    slope = abs(data['pitch']) > 10.
    summary = dict(reason=reason,duration_sim_s=float(data['t'][-1]),duration_wall_s=time.monotonic()-start,
                   final_position_m=data['pos'][-1].tolist(),max_abs_pitch_deg=float(abs(data['pitch']).max()),
                   progress_m=float(spawn[1]-data['pos'][-1,1]),
                   mean_abs_slip_on_slope=np.abs(data['slip'][slope]).mean(axis=0).tolist() if slope.any() else None,
                   max_sinkage_m=data['z'].max(axis=0).tolist(),hardware_commands_enabled=False)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
