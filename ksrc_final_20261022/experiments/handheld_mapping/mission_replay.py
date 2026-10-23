# ================================================================
# mission_replay.py
# 설명: 합성 pose와 정체를 실제 임무 runtime에 넣어 우회·정상·복귀를 재생한다.
# 사용법: python -m experiments.handheld_mapping.mission_replay MISSION OUTPUT --scenario stall30
# ================================================================
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from experiments.handheld_mapping.site_map import SiteMap
from experiments.handheld_mapping.site_mission import SiteMission
from experiments.handheld_mapping.mission_runtime import MissionRuntime,angle_wrap
from experiments.handheld_mapping.storage import atomic_json

SCENARIOS=('success','stall30','stall30_15','stall_all','tracking_loss','return_stall','descent_blocked','stall30_entry')


def demo_mission() -> SiteMission:
    """기하 경사와 견인 정체를 분리한 합성 3경로 demo 입력."""
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector([[0,0,0],[4,0,0],[4,3,0],[0,3,0]]),
                                  o3d.utility.Vector3iVector([[0,1,2],[0,2,3]]))
    site=SiteMap.from_mesh(mesh,resolution=.05,robot_radius=.1)
    routes={'10':[[.5,1.5],[1.,.5],[3.,.5],[3.5,1.5]],
            '15':[[.5,1.5],[1.,2.5],[3.,2.5],[3.5,1.5]],
            '30':[[.5,1.5],[1.,1.5],[3.5,1.5]]}
    return SiteMission(site,[.5,.5,0.],[.5,1.5],[3.5,1.5],routes,site_name='demo')


def step_plant(xy: np.ndarray, yaw: float, cmd: dict, dt: float) -> tuple[np.ndarray, float]:
    """몸 좌표 (vx, vy)를 세계로 돌려 한 스텝 적분한다. cmd에 vy_m_s가 없으면 0(기존 unicycle 적분과 같다)."""
    xy=xy+cmd['vx_m_s']*dt*np.array([np.cos(yaw),np.sin(yaw)])+cmd.get('vy_m_s',0.)*dt*np.array([-np.sin(yaw),np.cos(yaw)])
    return xy,angle_wrap(yaw+cmd['yaw_rate_rad_s']*dt)


def replay(mission: SiteMission, output: Path, scenario: str='stall30') -> dict:
    """일정 dt의 비물리 planar plant를 연결한다. 모래 물리/등판 성능 검증이 아니다."""
    if scenario not in SCENARIOS:raise ValueError('unknown scenario')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    runtime=MissionRuntime(mission);xy=mission.start[:2].copy();yaw=float(mission.start[2])
    stall={'success':set(),'stall30':{'30'},'stall30_15':{'30','15'},'stall_all':{'30','15','10'},
           'tracking_loss':set(),'return_stall':set(),'descent_blocked':set(),'stall30_entry':set()}[scenario]
    rows=[];dt=.1
    for step in range(20_000):
        stamp=round(step*dt,6)
        valid=not(scenario=='tracking_loss' and 12.<stamp<18.)
        packet={'stamp_s':stamp,'now_s':stamp,'frame':'site','site_identity':mission.site.identity,
                'position_m':xy.tolist(),'yaw_rad':yaw,'tracking_valid':valid}
        if (scenario=='descent_blocked' and runtime.state=='ascending'
                and np.linalg.norm(xy-mission.summit)<=mission.settings['arrival_tolerance_m']):
            col,row=mission.site.cells(mission.routes['10'][1])
            mission.site.blocked[max(0,row-1):row+2,max(0,col-1):col+2]=True
        result=runtime.update(packet,stamp)
        cmd=result['command'];rows.append({'stamp_s':stamp,'x_m':xy[0],'y_m':xy[1],
            'yaw_rad':yaw,'state':runtime.state,'route':runtime.route,'tracking_valid':valid,**cmd})
        if runtime.state in ('complete','failed'):break
        blocked=((runtime.state=='ascending' and runtime.route in stall and xy[0]>1.3)
                 or (scenario=='return_stall' and runtime.state=='returning')
                 or (scenario=='stall30_entry' and runtime.state=='trial_entry' and xy[0]>.65))
        if not blocked:
            xy,yaw=step_plant(xy,yaw,cmd,dt)
    with (output/'commands.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (output/'events.jsonl').open('w') as stream:
        for event in runtime.events:stream.write(json.dumps(event)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig=Figure(figsize=(8,6));FigureCanvasAgg(fig);axis=fig.subplots()
    for name,path in mission.routes.items():axis.plot(*path.T,'--',label=name+' deg named route')
    axis.plot([r['x_m'] for r in rows],[r['y_m'] for r in rows],label='synthetic execution')
    axis.set(xlabel='site X (m)',ylabel='site Y (m)',title='SIMULATION ONLY: '+scenario,aspect='equal');axis.legend()
    fig.savefig(output/'replay.png')
    summary={'state':runtime.state,'visits':runtime.visits,'failed_ascent_routes':runtime.failed_routes,
             'descent_route':'10','simulation_only':True,'hardware_commands_enabled':False,
             'paused_samples':runtime.paused_samples,'scenario':scenario,'samples':len(rows),
             'duration_s':rows[-1]['stamp_s'],'reason':result['reason'],
             'scope':'synthetic planar plant and explicit stalls; no real soil, actuator, localization accuracy validation'}
    atomic_json(output/'summary.json',summary)
    return summary


def main() -> None:
    parser=argparse.ArgumentParser(description='모터 없는 임무 시나리오 재생')
    parser.add_argument('mission',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--scenario',choices=SCENARIOS,default='stall30')
    args=parser.parse_args();print(json.dumps(replay(SiteMission.load(args.mission),args.output,args.scenario),ensure_ascii=False))


if __name__=='__main__':main()
