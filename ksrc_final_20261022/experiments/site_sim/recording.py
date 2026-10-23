# ================================================================
# recording.py
# 설명: simulator 임무의 지형 costmap·방문 경험·telemetry·결과와 관찰 파일을 저장한다.
# 사용법: from experiments.site_sim.recording import RunRecorder, geometry_costs
# ================================================================
from __future__ import annotations

import csv
import json
from pathlib import Path
import time

import numpy as np
from experiments.handheld_mapping.costmap import RoverLimits,build_costmap
from experiments.handheld_mapping.site_mission import SiteMission
from experiments.handheld_mapping.storage import atomic_json


def geometry_costs(mission: SiteMission) -> tuple[np.ndarray,np.ndarray]:
    """동일한 geometry에 normal/trial 경사 기준을 적용하고 site clearance를 보존한다."""
    s=mission.site;zero=np.zeros(s.observed.shape,np.float32)
    layers={'height_m':s.height_m,'slope_deg':s.slope_deg,'observed':s.observed,
            'valid':s.observed.copy(),'roughness_m':zero,'step_m':zero,'vertical_span_m':zero}
    result=[]
    for key in ('normal_slope_deg','trial_slope_deg'):
        limit=mission.settings[key]
        inclusive_limit=min(float(np.nextafter(np.float32(limit+1e-4),np.float32(np.inf))),89.999999)
        cost=build_costmap(layers,s.resolution,RoverLimits(max_slope_deg=inclusive_limit,max_step_m=s.max_step_m,
                                                       safety_radius_m=0.))
        cost[s.observed&~s.safe_mask(limit)]=254
        result.append(cost)
    return tuple(result)


class RunRecorder:
    """주행 시계열과 고정 크기 경험 grid를 새 run에 누적하며 부분 실패를 보존한다."""

    def __init__(self, mission: SiteMission, output: Path, *, max_snapshots: int=1200,
                 fault_injection: bool=False, pose_source: str='sim_ground_truth', control_dt: float=.05) -> None:
        if type(max_snapshots) is not int or max_snapshots<2:raise ValueError('at least two snapshots required')
        self.mission=mission;self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
        self.fault_injection=bool(fault_injection);self.fault_occurred=False
        self.pose_source=pose_source
        self.live=self.output/'live';self.live.mkdir();(self.output/'snapshots').mkdir()
        self.normal,self.trial=geometry_costs(mission)
        shape=mission.site.observed.shape
        self.visits=np.zeros(shape,np.uint32);self.stuck=np.zeros(shape,np.uint32)
        self.last_seen=np.full(shape,np.nan);self.last_stamp=-1.;self.last={};self.trajectory=[];self.estimated_trajectory=[]
        self.event_cursor=0;self.snapshot_count=0;self.max_snapshots=max_snapshots
        self.next_snapshot=0.;self.last_state=None;self.last_publish=-1.;self.done=False
        self.navigation_metadata: set[str]=set()
        self.telemetry=(self.output/'telemetry.jsonl').open('w')
        self.events=(self.output/'events.jsonl').open('w');self.csv_file=(self.output/'telemetry.csv').open('w')
        self.csv=csv.writer(self.csv_file);self.csv.writerow(['stamp_s','wall_s','x_m','y_m','z_m','state','route','vx_requested','vx_applied'])
        s=mission.site
        atomic_json(self.live/'map.json',{'origin':s.origin.tolist(),'resolution':s.resolution,
            'normal_cost':self.normal.tolist(),'trial_cost':self.trial.tolist(),
            'slope_deg':np.where(s.observed,s.slope_deg,0.).tolist(),
            'height_m':np.where(s.observed,s.height_m.astype(object),None).tolist(),
            'observed':s.observed.tolist(),
            'routes':{k:v.tolist() for k,v in mission.routes.items()},'start':mission.start.tolist(),
            'junction':mission.junction.tolist(),'summit':mission.summit.tolist()})
        atomic_json(self.output/'manifest.json',{'simulation_only':True,'pose_source':self.pose_source,
            'frame':'site','site_identity':s.identity,'physics_dt_s':.005,'control_dt_s':float(control_dt),
            'geometry_cost_source':'known simulator geometry','experience_source':'actual simulator pose samples',
            'slip':None,'sinkage':None,'hardware_commands_enabled':False,'mission_settings':mission.settings,
            'fault_injection':self.fault_injection,
            'grid_metadata':s.metadata})
        self.snapshot(0.,'initial')
        self.publish(status='preparing',reason='simulator startup')

    def snapshot(self, stamp_s: float, reason: str) -> Path | None:
        """최종 snapshot 공간을 예약하고 atomic NPZ로 현재 경험 층을 보존한다."""
        final=reason=='final'
        if not final and self.snapshot_count>=self.max_snapshots-1:return None
        name='final.npz' if final else f'{self.snapshot_count:04d}_{stamp_s:09.3f}.npz'
        path=self.output/'snapshots'/name;temporary=path.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream,stamp_s=stamp_s,origin=self.mission.site.origin,
                resolution=self.mission.site.resolution,normal_cost=self.normal,trial_cost=self.trial,
                visit_count=self.visits,last_seen_s=self.last_seen,stuck_count=self.stuck,
                frame='site',site_identity=self.mission.site.identity,reason=reason)
        temporary.replace(path);self.snapshot_count+=1
        return path

    def observe(self, row: dict, events: list[dict]) -> None:
        """유효한 monotonic 관측만 기록하고 새 event를 한 번씩 적용한다."""
        stamp=float(row['stamp_s']);position=np.asarray(row['position_m'],float)
        if not np.isfinite(stamp) or stamp<=self.last_stamp or position.shape!=(3,) or not np.isfinite(position).all():
            raise ValueError('finite monotonic telemetry required')
        encoded=json.dumps(row,allow_nan=False)
        navigation=row.get('mission',{}).get('navigation',{})
        if isinstance(navigation,dict):
            additions={f'navigation_{key}':navigation[key] for key in ('regions','portals')
                       if key not in self.navigation_metadata and isinstance(navigation.get(key),list)}
            if additions:
                map_data=json.loads((self.live/'map.json').read_text())
                map_data.update(additions);atomic_json(self.live/'map.json',map_data)
                self.navigation_metadata.update(key.removeprefix('navigation_') for key in additions)
        self.last_stamp=stamp;self.last=row;self.trajectory.append(position[:2].tolist())
        if 'estimated_position_m' in row:self.estimated_trajectory.append(row['estimated_position_m'][:2])
        s=self.mission.site;c=s.cells([position[:2]])[0];inside=0<=c[0]<s.observed.shape[1] and 0<=c[1]<s.observed.shape[0]
        if inside:self.visits[c[1],c[0]]+=1;self.last_seen[c[1],c[0]]=stamp
        self.telemetry.write(encoded+'\n')
        m=row['mission'];self.csv.writerow([stamp,row['wall_s'],*position,m['state'],m['route'],
                                           row['requested_command']['vx_m_s'],row['applied_command']['vx_m_s']])
        for event in events[self.event_cursor:]:
            self.events.write(json.dumps(event,allow_nan=False)+'\n')
            if event['type']=='fault_injected':self.fault_occurred=True
            if event['type']=='ascent_failed' and inside:self.stuck[c[1],c[0]]+=1
        self.event_cursor=len(events)
        state=(m['state'],m['route'])
        if stamp>=self.next_snapshot or state!=self.last_state:
            self.snapshot(stamp,'periodic' if state==self.last_state else 'phase')
            self.next_snapshot=stamp+1.;self.last_state=state
            self.telemetry.flush();self.events.flush();self.csv_file.flush()
        if time.monotonic()-self.last_publish>=.25:self.publish()

    def publish(self, *, status: str='running', reason: str='') -> None:
        """viewer에 전달할 최신 상태를 physics API와 독립적인 파일로 공개한다."""
        now=time.monotonic()
        if status=='running' and now-self.last_publish<.25:return
        self.last_publish=now
        yy,xx=np.nonzero(self.visits)
        atomic_json(self.live/'state.json',{'status':status,'reason':reason,'updated_wall_s':time.time(),
            'simulation_only':True,'pose_source':self.last.get('pose_source',self.pose_source),'recording':str(self.output),
            'fault_injection':self.fault_injection,'fault_occurred':self.fault_occurred,
            'snapshot_count':self.snapshot_count,'last':self.last,'trajectory':self.trajectory[-20000:],'estimated_trajectory':self.estimated_trajectory[-20000:],
            'experience':np.c_[xx,yy,self.visits[yy,xx],self.stuck[yy,xx]].tolist(),
            'visited_cells':int((self.visits>0).sum()),'stuck_events':int(self.stuck.sum())},indent=None)

    def finalize(self, status: str, reason: str) -> dict:
        """실제 결과·마지막 grid·궤적을 남기고 기록 파일을 닫는다."""
        if status not in ('complete','failed','timeout','interrupted'):raise ValueError('invalid final status')
        if self.done:return json.loads((self.output/'summary.json').read_text())
        self.done=True;self.snapshot(max(0.,self.last_stamp),'final')
        for f in (self.telemetry,self.events,self.csv_file):f.close()
        summary={'status':status,'reason':reason,'simulation_only':True,'pose_source':self.last.get('pose_source',self.pose_source),
            'fault_injection':self.fault_injection,'fault_occurred':self.fault_occurred,
            'fault_active':self.last.get('fault_active',False),
            'duration_sim_s':max(0.,self.last_stamp),'duration_wall_s':self.last.get('wall_s'),
            'mission':self.last.get('mission'),'snapshots':self.snapshot_count,'telemetry_samples':len(self.trajectory),
            'visited_cells':int((self.visits>0).sum()),'stuck_events':int(self.stuck.sum()),
            'hardware_commands_enabled':False,'slip':None,'sinkage':None}
        atomic_json(self.output/'summary.json',summary);atomic_json(self.output/'completion.json',summary)
        self.publish(status=status,reason=reason)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(9,7));s=self.mission.site
        extent=[s.origin[0],s.origin[0]+self.normal.shape[1]*s.resolution,
                s.origin[1],s.origin[1]+self.normal.shape[0]*s.resolution]
        ax.imshow(self.normal,origin='lower',extent=extent,cmap='magma',vmin=0,vmax=255)
        for key,p in self.mission.routes.items():ax.plot(p[:,0],p[:,1],label=key+' route')
        if self.trajectory:
            p=np.asarray(self.trajectory);ax.plot(p[:,0],p[:,1],'c-',label='actual simulator path (GT)')
        if self.estimated_trajectory:
            p=np.asarray(self.estimated_trajectory);ax.plot(p[:,0],p[:,1],'-',color='orange',label='ORB+ESKF estimate')
        ax.set(title=f'{status}: {reason}',xlabel='site X (m)',ylabel='site Y (m)',aspect='equal');ax.legend()
        fig.tight_layout();fig.savefig(self.output/'final.png',dpi=120);plt.close(fig)
        return summary
