# ================================================================
# site_mission.py
# 설명: 공동 정상에 이르는 세 경로와 대표점·우회 정책을 지도에 묶어 저장한다.
# 사용법: from ksrc_runtime.handheld_mapping.site_mission import SiteMission
# ================================================================
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ksrc_runtime.handheld_mapping.coverage_path import astar
from ksrc_runtime.handheld_mapping.site_map import SiteMap
from ksrc_runtime.handheld_mapping.storage import atomic_json

DEFAULT_SETTINGS={'normal_slope_deg':20.,'trial_slope_deg':35.,'arrival_tolerance_m':.12,
                  'max_speed_m_s':.15,'max_yaw_rate_rad_s':.6,'pose_max_age_s':.5,
                  'stuck_window_s':3.,'min_progress_m':.04,'max_attempt_s':60.,
                  'max_pose_jump_m':.5,'carrot_lookahead_m':.2,'carrot_brake_m':.3,'carrot_min_speed_factor':.35}
DEFAULT_SETTINGS.update(trial30_forward_limit_s=5.,trial30_elapsed_limit_s=10.)  # 사용자 정책:30°는 짧게 시도하고15°로 우회
DEFAULT_SETTINGS['stanley_gain']=0.   # 0이면 기존 lookahead 방향 추종, 양수면 횡방향 오차 항(Stanley식)을 더한다
DEFAULT_SETTINGS.update(follower_blend=0.,blend_heading_hold=0.,blend_max_yaw_rate_rad_s=.09,blend_ramp_slope_deg=8.,
                        blend_ramp_resume_deg=15.,blend_ramp_stop_deg=25.,blend_dir_rate_rad_s=1.,
                        blend_realign_stop_deg=90.,blend_realign_resume_deg=45.)   # 병진 방향 오차가 stop을 넘으면 정지·재정렬, resume 이하에서 출발   # follower_blend=1이면 yaw·vy 혼합 추종기(blend_follower.py)
DEFAULT_SETTINGS['summit_look']=0.   # 1이면 정상에서 30° 램프 쪽으로 돌아 잠깐 서서 관찰한 뒤 하행 방향으로 돌아선다(scenario.json의 look 필요)
DEFAULT_SETTINGS['descent_speed_scale']=1.   # 내리막(descending) 상태에서 MPPI 최대 속도·회전·옆걸음에 곱하는 비율(1이면 그대로)
ZERO_ALLOWED=('summit_look','stanley_gain','follower_blend','blend_heading_hold','trial30_forward_limit_s','trial30_elapsed_limit_s')


def dense_path(points: np.ndarray, spacing: float=.05) -> np.ndarray:
    """각 경유점을 보존하고 구간 사이만 등간격으로 나눈다."""
    points=np.asarray(points,float)
    output=[points[0]]
    for a,b in zip(points[:-1],points[1:]):
        count=max(1,int(np.ceil(np.linalg.norm(b-a)/spacing)))
        output.extend(np.linspace(a,b,count+1)[1:])
    return np.asarray(output)


def path_with_heading(points: np.ndarray) -> np.ndarray:
    """metric XY polyline에 다음 구간 heading(rad)을 추가한다."""
    points=dense_path(points)
    if len(points)==1:return np.column_stack((points,[0.]))
    delta=np.diff(points,axis=0);yaw=np.arctan2(delta[:,1],delta[:,0])
    return np.column_stack((points,np.r_[yaw,yaw[-1]]))


class SiteMission:
    """한 고정 지도에 묶인 출발부·공동 정상·방향별 대체 경로 설정."""

    def __init__(self, site: SiteMap, start: list | np.ndarray, junction: list | np.ndarray,
                 summit: list | np.ndarray, routes: dict, settings: dict | None=None,
                 site_name: str='yeouido', connector_waypoints: list | np.ndarray | None=None) -> None:
        self.site=site;self.start=np.asarray(start,float);self.junction=np.asarray(junction,float)
        self.summit=np.asarray(summit,float)
        self.routes={key:np.asarray(value,float).copy() for key,value in routes.items()}
        self.connector_waypoints=None if connector_waypoints is None else np.asarray(connector_waypoints,float).copy()
        values=settings or {}
        if set(values)-DEFAULT_SETTINGS.keys():raise ValueError('unknown mission setting')
        self.settings={**DEFAULT_SETTINGS,**values}
        if (any(type(v) not in (int,float) or not math.isfinite(v) or v<(0 if k in ZERO_ALLOWED else 1e-12) for k,v in self.settings.items())
                or not self.settings['normal_slope_deg']<=self.settings['trial_slope_deg']<90
                or self.settings['max_attempt_s']<=self.settings['stuck_window_s']
                or self.settings['max_pose_jump_m']<=self.settings['arrival_tolerance_m']
                or self.settings['max_speed_m_s']>.5 or self.settings['max_yaw_rate_rad_s']>2.
                or self.settings['follower_blend'] not in (0,1) or self.settings['blend_heading_hold'] not in (0,1)
                or self.settings['blend_max_yaw_rate_rad_s']>self.settings['max_yaw_rate_rad_s']
                or not self.settings['blend_ramp_resume_deg']<self.settings['blend_ramp_stop_deg']<90
                or not self.settings['blend_realign_resume_deg']<self.settings['blend_realign_stop_deg']<=180):
            raise ValueError('invalid mission settings')
        if site_name not in ('yeouido','sacheon','demo'):raise ValueError('invalid site name')
        self.site_name=site_name

    def path(self, route: str, direction: str='up') -> np.ndarray:
        if direction not in ('up','down'):raise ValueError('direction must be up/down')
        return self.routes[route].copy() if direction=='up' else self.routes[route][::-1].copy()

    def descent_path(self) -> np.ndarray:
        """10° 경로가 출발 waypoint를 지나면 그곳에서 하산을 끝낸다."""
        path=self.path('10','down')
        matches=np.flatnonzero(np.linalg.norm(path-self.start[:2],axis=1)<=1e-8)
        return path[:matches[0]+1] if len(matches) else path

    def connector(self) -> np.ndarray:
        """명시한 안전 점열을 보존하거나 safe grid에서 출발→갈림길을 연결한다."""
        if self.connector_waypoints is not None:
            points=self.connector_waypoints
            if (points.ndim!=2 or points.shape[1]!=2 or len(points)<2 or not np.isfinite(points).all()
                    or not np.allclose(points[0],self.start[:2],atol=1e-6,rtol=0)
                    or not np.allclose(points[-1],self.junction,atol=1e-6,rtol=0)
                    or np.any(np.linalg.norm(np.diff(points,axis=0),axis=1)<1e-5)):
                raise ValueError('invalid connector waypoints')
            if not self.site.validate_path(points,self.settings['normal_slope_deg'])['accepted']:
                raise ValueError('connector clearance failed')
            return points.copy()
        direct=np.asarray([self.start[:2],self.junction])
        slope=self.settings['normal_slope_deg']
        if self.site.validate_path(direct,slope)['accepted']:
            return direct if np.linalg.norm(direct[1]-direct[0])>1e-9 else direct[:1]
        cells=self.site.cells(direct);rows,cols=self.site.observed.shape
        if np.any(cells<0) or np.any(cells[:,0]>=cols) or np.any(cells[:,1]>=rows):
            raise ValueError('start/junction outside map')
        safe=self.site.safe_mask(slope)
        a,b=(tuple(cell[::-1]) for cell in cells)
        if not safe[a] or not safe[b]:raise ValueError('start/junction unsafe')
        path=astar(safe,a,b)
        if path is None:raise ValueError('start/junction unreachable')
        xy=self.site.origin+(np.asarray(path)[:,::-1]+.5)*self.site.resolution
        joined=np.vstack((direct[0],xy,direct[-1]))
        keep=np.r_[True,np.linalg.norm(np.diff(joined,axis=0),axis=1)>1e-9]
        joined=joined[keep]
        pulled=[joined[0]];index=0   # 격자 계단·시작점 뒤로 가는 점을 없애도록, 안전한 가장 먼 점으로 직선 단축
        while index<len(joined)-1:
            far=len(joined)-1
            while far>index+1 and not self.site.validate_path(np.asarray([joined[index],joined[far]]),slope)['accepted']:
                far-=1
            pulled.append(joined[far]);index=far
        joined=np.asarray(pulled)
        if not self.site.validate_path(joined,slope)['accepted']:
            raise ValueError('connector clearance failed')
        return joined

    def validate(self) -> dict:
        report={'accepted':False,'routes':{},'reason':''}
        if (self.start.shape!=(3,) or self.junction.shape!=(2,) or self.summit.shape!=(2,)
                or not np.isfinite(np.r_[self.start,self.junction,self.summit]).all()
                or set(self.routes)!= {'10','15','30'}
                or np.linalg.norm(self.summit-self.junction)<self.settings['arrival_tolerance_m']*2):
            return {**report,'reason':'invalid_start_junction_summit_or_routes'}
        for name,points in self.routes.items():
            if (points.ndim!=2 or points.shape[1]!=2 or len(points)<(3 if name=='30' else 2)
                    or not np.isfinite(points).all()
                    or not np.allclose(points[0],self.junction,atol=1e-6,rtol=0)
                    or not np.allclose(points[-1],self.summit,atol=1e-6,rtol=0)
                    or np.any(np.linalg.norm(np.diff(points,axis=0),axis=1)<1e-5)):
                report['routes'][name]={'accepted':False,'reason':'invalid_route_geometry'}
                continue
            limit=self.settings['trial_slope_deg' if name=='30' else 'normal_slope_deg']
            report['routes'][name]=self.site.validate_path(points,limit)
        try:
            self.connector();report['connector']={'accepted':True}
        except ValueError as error:
            report['connector']={'accepted':False,'reason':str(error)}
        report['accepted']=all(row['accepted'] for row in report['routes'].values()) and report['connector']['accepted']
        report['reason']='valid' if report['accepted'] else 'route_or_connector_rejected'
        return report

    def save(self, folder: Path) -> None:
        report=self.validate()
        if not report['accepted']:raise ValueError('invalid mission: '+json.dumps(report))
        folder=Path(folder).expanduser().resolve();folder.mkdir(parents=True,exist_ok=False)
        self.site.save(folder/'site')
        routes_dir=folder/'routes';routes_dir.mkdir()
        for name in ('10','15','30'):
            for direction in ('up','down'):
                np.savetxt(routes_dir/f'{name}_{direction}.csv',path_with_heading(self.path(name,direction)),
                           delimiter=',',header='x_m,y_m,yaw_rad',comments='')
        atomic_json(folder/'mission.json',{'schema_version':1,'site_name':self.site_name,
            'frame':self.site.frame,'units':'m','site_identity':self.site.identity,
            'start':self.start.tolist(),'junction':self.junction.tolist(),'summit':self.summit.tolist(),
            **({} if self.connector_waypoints is None else {'connector_waypoints':self.connector_waypoints.tolist()}),
            'routes':{key:value.tolist() for key,value in self.routes.items()},'settings':self.settings,
            'policy':{'ascent_order':['30','15','10'],'descent_route':'10',
                      'representative_visit_not_coverage':True,'trial_entry':self.routes['30'][1].tolist()},
            'validation':report,'hardware_commands_enabled':False})

    @classmethod
    def load(cls, folder: Path, *, load_mesh: bool=True) -> SiteMission:
        folder=Path(folder).expanduser().resolve();data=json.loads((folder/'mission.json').read_text())
        if data.get('schema_version')!=1 or data.get('frame')!='site' or data.get('units')!='m':
            raise ValueError('invalid mission schema/frame/units')
        site=SiteMap.load(folder/'site',load_mesh=load_mesh)
        if data['site_identity']!=site.identity:raise ValueError('mission/site hash mismatch')
        self=cls(site,data['start'],data['junction'],data['summit'],data['routes'],data['settings'],data['site_name'],
                 connector_waypoints=data.get('connector_waypoints'))
        if not self.validate()['accepted']:raise ValueError('saved mission route validation failed')
        return self
