# ================================================================
# mission_runtime.py
# 설명: site 좌표 pose로 대표점 방문·등판 중단·후퇴·우회·복귀와 속도 요청을 계산한다.
# 사용법: python -m experiments.handheld_mapping.mission_runtime MISSION INPUT.jsonl OUTPUT
# ================================================================
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiments.handheld_mapping.site_mission import SiteMission, dense_path
from experiments.handheld_mapping.storage import atomic_json
from experiments.handheld_mapping.sensor_safety import SensorSafety
from experiments.handheld_mapping.carrot_follower import CarrotPath, stanley_heading
from experiments.handheld_mapping.blend_follower import BlendFollower, BlendParams


def angle_wrap(value: float) -> float:
    return math.atan2(math.sin(value),math.cos(value))


class MissionRuntime:
    """유효한 site 관측에만 반응하고 실제 지나온 경로로 후퇴하는 임무 상태기계."""

    def __init__(self, mission: SiteMission, *, safety: SensorSafety | None=None) -> None:
        if not mission.validate()['accepted']:raise ValueError('valid mission required')
        self.safety=safety;self.safety_status=None
        self.mission,self.settings=mission,mission.settings
        self.state='to_junction';self.route='30';self.direction='up'
        self.visits={'entry30':'pending','entry15':'pending','entry10':'pending',
                     'summit':'pending','return':'pending'}
        self.route_status={name:{'up':'unknown','down':'unknown'} for name in ('10','15','30')}
        self.events: list[dict]=[];self.failed_routes: list[str]=[]
        self.last_stamp=None;self.last_valid_stamp=None;self.last_position=None
        self.pose_epoch=None;self.epoch_changed=False
        self.paused=True;self.started=False;self.paused_samples=0
        self.follower=None if not self.settings['follower_blend'] else BlendFollower(BlendParams(
            max_yaw_rate=self.settings['blend_max_yaw_rate_rad_s'],ramp_slope_deg=self.settings['blend_ramp_slope_deg'],
            resume_deg=self.settings['blend_ramp_resume_deg'],stop_deg=self.settings['blend_ramp_stop_deg'],
            dir_rate=self.settings['blend_dir_rate_rad_s'],heading_hold=bool(self.settings['blend_heading_hold']),
            realign_stop_deg=self.settings['blend_realign_stop_deg'],realign_resume_deg=self.settings['blend_realign_resume_deg']))
        self.command=self._zero_command()
        self.breadcrumbs: list[list]=[]
        self._set_path(mission.connector())

    def _set_path(self, points: np.ndarray) -> None:
        self.points=dense_path(points);self.index=0;self.carrot=CarrotPath(self.points)
        self.arcs=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(self.points,axis=0),axis=1))]
        self.phase_elapsed=self.effort=self.window_progress=0.

    def _zero_command(self) -> dict:
        """정지 명령. blend 모드에서만 vy_m_s를 싣는다(기본 모드의 dict는 그대로)."""
        if self.follower is None:return {'vx_m_s':0.,'yaw_rate_rad_s':0.}
        return {'vx_m_s':0.,'vy_m_s':0.,'yaw_rate_rad_s':0.}

    def _slope_deg(self, position: np.ndarray) -> float:
        """지도 칸의 경사(deg). 지도 밖이거나 NaN·비유한이면 0(게이트 비활성)."""
        position=np.asarray(position,float)
        if not np.isfinite(position).all():return 0.
        col,row=self.mission.site.cells(position)
        rows,cols=self.mission.site.slope_deg.shape
        if not (0<=col<cols and 0<=row<rows):return 0.
        value=float(self.mission.site.slope_deg[row,col])
        return value if math.isfinite(value) else 0.

    def _event(self, kind: str, stamp: float, **details) -> None:
        self.events.append({'type':kind,'stamp_s':stamp,'state':self.state,'route':self.route,**details})

    def _output(self, reason: str='active', command: dict | None=None) -> dict:
        self.command=command or self._zero_command()
        result={'state':self.state,'route':self.route,'direction':self.direction,'reason':reason,
                'visits':self.visits.copy(),'route_status':{k:v.copy() for k,v in self.route_status.items()},
                'command':self.command.copy(),'frame':'site','site_identity':self.mission.site.identity,
                'command_stamp_s':self.last_stamp,
                'valid_until_s':None if self.last_stamp is None else self.last_stamp+self.settings['pose_max_age_s'],
                'hardware_commands_enabled':False}
        if self.safety_status is not None:
            result['sensor_safety']=self.safety_status.copy()
            result['valid_until_s']=self.safety_status['valid_until_s']
        return result

    def observe_imu(self, stamp_s: float, now_s: float, valid: bool=True) -> None:
        """비동기 IMU 입력을 수용하고 watchdog에서 pose와 독립적으로 freshness를 검사한다."""
        if self.safety is not None:self.safety.observe_imu(stamp_s,now_s,valid)

    def watchdog(self, now_s: float) -> dict:
        """센서 입력과 독립적으로 주기 호출하며 끊김 시 두 속도를 0으로 만든다."""
        if self.state in ('complete','failed'):return self._output(self.state)
        if self.safety is None:return self._pause('sensor_watchdog_disabled')
        self.safety_status=self.safety.status(now_s)
        if self.safety_status['mode']=='stop':return self._pause(self.safety_status['reason'])
        return self._output('watchdog_ok',command=self.command.copy())

    def _pause(self, reason: str) -> dict:
        if self.safety is not None and reason!='recovering':
            self.safety.inhibit(reason)
            self.safety_status={'mode':'stop','reason':reason,'scale':0.,'valid_until_s':None}
        self.paused=True;self.effort=0.;self.paused_samples+=1
        if self.follower is not None:self.follower.invalidate()
        return self._output(reason)

    def update(self, sample: dict, now_s: float) -> dict:
        """stamp와 receipt가 같은 명시 clock의 관측 한 개를 처리한다."""
        if self.state in ('complete','failed'):return self._output(self.state)
        try:
            stamp=float(sample['stamp_s']);now=float(now_s)
            position=np.asarray(sample['position_m'],float);yaw=float(sample['yaw_rad'])
            if (position.shape not in ((2,),(3,)) or not np.isfinite(np.r_[position,yaw,stamp,now]).all()):
                raise ValueError('finite XY/XYZ/yaw/time required')
            position=position[:2]
        except (KeyError,ValueError,TypeError,OverflowError):return self._pause('invalid_pose')
        if sample.get('frame')!='site' or sample.get('site_identity')!=self.mission.site.identity:
            return self._pause('unaligned_frame')
        if now-stamp>self.settings['pose_max_age_s'] or now<stamp-.02:
            return self._pause('stale_pose')
        if self.last_stamp is not None and stamp<=self.last_stamp:return self._pause('nonmonotonic_pose')
        self.last_stamp=stamp
        if self.safety is not None:
            if 'imu_stamp_s' in sample:self.observe_imu(sample['imu_stamp_s'],now,sample.get('imu_valid',True))
            if 'filter_status' in sample:self.safety.observe_filter(sample['filter_status'],stamp)
            if 'slip_ratio' in sample:self.safety.observe_slip(sample['slip_ratio'],stamp)
            self.safety.observe_pose(stamp,now,sample.get('tracking_valid'),sample.get('tracking_quality','good'))
            self.safety_status=self.safety.status(now)
            if self.safety_status['mode']=='stop':return self._pause(self.safety_status['reason'])
        if sample.get('tracking_valid') is not True:return self._pause('tracking_lost')
        epoch=sample.get('pose_epoch',0)
        if type(epoch) not in (int,str):return self._pause('invalid_pose_epoch')
        if self.pose_epoch is None:self.pose_epoch=epoch
        if self.pose_epoch!=epoch:self.epoch_changed=True
        if self.epoch_changed:return self._pause('pose_epoch_changed')
        if not self.started and np.linalg.norm(position-self.mission.start[:2])>self.settings['arrival_tolerance_m']*2:
            return self._pause('initial_pose_away_from_start')
        if self.last_position is not None and np.linalg.norm(position-self.last_position)>self.settings['max_pose_jump_m']:
            return self._pause('pose_jump')
        if self.last_position is not None:
            limit=self.settings['trial_slope_deg' if self.route=='30' and self.state in ('trial_entry','ascending','retreating') else 'normal_slope_deg']
            # Observed pose may drift by the arrival tolerance that the planned-path radius already budgets.
            if not self.mission.site.validate_path([self.last_position,position],limit,self.settings['arrival_tolerance_m'])['accepted']:
                return self._pause('unsafe_observed_motion')
        dt=0. if self.paused or self.last_valid_stamp is None else stamp-self.last_valid_stamp
        if dt>self.settings['pose_max_age_s']:
            dt=0.;self.effort=0.
        self.started=True;self.paused=False;self.last_valid_stamp=stamp;self.last_position=position.copy()
        self.phase_elapsed+=dt
        if self.state in ('trial_entry','ascending'):
            if not self.breadcrumbs or np.linalg.norm(position-self.breadcrumbs[-1])>=.025:
                self.breadcrumbs.append(position.tolist())
        if len(self.breadcrumbs)>20_000:return self._fail(stamp,'breadcrumb_limit')
        entry=self.mission.routes[self.route][1]
        if self.visits['entry'+self.route]!='reached' and np.linalg.norm(position-entry)<=self.settings['arrival_tolerance_m']:
            self.visits['entry'+self.route]='reached';self._event('entry_reached',stamp)
        if np.linalg.norm(position-self.points[-1])<=self.settings['arrival_tolerance_m']:
            handoff_safe=True
            if self.state=='ascending':
                descent=self.mission.descent_path();normal=self.settings['normal_slope_deg']
                # Stay in ascent until the actual pose can enter the normal descent corridor.
                if self.mission.site.validate_path(descent,normal)['accepted']:
                    handoff_safe=self.mission.site.validate_path(np.vstack((position,descent)),normal)['accepted']
            if handoff_safe:return self._arrived(position,stamp)
        # 곡선 안쪽을 지나 점 허용 반경을 못 맞춰도 진행 점은 앞쪽 가장 가까운 점까지 나아간다(한 번에 최대 4점=0.2 m).
        window=self.points[self.index:self.index+5]
        self.index+=int(np.argmin(np.linalg.norm(window-position,axis=1)))
        while self.index<len(self.points)-1 and np.linalg.norm(position-self.points[self.index])<=self.settings['arrival_tolerance_m']:
            self.index+=1
        progress=float(self.arcs[self.index])
        if self.paused or progress-self.window_progress>=self.settings['min_progress_m']:
            self.window_progress=progress;self.effort=0.
        elif math.hypot(self.command['vx_m_s'],self.command.get('vy_m_s',0.))>.01*(1. if self.safety_status is None else self.safety_status['scale']):
            # 감속 중에는 진행이 느린 것이 정상이므로 stuck 시간과 최소 지령 기준을 모두 감속 배율만큼 낮춰 센다.
            self.effort+=dt*(1. if self.safety_status is None else self.safety_status['scale'])
        budget=(self.settings['max_attempt_s'] if self.state=='ascending' else
                max(self.settings['max_attempt_s'],self.arcs[-1]/self.settings['max_speed_m_s']*4))
        if self.effort>=self.settings['stuck_window_s'] or self.phase_elapsed>budget:
            if self.state in ('ascending','trial_entry'):return self._ascent_failed(position,stamp)
            return self._fail(stamp,'no_progress_'+self.state)
        reverse=self.state=='retreating'
        target=self.points[self.index]
        limit=self.settings['trial_slope_deg' if self.route=='30' and self.state in ('ascending','trial_entry','retreating')
                            else 'normal_slope_deg']
        carrot=self.carrot.target(self.index,self.settings['carrot_lookahead_m'])
        for end in (target,carrot):   # 바로 앞 점과 실제 조향 목표인 당근 점까지 모두 안전해야 한다.
            if not self.mission.site.validate_path([position,end],limit,self.settings['arrival_tolerance_m'])['accepted']:
                return self._pause('unsafe_local_segment')
        heading=math.atan2(carrot[1]-position[1],carrot[0]-position[0])
        if self.settings['stanley_gain']>0 and not reverse:   # Stanley식: 경로 접선 + 횡방향 오차 보정
            offset,tangent=self.carrot.cross_track(self.index,position)
            heading=stanley_heading(tangent,offset,self.settings['max_speed_m_s'],self.settings['stanley_gain'])
        if self.follower is not None:return self._blend_command(position,yaw,heading,carrot,reverse,dt)
        heading+=math.pi if reverse else 0.
        error=angle_wrap(heading-yaw)
        yaw_rate=float(np.clip(error*2.,-self.settings['max_yaw_rate_rad_s'],self.settings['max_yaw_rate_rad_s']))
        speed=self.settings['max_speed_m_s']*max(0.,math.cos(error)) if abs(error)<1. else 0.
        speed*=self.carrot.speed_factor(self.index,self.settings['carrot_brake_m'],self.settings['carrot_min_speed_factor'])
        scale=1. if self.safety_status is None else self.safety_status['scale']
        return self._output(command={'vx_m_s':float((-speed if reverse else speed)*scale),'yaw_rate_rad_s':yaw_rate*scale})

    def _blend_command(self, position: np.ndarray, yaw: float, heading: float, carrot: np.ndarray, reverse: bool, dt: float) -> dict:
        """blend 모드: 당근 방위로 병진하고 몸은 ψ_ref를 느리게 따라가는 (vx, vy, yaw_rate)를 만든다. 후진은 방위 그대로 넘겨 몸을 돌리지 않는다."""
        if np.linalg.norm(carrot-position)<1e-9:heading=yaw+(math.pi if reverse else 0.)   # atan2(0,0)으로 방향이 튀지 않게
        speed=self.settings['max_speed_m_s']*self.carrot.speed_factor(self.index,self.settings['carrot_brake_m'],self.settings['carrot_min_speed_factor'])
        out=self.follower.step(yaw=yaw,travel_heading=heading,speed=speed,slope_deg=self._slope_deg(position),dt=dt,retreat=reverse)
        scale=1. if self.safety_status is None else self.safety_status['scale']
        result=self._output(command={'vx_m_s':float(out.vx*scale),'vy_m_s':float(out.vy*scale),'yaw_rate_rad_s':float(out.yaw_rate*scale)})
        result['follower']={'psi_ref':float(out.psi_ref),'heading_gap_rad':float(out.heading_gap_rad),
                            'travel_gap_rad':float(out.travel_gap_rad),'gated':bool(out.gated),'realigning':bool(out.realigning)}
        return result

    def _arrived(self, position: np.ndarray, stamp: float) -> dict:
        if self.state=='to_junction':
            self.state='trial_entry';self.route='30';self.breadcrumbs=[position.tolist()]
            self.visits['entry30']='active';self._set_path(self.mission.routes['30'][:2])
        elif self.state=='trial_entry':
            self.visits['entry30']='reached';self.state='ascending'
            self._set_path(np.vstack((position,self.mission.routes['30'][1:])))
        elif self.state=='ascending':
            self.visits['summit']='reached';self.route_status[self.route]['up']='passed'
            self._event('summit_reached',stamp)
            if not self.mission.site.validate_path(self.mission.descent_path(),self.settings['normal_slope_deg'])['accepted']:
                self.route_status['10']['down']='failed'
                return self._fail(stamp,'descent_route_unavailable')
            self.state='descending';self.route='10';self.direction='down';self.visits['return']='active'
            self._set_path(np.vstack((position,self.mission.descent_path())))
        elif self.state=='retreating':
            remaining=[name for name in ('30','15','10') if name not in self.failed_routes]
            if not remaining:return self._fail(stamp,'all_ascent_routes_failed')
            self.route=remaining[0];self.direction='up';self.state='ascending'
            self.breadcrumbs=[position.tolist()];self.visits['entry'+self.route]='active'
            self._set_path(np.vstack((position,self.mission.path(self.route))))
            self._event('fallback_selected',stamp)
        elif self.state=='descending':
            self.route_status['10']['down']='passed'
            if np.linalg.norm(position-self.mission.start[:2])<=self.settings['arrival_tolerance_m']:
                self.visits['return']='reached';self.state='complete';self._event('return_reached',stamp)
            else:
                self.state='returning'
                self._set_path(np.vstack((position,self.mission.connector()[::-1])))
        elif self.state=='returning':
            self.visits['return']='reached';self.state='complete';self._event('return_reached',stamp)
        self._event('phase_started',stamp,direction=self.direction)
        return self._output('phase_transition')

    def _ascent_failed(self, position: np.ndarray, stamp: float) -> dict:
        self.failed_routes.append(self.route);self.route_status[self.route]['up']='failed'
        if self.visits['entry'+self.route]!='reached':self.visits['entry'+self.route]='failed'
        self._event('ascent_failed',stamp,reason='observed_no_progress_not_wheel_slip')
        points=np.asarray(self.breadcrumbs)
        if len(points)<2:points=np.vstack((position,self.mission.junction))
        self.state='retreating';self.direction='down'
        self._set_path(np.vstack((position,points[::-1])))
        self._event('retreat_started',stamp,path=self.points.tolist())
        return self._output('ascent_failed')

    def _fail(self, stamp: float, reason: str) -> dict:
        self.state='failed'
        if self.visits['summit']!='reached':self.visits['summit']='failed'
        if self.visits['return']!='reached':self.visits['return']='failed'
        self._event('mission_failed',stamp,reason=reason)
        return self._output(reason)


def run_jsonl(mission: SiteMission, input_path: Path, output: Path, *, safety: SensorSafety | None=None) -> dict:
    """명시한 receipt time의 입력을 읽고 파일로만 속도 요청을 기록한다. 송신은 없다."""
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    runtime=MissionRuntime(mission,safety=safety);last=None
    with (output/'commands.jsonl').open('w') as dest:
        try:
            with Path(input_path).open() as source:
                for line in source:
                    if not line.strip():continue
                    packet=json.loads(line)
                    if 'now_s' not in packet:raise ValueError('explicit same-clock now_s receipt required')
                    if packet.get('sensor_event')=='imu':
                        runtime.observe_imu(packet['imu_stamp_s'],packet['now_s'],packet.get('imu_valid',True))
                        last=runtime.watchdog(packet['now_s'])
                    elif packet.get('watchdog') is True:last=runtime.watchdog(packet['now_s'])
                    else:last=runtime.update(packet,packet['now_s'])
                    dest.write(json.dumps({**last,'stamp_s':packet.get('stamp_s')},allow_nan=False)+'\n')
        finally:
            dest.write(json.dumps({'state':runtime.state,'reason':'input_end_stop',
                                  'command':{'vx_m_s':0.,'yaw_rate_rad_s':0.}})+'\n')
    with (output/'events.jsonl').open('w') as stream:
        for event in runtime.events:stream.write(json.dumps(event)+'\n')
    summary={'state':runtime.state,'visits':runtime.visits,'input_exhausted':True,
             'hardware_commands_enabled':False,'last_result':last}
    atomic_json(output/'summary.json',summary)
    return summary


def main() -> None:
    parser=argparse.ArgumentParser(description='파일 기반 site pose 임무 실행; 실제 모터 송신 없음')
    parser.add_argument('mission',type=Path);parser.add_argument('input',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--sensor-safety',action='store_true')
    parser.add_argument('--require-imu',action='store_true')
    args=parser.parse_args()
    safety=SensorSafety(require_imu=args.require_imu) if args.sensor_safety or args.require_imu else None
    print(json.dumps(run_jsonl(SiteMission.load(args.mission),args.input,args.output,safety=safety),ensure_ascii=False))


if __name__=='__main__':main()
