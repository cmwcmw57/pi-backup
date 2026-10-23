# ================================================================
# drive.py
# 설명: simulator 정답 pose를 임무 판단과 swerve joint target에 연결한다.
# 사용법: from experiments.site_sim.drive import SimulationDrive
# ================================================================
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any
import numpy as np
from scipy.spatial.transform import Rotation
from experiments.handheld_mapping.mission_runtime import MissionRuntime
from experiments.handheld_mapping.sensor_safety import SensorSafety
from experiments.handheld_mapping.site_mission import SiteMission
from experiments.orbslam3_ros2.drive_runtime import yaw_from_wxyz
from experiments.orbslam3_ros2.swerve import body_twist_from_modules,body_twist_to_modules
from experiments.site_sim.joint_convention import from_joint,to_joint


@dataclass(frozen=True)
class SlipInjection:
    """지정한 시간 구간에 왼쪽(y>0)·오른쪽 바퀴 속도 목표에 배율을 곱해 슬립이나 좌우 속도 차를 흉내낸다."""
    left: float = 1.
    right: float = 1.
    start_s: float = 0.
    end_s: float = math.inf

    def __post_init__(self) -> None:
        for gain in (self.left,self.right):
            if not math.isfinite(gain) or not 0<=gain<=2:raise ValueError('wheel gain must be in [0, 2]')

    def gains(self, geometry: Any, sim_s: float) -> np.ndarray:
        """geometry 모듈 순서대로의 배율(구간 밖이면 모두 1)."""
        if not self.start_s<=sim_s<self.end_s:return np.ones(len(geometry.modules))
        return np.array([self.left if module.y_m>0 else self.right for module in geometry.modules],float)


class ActuationDelay:
    """목표를 [lo_s, hi_s] 균등 난수 지연 뒤에 적용한다(순서 보존). 실기 명령→적용 지연 17.6~46.5 ms(bringup 실측)를 흉내 낸다."""

    def __init__(self, lo_s: float, hi_s: float, seed: int=0) -> None:
        if not 0<=lo_s<=hi_s<=.5:raise ValueError('delay range must satisfy 0 <= lo <= hi <= 0.5 s')
        self.lo_s,self.hi_s,self.rng=float(lo_s),float(hi_s),np.random.default_rng(seed)
        self.queue=[];self.last_release=-math.inf

    def push(self, sim_s: float, steer: np.ndarray, wheel: np.ndarray) -> None:
        """이번 제어 주기의 (조향 목표, 바퀴 목표)를 지연 큐에 넣는다."""
        release=max(sim_s+self.rng.uniform(self.lo_s,self.hi_s),self.last_release)
        self.last_release=release;self.queue.append((release,np.array(steer,float),np.array(wheel,float)))

    def pop(self, sim_s: float) -> tuple | None:
        """sim_s까지 풀린 가장 최근 (조향, 바퀴)를 돌려주고 큐에서 뺀다. 없으면 None."""
        out=None
        while self.queue and self.queue[0][0]<=sim_s+1e-9:
            _,steer,wheel=self.queue.pop(0);out=(steer,wheel)
        return out

    def clear(self) -> None:
        """대기 중인 목표를 모두 버린다(정지 시)."""
        self.queue.clear()


class SimulationDrive:
    """물리 pose feedback만으로 임무를 진행하고 실제 적용 명령을 따로 보존한다."""

    def __init__(self, robot: Any, session: Any, mission: SiteMission, *, fault_injection: bool=False, hold_s: float=0.,
                 estimator: Any=None, slip: SlipInjection | None=None, rl: Any=None, delay: ActuationDelay | None=None, runtime: Any=None,
                 actuator: Any=None) -> None:
        self.slip=slip;self.robot,self.session=robot,session;self.hold_s=float(hold_s);self.estimator=estimator
        self.delay=delay;self._applied=(np.zeros(4),np.zeros(4))
        self.actuator=actuator
        self.rl=rl;self._rl_previous=None   # RlResidual이면 swerve 목표 아래에 KSRC 정책 잔차를 얹는다(기본 None)
        self._rl_previous_rotation=None
        self.runtime=runtime if runtime is not None else MissionRuntime(mission,safety=SensorSafety() if estimator is not None else None)
        if runtime is not None and estimator is None:raise ValueError("region runtime requires estimated pose")
        self.fault_injection=fault_injection
        self.fault_started=False;self.wall_start=time.monotonic()

    def _rl_state(self, sim_s: float, quaternion: np.ndarray) -> dict:
        """정책 관측용 body 축 gyro·비력과 FL,FR,RL,RR 조인트 값을 로봇 물리 상태에서 만든다(비력은 속도 차분)."""
        s=self.session;r=Rotation.from_quat([quaternion[1],quaternion[2],quaternion[3],quaternion[0]])
        velocity=np.asarray(self.robot.get_linear_velocity(),float)
        previous=self._rl_previous;self._rl_previous=(sim_s,velocity)
        acceleration=np.zeros(3) if previous is None or sim_s<=previous[0] else (velocity-previous[1])/(sim_s-previous[0])
        joints=np.asarray(self.robot.get_joint_positions(),float);rates=np.asarray(self.robot.get_joint_velocities(),float)
        steer_q,steer_dq,wheel_w=joints[s.steering_indices],rates[s.steering_indices],rates[s.wheel_indices]
        gyro=r.inv().apply(np.asarray(self.robot.get_angular_velocity(),float))
        if self.actuator is not None:
            # sim2sim sensors=real: 제어주기 회전 평균과 엔코더20ms 창을 정책에 전달한다.
            if previous is not None and sim_s>previous[0]:
                gyro=(self._rl_previous_rotation.inv()*r).as_rotvec()/(sim_s-previous[0])
            else:gyro=np.zeros(3)
            self._rl_previous_rotation=r
            wheel_w=self.actuator.read_encoder()
        converted=[from_joint(m,float(q),float(w)) for m,q,w in zip(s.geometry.modules,steer_q,wheel_w)]   # 조인트 → 펌웨어 규약
        steer_pos=np.array([c[0] for c in converted]);steer_vel=np.array([m.steering_sign*dq for m,dq in zip(s.geometry.modules,steer_dq)])
        if self.actuator is not None:
            quantum=2*math.pi/4096
            steer_pos=np.round(steer_pos/quantum)*quantum;steer_vel=np.round(steer_vel/quantum)*quantum
        return {'speed':float(np.linalg.norm(velocity)),'gyro_body':gyro,
                'force_body':r.inv().apply(acceleration+np.array([0.,0.,9.81])),
                'steer_pos':steer_pos,'steer_vel':steer_vel,
                'wheel_vel':np.array([c[1] for c in converted])}

    def stop(self) -> None:
        """네 wheel velocity target을 즉시 0으로 만든다."""
        s=self.session
        if self.delay is not None:self.delay.clear();self._applied=(np.zeros(4),np.zeros(4))
        if self.actuator is not None:
            self.actuator.stop();return
        s.controller.apply_action(s._action_type(joint_velocities=np.zeros(4),joint_indices=s.wheel_indices))

    def step(self, sim_s: float) -> dict:
        """한 simulation timestamp의 판단과 joint 적용 결과를 반환한다."""
        try:
            position,quaternion=(np.asarray(a,float) for a in self.robot.get_world_pose())
            sample={'stamp_s':sim_s,'frame':'site','site_identity':self.runtime.mission.site.identity,
                    'pose_epoch':0,'position_m':position.tolist(),'yaw_rad':yaw_from_wxyz(quaternion),
                    'tracking_valid':True}
            estimate=None
            if self.estimator is not None:
                if hasattr(self.estimator,'on_motion'):   # 측정한 조향·바퀴 조인트의 이동 방향을 추정기에 알린다(옵션이 꺼져 있으면 추정기가 버린다)
                    s=self.session;joints=np.asarray(self.robot.get_joint_positions(),float);rates=np.asarray(self.robot.get_joint_velocities(),float)
                    vx,vy,_,_=body_twist_from_modules(s.geometry,joints[s.steering_indices].tolist(),rates[s.wheel_indices].tolist())
                    self.estimator.on_motion(sim_s,vx,vy)
                # 임무 판단은 추정 위치만 본다. GT(position, quaternion)는 기록과 비교용이다.
                pose,valid,quality,status=self.estimator.pose(sim_s)
                q=Rotation.from_matrix(pose[:3,:3]).as_quat()
                sample.update(position_m=pose[:3,3].tolist(),yaw_rad=yaw_from_wxyz(np.array([q[3],q[0],q[1],q[2]])),
                              tracking_valid=bool(valid),tracking_quality=quality,filter_status=status)
                if hasattr(self.runtime,'observe_depth'):
                    roll,pitch,_=Rotation.from_matrix(pose[:3,:3]).as_euler('xyz')
                    sample.update(roll_rad=float(roll),pitch_rad=float(pitch))
                estimate={'estimated_position_m':pose[:3,3].tolist(),'estimated_yaw_rad':sample['yaw_rad'],
                          'estimator_status':status,'estimator_quality':quality,
                          'estimator_valid':bool(valid)}
            result=self.runtime.update(sample,now_s=sim_s)
            requested=result['command'];applied=requested.copy()
            if sim_s<self.hold_s:applied={'vx_m_s':0.,'yaw_rate_rad_s':0.}  # IMU 영점용 정지 구간
            fault=(self.fault_injection and self.runtime.state=='ascending' and self.runtime.route=='30'
                   and np.linalg.norm(position[:2]-self.runtime.mission.routes['30'][1])>=.18)
            if fault:
                applied={'vx_m_s':0.,'yaw_rate_rad_s':0.}
                if not self.fault_started:
                    self.runtime._event('fault_injected',sim_s,fault_type='zero_wheel_targets',position_m=position.tolist())
                    self.fault_started=True
            s=self.session
            rl_out=None
            if self.rl is None:
                modules=body_twist_to_modules(s.geometry,vx=applied['vx_m_s'],vy=applied.get('vy_m_s',0.),yaw_rate=applied['yaw_rate_rad_s'])
                steer=np.array([m.steering_angle_rad for m in modules]);wheel=np.array([m.wheel_angular_velocity_rad_s for m in modules])
            else:
                rl_state=self._rl_state(sim_s,quaternion)
                rl_state['vo']=self.estimator.policy_vo(sim_s) if self.estimator is not None and hasattr(self.estimator,'policy_vo') else None
                rl_out=self.rl.step((applied['vx_m_s'],applied.get('vy_m_s',0.),applied['yaw_rate_rad_s']),rl_state,sim_s)
                pairs=[to_joint(m,float(g),float(v),s.geometry.wheel_radius_m)   # 펌웨어 규약 목표 → 시뮬 조인트 규약(steering_sign, rolling_heading)
                       for m,g,v in zip(s.geometry.modules,rl_out['steer_goal'],rl_out['wheel_mps'])]
                steer=np.array([p[0] for p in pairs]);wheel=np.array([p[1] for p in pairs])
            if self.delay is not None:   # 목표는 지금 계산하되 지연 뒤에 조인트에 닿는다
                self.delay.push(sim_s,steer,wheel);released=self.delay.pop(sim_s)
                if released is not None:self._applied=released
                steer,wheel=self._applied
            gain=np.ones(len(steer)) if self.slip is None else self.slip.gains(s.geometry,sim_s)
            if self.actuator is None:
                s.controller.apply_action(s._action_type(joint_positions=steer,joint_indices=s.steering_indices))
                s.controller.apply_action(s._action_type(joint_velocities=wheel*gain,joint_indices=s.wheel_indices))
            else:self.actuator.command(steer,wheel*gain)
            return {'stamp_s':float(sim_s),'wall_s':time.monotonic()-self.wall_start,
                    'position_m':position.tolist(),'quaternion_wxyz':quaternion.tolist(),
                    'joint_positions':np.asarray(self.robot.get_joint_positions(),float).tolist(),
                    'joint_velocities':np.asarray(self.robot.get_joint_velocities(),float).tolist(),
                    'requested_command':requested,'applied_command':applied,'mission':result,
                    'pose_source':'orb_eskf_estimate' if estimate else 'sim_ground_truth',
                    'fault_injection':self.fault_injection,'fault_active':bool(fault),'wheel_gain':gain.tolist(),**(estimate or {}),
                    **({} if self.actuator is None else {'actuation':self.actuator.telemetry()}),
                    **({} if rl_out is None else {'rl_d_steer_rad':rl_out['d_steer'].tolist(),'rl_wheel_gain':rl_out['wheel_gain'].tolist(),
                                   'rl_action':np.round(rl_out['action'],4).tolist(),'rl_obs':np.round(rl_out['obs_frame'],5).tolist()})}
        except BaseException:
            try:self.stop()
            except Exception:pass
            raise
