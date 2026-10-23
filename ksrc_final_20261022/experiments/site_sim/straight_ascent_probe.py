# ================================================================
# straight_ascent_probe.py
# 설명: 시뮬레이션 15° 경사에서 RL을 유지하고 고정 직진 입력과 joint 응답을 기록한다.
# 사용법:
#   python -m experiments.site_sim.straight_ascent_probe [site_sim.run 인자]
#   --navigation region-mppi --mppi-max-speed 0.10 --start-noise-xy-m 0 --start-noise-yaw-deg 0
# ================================================================
from __future__ import annotations

import math

import numpy as np

from experiments.handheld_mapping.region_mission import RegionMission
from experiments.site_sim.drive import SimulationDrive


class StraightAscentProbe(RegionMission):
    """고정 body 직진 명령만 내고 센서·자세·통로 한계와 진단 종료를 검사한다."""

    def __init__(self, mission, **kwargs) -> None:
        super().__init__(mission, planner=object(), **kwargs)
        self.route, self.state, self.stage = '15', 'ascending', 'probe15'
        self.events = []

    def _output(self, reason: str, navigation: dict | None = None) -> dict:
        out = super()._output(reason, navigation)
        out['navigation'].update(mode='fixed-straight-ascent', diagnostic=True,
                                 fixed_speed_m_s=self.settings['max_speed_m_s'])
        return out

    def update(self, sample: dict, now_s: float) -> dict:
        self.last_stamp = float(sample['stamp_s'])
        xy = np.asarray(sample['position_m'], float)[:2]
        pitch, roll = float(sample.get('pitch_rad', 0.)), float(sample.get('roll_rad', 0.))
        if not np.isfinite(np.r_[xy, pitch, roll, self.last_stamp, now_s]).all():
            return self._finish('failed', 'probe_invalid_pose')
        if now_s-self.last_stamp > .5 or self.last_stamp > now_s+.02 or sample.get('tracking_valid') is not True:
            return self._stop('probe_tracking_stop')
        if self.safety is not None:
            self.safety.observe_pose(self.last_stamp, now_s, True, sample.get('tracking_quality', 'good'))
            self.safety.observe_filter(sample.get('filter_status', 'ok'), self.last_stamp)
            self.sensor_status = self.safety.status(now_s)
            if self.sensor_status['mode'] == 'stop':return self._stop(self.sensor_status['reason'])
        if self.last_stamp < self.hold_s:return self._stop('start_hold')
        if abs(roll) > math.radians(30.) or abs(pitch) > math.radians(40.):
            return self._finish('failed', 'probe_tilt_limit')
        if xy[0] < 3.4 or not 1.04 <= xy[1] <= 1.36:
            return self._finish('failed', 'probe_corridor_limit')
        if xy[0] >= 6.03:
            self.route_status['15']['up'] = 'succeeded'
            return self._finish('complete', 'probe_ascent_reached')
        if self.last_stamp-self.hold_s >= self.settings.get('probe_duration_s',60.):
            return self._finish('complete', 'probe_duration_elapsed')
        self.command = dict(vx_m_s=float(self.settings['max_speed_m_s']), yaw_rate_rad_s=0.)
        return self._output('probe_fixed_twist')

    def _finish(self, state: str, reason: str) -> dict:
        self.state = state
        self._event('probe_finished', self.last_stamp, reason=reason)
        return self._stop(reason)


class InstrumentedProbeDrive(SimulationDrive):
    """기존 구동 결과에 지연 뒤 joint 목표와 실제 모듈 측정값을 추가한다."""

    def step(self, sim_s: float) -> dict:
        row = super().step(sim_s)
        s = self.session
        q, dq = np.asarray(row['joint_positions']), np.asarray(row['joint_velocities'])
        row['diagnostic_actuation'] = dict(
            steering_joint_rad=q[s.steering_indices].tolist(),
            steering_joint_rad_s=dq[s.steering_indices].tolist(),
            wheel_joint_rad_s=dq[s.wheel_indices].tolist(),
            steering_indices=list(map(int, s.steering_indices)), wheel_indices=list(map(int, s.wheel_indices)),
            steering_target_joint_rad=None if self.delay is None else self._applied[0].tolist(),
            wheel_target_joint_rad_s=None if self.delay is None else self._applied[1].tolist())
        return row


def main() -> None:
    """독립 실행 프로세스 안에서만 구동·임무 factory를 진단 구현으로 연결한다."""
    from experiments.handheld_mapping import region_mission
    from experiments.site_sim import drive, run
    load_original = run.load_mission

    def load_probe(args):
        mission = load_original(args)
        if args.navigation != 'region-mppi' or not args.rl_policy:
            raise ValueError('probe requires region-mppi and the existing RL policy')
        if args.start_noise_xy_m != 0 or args.start_noise_yaw_deg != 0:
            raise ValueError('probe requires zero start perturbation')
        mission.start = np.array([3.75, 1.2, 0.])
        if args.soil_model == 'ksrc-legacy':mission.settings['probe_duration_s']=240.
        return mission

    run.load_mission = load_probe
    region_mission.RegionMission = StraightAscentProbe
    drive.SimulationDrive = InstrumentedProbeDrive
    run.main()


if __name__ == '__main__':
    main()
