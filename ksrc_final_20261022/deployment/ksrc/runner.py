# ================================================================
# runner.py
# 설명: 최종 f판 ESKF·RegionMission을 실시간 센서·gateway에 연결한다.
#       SPI는 열지 않으며 ORB 복구 때 마지막 site pose와 임무 단계를 보존한다.
# 사용법: python runner.py --config real.json --state-dir state
# ================================================================
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import base64
from collections import deque
import json
import math
from pathlib import Path
import select
import signal
import socket
import struct
import sys
import time

import numpy as np
import cv2
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'runtime'))
from ksrc_runtime.handheld_mapping.region_mission import RegionMission
from ksrc_runtime.handheld_mapping.sensor_safety import SensorSafety
from ksrc_runtime.handheld_mapping.site_mission import SiteMission
from ksrc_runtime.handheld_mapping.control_gateway import pack_cmd
from ksrc_runtime.localization.online import OnlineEstimator


def pack_vo(seq: int, stamp_ns: int, vx: float, vy: float, valid: bool) -> bytes:
    """기존 run_policy --vo-port와 같은 KVO1 패킷을 만든다."""
    return struct.pack('<4sIqffI', b'KVO1', seq & 0xFFFFFFFF, stamp_ns, vx, vy, int(valid))


ORB_LOST_RESTART_S = 10.   # 복구 중 이 시간 동안 ORB 프레임이 없을 때만 ORB 재시작을 요청한다(ORB가 스스로 복구할 시간)
ORB_STARTUP_GRACE_S = 30.  # 재시작한 ORB는 어휘 로드·초기화로 첫 pose까지 8~10 s 걸린다(Pi 실측). 그동안 다시 죽이지 않는다


def orb_restart_due(recovering: bool, now: float, next_restart: float, t: float, recovery_since: float, last_frame: float) -> bool:
    """복구 중이고 ORB 프레임이 ORB_LOST_RESTART_S 넘게 없으며 유예가 끝났을 때만 ORB 재시작을 요청한다."""
    return (recovering and now >= next_restart and t - recovery_since > ORB_LOST_RESTART_S
            and t - last_frame > ORB_LOST_RESTART_S)


GATEWAY_STATUS_KEYS = ('orb_ok', 'in_region1', 'done', 'start_aligned', 'active', 'recovering', 'run', 'stage', 'anchor_count')


def gateway_payload(status: dict) -> bytes:
    """gateway 수신 제한(256 B) 안에 들어가도록 조이스틱 화면에 필요한 필드만 압축해 만든다."""
    message = {key: status[key] for key in GATEWAY_STATUS_KEYS}
    message.update(stationary_s=round(status['stationary_s'], 1), reason=str(status['reason'])[:30])
    return json.dumps(message, separators=(',', ':')).encode()


def matrix(value: object) -> np.ndarray:
    """잘못된 pose가 필터를 오염시키기 전에 거부한다."""
    out = np.asarray(value, float)
    if (out.shape != (4, 4) or not np.isfinite(out).all()
            or not np.allclose(out[3], [0, 0, 0, 1])
            or not np.allclose(out[:3, :3].T @ out[:3, :3], np.eye(3), atol=1e-4)
            or np.linalg.det(out[:3, :3]) < .99):
        raise ValueError('finite SE(3) pose required')
    return out


class Navigation:
    """같은 monotonic clock의 센서로 준비·임무·ORB 복구를 관리한다."""

    def __init__(self, mission_dir: Path, config: dict, plan_cpu: str | None = None) -> None:
        self.mission_dir, self.config, self.plan_cpu = Path(mission_dir), config, plan_cpu
        self.mission = SiteMission.load(self.mission_dir, load_mesh=False)
        self.scenario = json.loads((self.mission_dir / 'scenario.json').read_text())
        regions = json.loads((self.mission_dir / self.scenario['regions_file']).read_text())
        self.scenario.update(regions=regions['regions'], portals=regions['portals'])
        self.prior = np.eye(4)
        self.prior[:3, :3] = Rotation.from_euler('z', self.mission.start[2]).as_matrix()
        self.prior[:3, 3] = [*self.mission.start[:2], float(config['start_z_m'])]
        self.checkpoint = self.prior.copy()
        self.estimator = self.new_estimator(self.prior, 0.)
        self.runtime = self.new_runtime(1)
        self.active = self.recovering = False
        self.reason, self.run_id = 'waiting', 0
        self.anchor_count = 0
        self.last_imu = self.last_frame = self.last_accept = -math.inf
        self.last_tick, self.last_map = -math.inf, None
        self.stationary_since = self.good_since = None
        self.recovery_since = None
        self.frame_count, self.recovery_count = 0, 0
        self.wheel_speed, self.gyro_norm = math.inf, math.inf
        self.last_vo = None
        self.last_frame_matrix = None
        self.events: list[dict] = []
        self.async_plan = None
        self.wall_yaw = self.wall_track = None
        if config.get('wall'):
            from ksrc_runtime.site_sim.wall_yaw import WallYaw
            from ksrc_runtime.site_sim.wall_track import WallTrack
            self.wall_yaw = WallYaw(config['wall'], site=self.mission.site, sigma_deg=4.)
            self.wall_track = WallTrack(config['wall'], self.mission.site)
        self.wall_hist = {'lateral_delta': deque(maxlen=3), 'front_delta': deque(maxlen=3), 'corner': deque(maxlen=3)}

    def new_estimator(self, pose: np.ndarray, stamp: float) -> OnlineEstimator:
        """IMU holdback을 유지하고 camera capture 시각까지 진행해 관측을 적용한다."""
        est = OnlineEstimator(pose, self.config['calibration'], lag_s=.5,
                              adaptive_visual_noise=True, absolute_attitude_sigma_deg=5.,
                              gyro_bias_init_hold_s=None, gyro_bias_sigma_dps=.003)
        est.filter_time = stamp
        est.last_accept, est.update_time = stamp, stamp
        est.filter_gyro = np.zeros(3)
        return est

    def new_runtime(self, run: int) -> RegionMission:
        """f판 규칙을 읽고 2회차에만 30° 생략·하행 0.7배를 적용한다."""
        mission = SiteMission.load(self.mission_dir, load_mesh=False)
        mission.settings['max_speed_m_s'] = .15
        mission.settings['descent_speed_scale'] = .7 if run == 2 else 1.
        if run == 2:
            mission.settings['summit_look'] = 0
        runtime = RegionMission(mission, safety=SensorSafety(require_imu=True), scenario=self.scenario)
        runtime.planner.max_lateral = 0.
        runtime.planner.clearance_weight, runtime.planner.clearance_margin = 1.5, .15
        runtime.corner_gate = True
        if run == 2:
            runtime.route = '15'
            runtime.visits['entry15'] = 'active'
            runtime._phase('entry15', 'to_entry15', 0.)
        return runtime

    def imu(self, stamp: float, dt: float, gyro: object, force: object, wheel_speed: float) -> bool:
        """SI·sensor 축 IMU를 받으며 중복·역순을 freshness에 세지 않는다."""
        if np.asarray(gyro).shape != (3,) or np.asarray(force).shape != (3,):
            return False
        values = np.r_[stamp, dt, gyro, force, wheel_speed]
        if not np.isfinite(values).all() or stamp <= self.last_imu or not 0 < dt <= .1:
            return False
        if stamp - self.last_imu > .1:
            self.stationary_since = None
        self.last_imu = float(stamp)
        self.wheel_speed, self.gyro_norm = abs(float(wheel_speed)), float(np.linalg.norm(gyro))
        stationary = self.wheel_speed < .01 and self.gyro_norm < .05 and abs(np.linalg.norm(force) - 9.81) < .8
        self.stationary_since = (stamp if self.stationary_since is None else self.stationary_since) if stationary else None
        self.estimator.on_imu(stamp, dt, np.asarray(gyro), np.asarray(force))
        self.runtime.observe_imu(stamp, stamp, True)
        return True

    def fail_tracking(self, stamp: float, reason: str) -> None:
        """정지 후 pose·단계를 보존하며 새 ORB 기준이 안정될 때까지 기다린다."""
        if not self.recovering:
            self.recovering, self.recovery_since = True, stamp
            self.events.append(dict(type='recovery_start', stamp_s=stamp, reason=reason,
                                    checkpoint=self.checkpoint.tolist(), stage=self.runtime.stage))
        self.reason, self.good_since, self.frame_count = reason, None, 0
        self.last_vo = None
        for history in self.wall_hist.values():
            history.clear()
        if self.wall_yaw:
            self.wall_yaw.candidate = None

    def frame(self, stamp: float, camera_pose: object, map_id: str, state: int) -> bool:
        """capture 시각의 camera pose를 받으며 새 map 원점을 보존한 site pose에 연결한다."""
        if not math.isfinite(stamp) or stamp <= self.last_frame:
            return False
        if state != 2:
            self.fail_tracking(stamp, 'tracking_lost')
            return False
        pose = matrix(camera_pose)
        if self.last_map is not None and map_id != self.last_map:
            self.fail_tracking(stamp, 'map_changed')
        # 이미 적분한 filter 시각보다 오래된 관측을 현재 상태에 밀어 넣지 않는다.
        if stamp < self.estimator.filter_time - .025:
            return False
        self.last_frame = stamp
        if self.recovering and self.good_since is None:
            if self.stationary_since is None or stamp - self.stationary_since < .5:
                return False
            self.estimator = self.new_estimator(self.checkpoint, stamp)
            self.runtime.safety = SensorSafety(require_imu=True)
            self.good_since = stamp
            self.last_frame_matrix = None
        est = self.estimator
        while est.pending and est.pending[0][0] <= stamp + 1e-9:
            end, dt, gyro, force = est.pending.popleft()
            if end <= est.filter_time:
                continue
            est.filter.propagate(gyro, force, min(dt, end - est.filter_time))
            est.filter_gyro, est.filter_time = gyro, end
        # on_frame 내부 stamp-lag가 실제 capture 시각이 된다.
        info = est.on_frame(stamp + est.lag_s, 2, '0', pose)
        if info['accepted'] or info.get('reason') == 'reanchored':
            self.last_accept = stamp
            self.frame_count += 1
            if self.good_since is None:
                self.good_since = stamp
            base = pose @ est.cam_from_base
            previous = self.last_frame_matrix
            if previous and 0 < stamp - previous[0] <= .3:
                velocity = base[:3, :3].T @ (base[:3, 3] - previous[1][:3, 3]) / (stamp - previous[0])
                if np.linalg.norm(velocity[:2]) <= 1.:
                    self.last_vo = (stamp, float(velocity[0]), float(velocity[1]))
            self.last_frame_matrix = (stamp, base)
        self.last_map = map_id
        return True

    def status(self, now: float) -> dict:
        """gateway START 조건과 감시 상태를 반환한다. 신규 START와 AUTO 복구를 구분한다."""
        fresh = now - self.last_frame <= .35 and now - self.last_accept <= .5 and now - self.last_imu <= .1
        stationary_s = 0. if self.stationary_since is None or now - self.last_imu > .1 else max(0., now - self.stationary_since)
        inside = self.runtime.graph.distance(1, self.checkpoint[:2, 3]) <= 0.
        yaw = math.atan2(self.checkpoint[1, 0], self.checkpoint[0, 0])
        heading_error = abs((yaw - self.mission.start[2] + math.pi) % (2 * math.pi) - math.pi)
        aligned = np.linalg.norm(self.checkpoint[:2, 3] - self.prior[:2, 3]) <= .08 and heading_error <= math.radians(8.)
        return dict(orb_ok=bool(fresh and not self.recovering), stationary_s=stationary_s,
                    in_region1=bool(inside and aligned), start_aligned=bool(aligned), active=self.active, recovering=self.recovering,
                    run=self.run_id, stage=self.runtime.stage, reason=self.reason, anchor_count=self.anchor_count,
                    done=self.runtime.state in ('complete', 'failed'), recovery_count=self.recovery_count,
                    pose_age_s=None if not math.isfinite(self.last_frame) else max(0., now - self.last_frame),
                    imu_age_s=None if not math.isfinite(self.last_imu) else max(0., now - self.last_imu))

    def start(self, run: int, now: float) -> bool:
        """명시 START는 정지·정상 추적·출발 구역을 요구한다."""
        status = self.status(now)
        if self.active or run not in (1, 2) or not status['orb_ok'] or not status['in_region1'] or status['stationary_s'] < 3.:
            return False
        if self.async_plan:
            self.async_plan.close()
        self.runtime = self.new_runtime(run)
        self.runtime.last_position = self.checkpoint[:2, 3].copy()
        self.active, self.run_id, self.reason = True, run, 'active'
        if self.plan_cpu is not None:
            from planner import AsyncPlanner
            self.async_plan = AsyncPlanner(self.runtime, self.plan_cpu)
        return True

    def image(self, stamp: float, gray: np.ndarray) -> None:
        """f판 벽 yaw·계획 offset·코너 앞벽 관측. depth는 사용하지 않는다."""
        if self.wall_track is None or self.recovering or self.last_tick - stamp > .3:
            return
        pose = self.estimator.filter.pose @ self.estimator.imu_from_base
        roll, pitch, yaw = Rotation.from_matrix(pose[:3, :3]).as_euler('xyz')
        rate = float((self.estimator.base_from_imu[:3, :3] @ (self.estimator.filter_gyro - self.estimator.filter.gyro_bias))[2])
        wy = self.wall_yaw.measure(gray, stamp, yaw, roll, pitch, rate, pose[:2, 3])
        if wy['valid']:
            self.estimator.on_wall_yaw(wy['yaw'], self.wall_yaw.sigma)
        runtime = self.runtime
        if not self.active or not runtime.stage.startswith('return') or runtime.state != 'descending':
            for history in self.wall_hist.values():
                history.clear()
            return
        m = self.wall_track.measure(gray, pose[:2, 3], yaw, roll, pitch, rate)
        if runtime.stage in runtime.CORNER_GATE:
            cf = self.wall_track.front_only(gray, yaw, roll, pitch, rate)
            history = self.wall_hist['corner']
            if cf['d'] is None:
                history.clear()
            else:
                history.append(cf['d'])
            runtime.front_obs = dict(stamp=stamp, d=float(np.median(history))) if len(history) == 3 and max(history) - min(history) <= .04 else None
        if m['axis'] is not None:
            heading = np.array([math.cos(m['axis']), math.sin(m['axis'])])
            offset = runtime.plan_offset.copy()
            for key, spread, axis in (('lateral_delta', .04, np.array([-heading[1], heading[0]])), ('front_delta', .03, heading)):
                history = self.wall_hist[key]
                if m[key] is not None:
                    history.append(m[key])
                elif m['reason']:
                    history.clear()
                if len(history) == 3 and max(history) - min(history) <= spread:
                    mismatch = float(np.median(history)) - float(offset @ axis)
                    trust = math.exp(-(mismatch / .05) ** 2)
                    offset += float(np.clip((1. - trust) * mismatch, -.12, .12)) * axis
            norm = float(np.linalg.norm(offset))
            runtime.plan_offset = offset * min(1., .4 / max(norm, 1e-9))

    def stop(self, reason: str = 'operator') -> None:
        """STOP 이후 ORB가 복구돼도 AUTO를 재개하지 않는다."""
        self.active, self.reason = False, reason

    def anchor(self, now: float) -> bool:
        """사용자가 실제 출발 pose에 정렬한 뒤 정지 상태에서 preset을 다시 기준으로 삼는다."""
        status = self.status(now)
        if self.active or not status['orb_ok'] or status['stationary_s'] < 3.:
            return False
        self.checkpoint = self.prior.copy()
        self.estimator = self.new_estimator(self.prior, now)
        self.last_frame_matrix = self.last_vo = None
        self.reason = 'anchored_at_preset'
        self.anchor_count += 1
        self.events.append(dict(type='operator_anchor', stamp_s=now, pose=self.prior.tolist()))
        return True

    def orb_reset(self, now: float) -> bool:
        """운영자의 ORB 재시작 요청. AUTO 중에는 거절하고, 대기 중이면 pose를 보존한 채 복구 절차로 들어간다."""
        if self.active:
            return False
        self.fail_tracking(now, 'operator_orb_reset')
        return True

    def tick(self, now: float) -> dict:
        """50 Hz 제어에서 센서가 무효하거나 복구 중이면 0 명령을 출력한다."""
        zero = dict(command=(0., 0., 0.))
        if now <= self.last_tick:
            return zero
        self.last_tick = now
        if self.last_map is not None and now - self.last_frame > .35 and not self.recovering:
            self.fail_tracking(now, 'pose_gap')
        if self.active and now - self.last_imu > .1 and not self.recovering:
            self.fail_tracking(now, 'imu_gap')
        if self.recovering:
            stable = self.good_since is not None and now - self.good_since >= 1. and self.frame_count >= 5
            fresh = now - self.last_frame <= .2 and now - self.last_accept <= .3 and now - self.last_imu <= .1
            if not fresh:
                self.good_since, self.frame_count = None, 0
                self.last_frame_matrix = self.last_vo = None
            if not (stable and fresh):
                return zero
            self.recovering = False
            self.recovery_count += 1
            gap = now - self.recovery_since
            self.runtime.phase_start_s += gap
            for name in ('progress_stamp', 'blocked_since', 'corner_wait'):
                value = getattr(self.runtime, name)
                if value is not None:
                    setattr(self.runtime, name, value + gap)
            self.runtime.progress.clear()
            self.runtime.last_position = self.checkpoint[:2, 3].copy()
            self.runtime.last_stamp = None
            self.runtime.paused = True
            self.runtime.last_plan, self.runtime.progress_anchor = {}, None
            if self.async_plan:
                self.async_plan.invalidate()
            self.reason = 'recovered'
            self.events.append(dict(type='recovery_done', stamp_s=now, stage=self.runtime.stage,
                                    gap_s=now - self.recovery_since, checkpoint=self.checkpoint.tolist()))
        if now - self.last_imu > .1 or now - self.last_accept > .5 or now - self.last_frame > .35:
            return zero
        pose, valid, quality, filter_status = self.estimator.pose(now)
        if not valid or not np.isfinite(pose).all():
            return zero
        if np.linalg.norm(pose[:2, 3] - self.checkpoint[:2, 3]) > .5:
            self.fail_tracking(now, 'pose_jump')
            return zero
        self.checkpoint = pose.copy()
        if not self.active:
            return zero
        roll, pitch, yaw = Rotation.from_matrix(pose[:3, :3]).as_euler('xyz')
        self.runtime.observe_imu(self.last_imu, now, True)
        sample = dict(stamp_s=now, frame='site', site_identity=self.runtime.mission.site.identity,
                      position_m=pose[:3, 3].tolist(), yaw_rad=float(yaw), roll_rad=float(roll), pitch_rad=float(pitch),
                      tracking_valid=True, tracking_quality=quality, filter_status=filter_status, pose_epoch=0)
        if self.async_plan:
            self.async_plan.apply(now)
        out = self.runtime.update(sample, now_s=now)
        self.reason = out['reason']
        if self.runtime.state in ('complete', 'failed'):
            self.active = False
        cmd = out['command']
        command = (float(cmd['vx_m_s']), float(cmd.get('vy_m_s', 0.)), float(cmd['yaw_rate_rad_s']))
        if not np.isfinite(command).all():
            self.stop('invalid_command')
            return zero
        return dict(command=command, pose=pose.tolist(), mission=out)

    def close(self) -> None:
        """MPPI worker를 유한 시간 안에 종료한다."""
        if self.async_plan:
            self.async_plan.close()


def main() -> None:
    parser = argparse.ArgumentParser(description='KSRC f navigation; SPI를 열지 않는다')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--mission', type=Path, default=HERE / 'mission')
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--sensor-port', type=int, default=5012)
    parser.add_argument('--control-port', type=int, default=5010)
    parser.add_argument('--auto-port', type=int, default=5007)
    parser.add_argument('--status-port', type=int, default=5011)
    parser.add_argument('--vo-port', type=int, default=5600)
    parser.add_argument('--plan-cpu', default=None, help='지정하면 MPPI를 별도 프로세스로 계산한다')
    args = parser.parse_args()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    nav = Navigation(args.mission, json.loads(args.config.read_text()), args.plan_cpu)
    sensors, control, out = (socket.socket(socket.AF_INET, socket.SOCK_DGRAM) for _ in range(3))
    sensors.bind(('127.0.0.1', args.sensor_port))
    control.bind(('127.0.0.1', args.control_port))
    sensors.setblocking(False)
    control.setblocking(False)
    origin, epoch_ns = time.monotonic(), time.time_ns()
    stopped = False

    def stop_signal(signum: int, frame: object) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    seq, next_tick, next_status, next_restart, logged = 0, origin, origin, origin + ORB_STARTUP_GRACE_S, 0
    with (args.state_dir / 'navigation.jsonl').open('a', buffering=1) as log:
        try:
            while not stopped:
                now = time.monotonic()
                ready, _, _ = select.select([sensors, control], [], [], max(0., min(.02, next_tick - now)))
                for source in ready:
                    for _ in range(128):
                        try:
                            data = json.loads(source.recv(65536))
                        except BlockingIOError:
                            break
                        except (ValueError, UnicodeError):
                            continue
                        if not isinstance(data, dict):
                            continue
                        try:
                            now = time.monotonic() - origin
                            if source is control:
                                if data.get('op') == 'START':
                                    if not nav.start(int(data['run']), now):
                                        nav.events.append(dict(type='start_rejected', stamp_s=now))
                                        out.sendto(b'{"done":true}', ('127.0.0.1', args.status_port))
                                elif data.get('op') == 'STOP':
                                    nav.stop()
                                elif data.get('op') == 'ANCHOR':
                                    nav.anchor(now)
                                elif data.get('op') == 'ORB_RESET':
                                    if nav.orb_reset(now):
                                        (args.state_dir / 'restart_orb').touch()   # ORB 프로세스는 supervisor가 재시작한다
                                        next_restart = time.monotonic() + ORB_STARTUP_GRACE_S   # 새 ORB가 올라올 때까지 자동 재시작 보류
                                continue
                            stamp = float(data['monotonic_s']) - origin
                            if stamp > now + .02 or now - stamp > .5 or stamp < 0:
                                continue
                            if data['kind'] == 'imu':
                                nav.imu(stamp, float(data['dt_s']), data['gyro'], data['force'], float(data['wheel_speed']))
                            elif data['kind'] == 'frame':
                                nav.frame(stamp, data['camera_pose'], str(data['map_id']), int(data['state']))
                            elif data['kind'] == 'tracking' and int(data['state']) != 2:
                                nav.fail_tracking(now, 'tracking_lost')
                            elif data['kind'] == 'image':
                                gray = cv2.imdecode(np.frombuffer(base64.b64decode(data['jpg']), np.uint8), cv2.IMREAD_GRAYSCALE)
                                if gray is not None and gray.shape == (480, 640):
                                    nav.image(stamp, gray)
                        except (ValueError, KeyError, TypeError, OverflowError):
                            nav.events.append(dict(type='invalid_input', stamp_s=now))
                now = time.monotonic()
                if now < next_tick:
                    continue
                t = now - origin
                result = nav.tick(t)
                seq += 1
                out.sendto(pack_cmd(seq, *result['command']), ('127.0.0.1', args.auto_port))
                vo = nav.last_vo
                valid_vo = vo is not None and not nav.recovering and t - vo[0] <= .35
                out.sendto(pack_vo(seq, epoch_ns + int((vo[0] if valid_vo else t) * 1e9),
                                   vo[1] if valid_vo else 0., vo[2] if valid_vo else 0., valid_vo), ('127.0.0.1', args.vo_port))
                if now >= next_status:
                    status = nav.status(t)
                    out.sendto(gateway_payload(status), ('127.0.0.1', args.status_port))
                    temporary = args.state_dir / 'status.tmp'
                    temporary.write_text(json.dumps(dict(status, stamp_s=t, command=result['command'], pose=nav.checkpoint.tolist())))
                    temporary.replace(args.state_dir / 'status.json')
                    log.write(json.dumps(dict(type='tick', stamp_s=t, command=result['command'], pose=nav.checkpoint.tolist(), **status)) + '\n')
                    next_status = now + .2
                if orb_restart_due(nav.recovering, now, next_restart, t, nav.recovery_since, nav.last_frame):
                    (args.state_dir / 'restart_orb').touch()
                    next_restart = now + ORB_STARTUP_GRACE_S
                for event in nav.events[logged:]:
                    log.write(json.dumps(event) + '\n')
                logged = len(nav.events)
                next_tick += .02
                if next_tick < now - .02:
                    next_tick = now + .02
        finally:
            for _ in range(3):
                seq += 1
                out.sendto(pack_cmd(seq, 0., 0., 0.), ('127.0.0.1', args.auto_port))
                out.sendto(pack_vo(seq, time.time_ns(), 0., 0., False), ('127.0.0.1', args.vo_port))
            nav.close()
            sensors.close()
            control.close()
            out.close()


if __name__ == '__main__':
    main()
