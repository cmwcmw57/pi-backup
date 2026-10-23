# ================================================================
# fusion.py
# 설명: 독립 ORB 영상 상대 이동·IMU·(선택) 바퀴 엔코더 body 속도를 error-state filter로 결합한다. 벽 보정은 포함하지 않는다.
# 사용법:
#   python -m experiments.localization.fusion --sequence sequence --orb orb --initial-prior prior.json \
#       --output estimate [--encoder wheel_odometry.csv] [--reanchor-maps]
# ================================================================
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from experiments.localization.inertial_filter import InertialFilter, skew


def read_poses(path: Path) -> dict[int, np.ndarray]:
    """TUM pose를 nanosecond key로 읽으며 빈 실패 궤적도 허용한다."""
    poses = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        values = np.array([float(value) for value in line.split()])
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(values[4:8]).as_matrix()
        pose[:3, 3] = values[1:4]
        poses[int(round(values[0]*1e9))] = pose
    return poses


class SlipScale:
    """엔코더 속력 대비 융합 속력의 지수 가중 비율(바퀴 슬립 보정 계수)을 추정한다."""

    def __init__(self, decay: float = .98, minimum_speed: float = .04, low: float = .3, high: float = 1.2) -> None:
        self.decay, self.minimum_speed, self.low, self.high = decay, minimum_speed, low, high
        self.numerator = self.denominator = 0.

    @property
    def value(self) -> float:
        """기록이 없으면 1(슬립 없음)을 반환한다."""
        if self.denominator <= 0:
            return 1.
        return float(np.clip(self.numerator/self.denominator, self.low, self.high))

    def update(self, encoder_speed: float, reference_speed: float) -> None:
        """엔코더 속력이 충분히 클 때만 기준(영상 융합) 속력과의 비율을 누적한다."""
        if encoder_speed < self.minimum_speed:
            return
        self.numerator = self.decay*self.numerator+reference_speed
        self.denominator = self.decay*self.denominator+encoder_speed


def propagate_interval(filters: list, imu: np.ndarray, a: float, b: float, max_gap_s: float = .1,
                       gap_twists: list | None = None) -> float | None:
    """[a,b]를 적분한다. 양옆 IMU 표본 간격이 max_gap_s를 넘으면 보간하지 않고 공백으로 처리하고 길이를 반환한다.

    gap_twists[i]가 있으면 공백 동안 filters[i]는 그 body twist로 평면 추측 항법을 한다.
    """
    mid = (a+b)/2
    index = int(np.searchsorted(imu[:, 0], mid))
    if 0 < index < len(imu) and imu[index, 0]-imu[index-1, 0] > max_gap_s:
        for position, filter_ in enumerate(filters):
            twist = gap_twists[position] if gap_twists is not None else None
            if twist is None:
                filter_.imu_gap(b-a)
            else:
                filter_.imu_gap(b-a, twist=twist)
        return b-a
    sample = np.array([np.interp(mid, imu[:, 0], imu[:, column]) for column in range(1, 7)])
    for filter_ in filters:
        filter_.propagate(sample[:3], sample[3:], float(b-a))
    return None


def needs_reanchor(previous_ok: bool, ok: bool, previous_map: str, map_id: str) -> bool:
    """ORB가 다시 추적을 잡았고 새 map일 수 있으면 True. 리셋된 ORB는 map id를 0부터 다시 시작하므로
    id만 믿지 않고 추적 상실 뒤 복귀(상실 → OK)도 새 map으로 본다."""
    return ok and (map_id != previous_map or not previous_ok)


def move_baseline(filter_, accepted: bool, refresh: bool) -> bool:
    """보정 뒤 기준 pose를 현재로 옮겨야 하면 True. 거부된 경우에도 refresh면 clone을 현재로 옮긴다.

    거부 때 기준선을 두면 다음 비교가 누적 예측 오차를 안은 더 긴 기준선이 되어 계속 거부되는 악순환이 생긴다.
    """
    if accepted:
        return True
    if refresh:
        filter_.clone_current()
        return True
    return False


def run(sequence: Path, orb: Path, prior_path: Path, output: Path, reanchor_maps: bool = False,
        encoder: Path | None = None, encoder_updates: bool = False, slip_limit: float = .6,
        refresh_baseline: bool = True) -> None:
    """센서·ORB 추정·시작 prior만 읽고 GT는 평가 프로그램에 맡긴다.

    encoder는 IMU 공백 중 추측 항법에만 쓰며, 슬립 계수가 slip_limit 아래면 신뢰하지 않고 속도 0으로 둔다.
    encoder_updates=True면 매 프레임 속도 관측으로도 넣는다(슬립이 큰 구간에서 해로워 기본은 끔).
    """
    started = time.perf_counter()
    if not (orb/'complete.txt').exists():
        raise ValueError('ORB run incomplete')
    calib = json.loads((sequence/'calibration.json').read_text())
    imu = np.loadtxt(sequence/'mav0/imu0/data.csv', delimiter=',', comments='#')
    imu[:, 0] *= 1e-9
    times_ns = np.loadtxt(sequence/'camera_times.txt', dtype=np.int64, ndmin=1)
    times = times_ns*1e-9
    if np.any(np.diff(imu[:, 0]) <= 0) or imu[0, 0] > times[0] or imu[-1, 0] < times[-1]:
        raise ValueError('IMU times invalid or do not cover images')
    with (orb/'tracking.csv').open() as file:
        tracking = list(csv.DictReader(file))
    if [int(row['timestamp_ns']) for row in tracking] != times_ns.tolist():
        raise ValueError('ORB and sensor frame times differ')
    if any(int(row.get('imu_samples', 0)) for row in tracking):
        raise ValueError('use visual-only ORB input, not ORB already fused with IMU')
    if tracking[0]['state'] != '2':
        raise ValueError('visual tracking must initialize at the known stationary start')
    visual = read_poses(orb/'base_trajectory.tum')
    prior = json.loads(prior_path.read_text())
    base = np.asarray(prior['world_from_base'], dtype=float)
    left = np.linalg.inv(np.asarray(calib['ROS.T_c1_base']))
    base_from_imu = left @ np.linalg.inv(np.asarray(calib['IMU.T_b_c1']))
    static = imu[(imu[:, 0] >= times[0]) & (imu[:, 0] < times[0]+.3), 1:]
    if len(static) < max(8, int(.09/np.median(np.diff(imu[:, 0])))) or np.max(static.std(axis=0)) > .2 \
            or np.linalg.norm(static[:, :3].mean(axis=0)) > .1:
        raise ValueError('initial 0.3 s must be stationary for gravity and gyro calibration')
    force = base_from_imu[:3, :3] @ static[:, 3:].mean(axis=0)
    if abs(np.linalg.norm(force)-9.81) > .3:
        raise ValueError('initial acceleration is inconsistent with static gravity')
    yaw = Rotation.from_matrix(base[:3, :3]).as_euler('xyz')[2]
    roll, pitch = np.arctan2(force[1], force[2]), np.arctan2(-force[0], np.linalg.norm(force[1:]))
    base[:3, :3] = Rotation.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
    position_sigma = np.asarray(prior.get('position_sigma_m', [.03]*3))
    angle_sigma = np.deg2rad(prior.get('rotation_sigma_deg', [2.]*3))
    gravity_sigma = np.sqrt(calib['IMU.NoiseAcc']**2/.3+.05**2)/9.81
    angle_sigma[:2] = 1/np.sqrt(1/angle_sigma[:2]**2+1/gravity_sigma**2)
    transform = np.zeros((6, 6))
    transform[:3, :3] = np.eye(3)
    transform[:3, 3:] = -base[:3, :3] @ skew(base_from_imu[:3, 3])
    transform[3:, 3:] = base_from_imu[:3, :3].T
    pose_covariance = transform @ np.diag(np.r_[position_sigma, angle_sigma]**2) @ transform.T
    initial_covariance = np.diag(np.array([.03]*3+[.1]*3+[.01]*3+[.005]*3+[.05]*3)**2)
    indices = [0, 1, 2, 6, 7, 8]
    initial_covariance[np.ix_(indices, indices)] = pose_covariance
    names = ['imu_only', 'visual_imu'] + (['visual_imu_enc'] if encoder is not None else [])
    visual_names = [name for name in names if name != 'imu_only']
    filters = {name: InertialFilter(base @ base_from_imu, calib, initial_covariance) for name in names}
    for filter_ in filters.values():
        filter_.gyro_bias = static[:, :3].mean(axis=0)
    wheel = None
    if encoder is not None:
        wheel = np.loadtxt(encoder, delimiter=',', skiprows=1)
        if wheel.ndim != 2 or wheel.shape[1] != 4 or np.any(np.diff(wheel[:, 0]) <= 0):
            raise ValueError('encoder csv needs increasing t_s, vx, vy, wz rows')
    slip = SlipScale()

    def encoder_twist(time_s: float) -> np.ndarray | None:
        if wheel is None or time_s < wheel[0, 0] or time_s > wheel[-1, 0]:
            return None
        return np.array([np.interp(time_s, wheel[:, 0], wheel[:, column]) for column in (1, 2, 3)])

    map_id = tracking[0]['map_id']
    references = {name: visual[int(times_ns[0])] @ base_from_imu for name in visual_names}
    last_update = {name: times[0] for name in visual_names}
    histories, diagnostics = {name: [] for name in names}, []
    output.mkdir(parents=True, exist_ok=False)
    previous = times[0]
    imu_gap_s, maps_reanchored, previous_ok = 0., 0, True
    for frame, timestamp in enumerate(times):
        knots = np.r_[previous, imu[(imu[:, 0] > previous+1e-10) & (imu[:, 0] < timestamp-1e-10), 0], timestamp]
        for a, b in zip(knots[:-1], knots[1:]):
            if b-a <= 1e-10:
                continue
            twist = encoder_twist((a+b)/2)
            usable = twist is not None and slip.value >= slip_limit
            gap_twists = [None if name != 'visual_imu_enc' or not usable else
                          (slip.value*twist[0], slip.value*twist[1], twist[2]) for name in names]
            if propagate_interval(list(filters.values()), imu, float(a), float(b), gap_twists=gap_twists) is not None:
                imu_gap_s += float(b-a)
        previous = timestamp
        row = {'timestamp_ns': int(times_ns[frame]), 'orb_state': tracking[frame]['state'],
               'map_id': tracking[frame]['map_id'], 'imu_status': {name: f.status for name, f in filters.items()}}
        ok = tracking[frame]['state'] == '2' and int(tracking[frame]['map_id']) >= 0
        same_map = not needs_reanchor(previous_ok, ok, map_id, tracking[frame]['map_id'])
        if reanchor_maps and not same_map and ok:
            # 새 ORB map은 원점이 다르므로 상대 관측만 쓴다: 필터 현재 pose를 기준으로 새 map의 첫 pose에 닻을 내린다.
            measured = visual[int(times_ns[frame])] @ base_from_imu
            for name in visual_names:
                filters[name].clone_current()
                references[name], last_update[name] = measured, timestamp
            map_id, same_map = tracking[frame]['map_id'], False
            maps_reanchored += 1
        previous_ok = ok
        for name in visual_names:
            info = {'accepted': False, 'reason': 'not_due_or_no_visual'}
            if tracking[frame]['state'] == '2' and same_map and timestamp-last_update[name] >= .1-1e-8:
                measured = visual[int(times_ns[frame])] @ base_from_imu
                info = filters[name].relative_update(np.linalg.inv(references[name]) @ measured)
                if move_baseline(filters[name], info['accepted'], refresh_baseline):
                    references[name], last_update[name] = measured, timestamp
            row[name] = info
        twist = encoder_twist(float(timestamp))
        if twist is not None:
            reference = filters['visual_imu']
            if reference.status == 'ok':
                body_velocity = reference.pose[:3, :3].T @ reference.velocity
                slip.update(float(np.hypot(twist[0], twist[1])), float(np.hypot(body_velocity[0], body_velocity[1])))
            enc_filter = filters['visual_imu_enc']
            if encoder_updates and enc_filter.status != 'imu_gap' and slip.value >= slip_limit:
                scaled = slip.value*twist[:2]
                row['encoder'] = enc_filter.velocity_update(scaled, sigma=.03+.1*float(np.linalg.norm(scaled)))
            row['slip_scale'] = slip.value
        for name, filter_ in filters.items():
            pose = filter_.pose @ np.linalg.inv(base_from_imu)
            histories[name].append([timestamp, *pose[:3, 3], *Rotation.from_matrix(pose[:3, :3]).as_quat()])
            row[name+'_state'] = {'velocity_imu_world': filter_.velocity.tolist(),
                'gyro_bias': filter_.gyro_bias.tolist(), 'accel_bias': filter_.accel_bias.tolist(),
                'covariance_diagonal': filter_.covariance.diagonal()[:15].tolist()}
        diagnostics.append(row)
    for name, history in histories.items():
        np.savetxt(output/f'{name}.tum', history, fmt='%.9f')
    inputs = [sequence/'calibration.json', sequence/'mav0/imu0/data.csv', sequence/'camera_times.txt',
              orb/'base_trajectory.tum', orb/'tracking.csv', prior_path, *([encoder] if encoder is not None else [])]
    report = {'frames': diagnostics, 'runtime_s': time.perf_counter()-started,
              'input_sha256': {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
              'static_calibration_s': .3, 'known_start': prior, 'visual_measurement_hz': 10,
              'state_order': 'IMU center: world p, world v, right body rotation error, gyro bias, accel bias',
              'imu_gap_total_s': imu_gap_s, 'maps_reanchored': maps_reanchored,
              'final_slip_scale': slip.value if encoder is not None else None,
              'limitations': ['loose relative ORB/IMU fusion; not official ORB inertial initialization',
                  'no wall-direction update in this package',
                  'assumed visual covariance; temporal correlation of ORB relative poses is not modeled',
                  'known starting position/yaw and initial 0.3 s static calibration required',
                  'fixed camera/IMU calibration and timestamps; no online time-offset estimation',
                  'IMU gaps over 0.1 s are not integrated: encoder dead reckoning (slip scaled) when available, else zero velocity',
                  'encoder twist ignores the IMU lever arm; slip scale is an exponential mean, not a calibrated model']}
    (output/'tracking.json').write_text(json.dumps(report, indent=2)+'\n')
    (output/'complete.txt').write_text('complete\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('sequence', 'orb', 'initial-prior', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--encoder', type=Path, default=None)
    parser.add_argument('--reanchor-maps', action='store_true', help='새 ORB map마다 상대 관측 기준만 다시 잡는다')
    parser.add_argument('--encoder-updates', action='store_true', help='엔코더를 매 프레임 속도 관측으로도 사용')
    args = parser.parse_args()
    run(args.sequence, args.orb, args.initial_prior, args.output, args.reanchor_maps, args.encoder, args.encoder_updates)
