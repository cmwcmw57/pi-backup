# ================================================================
# inertial.py
# 설명: 저장한 stereo·보정 IMU를 공통 시각으로 재생하고 실제 관성 초기화 여부를 별도 기록한다.
# 사용법: .venv-mapping/bin/python -m experiments.handheld_mapping.inertial RUN --allow-legacy
# ================================================================
import argparse
import csv
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

from experiments.handheld_mapping.compare import ROOT, runner_environment, summarize_timings, vocabularies
from experiments.handheld_mapping.offline import iter_pairs, write_settings
from experiments.handheld_mapping.storage import atomic_json


def validate_imu_provenance(run: Path) -> dict:
    """저장된 보정값이 capture 당시 파일·축 매핑에서 왔는지 확인한다."""
    from experiments.handheld_mapping.imu import load_calibration
    from experiments.handheld_mapping.time_alignment import file_sha256
    calibration = load_calibration(run/'imu_calibration.json')
    manifest = json.loads((run/'imu_manifest.json').read_text())
    if (manifest.get('calibration_sha256') != file_sha256(run/'imu_calibration.json')
            or manifest.get('urdf_sha256') != file_sha256(run/'robot.urdf')
            or manifest.get('chip_to_body_actual') != calibration['chip_to_body']
            or manifest.get('chip_to_body_expected') != calibration['chip_to_body']):
        raise ValueError('IMU calibration/mount provenance mismatch')
    return calibration


def imu_from_camera(urdf: Path, camera_from_base: np.ndarray) -> np.ndarray:
    """body 축으로 보정된 IMU 원점을 URDF 위치에 놓아 camera→IMU 변환을 만든다."""
    joint = ET.parse(urdf).getroot().find("./joint[@name='base_to_imu']")
    if joint is None or joint.get('type') != 'fixed':
        raise ValueError('fixed base_to_imu joint required')
    if joint.find('parent').get('link') != 'base_link' or joint.find('child').get('link') != 'imu_link':
        raise ValueError('base_to_imu reference frames disagree')
    origin = joint.find('origin')
    xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError('invalid IMU origin')
    base_from_imu = np.eye(4)
    base_from_imu[:3, 3] = xyz
    # calibration.frame=rover_body: 이미 축 permutation을 적용했으므로 URDF rotation을 다시 곱하지 않는다.
    return np.linalg.inv(base_from_imu) @ np.linalg.inv(camera_from_base)


def camera_from_urdf(urdf: Path, stereo: dict) -> np.ndarray:
    """저장된 CameraInfo rectification과 두 fixed 장착 joint로 camera←base를 만든다."""
    robot = ET.parse(urdf).getroot()
    base_from_camera = np.eye(4)
    for name, parent, child in [('base_to_camera', 'base_link', 'camera_link'),
                               ('camera_to_left_optical_frame', 'camera_link', 'camera_left_optical_frame')]:
        joint = robot.find(f"./joint[@name='{name}']")
        if joint is None or joint.get('type') != 'fixed' or joint.find('parent').get('link') != parent or joint.find('child').get('link') != child:
            raise ValueError('fixed camera mount joint required')
        origin = joint.find('origin')
        xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
        rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')
        if xyz.shape != (3,) or rpy.shape != (3,) or not np.isfinite(np.r_[xyz, rpy]).all():
            raise ValueError('invalid camera mount origin')
        cr, cp, cy = np.cos(rpy)
        sr, sp, sy = np.sin(rpy)
        transform = np.eye(4)
        transform[:3,:3] = [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                           [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]]
        transform[:3,3] = xyz
        base_from_camera = base_from_camera @ transform
    rectified = np.eye(4)
    rectified[:3,:3] = np.asarray(stereo['source_camera_info']['left']['r']).reshape(3,3)
    if not np.isfinite(rectified).all() or not np.allclose(rectified[:3,:3].T@rectified[:3,:3], np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rectified[:3,:3]), 1.):
        raise ValueError('invalid camera rectification')
    return rectified @ np.linalg.inv(base_from_camera)


def restore_camera_stamps(folder: Path, stamp_map: dict[int, int]) -> None:
    """공통 clock 재생 시각을 원본 camera 시각으로 복원해 TSDF 원본 대응을 유지한다."""
    keys = np.asarray(sorted(stamp_map), dtype=np.int64)
    for path in [folder/'timings.csv', *folder.glob('map_*_keyframes.csv')]:
        if not path.exists():
            continue
        with path.open() as stream:
            reader = csv.DictReader(stream)
            names = reader.fieldnames
            rows = list(reader)
        for row in rows:
            name = 'stamp_ns' if path.name == 'timings.csv' else 'source_stamp'
            value = int(row[name]) if name == 'stamp_ns' else round(float(row[name])*1e9)
            index = int(np.searchsorted(keys, value))
            candidates = [i for i in (index-1, index) if 0 <= i < len(keys)]
            closest = min(candidates, key=lambda i: abs(int(keys[i])-value))
            if abs(int(keys[closest])-value) > 2000:
                raise ValueError('unknown keyframe/camera stamp')
            restored = stamp_map[int(keys[closest])]
            row[name] = str(restored) if name == 'stamp_ns' else format(restored*1e-9, '.17g')
        temporary = path.with_suffix('.retimed')
        with temporary.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=names)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)


def fusion_evidence(path: Path, completion: dict) -> dict:
    """IMU를 전달했다는 사실과 map 관성 초기화가 실제 완료된 사실을 구분한다."""
    count = initialized = submitted = 0
    final = False
    with path.open() as stream:
        for row in csv.DictReader(stream):
            count += 1
            final = row.get('imu_initialized') == '1'
            initialized += int(final)
            submitted += int(row.get('imu_samples', 0))
    maps = completion['exported_imu_initialized_maps']
    if not isinstance(maps, list) or any(type(v) is not int or v < 0 for v in maps):
        raise ValueError('invalid exported inertial map evidence')
    return {'orb_fused': bool(maps), 'final_export_has_inertial_map': bool(maps),
            'exported_imu_initialized_maps': maps, 'last_tracked_frame_imu_initialized': final,
            'imu_initialized_frames': initialized, 'imu_initialized_fraction': initialized/count if count else 0.,
            'imu_samples_submitted': submitted}


def run_inertial(run: Path, output: Path | None = None, correction_s: float | None = None,
                 allow_legacy: bool = False, segment: int | None = None,
                 correlation_window_s: tuple[float, float] | None = None,
                 noise_override: dict[str, float] | None = None,
                 use_measured_constant: bool = False) -> dict:
    """비파괴 offline 비교를 실행하며 미검증 보정·noise를 실험 결과에 명시한다."""
    from experiments.handheld_mapping.inertial_timing import aligned_inputs
    from experiments.handheld_mapping.time_alignment import file_sha256
    run = Path(run).resolve()
    folder = Path(output).resolve() if output is not None else run/'orb_inertial'
    folder.mkdir(parents=True, exist_ok=False)
    result = {'status': 'preparing', 'imu_requested': True, 'orb_fused': False,
              'imu_clock_verified': False, 'extrinsic_verified': False, 'noise_verified': False,
              'source': str(run), 'backend': 'orb_inertial', 'raw_preserved': True,
              'gt_accuracy_evaluated': False}
    try:
        calibration = validate_imu_provenance(run)
        data = aligned_inputs(run, correction_s, allow_legacy, segment=segment,
                              correlation_window_s=correlation_window_s,
                              use_measured_constant=use_measured_constant)
        atomic_json(folder/'alignment.json', data['report'])
        times = data['imu_times_s']
        origin = data['camera_origin_ns']
        stamp_map = {}
        with (folder/'pairs.tsv').open('w') as stream:
            for stamp, left, right in iter_pairs(run):
                t = data['camera_times_s'][stamp]
                if t < times[1] or t >= times[-1]:
                    continue
                common = origin+round(t*1e9)
                stamp_map[common] = stamp
                stream.write(f'{common}\t{left}\t{right}\n')
        if len(stamp_map) < 30:
            raise ValueError('fewer than 30 stereo pairs covered by IMU')
        atomic_json(folder/'camera_stamp_map.json', {str(k): v for k, v in stamp_map.items()})
        with (folder/'imu.csv').open('w') as stream:
            writer = csv.writer(stream)
            writer.writerow(['timestamp_ns', 'gx_rad_s', 'gy_rad_s', 'gz_rad_s', 'ax_m_s2', 'ay_m_s2', 'az_m_s2'])
            for t, gyro, acc in zip(times, data['gyro'], data['acc']):
                writer.writerow([origin+round(float(t)*1e9), *gyro, *acc])
        settings = write_settings(run, folder)
        stereo = json.loads((run/'capture_calibration.json').read_text())['stereo']
        if calibration.get('frame') != 'rover_body':
            raise ValueError('body-axis corrected IMU calibration required')
        transform = imu_from_camera(run/'robot.urdf', camera_from_urdf(run/'robot.urdf', stereo))
        frequency = round(1/float(np.median(np.diff(times))))
        if not 10 <= frequency <= 1000:
            raise ValueError('implausible effective IMU frequency')
        # ORB EuRoC 시작값이다. 이 센서의 Allan variance 보정값으로 주장하지 않는다.
        noise = {'IMU.NoiseGyro': .00017, 'IMU.NoiseAcc': .002,
                 'IMU.GyroWalk': .000019393, 'IMU.AccWalk': .003, **(noise_override or {})}
        with settings.open('a') as stream:
            for key, value in {**noise, 'IMU.Frequency': float(frequency), 'IMU.InsertKFsWhenLost': 0}.items():
                stream.write(f'{key}: {value}\n')
            stream.write('IMU.T_b_c1: !!opencv-matrix\n  rows: 4\n  cols: 4\n  dt: f\n  data: [ '+
                         ', '.join(format(float(v), '.12g') for v in transform.flat)+' ]\n')
        result.update(paired_frames=len(stamp_map), imu_frequency=frequency, noise_starting_values=noise,
                      T_imu_c1=transform.tolist(), imu_axes='rover_body_at_imu_origin',
                      imu_time_correction_s=data['report']['imu_time_correction_s'],
                      time_source=data['report']['imu_time_source'], input_sha256=data['report']['input_sha256'])
        result['input_sha256']['robot.urdf'] = file_sha256(run/'robot.urdf')
        result['input_sha256']['imu_manifest.json'] = file_sha256(run/'imu_manifest.json')
        binary = ROOT/'outputs/handheld_mapping/orb3_imu_build/rover_orb3_imu_compare'
        orb_voc, line_voc = vocabularies()
        with (folder/'runner.log').open('wb') as log:
            subprocess.run([str(binary), str(orb_voc), str(line_voc), str(settings), str(folder/'pairs.tsv'),
                            str(folder), str(folder/'imu.csv')], cwd=folder, env=runner_environment(True),
                           stdout=log, stderr=log, check=True, timeout=3600)
        completion = json.loads((folder/'runner_complete.json').read_text())
        if completion.get('status') != 'complete' or completion.get('imu_requested') is not True:
            raise RuntimeError('inertial runner completion contract mismatch')
        restore_camera_stamps(folder, stamp_map)
        result.update(summarize_timings(folder/'timings.csv'))
        result.update(fusion_evidence(folder/'timings.csv', completion))
        result.update(final_tracked_map_initialized=completion.get('final_tracked_map_initialized'),
                      imu_frames_skipped=completion.get('imu_frames_skipped'),
                      imu_replay_paced=completion.get('imu_replay_paced'))
        result['status'] = 'initialized' if result['final_export_has_inertial_map'] else 'not_initialized'
        for name, checksum in result['input_sha256'].items():
            if file_sha256(run/name) != checksum:
                raise RuntimeError('input changed during inertial experiment')
    except Exception as error:
        result.update(status='failed', error=str(error))
        atomic_json(folder/'summary.json', result)
        raise
    atomic_json(folder/'summary.json', result)
    return result


def run_inertial_auto(run: Path, output: Path | None = None) -> dict:
    """한 boot의 운동 표본을 선택하고 시차 품질 검사 후 관성 재생을 자동 실행한다."""
    from experiments.handheld_mapping.imu import finite_vector, strict_json
    from experiments.handheld_mapping.inertial_timing import aligned_inputs
    run = Path(run).resolve()
    activity = {}
    with (run/'imu.jsonl').open() as stream:
        for count, line in enumerate(stream, 1):
            if count > 400000 or len(line) > 65536:
                raise ValueError('IMU automatic analysis limit exceeded')
            row = strict_json(line)
            if row.get('acceptance') != 'accepted':
                continue
            segment = row['clock'].get('segment', 0)
            if type(segment) is not int or segment < 0:
                raise ValueError('invalid IMU segment')
            gyro = np.radians(finite_vector(row['calibrated']['gyro_dps'], 'gyro_dps'))
            activity[segment] = activity.get(segment, 0)+int(np.linalg.norm(gyro) > .02)
    if not activity or max(activity.values()) < 20:
        raise ValueError('insufficient motion for automatic inertial alignment')
    selected = max(activity, key=lambda key: (activity[key], key))
    data = aligned_inputs(run, 0, segment=selected)
    moving = np.flatnonzero(np.linalg.norm(data['gyro'], axis=1) > .02)
    if len(moving) < 20 or data['imu_times_s'][moving[-1]]-data['imu_times_s'][moving[0]] < 10:
        raise ValueError('insufficient motion duration for automatic alignment')
    window = (float(data['imu_times_s'][moving[0]]-1), float(data['imu_times_s'][moving[-1]]+1))
    result = run_inertial(run, output=output, segment=selected, correlation_window_s=window)
    result['automatic_selection'] = {'selected_segment': selected, 'motion_samples_by_segment': activity,
                                     'motion_threshold_rad_s': .02, 'correlation_window_s': list(window),
                                     'quality_guards_relaxed': False, 'reset_boundaries_integrated': False}
    atomic_json((Path(output) if output is not None else run/'orb_inertial')/'summary.json', result)
    return result


def main() -> None:
    """명시한 run에 별도 결과를 생성한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--correction-s', type=float)
    parser.add_argument('--allow-legacy', action='store_true')
    parser.add_argument('--measured-constant', action='store_true',
                        help='기존 SPI 50Hz 경로에서 잰 -45ms 보정을 사용(패치 펌웨어·raw 경로에는 불가)')
    parser.add_argument('--segment', type=int)
    parser.add_argument('--correlation-window-s', type=float, nargs=2)
    parser.add_argument('--auto', action='store_true', help='운동량이 충분한 boot와 상관 구간을 자동 선택')
    args = parser.parse_args()
    window = tuple(args.correlation_window_s) if args.correlation_window_s is not None else None
    if args.auto:
        if (args.correction_s is not None or args.allow_legacy or args.measured_constant
                or args.segment is not None or window is not None):
            parser.error('--auto는 수동 시각/구간/legacy 옵션과 함께 사용할 수 없습니다')
        result = run_inertial_auto(args.run, args.output)
    else:
        result = run_inertial(args.run, args.output, args.correction_s, args.allow_legacy, args.segment, window,
                              use_measured_constant=args.measured_constant)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()


from experiments.handheld_mapping.inertial_verified import prepare_input
