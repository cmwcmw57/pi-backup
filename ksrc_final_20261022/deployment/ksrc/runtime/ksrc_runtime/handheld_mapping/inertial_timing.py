# ================================================================
# inertial_timing.py
# 설명: MCU 제어 시계에 평균 IMU 읽기 offset을 적용하고 카메라와 같은 상대 시각으로 변환한다.
# 사용법: from ksrc_runtime.handheld_mapping.inertial_timing import aligned_inputs
# ================================================================
import csv
import math
from pathlib import Path
from typing import Iterator

import numpy as np

from ksrc_runtime.handheld_mapping.imu import finite_vector, strict_json
from ksrc_runtime.handheld_mapping.time_alignment import estimate_offset, file_sha256, fit_clock


TIMING_KEYS = ('imu_timing_valid', 'imu_mean_read_offset_us', 'imu_time_source')


# 기존 SPI 50Hz telemetry 경로(제어 tick 시각)에서 영상 회전과 자이로 상관으로 잰 값이다.
# 근거 기록: 20261015_212049_bc26d3(-45ms), 20261016 패치 기록들(-45~-50ms), 20261017_213103_fa7cf6(-45ms).
# 평균 읽기 offset을 이미 더하는 패치 펌웨어나 raw 전송 경로에는 쓰지 않는다.
MEASURED_LEGACY_SPI_CORRECTION_S = -.045


def _rows(path: Path, limit: int) -> Iterator[dict]:
    """로그의 행 크기와 개수를 제한하여 원본을 읽는다."""
    with path.open() as stream:
        for count, line in enumerate(stream, 1):
            if count > limit or len(line) > 65536:
                raise ValueError(f'analysis log limit exceeded: {path.name}')
            yield strict_json(line)


def _stamp(value: object, name: str, upper: int = 2**63) -> int:
    """정밀도 손실 없이 양의 정수 clock 범위를 검사한다."""
    if type(value) is not int or not 0 <= value < upper:
        raise ValueError(f'invalid {name} clock value')
    return value


def _visual(run: Path, camera: dict[int, float]) -> dict:
    """같은 map에 속한 연속 유효 pose의 회전 크기를 구한다."""
    starts, ends, omega = [], [], []
    previous = None
    with (run/'orb_live/live_poses.csv').open() as stream:
        for count, row in enumerate(csv.DictReader(stream), 1):
            if count > 100000:
                raise ValueError('visual pose sample limit exceeded')
            stamp, state, map_id = int(row['stamp_ns']), int(row['state']), int(row['map_id'])
            if state != 2 or map_id < 0 or stamp not in camera:
                previous = None
                continue
            rotation = np.array([[float(row[f'm{i}{j}']) for j in range(3)] for i in range(3)])
            if (not np.isfinite(rotation).all()
                    or not np.allclose(rotation.T@rotation, np.eye(3), atol=.003)
                    or not np.isclose(np.linalg.det(rotation), 1., atol=.003)):
                raise ValueError('invalid visual rotation')
            current = (camera[stamp], map_id, rotation)
            if previous is not None:
                dt = current[0]-previous[0]
                if dt <= 0:
                    raise ValueError('nonmonotonic visual poses')
                if current[1] == previous[1] and .025 <= dt <= .15:
                    angle = np.arccos(np.clip((np.trace(previous[2].T@rotation)-1)*.5, -1., 1.))
                    starts.append(previous[0])
                    ends.append(current[0])
                    omega.append(angle/dt)
            previous = current
    return {'start': np.asarray(starts), 'end': np.asarray(ends), 'omega': np.asarray(omega)}


def aligned_inputs(run: Path, correction_s: float | None = None, allow_legacy: bool = False,
                   segment: int | None = None,
                   correlation_window_s: tuple[float, float] | None = None,
                   use_measured_constant: bool = False) -> dict:
    """검증되지 않은 실험용 IMU/camera 시간을 반환하며 불안정·공백·새 timing 누락은 거부한다."""
    if correction_s is not None and (isinstance(correction_s, bool)
            or not isinstance(correction_s, (int, float)) or not math.isfinite(correction_s)
            or abs(correction_s) > .5):
        raise ValueError('manual correction must be finite seconds within [-0.5, 0.5]')
    if use_measured_constant and correction_s is not None:
        raise ValueError('use either correction_s or use_measured_constant, not both')
    if segment is not None and (type(segment) is not int or segment < 0):
        raise ValueError('segment must be a nonnegative integer or None')
    if correlation_window_s is not None and (
            not isinstance(correlation_window_s, tuple) or len(correlation_window_s) != 2
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) for value in correlation_window_s)
            or correlation_window_s[0] >= correlation_window_s[1]):
        raise ValueError('correlation_window_s must be two finite increasing seconds')
    try:
        return _aligned_inputs(Path(run).resolve(), correction_s, allow_legacy, segment,
                               correlation_window_s, use_measured_constant)
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError(f'invalid alignment input: {error}') from error


def _aligned_inputs(run: Path, correction_s: float | None, allow_legacy: bool,
                    segment: int | None, correlation_window_s: tuple[float, float] | None,
                    use_measured_constant: bool = False) -> dict:
    """선택한 단일 Pi clock에서 제어 tick을 fit한 뒤 평균 읽기 offset을 더한다."""
    summary = run/'imu_summary.json'
    if not summary.is_file() or strict_json(summary.read_text()).get('state') != 'stopped':
        raise ValueError('stop IMU recording before alignment')
    imu, counts, discarded = [], {}, {}
    for row in _rows(run/'imu.jsonl', 400000):
        state = row.get('acceptance', 'unknown')
        counts[state] = counts.get(state, 0)+1
        if state == 'reset_boundary_discard':
            if segment is None:
                raise ValueError('MCU reset in recording')
            discarded[state] = discarded.get(state, 0)+1
            continue
        if state == 'accepted':
            row_segment = _stamp(row['clock'].get('segment', 0), 'MCU segment')
            if segment is None and row_segment != 0:
                raise ValueError('MCU reset segment in recording')
            if segment is not None and row_segment != segment:
                discarded['other_segment'] = discarded.get('other_segment', 0)+1
                continue
            imu.append(row)
        else:
            discarded[state] = discarded.get(state, 0)+1
    frames = [row for row in _rows(run/'frames.jsonl', 400000) if row.get('stream') == 'left']
    if len(imu) < 20 or len(imu) > 200000 or len(frames) < 2 or len(frames) > 100000:
        raise ValueError('insufficient or excessive IMU/camera samples')
    telemetry = [row.get('raw', {}).get('telemetry', {}) for row in imu]
    new_timing = any(any(key in item for key in TIMING_KEYS) for item in telemetry)
    if use_measured_constant and new_timing and not any(
            item.get('imu_timing_valid') not in (None, False) or item.get('imu_mean_read_offset_us') is not None
            or item.get('imu_time_source') is not None for item in telemetry):
        # 펌웨어가 타이밍 필드를 보내지만 전부 '없음'이면 평균 읽기 시각 패치가 없는 기존 경로다.
        new_timing = False
    if use_measured_constant:
        if new_timing:
            raise ValueError('measured constant is for legacy control-tick timing; patched timing already adds the mean read offset')
        correction_s, allow_legacy = MEASURED_LEGACY_SPI_CORRECTION_S, True
    if not new_timing and not allow_legacy:
        raise ValueError('legacy IMU control timestamps require allow_legacy=True')
    mean_offsets = []
    for item in telemetry:
        if new_timing:
            value = item.get('imu_mean_read_offset_us')
            if (item.get('imu_timing_valid') is not True
                    or item.get('imu_time_source') != 'mcu_mean_sample_read'
                    or type(value) is not int or not -32768 <= value <= 32767):
                raise ValueError('invalid or missing new IMU timing metadata')
            mean_offsets.append(value)
        else:
            mean_offsets.append(0)
    use_raw = (all(row.get('raw', {}).get('host_monotonic_raw_ns') is not None for row in imu)
               and all(row.get('pi_receipt_monotonic_raw_ns') is not None for row in frames))
    domain = 'CLOCK_MONOTONIC_RAW' if use_raw else 'CLOCK_MONOTONIC'
    mcu, receipt, realtime, gyro, acc = [], [], [], [], []
    for row in imu:
        clock = row['clock']
        mcu.append(_stamp(clock['mcu_t_ms'], 'MCU', 2**32))
        selected = (row['raw']['host_monotonic_raw_ns'] if use_raw else clock['pi_receipt_monotonic_ns'])
        receipt.append(_stamp(selected, domain))
        realtime.append(_stamp(clock['pi_receipt_realtime_ns'], 'IMU realtime'))
        g = finite_vector(row['calibrated']['gyro_dps'], 'gyro_dps')
        a = finite_vector(row['calibrated']['acc_g'], 'acc_g')
        if max(abs(v) for v in g) > 2500 or max(abs(v) for v in a) > 32:
            raise ValueError('IMU calibrated measurement outside sensor range')
        gyro.append(np.radians(g))
        acc.append(np.asarray(a)*9.80665)
    receipt = np.asarray(receipt, dtype=np.int64)
    realtime = np.asarray(realtime, dtype=np.int64)
    clock = fit_clock(np.asarray(mcu, dtype=np.int64), receipt)
    unwrapped = clock.pop('unwrapped_ms')
    clock['clock_domain'] = domain
    if clock['receipt_residual_trend_suspect'] is not False:
        raise ValueError('clock_model_unstable_or_insufficient_trend_evidence')
    # 제어 tick을 fit한다. 평균 읽기 시각에 직접 fit하면 상수 offset이 fit 절편에 흡수된다.
    times = ((unwrapped-clock['mcu_origin_unwrapped_ms'])*.001*clock['scale']
             + clock['offset_s']+np.asarray(mean_offsets)*1e-6*clock['scale'])
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('nonmonotonic IMU mean sample time')
    max_gap = float(np.max(np.diff(times)))
    if max_gap > .100000001:
        raise ValueError('IMU gap exceeds 100ms: split recording before alignment')
    offsets = (realtime-receipt).tolist()
    clock_pairs = list(zip(receipt.tolist(), offsets))
    fallback = int(np.median(realtime-receipt))
    camera, ages, camera_offsets = {}, [], []
    camera_stamps, camera_clock_values = [], []
    legacy_camera = 0
    for row in frames:
        stamp = _stamp(row['stamp_ns'], 'camera ROS')
        selected = row['pi_receipt_monotonic_raw_ns'] if use_raw else row['pi_receipt_monotonic_ns']
        pi = _stamp(selected, f'camera {domain}')
        if row.get('pi_receipt_realtime_ns') is None:
            if not allow_legacy:
                raise ValueError('legacy camera callback without realtime requires allow_legacy=True')
            offset = fallback
            legacy_camera += 1
        else:
            offset = _stamp(row['pi_receipt_realtime_ns'], 'camera realtime')-pi
            camera_offsets.append(offset)
            clock_pairs.append((pi, offset))
        age = (pi+offset-stamp)*1e-6
        if not -5 <= age <= 2000:
            raise ValueError('camera ROS header incompatible with Pi epoch')
        camera_stamps.append(stamp)
        camera_clock_values.append(stamp-offset)
        ages.append(age)
    offset_range_ns = max(offsets+camera_offsets)-min(offsets+camera_offsets)
    ordered_offsets = np.asarray([offset for _, offset in sorted(clock_pairs)], dtype=np.int64)
    max_local_change_ns = int(np.max(np.abs(np.diff(ordered_offsets))))
    # RAW에는 NTP slew가 적용되지 않는다. 정상적인 누적 offset 변화 대신 인접 step만 검사한다.
    relation_change_ns = max_local_change_ns if use_raw else offset_range_ns
    if relation_change_ns > 5_000_000:
        raise ValueError('Pi realtime/selected clock relation changed over 5ms')
    stamps = np.asarray(camera_stamps, dtype=np.int64)
    camera_clock = np.asarray(camera_clock_values, dtype=np.int64)
    if np.any(np.diff(stamps) <= 0) or np.any(np.diff(camera_clock) <= 0):
        raise ValueError('nonmonotonic camera timestamps')
    origin_pi = int(camera_clock[0])
    camera = {int(stamp): float((int(value)-origin_pi)*1e-9)
              for stamp, value in zip(stamps, camera_clock)}
    times -= (origin_pi-int(receipt[0]))*1e-9
    gyro = np.asarray(gyro)
    acc = np.asarray(acc)
    estimate = None
    if correction_s is None:
        visual = _visual(run, camera)
        if correlation_window_s is not None:
            start, end = correlation_window_s
            inside = (visual['start'] >= start) & (visual['end'] <= end)
            visual = {name: values[inside] for name, values in visual.items()}
        estimate = estimate_offset(times, np.linalg.norm(gyro, axis=1),
                                   visual['start'], visual['end'], visual['omega'])
        if estimate['status'] != 'estimated':
            raise ValueError('auto correction rejected: '+', '.join(estimate['reasons']))
        correction_s = estimate['imu_time_correction_s']
        source = 'visual_rotation_correlation'
    else:
        source = 'measured_constant_legacy_spi' if use_measured_constant else 'experimental_manual'
    times = times+float(correction_s)
    if times[-1] < 0 or times[0] > camera[int(stamps[-1])]:
        raise ValueError('IMU and camera have no aligned overlap')
    report = {'source_run': str(run), 'clock_domain': domain, 'clock_fit': clock,
              'selected_segment': segment, 'discarded_imu_rows': sum(discarded.values()),
              'correlation_window_s': list(correlation_window_s) if correlation_window_s is not None else None,
              'imu_selection_discard_counts': discarded,
              'clock_relation_guard': 'adjacent_pair_step' if use_raw else 'total_offset_range',
              'clock_relation_offset_range_ms': offset_range_ns*1e-6,
              'clock_relation_max_adjacent_change_ms': max_local_change_ns*1e-6,
              'clock_verified': False, 'imu_clock_verified': False,
              'imu_time_source': 'mcu_mean_sample_read' if new_timing else 'legacy_mcu_control_tick',
              'confidence': 'experimental_unverified' if new_timing else 'legacy_control_time_unverified',
              'correction_source': source, 'imu_time_correction_s': float(correction_s),
              'sign_convention': 'aligned_imu_time = fitted_control_time + mean_read_offset * clock_scale + correction_s',
              'accepted_imu_samples': len(imu), 'imu_acceptance_counts': counts,
              'camera_frames': len(camera), 'legacy_camera_frames': legacy_camera,
              'camera_header_age_p50_p95_ms': np.percentile(ages, [50, 95]).tolist(),
              'imu_max_gap_s': max_gap, 'mean_read_offset_us_min_max': [min(mean_offsets), max(mean_offsets)],
              'sensor_acquisition_time_verified': False,
              'clock_assumption': 'ROS header uses Pi realtime epoch; read completion is not sensor acquisition time',
              'input_sha256': {name: file_sha256(run/name) for name in (
                  'imu.jsonl', 'frames.jsonl', 'orb_live/live_poses.csv', 'capture_calibration.json',
                  'imu_calibration.json', 'imu_summary.json') if (run/name).is_file()}}
    if estimate is not None:
        report['offset_estimate'] = estimate
    return {'imu_times_s': times, 'gyro': gyro, 'acc': acc, 'camera_origin_ns': int(stamps[0]),
            'camera_times_s': camera, 'camera_stamps_ns': stamps, 'report': report}
