# ================================================================
# time_alignment.py
# 설명: MCU–Pi 시계 드리프트와 영상 회전–자이로 시차를 추정하되 검증된 동기화로 취급하지 않는다.
# 사용법: .venv-mapping/bin/python -m experiments.handheld_mapping.time_alignment RUN
# ================================================================
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.handheld_mapping.imu import finite_vector, strict_json
from experiments.handheld_mapping.storage import atomic_json


def file_sha256(path: Path) -> str:
    """원본 로그 전체를 RAM에 올리지 않고 hash를 계산한다."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fit_clock(mcu_ms: np.ndarray, pi_ns: np.ndarray) -> dict:
    """uint32 wrap을 풀고 수신 지터에 강한 MCU seconds→Pi seconds affine 모델을 구한다."""
    mcu_ms, pi_ns = np.asarray(mcu_ms, dtype=np.int64), np.asarray(pi_ns, dtype=np.int64)
    if len(mcu_ms) < 20 or mcu_ms.shape != pi_ns.shape or np.any(np.diff(pi_ns) <= 0):
        raise ValueError('insufficient or nonmonotonic clock samples')
    unwrapped = mcu_ms.copy()
    wraps = 0
    for i in range(1, len(mcu_ms)):
        if mcu_ms[i] < mcu_ms[i-1]:
            if mcu_ms[i-1] > 2**32-10000 and mcu_ms[i] < 10000:
                wraps += 1
            else:
                raise ValueError('MCU reset: split recording before alignment')
        unwrapped[i] += wraps*2**32
    if np.any(np.diff(unwrapped) <= 0):
        raise ValueError('duplicate or reversed MCU timestamps')
    x, y = (unwrapped-unwrapped[0])*.001, (pi_ns-pi_ns[0])*1e-9
    if x[-1] < 1:
        raise ValueError('clock duration below one second')
    keep = np.ones(len(x), bool)
    for _ in range(5):
        scale, intercept = np.polyfit(x[keep], y[keep], 1)
        residual = y-scale*x-intercept
        low, high = np.percentile(residual, [5, 95])
        keep = (residual >= low) & (residual <= high)
    if not .9 < scale < 1.1:
        raise ValueError('implausible MCU/Pi clock rate')
    # 20ms 제어 주기보다 큰 저주파 잔차 변화는 고정 affine 모델의 근거로 쓰지 않는다.
    edges = np.linspace(x[0], x[-1], 9)
    medians = []
    for i, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (x >= low) & ((x < high) if i < 7 else (x <= high))
        medians.append(float(np.median(residual[mask])*1000) if mask.sum() >= 20 else None)
    spread = float(max(medians)-min(medians)) if all(v is not None for v in medians) else None
    return {'scale': float(scale), 'offset_s': float(intercept),
            'mcu_origin_unwrapped_ms': int(unwrapped[0]), 'pi_origin_monotonic_ns': int(pi_ns[0]),
            'rate_difference_ppm': float((scale-1)*1e6),
            'receipt_fit_residual_p5_p50_p95_ms': (np.percentile(residual, [5, 50, 95])*1000).tolist(),
            'receipt_block_medians_ms': medians, 'receipt_block_median_range_ms': spread,
            'receipt_residual_trend_suspect': spread > 20 if spread is not None else None,
            'receipt_trend_limit_ms': 20,
            'absolute_receipt_delay_known': False, 'unwrapped_ms': unwrapped}


def estimate_offset(imu_t: np.ndarray, gyro_norm: np.ndarray, visual_start: np.ndarray,
                    visual_end: np.ndarray, visual_omega: np.ndarray) -> dict:
    """같은 비교 구간의 평균 각속도를 사용하고 IMU 시각에 더할 보정의 부호를 명시한다."""
    imu_t, gyro_norm, start, end, omega = [np.asarray(a, dtype=float) for a in
                                         (imu_t, gyro_norm, visual_start, visual_end, visual_omega)]
    result = {'status': 'rejected', 'reasons': [], 'imu_clock_verified': False,
              'orb_fused': False, 'imu_time_correction_s': None,
              'sign_convention': 'aligned_imu_time = fitted_imu_pi_time + imu_time_correction_s',
              'search_range_s': [-.5, .5], 'search_step_s': .005,
              'search_step_is_accuracy': False, 'excluded_gap_intervals': 0}
    if (len(imu_t) < 20 or len(start) < 2 or len(gyro_norm) != len(imu_t)
            or start.shape != end.shape or start.shape != omega.shape
            or any(not np.isfinite(a).all() for a in (imu_t, gyro_norm, start, end, omega))
            or np.any(np.diff(imu_t) <= 0) or np.any(end <= start)):
        result['reasons'] = ['invalid_or_insufficient_samples']
        return result
    inside = (start-.5 >= imu_t[0]) & (end+.5 <= imu_t[-1])
    gaps = np.flatnonzero(np.diff(imu_t) > .1)
    intersects = (np.searchsorted(imu_t[gaps], end+.5, side='left')
                  - np.searchsorted(imu_t[gaps+1], start-.5, side='right')) > 0
    result['excluded_gap_intervals'] = int(np.sum(inside & intersects))
    valid = inside & ~intersects
    start, end, omega = start[valid], end[valid], omega[valid]
    result['visual_intervals'] = len(start)
    if len(start) < 60 or end[-1]-start[0] < 10:
        result['reasons'] = ['insufficient_overlap']
        return result
    if np.std(omega) < .005 or np.std(gyro_norm) < .005:
        result['reasons'] = ['insufficient_motion']
        return result
    integral = np.r_[0., np.cumsum((gyro_norm[:-1]+gyro_norm[1:])*.5*np.diff(imu_t))]
    shifts = np.linspace(-.5, .5, 201)
    predictions = np.array([(np.interp(end+d, imu_t, integral)-np.interp(start+d, imu_t, integral))
                            /(end-start) for d in shifts])

    def curve(selection: np.ndarray) -> np.ndarray:
        observed = omega[selection]-np.mean(omega[selection])
        predicted = predictions[:, selection]-np.mean(predictions[:, selection], axis=1)[:, None]
        denominator = np.linalg.norm(predicted, axis=1)*np.linalg.norm(observed)
        return np.divide(predicted@observed, denominator, out=np.zeros(len(shifts)), where=denominator > 1e-12)

    scores = curve(np.ones(len(start), bool))
    best = int(np.argmax(scores))
    far = np.abs(shifts-shifts[best]) >= .125
    margin = float(scores[best]-np.max(scores[far]))
    halves = [start < np.median(start), start >= np.median(start)]
    reasons = []
    half_peaks, half_correlations = [], []
    for selection in halves:
        half_scores = curve(selection)
        peak = int(np.argmax(half_scores))
        half_correlations.append(float(half_scores[peak]))
        if (np.std(omega[selection]) < .005
                or np.std(predictions[peak, selection]) < .005):
            half_peaks.append(None)
            reasons.append('insufficient_half_motion')
        elif half_scores[peak] < .7 or peak in (0, len(shifts)-1):
            half_peaks.append(None)
            reasons.append('invalid_half_peak')
        else:
            half_peaks.append(float(shifts[peak]))
    if best in (0, len(shifts)-1): reasons.append('search_boundary')
    if scores[best] < .7: reasons.append('weak_correlation')
    if margin < .02: reasons.append('ambiguous_peak')
    if (all(peak is not None for peak in half_peaks)
            and abs(half_peaks[0]-half_peaks[1]) > .05):
        reasons.append('unstable_time_offset')
    reasons = list(dict.fromkeys(reasons))
    result.update(reasons=reasons, correlation=float(scores[best]), separated_peak_margin=margin,
                  raw_best_shift_s=float(shifts[best]), half_best_shift_s=half_peaks,
                  half_correlations=half_correlations,
                  lag_grid_s=shifts.tolist(), correlations=scores.tolist(),
                  comparison_visual_times_s=((start+end)*.5).tolist(),
                  visual_omega_rad_s=omega.tolist(),
                  matched_gyro_norm_rad_s=predictions[best].tolist())
    if not reasons:
        result.update(status='estimated', imu_time_correction_s=float(-shifts[best]))
    return result


def imu_samples(run: Path) -> dict:
    """한 줄씩 읽어 필요한 수치만 보관하며 reset을 넘는 시계 모델을 만들지 않는다."""
    rows = []
    counts = {}
    with (run/'imu.jsonl').open() as stream:
        for line in stream:
            if len(line) > 65536:
                raise ValueError('oversized IMU row')
            row = strict_json(line)
            state = row.get('acceptance', 'unknown')
            counts[state] = counts.get(state, 0)+1
            if state == 'reset_boundary_discard':
                raise ValueError('MCU reset in recording')
            if state != 'accepted':
                continue
            clock = row['clock']
            if clock.get('segment', 0) != 0:
                raise ValueError('MCU reset segment in recording')
            stamps = [clock[k] for k in ('mcu_t_ms', 'pi_receipt_monotonic_ns', 'pi_receipt_realtime_ns')]
            if any(type(v) is not int or not 0 <= v < 2**63 for v in stamps) or stamps[0] >= 2**32:
                raise ValueError('invalid IMU clock values')
            calibrated = row['calibrated']
            gyro = finite_vector(calibrated['gyro_dps'], 'gyro_dps')
            acc = finite_vector(calibrated['acc_g'], 'acc_g')
            rows.append((*stamps, *np.radians(gyro), *(np.asarray(acc)*9.80665),
                         row.get('raw', {}).get('telemetry', {}).get('imu_n', 0)))
            if len(rows) > 200000:
                raise ValueError('IMU analysis sample limit exceeded: split recording')
    if not rows:
        raise ValueError('no accepted IMU samples')
    # 장치 epoch의 ns 정밀도를 유지하기 위해 시각 열은 float 배열과 분리한다.
    return {'mcu_ms': np.array([r[0] for r in rows], dtype=np.int64),
            'pi_ns': np.array([r[1] for r in rows], dtype=np.int64),
            'realtime_ns': np.array([r[2] for r in rows], dtype=np.int64),
            'gyro': np.array([r[3:6] for r in rows]), 'acc': np.array([r[6:9] for r in rows]),
            'imu_n': [r[9] for r in rows], 'counts': counts}


def visual_samples(run: Path, pi_origin: int, fallback_offset: int, max_dt_s: float = .15) -> dict:
    """영상 ROS epoch를 Pi monotonic으로 옮기고 같은 map의 연속 유효 회전만 사용한다."""
    camera = {}
    offsets, ages, spans = [], [], []
    legacy = 0
    with (run/'frames.jsonl').open() as stream:
        for line in stream:
            row = strict_json(line)
            if row.get('stream') != 'left':
                continue
            stamp = row['stamp_ns']
            receipt = row['pi_receipt_monotonic_ns']
            if type(stamp) is not int or type(receipt) is not int:
                raise ValueError('invalid camera clock')
            realtime = row.get('pi_receipt_realtime_ns')
            if realtime is None:
                offset = fallback_offset
                legacy += 1
            else:
                if type(realtime) is not int:
                    raise ValueError('invalid camera realtime')
                offset = realtime-receipt
                offsets.append(offset)
                spans.append(row.get('pi_clock_sample_span_ns', 0))
            age_ms = (receipt+offset-stamp)*1e-6
            if not -5 <= age_ms <= 2000:
                raise ValueError('camera ROS header not compatible with Pi host epoch')
            ages.append(age_ms)
            camera[stamp] = (stamp-offset-pi_origin)*1e-9
            if len(camera) > 100000:
                raise ValueError('camera analysis sample limit exceeded: split recording')
    if offsets and max(offsets)-min(offsets) > 5_000_000:
        raise ValueError('Pi realtime/monotonic clock relation changed over 5ms')
    starts, ends, omega = [], [], []
    previous = None
    pose_file = run/'orb_live/live_poses.csv'
    with pose_file.open() as stream:
        for row in csv.DictReader(stream):
            stamp, state, map_id = int(row['stamp_ns']), int(row['state']), int(row['map_id'])
            if state != 2 or map_id < 0 or stamp not in camera:
                previous = None
                continue
            rotation = np.array([[float(row[f'm{i}{j}']) for j in range(3)] for i in range(3)])
            if (not np.isfinite(rotation).all() or not np.allclose(rotation.T@rotation, np.eye(3), atol=.003)
                    or not np.isclose(np.linalg.det(rotation), 1., atol=.003)):
                raise ValueError('invalid visual rotation')
            current = (camera[stamp], map_id, rotation)
            if previous is not None:
                dt = current[0]-previous[0]
                if current[1] == previous[1] and .025 <= dt <= max_dt_s:
                    angle = np.arccos(np.clip((np.trace(previous[2].T@rotation)-1)*.5, -1, 1))
                    starts.append(previous[0]); ends.append(current[0]); omega.append(angle/dt)
            previous = current
    return {'start': np.asarray(starts), 'end': np.asarray(ends), 'omega': np.asarray(omega),
            'clock_basis': 'legacy_IMU_host_clock_pair' if legacy else 'camera_callback_host_clock_pair',
            'legacy_camera_frames': legacy, 'camera_frames': len(camera),
            'camera_header_age_p50_p95_ms': np.percentile(ages, [50, 95]).tolist() if ages else [],
            'clock_sample_span_max_ns': int(max(spans, default=0)),
            'clock_domain_verified': False,
            'clock_assumption': 'ROS image header is a camera timestamp translated into Pi realtime epoch; hardware metadata not verified'}


def write_report(output: Path, result: dict) -> None:
    """기각 사유와 부호·원본 provenance를 남기고 비교 신호가 있으면 그림을 만든다."""
    atomic_json(output/'summary.json', result)
    correction = result.get('imu_time_correction_s')
    clock = result.get('clock_fit', {})
    lines = ['# 카메라–IMU 시간 정렬 진단', '', f'- 판정: `{result["status"]}`',
             f'- 이유: {", ".join(result.get("reasons", [])) or "품질 조건을 만족한 추정 후보"}',
             f'- IMU 시각에 더할 보정(s): {correction}',
             f'- MCU→Pi clock rate 차이(ppm): {clock.get("rate_difference_ppm")}',
             f'- 회전 속도 상관: {result.get("correlation")}', '',
             '**아직 검증된 하드웨어 동기화가 아니다. ORB 융합·자동 시각 교체에 사용하지 않는다.**', '',
             '- 부호: `aligned_imu_time = fitted_imu_pi_time + imu_time_correction_s`.',
             '- 양의 gyro query shift는 영상보다 늦은 IMU 모델 시각을 조회했다는 뜻이며, IMU 정렬 보정은 그 음수다.',
             '- MCU 시계 속도와 offset은 Pi 수신 시각으로 추정한다. SPI 지연의 절대값은 알 수 없다.',
             '- 회전 크기 비교는 축 변환에 덜 민감하지만 부호·장착 변환 검증을 대체하지 않는다.',
             '- 5ms 검색 격자는 정확도가 아니다. 15FPS 영상·평균 IMU·노출 시각 차이가 남아 있다.',
             '- 정지·반복 회전·부족한 겹침·IMU 공백·불안정한 시차는 거부한다.',
             '- 카메라 timestamp domain의 실제 metadata 검증 전까지 `imu_clock_verified=false`를 유지한다.',
             '- Pi와 노트북 수신 시각을 직접 빼서 latency로 쓰지 않는다.', '',
             '[전체 수치와 입력 hash](summary.json)']
    if 'lag_grid_s' in result:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
        times = np.asarray(result['comparison_visual_times_s'])
        axes[0].plot(times-times[0], result['visual_omega_rad_s'], label='Visual rotation rate', lw=1)
        axes[0].plot(times-times[0], result['matched_gyro_norm_rad_s'], label='IMU at candidate shift', lw=1)
        axes[0].set(xlabel='Time from first comparison interval (s)', ylabel='rad/s')
        axes[0].legend()
        axes[1].plot(np.asarray(result['lag_grid_s'])*1000, result['correlations'])
        axes[1].set(xlabel='Gyro query shift (ms); IMU time correction has opposite sign', ylabel='Correlation')
        for axis in axes: axis.grid(alpha=.2)
        fig.suptitle(f'Camera / external IMU alignment: {result["status"]} | NOT clock-verified')
        fig.savefig(output/'alignment.png', dpi=150)
        plt.close(fig)
        lines += ['', '[비교 그림](alignment.png)']
    (output/'README.md').write_text('\n'.join(lines)+'\n')


def analyze_run(run: Path, output: Path | None = None, max_visual_dt_s: float = .15) -> dict:
    """종료된 RUN을 변경하지 않고 새 폴더에 추정 후보 또는 명시적인 기각 보고서를 만든다."""
    run = Path(run).resolve()
    imu_summary = run/'imu_summary.json'
    if not imu_summary.is_file() or strict_json(imu_summary.read_text()).get('state') != 'stopped':
        raise ValueError('stop IMU recording before time alignment')
    output = Path(output).resolve() if output is not None else run/'imu_time_alignment'
    output.mkdir(parents=True, exist_ok=False)
    result = {'status': 'rejected', 'reasons': [], 'source_run': str(run),
              'imu_clock_verified': False, 'orb_fused': False, 'imu_time_correction_s': None,
              'input_sha256': {}}
    for relative in ('imu.jsonl', 'frames.jsonl', 'orb_live/live_poses.csv', 'capture_calibration.json', 'imu_calibration.json'):
        path = run/relative
        if path.is_file(): result['input_sha256'][relative] = file_sha256(path)
    try:
        imu = imu_samples(run)
        clock = fit_clock(imu['mcu_ms'], imu['pi_ns'])
        unwrapped = clock.pop('unwrapped_ms')
        times = (unwrapped-clock['mcu_origin_unwrapped_ms'])*.001*clock['scale']+clock['offset_s']
        offsets = imu['realtime_ns']-imu['pi_ns']
        if int(offsets.max())-int(offsets.min()) > 5_000_000:
            raise ValueError('Pi realtime/monotonic clock relation changed over 5ms')
        fallback = int(np.sort(offsets)[len(offsets)//2])
        visual = visual_samples(run, clock['pi_origin_monotonic_ns'], fallback, max_visual_dt_s)
        result.update(estimate_offset(times, np.linalg.norm(imu['gyro'], axis=1),
                                      visual['start'], visual['end'], visual['omega']))
        result.update(clock_fit=clock, camera_clock={k: v for k, v in visual.items() if k not in ('start', 'end', 'omega')},
                      accepted_imu_samples=len(times), imu_acceptance_counts=imu['counts'],
                      unique_imu_receipt_hz=(len(times)-1)*1e9/int(imu['pi_ns'][-1]-imu['pi_ns'][0]),
                      imu_n_median=(float(np.median([v for v in imu['imu_n'] if isinstance(v, (int, float))]))
                                    if any(isinstance(v, (int, float)) for v in imu['imu_n']) else None),
                      gyro_norm_std_rad_s=float(np.std(np.linalg.norm(imu['gyro'], axis=1))))
        if clock['receipt_residual_trend_suspect']:
            result.update(status='rejected', imu_time_correction_s=None,
                          reasons=[*result['reasons'], 'clock_model_unstable'])
        if result['status'] == 'estimated':
            with (output/'imu_alignment_candidate.csv').open('x') as stream:
                writer = csv.writer(stream)
                writer.writerow(['mcu_unwrapped_ms', 'fitted_pi_monotonic_ns', 'aligned_pi_monotonic_ns',
                                 'gx_rad_s', 'gy_rad_s', 'gz_rad_s', 'ax_m_s2', 'ay_m_s2', 'az_m_s2', 'clock_verified'])
                for i, t in enumerate(times):
                    fitted = clock['pi_origin_monotonic_ns']+round(t*1e9)
                    aligned = clock['pi_origin_monotonic_ns']+round((t+result['imu_time_correction_s'])*1e9)
                    writer.writerow([int(unwrapped[i]), fitted, aligned, *imu['gyro'][i], *imu['acc'][i], False])
    except (ValueError, KeyError, TypeError, OSError) as error:
        result.update(status='rejected', reasons=[str(error)], imu_time_correction_s=None)
    for relative, checksum in result['input_sha256'].items():
        if file_sha256(run/relative) != checksum:
            raise RuntimeError('input changed during analysis')
    result['raw_preserved'] = True
    result['output_dir'] = str(output)
    write_report(output, result)
    return result


def main() -> None:
    """RUN을 명시적으로 지정해 시간 정렬 진단을 실행한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-visual-dt', type=float, default=.15,
                        help='연속 영상 pose 간격 상한(s); 6 FPS 비교 실험에서만 .2로 올린다')
    args = parser.parse_args()
    result = analyze_run(args.run, args.output, args.max_visual_dt)
    print(json.dumps({k: v for k, v in result.items() if k not in
                     ('lag_grid_s', 'correlations', 'comparison_visual_times_s', 'visual_omega_rad_s', 'matched_gyro_norm_rad_s')},
                     indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
