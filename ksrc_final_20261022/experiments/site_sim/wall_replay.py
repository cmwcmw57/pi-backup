# ================================================================
# wall_replay.py
# 설명: 기록된 IMU·ORB 이벤트와 스테레오 영상으로 온라인 추정을 재생하면서 STL 벽 거리(R)·직선(L)·모서리(C) 보정을 켜고 끄며 정답 대비 오차를 비교한다.
# 사용법:
#   python -m experiments.site_sim.wall_replay RUN_DIR --variants V,R,R+L,R+L+C --noise 0 1 2
#   (RUN_DIR에 estimator_calibration.json, estimator_events.jsonl, stereo/calibration.json, 일부 stereo 영상이 있어야 한다)
# ================================================================
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from experiments.localization.online import OnlineEstimator
from experiments.localization.wall_range import (Camera, Scene, add_depth_noise, base_state, column_scan, corner_update,
                                                 detect_corners, fit_lines, level_points, line_update, range_update,
                                                 stereo_depth, detect_wall_top, wall_top_update)


def sgbm() -> 'cv2.StereoSGBM':
    """기존 wall_range 실험과 같은 수동 스테레오 매개변수."""
    return cv2.StereoSGBM_create(minDisparity=0, numDisparities=96, blockSize=7, P1=8*49, P2=32*49,
                                 uniquenessRatio=8, speckleWindowSize=80, speckleRange=2, disp12MaxDiff=1)


def load_depths(run: Path, camera: Camera) -> dict[int, np.ndarray]:
    """영상이 남은 프레임마다 SGBM 깊이를 만든다(float16로 보관)."""
    matcher, depths = sgbm(), {}
    base = run/'stereo/mav0'
    for left in sorted((base/'cam0/data').glob('*.png')):
        right = base/'cam1/data'/left.name
        if not right.is_file():
            continue
        z = stereo_depth(matcher, cv2.imread(str(left), 0), cv2.imread(str(right), 0), camera)
        depths[int(left.stem)] = z.astype(np.float16)
    return depths


def synthetic_scan(scene: Scene, gt_base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """정답 pose에서 STL을 쏴서 만든 열별 벽 깊이(진단용, 실제 센서가 아니다)."""
    z = scene.depth(gt_base, scene.fine, True).reshape(scene.fine_shape)
    fine = scene.fine.reshape(scene.fine_shape[0], scene.fine_shape[1], 3)
    rays, values = [], []
    for column in range(scene.fine_shape[1]):
        rows = np.flatnonzero(np.isfinite(z[:, column]))
        if len(rows) >= 5:
            mid = rows[len(rows)//2]
            rays.append(fine[mid, column])
            values.append(z[mid, column])
    return np.array(rays).reshape(-1, 3), np.array(values)


def apply_wall(est: OnlineEstimator, scene: Scene, depth: np.ndarray | None, flags: set[str], rng: np.random.Generator,
               noise: float, counts: dict, gt_base: np.ndarray | None = None, bias: float = 1.03,
               gray: np.ndarray | None = None) -> None:
    """현재 필터 pose에서 깊이 영상(또는 gt_base가 주어지면 합성 깊이)으로 R·L·C 중 켠 보정을 순서대로 적용한다."""
    if 'T' in flags:
        columns = np.arange(8, 640, 8.)
        info = wall_top_update(est.filter, scene, est.imu_from_base, detect_wall_top(gray, columns), columns)
        counts['T'][0] += 1
        counts['T'][1] += bool(info['accepted'])
    if not flags-{'T'}:
        return
    base, _ = base_state(est.filter, est.imu_from_base)
    rays, z = synthetic_scan(scene, gt_base) if gt_base is not None else column_scan(scene, base, depth.astype(np.float32), bias=bias)
    if len(z) < 6:
        return
    if noise > 0:
        z = add_depth_noise(z, rng, scale=noise)
    if 'R' in flags:
        info = range_update(est.filter, scene, est.imu_from_base, z, rays)
        counts['R'][0] += 1
        counts['R'][1] += bool(info['accepted'])
    if 'L' in flags or 'C' in flags:
        base, _ = base_state(est.filter, est.imu_from_base)
        points = level_points(scene, base, rays, z) if len(z) >= 10 else np.zeros((0, 2))
        lines = fit_lines(points, rng) if len(points) >= 10 else []
        if 'L' in flags:
            info = line_update(est.filter, scene, est.imu_from_base, lines)
            counts['L'][0] += 1
            counts['L'][1] += bool(info['accepted'])
        if 'C' in flags:
            info = corner_update(est.filter, scene, est.imu_from_base, detect_corners(lines))
            counts['C'][0] += 1
            counts['C'][1] += bool(info['accepted'])


def replay(run: Path, flags: set[str], noise: float, depths: dict[int, np.ndarray], camera: Camera, seed: int = 0,
           synthetic: bool = False, bias: float = 1.03, mesh: object | None = None) -> dict:
    """한 조합을 재생해 오차 요약과 시간별 오차를 돌려준다."""
    config = json.loads((run/'estimator_calibration.json').read_text())
    est = OnlineEstimator(np.asarray(config['start_world_from_base']), config['calibration'], lag_s=config['render_lag_s'],
                          adaptive_visual_noise=config.get('adaptive_visual_noise', False))
    scene, rng = Scene(camera, mesh=mesh), np.random.default_rng(seed)
    counts = {key: [0, 0] for key in 'RLCT'}
    times, errors = [], []
    ground_truth = []
    if synthetic:
        from scipy.spatial.transform import Rotation
        for line in (run/'estimator_events.jsonl').read_text().splitlines():
            event = json.loads(line)
            if event['kind'] == 'pose':
                q = event['quaternion_wxyz']
                pose = np.eye(4)
                pose[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
                pose[:3, 3] = event['position_m']
                ground_truth.append((event['stamp_s'], pose))
        gt_times = np.array([g[0] for g in ground_truth])
    with (run/'estimator_events.jsonl').open() as source:
        for line in source:
            event = json.loads(line)
            t = event['stamp_s']
            if event['kind'] == 'imu':
                est.on_imu(t, event['dt_s'], np.asarray(event['gyro']), np.asarray(event['force']))
            elif event['kind'] == 'frame':
                est.on_frame(t, event['orb_state'], event['map_id'], np.asarray(event['world_from_camera']))
                depth = depths.get(round(t*1e9))
                if depth is not None and flags:
                    gt_base = None
                    if synthetic:
                        gt_base = ground_truth[int(np.argmin(abs(gt_times-(t-config['render_lag_s']))))][1]
                    gray = cv2.imread(str(run/'stereo/mav0/cam0/data'/f'{round(t*1e9)}.png'), 0) if 'T' in flags else None
                    apply_wall(est, scene, depth, flags, rng, noise, counts, gt_base, bias, gray)
            else:
                pose, _, _, _ = est.pose(t)
                times.append(t)
                errors.append(np.linalg.norm(pose[:2, 3]-np.asarray(event['position_m'][:2]))*100)
    t, e = np.array(times), np.array(errors)
    return {'rmse_cm': float(np.sqrt((e**2).mean())), 'max_cm': float(e.max()), 'mean_cm': float(e.mean()),
            'last_cm': float(e[-1]), 'within_5cm': float((e <= 5).mean()), 'updates': counts,
            'by_40s': [float(e[(t >= lo) & (t < lo+40)].mean()) for lo in range(0, int(t[-1])+1, 40) if ((t >= lo) & (t < lo+40)).any()]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--variants', default='V,R,R+L,R+L+C', help='V는 벽 보정 없음')
    parser.add_argument('--noise', type=float, nargs='+', default=[0.], help='깊이에 더할 잡음 배율(0이면 SGBM 자체 잡음만)')
    parser.add_argument('--synthetic', action='store_true', help='스테레오 대신 정답 pose의 합성 벽 깊이(진단용)')
    parser.add_argument('--bias', type=float, default=1.03, help='스테레오 깊이에 곱하는 편향 보정(측정 비율 약 0.945 → 1.06)')
    parser.add_argument('--mesh', type=Path, default=None, help='경기장 전체 메쉬(.ply). 없으면 외벽 STL만 쓴다')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    camera = Camera.from_sim(json.loads((args.run/'stereo/calibration.json').read_text()))
    depths = load_depths(args.run, camera)
    mesh = None
    if args.mesh:
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(str(args.mesh))
    print('depth frames', len(depths), flush=True)
    report = {}
    for variant in args.variants.split(','):
        flags = set() if variant == 'V' else set(variant.split('+'))
        for noise in (args.noise if flags else [0.]):
            key = f'{variant} noise={noise:g}'
            report[key] = result = replay(args.run, flags, noise, depths, camera, synthetic=args.synthetic, bias=args.bias, mesh=mesh)
            print(f"{key:18s} rmse {result['rmse_cm']:5.1f} max {result['max_cm']:5.1f} mean {result['mean_cm']:5.1f} "
                  f"last {result['last_cm']:5.1f} <=5cm {result['within_5cm']*100:3.0f}% updates {result['updates']} "
                  f"by40s {[round(v, 1) for v in result['by_40s']]}", flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
