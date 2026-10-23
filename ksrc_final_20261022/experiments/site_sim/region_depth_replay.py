# ================================================================
# region_depth_replay.py
# 설명: 기록 stereo만으로 초기 위치 유무의3D 면 정합을 비교한다. GT 파일 접근은 차단한다.
# 사용법: python -m experiments.site_sim.region_depth_replay --stereo STEREO --mesh MESH --output OUT
# ================================================================
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import open3d as o3d

from experiments.localization.region_depth import DepthMap, Observation, observation, visual_motion
from experiments.localization.wall_range import Camera, stereo_depth
from experiments.site_sim.wall_replay import sgbm


def block_truth(event: str, args: tuple) -> None:
    """추정 과정에서 평가용 파일·주행 경로를 실수로 읽지 않도록 한다."""
    if event == 'open' and isinstance(args[0], (str, bytes)):
        name = Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).name
        if name in {'gt.jsonl', 'telemetry.jsonl', 'summary.json', 'mission.json', 'estimator_events.jsonl'}:
            raise RuntimeError('evaluation/controller input forbidden during estimation: '+name)


def sha(path: Path) -> str:
    """입력·실험 소스 hash를 계산한다."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(stereo: Path, camera: Camera, output: Path, limit: int) -> tuple[dict, dict]:
    """현재 좌우 영상에서만 관측과 VO를 만들고 재사용 가능한 배열로 저장한다."""
    selected = sorted((stereo/'mav0/cam0/data').glob('*.png'), key=lambda p: int(p.stem))[::5]
    last = max((stereo/'mav0/cam0/data').glob('*.png'), key=lambda p: int(p.stem))
    if selected[-1] != last:
        selected.append(last)
    if limit:
        selected = selected[:3*limit-2]
    source_files = [Path(__file__), Path(__file__).parents[1]/'localization/region_depth.py']
    signature = dict(stereo=str(stereo), source={str(p): sha(p) for p in source_files},
                     calibration_sha256=sha(stereo/'calibration.json'), frames=[p.name for p in selected])
    cache, metadata_path = output/'sensors.npz', output/'sensors.json'
    if cache.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata['signature'] == signature:
            with np.load(cache) as loaded:
                return {name: loaded[name] for name in loaded.files}, metadata
    cv2.setNumThreads(2);cv2.setRNGSeed(17)
    matcher, orb = sgbm(), cv2.ORB_create(nfeatures=2000, fastThreshold=8)
    arrays = dict(stamp_ns=[], points=[], rays=[], depth=[], count=[], movement=[], vo_ok=[], inliers=[], matches=[])
    previous, inputs = None, {}
    accumulated, accepted_steps, minimum_inliers, total_matches = np.eye(4), True, 10000, 0
    for index, path in enumerate(selected):
        right_path = stereo/'mav0/cam1/data'/path.name
        gray, right = cv2.imread(str(path), 0), cv2.imread(str(right_path), 0)
        assert gray is not None and right is not None
        depth = stereo_depth(matcher, gray, right, camera)
        mask = (np.isfinite(depth)&(depth >= .25)&(depth <= 2.)&(gray > 25)&(gray < 200)).astype(np.uint8)*255
        features, descriptors = orb.detectAndCompute(gray, cv2.erode(mask, np.ones((3, 3), np.uint8)))
        current = dict(pixels=np.array([key.pt for key in features]), descriptors=descriptors, depth=depth, stamp=int(path.stem)*1e-9)
        movement, info = (np.eye(4), dict(accepted=False, inliers=0, matches=0)) if previous is None else visual_motion(previous, current, camera, current['stamp']-previous['stamp'])
        previous = current
        accumulated = accumulated@movement
        accepted_steps &= info['accepted'];minimum_inliers = min(minimum_inliers, info['inliers']);total_matches += info['matches']
        inputs[str(path)] = sha(path);inputs[str(right_path)] = sha(right_path)
        if index%3 != 0 and index != len(selected)-1:
            continue
        observed = observation(gray, depth, camera)
        arrays['stamp_ns'].append(int(path.stem));arrays['movement'].append(accumulated)
        arrays['count'].append(len(observed.points));arrays['vo_ok'].append(accepted_steps)
        arrays['inliers'].append(minimum_inliers);arrays['matches'].append(total_matches)
        accumulated, accepted_steps, minimum_inliers, total_matches = np.eye(4), True, 10000, 0
        for name, values in [('points', observed.points), ('rays', observed.rays), ('depth', observed.depth)]:
            padded = np.zeros((1600, 3) if name != 'depth' else (1600,), np.float32)
            padded[:len(values)] = values;arrays[name].append(padded)
        if index%60 == 0:
            print('sensor', index, '/', len(selected), 'points', len(observed.points), 'VO', info, flush=True)
    arrays = {name: np.array(value) for name, value in arrays.items()}
    np.savez_compressed(cache, **arrays)
    metadata = dict(signature=signature, input_sha256=inputs, cache_sha256=sha(cache),
                    observations='current stereo only; no map ROI; intensity (25,200), depth [0.25,2]m',
                    imu_used=False, gt_used=False)
    metadata_path.write_text(json.dumps(metadata, indent=2)+'\n')
    return arrays, metadata


def serialized(hypotheses: list[dict]) -> list[dict]:
    """pose 배열을 그대로 보존하며 GT로 후보를 고르지 않는다."""
    return [{key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in h.items()} for h in hypotheses]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--stereo', type=Path, required=True);parser.add_argument('--mesh', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True);parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--initial-xy', type=float, nargs=2, default=[2.9, 1.2])
    parser.add_argument('--initial-yaw-deg', type=float, default=0.)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    sys.addaudithook(block_truth)
    args.output.mkdir(parents=True, exist_ok=True)
    camera = Camera.from_sim(json.loads((args.stereo/'calibration.json').read_text()))
    data, metadata = prepare(args.stereo, camera, args.output, args.limit)
    if args.prepare_only:
        print('prepared', len(data['stamp_ns']), 'frames', flush=True);return
    world = DepthMap(o3d.io.read_triangle_mesh(str(args.mesh)), camera)
    hypotheses = {key: [] for key in ('none', 'xy', 'xy_yaw')}
    xy, yaw = np.array(args.initial_xy), np.deg2rad(args.initial_yaw_deg)
    vo_pose = world.seed_poses(xy, yaw)[0]
    start = time.monotonic()
    with (args.output/'predictions.jsonl').open('w') as stream, (args.output/'global_restarts.jsonl').open('w') as restarts:
        for index, stamp in enumerate(data['stamp_ns']):
            count = int(data['count'][index])
            observed = Observation(data['points'][index, :count].astype(float), data['rays'][index, :count].astype(float), data['depth'][index, :count].astype(float))
            global_candidates = None
            if index%10 == 0 or index == len(data['stamp_ns'])-1:
                global_candidates = world.initialize(observed)
                restarts.write(json.dumps(dict(stamp_ns=int(stamp), candidates=serialized(global_candidates)), allow_nan=False)+'\n');restarts.flush()
            for key in hypotheses:
                if not hypotheses[key]:
                    hypotheses[key] = global_candidates if key == 'none' and global_candidates is not None else world.initialize(observed, None if key == 'none' else xy, yaw if key == 'xy_yaw' else None)
                else:
                    hypotheses[key] = world.track(hypotheses[key], data['movement'][index], observed)
            if index:
                vo_pose = vo_pose@data['movement'][index]
            row = dict(stamp_ns=int(stamp), variants={key: serialized(value) for key, value in hypotheses.items()},
                       vo_only_pose=vo_pose.tolist(), vo_accepted=bool(data['vo_ok'][index]),
                       vo_inliers=int(data['inliers'][index]), observed_points=count)
            stream.write(json.dumps(row, allow_nan=False)+'\n');stream.flush()
            if index%10 == 0:
                print('estimate', index, '/', len(data['stamp_ns']), 'elapsed', round(time.monotonic()-start, 1),
                      {key: None if not h else [round(float(v), 3) for v in [*h[0]['pose'][:2, 3], h[0]['score']]] for key, h in hypotheses.items()}, flush=True)
    manifest = dict(gt_access_blocked=True, truth_or_mission_files_read=False, imu_used=False,
                    sensor_metadata=metadata, source_sha256={str(p): sha(p) for p in [Path(__file__), Path(__file__).parents[1]/'localization/region_depth.py']},
                    mesh_sha256=sha(args.mesh), initial_prior=dict(xy=args.initial_xy, yaw_deg=args.initial_yaw_deg, applied='initialization only'),
                    predictions_sha256=sha(args.output/'predictions.jsonl'), restarts_sha256=sha(args.output/'global_restarts.jsonl'),
                    frame_count=len(data['stamp_ns']), elapsed_estimation_s=time.monotonic()-start)
    (args.output/'estimation_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('estimation complete, no GT read', len(data['stamp_ns']), flush=True)


if __name__ == '__main__':
    main()
