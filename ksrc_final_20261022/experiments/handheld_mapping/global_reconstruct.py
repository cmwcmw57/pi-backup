# ================================================================
# global_reconstruct.py
# 설명: 원본 ORB pose를 보존하며 depth fragment graph를 최적화하고 별도 TSDF를 만든다.
# 사용법: python -m experiments.handheld_mapping.global_reconstruct RUN OUTPUT --map-id 0
# ================================================================
import argparse
from bisect import bisect_right
import csv
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import resource
import shutil
import time
from typing import Callable

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation, Slerp

from experiments.handheld_mapping.recording import matched_depths, ordered_keyframes
from experiments.handheld_mapping.reconstruct import pose_matrix, reconstruct, transform
from experiments.handheld_mapping.registration import REGISTRATION_SETTINGS, correction_size, register_clouds
from experiments.handheld_mapping.storage import atomic_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _stamp(row: dict) -> int:
    return int((Decimal(row['source_stamp'])*1_000_000_000).to_integral_value())


def _depth_calibration(run: Path) -> tuple[dict, np.ndarray, o3d.camera.PinholeCameraIntrinsic]:
    try:
        info = json.loads((run/'capture_calibration.json').read_text())['depth']
        scale, width, height = float(info['depth_scale_m']), int(info['width']), int(info['height'])
        k = np.asarray(info['k'], dtype=float).reshape(3, 3)
        extrinsic = transform(info['T_left_depth'])
        if (not math.isfinite(scale) or not 0 < scale < .1 or width <= 0 or height <= 0
                or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0
                or not np.allclose(k[2], [0, 0, 1]) or not np.allclose(k[[0, 1], [1, 0]], 0)
                or any(abs(float(v)) > 1e-8 for v in info.get('d', []))):
            raise ValueError('unsupported depth calibration')
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, k[0, 0], k[1, 1], k[0, 2], k[1, 2])
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError('depth calibration required') from error
    return info, extrinsic, intrinsic


def _depth_cloud(run: Path, frame: dict, info: dict, intrinsic, depth_max: float,
                 voxel_size: float) -> tuple[o3d.geometry.PointCloud | None, Path]:
    path = (run/frame['path']).resolve()
    if not path.is_relative_to(run/'raw/depth'):
        raise ValueError('invalid depth path outside source raw/depth')
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.dtype != np.uint16 or depth.shape != (int(info['height']), int(info['width'])):
        raise ValueError('depth file/profile mismatch')
    meters = depth*float(info['depth_scale_m'])
    depth[(meters < .25) | (meters > depth_max)] = 0
    if np.count_nonzero(depth) < 100:
        return None, path
    cloud = o3d.geometry.PointCloud.create_from_depth_image(
        o3d.geometry.Image(depth), intrinsic, depth_scale=1/float(info['depth_scale_m']), depth_trunc=depth_max)
    cloud = cloud.voxel_down_sample(voxel_size)
    if len(cloud.points) > 100_000:
        raise ValueError('frame voxel point limit exceeded: 100000')
    return cloud, path


def build_fragments(run: Path, keyfile: Path, map_id: int, output: Path, info: dict,
                    extrinsic: np.ndarray, intrinsic, depth_max: float, voxel_size: float,
                    fragment_size: int, overlap: int, progress: Callable | None = None) -> tuple[list[dict], dict]:
    """32개 이하 frame 참조와 축약된 cloud만 유지하고 원본 pose는 순차 복사한다."""
    folder = output/'fragments'
    folder.mkdir()
    fragments: list[dict] = []
    pending: list[tuple[dict, dict]] = []
    unflushed, previous, segment = 0, None, 0
    totals = {'source_keyframes': 0, 'matched_depth_frames': 0, 'valid_depth_frames': 0,
              'missing_depth_frames': 0, 'invalid_depth_frames': 0}
    depth_digest = hashlib.sha256()

    def flush() -> None:
        if len(pending) < 3:
            return
        if len(fragments) >= 256:
            raise ValueError('fragment node limit exceeded: 256')
        anchor = pose_matrix(pending[0][0])
        cloud = o3d.geometry.PointCloud()
        for row, frame in pending:
            frame_cloud, _ = _depth_cloud(run, frame, info, intrinsic, depth_max, voxel_size)
            if frame_cloud is None:
                raise ValueError('depth changed while constructing fragments')
            frame_cloud.transform(np.linalg.inv(anchor) @ pose_matrix(row) @ extrinsic)
            cloud += frame_cloud
            cloud = cloud.voxel_down_sample(voxel_size)
            if len(cloud.points) > 100_000:
                raise ValueError('fragment voxel point limit exceeded: 100000')
        index = len(fragments)
        relative = f'fragments/fragment_{index:04d}.ply'
        if not o3d.io.write_point_cloud(str(output/relative), cloud):
            raise OSError('fragment cloud write failed')
        fragments.append({'id': index, 'segment': segment, 'anchor_pose': anchor.tolist(),
                          'anchor_stamp_ns': _stamp(pending[0][0]), 'end_stamp_ns': _stamp(pending[-1][0]),
                          'keyframe_stamps_ns': [_stamp(row) for row, _ in pending],
                          'depth_paths': [frame['path'] for _, frame in pending],
                          'points': len(cloud.points), 'cloud': relative})
        atomic_json(folder/'manifest.json', {'fragments': fragments, 'settings': {
            'size': fragment_size, 'overlap': overlap, 'minimum_frames': 3, 'gap_ns': 1_000_000_000,
            'maximum_nodes': 256, 'maximum_voxel_points_per_frame_and_fragment': 100_000}})
        if progress:
            progress(f'지도 {map_id}: depth fragment {index+1}개 ({len(cloud.points)}점)')

    with (output/'baseline_poses'/keyfile.name).open('w') as source_poses, (output/'input_depths.jsonl').open('w') as depth_manifest:
        writer = None
        for row, frame, _ in matched_depths(run, keyfile, map_id, info['depth_frame']):
            pose_matrix(row)
            if writer is None:
                writer = csv.DictWriter(source_poses, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            totals['source_keyframes'] += 1
            if frame is None:
                totals['missing_depth_frames'] += 1
                continue
            totals['matched_depth_frames'] += 1
            cloud, path = _depth_cloud(run, frame, info, intrinsic, depth_max, voxel_size)
            record = {'keyframe_stamp_ns': _stamp(row), 'depth_stamp_ns': frame['stamp_ns'],
                      'path': frame['path'], 'sha256': _sha256(path), 'valid_depth': cloud is not None}
            encoded = json.dumps(record, sort_keys=True)+'\n'
            depth_manifest.write(encoded)
            depth_digest.update(encoded.encode())
            if cloud is None:
                totals['invalid_depth_frames'] += 1
                continue
            totals['valid_depth_frames'] += 1
            stamp = _stamp(row)
            if previous is not None and stamp-previous > 1_000_000_000:
                if unflushed:
                    flush()
                pending, unflushed = [], 0
                segment += 1
            previous = stamp
            pending.append((row, frame))
            unflushed += 1
            if len(pending) == fragment_size:
                flush()
                pending = pending[-overlap:] if overlap else []
                unflushed = 0
        if unflushed:
            flush()
    if not (folder/'manifest.json').exists():
        atomic_json(folder/'manifest.json', {'fragments': [], 'reason': 'fewer_than_three_valid_frames'})
    totals['input_depth_manifest_sha256'] = depth_digest.hexdigest()
    return fragments, totals


def optimize_fragments(fragments: list[dict], output: Path, voxel_size: float = .02,
                       progress: Callable | None = None) -> tuple[list[np.ndarray], dict]:
    """ORB prior와 통과한 depth edge를 구분하고 첫 fragment를 고정한다."""
    initial = [transform(fragment['anchor_pose']) for fragment in fragments]
    graph = o3d.pipelines.registration.PoseGraph()
    for pose in initial:
        graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(pose))
    edges, accepted, loops = [], 0, 0
    cloud_cache: dict[int, o3d.geometry.PointCloud] = {}

    def cloud(index: int) -> o3d.geometry.PointCloud:
        if index not in cloud_cache:
            if len(cloud_cache) >= 2:
                cloud_cache.pop(next(iter(cloud_cache)))
            cloud_cache[index] = o3d.io.read_point_cloud(str(output/fragments[index]['cloud']))
        return cloud_cache[index]

    def add_edge(source: int, target: int, measurement: np.ndarray, information: np.ndarray,
                 kind: str, uncertain: bool) -> None:
        graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(source, target, measurement, information, uncertain))
        edges.append({'source': source, 'target': target, 'transformation': measurement.tolist(),
                      'information': information.tolist(), 'kind': kind, 'uncertain': uncertain,
                      'information_semantics': 'heuristic weak continuity weight' if kind == 'orb_prior'
                      else REGISTRATION_SETTINGS['information_semantics']})

    with (output/'registration.jsonl').open('w') as diagnostics:
        loop_degrees = [0]*len(fragments)
        for source in range(len(fragments)):
            candidates = []
            if source+1 < len(fragments):
                target = source+1
                prior = np.linalg.inv(initial[target]) @ initial[source]
                add_edge(source, target, prior, np.eye(6)*.001, 'orb_prior', False)
                if fragments[source].get('segment', 0) == fragments[target].get('segment', 0):
                    candidates.append(target)
            nearby = sorted((target for target in range(source+3, len(fragments))
                             if fragments[source].get('segment', 0) == fragments[target].get('segment', 0)
                             and np.linalg.norm(initial[source][:3, 3]-initial[target][:3, 3]) <= 1.),
                            key=lambda target: np.linalg.norm(initial[source][:3, 3]-initial[target][:3, 3]))
            for target in nearby:
                if loop_degrees[source] >= 8:
                    break
                if loop_degrees[target] < 8:
                    candidates.append(target)
                    loop_degrees[source] += 1
                    loop_degrees[target] += 1
            for target in candidates:
                prior = np.linalg.inv(initial[target]) @ initial[source]
                report = register_clouds(cloud(source), cloud(target), prior, voxel_size)
                report.update(source=source, target=target, kind='adjacent_depth' if target == source+1 else 'loop_depth')
                diagnostics.write(json.dumps(report, allow_nan=False)+'\n')
                diagnostics.flush()
                if report['accepted']:
                    add_edge(source, target, np.asarray(report['transformation']), np.asarray(report['information']),
                             report['kind'], target != source+1)
                    accepted += 1
                    loops += target != source+1
                if progress:
                    progress(f'depth 정합 {source}→{target}: {report["reason"]}')
    report = {'optimization_status': 'insufficient_fragments' if len(fragments) < 2 else 'prior_only',
              'accepted_depth_edges': accepted, 'loop_closure_used': bool(loops), 'accepted_loop_edges': loops,
              'candidate_search': 'prior-nearby <=1m, fragment gap >=3, maximum 8 loops per node'}
    final, proposed = initial, initial
    if accepted and len(fragments) > 1:
        registration = o3d.pipelines.registration
        registration.global_optimization(
            graph, registration.GlobalOptimizationLevenbergMarquardt(),
            registration.GlobalOptimizationConvergenceCriteria(),
            registration.GlobalOptimizationOption(max_correspondence_distance=voxel_size*1.5,
                                                   edge_prune_threshold=.25, reference_node=0))
        proposed = [np.asarray(node.pose).copy() for node in graph.nodes]
        node_corrections = []
        for index, (old, new) in enumerate(zip(initial, proposed)):
            try:
                transform(new)
                translation, angle = correction_size(old, new)
            except ValueError:
                node_corrections.append({'node': index, 'valid_se3': False,
                                         'translation_correction_m': None, 'rotation_correction_deg': None})
            else:
                node_corrections.append({'node': index, 'valid_se3': True,
                                         'translation_correction_m': translation, 'rotation_correction_deg': angle,
                                         'exceeds_prior_gate': translation > .5 or angle > 20})
        valid = [item for item in node_corrections if item['valid_se3']]
        report.update(node_corrections=node_corrections,
                      maximum_proposed_translation_correction_m=max((item['translation_correction_m'] for item in valid), default=0.),
                      maximum_proposed_rotation_correction_deg=max((item['rotation_correction_deg'] for item in valid), default=0.))
        try:
            for old, new in zip(initial, proposed):
                transform(new)
                translation, angle = correction_size(old, new)
                if translation > .5 or angle > 20:
                    raise ValueError('node correction exceeds prior gate')
            if not np.allclose(proposed[0], initial[0], atol=1e-6):
                raise ValueError('first node anchor changed')
        except ValueError as error:
            report.update(optimization_status='optimization_rejected', optimization_error=str(error))
        else:
            final = proposed
            report['optimization_status'] = 'optimized_unverified'
    atomic_json(output/'pose_graph.json', {'initial_nodes': [pose.tolist() for pose in initial],
                'final_nodes': [pose.tolist() for pose in final], 'edges': edges,
                'proposed_nodes': [pose.tolist() if np.isfinite(pose).all() else None for pose in proposed],
                'reference_node': 0, **report})
    return final, report


def interpolate_correction(stamps: list[int], corrections: list[np.ndarray], stamp: int) -> np.ndarray:
    if not stamps:
        return np.eye(4)
    index = bisect_right(stamps, stamp)-1
    if index < 0:
        return corrections[0].copy()
    if index >= len(stamps)-1:
        return corrections[-1].copy()
    fraction = (stamp-stamps[index])/(stamps[index+1]-stamps[index])
    result = np.eye(4)
    result[:3, :3] = Slerp([0., 1.], Rotation.from_matrix(np.stack([
        corrections[index][:3, :3], corrections[index+1][:3, :3]])))(fraction).as_matrix()
    result[:3, 3] = (1-fraction)*corrections[index][:3, 3]+fraction*corrections[index+1][:3, 3]
    return result


def _write_corrected_keyframes(source: Path, target: Path, map_id: int, fragments: list[dict],
                               poses: list[np.ndarray]) -> int:
    stamps = [fragment['anchor_stamp_ns'] for fragment in fragments]
    corrections = [pose @ np.linalg.inv(transform(fragment['anchor_pose'])) for pose, fragment in zip(poses, fragments)]
    count = 0
    with target.open('w') as stream:
        writer = None
        for stamp, row in ordered_keyframes(source, map_id):
            corrected = interpolate_correction(stamps, corrections, stamp) @ pose_matrix(row)
            values = [*corrected[:3, 3], *Rotation.from_matrix(corrected[:3, :3]).as_quat()]
            row.update({key: format(float(value), '.17g') for key, value in zip(('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'), values)})
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            count += 1
    return count


def run_global(run: Path, output: Path, backend: str = 'orb_offline', map_id: int = 0,
               depth_max: float = 1., voxel_size: float = .02, color: bool = False,
               progress: Callable | None = None, *, fragment_size: int = 32,
               fragment_overlap: int = 8) -> dict:
    run, output = Path(run).resolve(), Path(output).resolve()
    if backend not in ('orb', 'orb_offline', 'orb_inertial') or type(map_id) is not int or map_id < 0:
        raise ValueError('explicit source backend and map_id required')
    if (not math.isfinite(voxel_size) or not .002 <= voxel_size <= .1
            or not math.isfinite(depth_max) or not .3 < depth_max <= 10
            or type(fragment_size) is not int or not 3 <= fragment_size <= 32
            or type(fragment_overlap) is not int or not 0 <= fragment_overlap < fragment_size):
        raise ValueError('invalid global reconstruction settings')
    if output.is_relative_to(run/'raw') or output.is_relative_to(run/backend) or output == run:
        raise ValueError('output must not replace source inputs')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {'status': 'processing', 'method': 'depth_fragment_pose_graph_TSDF', 'source_run': str(run),
              'backend': backend, 'map_id': map_id, 'paths_relative_to': 'summary',
              'voxel_size_m': voxel_size, 'sdf_trunc_m': 4*voxel_size, 'depth_min_m': .25, 'depth_max_m': depth_max,
              'color_requested': bool(color), 'accuracy_verified_on_real_camera': False,
              'gravity_alignment_verified': False, 'coordinate_system': 'source ORB map; gravity unverified',
              'baseline_summary': 'baseline/summary.json', 'optimized_summary': 'optimized/summary.json',
              'keyframes': 'keyframes.csv', 'registration_settings': REGISTRATION_SETTINGS,
              'optimization_status': 'not_started', 'maps': []}
    atomic_json(output/'summary.json', result)
    try:
        source = run/backend/f'map_{map_id}_keyframes.csv'
        if not source.is_file() or not source.resolve().is_relative_to(run):
            raise ValueError('explicit source map keyframes missing or outside run')
        if backend == 'orb_inertial':
            inertial = json.loads((run/backend/'summary.json').read_text())
            if inertial.get('orb_fused') is not True or map_id not in inertial.get('exported_imu_initialized_maps', []):
                raise ValueError('initialized inertial export required for global reconstruction')
        info, extrinsic, intrinsic = _depth_calibration(run)
        source_hashes = {str(path.relative_to(run)): _sha256(path) for path in (
            run/'capture_calibration.json', run/'frames.jsonl', source)}
        for folder in ('baseline_poses', 'optimized_poses'):
            (output/folder).mkdir()
            if backend == 'orb_inertial':
                shutil.copyfile(run/backend/'summary.json', output/folder/'summary.json')
        fragments, totals = build_fragments(run, source, map_id, output, info, extrinsic, intrinsic,
                                            depth_max, voxel_size, fragment_size, fragment_overlap, progress)
        result.update(totals, source_hashes=source_hashes, fragment_count=len(fragments))
        result['source_input_sha256'] = hashlib.sha256(json.dumps({
            'files': source_hashes, 'depths': totals['input_depth_manifest_sha256']},
            sort_keys=True).encode()).hexdigest()
        if not totals['valid_depth_frames']:
            raise ValueError('No matching depth/keyframes with valid depth')
        if progress:
            progress(f'지도 {map_id}: 동일 depth 기준 baseline 융합')
        baseline = reconstruct(run, voxel_size, depth_max, progress, backend,
                               keyframe_dir=output/'baseline_poses', output_dir=output/'baseline', color=color)
        result['baseline_available'] = True
        atomic_json(output/'summary.json', result)
        poses, optimization = optimize_fragments(fragments, output, voxel_size, progress)
        result.update(optimization)
        count = _write_corrected_keyframes(output/'baseline_poses'/source.name, output/'keyframes.csv', map_id, fragments, poses)
        if count != totals['source_keyframes']:
            raise ValueError('optimized keyframe timestamp/count mismatch')
        result['optimized_keyframes'] = count
        shutil.copyfile(output/'keyframes.csv', output/'optimized_poses'/source.name)
        if progress:
            progress(f'지도 {map_id}: 최적화 pose로 원본 depth 재융합')
        optimized = reconstruct(run, voxel_size, depth_max, progress, backend,
                                keyframe_dir=output/'optimized_poses', output_dir=output/'optimized', color=color)
        if baseline['maps'][0]['integrated_frames'] != optimized['maps'][0]['integrated_frames']:
            raise ValueError('baseline/optimized observation count mismatch')
        for relative, summary in (('baseline', baseline), ('optimized', optimized)):
            item = summary['maps'][0]
            result[relative] = dict(item, mesh=f'{relative}/{item["mesh"]}', cloud=f'{relative}/{item["cloud"]}')
        result['maps'] = [result['optimized']]
        result['integrated_frames'] = optimized['maps'][0]['integrated_frames']
        result['color_status'] = optimized.get('color_status', 'disabled')
        result['status'] = 'partial' if (baseline['status'] != 'complete' or optimized['status'] != 'complete'
                                         or result['optimization_status'] == 'optimization_rejected') else 'complete'
    except Exception as error:
        result.update(status='partial' if result.get('baseline_available') else 'failed', error=str(error))
        raise
    finally:
        result.update(elapsed_s=time.monotonic()-started, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                      peak_rss_scope='process lifetime, including Open3D native allocations')
        atomic_json(output/'summary.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='원본을 보존하는 depth fragment 전역 재구성')
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--backend', choices=('orb', 'orb_offline', 'orb_inertial'), default='orb_offline')
    parser.add_argument('--map-id', type=int, required=True)
    parser.add_argument('--depth-max', type=float, default=1.)
    parser.add_argument('--voxel-size', type=float, default=.02)
    parser.add_argument('--color', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run_global(args.run, args.output, args.backend, args.map_id,
                                args.depth_max, args.voxel_size, args.color, print), indent=2))


if __name__ == '__main__':
    main()
