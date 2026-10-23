# ================================================================
# region_navigation_replay.py
# 설명: 기존 sensor-only stereo cache로 비학습 구역 확률·통로 목표를 재생한다.
#       GT와 제어 기록을 읽지 않고 구동기·RL·실기에는 연결하지 않는다.
# 사용법: python -m experiments.site_sim.region_navigation_replay --help
# ================================================================
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
import open3d as o3d

from experiments.localization.region_depth import DepthMap, Observation
from experiments.localization.region_navigation import RegionGraph, RegionBelief
from experiments.localization.wall_range import Camera
from experiments.site_sim.region_depth_replay import block_truth, sha

PORTALS = Path(__file__).parents[1]/'localization/assets/gyusa_region_portals.json'


def grounded_poses(world: DepthMap, xy: np.ndarray, yaw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """지도 바닥 법선과 명시적인 nominal base 높이16cm로 pose 후보를 만든다."""
    rays = np.c_[xy, np.full(len(xy), 3.), np.zeros((len(xy), 2)), -np.ones(len(xy))]
    hit = world.scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))
    distance, normal = hit['t_hit'].numpy(), hit['primitive_normals'].numpy().astype(float)
    normal *= np.where(normal[:, 2:3] < 0, -1., 1.)
    valid = np.isfinite(distance)&(normal[:, 2] > .65)
    normal[~valid] = [0., 0., 1.]
    forward = np.c_[np.cos(yaw), np.sin(yaw), np.zeros(len(yaw))]
    forward -= normal*np.sum(normal*forward, axis=1)[:, None]
    forward /= np.maximum(np.linalg.norm(forward, axis=1)[:, None], 1e-9)
    poses = np.repeat(np.eye(4)[None], len(xy), axis=0)
    poses[:, :3, :3] = np.stack([forward, np.cross(normal, forward), normal], axis=2)
    poses[:, :2, 3], poses[:, 2, 3] = xy, np.where(valid, 3.-distance+.16, 0.)
    return poses, valid


def proposals(world: DepthMap, graph: RegionGraph, beam: list[dict], movement: np.ndarray,
              current: int) -> tuple[np.ndarray, np.ndarray]:
    """VO 예측 근처의 작은 XY/yaw 격자를 평가하며 전역 초기화는 하지 않는다."""
    offsets = np.array(list(itertools.product((-.12, 0., .12), (-.12, 0., .12),
                                             (-np.deg2rad(8.), 0., np.deg2rad(8.)))))
    xy, yaw, priors, measured = [], [], [], []
    allowed = {current}|graph.adjacent[current]
    for h in beam:
        predicted = h['pose']@movement
        angle = np.arctan2(predicted[1, 0], predicted[0, 0])
        for dx, dy, da in offsets:
            p = predicted[:2, 3]+[dx, dy]
            if not allowed.intersection(graph.labels(p, overlap_m=.125)):
                continue
            xy.append(p);yaw.append(angle+da)
            candidate = predicted.copy()
            rotation = np.array([[np.cos(da), -np.sin(da), 0.], [np.sin(da), np.cos(da), 0.], [0., 0., 1.]])
            candidate[:3, :3] = rotation@predicted[:3, :3]
            candidate[:2, 3] = p
            measured.append(candidate)
            priors.append(.5*((dx/.12)**2+(dy/.12)**2+(da/np.deg2rad(8.))**2)-np.log(max(h['weight'], 1e-12)))
    if not xy:
        return np.empty((0, 4, 4)), np.empty(0)
    _, valid = grounded_poses(world, np.asarray(xy), np.asarray(yaw))
    # 경사 경계에서는 바퀴가 여러 면에 걸치므로 이동 중 차체 자세를 지도 법선으로 바꾸지 않는다.
    return np.asarray(measured)[valid], np.asarray(priors)[valid]


def depth_posterior(scores: np.ndarray, priors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """절대 depth 잔차가 큰 후보를 버린 뒤 상대 가중치를 만든다(확률 보정값은 아님)."""
    good = np.isfinite(scores)&np.isfinite(priors)&(scores <= 2.5)
    if not good.any():
        return np.empty(0), np.empty(0)
    costs = np.where(good, scores/.6+priors, np.inf)
    weights = np.exp(-(costs-costs.min()));weights /= weights.sum()
    return costs, weights


def replay(args: argparse.Namespace) -> dict:
    """검증한 센서 cache만으로 추정하고 GT 평가와 별도 출력에 저장한다."""
    args.output.mkdir(parents=True, exist_ok=False)
    sys.addaudithook(block_truth)
    metadata = json.loads(args.cache.with_suffix('.json').read_text())
    if metadata.get('gt_used') is not False or metadata.get('imu_used') is not False:
        raise ValueError('stereo-only cache metadata required')
    if sha(args.cache) != metadata['cache_sha256']:
        raise ValueError('sensor cache hash mismatch')
    if sha(args.calibration) != metadata['signature']['calibration_sha256']:
        raise ValueError('cache camera calibration mismatch')
    document = json.loads(args.regions.read_text());portals = json.loads(PORTALS.read_text())
    mesh_hash = sha(args.mesh)
    if document['source_mesh_sha256'] != mesh_hash or portals['source_mesh_sha256'] != mesh_hash:
        raise ValueError('region/portal/mesh source mismatch')
    graph = RegionGraph(document['regions'], portals['portals'])
    if args.start_region not in graph.labels(np.array(args.initial_xy)):
        raise ValueError('initial XY must be in declared start region')
    camera = Camera.from_sim(json.loads(args.calibration.read_text()))
    world = DepthMap(o3d.io.read_triangle_mesh(str(args.mesh)), camera)
    pose, good = grounded_poses(world, np.array([args.initial_xy]), np.deg2rad([args.initial_yaw_deg]))
    if not good[0]:
        raise ValueError('start is outside traversable map surface')
    beam = [dict(pose=pose[0], weight=1.)]
    belief = RegionBelief(graph, args.start_region)
    transitions, missing, start_time = [], 0, time.monotonic()
    with np.load(args.cache) as data, (args.output/'states.jsonl').open('w') as stream:
        stamps = data['stamp_ns'][:args.limit or None]
        if len(stamps) == 0 or np.any(np.diff(stamps) <= 0):
            raise ValueError('nonempty increasing sensor times required')
        for index, stamp in enumerate(stamps):
            count = int(data['count'][index])
            observed = Observation(data['points'][index, :count], data['rays'][index, :count], data['depth'][index, :count])
            movement = np.eye(4) if index == 0 else data['movement'][index]
            if not np.isfinite(movement).all():
                raise ValueError('nonfinite VO movement')
            valid = count >= 100 and (index == 0 or bool(data['vo_ok'][index]))
            likelihood = {}
            minimum_score = None
            if valid:
                poses, prior = proposals(world, graph, beam, movement, belief.current)
                if len(poses):
                    scores = world.scores(poses, observed);minimum_score = float(scores.min())
                    costs, weights = depth_posterior(scores, prior)
                    if len(weights):
                        for p, weight in zip(poses, weights):
                            labels = graph.labels(p[:2, 3], overlap_m=.125)
                            for k in labels:
                                likelihood[k] = likelihood.get(k, 0.)+float(weight)/len(labels)
                        ids = np.flatnonzero(weights > 0)
                        selected = ids[world.distinct(poses[ids], costs[ids], 6, distance=.08)]
                        total = sum(weights[k] for k in selected)
                        beam = [dict(pose=poses[k], weight=float(weights[k]/total)) for k in selected]
                    else:
                        valid = False
                else:
                    valid = False
            if not valid:
                beam = [dict(pose=h['pose']@movement, weight=h['weight']) for h in beam]
                missing += 1
            pose = beam[0]['pose']
            state = belief.update(float(stamp)*1e-9, likelihood, pose[:2, 3], valid=valid)
            route = graph.route(belief.current, args.goal_region, args.max_slope_deg)
            guidance_valid = state['status'] == 'observed' and state['probabilities'][str(belief.current)] >= .55
            guidance_valid = guidance_valid and graph.distance(belief.current, pose[:2, 3]) <= .125
            target = graph.portal_target(*route[:2], pose[:2, 3]) if guidance_valid and len(route) >= 2 else None
            if state['transition']:
                transitions.append(dict(stamp_ns=int(stamp), regions=state['transition']))
            row = dict(stamp_ns=int(stamp), **state, pose=pose.tolist(), route=route,
                       goal_region=args.goal_region, portal_target_xy_m=None if target is None else target.tolist(),
                       portal_segment_xy_m=None if target is None else graph.portals[tuple(sorted(route[:2]))].tolist(),
                       observed_points=count, vo_ok=bool(data['vo_ok'][index]),
                       minimum_depth_score=minimum_score, guidance_valid=bool(guidance_valid),
                       proposal_only=True, command_emitted=False)
            stream.write(json.dumps(row, allow_nan=False)+'\n')
            if index%30 == 0:
                print('frame', index, '/', len(stamps), 'region', belief.current, 'route', route, flush=True)
    sources = [Path(__file__), Path(__file__).parents[1]/'localization/region_navigation.py',
               Path(__file__).parents[1]/'localization/region_depth.py', PORTALS]
    manifest = dict(frame_count=len(stamps), transitions=transitions, observation_missing_frames=missing,
                    initial_prior=dict(xy=args.initial_xy, yaw_deg=args.initial_yaw_deg, region=args.start_region, applied='once'),
                    goal_region=args.goal_region, max_slope_deg=args.max_slope_deg, learning_used=False,
                    imu_used=False, truth_files_read=False, truth_access_guard=True,
                    lower_controller_modified=False, closed_loop=False, proposal_only=True,
                    method='local XY/yaw beam + stereo depth raycast + region Bayes filter; no ICP',
                    heuristics=dict(xy_step_m=.12, yaw_step_deg=8., depth_temperature=.6, beam_size=6,
                                    maximum_depth_score=2.5, boundary_overlap_m=.125,
                                    transition_confirmations=2, portal_clearance_m=.25),
                    source_sha256={str(p): sha(p) for p in sources},
                    input_sha256={str(p): sha(p) for p in [args.cache, args.cache.with_suffix('.json'), args.regions, args.calibration, args.mesh]},
                    output_sha256=sha(args.output/'states.jsonl'), elapsed_s=time.monotonic()-start_time)
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def main() -> None:
    """시뮬 기록 재생 CLI이며 실기 통신 옵션은 없다."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'regions', 'mesh', 'calibration', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--initial-xy', type=float, nargs=2, default=[2.9, 1.2])
    parser.add_argument('--initial-yaw-deg', type=float, default=0.)
    parser.add_argument('--start-region', type=int, default=1)
    parser.add_argument('--goal-region', type=int, default=4)
    parser.add_argument('--max-slope-deg', type=float, default=20.)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(replay(args), indent=2))


if __name__ == '__main__':
    main()
