# ================================================================
# terrain_replay.py
# 설명: 최종 map 하나의 depth를 한 장씩 재생해 중력 정렬 지형 층과 costmap을 저장한다.
# 사용법: .venv-mapping/bin/python -m experiments.handheld_mapping.terrain_replay --demo --output OUTPUT
# ================================================================
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.handheld_mapping.costmap import RoverLimits, build_costmap
from experiments.handheld_mapping.reconstruct import pose_matrix, transform
from experiments.handheld_mapping.storage import atomic_json
from experiments.handheld_mapping.terrain import GridSpec, TerrainGrid
from experiments.handheld_mapping.depth_filter import DepthFilter, filtered_depths
from experiments.handheld_mapping.ground import GroundGrid


def project_depth(depth: np.ndarray, info: dict, grid_from_depth: np.ndarray,
                  stride: int = 4, max_points: int = 100_000, depth_max: float = 3.) -> np.ndarray:
    """rectified Z16을 명시한 단위와 rigid transform으로 3D에 투영한다."""
    matrix = transform(grid_from_depth)
    height, width = int(info['height']), int(info['width'])
    k = np.asarray(info['k'], float).reshape(3, 3)
    scale = float(info['depth_scale_m'])
    if (depth.dtype != np.uint16 or depth.shape != (height, width) or not 0 < scale < .1
            or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.allclose(k[2], [0, 0, 1]) or not np.allclose(k[[0, 1], [1, 0]], 0)
            or any(abs(float(v)) > 1e-8 for v in info.get('d', []))
            or not isinstance(stride, int) or stride < 1 or not np.isfinite(depth_max) or not .3 < depth_max <= 10):
        raise ValueError('invalid rectified depth calibration/profile/stride')
    if ((height+stride-1)//stride)*((width+stride-1)//stride) > max_points:
        raise ValueError('projected points exceed update limit; increase stride')
    v, u = np.mgrid[0:height:stride, 0:width:stride]
    z = depth[::stride, ::stride].astype(np.float32)*scale
    valid = (z >= .25) & (z <= depth_max)
    points = np.column_stack(((u[valid]-k[0, 2])*z[valid]/k[0, 0],
                              (v[valid]-k[1, 2])*z[valid]/k[1, 1], z[valid]))
    return (points @ matrix[:3, :3].T+matrix[:3, 3]).astype(np.float32)


def save_grid(terrain: TerrainGrid, output: Path, limits: RoverLimits,
              min_observations: int, metadata: dict) -> dict:
    """지형 원자료 층·cost·설정을 새 폴더에 보존한다."""
    layers = terrain.layers(min_observations)
    cost = build_costmap(layers, terrain.spec.resolution, limits)
    output.mkdir(parents=True, exist_ok=False)
    with (output/'terrain.npz.tmp').open('wb') as stream:
        np.savez_compressed(stream, **layers, cost=cost)
    (output/'terrain.npz.tmp').replace(output/'terrain.npz')
    summary = {**metadata, 'grid': asdict(terrain.spec), 'rover_limits': asdict(limits),
               'coordinate_system': metadata.get('coordinate_system','user-aligned gravity frame; axis order y,x'),
               'min_observations': min_observations, 'stats': terrain.stats,
               'persistent_grid_bytes': terrain.persistent_bytes,
               'persistent_bytes_scope': 'grid arrays only; excludes temporary layers, one frame, native SLAM/TSDF and process RSS',
               'cost_values': {'unknown': 255, 'blocked': 254, 'valid_range': [0, 252]},
               'cells': {'unknown': int((cost == 255).sum()), 'blocked': int((cost == 254).sum()),
                         'valid': int((cost < 254).sum())},
               'ground_separation': terrain.metadata() if isinstance(terrain, GroundGrid) else None,
               'limitations': ['2.5D local ground model; occluded or disconnected ground can remain unknown',
                               'sensor/pose error may contribute to roughness and vertical span',
                               'observation statistics are not calibrated uncertainty',
                               'rover thresholds and circular clearance require real validation',
                               'no soil mechanics, slip, full footprint collision or Nav2 integration'],
               'accuracy_verified_on_real_camera': False}
    if 'ground' in layers:
        summary['surface_cells'] = {'ground': int(layers['ground'].sum()),
            'observed': int(layers['surface_observed'].sum()),
            'ground_unconfirmed': int((layers['surface_observed'] & ~layers['ground']).sum()),
            'obstacle_height_available': int(layers['obstacle_observed'].sum())}
    try:
        from experiments.handheld_mapping.terrain_preview import render
        render(layers, cost, terrain.spec, output/'preview.png')
        summary['preview_created'] = True
    except ImportError:
        summary['preview_created'] = False
    atomic_json(output/'summary.json', summary)
    return summary


def replay(run: Path, output: Path, map_id: int, spec: GridSpec, grid_from_world: np.ndarray,
           limits: RoverLimits | None = None, backend: str = 'orb', depth_stride: int = 4,
           min_observations: int = 2, alignment_metadata: dict | None = None,
           depth_max: float = 3.) -> dict:
    """고정 최종 pose와 근접 depth를 한 장씩 읽어 기준 지형 지도를 만든다."""
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if backend not in ('orb', 'orbline', 'orb_offline','orb_inertial') or not isinstance(map_id, int) or map_id < 0:
        raise ValueError('invalid backend/map identity')
    if not np.isfinite(depth_max) or not .3 < depth_max <= 10:
        raise ValueError('invalid maximum depth')
    alignment = transform(grid_from_world)
    info = json.loads((run/'capture_calibration.json').read_text())['depth']
    left_from_depth = transform(info['T_left_depth'])
    terrain = GroundGrid(spec)
    filtering, filter_stats = DepthFilter(max_depth_m=depth_max), {}
    integrated = total = 0
    for row, matched, _, depth in filtered_depths(run, run/backend/f'map_{map_id}_keyframes.csv',
                                                  map_id, info, alignment, filtering, filter_stats):
        total += 1
        if matched is None:
            continue
        grid_depth = alignment @ pose_matrix(row) @ left_from_depth
        points = project_depth(depth, info, grid_depth,
                               depth_stride, spec.max_points, depth_max=depth_max)
        terrain.update(points, matched['stamp_ns'], camera_position=grid_depth[:3, 3])
        integrated += 1
    if not integrated or not terrain.stats['accepted_points']:
        raise ValueError('no valid depth in configured grid bounds/keyframes')
    return save_grid(terrain, output, limits or RoverLimits(), min_observations,
                     {'status': 'complete' if integrated == total else 'partial', 'source': str(run),
                      'backend': backend, 'map_id': map_id, 'integrated_frames': integrated,
                      'unintegrated_keyframes': total-integrated, 'grid_from_world': alignment.tolist(),
                      'pose_kind': 'final keyframe poses; offline reference, not live correction',
                      'depth_stride': depth_stride, 'timestamp_tolerance_ns': 2_000_000,
                      'depth_max_m': depth_max,
                      'depth_filter': filtering.metadata(), 'depth_filter_stats': filter_stats,
                      'alignment_metadata':alignment_metadata,
                      'source_sha256': {name: hashlib.sha256((run/name).read_bytes()).hexdigest()
                                        for name in ('capture_calibration.json', f'{backend}/map_{map_id}_keyframes.csv')},
                      'coordinate_system':('estimated gravity z-up; axis order y,x; elevation relative to anchor camera'
                                           if alignment_metadata else 'user-aligned gravity frame; axis order y,x')})


def demo(output: Path) -> dict:
    """평면·경사·단차·벽·미관측 구역을 포함한 알려진 합성 형상을 생성한다."""
    spec = GridSpec((-2., 2., -2., 2.), .05)
    terrain = TerrainGrid(spec)
    x, y = np.meshgrid(np.arange(80)*.05-1.975, np.arange(80)*.05-1.975)
    z = .06*x+.02*y+.14*(x > 1.)
    z += .025*np.sin(32*x)*np.cos(32*y)*((x > -.7) & (x < .2) & (y > .3) & (y < 1.3))
    seen = ~((x < -1.) & (y > 1.))
    points = np.c_[x[seen], y[seen], z[seen]]
    wall = points[(np.abs(points[:, 1]+1.225) < .001) & (np.abs(points[:, 0]) < .8)].copy()
    wall[:, 2] += .4
    for stamp in range(1, 4):
        terrain.update(np.r_[points, wall], stamp)
    return save_grid(terrain, output, RoverLimits(), 2,
                     {'status': 'complete', 'source': 'known synthetic geometry; not real camera/SLAM',
                      'map_id': 0, 'integrated_frames': 3, 'pose_kind': 'known synthetic gravity frame'})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path, nargs='?')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--map-id', type=int)
    parser.add_argument('--alignment', type=Path, help='JSON 4x4 grid_from_world; z-up gravity alignment required')
    parser.add_argument('--bounds', type=float, nargs=4, default=(-5., 5., -5., 5.))
    parser.add_argument('--resolution', type=float, default=.05)
    parser.add_argument('--max-cells', type=int, default=250_000)
    parser.add_argument('--max-points', type=int, default=100_000)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--depth-max', type=float, default=3., help='사용할 최대 optical Z depth [m]')
    parser.add_argument('--min-observations', type=int, default=2)
    parser.add_argument('--backend', choices=('orb', 'orbline', 'orb_offline','orb_inertial'), default='orb')
    parser.add_argument('--rover-limits', type=Path, help='JSON with RoverLimits field names')
    args = parser.parse_args()
    if args.demo:
        result = demo(args.output)
    else:
        if args.run is None or args.alignment is None or args.map_id is None:
            parser.error('real replay requires run, --map-id and explicit --alignment')
        limits = RoverLimits(**json.loads(args.rover_limits.read_text())) if args.rover_limits else RoverLimits()
        result = replay(args.run, args.output, args.map_id, GridSpec(tuple(args.bounds), args.resolution,
                        args.max_cells, args.max_points),
                        np.asarray(json.loads(args.alignment.read_text())), limits, args.backend,
                        args.stride, args.min_observations, depth_max=args.depth_max)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
