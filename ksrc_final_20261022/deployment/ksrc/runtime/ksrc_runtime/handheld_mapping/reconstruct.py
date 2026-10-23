# ================================================================
# reconstruct.py
# 설명: 최종 map별 keyframe과 동기화된 원본 depth를 노트북에서 TSDF로 융합한다.
# 사용법: .venv-mapping/bin/python -m experiments.handheld_mapping.reconstruct RUN
# ================================================================
import argparse
from bisect import bisect_left
from contextlib import nullcontext
import json
import math
from pathlib import Path

import cv2
import numpy as np

from ksrc_runtime.handheld_mapping.storage import atomic_json
from ksrc_runtime.handheld_mapping.depth_filter import DepthFilter, filtered_depths
from ksrc_runtime.handheld_mapping.recording import matched_depths
from ksrc_runtime.handheld_mapping.color import color_matcher, project_depth_color


def transform(value: list) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("finite SE3 transform required")
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)):
        raise ValueError("invalid SE3 transform")
    return matrix


def pose_matrix(row: dict) -> np.ndarray:
    x, y, z, qx, qy, qz, qw = [float(row[key]) for key in ("x", "y", "z", "qx", "qy", "qz", "qw")]
    quaternion = np.array([qx, qy, qz, qw])
    norm = np.linalg.norm(quaternion)
    if not np.isfinite([x, y, z, *quaternion]).all() or not .98 < norm < 1.02:
        raise ValueError("invalid camera pose")
    qx, qy, qz, qw = quaternion / norm
    matrix = np.eye(4)
    matrix[:3, :3] = [[1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
                        [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
                        [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]]
    matrix[:3, 3] = [x, y, z]
    return matrix


def nearest(stamps: list[int], stamp: int, tolerance_ns: int = 2_000_000) -> int | None:
    index = bisect_left(stamps, stamp)
    candidates = stamps[max(0, index-1):index+1]
    if not candidates:
        return None
    result = min(candidates, key=lambda value: abs(value-stamp))
    return result if abs(result-stamp) <= tolerance_ns else None


def reconstruct(run_dir: Path, voxel_size: float = .01, depth_max: float = 1.,
                progress=None, backend: str = "orb", gravity_alignments: dict | None = None,
                depth_filter: DepthFilter | None = None, output_root: Path | None = None,
                *, keyframe_dir: Path | None = None,
                output_dir: Path | None = None, color: bool = False) -> dict:
    import open3d as o3d
    run_dir = Path(run_dir).resolve()
    filtering = depth_filter or DepthFilter(max_depth_m=depth_max)
    if not math.isclose(filtering.max_depth_m, depth_max, rel_tol=0., abs_tol=1e-9):
        raise ValueError('depth filter and TSDF maximum distance differ')
    if output_root is not None and output_dir is not None:
        raise ValueError('choose output_root or output_dir, not both')
    if backend not in ("orb", "orbline", "orb_offline", "orb_inertial"):
        raise ValueError("invalid reconstruction backend")
    if not math.isfinite(voxel_size) or not .002 <= voxel_size <= .1 or not .3 < depth_max <= 10:
        raise ValueError("invalid voxel/depth bounds")
    calibration = json.loads((run_dir / "capture_calibration.json").read_text())
    try:
        info = calibration["depth"]
        scale = float(info["depth_scale_m"])
        left_depth = transform(info["T_left_depth"])
        width, height = int(info["width"]), int(info["height"])
        k = np.asarray(info["k"], dtype=float).reshape(3, 3)
        if (not math.isfinite(scale) or not 0 < scale < .1 or width <= 0 or height <= 0
                or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0
                or not np.allclose(k[2], [0, 0, 1]) or not np.allclose(k[[0, 1], [1, 0]], 0)
                or any(abs(float(v)) > 1e-8 for v in info.get("d", []))):
            raise ValueError("unsupported depth calibration")
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("depth scale and depth-to-left TF calibration required") from error
    pose_dir = (run_dir / backend) if keyframe_dir is None else Path(keyframe_dir).resolve()
    keyfiles = sorted(pose_dir.glob("map_*_keyframes.csv"))
    if backend == 'orb_inertial':
        fusion = json.loads((run_dir/backend/'summary.json').read_text())
        identities = fusion.get('exported_imu_initialized_maps')
        if fusion.get('orb_fused') is not True or not isinstance(identities, list) or not identities or any(type(v) is not int or v < 0 for v in identities):
            raise ValueError('initialized inertial export required for TSDF')
        keyfiles = [path for path in keyfiles if path.name in {f'map_{v}_keyframes.csv' for v in identities}]
    if not keyfiles:
        raise ValueError("원본 depth 또는 최종 keyframe이 없습니다")
    output_name = {"orb": "dense", "orbline": "dense_orbline", "orb_offline": "dense_offline", "orb_inertial": "dense_inertial"}[backend]
    explicit_output = output_dir is not None
    destination = Path(output_root).resolve() if output_root is not None else run_dir
    output = Path(output_dir).resolve() if explicit_output else destination / output_name
    new_output = explicit_output or output_root is not None
    if new_output and (output == run_dir or run_dir.is_relative_to(output)
                            or output.is_relative_to(run_dir/'raw')
                            or output.is_relative_to(run_dir/backend) or output.is_relative_to(pose_dir)):
        raise ValueError('output overlaps source data or poses')
    output.mkdir(parents=True, exist_ok=not new_output)
    # 이전 결과는 원본과 분리한 채 새 summary가 성공한 경우에만 갱신한다.
    (output / "summary.json").unlink(missing_ok=True)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, k[0, 0], k[1, 1], k[0, 2], k[1, 2])
    result = {"status": "complete", "method": "keyframe_TSDF", "voxel_size_m": voxel_size,
              "source_run": str(run_dir), "backend": backend, "keyframe_dir": str(pose_dir),
              "depth_max_m": depth_max, "timestamp_tolerance_ns": 2_000_000,
              "coordinate_system": "per-map initial camera optical; gravity alignment not applied",
              "accuracy_verified_on_real_camera": False, "maps": [], "skipped_maps": []}
    result['depth_filter'] = filtering.metadata()
    if explicit_output:
        result['paths_relative_to'] = 'summary'
    result.update(color_requested=bool(color), color_status='unavailable' if color else 'disabled',
                  color_timestamp_tolerance_ns=34_000_000,
                  color_unknown_rgb=[128,128,128], color_occlusion_tolerance_m=.005)
    color_info = calibration.get('color')
    if color and color_info is not None and color_info.get('depth_frame') != info['depth_frame']:
        raise ValueError('color/depth frame identity mismatch')
    if backend == 'orb_inertial':
        result.update(coordinate_system='ORB stereo-inertial map; gravity estimate unverified',
                      imu_clock_verified=False, pose_source='orb_inertial')
    matcher_context = color_matcher(run_dir, output, color_info) if color else nullcontext(None)
    with matcher_context as matcher:
        for keyfile in keyfiles:
            map_id = keyfile.name.removeprefix("map_").removesuffix("_keyframes.csv")
            if not map_id.isdecimal():
                raise ValueError("keyframe map identity mismatch")
            from ksrc_runtime.handheld_mapping.gravity import load_alignment
            gravity = (load_alignment(run_dir, backend, int(map_id))
                       if gravity_alignments is None and keyframe_dir is None
                       else (gravity_alignments or {}).get(int(map_id)))
            world_from_slam = transform(gravity['world_from_slam']) if gravity else np.eye(4)
            can_color = matcher is not None and matcher.count > 0 and color_info is not None
            volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=voxel_size, sdf_trunc=voxel_size*4,
                color_type=(o3d.pipelines.integration.TSDFVolumeColorType.RGB8 if can_color else
                            o3d.pipelines.integration.TSDFVolumeColorType.NoColor))
            integrated = 0
            total = max_gap = 0
            colored_frames = colored_pixels = valid_pixels = max_color_gap = matched_color_frames = 0
            filter_stats = {}
            for row, frame, gap, depth in filtered_depths(run_dir, keyfile, int(map_id), info,
                                                         world_from_slam, filtering, filter_stats):
                total += 1
                if frame is None:
                    continue
                if np.count_nonzero(depth) < 100:
                    continue
                world_depth = world_from_slam @ pose_matrix(row) @ left_depth
                rgb = np.full((height,width,3), 128, np.uint8)
                valid_pixels += int(np.count_nonzero(depth))
                if can_color:
                    image, color_gap = matcher.match(frame['stamp_ns'])
                    if image is not None:
                        matched_color_frames += 1
                        rgb, mask = project_depth_color(depth,info,image,color_info)
                        pixels = int(np.count_nonzero(mask))
                        colored_pixels += pixels
                        colored_frames += int(pixels > 0)
                        max_color_gap = max(max_color_gap,color_gap)
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(rgb),
                    o3d.geometry.Image(depth), depth_scale=1/scale, depth_trunc=depth_max,
                    convert_rgb_to_intensity=False)
                volume.integrate(rgbd, intrinsic, np.linalg.inv(world_depth))
                integrated += 1
                max_gap = max(max_gap, gap)
                if progress:
                    progress(f"지도 {map_id}: {integrated} keyframe 융합")
            if integrated == 0:
                result["skipped_maps"].append(map_id)
                continue
            mesh, cloud = volume.extract_triangle_mesh(), volume.extract_point_cloud()
            if colored_pixels == 0:
                mesh.vertex_colors = o3d.utility.Vector3dVector()
                cloud.colors = o3d.utility.Vector3dVector()
            mesh.compute_vertex_normals()
            if len(mesh.triangles) == 0 or len(cloud.points) == 0:
                raise ValueError("empty TSDF surface")
            mesh_file, cloud_file = f"map_{map_id}_mesh.ply", f"map_{map_id}_cloud.ply"
            mesh_path = mesh_file if explicit_output else f'{output_name}/{mesh_file}'
            cloud_path = cloud_file if explicit_output else f'{output_name}/{cloud_file}'
            if not o3d.io.write_triangle_mesh(str(output / mesh_file), mesh):
                raise OSError("mesh write failed")
            if not o3d.io.write_point_cloud(str(output / cloud_file), cloud):
                raise OSError("point cloud write failed")
            unavailable_reason = None
            if color and not colored_pixels:
                unavailable_reason = ('missing_color_calibration' if color_info is None else
                                      'no_color_frames' if not matcher.count else
                                      'no_color_in_timestamp_window' if not matched_color_frames else
                                      'no_visible_color_projection')
            result["maps"].append({"map_id": int(map_id), "keyframes": total, "integrated_frames": integrated,
                                  "unintegrated_keyframes": total-integrated,
                                  "max_timestamp_difference_ns": max_gap, "mesh": mesh_path, "cloud": cloud_path,
                                  "vertices": len(mesh.vertices), "triangles": len(mesh.triangles),
                                  'coordinate_system': 'gravity_z_up' if gravity else 'initial_camera_optical',
                                  'world_from_slam': world_from_slam.tolist(),
                                  'gravity_method': gravity['method'] if gravity else None,
                                  'gravity_extrinsic_verified': gravity.get('extrinsic_verified', False) if gravity else False,
                                  'depth_filter_stats': filter_stats,
                                  'colored_frames': colored_frames, 'colored_depth_pixels': colored_pixels,
                                  'matched_color_frames': matched_color_frames,
                                  'color_unavailable_reason': unavailable_reason,
                                  'color_volume_type': 'RGB8' if can_color else 'NoColor',
                                  'valid_depth_pixels': valid_pixels,
                                  'color_coverage': colored_pixels/valid_pixels if valid_pixels else 0.,
                                  'max_color_timestamp_difference_ns': max_color_gap,
                                  'color_status': ('complete' if colored_pixels == valid_pixels else
                                                   'partial' if colored_pixels else
                                                   'unavailable') if color else 'disabled'})
    if not result["maps"]:
        raise ValueError("No matching depth/keyframes with valid depth: 일치하는 유효 depth가 없습니다")
    if result["skipped_maps"] or any(item["unintegrated_keyframes"] for item in result["maps"]):
        result["status"] = "partial"
    if color:
        statuses = {item['color_status'] for item in result['maps']}
        result['color_status'] = ('complete' if statuses == {'complete'} else
                                  'unavailable' if statuses == {'unavailable'} else 'partial')
    if any(row['coordinate_system'] == 'gravity_z_up' for row in result['maps']):
        result['coordinate_system'] = 'per-map; see maps[].coordinate_system; elevation relative to anchor camera'
    atomic_json(output / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--voxel-size", type=float, default=.01)
    parser.add_argument("--depth-max", type=float, default=1., help="지도 융합에 사용하는 최대 depth 거리(m)")
    parser.add_argument('--rgb', action='store_true', help='보정·동기화 가능한 RGB를 depth에 투영하여 색을 융합')
    parser.add_argument('--backend', choices=('orb','orbline','orb_offline','orb_inertial'), default='orb')
    parser.add_argument('--output', type=Path, help='새 출력 폴더; 기존 폴더는 거부')
    parser.add_argument('--output-root', type=Path, help='새 결과 루트 아래 backend별 dense 폴더 생성')
    parser.add_argument('--keyframe-dir', type=Path, help='명시한 map_*_keyframes.csv pose 폴더')
    args = parser.parse_args()
    print(json.dumps(reconstruct(args.run_dir, args.voxel_size, args.depth_max, backend=args.backend,
                                 keyframe_dir=args.keyframe_dir, output_dir=args.output, color=args.rgb,
                                 output_root=args.output_root), indent=2))


if __name__ == "__main__":
    main()
