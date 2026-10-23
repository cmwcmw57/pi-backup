# ================================================================
# reference.py
# 설명: 원본 지도를 보존하며 rigid 정렬·crop·출발 prior·검증된 지도 병합 버전을 관리한다.
# 사용법: from ksrc_runtime.handheld_mapping.reference import load_source, ReferenceMap
# ================================================================
from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from ksrc_runtime.handheld_mapping.reconstruct import pose_matrix, transform
from ksrc_runtime.handheld_mapping.storage import atomic_json

FOLDERS = {'orb_offline': 'dense_offline', 'orb_inertial': 'dense_inertial',
           'orb': 'dense', 'orbline': 'dense_orbline'}


def rigid_pose(x: float, y: float, z: float, yaw: float, roll: float = 0., pitch: float = 0.) -> np.ndarray:
    """degree 입력의 초기 SE3를 만들며 source→reference 방향을 사용한다."""
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()
    matrix[:3, 3] = [x, y, z]
    return transform(matrix)


def manual_gravity(up: list[float] | np.ndarray) -> dict:
    """명시한 source up을 reference +Z로 맞추며 실측 중력 검증으로 간주하지 않는다."""
    vector = np.asarray(up, float)
    if vector.shape != (3,) or not np.isfinite(vector).all() or np.linalg.norm(vector) < 1e-9:
        raise ValueError('finite nonzero up vector required')
    vector = vector / np.linalg.norm(vector)
    z = np.array([0., 0., 1.])
    cross = np.cross(vector, z)
    sine, cosine = np.linalg.norm(cross), float(np.dot(vector, z))
    rotation = (Rotation.from_rotvec(cross / sine * np.arctan2(sine, cosine)).as_matrix()
                if sine > 1e-10 else np.diag([1., -1., -1.]) if cosine < 0 else np.eye(3))
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    return {'accepted': True, 'source': 'manual_up_vector', 'verified': False,
            'quality': 'user_declared_not_measured', 'up_source': vector.tolist(),
            'transformation': matrix.tolist()}


def static_gravity(imu_times: np.ndarray, gyro: np.ndarray, acceleration: np.ndarray,
                   camera_times: np.ndarray, camera_poses: np.ndarray, imu_from_camera: np.ndarray) -> dict:
    """정지·연속 IMU와 근접 camera pose만 사용해 source 좌표계 up 후보를 추정한다."""
    times, camera_times = np.asarray(imu_times, float), np.asarray(camera_times, float)
    gyro, acceleration = np.asarray(gyro, float), np.asarray(acceleration, float)
    poses = np.asarray(camera_poses, float)
    extrinsic = transform(imu_from_camera)
    if (len(times) < 10 or gyro.shape != (len(times), 3) or acceleration.shape != gyro.shape
            or poses.shape != (len(camera_times), 4, 4) or len(poses) < 2
            or not np.isfinite(np.r_[times, gyro.ravel(), acceleration.ravel(), camera_times, poses.ravel()]).all()
            or np.any(np.diff(times) <= 0) or np.any(np.diff(camera_times) <= 0)):
        raise ValueError('finite ordered IMU and matched camera poses required')
    for pose in poses:
        transform(pose)
    accnorm = np.linalg.norm(acceleration, axis=1)
    still = (np.linalg.norm(gyro, axis=1) < .03) & (np.abs(accnorm - 9.80665) < .5)
    # 단 한 개의 정지 sample이나 불연속 구간은 중력 증거로 사용하지 않는다.
    runs: list[np.ndarray] = []
    start = None
    for index in range(len(times)):
        if not still[index] or (index and times[index] - times[index-1] > .1):
            if start is not None:
                runs.append(np.arange(start, index))
            start = index if still[index] else None
        elif start is None:
            start = index
    if start is not None:
        runs.append(np.arange(start, len(times)))
    selected = [run for run in runs if len(run) >= 10 and times[run[-1]] - times[run[0]] >= 1.
                and np.max(np.std(acceleration[run], axis=0)) <= .15]
    if not selected:
        raise ValueError('no continuous 1s static IMU window with low acceleration variance')
    indices = max(selected, key=len)
    up, matched_times = [], []
    for index in indices:
        nearest = int(np.argmin(np.abs(camera_times - times[index])))
        if abs(camera_times[nearest] - times[index]) > .075:
            continue
        # specific force points up at rest; T_imu_c1 maps camera→corrected body IMU.
        up.append(poses[nearest, :3, :3] @ extrinsic[:3, :3].T @ acceleration[index])
        matched_times.append(times[index])
    if len(up) < 10:
        raise ValueError('static IMU window lacks matched camera orientations')
    unit = np.asarray(up) / np.linalg.norm(up, axis=1)[:, None]
    mean = unit.mean(axis=0)
    mean /= np.linalg.norm(mean)
    angular = np.degrees(np.arccos(np.clip(unit @ mean, -1, 1)))
    if angular.max() > 5.:
        raise ValueError('camera/IMU world-up directions disagree')
    result = manual_gravity(mean)
    result.update(source='static_imu_matched_camera_urdf', verified=False,
                  quality={'static_samples': len(up), 'duration_s': float(times[indices[-1]] - times[indices[0]]),
                           'static_window_duration_s': float(times[indices[-1]] - times[indices[0]]),
                           'matched_camera_span_s': float(matched_times[-1] - matched_times[0]),
                           'max_up_deviation_deg': float(angular.max()), 'physical_accuracy_verified': False})
    return result


def file_hash(path: Path) -> str:
    """파일 내용을 streaming SHA256으로 보존 식별한다."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def estimate_gravity(run: Path, source: ReferenceMap, *, allow_unverified_clock: bool = False) -> dict:
    """provenance·시계·map pose 대응 확인 후에만 IMU 자동 후보를 허용한다."""
    from ksrc_runtime.handheld_mapping.inertial import (
        camera_from_urdf, imu_from_camera, validate_imu_provenance,
    )
    try:
        run = Path(run).resolve()
        validate_imu_provenance(run)
        folder = run/'orb_inertial'
        summary = json.loads((folder/'summary.json').read_text())
        alignment = json.loads((folder/'alignment.json').read_text())
        clock_verified = alignment.get('clock_verified') is True
        acquisition_verified = alignment.get('sensor_acquisition_time_verified') is True
        if not (clock_verified and acquisition_verified) and not allow_unverified_clock:
            raise ValueError('IMU/camera acquisition clock is unverified; manual up vector required')
        clock_fit = alignment.get('clock_fit', {})
        if (not alignment.get('clock_domain') or not alignment.get('imu_time_source')
                or clock_fit.get('receipt_residual_trend_suspect') is not False
                or not .9 < float(clock_fit.get('scale', 0)) < 1.1
                or not np.isfinite(float(alignment.get('imu_time_correction_s', np.nan)))):
            raise ValueError('no stable recorded IMU/camera time relation')
        hashes = {**alignment.get('input_sha256', {}), **summary.get('input_sha256', {})}
        for name in ('robot.urdf', 'imu_manifest.json', 'imu_calibration.json', 'capture_calibration.json'):
            if hashes.get(name) != file_hash(run/name):
                raise ValueError(f'gravity provenance hash mismatch: {name}')
        stereo = json.loads((run/'capture_calibration.json').read_text())['stereo']
        extrinsic = imu_from_camera(run/'robot.urdf', camera_from_urdf(run/'robot.urdf', stereo))
        if not np.allclose(transform(summary['T_imu_c1']), extrinsic, atol=1e-6):
            raise ValueError('gravity extrinsic does not match captured URDF')
        stamp_map = json.loads((folder/'camera_stamp_map.json').read_text())
        inverse = {int(original): int(common) for common, original in stamp_map.items()}
        if len(inverse) != len(stamp_map):
            raise ValueError('ambiguous camera clock mapping')
        if not len(source.poses) or source.source.get('trajectory_source') is None:
            raise ValueError('source has no camera orientations')
        camera_times = []
        for stamp in source.stamps:
            key = min(inverse, key=lambda item: abs(item - round(stamp*1e9)))
            if abs(key - round(stamp*1e9)) > 2000:
                raise ValueError('source camera pose has no common-clock correspondence')
            camera_times.append(inverse[key]*1e-9)
        with (folder/'imu.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        times = np.array([int(row['timestamp_ns'])*1e-9 for row in rows])
        gyro = np.array([[float(row[f'{axis}_rad_s']) for axis in ('gx', 'gy', 'gz')] for row in rows])
        acc = np.array([[float(row[f'{axis}_m_s2']) for axis in ('ax', 'ay', 'az')] for row in rows])
        # 이미 reference 정렬한 pose에는 같은 world-up 회전이 적용된다.
        poses = np.asarray([source.local_transform @ pose for pose in source.poses])
        result = static_gravity(times, gyro, acc, np.asarray(camera_times), poses, extrinsic)
        result['quality'].update(clock_verified=clock_verified, sensor_acquisition_time_verified=acquisition_verified,
                                 extrinsic_quality='captured_urdf_not_physically_verified',
                                 clock_quality='verified' if clock_verified and acquisition_verified else 'experimental_unverified')
        result['requires_explicit_user_application'] = True
        return result
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        return {'accepted': False, 'source': 'imu_rejected', 'verified': False, 'reason': str(error)}


def point_mask(points: np.ndarray, crop: dict | None) -> np.ndarray:
    """XY polygon 경계를 포함하고 유한 height 범위로 점을 선택한다."""
    points = np.asarray(points)
    mask = np.isfinite(points).all(axis=1)
    if not crop or len(points) == 0:
        return mask
    mask &= (points[:, 2] >= crop['height_min']) & (points[:, 2] <= crop['height_max'])
    polygon = np.asarray(crop['polygon'], float)
    if len(polygon):
        inside = np.zeros(len(points), bool)
        boundary = np.zeros(len(points), bool)
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            dx, dy = b - a
            cross = (points[:, 0] - a[0])*dy - (points[:, 1] - a[1])*dx
            boundary |= (np.abs(cross) < 1e-9) & ((points[:, 0]-a[0])*(points[:, 0]-b[0])
                                                       + (points[:, 1]-a[1])*(points[:, 1]-b[1]) <= 1e-9)
            if abs(dy) > 1e-12:
                crossing = ((a[1] > points[:, 1]) != (b[1] > points[:, 1]))
                inside ^= crossing & (points[:, 0] < dx*(points[:, 1] - a[1])/dy + a[0])
        mask &= inside | boundary
    return mask


class ReferenceMap:
    """원본 geometry와 pose를 불변 입력으로 보관하고 편집·미리보기·버전 저장을 수행한다."""

    def __init__(self, cloud: o3d.geometry.PointCloud, mesh: o3d.geometry.TriangleMesh | None = None,
                 poses: np.ndarray | None = None, stamps: np.ndarray | None = None, map_id: int | None = None,
                 source: dict | None = None, breaks: np.ndarray | None = None) -> None:
        if cloud.is_empty() or not np.isfinite(np.asarray(cloud.points)).all():
            raise ValueError('nonempty finite point cloud required')
        if map_id is not None and (type(map_id) is not int or map_id < 0):
            raise ValueError('invalid map identity')
        self.cloud, self.mesh = copy.deepcopy(cloud), copy.deepcopy(mesh)
        self.poses = np.asarray(poses, float).copy() if poses is not None else np.empty((0, 4, 4))
        if self.poses.shape != (len(self.poses), 4, 4):
            raise ValueError('camera poses must have shape Nx4x4')
        for pose in self.poses:
            transform(pose)
        self.stamps = np.asarray(stamps, float).copy() if stamps is not None else np.arange(len(self.poses), dtype=float)
        if self.stamps.shape != (len(self.poses),) or not np.isfinite(self.stamps).all():
            raise ValueError('finite camera timestamps required')
        self.breaks = np.asarray(breaks, bool).copy() if breaks is not None else np.zeros(len(self.poses), bool)
        if self.breaks.shape != (len(self.poses),):
            raise ValueError('trajectory breaks shape mismatch')
        self.local_transform, self.saved_transform = np.eye(4), np.eye(4)
        self.map_id, self.source = map_id, copy.deepcopy(source or {})
        self.path_kind = 'camera' if len(self.poses) else 'none'
        self.camera_from_base = None
        self.crop: dict | None = None
        self.saved_crop: dict | None = None
        self.gravity = {'source': 'unaligned', 'verified': False, 'quality': 'unknown'}
        self.start_pose: dict | None = None
        self.original_start = self.poses[0].tolist() if len(self.poses) else None
        self.parent_version = None
        self.merge: dict | None = None
        self.processing = {'max_triangles': 0, 'smooth_iterations': 0}
        self.saved_processing: dict | None = None
        self._geometry_cache: tuple | None = None

    @property
    def reference_transform(self) -> np.ndarray:
        return self.local_transform @ self.saved_transform

    def apply_transform(self, matrix: np.ndarray | list, gravity: dict | None = None) -> None:
        matrix = transform(matrix)
        self.local_transform = matrix @ self.local_transform
        self._geometry_cache = None
        if gravity is not None:
            self.gravity = copy.deepcopy(gravity)
        if self.start_pose is not None:
            self.start_pose['matrix'] = (matrix @ transform(self.start_pose['matrix'])).tolist()
        # Crop polygon은 reference XY 좌표이므로 frame을 바꿀 때 명시적으로 다시 지정한다.
        self.crop = None
        self.saved_crop = None

    def set_crop(self, polygon: list, height_min: float, height_max: float) -> None:
        polygon = np.asarray(polygon, float)
        if polygon.size == 0:
            polygon = np.empty((0, 2))
        if (polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) in (1, 2)
                or not np.isfinite(polygon).all() or not np.isfinite([height_min, height_max]).all()
                or height_min > height_max):
            raise ValueError('polygon needs >=3 finite XY vertices and ordered height bounds')
        if len(polygon):
            area = abs(np.dot(polygon[:, 0], np.roll(polygon[:, 1], 1)) - np.dot(polygon[:, 1], np.roll(polygon[:, 0], 1)))
            if area < 1e-10:
                raise ValueError('polygon area must be nonzero')
        self.crop = {'polygon': polygon.tolist(), 'height_min': float(height_min), 'height_max': float(height_max),
                     'mesh_boundary_policy': 'all_vertices_inside'}
        self.saved_crop = None
        self._geometry_cache = None

    def set_start(self, position: list[float], heading_point: list[float]) -> None:
        position, other = np.asarray(position, float), np.asarray(heading_point, float)
        if position.shape != (3,) or other.shape != (3,) or not np.isfinite(np.r_[position, other]).all():
            raise ValueError('finite start and heading positions required')
        delta = other[:2] - position[:2]
        if np.linalg.norm(delta) < 1e-6:
            raise ValueError('heading requires a second distinct XY point')
        matrix = rigid_pose(*position, np.degrees(np.arctan2(delta[1], delta[0])))
        self.start_pose = {'matrix': matrix.tolist(), 'provenance': 'manual_initial_prior_not_measured',
                           'height_source': 'explicit_user_value', 'localization_verified': False}

    def set_processing(self, max_triangles: int = 0, smooth_iterations: int = 0) -> None:
        """원본 mesh를 유지하며 명시한 간소화와 제한된 Taubin preview를 설정한다."""
        if (type(max_triangles) is not int or max_triangles < 0 or max_triangles in (1, 2, 3)
                or type(smooth_iterations) is not int or not 0 <= smooth_iterations <= 10):
            raise ValueError('max triangles must be 0 or >=4; Taubin iterations must be 0..10')
        if self.mesh is None or not len(self.mesh.triangles):
            raise ValueError('mesh processing requires a triangle mesh')
        self.processing = {'max_triangles': max_triangles, 'smooth_iterations': smooth_iterations,
                           'method': 'quadric_decimation_then_optional_taubin',
                           'small_geometry_may_be_lost': True, 'analysis_original_preserved': True}
        self._geometry_cache = None

    def geometry(self) -> tuple[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh | None]:
        if self._geometry_cache is not None:
            return copy.deepcopy(self._geometry_cache)
        cloud = copy.deepcopy(self.cloud).transform(self.local_transform)
        indices = np.flatnonzero(point_mask(np.asarray(cloud.points), self.crop))
        cloud = cloud.select_by_index(indices.tolist())
        mesh = copy.deepcopy(self.mesh)
        if mesh is not None:
            mesh.transform(self.local_transform)
            if self.crop is not None:
                mask = point_mask(np.asarray(mesh.vertices), self.crop)
                triangles = np.asarray(mesh.triangles)
                mesh.remove_triangles_by_mask(~mask[triangles].all(axis=1))
                mesh.remove_unreferenced_vertices()
            limit = self.processing['max_triangles']
            iterations = self.processing['smooth_iterations']
            if limit and len(mesh.triangles) > limit:
                mesh = mesh.simplify_quadric_decimation(limit)
                # decimation의 고립 vertex는 Taubin에서 NaN이 될 수 있다.
                mesh.remove_unreferenced_vertices()
            if iterations and len(mesh.triangles):
                mesh.remove_degenerate_triangles()
                mesh.remove_unreferenced_vertices()
                mesh = mesh.filter_smooth_taubin(number_of_iterations=iterations)
            if not np.isfinite(np.asarray(mesh.vertices)).all():
                raise ValueError('non-finite mesh geometry; preview/save rejected')
            if len(mesh.triangles):
                mesh.compute_vertex_normals()
        self._geometry_cache = cloud, mesh
        return copy.deepcopy(self._geometry_cache)

    def transformed_poses(self) -> np.ndarray:
        poses = np.asarray([self.local_transform @ pose for pose in self.poses]).reshape(-1, 4, 4)
        if self.camera_from_base is not None:
            poses = poses @ self.camera_from_base
        return poses

    def trajectory_segments(self, max_gap_s: float = 2.) -> list[list[list[float]]]:
        poses = self.transformed_poses()
        points = poses[:, :3, 3]
        keep = point_mask(points, self.crop)
        segments: list[list[list[float]]] = []
        current: list[list[float]] = []
        for index, point in enumerate(points):
            gap = index and (self.stamps[index] - self.stamps[index-1] > max_gap_s
                             or self.stamps[index] <= self.stamps[index-1] or self.breaks[index])
            if gap or not keep[index]:
                if current:
                    segments.append(current)
                current = []
            if keep[index]:
                current.append(point.tolist())
        if current:
            segments.append(current)
        return segments

    def _retained_trajectory(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """crop 밖 camera pose를 제거하고 끊긴 구간의 선 연결 금지를 보존한다."""
        keep = point_mask(self.transformed_poses()[:, :3, 3], self.crop)
        selected = np.flatnonzero(keep)
        breaks = self.breaks[selected].copy()
        if len(breaks) > 1:
            breaks[1:] |= np.diff(selected) != 1
        poses = np.asarray([self.local_transform @ self.poses[index] for index in selected]).reshape(-1, 4, 4)
        return poses, self.stamps[selected], breaks

    def metadata(self) -> dict:
        original = (self.local_transform @ transform(self.original_start)).tolist() if self.original_start is not None else None
        return {'schema_version': 1, 'units': 'm', 'map_id': self.map_id, 'source': copy.deepcopy(self.source),
                'source_hash': self.source.get('sha256', {}), 'T_reference_source': self.reference_transform.tolist(),
                'geometry_frame': 'reference', 'gravity': copy.deepcopy(self.gravity),
                'crop': copy.deepcopy(self.crop if self.crop is not None else self.saved_crop),
                'start_pose': copy.deepcopy(self.start_pose), 'original_start': original,
                'trajectory_kind': self.path_kind, 'trajectory_segments': self.trajectory_segments(),
                'parent_version': self.parent_version, 'merge': copy.deepcopy(self.merge),
                'processing': copy.deepcopy(self.saved_processing if self.saved_processing is not None and
                                            not any(self.processing[key] for key in ('max_triangles', 'smooth_iterations'))
                                            else self.processing),
                'accuracy_verified_on_real_camera': False}

    def save(self, destination: Path) -> dict:
        destination = Path(destination).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=False)
        try:
            cloud, mesh = self.geometry()
            if cloud.is_empty():
                raise ValueError('crop leaves no geometry')
            if not o3d.io.write_point_cloud(str(destination/'cloud.ply'), cloud):
                raise OSError('reference cloud write failed')
            if mesh is not None and len(mesh.triangles):
                if not o3d.io.write_triangle_mesh(str(destination/'mesh.ply'), mesh):
                    raise OSError('reference mesh write failed')
            original = destination/'original'
            original.mkdir()
            if not o3d.io.write_point_cloud(str(original/'cloud.ply'), self.cloud):
                raise OSError('original cloud preservation failed')
            if self.mesh is not None and len(self.mesh.triangles):
                if not o3d.io.write_triangle_mesh(str(original/'mesh.ply'), self.mesh):
                    raise OSError('original mesh preservation failed')
            atomic_json(original/'metadata.json', {'source': self.source, 'camera_poses': self.poses.tolist(),
                                                   'stamps_s': self.stamps.tolist(), 'breaks': self.breaks.tolist(),
                                                   'geometry_frame': 'loaded_input',
                                                   'T_reference_input': self.local_transform.tolist(),
                                                   'input_T_reference_source': self.saved_transform.tolist()})
            metadata = self.metadata()
            poses, stamps, breaks = self._retained_trajectory()
            metadata.update(cloud='cloud.ply', mesh='mesh.ply' if (destination/'mesh.ply').exists() else None,
                            camera_poses=poses.tolist(), stamps_s=stamps.tolist(), trajectory_breaks=breaks.tolist(),
                            T_camera_base=self.camera_from_base.tolist() if self.camera_from_base is not None else None)
            atomic_json(destination/'reference.json', metadata)
            return metadata
        except Exception:
            # 실패한 새 폴더에는 완료 reference.json을 두지 않는다. 원본·이전 버전은 건드리지 않는다.
            (destination/'reference.json').unlink(missing_ok=True)
            raise

    def preview_merge(self, candidate: ReferenceMap, initial: np.ndarray,
                      voxel_size: float = .02, registrar: Callable | None = None) -> dict:
        if registrar is None:
            from ksrc_runtime.handheld_mapping.registration import register_clouds
            registrar = register_clouds
        matrix = transform(initial)
        source, _ = candidate.geometry()
        target, _ = self.geometry()
        result = registrar(source, target, matrix, voxel_size=voxel_size)
        if result.get('accepted'):
            transform(result['transformation'])
        return result

    def merged(self, candidate: ReferenceMap, result: dict) -> ReferenceMap:
        if result.get('accepted') is not True:
            raise ValueError('accepted registration preview required')
        matrix = transform(result['transformation'])
        cloud, mesh = self.geometry()
        other_cloud, other_mesh = candidate.geometry()
        cloud += other_cloud.transform(matrix)
        if mesh is not None and other_mesh is not None:
            mesh += other_mesh.transform(matrix)
        else:
            # 일부 cloud만 있는 경우 incomplete mesh를 전체 map으로 표시하지 않는다.
            mesh = None
        # camera poses는 camera-frame 저장을 유지하고 base extrinsic은 동일한 경우만 공유한다.
        a, a_stamps, a_breaks = self._retained_trajectory()
        b, b_stamps, b_breaks = candidate._retained_trajectory()
        b = np.asarray([matrix @ pose for pose in b]).reshape(-1, 4, 4)
        breaks = np.r_[a_breaks, b_breaks]
        if len(b):
            breaks[len(a)] = True
        merged = ReferenceMap(cloud, mesh, np.concatenate([a, b]), np.r_[a_stamps, b_stamps],
                              self.map_id, self.source, breaks)
        merged.saved_transform = self.reference_transform.copy()
        merged.gravity, merged.start_pose = copy.deepcopy(self.gravity), copy.deepcopy(self.start_pose)
        merged.original_start = (self.local_transform @ transform(self.original_start)).tolist() if self.original_start else None
        merged.parent_version = self.parent_version
        if self.camera_from_base is not None and candidate.camera_from_base is not None and np.allclose(self.camera_from_base, candidate.camera_from_base):
            merged.camera_from_base = self.camera_from_base.copy()
            merged.path_kind = 'rover'
        merged.merge = {'method': 'registered_geometry_union_not_joint_tsdf', 'candidate': copy.deepcopy(candidate.source),
                        'T_reference_candidate': matrix.tolist(), 'registration': copy.deepcopy(result),
                        'baseline_frozen': True, 'watertight_verified': False}
        merged.saved_processing = self.metadata()['processing']
        return merged


def safe_path(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError('invalid geometry path')
    path = (root/value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('geometry path escapes summary/source root')
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return path


def _summary(target: Path, backend: str, variant: str) -> tuple[Path, Path, dict, Path | None, str]:
    """summary root와 원본 run을 별도로 반환해 legacy 경로 의미를 보존한다."""
    if backend not in ('auto', *FOLDERS):
        raise ValueError('unknown source backend')
    folder = target.parent if target.is_file() else target
    run: Path | None = None
    if target.name == 'summary.json' or (folder/'summary.json').exists():
        summary_path = target if target.is_file() else folder/'summary.json'
    else:
        selected = next(((name, folder/path/'summary.json') for name, path in FOLDERS.items()
                         if backend in ('auto', name) and (folder/path/'summary.json').is_file()), None)
        if selected is None:
            raise FileNotFoundError(f'no saved geometry summary: {folder}')
        backend, summary_path = selected
        run = folder
    data = json.loads(summary_path.read_text())
    if not isinstance(data, dict):
        raise ValueError('source summary must be an object')
    if backend == 'auto' and data.get('backend') in FOLDERS:
        backend = data['backend']
    if f'{variant}_summary' in data:
        if variant not in ('baseline', 'optimized'):
            raise ValueError('global variant must be baseline or optimized')
        run = Path(data.get('source_run', data.get('source', ''))).expanduser().resolve()
        backend = data.get('backend', backend)
        nested = safe_path(summary_path.parent, data[f'{variant}_summary'])
        return nested, nested.parent, json.loads(nested.read_text()), run, backend
    if run is None and summary_path.parent.name in FOLDERS.values():
        run = summary_path.parent.parent
        if backend == 'auto':
            backend = next(name for name, value in FOLDERS.items() if value == summary_path.parent.name)
    root = summary_path.parent if data.get('paths_relative_to') == 'summary' else run or summary_path.parent
    source_run = data.get('source_run', data.get('source'))
    if run is None and isinstance(source_run, str) and source_run:
        run = Path(source_run).expanduser().resolve()
    return summary_path, root, data, run, backend


def source_maps(target: Path, backend: str = 'auto', variant: str = 'optimized') -> list[int]:
    target = Path(target).expanduser().resolve()
    if target.suffix.lower() in ('.ply', '.stl') or (target/'reference.json').exists() or target.name == 'reference.json':
        return []
    _, _, summary, _, _ = _summary(target, backend, variant)
    maps = summary.get('maps')
    if not isinstance(maps, list) or any(not isinstance(row, dict) or type(row.get('map_id')) is not int or row['map_id'] < 0 for row in maps):
        raise ValueError('invalid source maps')
    identities = [row['map_id'] for row in maps]
    if len(set(identities)) != len(identities):
        raise ValueError('duplicate source map identity')
    return identities


def _read_geometry(cloud_path: Path | None, mesh_path: Path | None) -> tuple[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh | None]:
    mesh = o3d.io.read_triangle_mesh(str(mesh_path)) if mesh_path is not None else None
    if mesh is not None and not len(mesh.triangles):
        mesh = None
    cloud = o3d.io.read_point_cloud(str(cloud_path)) if cloud_path is not None else o3d.geometry.PointCloud()
    if cloud.is_empty() and mesh is not None:
        cloud.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
        if mesh.has_vertex_colors():
            cloud.colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors))
    if cloud.is_empty():
        raise ValueError('empty source geometry')
    return cloud, mesh


def load_source(target: Path, map_id: int | None = None, backend: str = 'auto',
                variant: str = 'optimized', units_confirmed: bool = False) -> ReferenceMap:
    """legacy RUN·explicit/global summary·direct geometry·저장 reference를 읽는다."""
    target = Path(target).expanduser().resolve()
    if target.is_dir() and (target/'reference.json').exists():
        target = target/'reference.json'
    if target.name == 'reference.json':
        data = json.loads(target.read_text())
        if data.get('schema_version') != 1 or data.get('units') != 'm' or data.get('geometry_frame') != 'reference':
            raise ValueError('unsupported reference schema, units or geometry frame')
        cloud, mesh = _read_geometry(safe_path(target.parent, data['cloud']),
                                     safe_path(target.parent, data['mesh']) if data.get('mesh') else None)
        poses = np.asarray(data.get('camera_poses', []), float).reshape(-1, 4, 4)
        value = ReferenceMap(cloud, mesh, poses, np.asarray(data.get('stamps_s', [])), data.get('map_id'),
                             data.get('source'), np.asarray(data.get('trajectory_breaks', np.zeros(len(poses))), bool))
        value.saved_transform = transform(data['T_reference_source'])
        value.gravity, value.start_pose = data.get('gravity'), data.get('start_pose')
        value.saved_crop = data.get('crop')
        value.saved_processing = data.get('processing')
        value.original_start, value.merge = data.get('original_start'), data.get('merge')
        if value.original_start is not None:
            transform(value.original_start)
        if value.start_pose is not None:
            transform(value.start_pose['matrix'])
        value.parent_version = str(target.parent)
        if data.get('T_camera_base') is not None:
            value.camera_from_base = transform(data['T_camera_base'])
            value.path_kind = 'rover'
        # geometry와 poses는 이미 reference frame이다. 저장 T/mesh 처리를 다시 적용하지 않는다.
        return value
    if target.suffix.lower() in ('.ply', '.stl'):
        if target.suffix.lower() == '.stl' and not units_confirmed:
            raise ValueError('STL units unknown: explicit meter confirmation required')
        cloud, mesh = _read_geometry(target if target.suffix.lower() == '.ply' else None, target)
        return ReferenceMap(cloud, mesh, map_id=map_id,
                            source={'kind': 'direct_geometry', 'path': str(target),
                                    'units_provenance': 'user_confirmed_meters' if target.suffix.lower() == '.stl' else 'assumed_meters',
                                    'sha256': {target.name: file_hash(target)}})
    summary_path, root, data, run, backend = _summary(target, backend, variant)
    maps = data.get('maps')
    if not isinstance(maps, list) or not maps or any(not isinstance(row, dict) for row in maps):
        raise ValueError('source summary has no maps')
    identities = [row.get('map_id') for row in maps]
    if any(type(identity) is not int or identity < 0 for identity in identities) or len(set(identities)) != len(identities):
        raise ValueError('invalid source map identity')
    if map_id is None:
        if len(maps) != 1:
            raise ValueError(f'explicit map_id required for separate maps: {identities}')
        map_id = identities[0]
    row = next((row for row in maps if row['map_id'] == map_id), None)
    if row is None:
        raise ValueError(f'source map {map_id} unavailable')
    cloud_path = safe_path(root, row['cloud']) if row.get('cloud') else None
    mesh_path = safe_path(root, row['mesh']) if row.get('mesh') else None
    cloud, mesh = _read_geometry(cloud_path, mesh_path)
    source = {'kind': 'summary', 'summary': str(summary_path), 'run': str(run) if run else None,
              'backend': backend, 'variant': variant, 'map_id': map_id,
              'sha256': {str(path): file_hash(path) for path in (summary_path, cloud_path, mesh_path) if path}}
    # global job root/keyframes.csv는 선택한 optimized camera pose를 저장한다.
    keyfile = summary_path.parent.parent/'keyframes.csv' if summary_path.parent.name == 'optimized' else None
    if (keyfile is None or not keyfile.is_file()) and data.get('keyframe_dir'):
        keydir = Path(data['keyframe_dir']).expanduser()
        if not keydir.is_absolute():
            keydir = summary_path.parent/keydir
        keyfile = keydir.resolve()/f'map_{map_id}_keyframes.csv'
    if keyfile is None or not keyfile.is_file():
        keyfile = run/backend/f'map_{map_id}_keyframes.csv' if run is not None and backend in FOLDERS else None
    poses, stamps = [], []
    if keyfile is not None and keyfile.exists():
        with keyfile.open() as stream:
            rows = sorted(csv.DictReader(stream), key=lambda row: float(row['source_stamp']))
        for entry in rows:
            if str(entry.get('map_id')) != str(map_id):
                raise ValueError('trajectory map identity mismatch')
            poses.append(pose_matrix(entry))
            stamps.append(float(entry['source_stamp']))
        source['trajectory_source'] = str(keyfile)
        source['sha256'][str(keyfile)] = file_hash(keyfile)
    value = ReferenceMap(cloud, mesh, np.asarray(poses).reshape(-1, 4, 4), np.asarray(stamps), map_id, source)
    if run is not None and (run/'robot.urdf').exists() and (run/'capture_calibration.json').exists():
        from ksrc_runtime.handheld_mapping.inertial import camera_from_urdf
        try:
            stereo = json.loads((run/'capture_calibration.json').read_text())['stereo']
            value.camera_from_base = transform(camera_from_urdf(run/'robot.urdf', stereo))
            value.path_kind = 'rover' if len(poses) else 'none'
            value.source['base_extrinsic_provenance'] = {'robot_urdf_sha256': file_hash(run/'robot.urdf'),
                                                       'calibration_sha256': file_hash(run/'capture_calibration.json')}
        except (ValueError, KeyError, OSError, AttributeError):
            pass
    return value
