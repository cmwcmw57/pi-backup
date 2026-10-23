# ================================================================
# region_depth.py
# 설명: 현재 stereo depth를 지도 면에 정합하는 초기 위치 비교용 오프라인 프로토타입이다.
#       GT·주행 경로·IMU를 읽지 않으며 실기나 주행 제어기에 연결하지 않는다.
# 사용법: from experiments.localization.region_depth import DepthMap, observation
# ================================================================
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import open3d as o3d

from experiments.localization.wall_range import Camera


@dataclass
class Observation:
    """지도나 pose 없이 현재 영상에서 선택한 근거리 모래 후보와 광선이다."""
    points: np.ndarray
    rays: np.ndarray
    depth: np.ndarray


def observation(gray: np.ndarray, depth: np.ndarray, camera: Camera, stride: int = 8) -> Observation:
    """고정 밝기·깊이 조건으로 배경·검은 벽·결측을 제외한다."""
    v, u = np.mgrid[0:gray.shape[0]:stride, 0:gray.shape[1]:stride]
    z = depth[v, u]
    keep = np.isfinite(z)&(z >= .25)&(z <= 2.)&(gray[v, u] > 25)&(gray[v, u] < 200)
    rays = np.c_[(u[keep]-camera.cx)/camera.fx, (v[keep]-camera.cy)/camera.fy, np.ones(keep.sum())]
    z = z[keep]
    if len(z) > 1600:
        indices = np.linspace(0, len(z)-1, 1600).astype(int)
        rays, z = rays[indices], z[indices]
    points = (rays*z[:, None])@camera.base_from_camera[:3, :3].T+camera.base_from_camera[:3, 3]
    return Observation(points, rays, z)


def base_motion(current_camera_from_previous: np.ndarray, base_from_camera: np.ndarray) -> np.ndarray:
    """PnP 방향을 이전 base에서 본 현재 base pose로 바꾼다."""
    return base_from_camera@np.linalg.inv(current_camera_from_previous)@np.linalg.inv(base_from_camera)


def visual_motion(previous: dict, current: dict, camera: Camera, dt: float = 1.) -> tuple[np.ndarray, dict]:
    """이전 stereo3D와 현재 영상2D의 PnP로 상대 이동만 추정한다."""
    info = dict(accepted=False, matches=0, inliers=0)
    if previous['descriptors'] is None or current['descriptors'] is None:
        return np.eye(4), info
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(previous['descriptors'], current['descriptors'], k=2)
    good = [a for pair in pairs if len(pair) == 2 for a, b in [pair] if a.distance < .75*b.distance]
    info['matches'] = len(good)
    if len(good) < 20:
        return np.eye(4), info
    old = np.array([previous['pixels'][m.queryIdx] for m in good])
    new = np.array([current['pixels'][m.trainIdx] for m in good])
    u, v = np.round(old).astype(int).T
    z = previous['depth'][v, u]
    valid = np.isfinite(z)&(z >= .25)&(z <= 2.)
    if valid.sum() < 20:
        return np.eye(4), info
    xyz = np.c_[(old[valid, 0]-camera.cx)/camera.fx, (old[valid, 1]-camera.cy)/camera.fy, np.ones(valid.sum())]*z[valid, None]
    matrix = np.array([[camera.fx, 0., camera.cx], [0., camera.fy, camera.cy], [0., 0., 1.]])
    success, rvec, tvec, inliers = cv2.solvePnPRansac(xyz, new[valid], matrix, None, iterationsCount=150,
                                                    reprojectionError=2.5, confidence=.999, flags=cv2.SOLVEPNP_EPNP)
    if not success or inliers is None or len(inliers) < 20:
        return np.eye(4), info
    ids = inliers.ravel()
    success, rvec, tvec = cv2.solvePnP(xyz[ids], new[valid][ids], matrix, None, rvec, tvec, True, flags=cv2.SOLVEPNP_ITERATIVE)
    transform = np.eye(4);transform[:3, :3] = cv2.Rodrigues(rvec)[0];transform[:3, 3] = tvec.ravel()
    movement = base_motion(transform, camera.base_from_camera)
    angle = np.arccos(np.clip((np.trace(movement[:3, :3])-1)/2, -1, 1))
    accepted = success and np.isfinite(movement).all() and np.linalg.norm(movement[:3, 3]) < .25*dt+.03 and angle < np.deg2rad(50.*dt+5.)
    info.update(accepted=bool(accepted), inliers=len(ids))
    return (movement if accepted else np.eye(4)), info


class DepthMap:
    """전체 지도에서 초기 pose 후보를 찾고 가까운 면과의 정합을 갱신한다."""

    def __init__(self, mesh: o3d.geometry.TriangleMesh, camera: Camera) -> None:
        self.camera = camera
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        o3d.utility.random.seed(17)
        self.target = mesh.sample_points_uniformly(60000, use_triangle_normal=True).voxel_down_sample(.025)
        self.bounds = np.array([mesh.get_min_bound(), mesh.get_max_bound()])
        self.estimator = o3d.pipelines.registration.TransformationEstimationPointToPlane(o3d.pipelines.registration.TukeyLoss(.06))
        self.criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=25)
        xs = np.arange(self.bounds[0, 0]+.125, self.bounds[1, 0], .25)
        ys = np.arange(self.bounds[0, 1]+.125, self.bounds[1, 1], .25)
        self.xy_grid = np.array(np.meshgrid(xs, ys)).reshape(2, -1).T

    def seed_poses(self, xy: np.ndarray | None = None, yaw: float | None = None) -> np.ndarray:
        """지도의 표면과 rover 명목 높이로 seed를 만든다. 주행 시작점은 내장하지 않는다."""
        points = self.xy_grid if xy is None else np.asarray(xy).reshape(1, 2)
        rays = np.c_[points, np.full(len(points), 3.), np.zeros((len(points), 2)), -np.ones(len(points))].astype(np.float32)
        hits = self.scene.cast_rays(o3d.core.Tensor(rays))
        depth, normal = hits['t_hit'].numpy(), hits['primitive_normals'].numpy().astype(float)
        keep = np.isfinite(depth)&(normal[:, 2] >= np.cos(np.deg2rad(35.)))
        poses = []
        for p, z, n in zip(points[keep], 3.-depth[keep]+.16, normal[keep]):
            for angle in (np.arange(-np.pi, np.pi, np.deg2rad(15.)) if yaw is None else [yaw]):
                forward = np.array([np.cos(angle), np.sin(angle), 0.])
                forward -= n*(n@forward);forward /= np.linalg.norm(forward)
                pose = np.eye(4);pose[:3, :3] = np.c_[forward, np.cross(n, forward), n];pose[:3, 3] = [*p, z]
                poses.append(pose)
        return np.array(poses).reshape(-1, 4, 4)

    def scores(self, poses: np.ndarray, observed: Observation) -> np.ndarray:
        """현재 유효 depth 광선에서 가림을 포함한 지도 깊이 잔차를 계산한다."""
        if not len(observed.depth):
            return np.full(len(poses), 4.)
        ids = np.linspace(0, len(observed.depth)-1, min(160, len(observed.depth))).astype(int)
        rays, measured = observed.rays[ids], observed.depth[ids]
        sigma = .015+.02*measured**2
        scores = []
        for start in range(0, len(poses), 128):
            cameras = poses[start:start+128]@self.camera.base_from_camera
            directions = np.einsum('pj,bkj->bpk', rays, cameras[:, :3, :3])
            origins = np.broadcast_to(cameras[:, None, :3, 3], directions.shape)
            query = np.concatenate([origins, directions], axis=2).reshape(-1, 6).astype(np.float32)
            predicted = self.scene.cast_rays(o3d.core.Tensor(query))['t_hit'].numpy().reshape(len(cameras), -1)
            error = np.abs(predicted-measured)/sigma
            scores.extend(np.minimum(error, 4.).mean(axis=1).tolist())
        return np.array(scores)

    def refine(self, observed: Observation, pose: np.ndarray, local: bool = False) -> tuple[np.ndarray, bool]:
        """robust point-to-plane ICP 결과가 현재 depth와 일관될 때만 채택한다."""
        if len(observed.points) < 100:
            return pose.copy(), False
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(observed.points)).voxel_down_sample(.025)
        result = o3d.pipelines.registration.registration_icp(cloud, self.target, .12, pose, self.estimator, self.criteria)
        candidate = result.transformation
        shift = np.linalg.norm(candidate[:3, 3]-pose[:3, 3])
        delta = candidate[:3, :3]@pose[:3, :3].T
        angle = np.arccos(np.clip((np.trace(delta)-1)/2, -1, 1))
        sane = np.isfinite(candidate).all() and candidate[2, 2] > .65 and result.fitness >= .35
        sane = sane and shift <= (.25 if local else .4) and angle <= np.deg2rad(20. if local else 45.)
        if sane:
            scores = self.scores(np.array([pose, candidate]), observed)
            if scores[1] <= scores[0]:
                return candidate.copy(), True
        return pose.copy(), False

    @staticmethod
    def distinct(poses: np.ndarray, scores: np.ndarray, count: int, distance: float = .3) -> list[int]:
        """가까운 중복을 제외하고 서로 다른 장소·방향의 후보를 남긴다."""
        kept = []
        for index in np.argsort(scores):
            yaw = np.arctan2(poses[index, 1, 0], poses[index, 0, 0])
            duplicate = False
            for k in kept:
                other_yaw = np.arctan2(poses[k, 1, 0], poses[k, 0, 0])
                dyaw = np.arctan2(np.sin(yaw-other_yaw), np.cos(yaw-other_yaw))
                if np.linalg.norm(poses[index, :2, 3]-poses[k, :2, 3]) < distance and abs(dyaw) < np.deg2rad(25.):
                    duplicate = True;break
            if not duplicate:
                kept.append(int(index))
                if len(kept) == count:
                    break
        return kept

    def initialize(self, observed: Observation, xy: np.ndarray | None = None, yaw: float | None = None) -> list[dict]:
        """전역/XY-only/XY+yaw에 동일한 센서 점수를 적용한다."""
        if len(observed.points) < 100:
            return []
        seeds = self.seed_poses(xy, yaw)
        if not len(seeds):
            return []
        scores = self.scores(seeds, observed)
        ids = self.distinct(seeds, scores, min(32, len(seeds)), distance=.2)
        refined = np.array([self.refine(observed, seeds[k])[0] for k in ids])
        if xy is not None:
            for index, k in enumerate(ids):
                shifted = np.linalg.norm(refined[index, :2, 3]-np.asarray(xy)) > .05
                heading = np.arctan2(refined[index, 1, 0], refined[index, 0, 0])
                turned = yaw is not None and abs(np.arctan2(np.sin(heading-yaw), np.cos(heading-yaw))) > np.deg2rad(5.)
                if shifted or turned:
                    refined[index] = seeds[k]
        scores = self.scores(refined, observed)
        selected = self.distinct(refined, scores, 3)
        return [dict(pose=refined[k], score=float(scores[k]), accepted=True) for k in selected]

    def track(self, hypotheses: list[dict], movement: np.ndarray, observed: Observation) -> list[dict]:
        """영상 상대 이동으로 예측하고 동일한 국소 면 정합으로 후보를 갱신한다."""
        output = []
        for previous in hypotheses:
            predicted = previous['pose']@movement
            pose, accepted = self.refine(observed, predicted, local=True)
            score = float(self.scores(pose[None], observed)[0])
            output.append(dict(pose=pose, score=.65*previous['score']+.35*score, accepted=accepted, instant_score=score))
        return sorted(output, key=lambda h: h['score'])
