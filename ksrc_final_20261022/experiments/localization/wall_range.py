# ================================================================
# wall_range.py
# 설명: 고정 STL 벽 지도와 스테레오 깊이로 벽 거리(R)·직선(L)·모서리(C)를 만들어 ESKF의 x,y,yaw를 보정한다.
#       ~/rover의 wall_range_fusion.py(2026-10-16)에서 핵심만 가져온 시뮬 전용 실험 코드이며 주행 경로에는 연결하지 않았다.
# 사용법:
#   from experiments.localization.wall_range import Camera, Scene, stereo_depth, column_scan, range_update
# ================================================================
from __future__ import annotations

import struct
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import chi2

from experiments.localization.inertial_filter import InertialFilter

FENCE_STL = Path(__file__).parent/'assets'/'gyusa_fence40.stl'
STL_UNIT_M = .1                  # 원본 STL 한 단위가 0.1 m(경쟁장 bounds 25~65 → 2.5~6.5 m)
DEPTH_RANGE = (.3, 3.)


@dataclass
class Camera:
    """왼쪽 카메라 내부 파라미터, 좌우 기선, base에서 본 왼쪽 광학 프레임이다."""
    fx: float
    fy: float
    cx: float
    cy: float
    baseline: float
    base_from_camera: np.ndarray

    @classmethod
    def from_sim(cls, calibration: dict) -> 'Camera':
        """시뮬 stereo `calibration.json`에서 만든다."""
        return cls(calibration['fx'], calibration['fy'], calibration['cx'], calibration['cy'],
                   abs(float(np.asarray(calibration['T_left_right'])[0, 3])), np.asarray(calibration['T_base_left_optical'], float))


def load_stl(path: Path = FENCE_STL, unit_m: float = STL_UNIT_M) -> np.ndarray:
    """바이너리 STL을 (n,3,3) 삼각형(m)으로 읽는다."""
    data = Path(path).read_bytes()
    count = struct.unpack('<I', data[80:84])[0]
    if len(data) != 84+50*count:
        raise ValueError('not a binary STL')
    values = np.frombuffer(data, dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]), count=count, offset=84)
    return values['v'].astype(float)*unit_m


def load_scene(triangles: np.ndarray | None = None) -> tuple[np.ndarray, list[dict], np.ndarray]:
    """삼각형, 수직 벽면(평면+선분 범위), 두 벽면이 만나는 수직 모서리 (x,y)를 만든다."""
    triangles = load_stl() if triangles is None else triangles
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    planes: dict = {}
    for tri, normal in zip(triangles, normals):
        if abs(normal[2]) > .2:
            continue
        n2 = normal[:2]/np.linalg.norm(normal[:2])
        if n2[0] < -1e-9 or (abs(n2[0]) < 1e-9 and n2[1] < 0):
            n2 = -n2
        d = float(n2 @ tri[0, :2])
        key = (round(np.degrees(np.arctan2(n2[1], n2[0])), 0), round(d, 3))
        along = np.array([-n2[1], n2[0]]) @ tri[:, :2].T
        entry = planes.setdefault(key, {'n': n2, 'd': d, 'lo': 1e9, 'hi': -1e9})
        entry['lo'], entry['hi'] = min(entry['lo'], along.min()), max(entry['hi'], along.max())
    plane_list = list(planes.values())
    corners = []
    for a, b in product(plane_list, plane_list):
        if abs(a['n'] @ b['n']) > .1 or id(a) >= id(b):
            continue
        point = np.linalg.solve(np.array([a['n'], b['n']]), [a['d'], b['d']])
        if all(p['lo']-.03 <= np.array([-p['n'][1], p['n'][0]]) @ point <= p['hi']+.03 for p in (a, b)):
            corners.append(point)
    unique = {tuple(np.round(c, 2)): c for c in corners}
    return triangles, plane_list, np.array(list(unique.values()))


def raycast(origin: np.ndarray, directions: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """origin에서 각 방향(z=1 정규화)으로 쏜 ray의 (삼각형 교차 광축 깊이, 삼각형 번호)다. 없으면 inf."""
    e1, e2 = triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]
    p = np.cross(directions[:, None], e2[None])
    det = np.einsum('tj,ntj->nt', e1, p)
    inv = np.divide(1., det, out=np.zeros_like(det), where=abs(det) > 1e-10)
    off = origin-triangles[:, 0]
    q = np.cross(off, e1)
    u = np.einsum('tj,ntj->nt', off, p)*inv
    v = np.einsum('nj,tj->nt', directions, q)*inv
    t = np.einsum('tj,tj->t', e2, q)[None]*inv
    valid = (abs(det) > 1e-10) & (u >= 0) & (v >= 0) & (u+v <= 1) & (t > 1e-3)
    t = np.where(valid, t, np.inf)
    return t.min(axis=1), t.argmin(axis=1)


class Scene:
    """STL 지도와 카메라 모델로 추정 pose에서 벽이 보일 깊이를 예측한다."""

    def __init__(self, camera: Camera, triangles: np.ndarray | None = None, mesh: object | None = None) -> None:
        self.triangles, self.planes, self.corners = load_scene(triangles)
        self._caster = None
        if mesh is not None:   # 경기장 전체 메쉬(내부 경사로 벽 포함)는 Open3D 가속 구조로 쏜다
            import open3d as o3d
            self._caster = o3d.t.geometry.RaycastingScene()
            self._caster.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            mesh_triangles = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)]
            self.triangles = mesh_triangles
        cross = np.cross(self.triangles[:, 1]-self.triangles[:, 0], self.triangles[:, 2]-self.triangles[:, 0])
        self.vertical = abs(cross[:, 2])/np.maximum(np.linalg.norm(cross, axis=1), 1e-12) < .2
        self.camera = camera
        self.left = camera.base_from_camera
        self.grid = self._rays(np.arange(8, 640, 8.), np.arange(8, 480, 8.))
        self.columns = np.arange(14, 640, 12.)
        self.rows = np.arange(2, 480, 3.)
        self.fine = self._rays(self.columns, self.rows)
        self.fine_shape = (len(self.rows), len(self.columns))
        self.sparse = self._rays(np.linspace(40, 600, 16), np.array([150., 240., 330.]))

    def _rays(self, us: np.ndarray, vs: np.ndarray) -> np.ndarray:
        c = self.camera
        uu, vv = np.meshgrid(us, vs)
        return np.stack([(uu.ravel()-c.cx)/c.fx, (vv.ravel()-c.cy)/c.fy, np.ones(uu.size)], axis=1)

    def depth(self, base_pose: np.ndarray, rays: np.ndarray, walls_only: bool) -> np.ndarray:
        """base pose에서 본 광축 깊이(가림·범위 밖은 nan). walls_only면 수직 벽면에 맞은 ray만 남긴다."""
        camera = base_pose @ self.left
        directions = rays @ camera[:3, :3].T
        if self._caster is not None:
            import open3d as o3d
            query = np.concatenate([np.broadcast_to(camera[:3, 3], directions.shape), directions], axis=1).astype(np.float32)
            hit = self._caster.cast_rays(o3d.core.Tensor(query))
            z, index = hit['t_hit'].numpy().astype(float), hit['primitive_ids'].numpy().astype(np.int64)
            index = np.where(index < len(self.vertical), index, 0)
        else:
            z, index = raycast(camera[:3, 3], directions, self.triangles)
        keep = (z >= DEPTH_RANGE[0]) & (z <= DEPTH_RANGE[1])
        if walls_only:
            keep &= self.vertical[index]
        return np.where(keep, z, np.nan)


def stereo_depth(sgbm, left_image: np.ndarray, right_image: np.ndarray, camera: Camera) -> np.ndarray:
    """수동 스테레오(SGBM, 프로젝터 없음) 깊이 영상(z, 무효는 nan)이다."""
    disparity = sgbm.compute(left_image, right_image).astype(np.float32)/16.
    return np.where(disparity > 1, camera.fx*camera.baseline/np.maximum(disparity, 1e-3), np.nan)


def column_scan(scene: Scene, base_pose: np.ndarray, depth_image: np.ndarray, bias: float = 1.03):
    """열마다 STL 예측 벽 픽셀의 (측정-예측) 깊이 잔차 중앙값을 빔 하나로 만든다. (rays, z_meas)를 돌려준다."""
    c = scene.camera
    z_pred = scene.depth(base_pose, scene.fine, True).reshape(scene.fine_shape)
    u = np.round(scene.fine[:, 0]*c.fx+c.cx).astype(int).clip(0, 639)
    v = np.round(scene.fine[:, 1]*c.fy+c.cy).astype(int).clip(0, 479)
    z_meas = (depth_image[v, u]*bias).reshape(scene.fine_shape)
    residual = z_meas-z_pred
    fine = scene.fine.reshape(scene.fine_shape[0], scene.fine_shape[1], 3)
    rays, values = [], []
    for column in range(scene.fine_shape[1]):
        rows = np.flatnonzero(np.isfinite(z_pred[:, column]) & np.isfinite(z_meas[:, column]))
        if len(rows) < 5:
            continue
        r = residual[rows, column]
        center = np.median(r)
        if abs(center) > max(.12, .12*np.median(z_pred[rows, column])) or np.median(abs(r-center)) > .06:
            continue
        mid = rows[len(rows)//2]
        rays.append(fine[mid, column])
        values.append(z_pred[mid, column]+center)
    return np.array(rays).reshape(-1, 3), np.array(values)


def add_depth_noise(z: np.ndarray, rng: np.random.Generator, scale: float = 1., dropout: float = .15,
                    outlier: float = .03) -> np.ndarray:
    """깊이에 D435식 잡음(σ=scale·(1%·z+5 mm)), 결측(dropout), 이상치(±0.3 m, outlier 비율)를 입힌다."""
    z = np.asarray(z, float).copy()
    z = z+rng.normal(0, (.01*z+.005)*scale)
    z[rng.random(len(z)) < dropout] = np.nan
    bad = rng.random(len(z)) < outlier
    z[bad] += rng.uniform(-.3, .3, bad.sum())
    return z


def fit_lines(points: np.ndarray, rng: np.random.Generator) -> list[dict]:
    """2D 점군에서 순차 RANSAC으로 직선을 뽑는다. 직선은 n·p=rho(rho>0), 점 범위(span)를 가진다."""
    pts, lines = points.copy(), []
    for _ in range(4):
        if len(pts) < 10:
            break
        best = None
        for _ in range(60):
            i, j = rng.choice(len(pts), 2, replace=False)
            direction = pts[j]-pts[i]
            if np.linalg.norm(direction) < .08:
                continue
            normal = np.array([-direction[1], direction[0]])/np.linalg.norm(direction)
            inliers = abs((pts-pts[i]) @ normal) < .03
            if best is None or inliers.sum() > best.sum():
                best = inliers
        if best is None or best.sum() < 10:
            break
        group = pts[best]
        center = group.mean(axis=0)
        normal = np.linalg.svd(group-center)[2][1]
        rho = float(normal @ center)
        if rho < 0:
            normal, rho = -normal, -rho
        along = np.array([-normal[1], normal[0]]) @ group.T
        if float(along.max()-along.min()) >= .2:
            lines.append({'n': normal, 'rho': rho, 'alpha': float(np.arctan2(normal[1], normal[0])),
                          'lo': float(along.min()), 'hi': float(along.max()), 'count': int(best.sum())})
        pts = pts[~best]
    return lines


def level_points(scene: Scene, base_pose: np.ndarray, rays: np.ndarray, z: np.ndarray) -> np.ndarray:
    """깊이 점을 추정 자세로 중력 정렬한 뒤 yaw를 뺀 로버 수평 좌표(xy)로 바꾼다."""
    ok = np.isfinite(z)
    base_points = (rays[ok]*z[ok, None]) @ scene.left[:3, :3].T+scene.left[:3, 3]
    yaw = Rotation.from_matrix(base_pose[:3, :3]).as_euler('xyz')[2]
    rot = Rotation.from_euler('z', -yaw).as_matrix() @ base_pose[:3, :3]
    return (base_points @ rot.T)[:, :2]


def detect_corners(lines: list[dict]) -> list[np.ndarray]:
    """거의 직각인 두 직선의 교점이 두 선분 끝 근처에 있으면 모서리로 본다."""
    corners = []
    for a, b in product(lines, lines):
        if id(a) >= id(b):
            continue
        angle = abs(((a['alpha']-b['alpha'])+np.pi) % (2*np.pi)-np.pi)
        if abs(angle-np.pi/2) > np.deg2rad(10):
            continue
        point = np.linalg.solve(np.array([a['n'], b['n']]), [a['rho'], b['rho']])
        if all(l['lo']-.15 <= np.array([-l['n'][1], l['n'][0]]) @ point <= l['hi']+.15 for l in (a, b)):
            corners.append(point)
    return corners


def wrap(angle: float) -> float:
    """각도를 [-pi, pi]로 접는다."""
    return float((angle+np.pi) % (2*np.pi)-np.pi)


def base_state(filter_: InertialFilter, imu_from_base: np.ndarray) -> tuple[np.ndarray, float]:
    """filter의 base pose와 yaw를 돌려준다."""
    base = filter_.pose @ imu_from_base
    return base, float(Rotation.from_matrix(base[:3, :3]).as_euler('xyz')[2])


def planar_update(filter_: InertialFilter, residual: np.ndarray, jac3: np.ndarray, noise: np.ndarray) -> dict:
    """jac3(열: x,y,yaw)를 21차 상태로 옮기고 x·y·yaw 상태만 고치도록 gain을 투영해 적용한다."""
    jac = np.zeros((len(residual), 21))
    jac[:, 0:2] = jac3[:, 0:2]
    yaw_axis = filter_.pose[:3, :3].T @ [0., 0., 1.]
    jac[:, 6:9] = np.outer(jac3[:, 2], yaw_axis)
    projection = np.zeros((21, 21))
    projection[:2, :2] = np.eye(2)
    projection[3:5, 3:5] = np.eye(2)
    projection[15:17, 15:17] = np.eye(2)
    for start, rotation in ((6, filter_.pose[:3, :3]), (18, filter_.clone[:3, :3])):
        axis = rotation.T @ [0., 0., 1.]
        projection[start:start+3, start:start+3] = np.outer(axis, axis)
    projection[9:12, 9:12] = np.outer(yaw_axis, yaw_axis)
    return filter_.update(residual, jac, noise, float(chi2.ppf(.99, len(residual))), projection)


def range_update(filter_: InertialFilter, scene: Scene, imu_from_base: np.ndarray, z_meas: np.ndarray, rays: np.ndarray,
                 rel_sigma: float = .05) -> dict:
    """측정 깊이와 현재 추정 pose의 ray casting 예측 깊이의 차이로 x,y,yaw를 보정한다(R)."""
    base, _ = base_state(filter_, imu_from_base)

    def predict(dx: float, dy: float, dyaw: float) -> np.ndarray:
        moved = base.copy()
        moved[:3, :3] = Rotation.from_euler('z', dyaw).as_matrix() @ base[:3, :3]
        moved[:2, 3] += (dx, dy)
        return scene.depth(moved, rays, True)

    z_pred = predict(0, 0, 0)
    use = np.isfinite(z_meas) & np.isfinite(z_pred) & (abs(z_meas-z_pred) < .25)
    if use.sum() < 6:
        return {'accepted': False, 'reason': 'few_rays', 'n': int(use.sum())}
    eps = 1e-4
    jac3 = np.stack([(predict(*(np.eye(3)[i]*eps))-z_pred)/eps for i in range(3)], axis=1)[use]
    ok = np.isfinite(jac3).all(axis=1)
    use_idx = np.flatnonzero(use)[ok]
    sigma = rel_sigma*z_meas[use_idx]+.005
    info = planar_update(filter_, (z_meas-z_pred)[use_idx], jac3[ok], np.diag((sigma*2.)**2))
    return {**info, 'n': int(len(use_idx))}


def line_update(filter_: InertialFilter, scene: Scene, imu_from_base: np.ndarray, lines: list[dict]) -> dict:
    """검출한 벽 직선 (각도, 거리)를 STL 벽면과 짝지어 x,y,yaw를 보정한다(L)."""
    base, yaw = base_state(filter_, imu_from_base)
    xy = base[:2, 3]
    rows, jacs, sigmas = [], [], []
    for line in lines:
        best = None
        for plane in scene.planes:
            sign = 1. if plane['d']-plane['n'] @ xy > 0 else -1.
            normal = sign*plane['n']
            rho_pred = float(sign*(plane['d']-plane['n'] @ xy))
            d_alpha, d_rho = wrap(line['alpha']-wrap(np.arctan2(normal[1], normal[0])-yaw)), line['rho']-rho_pred
            if abs(d_alpha) < np.deg2rad(8) and abs(d_rho) < .25 and (best is None or abs(d_rho) < abs(best[1])):
                best = (d_alpha, d_rho, normal)
        if best is not None:
            d_alpha, d_rho, normal = best
            rows += [d_alpha, d_rho]
            jacs += [[0, 0, -1.], [-normal[0], -normal[1], 0.]]
            sigmas += [np.deg2rad(2.), .03+.01*line['rho']]
    if not rows:
        return {'accepted': False, 'reason': 'no_match', 'n': 0}
    return {**planar_update(filter_, np.array(rows), np.array(jacs), np.diag(np.array(sigmas)**2)), 'n': len(rows)//2}


def corner_update(filter_: InertialFilter, scene: Scene, imu_from_base: np.ndarray, corners: list[np.ndarray]) -> dict:
    """검출한 모서리 점(로버 수평 좌표)을 가장 가까운 STL 모서리와 짝지어 x,y,yaw를 보정한다(C)."""
    base, yaw = base_state(filter_, imu_from_base)
    xy = base[:2, 3]

    def predicted(corner: np.ndarray, dx: float, dy: float, dyaw: float) -> np.ndarray:
        c, s = np.cos(yaw+dyaw), np.sin(yaw+dyaw)
        return np.array([[c, s], [-s, c]]) @ (corner-xy-(dx, dy))

    rows, jacs = [], []
    for measured in corners:
        index = int(np.argmin([np.linalg.norm(predicted(c, 0, 0, 0)-measured) for c in scene.corners]))
        corner = scene.corners[index]
        base_pred = predicted(corner, 0, 0, 0)
        if np.linalg.norm(base_pred-measured) > .35:
            continue
        rows += list(measured-base_pred)
        eps = 1e-5
        jacs += list(np.stack([(predicted(corner, *(np.eye(3)[i]*eps))-base_pred)/eps for i in range(3)], axis=1))
    if not rows:
        return {'accepted': False, 'reason': 'no_match', 'n': 0}
    return {**planar_update(filter_, np.array(rows), np.array(jacs), np.eye(len(rows))*.05**2), 'n': len(rows)//2}


# ---------------------------------------------------------------- 벽 윗선(허공-벽 경계) 거리 관측
VOID_GRAY = 75        # 이보다 어두우면 벽 너머 허공으로 본다(시뮬 배경 약 50)
GATE_LIMITS = (1., 1.5, 2., 3.)


def detect_wall_top(gray: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """열마다 위쪽 허공이 끝나고 벽이 시작하는 행(윗선)을 찾는다. 맨 위가 허공이 아닌 열은 nan이다."""
    import cv2
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    out = np.full(len(columns), np.nan)
    for j, column in enumerate(np.asarray(columns, int)):
        values = blurred[:, column]
        if not (values[:6] < VOID_GRAY).all():
            continue
        below = np.flatnonzero(values >= VOID_GRAY)
        if len(below):
            out[j] = below[0]
    return out


def gate_wall_top(distance: np.ndarray, min_columns: int = 8) -> tuple[np.ndarray, float]:
    """예측 거리 1 m 이내 열만 쓰고, 열이 min_columns보다 적으면 1.5·2·3 m로 단계적으로 넓힌다. 뒤쪽 먼 벽은 기본에서 빠진다."""
    distance = np.asarray(distance, float)
    keep = np.zeros(len(distance), bool)
    limit = GATE_LIMITS[-1]
    for limit in GATE_LIMITS:
        keep = np.isfinite(distance) & (distance <= limit)
        if keep.sum() >= min_columns:
            break
    return keep, limit


def wall_top_residual(measured: np.ndarray, predicted: np.ndarray, limit_px: float = 15.) -> tuple[np.ndarray, np.ndarray]:
    """(측정-예측) 픽셀 잔차 중 한계 안인 열의 잔차와 그 마스크를 돌려준다. 윗선이 아닌 것에 걸린 열은 버린다."""
    delta = np.asarray(measured, float)-np.asarray(predicted, float)
    keep = np.isfinite(delta) & (abs(delta) <= limit_px)
    return delta[keep], keep


def wall_top_agrees(measured: np.ndarray, predicted: np.ndarray, tolerance_px: float = 4., min_fraction: float = .6) -> bool:
    """검출과 예측이 둘 다 있는 열 중 tolerance_px 이내인 비율이 min_fraction 이상일 때만 그 프레임의 윗선을 믿는다(지도에 없는 구조물 거부)."""
    both = np.isfinite(measured) & np.isfinite(predicted)
    if both.sum() < 4:
        return False
    return float(np.mean(abs(measured[both]-predicted[both]) <= tolerance_px)) >= min_fraction


def predicted_wall_top(scene: Scene, base_pose: np.ndarray, columns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """추정 pose에서 STL 벽이 처음 보이는 행(윗선)과 그 광축 거리를 열마다 구한다(성기게 찾고 4행 안에서 0.25픽셀 단위로 다듬는다)."""
    c = scene.camera
    columns = np.asarray(columns, float)
    coarse = np.arange(0, 480, 4.)
    z = scene.depth(base_pose, scene._rays(columns, coarse), False).reshape(len(coarse), len(columns))
    rows, dist = np.full(len(columns), np.nan), np.full(len(columns), np.nan)
    for j in range(len(columns)):
        hit = np.flatnonzero(np.isfinite(z[:, j]))
        if not len(hit):
            continue
        k = hit[0]
        candidates = np.arange(max(0., coarse[k]-3.), coarse[k]+.001, .25)
        rays = np.stack([np.full(len(candidates), (columns[j]-c.cx)/c.fx), (candidates-c.cy)/c.fy, np.ones(len(candidates))], axis=1)
        fine = scene.depth(base_pose, rays, False)
        found = np.flatnonzero(np.isfinite(fine))
        if len(found):
            rows[j], dist[j] = candidates[found[0]], fine[found[0]]
    return rows, dist


def wall_top_update(filter_: InertialFilter, scene: Scene, imu_from_base: np.ndarray, measured: np.ndarray, columns: np.ndarray,
                    min_columns: int = 8, limit_px: float = 15., sigma_px: float = 3., agree_px: float = 8.) -> dict:
    """검출한 벽 윗선 행을 예측과 비교해 x,y,yaw를 보정한다. 거리 제한 단계가 넓을수록 잡음을 키운다."""
    base, _ = base_state(filter_, imu_from_base)
    predicted, distance = predicted_wall_top(scene, base, columns)
    usable = np.isfinite(predicted) & (predicted > 6) & np.isfinite(measured)
    gate, limit = gate_wall_top(np.where(usable, distance, np.nan), min_columns)
    if not wall_top_agrees(np.where(gate, measured, np.nan), predicted, agree_px):
        return {'accepted': False, 'reason': 'map_mismatch', 'n': 0, 'limit_m': limit}
    residual, close = wall_top_residual(np.where(gate, measured, np.nan), predicted, limit_px)
    if close.sum() < 4:
        return {'accepted': False, 'reason': 'few_columns', 'n': int(close.sum()), 'limit_m': limit}

    def at(dx: float, dy: float, dyaw: float) -> np.ndarray:
        moved = base.copy()
        moved[:3, :3] = Rotation.from_euler('z', dyaw).as_matrix() @ base[:3, :3]
        moved[:2, 3] += (dx, dy)
        return predicted_wall_top(scene, moved, np.asarray(columns)[close])[0]

    eps = .01   # 예측 행이 0.25픽셀 단위라 미분 간격을 1 cm·0.01 rad로 둔다
    jac3 = np.stack([(at(*(np.eye(3)[i]*eps))-at(*(-np.eye(3)[i]*eps)))/(2*eps) for i in range(3)], axis=1)
    ok = np.isfinite(jac3).all(axis=1)
    if ok.sum() < 4:
        return {'accepted': False, 'reason': 'no_jacobian', 'n': int(ok.sum()), 'limit_m': limit}
    noise = np.eye(int(ok.sum()))*(sigma_px*limit)**2
    return {**planar_update(filter_, residual[ok], jac3[ok], noise), 'n': int(ok.sum()), 'limit_m': limit}
