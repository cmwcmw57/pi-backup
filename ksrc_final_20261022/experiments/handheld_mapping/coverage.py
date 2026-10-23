# ================================================================
# coverage.py
# 설명: 고정 크기 관측 격자와 정확한 stereo 시각의 bounded FIFO 입력을 관리한다.
# 사용법: from experiments.handheld_mapping.coverage import CoverageGrid, StereoMailbox
# ================================================================
from collections import OrderedDict, deque
from pathlib import Path
import threading
import time

import numpy as np

from experiments.handheld_mapping.reconstruct import transform


class CoverageGrid:
    """초기 camera optical X-Z 평면에 실제 관측 표면과 시점 변화 횟수를 투영한다."""

    def __init__(self, bounds: tuple = (-10., 10., -10., 10.), resolution: float = .05,
                 gravity_aligned: bool = False) -> None:
        self.axes = (0,1) if gravity_aligned else (0,2)
        self.gravity_aligned = gravity_aligned
        self.bounds = tuple(float(v) for v in bounds)
        self.resolution = float(resolution)
        xmin, xmax, zmin, zmax = self.bounds
        if (not np.isfinite([*self.bounds, resolution]).all() or resolution <= 0
                or xmax <= xmin or zmax <= zmin):
            raise ValueError('invalid coverage bounds')
        shape = (int(np.ceil((zmax-zmin)/resolution)), int(np.ceil((xmax-xmin)/resolution)))
        if shape[0]*shape[1] > 250_000:
            raise ValueError('coverage cell limit')
        self.counts = np.zeros(shape, np.uint16)
        self.view_x = np.full(shape, np.nan, np.float32)
        self.view_z = np.full(shape, np.nan, np.float32)
        self.view_yaw = np.full(shape, np.nan, np.float32)
        self.map_id = None
        self.path = deque(maxlen=2000)
        self.camera = None
        self.outside_points = 0

    def update_pose(self, map_id: int, pose: np.ndarray) -> None:
        """depth가 빠져도 카메라 위치는 갱신하며 미관측 셀을 채우지 않는다."""
        pose = transform(pose)
        if self.map_id != map_id:
            self.counts.fill(0)
            for array in (self.view_x, self.view_z, self.view_yaw): array.fill(np.nan)
            self.path.clear()
            self.outside_points = 0
            self.map_id = map_id
        x, z = pose[list(self.axes),3]
        yaw = float(np.arctan2(pose[self.axes[0],2], pose[self.axes[1],2]))
        self.camera = (float(x), float(z), yaw)
        if not self.path or np.linalg.norm(np.array(self.path[-1])-[x,z]) >= .02:
            self.path.append((float(x),float(z)))

    def update(self, map_id: int, pose: np.ndarray, depth: np.ndarray, info: dict, stride: int = 8) -> None:
        pose = transform(pose)
        extrinsic = transform(info['T_left_depth'])
        scale = float(info['depth_scale_m'])
        k = np.asarray(info['k'], dtype=float).reshape(3,3)
        if (depth.dtype != np.uint16 or depth.shape != (info['height'], info['width'])
                or not np.isfinite(k).all() or min(k[0,0], k[1,1]) <= 0
                or not np.isfinite(scale) or not 0 < scale < .1 or not 1 <= stride <= 64):
            raise ValueError('invalid coverage depth/calibration')
        self.update_pose(map_id,pose)
        x,z,yaw = self.camera
        vv, uu = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
        distance = depth[::stride,::stride].astype(np.float32)*scale
        valid = (distance >= .25) & (distance <= 3.)
        camera_points = np.stack(((uu-k[0,2])*distance/k[0,0],
                                  (vv-k[1,2])*distance/k[1,1], distance),axis=-1)[valid]
        world_depth = pose @ extrinsic
        points = camera_points @ world_depth[:3,:3].T + world_depth[:3,3]
        points = points[:,list(self.axes)]
        xmin, xmax, zmin, zmax = self.bounds
        inside = ((points[:,0] >= xmin) & (points[:,0] < xmax)
                  & (points[:,1] >= zmin) & (points[:,1] < zmax))
        self.outside_points += int(np.count_nonzero(~inside))
        points = points[inside]
        cols = np.floor((points[:,0]-xmin)/self.resolution).astype(int)
        rows = np.floor((points[:,1]-zmin)/self.resolution).astype(int)
        indices = np.unique(rows*self.counts.shape[1]+cols)
        count = self.counts.ravel()
        px, pz, pyaw = self.view_x.ravel(), self.view_z.ravel(), self.view_yaw.ravel()
        difference = np.arctan2(np.sin(yaw-pyaw[indices]), np.cos(yaw-pyaw[indices]))
        novel = ((count[indices] == 0) | (np.hypot(x-px[indices], z-pz[indices]) >= .1)
                 | (np.abs(difference) >= np.deg2rad(10)))
        indices = indices[novel]
        count[indices] = np.minimum(count[indices].astype(np.uint32)+1, 65535).astype(np.uint16)
        px[indices], pz[indices], pyaw[indices] = x, z, yaw

    def snapshot(self) -> dict:
        image = np.empty((*self.counts.shape,3), np.uint8)
        image[:] = [45,53,63]
        image[(self.counts > 0) & (self.counts < 3)] = [222,177,63]
        image[self.counts >= 3] = [53,175,133]
        return {'map_id': self.map_id, 'image': image, 'bounds': self.bounds,
                'gravity_aligned':self.gravity_aligned,
                'coordinate_system':'gravity_z_up_XY' if self.gravity_aligned else 'initial_camera_optical_XZ_not_gravity_aligned',
                'resolution': self.resolution, 'path': list(self.path), 'camera': self.camera,
                'observed_cells': int(np.count_nonzero(self.counts)),
                'max_observations': int(self.counts.max()), 'outside_points': self.outside_points,
                'grid_bytes': sum(array.nbytes for array in (self.counts,self.view_x,self.view_z,self.view_yaw))}


class StereoMailbox:
    """저장된 경로를 FIFO로 보유하고 포화 시 새 쌍만 버려 대기 구간을 보존한다."""

    PAIR_LIMIT = 256
    UNMATCHED_LIMIT = 256
    DEPTH_LIMIT = 512

    def __init__(self, run: Path) -> None:
        self.run = Path(run)
        self.condition = threading.Condition()
        self.frames = {side: OrderedDict() for side in ('left','right')}
        self.depths = deque(maxlen=self.DEPTH_LIMIT)
        self.pairs = deque()
        self.closed = False
        self.dropped = 0
        self.unmatched_dropped = 0

    def offer(self, header: dict) -> None:
        side, stamp = header['stream'], header['stamp_ns']
        path = str(self.run/'raw'/side/f'{stamp}.png')
        with self.condition:
            if self.closed: return
            if side == 'depth':
                self.depths.append({**header,'path':path})
                self.condition.notify_all()
                return
            own = self.frames[side]
            other = self.frames['right' if side == 'left' else 'left']
            own[stamp] = {'path':path,'pi_receipt_monotonic_ns':header.get('pi_receipt_monotonic_ns'),
                          'laptop_receipt_monotonic_ns':header.get('laptop_receipt_monotonic_ns')}
            if stamp in other:
                records = {side: own.pop(stamp), 'right' if side == 'left' else 'left':other.pop(stamp)}
                paths = {key:value['path'] for key,value in records.items()}
                if len(self.pairs) >= self.PAIR_LIMIT:
                    self.dropped += 1
                else:
                    # 수신 큐 대기도 지연에 포함하도록 더 오래된 노트북 수신 시각을 보존한다.
                    received = [value['laptop_receipt_monotonic_ns'] for value in records.values()
                                if value['laptop_receipt_monotonic_ns'] is not None]
                    offered_at = time.monotonic()
                    self.pairs.append({'stamp_ns':stamp,**paths,
                                       'offered_at':min(received)/1e9 if received else offered_at,
                                       'pi_receipt_monotonic_ns':records['left']['pi_receipt_monotonic_ns']})
                    self.condition.notify_all()
            while len(own) > self.UNMATCHED_LIMIT:
                own.popitem(last=False)
                self.unmatched_dropped += 1

    def take(self, timeout: float = .1) -> dict | None:
        with self.condition:
            self.condition.wait_for(lambda: self.closed or self.pairs, timeout)
            return self.pairs.popleft() if self.pairs else None

    def depth_for(self, stamp: int) -> dict | None:
        with self.condition:
            closest = min(self.depths, key=lambda row: abs(row['stamp_ns']-stamp), default=None)
            return dict(closest) if closest and abs(closest['stamp_ns']-stamp) <= 2_000_000 else None

    def snapshot(self) -> dict:
        with self.condition:
            return {'queued_pairs':len(self.pairs), 'dropped_pairs':self.dropped,
                    'pair_limit':self.PAIR_LIMIT, 'overflow_policy':'drop_new',
                    'unmatched_frames':sum(len(rows) for rows in self.frames.values()),
                    'unmatched_dropped':self.unmatched_dropped, 'depth_metadata':len(self.depths)}

    def close(self) -> None:
        with self.condition:
            self.closed = True
            self.condition.notify_all()
