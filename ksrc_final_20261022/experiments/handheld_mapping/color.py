# ================================================================
# color.py
# 설명: 제한된 디스크 인덱스로 RGB를 매칭하고 depth 점에 보정·가림 검증 후 색을 투영한다.
# 사용법: from experiments.handheld_mapping.color import ColorMatcher, project_depth_color
# ================================================================
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from typing import Iterator

import cv2
import numpy as np

from experiments.handheld_mapping.recording import disk_index


def color_calibration(info: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """지원하는 RGB pinhole 보정과 depth→color rigid 변환을 검증한다."""
    from experiments.handheld_mapping.reconstruct import transform
    try:
        k = np.asarray(info['k'], dtype=float).reshape(3, 3)
        d = np.asarray(info.get('d', []), dtype=float)
        model = info.get('distortion_model', '')
        if model not in ('plumb_bob', 'rational_polynomial', ''):
            raise ValueError('unsupported color distortion model')
        if (d.ndim != 1 or not np.isfinite(d).all()
                or (model == 'plumb_bob' and len(d) not in (0, 4, 5))
                or (model == 'rational_polynomial' and len(d) != 8)
                or (not model and np.any(d))):
            raise ValueError('invalid color distortion coefficients')
        if (not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0
                or not np.allclose(k[2], [0, 0, 1]) or not np.allclose(k[[0,1],[1,0]], 0)
                or int(info['width']) <= 0 or int(info['height']) <= 0):
            raise ValueError('invalid color intrinsics')
        return k, d, transform(info['T_color_depth'])
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError('color intrinsics and T_color_depth required') from error


def project_depth_color(depth: np.ndarray, depth_info: dict, color_bgr: np.ndarray,
                        color_info: dict) -> tuple[np.ndarray, np.ndarray]:
    """왜곡된 RGB에 depth 점을 투영하며 동일 pixel의 뒤쪽 점은 neutral로 남긴다."""
    ck, distortion, extrinsic = color_calibration(color_info)
    if (color_bgr.dtype != np.uint8 or color_bgr.shape !=
            (int(color_info['height']), int(color_info['width']), 3)):
        raise ValueError('color file/profile mismatch')
    k = np.asarray(depth_info['k'], dtype=float).reshape(3, 3)
    yy, xx = np.indices(depth.shape)
    z = depth.astype(float)*float(depth_info['depth_scale_m'])
    points = np.stack(((xx-k[0,2])*z/k[0,0], (yy-k[1,2])*z/k[1,1], z), axis=-1)
    transformed = points @ extrinsic[:3,:3].T + extrinsic[:3,3]
    rgb = np.full((*depth.shape, 3), 128, np.uint8)
    valid = (depth > 0) & np.isfinite(transformed).all(axis=2) & (transformed[:,:,2] > 0)
    flat = np.flatnonzero(valid)
    if not len(flat):
        return rgb, np.zeros(depth.shape, bool)
    projected, _ = cv2.projectPoints(transformed.reshape(-1,3)[flat], np.zeros(3), np.zeros(3),
                                     ck, distortion if len(distortion) else None)
    uv = projected.reshape(-1,2)
    finite = np.isfinite(uv).all(axis=1)
    flat, uv = flat[finite], uv[finite]
    u, v = np.rint(uv).astype(np.int64).T
    inside = (u >= 0) & (v >= 0) & (u < color_bgr.shape[1]) & (v < color_bgr.shape[0])
    flat, u, v = flat[inside], u[inside], v[inside]
    pixel = v*color_bgr.shape[1]+u
    distance = transformed.reshape(-1,3)[flat,2]
    zbuffer = np.full(color_bgr.shape[0]*color_bgr.shape[1], np.inf)
    np.minimum.at(zbuffer, pixel, distance)
    visible = distance <= zbuffer[pixel]+.005
    flat, u, v = flat[visible], u[visible], v[visible]
    mask = np.zeros(depth.shape, bool)
    mask.ravel()[flat] = True
    rgb.reshape(-1,3)[flat] = color_bgr[v,u,::-1]
    return rgb, mask


class ColorMatcher:
    """SQLite의 timestamp 인덱스로 RGB 한 장씩만 읽고 시각·경로·frame을 검증한다."""

    def __init__(self, run: Path, database: sqlite3.Connection, info: dict | None) -> None:
        self.run, self.database, self.info = run, database, info
        self.count = 0
        database.execute('CREATE TABLE colors(stamp INTEGER PRIMARY KEY, row TEXT NOT NULL)')
        previous = -1
        with (run/'frames.jsonl').open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['stream'] != 'color':
                    continue
                stamp = row['stamp_ns']
                if type(stamp) is not int or not previous < stamp < 2**63:
                    raise ValueError('invalid color stamp')
                previous = stamp
                if info is not None and row.get('frame_id') != info.get('color_frame'):
                    raise ValueError('color frame identity mismatch')
                if info is not None and (row.get('width'),row.get('height')) != (info['width'],info['height']):
                    raise ValueError('color index/profile mismatch')
                if row.get('encoding') not in ('rgb8', 'bgr8'):
                    raise ValueError('invalid color encoding')
                database.execute('INSERT INTO colors VALUES (?,?)', (stamp,json.dumps(row)))
                self.count += 1
        database.commit()
        if self.count and info is not None:
            color_calibration(info)

    def match(self, stamp: int, tolerance_ns: int = 34_000_000) -> tuple[np.ndarray | None, int | None]:
        if self.info is None:
            return None, None
        rows = self.database.execute(
            'SELECT stamp,row FROM colors WHERE stamp BETWEEN ? AND ? ORDER BY ABS(stamp-?),stamp LIMIT 1',
            (stamp-tolerance_ns,stamp+tolerance_ns,stamp)).fetchone()
        if rows is None:
            return None, None
        matched, encoded = rows
        row = json.loads(encoded)
        path = (self.run/row['path']).resolve()
        if not path.is_relative_to(self.run/'raw/color'):
            raise ValueError('invalid color path')
        image = cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
        if (image is None or image.dtype != np.uint8 or image.shape !=
                (int(self.info['height']),int(self.info['width']),3)):
            raise ValueError('color file/profile mismatch')
        return image, abs(matched-stamp)


@contextmanager
def color_matcher(run: Path, parent: Path, info: dict | None) -> Iterator[ColorMatcher]:
    """수집 원본을 변경하지 않는 임시 인덱스의 수명을 융합 작업으로 제한한다."""
    with disk_index(parent) as database:
        yield ColorMatcher(run,database,info)
