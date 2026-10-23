# ================================================================
# stereo_depth_check.py
# 설명: 스테레오(SGBM) 깊이를 정답 pose의 STL 벽 깊이와 픽셀 단위로 비교해 거리별 검출률·편향·이상치를 보고한다.
#       시뮬 기록과 같은 지표를 실기 녹화(정답 위치가 있을 때)에도 적용하기 위한 진단이다.
# 사용법:
#   python -m experiments.site_sim.stereo_depth_check RUN_DIR [--every 25] [--lag-s 0.1]
#   (RUN_DIR: estimator_events.jsonl의 pose 이벤트, stereo/calibration.json, stereo/mav0/cam{0,1}/data 필요)
# ================================================================
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from experiments.localization.wall_range import Camera, Scene, stereo_depth
from experiments.site_sim.wall_replay import sgbm

BINS = ((.3, .8), (.8, 1.5), (1.5, 3.1))


def depth_agreement(truth: np.ndarray, measured: np.ndarray, min_pixels: int = 20) -> list[dict]:
    """정답 깊이 구간별로 검출률, 측정/정답 비율의 중앙값, 비율의 강건 표준편차, 10%·25% 넘게 틀린 비율을 돌려준다."""
    truth, measured = np.asarray(truth, float), np.asarray(measured, float)
    rows = []
    for low, high in BINS:
        inside = (truth >= low) & (truth < high)
        valid = inside & np.isfinite(measured)
        if valid.sum() < min_pixels:
            continue
        ratio = measured[valid]/truth[valid]
        center = float(np.median(ratio))
        rows.append({'bin': f'{low:g}-{high:g}', 'pixels': int(inside.sum()), 'detection': float(valid.sum()/inside.sum()),
                     'median_ratio': center, 'robust_sigma': float(1.4826*np.median(abs(ratio-center))),
                     'outlier_10pct': float(np.mean(abs(ratio-1) > .10)), 'outlier_25pct': float(np.mean(abs(ratio-1) > .25))})
    return rows


def check(run: Path, every: int = 25, lag_s: float = .1) -> list[dict]:
    camera = Camera.from_sim(json.loads((run/'stereo/calibration.json').read_text()))
    scene = Scene(camera)
    poses = [(e['stamp_s'], np.array(e['position_m']), np.array(e['quaternion_wxyz']))
             for e in map(json.loads, (run/'estimator_events.jsonl').read_text().splitlines()) if e['kind'] == 'pose']
    times = np.array([p[0] for p in poses])
    us, vs = np.arange(4, 640, 4.), np.arange(4, 480, 4.)
    rays = scene._rays(us, vs)
    uu, vv = np.meshgrid(us.astype(int), vs.astype(int))
    uu, vv = uu.ravel(), vv.ravel()
    matcher, truth, measured = sgbm(), [], []
    for left in sorted((run/'stereo/mav0/cam0/data').glob('*.png'))[::every]:
        right = run/'stereo/mav0/cam1/data'/left.name
        index = int(np.argmin(abs(times-(int(left.stem)*1e-9-lag_s))))
        _, position, q = poses[index]
        base = np.eye(4)
        base[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        base[:3, 3] = position
        wall = scene.depth(base, rays, True)
        ok = np.isfinite(wall)
        if ok.sum() < 30:
            continue
        depth = stereo_depth(matcher, cv2.imread(str(left), 0), cv2.imread(str(right), 0), camera)
        truth.append(wall[ok])
        measured.append(depth[vv[ok], uu[ok]])
    return depth_agreement(np.concatenate(truth), np.concatenate(measured))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--every', type=int, default=25)
    parser.add_argument('--lag-s', type=float, default=.1)
    args = parser.parse_args()
    for row in check(args.run, args.every, args.lag_s):
        print(f"gt {row['bin']} m: pixels {row['pixels']:6d} detect {row['detection']*100:3.0f}% median ratio {row['median_ratio']:.3f} "
              f"sigma {row['robust_sigma']:.3f} >10% {row['outlier_10pct']*100:3.0f}% >25% {row['outlier_25pct']*100:3.0f}%")


if __name__ == '__main__':
    main()
