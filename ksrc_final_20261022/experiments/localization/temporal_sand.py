# ================================================================
# temporal_sand.py
# 설명: 이전 모래 영역의 영상 이동으로 현재 검출의 탐색 범위를 제한하는 실험용 추적기.
# 사용법: from experiments.localization.temporal_sand import TemporalSand
# ================================================================
from __future__ import annotations
import cv2
import numpy as np


class TemporalSand:
    """현재 검출만 출력하며 특징점 추적이 끊기면 이전 마스크를 버리고 재획득한다."""

    def __init__(self) -> None:
        self.previous: tuple[float, np.ndarray, np.ndarray] | None = None

    def update(self, stamp: float, gray: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, dict]:
        """GT·지도 정답 없이 왕복 광류로 모래 후보를 시간상 연결한다."""
        if gray.ndim != 2 or gray.dtype != np.uint8 or candidate.shape != gray.shape:
            raise ValueError('matching grayscale image and mask required')
        if not np.isfinite(stamp) or (self.previous is not None and stamp <= self.previous[0]):
            raise ValueError('finite increasing timestamp required')
        current = (candidate>0).astype(np.uint8)
        result = current.copy(); info = dict(tracked=False, reason='new_track')
        if not current.any():
            self.previous = float(stamp), gray.copy(), current.copy()
            return current, dict(tracked=False, reason='no_current_detection')
        if self.previous is not None:
            time, previous, mask = self.previous
            if stamp-time > 1.5:
                info['reason'] = 'time_gap_reset'
            else:
                info['reason'] = 'flow_reset'
                points = cv2.goodFeaturesToTrack(previous, maxCorners=250, qualityLevel=.01,
                                                minDistance=5, mask=mask*255)
                info['reason'] = 'few_image_features'
                info['sampled_points'] = 0 if points is None else len(points)
                if points is not None and len(points) >= 10:
                    forward, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, points, None, winSize=(21,21), maxLevel=2)
                    backward, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, previous, forward, None, winSize=(21,21), maxLevel=2)
                    valid = (status.ravel()>0)&(back_status.ravel()>0)&(np.linalg.norm((backward-points).reshape(-1,2),axis=1)<=1.)
                    info['reason'] = 'few_reversible_points'
                    info['reversible_points'] = int(valid.sum())
                    if valid.sum() >= 10 and valid.mean() >= .5:
                        # 지면의 원근 이동을 표현한다. 단일 평면 가설이 맞지 않으면 재획득한다.
                        homography, inliers = cv2.findHomography(points[valid], forward[valid], method=cv2.RANSAC, ransacReprojThreshold=2.)
                        info['reason'] = 'no_consistent_plane_motion'
                        if homography is not None and inliers.mean() >= .7:
                            predicted = cv2.warpPerspective(mask, homography, (gray.shape[1],gray.shape[0]), flags=cv2.INTER_NEAREST)
                            search = cv2.dilate(predicted, np.ones((21,21),np.uint8))
                            retained = current & search
                            info['reason'] = 'current_detection_moved_outside_prediction'
                            # 검출 대부분을 잃었을 때는 잘못된 warp를 계속 누적하지 않는다.
                            if retained.sum() >= .5*current.sum():
                                result = retained
                                info = dict(tracked=True, reason='current_detection_in_tracked_region',
                                            reversible_points=int(valid.sum()), inlier_fraction=float(inliers.mean()),
                                            removed_pixels=int(current.sum()-retained.sum()))
        self.previous = float(stamp), gray.copy(), result.copy()
        return result, info
