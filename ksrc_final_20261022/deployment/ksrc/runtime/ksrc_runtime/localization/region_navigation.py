# ================================================================
# region_navigation.py
# 설명: 학습 없이 명시 통로 그래프와 관측 likelihood로 현재 구역·다음 통로 영역을 유지한다.
#       구동 명령을 보내지 않는 상위 관측·계획 모듈이다.
# 사용법: from ksrc_runtime.localization.region_navigation import RegionGraph, RegionBelief
# ================================================================
from __future__ import annotations

import heapq
import math

import cv2
import numpy as np


def segment_distance(point: np.ndarray, segment: np.ndarray) -> float:
    """점과 유한 선분의 거리다."""
    a, b = segment
    u = np.clip((point-a)@(b-a)/np.sum((b-a)**2), 0., 1.)
    return float(np.linalg.norm(point-a-u*(b-a)))


class RegionGraph:
    """구역 polygon과 사람이 명시한 통로만 연결하는 제안 지도다."""

    def __init__(self, regions: list[dict], portals: list[dict]) -> None:
        self.regions = {r['id']: r for r in regions}
        self.polygons = {k: np.asarray(r['polygon_xy_m'], np.float32) for k, r in self.regions.items()}
        self.adjacent = {k: set() for k in self.regions}
        self.portals: dict[tuple[int, int], np.ndarray] = {}
        for row in portals:
            a, b = row['regions']
            segment = np.asarray(row['segment_xy_m'], float)
            if a == b or a not in self.regions or b not in self.regions or segment.shape != (2, 2):
                raise ValueError('portal needs two existing regions and a 2x2 segment')
            if not np.isfinite(segment).all() or np.linalg.norm(segment[1]-segment[0]) <= 0:
                raise ValueError('portal must have finite positive width')
            self.adjacent[a].add(b);self.adjacent[b].add(a)
            self.portals[tuple(sorted((a, b)))] = segment

    def distance(self, region: int, xy: np.ndarray) -> float:
        """구역 내부는0, 외부는 polygon 경계까지의 거리다."""
        signed = cv2.pointPolygonTest(self.polygons[region], tuple(map(float, xy)), True)
        return max(0., -signed)

    def labels(self, xy: np.ndarray, overlap_m: float = 0.) -> list[int]:
        """경계 중첩을 허용한 후보 번호다. 관측된 전방 면의 번호와 다르다."""
        return [k for k in self.regions if self.distance(k, xy) <= overlap_m+1e-7]

    def route(self, start: int, goal: int, max_slope_deg: float = 20.,
              blocked: set[tuple[int, int]] | None = None) -> list[int]:
        """통로 길이와 목적 구역 경사를 사용해 Dijkstra 경로를 찾는다."""
        if start not in self.regions or goal not in self.regions:
            raise ValueError('unknown region')
        blocked = {tuple(sorted(e)) for e in (blocked or set())}
        queue, seen = [(0., [start])], set()
        while queue:
            cost, path = heapq.heappop(queue)
            node = path[-1]
            if node == goal:
                return path
            if node in seen:
                continue
            seen.add(node)
            for other in sorted(self.adjacent[node]):
                if tuple(sorted((node, other))) in blocked:
                    continue
                slope = abs(self.regions[other]['median_mesh_slope_deg'])
                if slope > max_slope_deg:
                    continue
                centers = [self.polygons[k].mean(axis=0) for k in (node, other)]
                length = float(np.linalg.norm(centers[1]-centers[0]))
                heapq.heappush(queue, (cost+length*(1.+slope/30.), path+[other]))
        return []

    def portal_target(self, current: int, following: int, xy: np.ndarray,
                      clearance_m: float = .25) -> np.ndarray:
        """선분 양끝에 여유를 남긴 영역 안에서 가장 가까운 통과점을 제안한다."""
        key = tuple(sorted((current, following)))
        if key not in self.portals:
            raise ValueError('regions are not connected by a portal')
        a, b = self.portals[key];length = float(np.linalg.norm(b-a))
        if not math.isfinite(clearance_m) or clearance_m < 0 or length <= 2*clearance_m:
            raise ValueError('portal is too narrow for the requested clearance')
        u = np.clip((xy-a)@(b-a)/length**2, clearance_m/length, 1.-clearance_m/length)
        return a+u*(b-a)


class RegionBelief:
    """시작 구역 prior와 이웃 연결·통로 근접·연속 관측으로 구역 전이를 확인한다."""

    def __init__(self, graph: RegionGraph, start: int = 1) -> None:
        if start not in graph.regions:
            raise ValueError('unknown start region')
        self.graph, self.ids = graph, list(graph.regions)
        self.current = start
        self.probabilities = np.array([float(k == start) for k in self.ids])
        self.last_stamp: float | None = None
        self.last_xy: np.ndarray | None = None
        self.pending, self.confirmations = None, 0

    def update(self, stamp_s: float, likelihood: dict[int, float], xy: np.ndarray,
               valid: bool = True) -> dict:
        """약1Hz 관측을 반영한다. 유효한 연속2회와 경계 밖125mm 진입이 필요하다."""
        xy = np.asarray(xy, float)
        if not math.isfinite(stamp_s) or (self.last_stamp is not None and stamp_s <= self.last_stamp):
            raise ValueError('stamp must be finite and strictly increasing')
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError('finite XY required')
        dt = 0. if self.last_stamp is None else stamp_s-self.last_stamp
        gap = self.last_stamp is not None and dt > 2.5
        allowed = {self.current}|self.graph.adjacent[self.current]
        mask = np.array([k in allowed for k in self.ids])
        evidence = np.array([likelihood.get(k, 0.) if k in allowed else 0. for k in self.ids])
        if not np.isfinite(evidence).all() or np.any(evidence < 0):
            raise ValueError('likelihood must be finite and nonnegative')
        self.last_stamp = stamp_s
        valid = valid and not gap and evidence.sum() > 0
        if valid:
            # 소량의 이웃 prior를 남기되 지도 연결 밖으로 바로 점프하지 않는다.
            prior = .9*self.probabilities*mask+.1*mask/mask.sum()
            posterior = prior*evidence
            self.probabilities = posterior/posterior.sum()
        else:
            self.probabilities = .95*self.probabilities*mask+.05*mask/mask.sum()
            self.probabilities /= self.probabilities.sum()
        best = self.ids[int(self.probabilities.argmax())]
        transition = None
        key = tuple(sorted((self.current, best)))
        can_enter = valid and best != self.current and self.probabilities.max() >= .65
        can_enter = can_enter and key in self.graph.portals and self.graph.distance(best, xy) < 1e-7
        can_enter = can_enter and self.graph.distance(self.current, xy) > .125
        if can_enter:
            near = segment_distance(xy, self.graph.portals[key]) <= .45
            continuing = self.pending == best and self.confirmations > 0 and self.last_xy is not None
            continuing = continuing and np.linalg.norm(xy-self.last_xy) <= .3*dt+.05
            can_enter = near or continuing
        if can_enter:
            self.confirmations = self.confirmations+1 if self.pending == best else 1
            self.pending = best
            if self.confirmations >= 2:
                transition = [self.current, best];self.current = best
                self.pending, self.confirmations = None, 0
        else:
            self.pending, self.confirmations = None, 0
        self.last_xy = xy.copy()
        return dict(current_region=self.current, transition=transition,
                    candidates=[k for k, p in zip(self.ids, self.probabilities) if p >= .05],
                    probabilities={str(k): float(p) for k, p in zip(self.ids, self.probabilities)},
                    status='observed' if valid else 'observation_missing')
