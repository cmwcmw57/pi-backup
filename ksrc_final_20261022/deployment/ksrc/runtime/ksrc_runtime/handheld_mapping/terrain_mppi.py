# ================================================================
# terrain_mppi.py
# 설명: 알려진 격자와 현재 장애물로 시뮬레이션용 비학습 MPPI 속도를 선택한다.
# 사용법: from ksrc_runtime.handheld_mapping.terrain_mppi import TerrainMppi
# ================================================================
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt, label, map_coordinates
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra


class TerrainMppi:
    """차체 여유 공간의 geodesic 비용으로 Gaussian 제어 표본을 가중 갱신한다."""

    def __init__(self, site: Any, seed: int = 17, max_speed: float = .08,
                 max_yaw_rate: float = .3, start_tolerance_m: float = 0.) -> None:
        if not (0. < max_speed <= .15 and 0. < max_yaw_rate <= .3):
            raise ValueError('simulation speed/yaw bounds must be in (0,.15]/(0,.3]')
        self.site = site
        self.max_speed, self.max_yaw_rate = float(max_speed), float(max_yaw_rate)
        self.start_tolerance = float(start_tolerance_m)  # 위치 추정 오차 상한: 여유 부족이 이 안이면 시작을 허용한다.
        self.rng = np.random.default_rng(seed)
        self.samples, self.horizon = 192, 30
        self.dt = min(.2, float(site.resolution) * .4 / self.max_speed)
        self.temperature = .12
        self.controls = np.zeros((self.horizon, 3))   # (vx, 요 속도, vy)
        self.max_lateral = 0.        # vy 상한 [m/s]; 0이면 vy 없이 기존과 동일
        self.clearance_weight = 0.   # 벽 여유 비용 가중치; 0이면 꺼짐
        self.clearance_margin = .12  # 여유 비용이 0이 되는 벽 거리 = 안전 한계 + 이 값 [m]
        self._field_key: tuple | None = None
        self._field: np.ndarray | None = None
        self._reverse = False

    def _stop(self, reason: str, rejected: int = 0) -> dict:
        self.controls[:] = 0.
        return {'valid': False, 'reason': reason,
                'command': {'vx_m_s': 0., 'yaw_rate_rad_s': 0.},
                'selected_trajectory': [], 'candidate_trajectories': [],
                'rejected_candidates': int(rejected)}

    def _cells(self, xy: np.ndarray) -> np.ndarray:
        return np.floor((xy - np.asarray(self.site.origin)) / self.site.resolution).astype(int)

    def _trajectory_safe(self, trajectories: np.ndarray, safe: np.ndarray) -> np.ndarray:
        cells = self._cells(trajectories)
        rows, cols = safe.shape
        inside = ((cells[..., 0] >= 0) & (cells[..., 0] < cols)
                  & (cells[..., 1] >= 0) & (cells[..., 1] < rows))
        good = safe[np.clip(cells[..., 1], 0, rows - 1), np.clip(cells[..., 0], 0, cols - 1)]
        valid = (inside & good).all(axis=1)
        # A diagonal grid crossing must also clear both side cells (supercover).
        # Rollout steps are shorter than one cell, so no other cells can be skipped.
        a, b = cells[:, :-1], cells[:, 1:]
        diagonal = (a[..., 0] != b[..., 0]) & (a[..., 1] != b[..., 1])
        sides = (safe[np.clip(a[..., 1], 0, rows - 1), np.clip(b[..., 0], 0, cols - 1)]
                 & safe[np.clip(b[..., 1], 0, rows - 1), np.clip(a[..., 0], 0, cols - 1)])
        return valid & (~diagonal | sides).all(axis=1)

    @staticmethod
    def _bound_speed(values: np.ndarray, maximum: float, minimum: float, reverse: bool) -> np.ndarray:
        """정지 또는 유효 속도 범위로 제한하고 제한된 값으로 궤적을 검사한다."""
        values = np.clip(values, -maximum if reverse else 0., 0. if reverse else maximum)
        if minimum > 0:
            values = (-1. if reverse else 1.) * np.where(np.abs(values) > 1e-9,
                                                         np.maximum(np.abs(values), minimum), 0.)
        return values

    def _rollout(self, xy: np.ndarray, yaw: float, controls: np.ndarray) -> np.ndarray:
        # Midpoint integration; displacement <= .4 cells prevents jumping over occupied cells.
        heading = yaw + np.cumsum(controls[..., 1] * self.dt, axis=1)
        middle = heading - controls[..., 1] * self.dt * .5
        forward = controls[..., :1] * np.stack((np.cos(middle), np.sin(middle)), axis=-1)
        sideways = controls[..., 2:3] * np.stack((-np.sin(middle), np.cos(middle)), axis=-1) if controls.shape[-1] > 2 else 0.
        delta = (forward + sideways) * self.dt
        points = xy + np.cumsum(delta, axis=1)
        return np.concatenate((np.broadcast_to(xy, (len(controls), 1, 2)), points), axis=1)

    def _distance_field(self, safe: np.ndarray, target: tuple[int, int]) -> np.ndarray:
        key = (safe.tobytes(), target)
        if key == self._field_key:
            return self._field
        rows, cols = safe.shape
        indices = np.arange(rows * cols).reshape(safe.shape)
        source, dest = [], []
        for a, b in ((indices[:, :-1], indices[:, 1:]), (indices[:-1], indices[1:])):
            valid = safe.ravel()[a] & safe.ravel()[b]
            source.extend((a[valid], b[valid]))
            dest.extend((b[valid], a[valid]))
        ii, jj = np.concatenate(source), np.concatenate(dest)
        graph = coo_matrix((np.full(len(ii), self.site.resolution), (ii, jj)),
                           shape=(rows * cols, rows * cols)).tocsr()
        self._field = dijkstra(graph, directed=False, indices=target[0] * cols + target[1]).reshape(safe.shape)
        self._field_key = key
        return self._field

    def plan(self, xy: Any, yaw: float, target_xy: Any, allowed_mask: np.ndarray,
             max_slope_deg: float, *, reverse: bool = False,
             obstacles_xy: Any = None, min_speed_m_s: float = 0.,
             command_scale: float = 1.) -> dict:
        xy, target = np.asarray(xy, float), np.asarray(target_xy, float)
        allowed = np.asarray(allowed_mask, bool)
        if (xy.shape != (2,) or target.shape != (2,) or not np.isfinite(xy).all()
                or not np.isfinite(target).all() or not math.isfinite(yaw)
                or allowed.shape != self.site.observed.shape
                or not math.isfinite(max_slope_deg) or not 0. < max_slope_deg < 90.
                or not math.isfinite(command_scale) or not 0. < command_scale <= 1.
                or not math.isfinite(min_speed_m_s) or not 0. <= min_speed_m_s <= self.max_speed):
            return self._stop('invalid_input')
        max_speed, min_speed = self.max_speed*command_scale, min_speed_m_s*command_scale
        max_yaw_rate = self.max_yaw_rate*command_scale
        max_lateral = self.max_lateral*command_scale
        clear = (allowed & self.site.observed & ~self.site.blocked
                 & np.isfinite(self.site.slope_deg) & (self.site.slope_deg <= max_slope_deg + 1e-4))
        if obstacles_xy is not None:
            obstacles = np.asarray(obstacles_xy, float)
            if obstacles.size:
                if obstacles.ndim != 2 or obstacles.shape[1] != 2 or not np.isfinite(obstacles).all():
                    return self._stop('invalid_obstacles')
                cells = self._cells(obstacles)
                inside = ((cells[:, 0] >= 0) & (cells[:, 0] < clear.shape[1])
                          & (cells[:, 1] >= 0) & (cells[:, 1] < clear.shape[0]))
                clear[cells[inside, 1], cells[inside, 0]] = False
        distance = distance_transform_edt(np.pad(clear, 1))[1:-1, 1:-1] * self.site.resolution
        safe = clear & (distance >= self.site.robot_radius + self.site.resolution * math.sqrt(2) - 1e-9)
        cells = self._cells(np.stack((xy, target)))
        rows, cols = safe.shape
        if np.any(cells < 0) or np.any(cells[:, 0] >= cols) or np.any(cells[:, 1] >= rows):
            return self._stop('outside_map')
        start_cell, target_cell = (tuple(cell[::-1]) for cell in cells)
        relaxed = not safe[start_cell]
        if relaxed:
            # 추정 오차로 시작점이 여유 대역에 들어온 경우 현재 여유 이상인 곳으로만 움직인다.
            start_clearance = distance[start_cell]
            need = self.site.robot_radius + self.site.resolution * math.sqrt(2)
            if not clear[start_cell] or start_clearance + self.start_tolerance < need - 1e-9:
                return self._stop('unsafe_start')
            safe = clear & (distance >= start_clearance - 1e-9)
        components, _ = label(safe)
        reachable = components == components[start_cell]
        if not safe[target_cell]:
            # Permit a portal on an inflated boundary, but never snap through a blocked wall.
            yy, xx = np.nonzero(reachable)
            distances = np.hypot((xx + .5) * self.site.resolution + self.site.origin[0] - target[0],
                                 (yy + .5) * self.site.resolution + self.site.origin[1] - target[1])
            nearest = int(np.argmin(distances))
            if not clear[target_cell] or distances[nearest] > self.site.robot_radius + 2 * self.site.resolution:
                return self._stop('unsafe_target')
            target_cell = (int(yy[nearest]), int(xx[nearest]))
        if not reachable[target_cell]:
            return self._stop('unreachable_target')
        field = self._distance_field(safe, target_cell)
        target_center = np.asarray(self.site.origin) + (np.asarray(target_cell[::-1]) + .5) * self.site.resolution
        if np.linalg.norm(xy - target_center) <= max(.04, self.site.resolution):
            return self._stop('target_reached')
        # Descend the nonconvex distance field for a short heading reference.
        waypoint = start_cell
        for _ in range(max(1, int(.25 / self.site.resolution))):
            y, x = waypoint
            neighbors = [(ny, nx) for ny, nx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1))
                         if 0 <= ny < rows and 0 <= nx < cols and safe[ny, nx]]
            next_cell = min(neighbors, key=lambda c: field[c], default=waypoint)
            if field[next_cell] >= field[waypoint]:
                break
            next_xy = np.asarray(self.site.origin) + (np.asarray(next_cell[::-1]) + .5) * self.site.resolution
            # 꺾인 거리장 경로의 lookahead가 벽 모서리를 가로지르지 않게 한다.
            if not self._trajectory_safe(np.array([[xy, next_xy]]), safe)[0]:
                break
            waypoint = next_cell
        reference = np.asarray(self.site.origin) + (np.asarray(waypoint[::-1]) + .5) * self.site.resolution
        desired = math.atan2(reference[1] - xy[1], reference[0] - xy[0]) + (math.pi if reverse else 0.)
        error = math.atan2(math.sin(desired - yaw), math.cos(desired - yaw))
        aligning = abs(error) > math.radians(30.)
        nominal = np.array([(-1. if reverse else 1.) * max_speed * max(.05, math.cos(error)),
                            np.clip(error * .8, -max_yaw_rate, max_yaw_rate), 0.])
        if aligning:
            # 급한 목표 방향 전환에서 좁은 통로 쪽으로 전진하는 원호를 만들지 않는다.
            nominal[0] = 0.
            self.controls[:, 0] = 0.
        if reverse != self._reverse:
            self.controls[:] = nominal
        self._reverse = reverse
        self.controls = .3 * np.vstack((self.controls[1:], self.controls[-1:])) + .7 * nominal
        rejected = 0
        trajectories = None
        valid = None
        finite_field = np.where(np.isfinite(field), field, rows * cols * self.site.resolution)
        for _ in range(2):
            noise = self.rng.normal(size=(self.samples, self.horizon, 2)) * [.045, .22]
            # Correlate yaw perturbations to explore coherent turns instead of cancelling noise.
            noise[..., 1] = .6 * noise[..., 1] + .4 * noise[:, :1, 1]
            lateral_noise = (self.rng.normal(size=(self.samples, self.horizon, 1)) * (.6 * max_lateral) if max_lateral > 0
                             else np.zeros((self.samples, self.horizon, 1)))
            noise = np.concatenate((noise, lateral_noise), axis=-1)
            candidates = self.controls[None] + noise
            candidates[0] = self.controls
            # 좁은 코너에서도 제자리 회전과 정지를 명시적으로 비교한다.
            candidates[1] = [0., nominal[1], 0.]
            candidates[2] = [0., 0., 0.]
            if min_speed > 0:
                # 정상 출구 근처에서는 최소속도로 전진한 뒤 목표 앞에서 정지하는 후보를 포함한다.
                stop_after = max(1, min(self.horizon, int(field[start_cell]/(max_speed*self.dt))))
                candidates[3] = nominal
                candidates[3, stop_after:] = 0.
            if max_lateral > 0:
                # 옆걸음 후보를 명시적으로 넣어 좁은 통로에서 회전 없이 횡이동하는 해를 비교한다.
                candidates[4] = nominal;candidates[4, :, 2] = max_lateral
                candidates[5] = nominal;candidates[5, :, 2] = -max_lateral
            candidates[..., 0] = self._bound_speed(candidates[..., 0], max_speed, min_speed, reverse)
            if aligning:candidates[..., 0] = 0.
            candidates[..., 1] = np.clip(candidates[..., 1], -max_yaw_rate, max_yaw_rate)
            candidates[..., 2] = 0. if (aligning or max_lateral <= 0) else np.clip(candidates[..., 2], -max_lateral, max_lateral)
            trajectories = self._rollout(xy, yaw, candidates)
            valid = self._trajectory_safe(trajectories, safe)
            rejected += int((~valid).sum())
            if not valid.any():
                return self._stop('no_safe_rollout', rejected)
            coords = (trajectories[..., ::-1] - np.asarray(self.site.origin)[::-1]) / self.site.resolution - .5
            potential = map_coordinates(finite_field, [coords[..., 0], coords[..., 1]], order=1, mode='nearest')
            end_yaw = yaw + candidates[..., 1].sum(axis=1) * self.dt
            heading_cost = 1. - np.cos(end_yaw - desired)
            cost = (3. * potential[:, -1] + .3 * potential.mean(axis=1)
                    + .18 * heading_cost + .02 * np.square(candidates[..., 1]).mean(axis=1))
            if self.clearance_weight > 0:
                # 벽 여유가 안전 한계(로봇 반경)+12 cm보다 작은 정도를 비용으로 줘서 통로 가운데를 선호하게 한다.
                near = map_coordinates(distance, [coords[..., 0], coords[..., 1]], order=1, mode='nearest')
                floor_distance = self.site.robot_radius + self.site.resolution * math.sqrt(2)
                cost = cost + self.clearance_weight * np.clip((floor_distance + self.clearance_margin - near) / self.clearance_margin, 0., 1.).mean(axis=1)
            if max_lateral > 0:
                cost = cost + .02 * np.square(candidates[..., 2] / max_lateral).mean(axis=1)
            cost[~valid] = np.inf
            weights = np.exp(-(cost - cost.min()) / self.temperature)
            weights /= weights.sum()
            # Weighted perturbation update is MPPI; collision candidates have exactly zero weight.
            self.controls += np.einsum('k,khi->hi', weights, candidates - self.controls)
            self.controls[:, 0] = self._bound_speed(self.controls[:, 0], max_speed, min_speed, reverse)
            self.controls[:, 1] = np.clip(self.controls[:, 1], -max_yaw_rate, max_yaw_rate)
            self.controls[:, 2] = 0. if (aligning or max_lateral <= 0) else np.clip(self.controls[:, 2], -max_lateral, max_lateral)
        chosen = self._rollout(xy, yaw, self.controls[None])
        if not self._trajectory_safe(chosen, safe)[0]:
            if min_speed <= 0:
                return self._stop('unsafe_weighted_mean', rejected)
            # 정지/최소속도의 비연속 제약에서는 평균이 위험해질 수 있다. 이미 검사한 최저비용 후보도 다시 검사한다.
            self.controls = candidates[int(np.argmin(cost))].copy()
            chosen = self._rollout(xy, yaw, self.controls[None])
            if not self._trajectory_safe(chosen, safe)[0]:
                return self._stop('unsafe_weighted_mean', rejected)
        return {'valid': True, 'reason': 'relaxed_start' if relaxed else 'aligning' if aligning else 'clear',
                'command': {'vx_m_s': float(self.controls[0, 0]),
                            'yaw_rate_rad_s': float(self.controls[0, 1]),
                            **({'vy_m_s': float(self.controls[0, 2])} if max_lateral > 0 else {})},
                'selected_trajectory': chosen[0].tolist(),
                'candidate_trajectories': trajectories[np.flatnonzero(valid)[:16]].tolist(),
                'rejected_candidates': rejected}
