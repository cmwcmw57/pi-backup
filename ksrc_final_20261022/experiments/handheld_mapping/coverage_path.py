# ================================================================
# coverage_path.py
# 설명: 지도 위에서 칠한 구역을 지그재그로 모두 덮고 출발점으로 돌아오는 폐루프 경로(웨이포인트)를 만든다. 지도는 높이맵의 경사·단차 또는 Nav2 지도에서 읽는다.
# 사용법:
#   from experiments.handheld_mapping.coverage_path import load_heightmap, coverage_loop
#   result = coverage_loop(load_heightmap(npz), region=mask, lane=0.4, robot_radius=0.21, start=(x, y))
# ================================================================
import heapq
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

NEIGHBOURS = [(-1, 0, 1.), (1, 0, 1.), (0, -1, 1.), (0, 1, 1.),
              (-1, -1, 2 ** .5), (-1, 1, 2 ** .5), (1, -1, 2 ** .5), (1, 1, 2 ** .5)]


@dataclass
class GridMap:
    """바닥 격자. free[iy, ix]가 True면 주행 가능, origin은 아래-왼쪽 모서리의 월드 좌표(m)다."""
    free: np.ndarray
    resolution: float
    origin: tuple[float, float]
    height: np.ndarray | None = None
    slope_deg: np.ndarray | None = None

    def to_cell(self, xy: np.ndarray) -> np.ndarray:
        """월드 (x, y) → 격자 (iy, ix)."""
        xy = np.atleast_2d(xy)
        return np.stack([np.floor((xy[:, 1] - self.origin[1]) / self.resolution),
                         np.floor((xy[:, 0] - self.origin[0]) / self.resolution)], axis=1).astype(int)

    def to_world(self, cells: np.ndarray) -> np.ndarray:
        cells = np.atleast_2d(cells)
        return np.stack([self.origin[0] + (cells[:, 1] + .5) * self.resolution,
                         self.origin[1] + (cells[:, 0] + .5) * self.resolution], axis=1)

    def clearance(self) -> np.ndarray:
        """주행 불가 칸까지의 거리(m). 지도 바깥은 벽(펜스)으로 본다."""
        padded = np.pad(self.free, 1, constant_values=False)
        return ndimage.distance_transform_edt(padded)[1:-1, 1:-1] * self.resolution

    def min_clearance(self, path_xy: np.ndarray) -> float:
        cells = np.clip(self.to_cell(path_xy), 0, np.array(self.free.shape) - 1)
        return float(self.clearance()[cells[:, 0], cells[:, 1]].min())


def load_heightmap(npz: Path, max_slope_deg: float = 21., cell: float = .02) -> GridMap:
    """경기장 모래 높이맵(axis0=x, axis1=y)을 cell 크기로 평균해 경사 상한 이하인 칸만 주행 가능으로 만든다.

    경사 상한 21°는 KSRC 견인력 예산(마찰 0.53 − 저항 0.14)에서 유도된 값이다(ksrc `config.max_slope_deg`).
    """
    data = np.load(npz)
    raw, size = data["height"], float(data["cell"])
    factor = max(1, round(cell / size))
    nx, ny = raw.shape[0] // factor, raw.shape[1] // factor
    coarse = np.nanmean(raw[:nx * factor, :ny * factor].reshape(nx, factor, ny, factor), axis=(1, 3))
    height = coarse.T                                         # [iy, ix]
    res = size * factor
    gy, gx = np.gradient(height, res)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    return GridMap(free=np.isfinite(height) & (slope <= max_slope_deg), resolution=res,
                   origin=(float(data["x0"]), float(data["y0"])), height=height, slope_deg=slope)


def load_nav2_map(yaml_path: Path) -> GridMap:
    """Nav2 지도(yaml + pgm)를 읽는다. 254(자유)만 주행 가능이다."""
    meta = {}
    for line in Path(yaml_path).read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    raw = (Path(yaml_path).parent / meta["image"]).read_bytes()
    pos, values = 2, []
    while len(values) < 3:                                    # P5 헤더, 주석 줄은 건너뛴다.
        while raw[pos:pos + 1].isspace():
            pos += 1
        if raw[pos:pos + 1] == b"#":
            pos = raw.index(b"\n", pos) + 1
            continue
        end = pos
        while raw[end:end + 1].isdigit():
            end += 1
        values.append(int(raw[pos:end]))
        pos = end
    width, height, _ = values
    image = np.frombuffer(raw[pos + 1:], dtype=np.uint8).reshape(height, width)
    origin = [float(v) for v in meta["origin"].strip("[]").split(",")]
    return GridMap(free=image[::-1] >= 250, resolution=float(meta["resolution"]), origin=(origin[0], origin[1]))


def region_from_polygon(grid: GridMap, polygon_xy: np.ndarray) -> np.ndarray:
    """월드 좌표 다각형 안의 칸을 True로 하는 구역 마스크."""
    from matplotlib.path import Path as Polygon
    rows, cols = np.indices(grid.free.shape)
    centres = grid.to_world(np.stack([rows.ravel(), cols.ravel()], axis=1))
    return Polygon(np.asarray(polygon_xy)).contains_points(centres).reshape(grid.free.shape)


def visible(safe: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> bool:
    """두 칸을 잇는 직선이 모두 safe인가."""
    count = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) * 2 + 1
    rows = np.rint(np.linspace(a[0], b[0], count)).astype(int)
    cols = np.rint(np.linspace(a[1], b[1], count)).astype(int)
    return bool(safe[rows, cols].all())


def astar(safe: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    """8방향 A*. 대각선은 양옆 칸이 모두 자유일 때만 허용한다. 길이 없으면 None."""
    heap, best, parent = [(0., 0., start)], {start: 0.}, {}
    while heap:
        _, cost, node = heapq.heappop(heap)
        if node == goal:
            path = [node]
            while node in parent:
                node = parent[node]
                path.append(node)
            return path[::-1]
        if cost > best.get(node, 1e18):
            continue
        for dr, dc, step in NEIGHBOURS:
            nxt = (node[0] + dr, node[1] + dc)
            if not (0 <= nxt[0] < safe.shape[0] and 0 <= nxt[1] < safe.shape[1]) or not safe[nxt]:
                continue
            if dr and dc and not (safe[node[0] + dr, node[1]] and safe[node[0], node[1] + dc]):
                continue
            new = cost + step
            if new < best.get(nxt, 1e18):
                best[nxt], parent[nxt] = new, node
                heapq.heappush(heap, (new + np.hypot(goal[0] - nxt[0], goal[1] - nxt[1]), new, nxt))
    return None


def connect(safe: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]] | None:
    """직선이 가능하면 직선, 아니면 A*를 시선 단축(string pulling)한 꼭짓점 목록."""
    if visible(safe, a, b):
        return [a, b]
    path = astar(safe, a, b)
    if path is None:
        return None
    out, i = [path[0]], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not visible(safe, path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out


def lane_segments(target: np.ndarray, count: int, shift: float, along_x: bool) -> list[list[tuple[tuple[int, int], tuple[int, int]]]]:
    """구역 마스크를 구역 폭에 균등하게 나눈 count개의 줄로 자르고, 줄마다 끊기지 않는 구간(양 끝 칸)을 모은다. shift는 줄 간격 단위의 이동."""
    mask = target if along_x else target.T
    rows = np.flatnonzero(mask.any(axis=1))
    if not len(rows):
        return []
    extent = rows.max() - rows.min() + 1
    positions = sorted({int(np.clip(round(rows.min() + (k + .5 + shift) * extent / count), rows.min(), rows.max())) for k in range(count)})
    lanes = []
    for r in positions:
        runs, c = [], 0
        while c < mask.shape[1]:
            if mask[r, c]:
                end = c
                while end + 1 < mask.shape[1] and mask[r, end + 1]:
                    end += 1
                runs.append(((r, c), (r, end)) if along_x else ((c, r), (end, r)))
                c = end + 1
            else:
                c += 1
        if runs:
            lanes.append(runs)
    return lanes


def lane_orders(lanes: list) -> list[list]:
    """줄 방문 순서 후보: 위에서 아래로 지그재그, 그리고 한 줄 걸러 갔다가 나머지로 돌아오는 빗살 두 가지(끝이 시작 근처가 된다)."""
    return [lanes, lanes[0::2] + lanes[1::2][::-1], lanes[1::2] + lanes[0::2][::-1]]


def tour(safe: np.ndarray, origin: tuple[int, int], order: list) -> tuple[list[tuple[int, int]], int, float]:
    """줄을 순서대로, 줄 안에서는 가까운 구간부터 가까운 끝에서 들어가 반대 끝으로 나오며 이어 붙인 꼭짓점, 건너뛴 구간 수, 복귀 구간 길이(칸)."""
    here, vertices, skipped = origin, [origin], 0
    dist = lambda a, b: float(np.hypot(a[0] - b[0], a[1] - b[1]))
    for runs in order:
        pending = list(runs)
        while pending:
            pick = min(pending, key=lambda r: min(dist(r[0], here), dist(r[1], here)))
            pending.remove(pick)
            near, far = (pick[0], pick[1]) if dist(pick[0], here) <= dist(pick[1], here) else (pick[1], pick[0])
            link = connect(safe, here, near)
            if link is None:
                skipped += 1
                continue
            vertices += link[1:] + [far]
            here = far
    back = connect(safe, here, origin)
    if back is None:
        raise ValueError("출발점으로 돌아오는 길이 없습니다")
    return vertices + back[1:], skipped, polyline_length(back)


def polyline_length(vertices: list[tuple[int, int]]) -> float:
    return float(np.linalg.norm(np.diff(np.array(vertices, float), axis=0), axis=1).sum()) if len(vertices) > 1 else 0.


def coverage_of(grid: GridMap, target: np.ndarray, points: np.ndarray, lane: float) -> float:
    """목표 칸 중 경로에서 lane/2 이내인 비율."""
    trace = np.zeros(target.shape, bool)
    cells = np.clip(grid.to_cell(points), 0, np.array(target.shape) - 1)
    trace[cells[:, 0], cells[:, 1]] = True
    reach = ndimage.distance_transform_edt(~trace) * grid.resolution
    return float((reach[target] <= lane / 2 + grid.resolution).mean())


def coverage_loop(grid: GridMap, region: np.ndarray, lane: float, robot_radius: float,
                  start: tuple[float, float] | None, step: float = .05) -> dict:
    """구역을 줄 간격 lane(m)으로 덮고 start로 돌아오는 닫힌 경로를 만든다.

    로버 중심은 벽·급경사에서 robot_radius 이상 떨어진 칸(safe)에서만 지나간다. 줄 수 2가지(간격이 lane 이하가 되는 최소 수와 +1) × 줄 위치 3가지 × 방문 순서 3가지를 시험해
    가장 많이 덮고(1%p 이내 차이면) 가장 짧은 경로를 고른다. start가 None이면 첫 줄의 한쪽 끝에서 시작한다.
    start에서 닿지 못하는 줄은 건너뛰고 보고하며, 닿는 줄이 하나도 없으면 ValueError를 낸다.
    """
    safe = grid.free & (grid.clearance() >= robot_radius)
    target = region & safe
    if not target.any():
        raise ValueError("칠한 구역 안에 주행 가능한 칸이 없습니다")
    rows, cols = np.nonzero(target)
    along_x = (cols.max() - cols.min()) >= (rows.max() - rows.min())      # 긴 쪽으로 줄을 그어 회전을 줄인다.
    extent_m = ((rows.max() if along_x else cols.max()) - (rows.min() if along_x else cols.min()) + 1) * grid.resolution
    base = max(1, int(np.ceil(extent_m / lane)))                          # 줄 간격이 lane 이하가 되는 최소 줄 수
    labels, _ = ndimage.label(safe)
    candidates = []
    for count, shift in [(n, f) for n in (base, base + 1) for f in (-.25, 0., .25)]:
        lanes = lane_segments(target, count, shift, along_x)
        if not lanes:
            continue
        if start is None:
            origin = lanes[0][0][0]
        else:
            origin = tuple(np.clip(grid.to_cell(np.array(start))[0], 0, np.array(safe.shape) - 1))
            if not safe[origin]:                                          # 가까운 안전 칸으로 옮긴다.
                _, (near_r, near_c) = ndimage.distance_transform_edt(~safe, return_indices=True)
                origin = (int(near_r[origin]), int(near_c[origin]))
        lanes = [[r for r in runs if labels[r[0]] == labels[origin]] for runs in lanes]
        lanes = [runs for runs in lanes if runs]
        if not lanes:
            raise ValueError("시작점에서 도달할 수 있는 구역이 없습니다")
        for order in lane_orders(lanes):
            vertices, skipped, back = tour(safe, origin, order)
            xy = grid.to_world(np.array(vertices))
            length = polyline_length(vertices) * grid.resolution
            sample = np.arange(0, length, step)
            seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
            points = np.stack([np.interp(sample, seg, xy[:, 0]), np.interp(sample, seg, xy[:, 1])], axis=1)
            candidates.append({"xy": xy, "points": points, "length": float(seg[-1]), "skipped": skipped, "closing": back * grid.resolution,
                               "coverage": coverage_of(grid, target, points, lane), "segments": sum(len(r) for r in order)})
    if not candidates:
        raise ValueError("시작점에서 도달할 수 있는 구역이 없습니다")
    best_cover = max(c["coverage"] for c in candidates)
    best = min((c for c in candidates if c["coverage"] >= best_cover - .01), key=lambda c: c["length"])
    points = np.vstack([best["points"], best["xy"][-1]])
    heading = np.arctan2(*np.diff(points, axis=0)[:, ::-1].T)
    waypoints = np.column_stack([points, np.r_[heading, heading[-1]]])
    return {"waypoints": waypoints, "coverage": best["coverage"], "length_m": best["length"], "lane_m": lane,
            "robot_radius_m": robot_radius, "segments": best["segments"], "unreachable_segments": best["skipped"],
            "closing_leg_m": best["closing"]}


def best_rectangle(safe: np.ndarray, min_side: int) -> tuple[int, int, int, int] | None:
    """네 변이 모두 safe인 축 정렬 사각형(r0, c0, r1, c1) 중 둘레가 가장 긴 것. 변의 칸만 보므로 안쪽에 장애물이 있어도 된다."""
    blocked = ~safe
    by_col = np.concatenate([np.zeros((1, safe.shape[1]), int), np.cumsum(blocked, axis=0)])   # 열마다 아래로 누적
    best, best_score = None, -1
    for r0 in range(safe.shape[0]):
        for r1 in range(r0 + min_side, safe.shape[0]):
            side_ok = (by_col[r1 + 1] - by_col[r0]) == 0                  # 이 열은 r0..r1이 모두 safe
            row_ok = safe[r0] & safe[r1]
            c = 0
            while c < safe.shape[1]:
                if not row_ok[c]:
                    c += 1
                    continue
                end = c
                while end + 1 < safe.shape[1] and row_ok[end + 1]:
                    end += 1
                cols = np.flatnonzero(side_ok[c:end + 1]) + c            # 이 구간에서 양쪽 세로변이 될 수 있는 열
                if len(cols) >= 2 and cols[-1] - cols[0] >= min_side:
                    score = (r1 - r0) + (cols[-1] - cols[0])
                    if score > best_score:
                        best, best_score = (r0, int(cols[0]), r1, int(cols[-1])), score
                c = end + 1
    return best


def rectangle_loop(grid: GridMap, robot_radius: float, start: tuple[float, float] | None = None,
                   rect: tuple[float, float, float, float] | None = None, laps: int = 1, step: float = .05,
                   max_slope_deg: float | None = None, min_side: float = .4) -> dict:
    """ㅁ자 직사각형 둘레의 폐루프: 직선 네 변과 직각 모서리뿐이다.

    rect(xmin, ymin, xmax, ymax)를 주면 그 사각형을, 없으면 네 변이 모두 safe인 가장 큰 사각형을 쓴다.
    max_slope_deg를 주면 그보다 가파른 칸을 피한다(평평한 곳만 돌 때). start가 있으면 둘레 위 가장 가까운 점에서 시작해 반시계로 돈다.
    corners는 닫힌 꼭짓점 목록(Isaac LoopRoute가 점에서 점으로 직진하는 입력), waypoints는 변마다 step 간격으로 나눈 (x, y, yaw)다.
    """
    safe = grid.free & (grid.clearance() >= robot_radius)
    if max_slope_deg is not None:
        if grid.slope_deg is None:
            raise ValueError("경사 한계를 쓰려면 경사 정보가 있는 지도가 필요합니다")
        safe &= grid.slope_deg <= max_slope_deg
    if rect is None:
        cells = best_rectangle(safe, max(1, round(min_side / grid.resolution)))
        if cells is None:
            raise ValueError("네 변이 모두 안전한 사각형을 찾지 못했습니다")
        (x0, y0), (x1, y1) = grid.to_world(np.array([cells[:2], cells[2:]]))
    else:
        x0, y0, x1, y1 = rect
        (r0, c0), (r1, c1) = grid.to_cell(np.array([[x0, y0], [x1, y1]]))
        if min(r0, c0) < 0 or r1 >= safe.shape[0] or c1 >= safe.shape[1] or not np.concatenate(
                [safe[r0, c0:c1 + 1], safe[r1, c0:c1 + 1], safe[r0:r1 + 1, c0], safe[r0:r1 + 1, c1]]).all():
            raise ValueError("지정한 사각형의 둘레가 안전하지 않습니다")
    ring = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])                # 반시계 네 꼭짓점
    vertices = [ring[k] for k in range(4)]
    if start is not None:                                                     # 둘레 위에서 start에 가장 가까운 점을 새 시작점으로 끼운다.
        edges = [(ring[k], ring[(k + 1) % 4]) for k in range(4)]
        def foot(a, b):
            t = np.clip(np.dot(np.asarray(start) - a, b - a) / np.dot(b - a, b - a), 0, 1)
            return a + t * (b - a), t
        k = min(range(4), key=lambda i: np.linalg.norm(foot(*edges[i])[0] - np.asarray(start)))
        point, t = foot(*edges[k])
        if t < 1e-9:
            order = [ring[(k + j) % 4] for j in range(4)]
        elif t > 1 - 1e-9:
            order = [ring[(k + 1 + j) % 4] for j in range(4)]
        else:
            order = [point] + [ring[(k + 1 + j) % 4] for j in range(4)]       # 변 중간에서 시작하면 점이 하나 늘어난다.
        vertices = order
    closed = vertices + [vertices[0]]
    corners = [closed[0]] + closed[1:] * laps
    dense = [corners[0]]
    for a, b in zip(corners[:-1], corners[1:]):                               # 변마다 모서리를 정확히 포함해 나눈다.
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        dense += [a + (b - a) * j / n for j in range(1, n + 1)]
    points = np.array(dense)
    heading = np.arctan2(*np.diff(points, axis=0)[:, ::-1].T)
    waypoints = np.column_stack([points, np.r_[heading, heading[-1]]])
    perimeter = 2 * (abs(x1 - x0) + abs(y1 - y0))
    return {"waypoints": waypoints, "corners": np.array(corners), "rect": [float(x0), float(y0), float(x1), float(y1)],
            "perimeter_m": float(perimeter), "length_m": float(perimeter * laps), "laps": laps, "robot_radius_m": robot_radius}


def save_path(result: dict, grid: GridMap, region: np.ndarray, folder: Path) -> None:
    """웨이포인트(JSON·CSV)와 지도 위 그림(PNG)을 새 폴더에 저장한다."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    waypoints = result["waypoints"]
    meta = {k: v for k, v in result.items() if k != "waypoints"}
    (folder / "path.json").write_text(json.dumps({**meta, "frame": "map", "unit": "m, rad",
                                                  "waypoints": np.round(waypoints, 4).tolist()},
                                                 indent=1, default=lambda v: np.round(v, 4).tolist() if isinstance(v, np.ndarray) else float(v)))
    np.savetxt(folder / "path.csv", waypoints, delimiter=",", header="x_m,y_m,yaw_rad", comments="", fmt="%.4f")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 7))
    h, w = grid.free.shape
    extent = [grid.origin[0], grid.origin[0] + w * grid.resolution, grid.origin[1], grid.origin[1] + h * grid.resolution]
    base = np.where(grid.free, 1., 0.)
    ax.imshow(base, origin="lower", extent=extent, cmap="gray", vmin=-.3, vmax=1.3)
    ax.imshow(np.ma.masked_where(~region, region), origin="lower", extent=extent, cmap="Blues", alpha=.35)
    ax.plot(waypoints[:, 0], waypoints[:, 1], color="#e8873a", lw=1.2)
    ax.plot(*waypoints[0, :2], "o", color="#3fbf6f", ms=9, label="시작·복귀")
    ax.set_aspect("equal")
    if "lane_m" in result:
        ax.set_title(f"탐색 폐루프 — 길이 {result['length_m']:.1f} m · 줄 간격 {result['lane_m']} m · 덮은 비율 {result['coverage'] * 100:.0f}%")
    else:
        ax.set_title(f"ㅁ 폐루프 — 길이 {result['length_m']:.1f} m · 한 바퀴 {result['perimeter_m']:.1f} m · {result['laps']}바퀴")
    fig.savefig(folder / "path.png", dpi=120, bbox_inches="tight")
