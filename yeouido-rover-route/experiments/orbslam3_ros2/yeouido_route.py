# ================================================================
# yeouido_route.py
# 설명: docs/yeouido_route_plan.json의 segment를 순서대로 실행하는 body twist 생성기. 제자리 회전·옆걸음·후진·체크포인트 정지·IMU 기울기 중단을 포함하고, RL 구간은 외부 policy를 끼울 수 있다. 하드웨어를 직접 다루지 않는다.
# 사용법:
#   from experiments.orbslam3_ros2.yeouido_route import RouteRunner
#   runner = RouteRunner.from_json("docs/yeouido_route_plan.json")
#   cmd = runner.command(elapsed_s=t, x=x, y=y, yaw_rad=yaw, pitch_rad=p, roll_rad=r)
# ================================================================
import json
import math
from pathlib import Path
from typing import Callable

from experiments.orbslam3_ros2.swerve import BodyCommand

Policy = Callable[[dict], BodyCommand]
STOP = BodyCommand(0.0, 0.0, 0.0)


def _wrap(angle: float) -> float:
    """각도를 [-pi, pi]로 접는다."""
    return math.atan2(math.sin(angle), math.cos(angle))


def _checkpoints(points: list[tuple[float, float]], spacing: float) -> list[tuple[float, float]]:
    """polyline을 따라 spacing 간격 지점과 각 꼭짓점을 순서대로 반환한다."""
    out = [points[0]]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        length = math.hypot(x1 - x0, y1 - y0)
        for k in range(1, int(length / spacing)):
            out.append((x0 + (x1 - x0) * k * spacing / length, y0 + (y1 - y0) * k * spacing / length))
        out.append((x1, y1))
    return out[1:]


class RouteRunner:
    """segment 목록을 move/rotate/wait step으로 풀어 현재 pose에서 다음 body twist를 만든다."""

    def __init__(self, spec: dict, policy: Policy | None = None) -> None:
        d = spec["defaults"]
        self.max_speed = float(d["max_speed_m_s"])
        self.stop_s = float(d["checkpoint_stop_s"])
        self.abort_pitch = math.radians(d["abort_pitch_deg"])
        self.abort_roll = math.radians(d["abort_roll_deg"])
        self.policy = policy
        self.tol_m, self.tol_yaw, self.ramp_s, self.stall_s = 0.04, math.radians(5.0), 2.0, 8.0
        self.steps = self._build(spec, float(d["checkpoint_spacing_m"]))
        self.index = 0
        self.aborted = False
        self._t_step: float | None = None
        self._progress: tuple[float, float] | None = None
        self._t0: float | None = None

    @classmethod
    def from_json(cls, path: str | Path, policy: Policy | None = None) -> "RouteRunner":
        """경로 JSON 파일에서 runner를 만든다."""
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), policy)

    @property
    def complete(self) -> bool:
        """모든 step을 마쳤는지 반환한다."""
        return self.index >= len(self.steps)

    @property
    def label(self) -> str:
        """현재 step의 segment 이름을 로그용으로 반환한다."""
        return "done" if self.complete else self.steps[self.index]["name"]

    def _build(self, spec: dict, spacing: float) -> list[dict]:
        """JSON segment를 단순 step(move/rotate/wait) 목록으로 변환한다."""
        steps: list[dict] = []
        yaw = math.radians(spec["start"]["yaw_deg"])
        pos = tuple(spec["start"]["base_link_xyz_m"][:2])

        def move(name: str, pts: list, hold: float, rl: bool, stops: bool, stall: bool = False) -> None:
            nonlocal pos
            for p in (_checkpoints([pos, *pts], spacing) if stops else pts):
                steps.append({"kind": "move", "name": name, "xy": tuple(p), "yaw": hold, "rl": rl, "stall": stall})
                if stops:
                    steps.append({"kind": "wait", "name": name, "s": self.stop_s})
            pos = tuple(pts[-1])

        def rotate(name: str, to: float) -> None:
            nonlocal yaw
            steps.append({"kind": "rotate", "name": name, "yaw": to})
            yaw = to

        for seg in spec["segments"]:
            name, rl = seg["name"], seg["controller"] == "RL"
            if "yaw_to_deg" in seg:
                rotate(name, math.radians(seg["yaw_to_deg"]))
                continue
            hold = math.radians(seg.get("hold_yaw_deg", seg.get("yaw_deg", math.degrees(yaw))))
            if abs(_wrap(hold - yaw)) > self.tol_yaw:
                rotate(name, hold)
            if "waypoints_xy_m" in seg:
                fail = seg.get("scripted_failure")
                pts = [tuple(p) for p in seg["waypoints_xy_m"]]
                pts = [p for p in pts if math.hypot(p[0] - pos[0], p[1] - pos[1]) > 0.02]
                move(name, pts, hold, rl, stops=fail is None, stall=fail is not None)
                if fail:
                    steps.append({"kind": "wait", "name": name, "s": fail["wait_s"]})
                    move(name, [tuple(fail["retreat_to_xy_m"])], hold, False, stops=False)
            for sub in seg.get("then", []):
                if "rotate_in_place_to_yaw_deg" in sub:
                    rotate(name, math.radians(sub["rotate_in_place_to_yaw_deg"]))
                    hold = yaw
                else:
                    move(name, [tuple(sub["crab_to_xy_m"])], math.radians(sub["hold_yaw_deg"]), False, False)
            if "crab_to_xy_m" in seg:
                move(name, [tuple(seg["crab_to_xy_m"])], hold, False, stops=False)
        return steps

    def command(self, *, elapsed_s: float, x: float, y: float, yaw_rad: float,
                pitch_rad: float = 0.0, roll_rad: float = 0.0) -> BodyCommand:
        """pose·자세로 현재 step의 body-frame twist를 계산한다. 기울기 한계를 넘으면 영구 정지."""
        if abs(pitch_rad) > self.abort_pitch or abs(roll_rad) > self.abort_roll:
            self.aborted = True
        if self.aborted or self.complete:
            return STOP
        if self._t0 is None:
            self._t0 = elapsed_s
        if self._t_step is None:
            self._t_step = elapsed_s
        step = self.steps[self.index]

        if step["kind"] == "wait":
            if elapsed_s - self._t_step >= step["s"]:
                self._advance()
            return STOP
        err_yaw = _wrap(step["yaw"] - yaw_rad)
        if step["kind"] == "rotate":
            if abs(err_yaw) <= self.tol_yaw:
                self._advance()
                return STOP
            return BodyCommand(0.0, 0.0, max(-0.4, min(0.4, 1.5 * err_yaw)))
        dx, dy = step["xy"][0] - x, step["xy"][1] - y
        dist = math.hypot(dx, dy)
        if dist <= self.tol_m:
            self._advance()
            return STOP
        if step["stall"]:
            best, t_best = self._progress or (dist, elapsed_s)
            if dist < best - 0.02:
                best, t_best = dist, elapsed_s
            self._progress = (best, t_best)
            if elapsed_s - t_best > self.stall_s:  # 실패 가정 구간에서 못 오르면 다음(대기→후퇴)으로
                while not self.complete and self.steps[self.index]["kind"] == "move" and self.steps[self.index]["stall"]:
                    self._advance()
                return STOP
        if step["rl"] and self.policy is not None:
            return self.policy({"x": x, "y": y, "yaw": yaw_rad, "target_xy": step["xy"],
                                "hold_yaw": step["yaw"], "max_speed": self.max_speed})
        speed = min(self.max_speed * min(1.0, (elapsed_s - self._t0) / self.ramp_s), dist)
        wx, wy = speed * dx / dist, speed * dy / dist
        c, s = math.cos(yaw_rad), math.sin(yaw_rad)
        return BodyCommand(c * wx + s * wy, -s * wx + c * wy, max(-0.4, min(0.4, 1.5 * err_yaw)))

    def _advance(self) -> None:
        """다음 step으로 넘어가고 step 시작 시각을 초기화한다."""
        self.index += 1
        self._t_step = None
        self._progress = None
