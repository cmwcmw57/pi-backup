#!/usr/bin/env python3
"""Nav2 `/plan` 의 꺾임각 분포를 집계한다.

왜 필요한가.  학습·평가 경로를 만드는 `sim/rl/path.py` `make_path` 가 막혔을 때
선회각 제한을 170도까지 풀어서, 실측 꺾임각이 **p90 148~166도, 최대 168도** 다
(헤어핀 — 로버가 멈춰서 제자리 회전을 해야 통과한다).  격자 기반 planner 는
한 스텝 최대 45도이고 smoother 를 거치면 더 완만하므로, **Nav2 가 실제로 내는
경로와 모양부터 다르다.**  그 차이가 순수 IK 기준선·s9·s10 판정·기구선호 접기
기각 전부에 실려 있다.

이 스크립트로 실제 분포를 재서 `make_path` 의 fallback 상한을 정한다.

사용법 (Nav2 를 띄운 뒤 별 터미널에서):
    cd ~/ksrc/ros
    source /opt/ros/humble/setup.bash
    export ROS_DOMAIN_ID=77
    export ROS_LOCALHOST_ONLY=1
    python3 plan_stats.py

    # RViz 의 "2D Goal Pose" 로 목표를 여러 번 찍는다.
    # 경로가 올 때마다 한 줄 요약이 나오고, Ctrl-C 로 누적 분포를 출력한다.

꺾임각 정의는 `sim/rl/path.py` 의 측정과 같다: 연속한 두 선분 **진행방향** 사이의
각도 (0도 = 직선, 180도 = 완전 역주).  Nav2 경로는 점 간격이 촘촘해서(코스트맵
해상도 단위) 점마다 재면 양자화 잡음이 지배한다 — `--resample` 간격으로 다시
샘플링한 뒤 잰다.  기본 0.264m 는 pure pursuit 의 선행거리 L 과 같다.
"""
import argparse
import math
import signal
import sys

import numpy as np
import rclpy
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy


def resample(pts, step):
    """호길이 `step` 간격으로 다시 샘플링한 점열."""
    if len(pts) < 2:
        return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(s[-1])
    if total < step * 2:
        return np.array([pts[0], pts[-1]])
    want = np.arange(0.0, total + 1e-9, step)
    return np.stack([np.interp(want, s, pts[:, 0]),
                     np.interp(want, s, pts[:, 1])], axis=1)


def turn_angles(pts):
    """연속 선분 진행방향 사이의 각도 [deg].  0=직선, 180=역주."""
    out = []
    for i in range(1, len(pts) - 1):
        a = pts[i] - pts[i - 1]
        b = pts[i + 1] - pts[i]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            continue
        c = float(np.dot(a, b) / (na * nb))
        out.append(math.degrees(math.acos(max(-1.0, min(1.0, c)))))
    return out


class PlanStats(Node):
    def __init__(self, args):
        super().__init__("plan_stats")
        self.args = args
        self.all_ang = []
        self.n_plans = 0
        self.lengths = []
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Path, args.topic, self.on_plan, qos)
        print(f"{args.topic} 구독 중 | 재샘플 {args.resample:.3f} m | "
              f"RViz 의 '2D Goal Pose' 로 목표를 찍으세요 (Ctrl-C 로 집계)",
              flush=True)

    def on_plan(self, msg):
        pts = np.array([[p.pose.position.x, p.pose.position.y]
                        for p in msg.poses], float)
        if len(pts) < 3:
            return
        raw_len = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        if raw_len < self.args.min_len:
            print(f"  (경로 {raw_len:.2f} m — {self.args.min_len} m 미만, 건너뜀)",
                  flush=True)
            return
        rs = resample(pts, self.args.resample)
        ang = turn_angles(rs)
        if not ang:
            return
        self.n_plans += 1
        self.all_ang += ang
        self.lengths.append(raw_len)
        a = np.array(ang)
        print(f"  경로 {self.n_plans}: 길이 {raw_len:5.2f} m, 원점 {len(pts)}개 "
              f"-> 재샘플 {len(rs)}개 | 꺾임각 평균 {a.mean():5.1f} "
              f"p90 {np.percentile(a, 90):5.1f} 최대 {a.max():5.1f} 도", flush=True)

    def report(self):
        if not self.all_ang:
            print("\n수집된 경로가 없습니다. Nav2 가 떠 있고 ROS_DOMAIN_ID 가 "
                  "런치와 같은지(기본 77) 확인하세요.")
            return
        a = np.array(self.all_ang)
        print(f"\n{'='*62}")
        print(f"Nav2 /plan 꺾임각 분포  (경로 {self.n_plans}개, 꼭짓점 {len(a)}개, "
              f"재샘플 {self.args.resample:.3f} m)")
        print(f"{'='*62}")
        print(f"  경로 길이   평균 {np.mean(self.lengths):.2f} m  "
              f"최소 {min(self.lengths):.2f}  최대 {max(self.lengths):.2f}")
        print(f"  꺾임각      평균 {a.mean():5.1f}  p50 {np.percentile(a,50):5.1f}  "
              f"p90 {np.percentile(a,90):5.1f}  p99 {np.percentile(a,99):5.1f}  "
              f"최대 {a.max():5.1f} 도")
        for t in (30, 45, 60, 90, 120):
            print(f"  {t:3d}도 초과   {100*np.mean(a > t):5.1f} %")
        print(f"\n비교 — sim/rl/path.py make_path (seed 4321, 30경로):")
        print(f"  꺾임각      평균  82.9  p50  64.5  p90 165.9  최대 168.4 도")
        print(f"  60도 초과   56.9 %")
        print(f"\n=> make_path 의 fallback 상한(현재 170도)을 위 p99 근처로 낮출 것.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--topic", default="/plan")
    p.add_argument("--resample", type=float, default=0.264,
                   help="재샘플 간격 [m]. 기본값은 pure pursuit 선행거리 L")
    p.add_argument("--min-len", type=float, default=0.5,
                   help="이 길이 미만의 경로는 무시 [m]")
    args = p.parse_args()
    if args.resample <= 0 or args.min_len <= 0:
        p.error("--resample 과 --min-len 은 0보다 커야 합니다")

    rclpy.init()
    node = PlanStats(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.report()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
