"""지형 지도에서 **마우스 클릭으로 경로를 찍는다**.

MuJoCo passive viewer 는 마우스 콜백이 없다 (`launch_passive` 는 `key_callback`
만 받고, `Handle.perturb.select` 는 body 선택이라 지면의 한 점을 못 준다).
그래서 위에서 본 2D 지도에서 찍고, 그 경로를 `view.py --path` 로 재생한다.

나중에 Nav2 가 붙으면 RViz 의 "2D Goal Pose" 가 이 역할을 대체한다.

조작
  좌클릭      웨이포인트 추가
  우클릭 / z  마지막 점 취소
  c           전부 지우기
  Enter / s   저장하고 종료
  q           저장 없이 종료
"""
import argparse
import pathlib

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import numpy as np

import path as pth
import terrain as terr
from config import RoverCfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrain", default="sand", choices=["sand", "rock"])
    ap.add_argument("--out", default="path.npy")
    ap.add_argument("--max-slope", type=float, default=None,
                    help="주행가능 판정 경사 상한 [deg]. 기본은 config.max_slope_deg")
    args = ap.parse_args()

    cfg = RoverCfg()
    lim = args.max_slope if args.max_slope is not None else cfg.max_slope_deg
    Z = terr.load_arena(args.terrain)
    ex, ey = terr.eval_extent(args.terrain)
    drive, slope = terr.drivable_mask(Z, ex, ey, max_slope_deg=lim)
    drive = terr.center_clearance_mask(drive, ex, ey, cfg.route_clearance)

    fig, ax = plt.subplots(figsize=(7.5, 9))
    ext = [-ex / 2, ex / 2, -ey / 2, ey / 2]
    ax.imshow(1000 * (Z - Z.min()), origin="lower", extent=ext, cmap="terrain")
    # 주행 불가 영역을 빨갛게 덮는다
    bad = np.ma.masked_where(drive, np.ones_like(Z))
    ax.imshow(bad, origin="lower", extent=ext, cmap="autumn", alpha=0.35)
    ax.set_xlabel("x [m]  (전진)")
    ax.set_ylabel("y [m]  (좌측)")
    ax.set_title(f"{args.terrain}  경사상한 {lim:.0f}도 (빨강=주행불가)\n"
                 "좌클릭 추가 · 우클릭/z 취소 · c 전체삭제 · Enter 저장 · q 취소",
                 fontsize=10)
    ax.grid(alpha=0.25, linewidth=0.5)

    pts = []
    line, = ax.plot([], [], "o-", color="black", ms=6, lw=2, zorder=5)
    warn = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top",
                   fontsize=9, color="crimson")

    def redraw():
        if pts:
            P = np.array(pts)
            line.set_data(P[:, 0], P[:, 1])
        else:
            line.set_data([], [])
        # 각 구간이 주행 가능한지 검사해서 알려준다
        msgs = []
        for i in range(len(pts) - 1):
            if not terr.line_clear(drive, ex, ey, np.array(pts[i]), np.array(pts[i + 1])):
                msgs.append(f"구간 {i}-{i+1} 통과 불가")
        tot = sum(float(np.linalg.norm(np.array(pts[i + 1]) - np.array(pts[i])))
                  for i in range(len(pts) - 1))
        warn.set_text(("점 %d개  길이 %.2fm\n" % (len(pts), tot)) + "\n".join(msgs))
        fig.canvas.draw_idle()

    def on_click(ev):
        if ev.inaxes is not ax or ev.xdata is None:
            return
        if ev.button == 1:
            pts.append((float(ev.xdata), float(ev.ydata)))
        elif ev.button == 3 and pts:
            pts.pop()
        redraw()

    state = {"save": False}

    def on_key(ev):
        if ev.key in ("z", "backspace") and pts:
            pts.pop(); redraw()
        elif ev.key == "c":
            pts.clear(); redraw()
        elif ev.key in ("enter", "s"):
            state["save"] = True; plt.close(fig)
        elif ev.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    plt.tight_layout()
    plt.show()

    if not state["save"]:
        print("저장하지 않음"); return
    if len(pts) < 2:
        print("점이 2개 미만이라 저장하지 않음"); return
    P = np.array(pts, float)
    rp = pth.RefPath(P)
    np.save(args.out, P)
    print("저장: %s   꼭짓점 %d개   길이 %.2fm" % (args.out, len(rp.P), rp.total))
    bad_seg = [i for i in range(len(rp.P) - 1)
               if not terr.line_clear(drive, ex, ey, rp.P[i], rp.P[i + 1])]
    if bad_seg:
        print("경고: 통과 불가 구간 %s -- 로버가 못 갈 수 있다" % bad_seg)
    print("재생:  python3 view.py --terrain %s --path %s "
          "[--model runs/s1/final --vecnorm runs/s1/vecnorm.pkl]"
          % (args.terrain, args.out))


if __name__ == "__main__":
    main()
