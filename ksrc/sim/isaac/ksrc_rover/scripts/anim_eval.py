"""eval_log.py 가 덤프한 npz 를 탑다운 애니메이션(mp4)으로 — Isaac 렌더 불필요(numpy+matplotlib).

이 이미지엔 omni.replicator/Xvfb 가 없어 Isaac 오프스크린 렌더가 세그폴트한다. 대신
npz 의 root_pos/quat·steer·cmd/vel 로 로버 거동을 그린다: 차체(yaw)·4바퀴 조향·이동경로·
명령속도(초록)vs실제속도(빨강) 화살표. 노트북에서 돌리고 바로 재생하면 된다.

사용:
  python3 anim_eval.py B_test.npz out.mp4 --envs 0,1,2,3 --fps 25
"""
import argparse

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import Rectangle

# 차체 치수(모듈 범위에서 대략). module_xy 로 그려서 실제 트랙폭 반영.
BODY_L, BODY_W = 0.33, 0.33


def yaw_from_quat(q):  # q: (...,4) w,x,y,z
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def rot(a):  # 2x2 회전행렬 스택, a:(...)
    c, s = np.cos(a), np.sin(a)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("out")
    ap.add_argument("--envs", default="0,1,2,3")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--trail", type=int, default=40, help="경로 꼬리 길이(스텝)")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    envs = [int(e) for e in args.envs.split(",")]
    T = z["cmd"].shape[0]
    pos = z["root_pos"][..., :2].astype(float)          # (T,N,2) world
    yaw = yaw_from_quat(z["root_quat"].astype(float))    # (T,N)
    steer = z["steer"].astype(float)                     # (T,N,4)
    cmd = z["cmd"].astype(float); vel = z["vel"].astype(float)  # (T,N,3) body
    mu_l, mu_r = z["mu_l"], z["mu_r"]
    done = z["done"].astype(bool)
    mod = np.asarray(z["module_xy"], float)              # (4,2) body

    # env 별로 시작점 기준 재중심(겹쳐 보기).
    p0 = pos[0:1, :, :].copy()
    pos = pos - p0

    n = len(envs)
    ncol = 2 if n > 1 else 1
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 5.0 * nrow))
    axes = np.atleast_1d(axes).ravel()

    # 축 범위: 각 env 경로 범위 + 여유.
    arts = []
    for ax, e in zip(axes, envs):
        pe = pos[:, e, :]
        lo = np.nanmin(pe, 0) - 0.6; hi = np.nanmax(pe, 0) + 0.6
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_title(f"env {e}")
        trail, = ax.plot([], [], "-", color="0.6", lw=1)
        body = Rectangle((0, 0), BODY_L, BODY_W, fc="none", ec="k", lw=1.5)
        ax.add_patch(body)
        wheels = [ax.plot([], [], "-", color="tab:blue", lw=2.5)[0] for _ in range(4)]
        cmd_a = ax.annotate("", (0, 0), (0, 0), arrowprops=dict(arrowstyle="->", color="tab:green", lw=2))
        vel_a = ax.annotate("", (0, 0), (0, 0), arrowprops=dict(arrowstyle="->", color="tab:red", lw=2))
        txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=8,
                      bbox=dict(fc="white", alpha=0.7, lw=0))
        arts.append(dict(ax=ax, e=e, trail=trail, body=body, wheels=wheels,
                         cmd=cmd_a, vel=vel_a, txt=txt))
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"{str(z['task'])}   green=cmd vel, red=actual vel   wheel-seed={int(z['wheel_seed'])}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    def frame(t):
        for a in arts:
            e = a["e"]; c, s = np.cos(yaw[t, e]), np.sin(yaw[t, e])
            R = np.array([[c, -s], [s, c]]); P = pos[t, e]
            # 경로(가장 최근 리셋 이후부터만 — 점프선 방지)
            lo = max(0, t - args.trail)
            dmask = done[lo:t + 1, e]
            start = lo + int(np.where(dmask)[0][-1]) + 1 if dmask.any() else lo
            seg = np.arange(start, t + 1)
            a["trail"].set_data(pos[seg, e, 0], pos[seg, e, 1])
            # 차체 사각형(중심정렬 + yaw)
            a["body"].set_xy((-BODY_L / 2, -BODY_W / 2))
            tr = matplotlib.transforms.Affine2D().rotate(yaw[t, e]).translate(*P) + a["ax"].transData
            a["body"].set_transform(tr)
            # 바퀴(모듈위치 + 조향각), body→world
            for j in range(4):
                wc = R @ mod[j] + P
                d = R @ (np.array([np.cos(steer[t, e, j]), np.sin(steer[t, e, j])]) * 0.09)
                a["wheels"][j].set_data([wc[0] - d[0], wc[0] + d[0]], [wc[1] - d[1], wc[1] + d[1]])
            # 속도 화살표(body→world)
            cv = R @ cmd[t, e, :2]; vv = R @ vel[t, e, :2]
            a["cmd"].set_position((P[0], P[1])); a["cmd"].xy = (P[0] + cv[0], P[1] + cv[1])
            a["vel"].set_position((P[0], P[1])); a["vel"].xy = (P[0] + vv[0], P[1] + vv[1])
            a["txt"].set_text(f"t={t}\nμL={mu_l[t,e]:.2f} μR={mu_r[t,e]:.2f}\n"
                              f"cmd wz={cmd[t,e,2]:+.2f}\nerr|v|={np.linalg.norm(cmd[t,e,:2]-vel[t,e,:2]):.2f}")
        return []

    ani = FuncAnimation(fig, frame, frames=T, interval=1000 / args.fps, blit=False)
    ani.save(args.out, writer=FFMpegWriter(fps=args.fps, bitrate=2400))
    print(f"ANIM_SAVED {args.out}  frames={T} envs={envs}")


if __name__ == "__main__":
    main()
