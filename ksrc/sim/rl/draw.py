"""컴파일된 MJCF 를 2D 직교투영으로 그린다. 원통은 축 방향에 따라 원/사각으로."""
import numpy as np, mujoco, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from config import RoverCfg; from mjcf import build
G = mujoco.mjtGeom

def _poly(ax, pts, col):
    k = pts[np.argsort(np.arctan2(pts[:,1]-pts[:,1].mean(), pts[:,0]-pts[:,0].mean()))]
    ax.add_patch(Polygon(k, fc=col, ec='k', lw=.5, alpha=.85))

def draw(ax, m, d, i, j, title, ground=True):
    for g in range(m.ngeom):
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "") == "ground": continue
        p, R, s, t = d.geom_xpos[g], d.geom_xmat[g].reshape(3,3), m.geom_size[g], m.geom_type[g]
        col = tuple(m.geom_rgba[g][:3])
        if t in (G.mjGEOM_CYLINDER, G.mjGEOM_CAPSULE):
            ax_w = R[:, 2]                                  # 원통 축 (월드)
            n = np.zeros(3); n[[k for k in (0,1,2) if k not in (i,j)][0]] = 1.0
            if abs(ax_w @ n) > 0.95:                        # 축이 화면과 수직 -> 원
                ax.add_patch(Circle((p[i], p[j]), s[0], fc=col, ec='k', lw=.5, alpha=.85))
                continue
            half = s[1] + (s[0] if t == G.mjGEOM_CAPSULE else 0.0)
            e1 = ax_w * half
            e2 = np.cross(ax_w, n); e2 = e2/ (np.linalg.norm(e2)+1e-12) * s[0]
            c = np.array([p+e1+e2, p+e1-e2, p-e1-e2, p-e1+e2])
            _poly(ax, c[:, [i, j]], col); continue
        if t != G.mjGEOM_BOX: continue
        c = np.array(np.meshgrid([-1,1],[-1,1],[-1,1])).T.reshape(-1,3) * s[:3]
        _poly(ax, ((R @ c.T).T + p)[:, [i, j]], col)
    ax.set_aspect('equal'); ax.grid(alpha=.3, lw=.4); ax.set_title(title, fontsize=10)
    ax.relim(); ax.autoscale_view(); ax.tick_params(labelsize=7)
    if ground: ax.axhline(0, color='saddlebrown', lw=1.2, alpha=.7)

if __name__ == "__main__":
    C = RoverCfg()
    xml = build(C).replace('<body name="chassis"',
        '<geom name="bump" type="box" pos="%.4f %.4f 0.02" size="0.05 0.05 0.02" '
        'rgba="0.75 0.25 0.2 1"/>\n    <body name="chassis"' % (C.axle_x, C.track/2))
    m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
    for _ in range(1800): mujoco.mj_step(m, d)
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    draw(ax[0], m, d, 0, 1, "top  X-Y   (x=forward, y=left)", False)
    draw(ax[1], m, d, 0, 2, "side X-Z   (40mm bump under front-left)")
    draw(ax[2], m, d, 1, 2, "front Y-Z   <- motor sticks out sideways")
    plt.tight_layout(); plt.savefig("rover_model.png", dpi=115)
    V = np.array([d.geom_xpos[g] for g in range(m.ngeom)
                  if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "") not in ("ground","bump")])
    bp = lambda w: d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"wheel_{w}")]
    print("휠베이스 %.1f  트랙 %.1f mm" % (abs(bp('fl')[0]-bp('rl')[0])*1000, abs(bp('fl')[1]-bp('fr')[1])*1000))
    print("전장 %.1f mm   전폭(모터 포함) %.1f mm" % (
        (abs(bp('fl')[0]-bp('rl')[0]) + 2*C.wheel_r)*1000,
        (abs(bp('fl')[1]-bp('fr')[1]) + C.wheel_w)*1000))
    print("모터 방향 %+.0f (%s)" % (C.motor_y_dir, "안쪽" if C.motor_y_dir < 0 else "바깥쪽"))
