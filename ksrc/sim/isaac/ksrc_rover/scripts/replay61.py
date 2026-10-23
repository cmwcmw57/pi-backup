"""Replay a record_play.py rollout inside the Isaac Sim 6.1 streaming app (kinematic, no physics).

Runs as a Kit startup script: runheadless.sh ... --exec /workspace/scripts/replay61.py
Settings come from env vars (Kit --exec does not forward argv):
  REPLAY_NPZ     recording (default /workspace/logs/play/play.npz)
  REPLAY_USD     rover USD (default /workspace/assets/usd/rover.usd)
  REPLAY_SPEED   playback speed (default 1.0), loops forever
  REPLAY_FOLLOW  robot index the viewport camera follows, -1 = free camera (default 0)
  REPLAY_MARGIN  terrain kept around the robots' path [m] (default 3)
  REPLAY_LIGHT   light intensity multiplier (default 1.0; lower if the scene looks washed out)
  REPLAY_ARROW   arrow length per m/s [m] (default 2.0); 0 = no arrows

Arrows above each robot: green = commanded velocity (vx, vy) and yaw-rate arc,
blue = measured base velocity. Both are body-frame values rotated by the base yaw.

Every link prim gets its recorded world pose each frame (translate + orient ops), so the
robot moves exactly as in the Isaac 5.1 rollout. The timeline is never played, so the
robot's physics APIs stay inert.
"""

import os
import time

import numpy as np
import omni.kit.app
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt

NPZ = os.environ.get("REPLAY_NPZ", "/workspace/logs/play/play.npz")
USD = os.environ.get("REPLAY_USD", "/workspace/assets/usd/rover.usd")
SPEED = float(os.environ.get("REPLAY_SPEED", "1.0"))
FOLLOW = int(os.environ.get("REPLAY_FOLLOW", "0"))
LIGHT = float(os.environ.get("REPLAY_LIGHT", "1.0"))
ARROW = float(os.environ.get("REPLAY_ARROW", "2.0"))


def log(msg):
    print(f"[replay61] {msg}", flush=True)


d = np.load(NPZ, allow_pickle=True)
link_paths = [str(p) for p in d["link_paths"]]
pos, quat = d["body_pos"], d["body_quat"]            # (T,K,B,3), (T,K,B,4 wxyz)
T, K, B, _ = pos.shape
dt = float(d["dt"])
cmd_all, vel_all = d["cmd"], d["vel"]                # (T,K,3) body frame; npz members decompress on every access
log(f"{NPZ}: T={T} K={K} links={B} dt={dt} task={d['task']} ckpt={d['checkpoint']}")

stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
world = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world.GetPrim())


def material(name, rgb, rough=0.6, metal=0.0, emissive=0.0):
    """UsdPreviewSurface material (RTX ignores displayColor on prims that carry the default white material)."""
    mat = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
    sh = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
    if emissive > 0:
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(c * emissive for c in rgb)))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def bind(prim, mat):
    api = UsdShade.MaterialBindingAPI.Apply(prim)
    api.Bind(mat, UsdShade.Tokens.strongerThanDescendants)


MAT = {
    "terrain": material("Terrain", (0.36, 0.30, 0.23), rough=0.95),
    "base": material("Base", (0.18, 0.22, 0.30), rough=0.5),
    "steer": material("Steer", (0.85, 0.45, 0.10), rough=0.5),
    "wheel": material("Wheel", (0.05, 0.05, 0.05), rough=0.9),
    "marker": material("Marker", (0.90, 0.12, 0.08), rough=0.4),
    "cmd": material("Cmd", (0.10, 0.85, 0.20), rough=0.4, emissive=0.6),
    "vel": material("Vel", (0.15, 0.45, 1.00), rough=0.4, emissive=0.6),
}


def link_material(rel):
    name = rel.rsplit("/", 1)[-1]
    for key in ("wheel", "steer", "marker"):
        if key in name:
            return MAT[key]
    return MAT["base"]

# terrain, cropped to where the robots go (+ REPLAY_MARGIN m) -- the full 8x16 m map is 1.3 M points
tp, tf = d["terrain_points"], d["terrain_faces"]
if len(tp):
    margin = float(os.environ.get("REPLAY_MARGIN", "3.0"))
    base = pos[:, :, 0, :2].reshape(-1, 2)
    lo, hi = base.min(0) - margin, base.max(0) + margin
    inside = np.all((tp[:, :2] >= lo) & (tp[:, :2] <= hi), axis=1)
    keep = inside[tf].all(axis=1)
    tf = tf[keep]
    used = np.unique(tf)
    remap = np.full(len(tp), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    tp, tf = tp[used], remap[tf]
    mesh = UsdGeom.Mesh.Define(stage, "/World/Terrain")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(tp, dtype=np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(tf), 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(tf.reshape(-1), dtype=np.int32)))
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.36, 0.30, 0.23)])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    bind(mesh.GetPrim(), MAT["terrain"])
    log(f"terrain {len(tp)} points")
dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
dome.CreateIntensityAttr(300.0 * LIGHT)
dome.CreateColorAttr(Gf.Vec3f(0.80, 0.86, 1.0))
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(1200.0 * LIGHT)
sun.CreateAngleAttr(1.0)
UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 20.0, 0.0))

# robots: reference the rover USD K times, take over every link's transform
ops = []          # per robot: list of (translate_op, orient_op) per link, None if link missing
for k in range(K):
    root = f"/World/Rover_{k}"
    prim = stage.DefinePrim(root, "Xform")
    prim.GetReferences().AddReference(USD)
    per = []
    missing = []
    for rel in link_paths:
        lp = stage.GetPrimAtPath(f"{root}/{rel}")
        if not lp.IsValid():
            per.append(None)
            missing.append(rel)
            continue
        xf = UsdGeom.Xformable(lp)
        xf.ClearXformOpOrder()
        t = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        o = xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        per.append((t, o, lp.GetParent().GetPath().pathString))
        bind(lp, link_material(rel))
    ops.append(per)
    if missing:
        log(f"robot {k}: links not found in USD: {missing}")
link_index = {f"/World/Rover_{k}/{rel}": i for k in range(K) for i, rel in enumerate(link_paths)}


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _qrot(q, v):
    w, x, y, z = q
    u = np.array([x, y, z])
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


# velocity arrows: shaft = unit cylinder along +X scaled to length, head = cone at the tip
def make_arrow(path, mat, radius):
    root = UsdGeom.Xform.Define(stage, path)
    r_t, r_o = root.AddTranslateOp(), root.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    shaft = UsdGeom.Cylinder.Define(stage, f"{path}/Shaft")
    shaft.CreateAxisAttr("X")
    shaft.CreateHeightAttr(1.0)
    shaft.CreateRadiusAttr(radius)
    s_t, s_s = shaft.AddTranslateOp(), shaft.AddScaleOp()
    head = UsdGeom.Cone.Define(stage, f"{path}/Head")
    head.CreateAxisAttr("X")
    head.CreateHeightAttr(radius * 5)
    head.CreateRadiusAttr(radius * 2.5)
    h_t = head.AddTranslateOp()
    for g in (shaft, head):
        bind(g.GetPrim(), mat)
    return dict(img=UsdGeom.Imageable(root), t=r_t, o=r_o, st=s_t, ss=s_s, ht=h_t, hl=radius * 5)


def make_arc(path, mat):
    curve = UsdGeom.BasisCurves.Define(stage, path)
    curve.CreateTypeAttr(UsdGeom.Tokens.linear)
    curve.CreateWidthsAttr([0.02])
    curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    bind(curve.GetPrim(), mat)
    head = UsdGeom.Cone.Define(stage, f"{path}_head")
    head.CreateHeightAttr(0.06)
    head.CreateRadiusAttr(0.025)
    h_t, h_o = head.AddTranslateOp(), head.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    bind(head.GetPrim(), mat)
    return dict(curve=curve, head=UsdGeom.Imageable(head), ht=h_t, ho=h_o)


def _yawq(yaw):
    return Gf.Quatd(np.cos(yaw / 2), Gf.Vec3d(0, 0, np.sin(yaw / 2)))


def set_arrow(a, origin, yaw, vxy):
    """vxy: body-frame (vx, vy) [m/s]; drawn in world frame at origin."""
    L = float(np.hypot(*vxy)) * ARROW
    if L < 0.01:
        a["img"].MakeInvisible()
        return
    a["img"].MakeVisible()
    a["t"].Set(Gf.Vec3d(*origin))
    a["o"].Set(_yawq(yaw + float(np.arctan2(vxy[1], vxy[0]))))
    shaft = max(L - a["hl"], 0.005)
    a["st"].Set(Gf.Vec3d(shaft / 2, 0, 0))
    a["ss"].Set(Gf.Vec3f(shaft, 1, 1))
    a["ht"].Set(Gf.Vec3d(shaft + a["hl"] / 2, 0, 0))


ARC_R = 0.32          # yaw-rate arc radius [m]; 1 rad/s -> 90 deg of arc


def set_arc(a, origin, yaw, wz):
    sweep = float(np.clip(wz, -2, 2)) * np.pi / 2
    if abs(sweep) < 0.05:
        a["curve"].MakeInvisible()
        a["head"].MakeInvisible()
        return
    a["curve"].MakeVisible()
    a["head"].MakeVisible()
    th = yaw + np.linspace(0, sweep, 24)
    pts = np.c_[origin[0] + ARC_R * np.cos(th), origin[1] + ARC_R * np.sin(th), np.full(len(th), origin[2])]
    a["curve"].GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
    a["curve"].GetCurveVertexCountsAttr().Set([len(th)])
    a["ht"].Set(Gf.Vec3d(*pts[-1]))
    # cone axis is +Z by default: tip along the tangent direction (sign of wz)
    tang = np.array([-np.sin(th[-1]), np.cos(th[-1]), 0.0]) * np.sign(sweep)
    q = Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*tang)).GetQuat()
    a["ho"].Set(q)


arrows = []
if ARROW > 0:
    for k in range(K):
        arrows.append(dict(cmd=make_arrow(f"/World/Arrows/Cmd_{k}", MAT["cmd"], 0.012),
                           vel=make_arrow(f"/World/Arrows/Vel_{k}", MAT["vel"], 0.009),
                           arc=make_arc(f"/World/Arrows/Yaw_{k}", MAT["cmd"])))
    log(f"arrows: green = command, blue = measured, {ARROW} m per m/s")


def set_frame(f):
    with Sdf.ChangeBlock():
        for k in range(K):
            for i, item in enumerate(ops[k]):
                if item is None:
                    continue
                t_op, o_op, parent = item
                p, q = pos[f, k, i].astype(float), quat[f, k, i].astype(float)
                j = link_index.get(parent)
                if j is not None:      # nested link: express relative to the parent link's recorded pose
                    pp, pq = pos[f, k, j].astype(float), quat[f, k, j].astype(float)
                    pqi = pq * np.array([1, -1, -1, -1])
                    p = _qrot(pqi, p - pp)
                    q = _qmul(pqi, q)
                t_op.Set(Gf.Vec3d(*p))
                o_op.Set(Gf.Quatd(q[0], Gf.Vec3d(*q[1:])))
        for k, a in enumerate(arrows):
            bp, bq = pos[f, k, 0].astype(float), quat[f, k, 0].astype(float)
            w, x, y, z = bq
            yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
            c, v = cmd_all[f, k], vel_all[f, k]
            set_arrow(a["cmd"], bp + [0, 0, 0.30], yaw, c[:2])
            set_arrow(a["vel"], bp + [0, 0, 0.27], yaw, v[:2])
            set_arc(a["arc"], bp + [0, 0, 0.30], yaw, c[2])


cam = UsdGeom.Camera.Define(stage, "/World/FollowCam")
cam_t = UsdGeom.Xformable(cam).AddTranslateOp()
cam_r = UsdGeom.Xformable(cam).AddRotateXYZOp()
cam.CreateFocalLengthAttr(18.0)
try:   # both modes use FollowCam; free mode places it once and then leaves it to the mouse
    from omni.kit.viewport.utility import get_active_viewport
    get_active_viewport().camera_path = "/World/FollowCam"
    log("viewport camera -> /World/FollowCam " + ("(follows robot)" if FOLLOW >= 0 else "(free: right-drag/scroll)"))
except Exception as e:  # viewport API differs between Kit versions
    log(f"could not set viewport camera: {e}")

set_frame(0)
_c = pos[0, 0, 0].astype(float)
cam_t.Set(Gf.Vec3d(_c[0] - 1.6, _c[1] - 1.6, _c[2] + 1.1))
cam_r.Set(Gf.Vec3f(62.0, 0.0, -45.0))
state = {"t0": time.time(), "last": -1}


def on_update(_):
    f = int((time.time() - state["t0"]) * SPEED / dt) % T
    if f == state["last"]:
        return
    state["last"] = f
    set_frame(f)
    if FOLLOW >= 0:
        c = pos[f, FOLLOW, 0].astype(float)            # base_link is link 0
        cam_t.Set(Gf.Vec3d(c[0] - 1.6, c[1] - 1.6, c[2] + 1.1))
        cam_r.Set(Gf.Vec3f(62.0, 0.0, -45.0))
    if f % int(round(1.0 / dt)) == 0:
        cmd = cmd_all[f, max(FOLLOW, 0)]
        vel = vel_all[f, max(FOLLOW, 0)]
        log(f"t={f * dt:6.2f}s cmd vx {cmd[0]:+.2f} vy {cmd[1]:+.2f} wz {cmd[2]:+.2f} | "
            f"meas vx {vel[0]:+.2f} vy {vel[1]:+.2f} wz {vel[2]:+.2f}")


_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(on_update, name="ksrc_replay")
log(f"playing {T * dt:.1f}s x{SPEED} in a loop")
