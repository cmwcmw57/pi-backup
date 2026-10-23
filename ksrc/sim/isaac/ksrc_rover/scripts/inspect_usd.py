"""로버 USD 의 충돌 형상·접촉 오프셋·물리 재질을 출력한다 (바퀴 요동 진단 1단계).

  bash run.sh -u /workspace/scripts/inspect_usd.py --headless [--usd /workspace/assets/usd/rover.usd]
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--usd", default="/workspace/assets/usd/rover.usd")
parser.add_argument("--filter", default="", help="경로에 이 문자열이 든 prim 만 (예: wheel)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

try:
    from pxr import PhysxSchema  # noqa: E402
except ImportError:
    PhysxSchema = None

stage = Usd.Stage.Open(args.usd)
print(f"USD {args.usd}  metersPerUnit={UsdGeom.GetStageMetersPerUnit(stage)}", flush=True)


BBOX = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide], useExtentsHint=False)


def attr(prim, name):
    a = prim.GetAttribute(name)
    return a.Get() if a and a.HasAuthoredValue() else None


for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
    path = str(prim.GetPath())
    if args.filter and args.filter not in path:
        continue
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        continue
    info = [f"type={prim.GetTypeName()}"]
    if prim.IsA(UsdGeom.Cylinder):
        c = UsdGeom.Cylinder(prim)
        info.append(f"cyl r={c.GetRadiusAttr().Get()} h={c.GetHeightAttr().Get()} axis={c.GetAxisAttr().Get()}")
    if prim.IsA(UsdGeom.Mesh):
        pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        info.append(f"mesh verts={len(pts) if pts else 0}")
    if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
        info.append(f"approx={UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()}")
    for n in ("physxCollision:contactOffset", "physxCollision:restOffset",
              "physxCollision:torsionalPatchRadius", "physxCollision:minTorsionalPatchRadius",
              "physxConvexHullCollision:hullVertexLimit"):
        v = attr(prim, n)
        if v is not None:
            info.append(f"{n.split(':')[1]}={v}")
    rel = UsdShade.MaterialBindingAPI(prim).GetDirectBinding(materialPurpose="physics").GetMaterial()
    if rel:
        m = rel.GetPrim()
        info.append(f"mat={m.GetPath()} μs={attr(m, 'physics:staticFriction')} "
                    f"μd={attr(m, 'physics:dynamicFriction')} rest={attr(m, 'physics:restitution')}")
    box = BBOX.ComputeWorldBound(prim).ComputeAlignedRange()
    lo, hi = box.GetMin(), box.GetMax()
    info.append(f"AABB x[{lo[0]:+.4f},{hi[0]:+.4f}] y[{lo[1]:+.4f},{hi[1]:+.4f}] z[{lo[2]:+.4f},{hi[2]:+.4f}]")
    print(f"{path}\n    " + "  ".join(info), flush=True)

for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
    if prim.HasAPI(UsdPhysics.MassAPI) and (not args.filter or args.filter in str(prim.GetPath())):
        m = UsdPhysics.MassAPI(prim)
        print(f"MASS {prim.GetPath()} m={m.GetMassAttr().Get()} I={m.GetDiagonalInertiaAttr().Get()}", flush=True)
app.close()
