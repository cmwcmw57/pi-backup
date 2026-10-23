# ================================================================
# friction.py
# 설명: 시뮬 지면 mesh와 바퀴 충돌체에 같은 정지·동마찰 mu의 PhysX 재질을 묶는다(모래 마찰을 임무 결과에 맞춰 보정하는 용도).
# 사용법: ground, wheels = bind_friction(stage, 0.25)   # Isaac 안에서 run.py의 --mu가 호출
# ================================================================


def bind_friction(stage: object, mu: float, ground_path: str = '/World/SandScene', rover_path: str = '/World/Rover') -> tuple[int, int]:
    """(묶은 지면 mesh 수, 바퀴 충돌체 수)를 돌려준다. 둘 중 하나라도 0이면 예외를 낸다."""
    if not 0 < mu <= 2:
        raise ValueError('mu must be in (0, 2]')
    from pxr import Usd, UsdPhysics, UsdShade
    material = UsdShade.Material.Define(stage, '/World/Physics/FrictionMat')
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(mu)
    api.CreateDynamicFrictionAttr(mu)
    api.CreateRestitutionAttr(0.0)
    ground = [p for p in Usd.PrimRange(stage.GetPrimAtPath(ground_path), Usd.TraverseInstanceProxies()) if p.GetTypeName() == 'Mesh']
    wheels = [p for p in Usd.PrimRange(stage.GetPrimAtPath(rover_path)) if p.HasAPI(UsdPhysics.CollisionAPI) and 'wheel' in str(p.GetPath()).lower()]
    if not ground or not wheels:
        raise RuntimeError(f'friction binding found ground={len(ground)} wheel colliders={len(wheels)}')
    for prim in ground+wheels:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.weakerThanDescendants, 'physics')
    return len(ground), len(wheels)
