# ================================================================
# wall_material.py
# 설명: 시뮬 경기장 벽에 반사가 적은 검은색 시각 재질을 적용한다.
# 사용법: from experiments.site_sim.wall_material import set_black_walls
# ================================================================
from __future__ import annotations


def set_black_walls(stage: object) -> dict:
    """Arena 벽 메시의 시각 재질만 무광 검정으로 바꾼다."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    path = '/World/SandScene/Arena'
    wall = stage.GetPrimAtPath(path)
    if not wall.IsValid() or not wall.IsA(UsdGeom.Mesh):
        raise ValueError('arena wall mesh is unavailable: '+path)
    material = UsdShade.Material.Define(stage, '/World/Looks/BlackWalls')
    shader = UsdShade.Shader.Define(stage, '/World/Looks/BlackWalls/Shader')
    shader.CreateIdAttr('UsdPreviewSurface')
    color = (.008, .008, .008)
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(1.)
    shader.CreateInput('useSpecularWorkflow', Sdf.ValueTypeNames.Int).Set(1)
    shader.CreateInput('specularColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    UsdShade.MaterialBindingAPI.Apply(wall).Bind(material)
    return {'prim': path, 'material': str(material.GetPath()), 'diffuse_color_linear': color,
            'roughness': 1., 'specular_color': [0., 0., 0.], 'physics_changed': False}
