# ================================================================
# lighting.py
# 설명: 모래 경기장에 그늘이 적은 환경광을 설정하고 광원별 변경값을 기록한다.
# 사용법: from experiments.site_sim.lighting import configure_lighting
# ================================================================
from __future__ import annotations


def configure_lighting(stage: object, profile: str = 'diffuse') -> dict:
    """diffuse는 강한 환경광·약한 태양광, legacy는 기존 전체 광량 ×0.1을 적용한다."""
    from pxr import UsdLux

    if profile not in ('legacy', 'diffuse'):
        raise ValueError('lighting profile must be legacy or diffuse')
    lights = [p for p in stage.Traverse() if p.HasAPI(UsdLux.LightAPI)]
    before = [{'path': str(p.GetPath()), 'type': p.GetTypeName(),
               'intensity': p.GetAttribute('inputs:intensity').Get()} for p in lights]
    for prim in lights:
        attr = prim.GetAttribute('inputs:intensity')
        value = attr.Get()
        if value is not None:
            attr.Set(float(value)*.1 if profile == 'legacy' else 0.)
    if profile == 'diffuse':
        # RTX의 광원 선택 설정에 영향받지 않도록 활성 DomeLight를 하나만 유지한다.
        domes = [p for p in lights if p.IsA(UsdLux.DomeLight)]
        dome = domes[0] if domes else UsdLux.DomeLight.Define(stage, '/World/Lighting').GetPrim()
        dome.GetAttribute('inputs:intensity').Set(600.)
        for other in domes[1:]:
            other.SetActive(False)
        suns = [p for p in lights if p.IsA(UsdLux.DistantLight)]
        if suns:
            suns[0].GetAttribute('inputs:intensity').Set(60.)
    after = [{'path': str(p.GetPath()), 'type': p.GetTypeName(),
              'intensity': p.GetAttribute('inputs:intensity').Get()}
             for p in stage.Traverse() if p.HasAPI(UsdLux.LightAPI)]
    import carb
    return {'profile': profile, 'before': before, 'after': after,
            'renderer_skip_most_lights': carb.settings.get_settings().get('/rtx/scenedb/skipMostLights'),
            'note': 'renderer intensities; not measured real-world illuminance'}
