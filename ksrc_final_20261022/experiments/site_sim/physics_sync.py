# ================================================================
# physics_sync.py
# 설명: 원래 충돌 형상을 유지하고 Claude 정합의 지면·바퀴 재질과 쿨롱 마찰을 임무에 연결한다.
# 사용법: python -m experiments.site_sim.run ... --parity-physics
# ================================================================
from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Any

import numpy as np

from experiments.site_sim.ksrc_actuation import FirmwareActuation, motor_effort
from experiments.site_sim.ksrc_soil import IsaacSoil, SoilParameters, SoilState

WHEEL_FRICTION_NM = 0.13001251564455568
WHEEL_INERTIA = 6.23819e-5 + .0098
REFERENCE_SOIL = SoilParameters(K=.16895121324729193, alpha=.04646353320394744,
    beta=.5714808553175139, c_r=.04067171150610929, k_z=8.73489742857582,
    z_max=.056623766036963945, mu_lat=.5221573489446142)


def validate_options(args: argparse.Namespace) -> None:
    """정합 물리는 기록된 μ=.4와 KSRC 구동·토양 조합에서만 허용한다."""
    if not getattr(args, 'parity_physics', False):return
    if args.actuator_model != 'ksrc-pi' or args.soil_model != 'ksrc-legacy' or args.mu is None or not np.isclose(args.mu, .4):
        raise ValueError('--parity-physics requires ksrc-pi, ksrc-legacy and --mu .4')


def bind_ground_friction(stage: Any) -> list[str]:
    """정적 지면 collider에 μ=.001/max를 physics 초기화 전에 바인딩한다."""
    from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade
    mat = UsdShade.Material.Define(stage, '/World/Physics/SoilGround')
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(.001);api.CreateDynamicFrictionAttr(.001);api.CreateRestitutionAttr(0.)
    PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim()).CreateFrictionCombineModeAttr('max')
    meshes = [p for p in Usd.PrimRange(stage.GetPrimAtPath('/World/SandScene'), Usd.TraverseInstanceProxies())
              if p.GetTypeName() == 'Mesh' and p.HasAPI(UsdPhysics.CollisionAPI)]
    if not meshes:raise RuntimeError('no ground collider meshes to bind friction to')
    for p in meshes:
        UsdShade.MaterialBindingAPI.Apply(p).Bind(mat, UsdShade.Tokens.strongerThanDescendants, 'physics')
    return [str(p.GetPath()) for p in meshes]


def write_wheel_materials(views: list, friction: list, *, verify: bool = False) -> list:
    """각 바퀴 solver μ를 직접 갱신하고 요청 시 실제 tensor 값을 읽어 검증한다."""
    read = []
    for view, mu in zip(views, friction):
        values = np.array([[[mu, mu, 0.]]], dtype=np.float32)
        view._physics_view.set_material_properties(values, np.array([0], dtype=np.int32))
        if verify:
            actual = np.asarray(view._physics_view.get_material_properties())
            if actual.shape != values.shape or not np.allclose(actual, values, atol=1e-6):
                raise RuntimeError(f'solver material mismatch: {actual} != {values}')
            read.extend(actual.tolist())
    return read


class TensorMaterialSoil(IsaacSoil):
    """기존 모래 저항에 PhysX solver μ 갱신과 정합의 고정 계수를 적용한다."""

    def initialize(self) -> None:
        super().initialize()
        self.state = SoilState(.4, REFERENCE_SOIL)
        self.solver_materials = write_wheel_materials(self.views, [.4*REFERENCE_SOIL.mu_lat]*4, verify=True)

    def apply(self) -> None:
        super().apply()
        write_wheel_materials(self.views, self.record['friction'])
        self.record['material_update'] = 'physx-tensor'


def coulomb_friction_torque(speed: np.ndarray, motor: np.ndarray) -> np.ndarray:
    """한 5ms step의 정지 필요 토크를 쿨롱 한계로 제한해 역회전을 방지한다."""
    return -np.clip(WHEEL_INERTIA*np.asarray(speed)/.005 + np.asarray(motor), -WHEEL_FRICTION_NM, WHEEL_FRICTION_NM)


class CoulombActuation(FirmwareActuation):
    """기존 펌웨어 PI·서보 위에 정지 중에도 작동하는 바퀴 쿨롱 토크를 더한다."""

    def apply(self) -> None:
        s = self.session
        profile = self.servo.step(self.goal)
        speed = np.asarray(self.robot.get_joint_velocities())[s.wheel_indices]
        motor = motor_effort(self.duty, speed)
        self.friction_effort = coulomb_friction_torque(speed, motor)
        self.effort = motor + self.friction_effort
        s.controller.apply_action(s._action_type(joint_positions=profile, joint_indices=s.steering_indices))
        s.controller.apply_action(s._action_type(joint_efforts=self.effort, joint_indices=s.wheel_indices))

    def stop(self) -> None:
        self.duty[:] = 0.
        for p in self.pi:p.reset()
        self.apply()

    def telemetry(self) -> dict:
        return {**super().telemetry(), 'wheel_friction_effort_nm':self.friction_effort.tolist()}


def settle_stopped(world: Any, actuator: CoulombActuation, soil: TensorMaterialSoil, steps: int) -> None:
    """종료 구간에도 매 물리 step 토크와 모래 저항을 갱신해 고정 제동 토크의 역회전을 막는다."""
    for _ in range(steps):
        actuator.stop();soil.apply();world.step(render=False)


def physics_evidence(stage: Any, soil: TensorMaterialSoil, ground_paths: list[str]) -> dict:
    """실제 임무 지형·네 바퀴 collider 및 solver 재질을 실행 산출물에 남긴다."""
    from pxr import Usd, UsdGeom, UsdPhysics
    wheels = []
    for view in soil.views:
        body = view.prim_paths[0]
        for prim in Usd.PrimRange(stage.GetPrimAtPath(body), Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.CollisionAPI) and prim.IsA(UsdGeom.Mesh):
                pts = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
                extent = np.ptp(pts, axis=0)
                approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
                wheels.append({'path':str(prim.GetPath()), 'points':len(pts),
                    'extent_m':extent.tolist(), 'approximation':approximation})
    if len(wheels) != 4:raise RuntimeError(f'expected 4 wheel mesh colliders, got {len(wheels)}')
    return {'enabled':True, 'reference_commit':'fc716e2', 'ground_colliders':ground_paths,
        'wheel_geometry':'original URDF mesh; cylinder replacement excluded by user',
        'ground_friction':.001, 'combine':'max', 'wheel_colliders':wheels,
        'solver_materials':[np.asarray(v._physics_view.get_material_properties()).tolist() for v in soil.views],
        'soil':asdict(REFERENCE_SOIL), 'mu_max':.4, 'wheel_friction_nm':WHEEL_FRICTION_NM,
        'wheel_armature':.0098, 'initial_sinkage_m':[0.]*4,
        'scope':'current mission terrain; conditional result-level parity, not full physical equivalence'}
