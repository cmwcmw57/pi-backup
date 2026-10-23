# ================================================================
# ksrc_soil.py
# 설명: sim2sim legacy 모래의 바퀴별 슬립·가상 침하·마찰·소산 저항을 Isaac에 연결한다.
# 사용법: IsaacSoil(stage, robot, session, mu_max=.4), initialize(), apply()
# 참고: YunBeeeen/ksrc @6023c2a sim/rl/terramech.py (z는 시각 mesh 변형이 아님)
# ================================================================
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from experiments.site_sim.joint_convention import from_joint


def prepare_contact_reports(stage) -> None:
    """물리 actor 생성 전에 접촉 보고를 켜서 초기화 뒤의 누락을 피한다."""
    from pxr import PhysxSchema, Usd, UsdPhysics
    for root in ('/World/Rover','/World/SandScene'):
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies()):
            if prim.IsInstanceProxy():continue
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.CollisionAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)


@dataclass(frozen=True)
class SoilParameters:
    """MuJoCo와 Isaac의 고정 legacy 비교에 함께 사용하는 토양 계수."""

    K: float = .15
    alpha: float = .058
    beta: float = .5
    c_r: float = .05
    k_z: float = 8.
    z_max: float = .05
    mu_lat: float = .6


class SoilState:
    """legacy 모래 식을 module 순서 FL,FR,RL,RR의 독립 상태로 적분한다."""

    def __init__(self, mu_max: float, parameters: SoilParameters = SoilParameters()) -> None:
        self.mu_max = float(mu_max)
        self.parameters = parameters
        self.z = np.zeros(4)

    def step(self, rim: np.ndarray, ground: np.ndarray, normal: np.ndarray) -> tuple:
        p = self.parameters
        den = np.maximum(np.maximum(abs(rim), abs(ground)), 1e-3)
        slip = np.clip((rim-ground)/den, -1., 1.)
        dig = (normal > 1e-6)*p.alpha*abs(slip)*np.maximum(abs(rim), abs(ground))
        self.z = np.clip(self.z+.005*(dig-p.beta*self.z), 0., p.z_max)
        mu = self.mu_max*np.maximum(1.-np.exp(-abs(slip)/p.K), p.mu_lat)
        resistance = normal*(p.c_r+p.k_z*self.z)
        return slip, mu, resistance


class IsaacSoil:
    """바퀴 contact normal을 읽고 매 물리 step 재질 마찰과 CoM 저항력을 적용한다."""

    def __init__(self, stage, robot, session, mu_max: float) -> None:
        from isaacsim.core.prims import RigidPrim
        from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade
        self.robot, self.session = robot, session
        self.state = SoilState(mu_max)
        self.views, self.materials = [], []
        ground = [p for p in Usd.PrimRange(stage.GetPrimAtPath('/World/SandScene'), Usd.TraverseInstanceProxies())
                  if p.GetTypeName() == 'Mesh']
        if not ground:raise RuntimeError('soil model requires ground meshes')
        self.record = {}
        self.steps = 0;self.max_normal = 0.

        def material(path, friction):
            mat = UsdShade.Material.Define(stage, path)
            api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
            api.CreateStaticFrictionAttr(friction);api.CreateDynamicFrictionAttr(friction);api.CreateRestitutionAttr(0.)
            PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim()).CreateFrictionCombineModeAttr('max')
            return mat, api

        floor, _ = material('/World/Physics/SoilGround', .001)
        for prim in ground:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(floor, UsdShade.Tokens.strongerThanDescendants, 'physics')
        prims = list(Usd.PrimRange(stage.GetPrimAtPath('/World/Rover'), Usd.TraverseInstanceProxies()))
        for index, module in enumerate(session.geometry.modules):
            joints = [p for p in prims if p.GetName() == module.wheel_joint and p.IsA(UsdPhysics.Joint)]
            if len(joints) != 1:raise RuntimeError(f'wheel joint is ambiguous: {module.wheel_joint}')
            body = str(UsdPhysics.Joint(joints[0]).GetBody1Rel().GetTargets()[0])
            view = RigidPrim(prim_paths_expr=body, name=f'soil_wheel_{index}', reset_xform_properties=False,
                             track_contact_forces=True, contact_filter_prim_paths_expr=[str(p.GetPath()) for p in ground],
                             max_contact_count=256)
            self.views.append(view)
            mat, api = material(f'/World/Physics/SoilWheel{index}', mu_max*self.state.parameters.mu_lat)
            colliders = [p for p in Usd.PrimRange(stage.GetPrimAtPath(body), Usd.TraverseInstanceProxies())
                         if p.HasAPI(UsdPhysics.CollisionAPI)]
            if not colliders:raise RuntimeError(f'no collider below {body}')
            # imported collider mesh는 instance proxy다. 편집 가능한 wheel body에 강한 상속 바인딩을 준다.
            UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(body)).Bind(mat, UsdShade.Tokens.strongerThanDescendants, 'physics')
            self.materials.append(api)

    def initialize(self) -> None:
        for view in self.views:view.initialize()

    def apply(self) -> None:
        s = self.session
        q = np.asarray(self.robot.get_joint_positions())[s.steering_indices]
        w = np.asarray(self.robot.get_joint_velocities())[s.wheel_indices]
        converted = [from_joint(m, float(a), float(b)) for m,a,b in zip(s.geometry.modules,q,w)]
        angle, speed = np.asarray(converted).T
        _, quat = self.robot.get_world_pose()
        rot = Rotation.from_quat([quat[1],quat[2],quat[3],quat[0]])
        direction = rot.apply(np.column_stack((np.cos(angle),np.sin(angle),np.zeros(4))))
        direction[:,2] = 0.
        direction /= np.maximum(np.linalg.norm(direction,axis=1,keepdims=True), 1e-12)
        normal, velocity = np.zeros(4), np.zeros(4)
        net = []
        centers = []
        for k, view in enumerate(self.views):
            forces, _, _, _, counts, starts = view.get_contact_force_data(dt=.005)
            forces, counts, starts = np.asarray(forces).reshape(-1), np.asarray(counts), np.asarray(starts)
            for count, start in zip(counts.ravel(), starts.ravel()):
                normal[k] += np.abs(forces[int(start):int(start)+int(count)]).sum()
            net.append(np.asarray(view.get_net_contact_forces(dt=.005))[0].tolist())
            velocity[k] = np.asarray(view.get_linear_velocities())[0]@direction[k]
            p, orientation = view.get_world_poses();local, _ = view.get_coms()
            center = np.asarray(p)[0]+Rotation.from_quat(np.asarray(orientation)[0][[1,2,3,0]]).apply(np.asarray(local).reshape(-1,3)[0])
            centers.append(center)
        slip, mu, resistance = self.state.step(speed*s.geometry.wheel_radius_m, velocity, normal)
        self.steps += 1;self.max_normal = max(self.max_normal,float(normal.sum()))
        if self.steps >= 200 and self.max_normal <= 1e-6:
            raise RuntimeError(f'soil contact normal remains zero; net contact diagnostic={net}')
        magnitude = resistance*np.sign(velocity)*(abs(velocity)>1e-4)
        force = -magnitude[:,None]*direction
        for k, (view, api) in enumerate(zip(self.views, self.materials)):
            api.GetStaticFrictionAttr().Set(float(mu[k]));api.GetDynamicFrictionAttr().Set(float(mu[k]))
            view.apply_forces_and_torques_at_pos(forces=force[k:k+1], positions=np.asarray(centers[k:k+1]), is_global=True)
        self.record = dict(model='legacy-slip-sinkage', mu_max=self.state.mu_max, slip=slip.tolist(),
                           parameters=asdict(self.state.parameters),
                           sinkage_m=self.state.z.tolist(), normal_force_n=normal.tolist(),
                           friction=mu.tolist(), resistance_n=resistance.tolist(),
                           net_contact_force_n=net,
                           resistance_power_w=float(np.sum(-magnitude*velocity)))
