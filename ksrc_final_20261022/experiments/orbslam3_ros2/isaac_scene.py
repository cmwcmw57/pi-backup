# ================================================================
# isaac_scene.py
# 설명: settled-sand USD와 portable rover URDF를 불러와 stereo camera와 IMU prim을 만든다.
# 사용법:
#   from experiments.orbslam3_ros2.isaac_scene import build_isaac_scene
# ================================================================

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from experiments.orbslam3_offline.calibration import load_frame_transform


@dataclass(frozen=True)
class SceneHandles:
    """생성된 rover stage의 frame, camera, IMU 및 import 결과 경로를 보관한다."""

    frame_paths: dict[str, str]
    left_camera_path: str
    right_camera_path: str
    imu_path: str
    imported_usd: str
    frame_transform_max_errors: dict[str, float]


def select_physics_variant(rover_prim: object) -> None:
    """URDF importer가 만든 PhysX payload를 rover prim에 활성화한다."""
    variant_set = rover_prim.GetVariantSets().GetVariantSet("Physics")
    if "physx" not in variant_set.GetVariantNames():
        raise RuntimeError("physx variant is unavailable on imported rover")
    if not variant_set.SetVariantSelection("physx"):
        raise RuntimeError("failed to select imported rover physx variant")


def enable_static_mesh_collisions(
    stage: object,
    root_path: str,
    *,
    collision_api: object | None = None,
) -> tuple[str, ...]:
    """시각화용 terrain reference 아래 모든 Mesh를 정적 collider로 활성화한다."""
    if collision_api is None:
        from pxr import UsdPhysics

        collision_api = UsdPhysics.CollisionAPI

    pending = list(stage.GetPrimAtPath(root_path).GetChildren())
    meshes: list[object] = []
    while pending:
        prim = pending.pop()
        pending.extend(prim.GetChildren())
        if prim.GetTypeName() == "Mesh":
            meshes.append(prim)
    if not meshes:
        raise RuntimeError(f"static terrain has no mesh below {root_path}")

    meshes.sort(key=lambda prim: str(prim.GetPath()))
    for prim in meshes:
        collision_api.Apply(prim)
    return tuple(str(prim.GetPath()) for prim in meshes)


def _validate_portable_urdf(urdf: Path) -> None:
    """URDF mesh가 존재하고 중복 scale 보정이 없는지 확인한다."""
    for mesh in ElementTree.parse(urdf).findall(".//mesh"):
        mesh_path = urdf.parent / mesh.attrib["filename"]
        if not mesh_path.is_file():
            raise ValueError(f"missing URDF mesh: {mesh.attrib['filename']}")
        scale = [float(value) for value in mesh.get("scale", "1 1 1").split()]
        if not np.allclose(scale, 1.0):
            raise ValueError("portable URDF mesh scale must be 1")


def _find_unique_descendant(stage: object, root_path: str, name: str) -> object:
    """rover root 아래에서 이름이 정확히 일치하는 prim 하나를 찾는다."""
    from pxr import Usd

    root = stage.GetPrimAtPath(root_path)
    matches = [
        prim
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
        if (
            prim.GetName() == name
            and not prim.IsInstance()
            and prim.GetParent().GetName() != name
        )
    ]
    # Mesh collision wrappers can contain a same-named descendant; the frame is the outermost match.
    paths = [str(prim.GetPath()) for prim in matches]
    matches = [
        prim for prim, path in zip(matches, paths)
        if not any(path.startswith(other + "/") for other in paths)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"imported frame must be unique: {name}, found {len(matches)}: "
            + ", ".join(str(prim.GetPath()) for prim in matches)
        )
    return matches[0]


def _column_world_matrix(prim: object) -> np.ndarray:
    """USD row-major world transform을 column-vector convention으로 반환한다."""
    from pxr import UsdGeom

    return np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).T


def _create_camera(stage: object, path: str, tick_rate_hz: float) -> None:
    """ROS optical frame 아래에 fx=fy=480인 USD camera를 만든다."""
    from pxr import Gf, Sdf, UsdGeom

    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateProjectionAttr("perspective")
    camera.CreateFocalLengthAttr(15.0)
    camera.CreateHorizontalApertureAttr(20.0)
    camera.CreateVerticalApertureAttr(15.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    camera.GetPrim().CreateAttribute(
        "omni:sensor:tickRate", Sdf.ValueTypeNames.Float
    ).Set(float(tick_rate_hz))
    xform = UsdGeom.Xformable(camera.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(0.0, 1.0, 0.0, 0.0))


def _create_imu(stage: object, frame_prim: object) -> str:
    """IMU frame pose를 가장 가까운 rigid-body parent 아래 sensor prim으로 옮긴다."""
    from isaacsim.sensors.experimental.physics import IMU
    from pxr import Gf, UsdGeom, UsdPhysics

    rigid_prim = frame_prim
    while rigid_prim and rigid_prim.IsValid() and not rigid_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        rigid_prim = rigid_prim.GetParent()
    if not rigid_prim or not rigid_prim.IsValid():
        raise RuntimeError("imu_sensor_frame has no rigid-body ancestor")

    frame_world = UsdGeom.Xformable(frame_prim).ComputeLocalToWorldTransform(0)
    rigid_world = UsdGeom.Xformable(rigid_prim).ComputeLocalToWorldTransform(0)
    local_transform = Gf.Transform(frame_world * rigid_world.GetInverse())
    translation = local_transform.GetTranslation()
    quaternion = local_transform.GetRotation().GetQuat()
    imaginary = quaternion.GetImaginary()
    imu_path = f"{rigid_prim.GetPath()}/ORB_IMU"
    IMU.create(
        imu_path,
        translations=[[translation[0], translation[1], translation[2]]],
        orientations=[[
            quaternion.GetReal(),
            imaginary[0],
            imaginary[1],
            imaginary[2],
        ]],
        linear_acceleration_filter_size=1,
        angular_velocity_filter_size=1,
        orientation_filter_size=1,
    )
    if not stage.GetPrimAtPath(imu_path).IsValid():
        raise RuntimeError(f"failed to create IMU prim: {imu_path}")
    return imu_path


def build_isaac_scene(
    *,
    urdf: Path,
    scene_usd: Path,
    output: Path,
    camera_hz: float = 30.0,
    start_position: tuple[float, float, float] = (3.5, -0.4, 0.42),
) -> tuple[object, SceneHandles]:
    """settled-sand stage, rover articulation, sensor prim을 만들고 World를 반환한다."""
    import omni.usd
    from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
    from isaacsim.core.api import World
    from pxr import Gf, UsdGeom, UsdLux

    urdf = Path(urdf).resolve(strict=True)
    scene_usd = Path(scene_usd).resolve(strict=True)
    output = Path(output)
    _validate_portable_urdf(urdf)

    imported_usd = URDFImporter(
        URDFImporterConfig(
            urdf_path=str(urdf),
            usd_path=str(output / "assets"),
            merge_fixed_joints=False,
            merge_mesh=False,
            collision_from_visuals=False,
            allow_self_collision=False,
            fix_base=False,
        )
    ).import_urdf()

    omni.usd.get_context().new_stage()
    world = World(physics_dt=0.005, rendering_dt=0.005, stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    stage.DefinePrim("/World/SandScene", "Xform").GetReferences().AddReference(str(scene_usd))
    enable_static_mesh_collisions(stage, "/World/SandScene")
    rover = stage.DefinePrim("/World/Rover", "Xform")
    rover.GetReferences().AddReference(str(imported_usd))
    select_physics_variant(rover)
    UsdGeom.Xformable(rover).AddTranslateOp().Set(Gf.Vec3d(*start_position))
    UsdLux.DomeLight.Define(stage, "/World/Lighting").CreateIntensityAttr(900.0)

    frame_names = (
        "base_link",
        "camera_infra1_optical_frame",
        "camera_infra2_optical_frame",
        "imu_sensor_frame",
    )
    frame_prims = {
        name: _find_unique_descendant(stage, "/World/Rover", name)
        for name in frame_names
    }
    frame_paths = {name: str(prim.GetPath()) for name, prim in frame_prims.items()}

    base_world = _column_world_matrix(frame_prims["base_link"])
    errors: dict[str, float] = {}
    for name in frame_names[1:]:
        actual = np.linalg.inv(base_world) @ _column_world_matrix(frame_prims[name])
        expected = load_frame_transform(urdf, "base_link", name)
        errors[name] = float(np.max(np.abs(actual - expected)))
        if not np.allclose(actual, expected, atol=1e-5, rtol=0.0):
            raise RuntimeError(
                f"imported frame transform mismatch: {name}, error={errors[name]}"
            )

    left_camera_path = frame_paths["camera_infra1_optical_frame"] + "/Camera"
    right_camera_path = frame_paths["camera_infra2_optical_frame"] + "/Camera"
    _create_camera(stage, left_camera_path, camera_hz)
    _create_camera(stage, right_camera_path, camera_hz)
    imu_path = _create_imu(stage, frame_prims["imu_sensor_frame"])

    handles = SceneHandles(
        frame_paths=frame_paths,
        left_camera_path=left_camera_path,
        right_camera_path=right_camera_path,
        imu_path=imu_path,
        imported_usd=str(imported_usd),
        frame_transform_max_errors=errors,
    )
    return world, handles
