# ================================================================
# visuals.py
# 설명: portable URDF의 실제 STL 형상을 importer의 시각 payload와 분리해 관찰용으로 그린다.
# 사용법: from experiments.site_sim.visuals import install_rover_visuals
# ================================================================
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np


def read_binary_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """미터 단위 binary STL을 중복 정점 없는 USD 삼각형 배열로 읽는다."""
    with Path(path).open('rb') as stream:
        header=stream.read(84)
        if len(header)!=84:raise ValueError(f'invalid binary STL: {path}')
        count=int.from_bytes(header[80:84],'little')
        dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attr','<u2')])
        data=np.fromfile(stream,dtype=dtype,count=count)
    if len(data)!=count or not count:raise ValueError(f'truncated/empty STL: {path}')
    points,indices=np.unique(data['vertices'].reshape(-1,3),axis=0,return_inverse=True)
    if not np.isfinite(points).all():raise ValueError(f'nonfinite STL: {path}')
    return points,indices.reshape(-1,3)


def install_rover_visuals(stage: object, urdf: Path) -> None:
    """실제 articulation link에 STL 시각 mesh를 부착하며 충돌·질량은 건드리지 않는다."""
    from pxr import Gf, Usd, UsdGeom
    from experiments.orbslam3_ros2.isaac_scene import _find_unique_descendant
    root=ET.parse(urdf).getroot()
    # Hide only imported geometry instances; retain their collision API and physical links.
    for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/Rover')):
        if prim.IsInstance():UsdGeom.Imageable(prim).CreateVisibilityAttr('invisible')
    for link in root.findall('link'):
        if not link.findall('visual'):continue
        parent=_find_unique_descendant(stage,'/World/Rover',link.get('name'))
        for i,visual in enumerate(link.findall('visual')):
            source=visual.find('geometry/mesh')
            if source is None:raise ValueError('portable rover visual must be a mesh')
            points,faces=read_binary_stl(urdf.parent/source.get('filename'))
            mesh=UsdGeom.Mesh.Define(stage,str(parent.GetPath())+f'/ObserverVisual_{i}')
            mesh.CreatePointsAttr(points);mesh.CreateFaceVertexCountsAttr(np.full(len(faces),3,dtype=np.int32))
            mesh.CreateFaceVertexIndicesAttr(faces.ravel());mesh.CreateSubdivisionSchemeAttr('none')
            mesh.CreateDoubleSidedAttr(True)
            mesh.CreateExtentAttr([Gf.Vec3f(*points.min(0).tolist()),Gf.Vec3f(*points.max(0).tolist())])
            color=visual.find('material/color')
            rgb=[float(v) for v in color.get('rgba').split()[:3]] if color is not None else [.6,.6,.6]
            mesh.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
            origin=visual.find('origin');xf=UsdGeom.Xformable(mesh)
            if origin is not None:
                xyz=[float(v) for v in origin.get('xyz','0 0 0').split()]
                rpy=np.degrees([float(v) for v in origin.get('rpy','0 0 0').split()])
                xf.AddTranslateOp().Set(Gf.Vec3d(*xyz));xf.AddRotateXYZOp().Set(Gf.Vec3f(*rpy.tolist()))
            scale=[float(v) for v in source.get('scale','1 1 1').split()]
            xf.AddScaleOp().Set(Gf.Vec3f(*scale))
