# ================================================================
# rover_assets.py
# 설명: 최신 rocker_bogie_4wd_rover URDF를 Isaac 가져오기용 portable 자산(미터 단위 mesh,
#       상대 경로, 색상, 기존 frame 별칭)과 FK 기반 swerve_geometry.json으로 변환한다.
# 사용법:
#   python -m experiments.site_sim.rover_assets --output outputs/site_sim/rover_assets_new
# ================================================================
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

SOURCE_URDF=Path(__file__).resolve().parents[2]/'ros2_ws/src/rover_description/urdf/rocker_bogie_4wd_rover.urdf'
MESH_DIR=Path('/home/su/rocker_bogie_4wd_rover/meshes')
# 차체는 모래색 바닥과 구분되는 주황, 나머지는 어두운 색으로 칠한다.
COLORS={'base_link':(.95,.4,.05,1.),'camera_link':(.1,.1,.1,1.),'imu_link':(.1,.1,.1,1.)}
DEFAULT_COLOR=(.12,.14,.2,1.)
ALIASES=(('camera_infra1_optical_frame','camera_left_optical_frame'),
         ('camera_infra2_optical_frame','camera_right_optical_frame'),('imu_sensor_frame','imu_link'))


def _vec(text: str | None, default: str='0 0 0') -> np.ndarray:
    return np.array([float(v) for v in (text or default).split()])


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    r,p,y=rpy
    rx=np.array([[1,0,0],[0,math.cos(r),-math.sin(r)],[0,math.sin(r),math.cos(r)]])
    ry=np.array([[math.cos(p),0,math.sin(p)],[0,1,0],[-math.sin(p),0,math.cos(p)]])
    rz=np.array([[math.cos(y),-math.sin(y),0],[math.sin(y),math.cos(y),0],[0,0,1]])
    return rz@ry@rx


def _transform(joint: ET.Element) -> np.ndarray:
    origin=joint.find('origin');T=np.eye(4)
    if origin is not None:
        T[:3,:3]=_rpy_matrix(_vec(origin.get('rpy')));T[:3,3]=_vec(origin.get('xyz'))
    return T


def zero_pose_kinematics(root: ET.Element) -> tuple[dict, dict]:
    """모든 joint를 0으로 둔 base_link 기준 link 변환과 joint frame 변환을 구한다."""
    joints={j.get('name'):j for j in root.findall('joint')}
    by_child={j.find('child').get('link'):j for j in joints.values()}
    links={'base_link':np.eye(4)};frames={}
    def link_T(name: str) -> np.ndarray:
        if name not in links:
            j=by_child[name];frames[j.get('name')]=link_T(j.find('parent').get('link'))@_transform(j)
            links[name]=frames[j.get('name')]
        return links[name]
    for name in list(by_child):link_T(name)
    return links,frames


def apply_ksrc_signs(root: ET.Element) -> list[str]:
    """KSRC 규약으로 joint axis의 부호만 뒤집는다: 조향축은 base +z, 바퀴축은 base +y(+속도로 +x 구름). 위치·이름·mesh·한계(대칭으로 뒤집음)는 그대로이며 뒤집은 joint 이름을 돌려준다."""
    _,frames=zero_pose_kinematics(root)
    flipped=[]
    for joint in root.findall('joint'):
        name=joint.get('name');kind=joint.get('type');axis=joint.find('axis')
        if axis is None:continue
        direction=frames[name][:3,:3]@_vec(axis.get('xyz'))
        if kind=='revolute' and 'steering' in name:wrong=direction[2]<0
        elif kind=='continuous' and 'wheel' in name:wrong=direction[1]<0
        else:continue
        if not wrong:continue
        axis.set('xyz',' '.join(f'{-v:g}' for v in _vec(axis.get('xyz'))))
        limit=joint.find('limit')
        if kind=='revolute' and limit is not None and limit.get('lower') is not None:
            lower,upper=float(limit.get('lower')),float(limit.get('upper'))
            limit.set('lower',f'{-upper:.12g}');limit.set('upper',f'{-lower:.12g}')
        flipped.append(name)
    return flipped


def swerve_geometry(root: ET.Element, wheel_radius: float) -> dict:
    """steering joint의 child link에 붙은 wheel joint를 짝지어 위치·부호·기준 방향을 계산한다."""
    links,frames=zero_pose_kinematics(root)
    joints={j.get('name'):j for j in root.findall('joint')}
    modules=[]
    for name,joint in joints.items():
        if joint.get('type')!='revolute' or 'steering' not in name:continue
        leg=joint.find('child').get('link')
        wheel=[j for j in joints.values() if j.get('type')=='continuous' and j.find('parent').get('link')==leg]
        if len(wheel)!=1:raise ValueError(f'{name}: exactly one wheel joint required on {leg}')
        wheel=wheel[0];F=frames[name];axis=F[:3,:3]@_vec(joint.find('axis').get('xyz'))
        if abs(abs(axis[2])-1)>1e-6:raise ValueError(f'{name}: steering axis must be vertical')
        wa=frames[wheel.get('name')][:3,:3]@_vec(wheel.find('axis').get('xyz'))
        if abs(wa[2])>1e-6:raise ValueError(f'{wheel.get("name")}: wheel axis must be horizontal')
        # rolling without slip: v_center = omega * r * (axis x z_up)
        roll=np.cross(wa,[0,0,1.])
        x,y=F[0,3],F[1,3]
        modules.append({'name':('front' if x>0 else 'rear')+'_'+('left' if y>0 else 'right'),
            'steering_joint':name,'wheel_joint':wheel.get('name'),
            'steering_pivot_base_link_m':[x,y,F[2,3]],'steering_sign':float(np.sign(axis[2])),
            'rolling_heading_at_zero_rad':math.atan2(roll[1],roll[0]),'wheel_axis_base_link_zero_pose':wa.tolist()})
    if len(modules)!=4 or len({m['name'] for m in modules})!=4:raise ValueError('four distinct swerve modules required')
    return {'frame':'base_link','axes':'X forward, Y left, Z up','units':'metres; radians; seconds',
            'source':str(SOURCE_URDF.name),'wheel_radius_m':wheel_radius,
            'steering_convention':'q = steering_sign * (heading - rolling_heading_at_zero_rad), folded to +-90 deg',
            'modules':sorted(modules,key=lambda m:m['name'])}


def make_portable_urdf(root: ET.Element, output: Path, mesh_dir: Path) -> ET.Element:
    """mesh를 미터 단위 STL로 복사하고 상대 경로·색상·별칭 frame을 가진 URDF를 만든다."""
    import open3d as o3d
    portable=copy.deepcopy(root);portable.set('name','rover')  # importer-generated prim name expected by the scene code
    (output/'meshes').mkdir(parents=True,exist_ok=True)
    for link in portable.findall('link'):
        for tag in ('visual','collision'):
            for item in link.findall(tag):
                mesh=item.find('geometry/mesh')
                if mesh is None:continue
                name=Path(mesh.get('filename')).name;scale=_vec(mesh.get('scale'),'1 1 1')
                target=output/'meshes'/name
                if not target.exists():
                    m=o3d.io.read_triangle_mesh(str(mesh_dir/name))
                    if m.is_empty():raise ValueError(f'cannot read mesh {name}')
                    m.vertices=o3d.utility.Vector3dVector(np.asarray(m.vertices)*scale)
                    m.compute_triangle_normals();o3d.io.write_triangle_mesh(str(target),m)
                mesh.set('filename','meshes/'+name);mesh.attrib.pop('scale',None)
                if tag=='collision':item.set('name',link.get('name')+'_collision_0')  # keeps importer prim names unique
                if tag=='visual':
                    for old in item.findall('material'):item.remove(old)
                    material=ET.SubElement(item,'material',name=link.get('name')+'_material')
                    ET.SubElement(material,'color',rgba=' '.join(map(str,COLORS.get(link.get('name'),DEFAULT_COLOR))))
    for alias,source in ALIASES:
        ET.SubElement(portable,'link',name=alias)
        joint=ET.SubElement(portable,'joint',name='alias_'+alias,type='fixed')
        ET.SubElement(joint,'parent',link=source);ET.SubElement(joint,'child',link=alias)
        ET.SubElement(joint,'origin',xyz='0 0 0',rpy='0 0 0')
    return portable


def wheel_radius(output: Path, root: ET.Element) -> float:
    """바퀴 mesh의 높이·폭 중 큰 지름 방향 반치수로 반지름을 추정한다."""
    import open3d as o3d
    mesh=o3d.io.read_triangle_mesh(str(output/'meshes/wheel_1.stl'))
    ext=np.asarray(mesh.get_axis_aligned_bounding_box().get_extent())
    return float(np.sort(ext)[-1]/2)


def build(output: Path, source: Path=SOURCE_URDF, mesh_dir: Path=MESH_DIR, ksrc_signs: bool=False) -> dict:
    output=Path(output).resolve()
    root=ET.parse(source).getroot()
    if ksrc_signs:apply_ksrc_signs(root)   # RL·펌웨어와 같은 부호 규약(steering_sign +1, rolling_heading 0)
    portable=make_portable_urdf(root,output,mesh_dir)
    ET.indent(portable);ET.ElementTree(portable).write(output/'rover_portable.urdf',encoding='utf-8',xml_declaration=True)
    geometry=swerve_geometry(root,wheel_radius(output,root))
    (output/'config').mkdir(exist_ok=True)
    (output/'config/swerve_geometry.json').write_text(json.dumps(geometry,indent=2),encoding='utf-8')
    return geometry


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--ksrc-signs',action='store_true',help='joint axis 부호만 KSRC 규약으로 뒤집는다(조향 +z, 바퀴 +y)')
    args=parser.parse_args();g=build(args.output,ksrc_signs=args.ksrc_signs)
    for m in g['modules']:print(m['name'],m['steering_joint'],m['wheel_joint'],np.round(m['steering_pivot_base_link_m'],4),
                                m['steering_sign'],round(math.degrees(m['rolling_heading_at_zero_rad']),1))
    print('wheel_radius_m',g['wheel_radius_m'])


if __name__=='__main__':main()
