"""Generate URDF and a native MuJoCo model from the measured configuration."""
from pathlib import Path
import json, math, sys
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]

def nums(v):
    return ' '.join(format(float(x), '.12g') for x in v)

def inertia_attributes(I):
    return dict(zip(['ixx','iyy','izz','ixy','ixz','iyz'],map(str,[I[0][0],I[1][1],I[2][2],I[0][1],I[0][2],I[1][2]])))

def generate_urdf(config, portable):
    robot = ET.Element('robot',name=config['robot_name'])
    robot.append(ET.Comment(' Measured masses; approximate inertia, collision, limits and motor ratings. See config/model.json. '))
    for s in config['links']:
        link = ET.SubElement(robot,'link',name=s['name'])
        if 'mass_kg' not in s:
            continue
        inertial=ET.SubElement(link,'inertial')
        ET.SubElement(inertial,'origin',xyz=nums(s['com_m']),rpy='0 0 0')
        ET.SubElement(inertial,'mass',value=str(s['mass_kg']))
        ET.SubElement(inertial,'inertia',**inertia_attributes(s['inertia_kg_m2']))
        visual=ET.SubElement(link,'visual')
        ET.SubElement(visual,'origin',xyz='0 0 0',rpy='0 0 0')
        geo=ET.SubElement(visual,'geometry')
        uri=s['mesh'] if portable else 'package://rover_description/'+s['mesh']
        ET.SubElement(geo,'mesh',filename=uri)
        material=ET.SubElement(visual,'material',name=s['name']+'_material')
        ET.SubElement(material,'color',rgba=nums(s['color']))
        for i,col in enumerate(s['collisions']):
            collision=ET.SubElement(link,'collision',name=s['name']+'_collision_'+str(i))
            ET.SubElement(collision,'origin',xyz=nums(col['xyz']),rpy='0 0 0')
            geometry=ET.SubElement(collision,'geometry')
            if col['type']=='box':
                ET.SubElement(geometry,'box',size=nums(col['size']))
            else:
                ET.SubElement(geometry,'cylinder',radius=str(col['radius']),length=str(col['length']))
    for s in config['joints']:
        j=ET.SubElement(robot,'joint',name=s['name'],type=s['type'])
        ET.SubElement(j,'parent',link=s['parent'])
        ET.SubElement(j,'child',link=s['child'])
        ET.SubElement(j,'origin',xyz=nums(s['xyz']),rpy=nums(s['rpy']))
        if s['type']=='fixed':
            continue
        ET.SubElement(j,'axis',xyz=nums(s['axis']))
        lim={'effort':str(s['effort']),'velocity':str(s['velocity'])}
        if s['type']=='revolute':
            lim.update(lower=str(s['lower']),upper=str(s['upper']))
        ET.SubElement(j,'limit',**lim)
        ET.SubElement(j,'dynamics',damping=str(s['damping']),friction='0')
        if 'mimic' in s:
            ET.SubElement(j,'mimic',**{k:str(v) for k,v in s['mimic'].items()})
    ET.indent(robot,space='  ')
    target=ROOT/'rover_portable.urdf' if portable else ROOT/'urdf/rover.urdf'
    ET.ElementTree(robot).write(target,encoding='utf-8',xml_declaration=True)

def mujoco_quat(rpy):
    q=Rotation.from_euler('xyz',rpy).as_quat()
    return nums([q[3],q[0],q[1],q[2]])

def update_zero_pose_reference(config):
    """Recompute mass properties from current link poses, including sensor tilt."""
    ref_path = ROOT/'config/cad_reference.json'
    refs = json.loads(ref_path.read_text())
    frames = {'base_link': np.eye(4)}
    pending = list(config['joints'])
    while pending:
        before = len(pending)
        for j in pending[:]:
            if j['parent'] not in frames:
                continue
            local = np.eye(4)
            local[:3, :3] = Rotation.from_euler('xyz', j['rpy']).as_matrix()
            local[:3, 3] = j['xyz']
            frames[j['child']] = frames[j['parent']] @ local
            pending.remove(j)
        if len(pending) == before:
            raise ValueError('Joint tree is disconnected or cyclic')
    physical = []
    bounds = []
    for link in config['links']:
        if 'mass_kg' not in link:
            continue
        T = frames[link['name']]
        center = T[:3, :3] @ np.array(link['com_m']) + T[:3, 3]
        inertia = T[:3, :3] @ np.array(link['inertia_kg_m2']) @ T[:3, :3].T
        physical.append((link['mass_kg'], center, inertia))
        mesh = trimesh.load_mesh(ROOT/link['mesh'], process=False)
        mesh.apply_transform(T)
        bounds.append(mesh.bounds)
    mass = sum(p[0] for p in physical)
    com = sum(m*c for m,c,_ in physical)/mass
    aggregate = np.zeros((3,3))
    for m,c,I in physical:
        delta = c-com
        aggregate += I + m*(np.dot(delta,delta)*np.eye(3)-np.outer(delta,delta))
    bounds = np.array(bounds)
    limits = np.array([bounds[:,0].min(axis=0), bounds[:,1].max(axis=0)])
    refs['global_link_frames_m'] = {name:T.tolist() for name,T in frames.items()}
    refs['total_mass_kg'] = mass
    refs['zero_pose_com_ros_m'] = com.tolist()
    refs['zero_pose_com_cad_mm'] = (np.array(refs['CAD_to_ROS_rotation']).T @ (com*1000) + refs['base_origin_cad_mm']).tolist()
    refs['zero_pose_aggregate_inertia_ros_kg_m2'] = aggregate.tolist()
    refs['zero_pose_visual_bounds_ros_m'] = limits.tolist()
    refs['zero_pose_visual_envelope_ros_xyz_mm'] = ((limits[1]-limits[0])*1000).tolist()
    refs['mass_properties_method'] = 'Measured masses; all 28 available CAD/sensor parts use prepared STL volume COM/inertia with uniform effective density inside each part. Repairs and shape proxies are recorded. Three omitted gears retain lumped inertia. Current joint transforms and parallel-axis aggregation; actual internal density/infill distribution is not modeled.'
    ref_path.write_text(json.dumps(refs, indent=2))

def generate_mjcf(config):
    root=ET.Element('mujoco',model='rover_measured')
    ET.SubElement(root,'compiler',angle='radian',meshdir='../meshes',autolimits='true',inertiafromgeom='false')
    ET.SubElement(root,'option',timestep='0.002',gravity='0 0 -9.81',integrator='implicitfast',solver='Newton',iterations='100')
    ET.SubElement(root,'visual')
    default=ET.SubElement(root,'default')
    ET.SubElement(default,'geom',friction='1.0 0.005 0.0001',solref='0.01 1',solimp='0.95 0.99 0.001')
    assets=ET.SubElement(root,'asset')
    for s in config['links']:
        if 'mesh' in s:
            filename=Path(s['mesh']).name
            source=trimesh.load_mesh(ROOT/s['mesh'],process=False)
            if len(source.faces)>190000:
                # MuJoCo's binary STL decoder accepts at most 200,000 faces.
                # Keep full CAD STL for URDF; simplify only the MJCF visual.
                import vtk
                from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
                points=vtk.vtkPoints()
                points.SetData(numpy_to_vtk(np.asarray(source.vertices),deep=True))
                cells=vtk.vtkCellArray()
                cells.SetCells(len(source.faces),numpy_to_vtkIdTypeArray(np.c_[np.full(len(source.faces),3),source.faces].astype(np.int64).ravel(),deep=True))
                data=vtk.vtkPolyData()
                data.SetPoints(points)
                data.SetPolys(cells)
                dec=vtk.vtkQuadricDecimation()
                dec.SetInputData(data)
                dec.SetTargetReduction(1-100000/len(source.faces))
                dec.Update()
                output=dec.GetOutput()
                visual=trimesh.Trimesh(vertices=vtk_to_numpy(output.GetPoints().GetData()),faces=vtk_to_numpy(output.GetPolys().GetData()).reshape(-1,4)[:,1:],process=False)
                filename=s['name']+'_sim.stl'
                visual.export(ROOT/'meshes'/filename)
            ET.SubElement(assets,'mesh',name=s['name']+'_mesh',file=filename)
    world=ET.SubElement(root,'worldbody')
    ET.SubElement(world,'light',pos='0 -1 2',dir='0 0 -1')
    ET.SubElement(world,'geom',name='ground',type='plane',size='5 5 .1',rgba='.75 .72 .65 1',contype='1',conaffinity='2')
    base=ET.SubElement(world,'body',name='base_link',pos='0 0 0.161813')
    ET.SubElement(base,'freejoint',name='floating_base')
    bodies={'base_link':base}
    link_specs={s['name']:s for s in config['links']}
    for j in config['joints']:
        body=ET.SubElement(bodies[j['parent']],'body',name=j['child'],pos=nums(j['xyz']),quat=mujoco_quat(j['rpy']))
        bodies[j['child']]=body
        if j['type']=='fixed':
            continue
        attrs={'name':j['name'],'type':'hinge','axis':nums(j['axis']),'damping':str(j['damping'])}
        if j['type']=='revolute':
            attrs['range']=nums([j['lower'],j['upper']])
        ET.SubElement(body,'joint',**attrs)
    for name,body in bodies.items():
        s=link_specs[name]
        if 'mass_kg' not in s:
            continue
        I=s['inertia_kg_m2']
        full=[I[0][0],I[1][1],I[2][2],I[0][1],I[0][2],I[1][2]]
        ET.SubElement(body,'inertial',mass=str(s['mass_kg']),pos=nums(s['com_m']),fullinertia=nums(full))
        ET.SubElement(body,'geom',name=name+'_visual',type='mesh',mesh=name+'_mesh',rgba=nums(s['color']),contype='0',conaffinity='0',group='2')
        for i,c in enumerate(s['collisions']):
            attrs={'name':name+'_collision_'+str(i),'type':c['type'],'pos':nums(c['xyz']),
                   'contype':'2','conaffinity':'1','rgba':'.3 .5 .8 .2','group':'3'}
            attrs['size']=nums(np.array(c['size'])/2) if c['type']=='box' else nums([c['radius'],c['length']/2])
            ET.SubElement(body,'geom',**attrs)
    equality=ET.SubElement(root,'equality')
    ET.SubElement(equality,'joint',name='rocker_differential',joint1='right_rocker_joint',joint2='left_rocker_joint',polycoef='0 -1 0 0 0',solref='0.002 1')
    actuator=ET.SubElement(root,'actuator')
    for j in config['joints']:
        if not j.get('actuated'):
            continue
        limits=nums([-j['effort'],j['effort']])
        if j['name'].endswith('_wheel_joint'):
            ET.SubElement(actuator,'velocity',name=j['name']+'_actuator',joint=j['name'],kv='.2',ctrlrange=nums([-j['velocity'],j['velocity']]),forcerange=limits)
        else:
            ET.SubElement(actuator,'position',name=j['name']+'_actuator',joint=j['name'],kp='10',kv='.3',ctrlrange=nums([j['lower'],j['upper']]),forcerange=limits)
    for name in ['camera_infra1_optical_frame','camera_infra2_optical_frame']:
        # MuJoCo cameras look along local -Z with local +Y up; ROS optical
        # frames look along +Z with local +Y down, hence Rx(pi).
        ET.SubElement(bodies[name],'camera',name=name.replace('_optical_frame',''),pos='0 0 0',quat='0 1 0 0',fovy='58')
    ET.SubElement(bodies['imu_sensor_frame'],'site',name='imu_site',pos='0 0 0',size='.001',rgba='1 0 0 1')
    sensors=ET.SubElement(root,'sensor')
    ET.SubElement(sensors,'accelerometer',name='imu_accelerometer',site='imu_site')
    ET.SubElement(sensors,'gyro',name='imu_gyro',site='imu_site')
    ET.SubElement(sensors,'framequat',name='base_orientation',objtype='body',objname='base_link')
    ET.indent(root,space='  ')
    ET.ElementTree(root).write(ROOT/'mjcf/rover.xml',encoding='utf-8',xml_declaration=True)

def main():
    config=json.loads((ROOT/'config/model.json').read_text())
    generate_urdf(config,False)
    generate_urdf(config,True)
    generate_mjcf(config)
    update_zero_pose_reference(config)
    print('Generated ROS URDF, portable URDF, and MuJoCo MJCF.')

if __name__=='__main__':
    main()
