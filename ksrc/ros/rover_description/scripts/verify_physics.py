"""Basic rigid-ground MuJoCo verification; not a sand or hardware validation."""
from pathlib import Path
import json, sys, math
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

ROOT=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[1]

def joint_angle(m,d,name):
    j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name)
    return float(d.qpos[m.jnt_qposadr[j]])

def advance(m,d,seconds):
    residual=0.
    for _ in range(round(seconds/m.opt.timestep)):
        mujoco.mj_step(m,d)
        residual=max(residual,abs(joint_angle(m,d,'left_rocker_joint')+joint_angle(m,d,'right_rocker_joint')))
    assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
    assert not any(w.number for w in d.warning), 'MuJoCo numerical warning'
    return residual

def load(bump=False,slope=0.):
    path=(ROOT/'mjcf/rover.xml').resolve()
    xml=ET.parse(path).getroot()
    xml.find('compiler').set('meshdir',str((ROOT/'meshes').resolve()))
    if bump:
        ET.SubElement(xml.find('worldbody'),'geom',name='test_bump',type='box',pos='.10243 .135 .0075',size='.07 .045 .0075',contype='1',conaffinity='2',rgba='.5 .4 .3 1')
        xml.find("worldbody/body[@name='base_link']").set('pos','0 0 .182')
    if slope:
        R=Rotation.from_euler('y',-slope)
        q=R.as_quat();quat=' '.join(map(str,[q[3],q[0],q[1],q[2]]))
        xml.find("worldbody/geom[@name='ground']").set('quat',quat)
        base=xml.find("worldbody/body[@name='base_link']")
        base.set('quat',quat)
        base.set('pos',' '.join(map(str,R.apply([0,0,.164]))))
    m=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'))
    return m,mujoco.MjData(m)

def main():
    model=json.loads((ROOT/'config/model.json').read_text())
    refs=json.loads((ROOT/'config/cad_reference.json').read_text())
    report={'status':'passed','mujoco_version':mujoco.__version__,
            'total_mass_kg':refs['total_mass_kg'],
            'limitations':['Rigid ground; no deformable sand or calibrated soil slip.',
                           'Robot self-collision disabled in MJCF because collisions are primitive approximations.',
                           'Actuator limits and gains are initial estimates; tests do not establish hardware performance.',
                           'Camera frames and ideal simulated cameras are present; stereo image matching and real depth error are not validated.',
                           'IMU site is at STL origin, not a measured sensing-die position.',
                           'Gear masses are lumped into base/rockers with approximate inertia.']}
    m,d=load();res=advance(m,d,2.)
    assert m.nq==17 and m.nv==16 and m.nu==8 and m.neq==1
    assert abs(sum(m.body_mass)-2.3263)<1e-8
    loads={c:0. for c in ['fl','fr','rl','rr']}
    for i,con in enumerate(d.contact):
        a=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,con.geom1) or ''
        b=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,con.geom2) or ''
        force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force)
        for c in loads:
            if c+'_wheel_link_collision' in a or c+'_wheel_link_collision' in b:
                loads[c]+=float(force[0])
    assert abs(sum(loads.values())-2.3263*9.81)<.2
    start=d.qpos[:3].copy()
    for c in ['fl','fr','rl','rr']:
        aid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,c+'_wheel_joint_actuator')
        d.ctrl[aid]=2.
    res=max(res,advance(m,d,1.5))
    displacement=(d.qpos[:3]-start).tolist()
    assert displacement[0]>.05 and abs(displacement[1])<.01
    report['flat_ground']={'settle_s':2.,'wheel_normal_loads_N':loads,'drive_s':1.5,
                          'wheel_velocity_command_rad_s':2.,'displacement_m':displacement,
                          'maximum_rocker_sum_error_rad':res,'numerical_warnings':0}
    m,d=load();m.opt.gravity[:]=0.;d.qpos[2]=1.
    lid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'left_rocker_link')
    d.xfrc_applied[lid,4]=.02
    res=advance(m,d,.3)
    left=joint_angle(m,d,'left_rocker_joint');right=joint_angle(m,d,'right_rocker_joint')
    assert left>.005 and right<-.005 and res<.001
    report['passive_coupling']={'left_torque_Nm':.02,'left_rad':left,'right_rad':right,
                               'maximum_rocker_sum_error_rad':res,'actuator_on_rockers':False}
    m,d=load(bump=True);res=advance(m,d,2.5)
    left=joint_angle(m,d,'left_rocker_joint');right=joint_angle(m,d,'right_rocker_joint')
    assert abs(left)>.005 and res<.005
    report['asymmetric_15mm_step']={'left_rad':left,'right_rad':right,
                                  'maximum_rocker_sum_error_rad':res,'base_z_m':float(d.qpos[2])}
    slope=math.radians(30)
    m,d=load(slope=slope);res=advance(m,d,2.)
    start=d.qpos[:3].copy()
    for c in ['fl','fr','rl','rr']:
        d.ctrl[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,c+'_wheel_joint_actuator')]=2.
    res=max(res,advance(m,d,1.5))
    displacement=(d.qpos[:3]-start)
    up=np.array([math.cos(slope),0,math.sin(slope)])
    along=float(displacement@up)
    assert along>.025
    report['rigid_30deg_slope']={'friction':1.,'uphill_displacement_m':along,
                                'maximum_rocker_sum_error_rad':res,
                                'meaning':'Smoke test under assumed rigid-ground friction and actuator settings; not validation of 30 degree sand ascent.'}
    com=np.array(refs['zero_pose_com_cad_mm'])*.001
    # Root ROS global origin has the same ground Z as CAD.
    height=float(com[2]);forward=float(com[1]);lateral=float(-com[0])
    contacts=[];radii=[]
    for s in model['links']:
        if not s['name'].endswith('_wheel_link'):continue
        T=np.array(refs['global_link_frames_m'][s['name']])
        col=s['collisions'][0]
        contacts.append(T[:3,:3]@np.array(col['xyz'])+T[:3,3])
        radii.append(col['radius'])
    points=np.array(contacts)
    fmargin=float(points[:,0].max()-forward);rmargin=float(forward-points[:,0].min())
    side_margin=float(min(points[:,1].max()-lateral,lateral-points[:,1].min()))
    report['quasistatic_estimates']={
        'zero_pose_com_cad_mm':refs['zero_pose_com_cad_mm'],
        'flat_support_polygon_contact_centers_ros_m':points[:,:2].tolist(),
        'forward_tip_deg':math.degrees(math.atan2(fmargin,height)),
        'backward_tip_deg':math.degrees(math.atan2(rmargin,height)),
        'lateral_tip_deg':math.degrees(math.atan2(side_margin,height)),
        'wheel_radius_m':float(np.mean(radii)),
        'per_wheel_gravity_only_torque_30deg_Nm':2.3263*9.81*float(np.mean(radii))*.5/4,
        'caveat':'Level rocker posture, rigid ground, contact-center support polygon, approximate component COM; no acceleration, sinkage or soil rolling resistance.'}
    dims=refs['zero_pose_visual_envelope_ros_xyz_mm']
    report['size_check']={'length_mm':dims[0],'width_mm':dims[1],'height_mm':dims[2],
                         'limit_mm':[300,300,200],
                         'within_limit':dims[0]<=300 and dims[1]<=300 and dims[2]<=200,
                         'width_excess_mm':max(0,dims[1]-300),
                         'scope':'Zero steering/rocker pose only; not the full steering swept envelope.'}
    (ROOT/'validation_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
