"""Refresh the existing rover description from the supplied Fusion archive.

CAD geometry is retained. Masses are measured; component inertias are primitive
approximations with the parallel-axis theorem, not CAD material-density results.
"""
from pathlib import Path
import json, re, struct, shutil, hashlib, csv, math, sys
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

WORK = Path(__file__).resolve().parent
CAD = WORK / 'cad_current'
OUT = WORK / 'rover_description_rl'
sys.path.insert(0, str(WORK / 'cad_final'))
import inspect_geometry as ig

C = np.array([[0., 1, 0], [-1, 0, 0], [0, 0, 1]])
BASE_CAD_M = np.array([0., 0., 0.161813])

def transform(v, T):
    return v @ T[:3, :3].T + T[:3, 3]

def pose(p, R=None):
    T = np.eye(4)
    T[:3, 3] = p
    if R is not None:
        T[:3, :3] = R
    return T

def source_placements():
    bulk = (CAD / 'rover/FusionAssetName[Active]/FusionDesignSegmentType1/BulkStream.dat').read_bytes()
    desc = json.loads((CAD / 'DesignDescription.json').read_text())['designDescription']['designGraphs'][0]['designObjects']
    names = {x['id']: x['friendlyName'] for x in desc}
    root = next(x for x in desc if x['friendlyName'] == 'rover')
    roles = {e['metadata']['neutronRole']: names[e['id']] for e in root['references'][0]['relationships']}
    models = {}
    for m in re.finditer(rb'306(.{8})', bulk, re.DOTALL):
        oid = int.from_bytes(m.group(1), 'little')
        region = bulk[m.end():m.end()+750]
        hits = [name for role, name in roles.items() if role.encode('utf-16le') in region]
        if len(hits) == 1:
            models[oid] = hits[0]
    # The copied second driver shares the source of the original occurrence.
    models[5516] = models[5293]
    world = (CAD / 'rover/FusionAssetName[Active]/OGS.BlobFolder/OGS/DefaultScene/world').read_bytes()
    paths = list(re.finditer('PersistentPassiveNodePath'.encode('utf-16le'), world))
    result = []
    for m in re.finditer('OverrideTransformAttribute'.encode('utf-16le'), world):
        pm = next(v for v in paths if v.start() > m.start())
        p = pm.end()
        if struct.unpack_from('<I', world, p+36)[0] != 2:
            continue
        oid = int.from_bytes(world[p+56:p+64], 'big')
        T = np.frombuffer(world, '<f4', 16, m.end()+5).reshape(4, 4).T.astype(float)
        if oid in models:
            result.append({'occurrence_id': oid, 'model': models[oid], 'matrix_cm': T.tolist()})
    old = json.loads((WORK / 'cad_final/placements.json').read_text())
    # The grounded chassis has no override; its unchanged rigid transform is explicit.
    result.insert(0, next(x for x in old if x['occurrence_id'] == 203))
    assert len(result) == 28, f'Expected 28 external occurrences, got {len(result)}'
    return result

def native_mesh(name):
    cached = CAD / (name + '.npz')
    if cached.exists():
        d = np.load(cached)
        return d['vertices_cm'], d['triangles']
    ig.ROOT = CAD
    if name == 'NUCLEO-F411RE':
        # Fusion archive has B-reps but no display tessellation for this board.
        # Board footprint: ST UM1724 MB1136, 70 x 82.5 mm.
        # Local center is derived below from the CAD assembly mating origins.
        v = trimesh.creation.box(extents=[8.25, 7.0, 2.0])
        v.apply_translation([4.125, 3.5, -0.5785])
        return np.asarray(v.vertices), np.asarray(v.faces)
    v, f = ig.extract(name)
    np.savez(cached, vertices_cm=v, triangles=f)
    return v, f

def box_inertia(mass, size):
    x, y, z = size
    return np.diag([mass*(y*y+z*z)/12, mass*(x*x+z*z)/12, mass*(x*x+y*y)/12])

def inertial(specs):
    mass = sum(x['mass_kg'] for x in specs)
    com = sum(x['mass_kg'] * np.array(x['com_m']) for x in specs) / mass
    I = np.zeros((3, 3))
    for s in specs:
        d = np.array(s['com_m']) - com
        I += np.array(s['inertia_kg_m2']) + s['mass_kg'] * (np.dot(d, d)*np.eye(3) - np.outer(d, d))
    return mass, com.tolist(), I.tolist()

def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(WORK / 'rover_description', OUT)
    for f in (OUT / 'meshes').glob('*'):
        f.unlink()
    (OUT / 'mjcf').mkdir(exist_ok=True)
    placements = source_placements()
    by_id = {x['occurrence_id']: x for x in placements}
    ignored = {'Adafruit_ISM330DHXC', 'IntelRealsense_D435_Multibody'}
    placements = [x for x in placements if x['model'] not in ignored]
    (CAD / 'placements.json').write_text(json.dumps(placements, indent=2))

    config = json.loads((WORK / 'rover_description/config/model.json').read_text())
    frames = {'base_link': np.eye(4)}
    for j in config['joints']:
        T = pose(j['xyz'], Rotation.from_euler('xyz', j['rpy']).as_matrix())
        frames[j['child']] = frames[j['parent']] @ T
    # Structural joint locations are unchanged in the final assembly; the
    # latest occurrence transforms update each component's assembled mesh pose.
    groups = {'base_link': [203, 5132, 5293, 5516, 5661, 6038, 6734, 5010],
              'left_rocker_link': [211, 1501, 1783],
              'right_rocker_link': [338, 465, 674]}
    corners = {'fl': (2394, 3608, 4301), 'fr': (2946, 3596, 4165),
               'rl': (2812, 3247, 4309), 'rr': (2063, 3080, 3929)}
    masses = {203: .4085, 5132: .1781, 5293: .020, 5516: .020, 6734: .0129,
              5661: .0712, 6038: .0344, 5010: .0312, 211: .0608, 338: .0608}
    for oid in [465, 674, 1501, 1783]:
        masses[oid] = .0598
    for c, (hub, motor, wheel) in corners.items():
        groups[c+'_steering_link'] = [hub, motor]
        groups[c+'_wheel_link'] = [wheel]
        masses.update({hub: .0216, motor: .2032, wheel: .0471})
    sources = {name: native_mesh(name) for name in {x['model'] for x in placements}}
    components = []
    rendered = []
    link_specs = []
    for name, ids in groups.items():
        vertices, faces, pieces, collisions = [], [], [], []
        offset = 0
        for oid in ids:
            occurrence = by_id[oid]
            v, f = sources[occurrence['model']]
            Tg = np.array(occurrence['matrix_cm'])
            global_v = (transform(v, Tg)*.01 - BASE_CAD_M) @ C.T
            local = transform(global_v, np.linalg.inv(frames[name]))
            lo, hi = local.min(0), local.max(0)
            center, size = (lo+hi)/2, hi-lo
            # Use the component's own bounding box orientation, not the broad
            # group box, to avoid spreading motor and battery mass arbitrarily.
            R = frames[name][:3, :3].T @ C @ Tg[:3, :3]
            native_size = np.ptp(v, axis=0)*.01
            I = R @ box_inertia(masses[oid], native_size) @ R.T
            if name.endswith('_wheel_link'):
                radius = float(np.linalg.norm(local[:, :2], axis=1).max())
                length = float(np.ptp(local[:, 2]))
                center = np.array([0., 0., center[2]])
                I = np.diag([masses[oid]*(3*radius**2+length**2)/12]*2 + [masses[oid]*radius**2/2])
                collisions = [{'type': 'cylinder', 'xyz': center.tolist(), 'radius': radius, 'length': length}]
            else:
                collisions.append({'type': 'box', 'xyz': center.tolist(), 'size': size.tolist()})
            part = {'occurrence_id': oid, 'source_model': occurrence['model'], 'link': name,
                    'mass_kg': masses[oid], 'com_m': center.tolist(), 'inertia_kg_m2': I.tolist(),
                    'cad_transform_cm': occurrence['matrix_cm'],
                    'global_com_cad_mm': ((C.T @ (frames[name][:3,:3] @ center + frames[name][:3,3]) + BASE_CAD_M)*1000).tolist(),
                    'approximation': 'Uniform component bounding box; measured mass'}
            if oid == 6038:
                part['approximation'] = 'Nucleo board box proxy: no OGS tessellation in F3D; footprint from ST UM1724; native corner/height alignment estimated from CAD mating origins.'
            if oid == 6734:
                part['usage'] = 'Steering driver; MDD3A mesh is a user-authorized visual substitute.'
            if oid == 5661:
                part['usage'] = 'Raspberry Pi plus AI HAT combined measured mass; AI HAT inertia lumped into board envelope.'
            pieces.append(part)
            components.append(part)
            vertices.append(local)
            faces.append(f + offset)
            offset += len(v)
            rendered.append((occurrence['model'], global_v*100, f))
        # Omitted gears still contribute finite approximate inertial properties.
        gear_mass = .0156 if name == 'base_link' else (.0058 if name in ['left_rocker_link', 'right_rocker_link'] else 0)
        if gear_mass:
            p = {'source_model': 'omitted_large_gear' if name == 'base_link' else 'omitted_small_gear',
                 'link': name, 'mass_kg': gear_mass, 'com_m': [0, 0, 0],
                 'inertia_kg_m2': (np.eye(3)*gear_mass*.01**2).tolist(),
                 'approximation': 'Gear mass lumped at pivot; 10 mm inertial radius; large gear assigned to chassis, small gears to respective rocker.'}
            pieces.append(p)
            components.append(p)
        v, f = np.concatenate(vertices), np.concatenate(faces)
        trimesh.Trimesh(vertices=v, faces=f, process=False).export(OUT/'meshes'/(name+'.stl'))
        mass, com, I = inertial(pieces)
        link_specs.append({'name': name, 'mesh': 'meshes/'+name+'.stl', 'mass_kg': mass,
                           'com_m': com, 'inertia_kg_m2': I, 'collisions': collisions,
                           'color': [.18,.2,.22,1] if name.endswith('_wheel_link') else [.7,.73,.76,1]})

    sensor_params = {'camera_origin_cad_mm': [0,116.619,157.691], 'imu_origin_cad_mm': [0,0,137.213],
                     'stereo_baseline_m': .050, 'front_plate_to_depth_plane_m': .0043,
                     'camera_forward': '+CAD Y / +ROS X projected heading; 35 degrees downward about camera STL origin',
                     'camera_mesh_rpy': [math.pi/2,0,math.pi/2],
                     'imu_rpy_ros': [0,0,-math.pi/2],
                     'imu_measurement_origin': 'STL origin; chip-specific offset not measured'}
    for name, filename, mass, point, rpy in [
            ('camera_link', 'IntelRealsense_D435_Multibody.stl', .072, sensor_params['camera_origin_cad_mm'], [0,math.radians(35),0]),
            ('imu_link', 'imu.stl', .0024, sensor_params['imu_origin_cad_mm'], sensor_params['imu_rpy_ros'])]:
        T = pose(C @ (np.array(point)*.001-BASE_CAD_M), Rotation.from_euler('xyz', rpy).as_matrix())
        frames[name] = T
        mesh = trimesh.load_mesh(WORK/'upload'/filename, process=False)
        mesh.vertices *= .001
        if name == 'camera_link':
            mesh.vertices = mesh.vertices @ Rotation.from_euler('xyz', sensor_params['camera_mesh_rpy']).as_matrix().T
        mesh.export(OUT/'meshes'/(name+'.stl'))
        lo, hi = mesh.bounds
        com = (lo+hi)/2
        I = box_inertia(mass, hi-lo)
        link_specs.append({'name': name, 'mesh': 'meshes/'+name+'.stl', 'mass_kg': mass,
                           'com_m': com.tolist(), 'inertia_kg_m2': I.tolist(),
                           'collisions': [{'type':'box','xyz':com.tolist(),'size':(hi-lo).tolist()}], 'color':[.45,.5,.55,1]})
        config['joints'].append({'name':name+'_joint','type':'fixed','parent':'base_link','child':name,
                                 'xyz':T[:3,3].tolist(),'rpy':rpy})
        rendered.append((name, transform(mesh.vertices,T)*100, mesh.faces))

    # camera_link is the STL front-plate midpoint, distinct from the left
    # camera's nominal zero-depth plane used by the RealSense depth stream.
    for child, xyz, rpy, parent in [
        ('camera_depth_frame',[-.0043,.025,0],[0,0,0],'camera_link'),
        ('camera_infra1_frame',[-.0043,.025,0],[0,0,0],'camera_link'),
        ('camera_infra2_frame',[-.0043,-.025,0],[0,0,0],'camera_link'),
        ('camera_stereo_center_frame',[-.0043,0,0],[0,0,0],'camera_link'),
        ('imu_sensor_frame',[0,0,0],[0,0,0],'imu_link')]:
        link_specs.append({'name':child})
        config['joints'].append({'name':child+'_joint','type':'fixed','parent':parent,'child':child,'xyz':xyz,'rpy':rpy})
        frames[child] = frames[parent] @ pose(xyz)
    for parent in ['camera_depth_frame','camera_infra1_frame','camera_infra2_frame','camera_stereo_center_frame']:
        child = parent.replace('_frame','_optical_frame')
        rpy = [-math.pi/2,0,-math.pi/2]
        link_specs.append({'name':child})
        config['joints'].append({'name':child+'_joint','type':'fixed','parent':parent,'child':child,'xyz':[0,0,0],'rpy':rpy})

    config['links'] = link_specs
    config['source'] = 'rover.f3z (2026-10-04 22:28 upload), plus separate sensor STLs'
    config['source_sha256'] = hashlib.sha256((WORK/'upload/rover.f3z').read_bytes()).hexdigest()
    config['sensor_parameters'] = sensor_params
    config['assumptions'].update({'mass_and_inertia':'Measured total 2.3263 kg; per-component primitive inertias and parallel-axis aggregation; wires/connectors/additional mounts not included.',
        'gear_inertia':'Gear geometry omitted. Large gear mass on base, two small gears on rocker pivots; exact gear inertia/attachment remains approximate.',
        'camera':'Passive monochrome stereo, emitter off; 50 mm nominal baseline and 4.3 mm nominal zero-depth offset; 35 degree downward pitch about unchanged STL origin. Calibrated extrinsics take precedence.',
        'imu':'STL origin at CAD (0,0,137.213) mm; native STL axes retained; measurement site at STL origin is an approximation.',
        'collisions':'Per-component boxes; wheel cylinders including grousers. Rigid terrain only; not a deformable sand model.',
        'electronics':'MDD3A occurrences 5293/5516 are drive drivers at 20 g each; occurrence 6734 is steering-driver substitute at 12.9 g.',
        'stm_visual':'F411RE CAD proxy for actual F446RE; simplified MB1136 box because source has no display tessellation.'})
    (OUT/'config/model.json').write_text(json.dumps(config, indent=2))
    (OUT/'config/components.json').write_text(json.dumps(components, indent=2))
    bounds = np.stack([np.concatenate([v for _,v,_ in rendered]).min(0),np.concatenate([v for _,v,_ in rendered]).max(0)])*.01
    mass = sum(x.get('mass_kg',0) for x in link_specs)
    com = sum(x.get('mass_kg',0)*(frames[x['name']][:3,:3]@np.array(x['com_m'])+frames[x['name']][:3,3]) for x in link_specs if 'mass_kg' in x)/mass
    reference = {'base_origin_cad_mm':(BASE_CAD_M*1000).tolist(),'CAD_to_ROS_rotation':C.tolist(),
                 'cad_occurrences_used':placements,'ignored_cad_sensor_ids':[4811,4991],
                 'global_link_frames_m':{k:v.tolist() for k,v in frames.items()},
                 'total_mass_kg':mass,'zero_pose_com_ros_m':com.tolist(),
                 'zero_pose_com_cad_mm':((C.T@com+BASE_CAD_M)*1000).tolist(),
                 'zero_pose_visual_bounds_ros_m':bounds.tolist(),
                 'zero_pose_visual_envelope_ros_xyz_mm':((bounds[1]-bounds[0])*1000).tolist()}
    (OUT/'config/cad_reference.json').write_text(json.dumps(reference, indent=2))
    with (OUT/'config/component_masses.csv').open('w',newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['source_model','occurrence_id','link','mass_g','com_x_m','com_y_m','com_z_m'])
        for p in components:
            writer.writerow([p['source_model'],p.get('occurrence_id',''),p['link'],p['mass_kg']*1000,*p['com_m']])
    shutil.copy(WORK/'test_rover_requirements.py',OUT/'scripts/test_requirements.py')
    (OUT/'scripts/validate_model.py').unlink(missing_ok=True)
    (OUT/'validation_report.json').unlink(missing_ok=True)
    shutil.copy(WORK/'cad_new_preview.png',OUT/'docs/source_cad_preview.png')
    ig.render(rendered,OUT/'docs/assembly_preview.png')
    print(json.dumps({'mass_kg':mass,'com_cad_mm':reference['zero_pose_com_cad_mm'],
                      'envelope_mm':reference['zero_pose_visual_envelope_ros_xyz_mm'],'links':len(link_specs)},indent=2))

if __name__ == '__main__':
    main()
