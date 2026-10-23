"""Use watertight component STLs and measured masses, then aggregate each link."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]

def measured_mesh_properties(mesh, mass_kg):
    if not np.isfinite(mass_kg) or mass_kg <= 0:
        raise ValueError('Measured mass must be finite and positive')
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise ValueError('Volume inertia requires a closed, consistently wound mesh')
    properties = mesh.mass_properties
    if not np.isfinite(properties.volume) or properties.volume <= 0:
        raise ValueError('Mesh must enclose positive finite volume')
    inertia = np.array(properties.inertia) * (mass_kg/properties.mass)
    eigenvalues = np.linalg.eigvalsh(inertia)
    if eigenvalues[0] <= 0 or eigenvalues[2] > sum(eigenvalues[:2]) + 1e-12:
        raise ValueError('Mesh inertia is not physically valid')
    return {'mass_kg': float(mass_kg), 'com_m': properties.center_mass.tolist(),
            'inertia_kg_m2': inertia.tolist(), 'volume_m3': float(properties.volume),
            'effective_density_kg_m3': float(mass_kg/properties.volume)}

def main():
    config_path = ROOT/'config/model.json'
    config = json.loads(config_path.read_text())
    components = json.loads((ROOT/'config/components.json').read_text())
    refs = json.loads((ROOT/'config/cad_reference.json').read_text())
    total_before = sum(l.get('mass_kg',0) for l in config['links'])
    audit = {'method': 'Uniform volume density independently for each available component, normalized to its measured mass.',
             'limitations': ['FDM infill and perimeters are not spatially modeled.',
                             'Internal material differences in electronics, motors and sensors are intentionally ignored.',
                             'Prepared meshes can include recorded repairs and convex shell/hull proxies.',
                             'Three gears have no supplied geometry and retain their documented lumped approximation.',
                             'Collision geometry remains primitive; STL is used for component inertial properties.'],
             'parts': []}
    for part in components:
        source = part.get('inertial_mesh')
        if not source:
            continue
        path = ROOT/source
        mesh = trimesh.load_mesh(path, process=True)
        properties = measured_mesh_properties(mesh, part['mass_kg'])
        part.setdefault('prior_box_com_m', part['com_m'])
        part.setdefault('prior_box_inertia_kg_m2', part['inertia_kg_m2'])
        part.update(properties)
        part['approximation'] = 'Prepared closed component STL volume; uniform effective density normalized to measured mass; internal density differences ignored. See mesh_preparation for repairs/proxies.'
        T = np.array(refs['global_link_frames_m'][part['link']])
        ros = T[:3,:3] @ np.array(part['com_m']) + T[:3,3]
        part['global_com_cad_mm'] = (np.array(refs['CAD_to_ROS_rotation']).T @ (ros*1000) + refs['base_origin_cad_mm']).tolist()
        audit['parts'].append({'occurrence_id': part['occurrence_id'], 'source_model': part['source_model'],
            'mesh': source, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'watertight': bool(mesh.is_watertight), 'winding_consistent': bool(mesh.is_winding_consistent),
            'measured_mass_kg': part['mass_kg'], 'volume_cm3': properties['volume_m3']*1e6,
            'effective_density_kg_m3': properties['effective_density_kg_m3'],
            'prior_box_com_local_mm': (np.array(part['prior_box_com_m'])*1000).tolist(),
            'mesh_com_local_mm': (np.array(part['com_m'])*1000).tolist(),
            'mesh_com_cad_mm': part['global_com_cad_mm'], 'inertia_kg_m2': part['inertia_kg_m2'],
            'mesh_preparation':part.get('mesh_preparation',{}),'source_proxy':bool(part.get('source_proxy',False))})
    if len(audit['parts']) != 28:
        raise ValueError('Expected 28 available CAD/sensor component STL sources')
    for link in config['links']:
        parts = [p for p in components if p['link'] == link['name']]
        if not parts:
            continue
        mass = sum(p['mass_kg'] for p in parts)
        com = sum(p['mass_kg']*np.array(p['com_m']) for p in parts)/mass
        inertia = np.zeros((3,3))
        for part in parts:
            delta = np.array(part['com_m'])-com
            inertia += np.array(part['inertia_kg_m2']) + part['mass_kg']*(delta@delta*np.eye(3)-np.outer(delta,delta))
        link.update(mass_kg=mass, com_m=com.tolist(), inertia_kg_m2=inertia.tolist())
    total_after = sum(l.get('mass_kg',0) for l in config['links'])
    if abs(total_after-total_before) > 1e-12:
        raise ValueError('Inertia refinement must not change measured masses')
    config['assumptions']['mass_and_inertia'] = 'Measured total 2.3263 kg; all 28 available component meshes use prepared STL volume COM/inertia, uniform density within each component normalized to measured mass; repaired and proxy geometry is recorded; 3 gears retain lumped approximations; parallel-axis aggregation; wires/connectors/additional mounts omitted.'
    config_path.write_text(json.dumps(config, indent=2))
    (ROOT/'config/components.json').write_text(json.dumps(components, indent=2))
    with (ROOT/'config/component_masses.csv').open('w',newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['source_model','occurrence_id','link','mass_g','com_x_m','com_y_m','com_z_m'])
        for part in components:
            writer.writerow([part['source_model'],part.get('occurrence_id',''),part['link'],part['mass_kg']*1000,*part['com_m']])
    (ROOT/'config/mesh_inertia_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))

if __name__ == '__main__':
    main()
