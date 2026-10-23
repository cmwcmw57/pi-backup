"""Prepare component volume meshes; preserve visual meshes and audit repairs/proxies."""
from pathlib import Path
import json
import numpy as np
import trimesh
import pymeshfix
from scipy.spatial import ConvexHull

ROOT=Path(__file__).resolve().parents[1]

def closed_convex_hull(mesh):
    hull=mesh.convex_hull
    # Float32 CAD tessellation can leave one numerical triangle hole in a hull.
    trimesh.repair.fill_holes(hull)
    trimesh.repair.fix_normals(hull,multibody=True)
    if not hull.is_volume:
        # Keep Qhull's complete indexed face set: removing nearly planar triangles
        # during automatic cleanup can open extremely thin CAD shell hulls.
        indexed=ConvexHull(mesh.vertices)
        hull=trimesh.Trimesh(vertices=mesh.vertices,faces=indexed.simplices,process=False)
        trimesh.repair.fix_normals(hull,multibody=True)
        hull.remove_unreferenced_vertices()
    if not hull.is_volume:
        raise ValueError('Convex proxy did not produce a closed positive volume')
    return hull

def prepare_inertial_mesh(mesh):
    original=mesh.copy()
    audit={'source_watertight':bool(mesh.is_watertight), 'source_winding_consistent':bool(mesh.is_winding_consistent),
           'source_faces':len(mesh.faces), 'repaired_shells':0, 'convex_shell_proxies':0,
           'discarded_surface_fragments':0, 'void_shells':0, 'whole_component_hull_proxy':False}
    if mesh.is_volume and mesh.body_count==1:
        result=mesh.copy();audit['method']='original_closed_single_solid'
    else:
        solids=[];voids=[]
        shells=mesh.split(only_watertight=False,repair=False)
        for shell in shells:
            if not shell.is_watertight:
                before=len(shell.faces)
                trimesh.repair.fill_holes(shell)
                if len(shell.faces)!=before:audit['repaired_shells']+=1
            if shell.is_watertight and shell.is_winding_consistent:
                with np.errstate(divide='ignore',invalid='ignore'):
                    volume=float(shell.volume)
                if abs(volume)<1e-18:
                    audit['discarded_surface_fragments']+=1;continue
                if volume<0:
                    shell.invert();voids.append(shell);continue
                solids.append(shell);continue
            # Planar display labels cannot define a 3D density or contribute solid volume.
            if len(shell.vertices)<4 or np.linalg.matrix_rank(shell.vertices-shell.vertices.mean(0),tol=1e-8)<3:
                audit['discarded_surface_fragments']+=1;continue
            if min(shell.extents)<1e-5:
                audit['discarded_surface_fragments']+=1;continue
            repaired=None
            if len(shell.faces)>=32:
                v,f=pymeshfix.clean_from_arrays(np.asarray(shell.vertices,dtype=np.float64),
                    np.asarray(shell.faces,dtype=np.int32),verbose=False,joincomp=False,remove_smallest_components=False)
                if len(v) and len(f):
                    candidate=trimesh.Trimesh(vertices=v,faces=f,process=True)
                    if candidate.is_volume and np.max(np.abs(candidate.bounds-shell.bounds))<=.0002:
                        repaired=candidate;audit['repaired_shells']+=1
            if repaired is None:
                repaired=closed_convex_hull(shell);audit['convex_shell_proxies']+=1
            solids.append(repaired)
        audit['positive_shells']=len(solids);audit['void_shells']=len(voids)
        try:
            if not solids:raise ValueError('No usable 3D solids')
            result=solids[0].copy() if len(solids)==1 else trimesh.boolean.union(solids,engine='manifold')
            if voids:
                cavity=voids[0] if len(voids)==1 else trimesh.boolean.union(voids,engine='manifold')
                result=trimesh.boolean.difference([result,cavity],engine='manifold')
            if not result.is_volume:
                before=len(result.faces)
                trimesh.repair.fill_holes(result)
                audit['numerical_union_hole_faces_added']=len(result.faces)-before
            if not result.is_volume:raise ValueError('Prepared union is not a valid volume')
            if np.max(np.abs(result.bounds-original.bounds))>.0002:
                raise ValueError('Repair lost more than 0.2 mm of the source envelope')
            audit['method']='closed_shell_union_with_recorded_repairs'
        except (ValueError,RuntimeError) as exc:
            result=closed_convex_hull(original)
            audit['method']='whole_component_convex_hull_proxy'
            audit['whole_component_hull_proxy']=True
            audit['proxy_reason']=str(exc)
    if not result.is_volume:
        raise ValueError('Prepared inertial mesh must enclose a positive consistently oriented volume')
    audit['output_faces']=len(result.faces)
    audit['output_volume_cm3']=float(result.volume*1e6)
    audit['bounds_difference_mm']=float(np.max(np.abs(result.bounds-original.bounds))*1000)
    return result,audit

def serialize_inertial_mesh(mesh, destination):
    """Preserve indexed topology across STL's triangle-soup/float32 round trip.

    Boolean unions can retain distinct vertices at solid-to-solid edge contacts.
    Welding identical coordinates on STL import otherwise creates nonmanifold edges.
    Separate only those numerical clusters by at most 1 micrometre per coordinate,
    then verify closedness and the negligible volume/centroid change explicitly.
    """
    mesh.export(destination)
    check=trimesh.load_mesh(destination,process=True)
    shift=0.
    if not check.is_volume:
        _,indices,counts=np.unique(np.round(mesh.vertices,8),axis=0,return_inverse=True,return_counts=True)
        mask=counts[indices]>1
        for epsilon in [1e-7,1e-6]:
            candidate=mesh.copy()
            rng=np.random.default_rng(12345)
            offsets=rng.uniform(-epsilon,epsilon,size=(int(mask.sum()),3))
            candidate.vertices[mask]+=offsets
            candidate.export(destination)
            check=trimesh.load_mesh(destination,process=True)
            if check.is_volume:
                shift=float(np.max(np.linalg.norm(offsets,axis=1))) if len(offsets) else 0.
                break
    if not check.is_volume:
        raise ValueError('STL round trip did not preserve a valid volume: '+str(destination))
    volume_change=abs(check.volume-mesh.volume)/mesh.volume
    centroid_shift=float(np.linalg.norm(check.center_mass-mesh.center_mass))
    if volume_change>1e-4 or centroid_shift>2e-6:
        raise ValueError('STL round trip changed mass distribution beyond the numerical tolerance')
    return check,{'maximum_vertex_shift_mm':shift*1000,
                 'relative_volume_change':float(volume_change),'centroid_shift_mm':centroid_shift*1000}

def source_mesh(part):
    visual=ROOT/part['source_visual_mesh']
    raw=visual.read_bytes()
    n=int.from_bytes(raw[80:84],'little')
    dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attribute','<u2')])
    start,stop=part['source_visual_face_range']
    if not 0<=start<stop<=n:raise ValueError('Component triangle range outside visual STL')
    triangles=np.frombuffer(raw,dtype=dtype,offset=84,count=n)['vertices'][start:stop]
    return trimesh.Trimesh(vertices=triangles.reshape(-1,3),faces=np.arange(len(triangles)*3).reshape(-1,3),process=True)

def main():
    path=ROOT/'config/components.json'
    components=json.loads(path.read_text())
    audit={'scope':'Each available physical component independently, using its measured mass after preparation.',
           'parts':[],'omitted_geometry':[]}
    for part in components:
        if 'source_visual_mesh' not in part:
            audit['omitted_geometry'].append(part['source_model']);continue
        key=part['occurrence_id']
        prepared,record=prepare_inertial_mesh(source_mesh(part))
        destination=ROOT/'meshes/inertial_parts'/('component_'+str(key)+'.stl')
        check,serialization=serialize_inertial_mesh(prepared,destination)
        record['stl_serialization']=serialization
        record['output_volume_cm3']=float(check.volume*1e6)
        part['inertial_mesh']=str(destination.relative_to(ROOT))
        part['mesh_preparation']=record
        record.update(occurrence_id=key,source_model=part['source_model'],mesh=part['inertial_mesh'],
                      source_proxy=bool(part.get('source_proxy',False)))
        audit['parts'].append(record)
        print(str(key)+' '+part['source_model']+' '+record['method']+' '+str(round(record['output_volume_cm3'],4))+' cm3',flush=True)
    if len(audit['parts'])!=28:raise ValueError('Expected 28 CAD/sensor components with supplied visual meshes')
    path.write_text(json.dumps(components,indent=2))
    (ROOT/'config/mesh_preparation_audit.json').write_text(json.dumps(audit,indent=2))

if __name__=='__main__':
    main()
