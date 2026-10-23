from pathlib import Path
import json, re, struct
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

ROOT = Path(__file__).parent

def transform(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]

def extract(model):
    folder = ROOT / model / 'FusionAssetName[Active]/OGS.BlobFolder/OGS/DefaultScene'
    world = (folder / 'world').read_bytes()
    blob = (folder / 'Fusion_mesh_000').read_bytes()
    tokens = list(re.finditer(rb'(?:[\x20-\x7e]\x00){4,}', world))
    component_transforms = []
    for i, token in enumerate(tokens):
        if token.group().decode('utf-16le') != 'TransformAttribute':
            continue
        near = [x.group().decode('utf-16le') for x in tokens[max(0, i-3):i]]
        if 'Component' not in near:
            continue
        matrix = np.frombuffer(world, '<f4', 16, token.end()+5).reshape(4,4).T.copy()
        component_transforms.append((token.start(), matrix))
    vertices, triangles, seen = [], [], set()
    for m in re.finditer(re.escape(struct.pack('<I', 4)+'Face'.encode('utf-16le')), world):
        off, p, n, uv, ni = struct.unpack_from('<5I', world, m.end()+57)
        if not p or p % 3 or n != p or uv not in (0, p//3*2):
            continue
        if off + (p+n+uv+ni)*4 > len(blob):
            continue
        nv = p//3
        data = np.frombuffer(blob, '<f4', p+n+uv, off).reshape(nv, (p+n+uv)//nv)
        indices = np.frombuffer(blob, '<u4', ni, off+(p+n+uv)*4)
        if ni % 3 or indices.max(initial=0) >= nv or not np.isfinite(data).all():
            continue
        ci = next((i for i, (pos, _) in enumerate(component_transforms) if pos > m.start()), None)
        matrix = np.eye(4) if ci is None else component_transforms[ci][1].copy()
        # This source contains four bolt instances inside one transformed subassembly.
        if model == 'STS3215_c' and ci in (3,4,5,6):
            matrix = component_transforms[7][1] @ matrix
        key = (off, ci)
        if key in seen:
            continue
        seen.add(key)
        assert np.allclose(np.linalg.norm(data[:,3:6], axis=1), 1, atol=2e-6)
        vertices.append(transform(data[:,:3], matrix))
        triangles.append(indices.reshape(-1,3) + sum(len(v) for v in vertices[:-1]))
    return np.concatenate(vertices), np.concatenate(triangles)

def polydata(vertices, triangles):
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.asarray(vertices, dtype=np.float64), deep=True))
    cells = vtk.vtkCellArray()
    cells.SetCells(len(triangles), numpy_to_vtkIdTypeArray(np.c_[np.full(len(triangles),3), triangles].astype(np.int64).ravel(), deep=True))
    data = vtk.vtkPolyData(); data.SetPoints(points); data.SetPolys(cells)
    normals = vtk.vtkPolyDataNormals(); normals.SetInputData(data); normals.SetFeatureAngle(45); normals.Update()
    return normals.GetOutput()

def render(parts, path, size=(1400,1100)):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from vtk.util.numpy_support import vtk_to_numpy
    fig=plt.figure(figsize=(size[0]/140,size[1]/140),dpi=140)
    ax=fig.add_subplot(111,projection='3d')
    all_vertices=[]; all_faces=[]; all_colors=[]
    for name, vertices, triangles in parts:
        data=polydata(vertices,triangles)
        if len(triangles)>4500:
            dec=vtk.vtkQuadricDecimation();dec.SetInputData(data);dec.SetTargetReduction(1-4500/len(triangles));dec.Update();data=dec.GetOutput()
        v=vtk_to_numpy(data.GetPoints().GetData());f=vtk_to_numpy(data.GetPolys().GetData()).reshape(-1,4)[:,1:]
        faces=v[f];normals=np.cross(faces[:,1]-faces[:,0],faces[:,2]-faces[:,0]);normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
        light=np.array([0.4,-0.6,0.7]);light/=np.linalg.norm(light)
        shade=0.6+0.4*np.maximum(normals@light,0)
        color=np.array((0.13,0.15,0.18) if 'STS' in name else ((0.52,0.55,0.59) if 'wheel' in name else (0.7,0.74,0.79)))
        all_faces.append(faces);all_colors.append(shade[:,None]*color)
        all_vertices.append(vertices)
    ax.add_collection3d(Poly3DCollection(np.concatenate(all_faces),facecolors=np.concatenate(all_colors),edgecolors='none',zsort='average'))
    all_vertices=np.concatenate(all_vertices);lo=all_vertices.min(0);hi=all_vertices.max(0);mid=(lo+hi)/2;span=max(hi-lo)*0.58
    ax.set_xlim(mid[0]-span,mid[0]+span);ax.set_ylim(mid[1]-span,mid[1]+span);ax.set_zlim(mid[2]-span,mid[2]+span)
    ax.set_box_aspect((1,1,1));ax.view_init(elev=28,azim=-55);ax.set_axis_off();fig.tight_layout();fig.savefig(path,bbox_inches='tight',facecolor='white');plt.close(fig)

if __name__=='__main__':
    placements=json.loads((ROOT/'placements.json').read_text())
    models={x['model'] for x in placements};meshes={name:extract(name) for name in models}
    parts=[]
    for name,(v,f) in meshes.items():
        np.savez(ROOT/(name+'.npz'),vertices_cm=v,triangles=f)
        print(name,len(v),len(f),'bounds cm',v.min(0).round(4),v.max(0).round(4))
    for x in placements:
        v,f=meshes[x['model']];parts.append((x['model'],transform(v,np.array(x['matrix_cm'])),f))
    render(parts,ROOT/'assembly_check.png')
