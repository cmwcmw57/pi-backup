# ================================================================
# site_map.py
# 설명: 팀원 STL을 meter·중력 기준 지형 격자로 변환하고 경로와 저장본을 검증한다.
# 사용법: from ksrc_runtime.handheld_mapping.site_map import load_stl, SiteMap
# ================================================================
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from typing import TYPE_CHECKING
from scipy.ndimage import distance_transform_edt

from ksrc_runtime.handheld_mapping.storage import atomic_json

if TYPE_CHECKING:
    import open3d as o3d


def file_hash(path: Path) -> str:
    """대형 원본을 chunk 단위로 fingerprint한다."""
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


class SiteMap:
    """STL 형상·좌표 provenance와 보수적인 지면/장애물 grid를 함께 유지한다."""

    @classmethod
    def from_mesh(cls, mesh: o3d.geometry.TriangleMesh, *, units: str='m',
                  up: list[float] | None=None, resolution: float=.05,
                  robot_radius: float=.15, max_step_m: float=.06,
                  source: Path | None=None, surface_mode: str='conservative') -> SiteMap:
        from ksrc_runtime.handheld_mapping.reference import manual_gravity
        if units not in ('m','mm') or not all(math.isfinite(v) for v in (resolution,robot_radius,max_step_m)):
            raise ValueError('invalid units or grid parameters')
        if not .005<=resolution<=.5 or robot_radius<0 or max_step_m<=0:
            raise ValueError('invalid grid resolution/radius/step')
        if surface_mode not in ('conservative','top'):raise ValueError('invalid surface selection')
        mesh=copy.deepcopy(mesh)
        vertices,triangles=np.asarray(mesh.vertices),np.asarray(mesh.triangles)
        if (len(vertices)<3 or not len(triangles) or len(triangles)>2_000_000
                or not np.isfinite(vertices).all() or triangles.min()<0 or triangles.max()>=len(vertices)):
            raise ValueError('nonempty finite triangle mesh required; maximum 2000000 triangles')
        geometry_hash=hashlib.sha256(vertices.tobytes()+triangles.tobytes()).hexdigest()
        matrix=np.asarray(manual_gravity([0,0,1] if up is None else up)['transformation'])
        scale=1. if units=='m' else .001
        mesh.scale(scale,center=[0,0,0]);mesh.transform(matrix);mesh.compute_triangle_normals()
        points=np.asarray(mesh.vertices)
        origin=np.floor(points[:,:2].min(axis=0)/resolution)*resolution
        cols,rows=np.ceil((points[:,:2].max(axis=0)-origin)/resolution).astype(int)
        if min(cols,rows)<2 or cols*rows>1_000_000:
            raise ValueError('grid extent requires 4..1000000 cells')
        self=cls()
        self.mesh,self.resolution,self.origin=mesh,float(resolution),origin
        self.robot_radius,self.max_step_m=float(robot_radius),float(max_step_m)
        self.frame='site'
        self.source=Path(source).resolve() if source is not None else None
        source_hash=file_hash(self.source) if self.source is not None else geometry_hash
        self.metadata={'schema_version':1,'grid_revision':3,'frame':self.frame,'units':'m','source_units':units,
                       'source_path':str(self.source) if self.source else None,'source_sha256':source_hash,
                       'source_geometry_sha256':geometry_hash,'scale_to_m':scale,
                       'T_site_source_axes':matrix.tolist(),'origin':origin.tolist(),
                       'resolution':self.resolution,'robot_radius':self.robot_radius,'max_step_m':self.max_step_m,
                       'shape':[int(rows),int(cols)],'accuracy_verified':False,'surface_mode':surface_mode,
                       'grid_method':'vertical surface; up to 16 lower hits; multiple layers unknown; exposed steep walls blocked',
                       'limitation':'2.5D; thick closed solids or overlapping floors can remain unknown'}
        self.identity=hashlib.sha256(json.dumps(self.metadata,sort_keys=True).encode()).hexdigest()
        self._rasterize(int(rows),int(cols))
        if not self.observed.any():
            raise ValueError('no unambiguous ground surface: check units/up; explicitly select top surface for a solid base')
        return self

    def _rasterize(self, rows: int, cols: int) -> None:
        import open3d as o3d
        scene=o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(self.mesh))
        vertices=np.asarray(self.mesh.vertices)
        top=float(vertices[:,2].max()+1.)
        height=np.full(rows*cols,np.nan);slope=np.full(rows*cols,np.nan)
        observed=np.zeros(rows*cols,bool)
        for first in range(0,rows*cols,50_000):
            idx=np.arange(first,min(rows*cols,first+50_000))
            rays=np.zeros((len(idx),6),np.float32)
            rays[:,0]=self.origin[0]+(idx%cols+.5)*self.resolution
            rays[:,1]=self.origin[1]+(idx//cols+.5)*self.resolution
            rays[:,2]=top;rays[:,5]=-1.
            hit=scene.cast_rays(o3d.core.Tensor(rays))
            distance=hit['t_hit'].numpy();normal=hit['primitive_normals'].numpy()
            valid=np.isfinite(distance)
            z=top-distance
            nz=np.abs(normal[:,2])/np.maximum(np.linalg.norm(normal,axis=1),1e-12)
            epsilon=max(1e-4,self.resolution*1e-3)
            multi=np.zeros(len(idx),bool)
            if self.metadata['surface_mode']=='conservative':
                active=np.flatnonzero(valid & (nz>=.35))
                rays[active,2]=z[active]-epsilon
                for _ in range(16):
                    if not len(active):break
                    next_hit=scene.cast_rays(o3d.core.Tensor(rays[active]))['t_hit'].numpy()
                    finite=np.isfinite(next_hit)
                    lower=rays[active,2]-next_hit
                    separated=finite & (z[active]-lower>self.max_step_m)
                    multi[active[separated]]=True
                    keep=finite & ~separated
                    remaining=active[keep]
                    rays[remaining,2]=lower[keep]-epsilon
                    active=remaining
                # More unresolved close hits cannot establish a unique 2.5D surface.
                multi[active]=True
            good=valid & (nz>=.35) & (~multi if self.metadata['surface_mode']=='conservative' else True)
            height[idx]=np.where(good,z,np.nan)
            slope[idx]=np.where(good,np.degrees(np.arccos(np.clip(nz,0,1))),np.nan)
            observed[idx]=good
        self.height_m=height.reshape(rows,cols)
        self.slope_deg=slope.reshape(rows,cols)
        self.observed=observed.reshape(rows,cols)
        walls=np.zeros((rows,cols),bool)
        normals=np.asarray(self.mesh.triangle_normals)
        steep=np.abs(normals[:,2])<.35
        budget=0
        for triangle in np.asarray(self.mesh.triangles)[steep]:
            xyz=vertices[triangle]
            for a,b in zip(xyz,np.roll(xyz,-1,axis=0)):
                count=max(2,int(np.ceil(np.linalg.norm(b[:2]-a[:2])/self.resolution*2))+1)
                budget+=count
                if budget>5_000_000:raise ValueError('projected wall raster budget exceeded')
                samples=np.linspace(a,b,count)
                cells=np.floor((samples[:,:2]-self.origin)/self.resolution).astype(int)
                valid=(cells[:,0]>=0)&(cells[:,0]<cols)&(cells[:,1]>=0)&(cells[:,1]<rows)
                if self.metadata['surface_mode']=='top':
                    ids=np.flatnonzero(valid);xx,yy=cells[ids].T
                    # Buried CAD side faces cannot obstruct the explicitly selected upper surface.
                    valid[ids]&=~self.observed[yy,xx] | (samples[ids,2]>=self.height_m[yy,xx]-1e-4)
                walls[cells[valid,1],cells[valid,0]]=True
        step=np.zeros((rows,cols),bool)
        for axis in (0,1):
            diff=np.abs(np.diff(self.height_m,axis=axis))>self.max_step_m
            if axis==0:step[:-1]|=diff;step[1:]|=diff
            else:step[:,:-1]|=diff;step[:,1:]|=diff
        self.blocked=walls|step

    def safe_mask(self, max_slope_deg: float=20., slack_m: float=0.) -> np.ndarray:
        if not math.isfinite(max_slope_deg) or not 0<max_slope_deg<90:
            raise ValueError('slope must be in (0,90)')
        clear=self.observed & ~self.blocked & (self.slope_deg<=max_slope_deg+1e-4)
        distance=distance_transform_edt(np.pad(clear,1,constant_values=False))[1:-1,1:-1]*self.resolution
        # cell centers require at least the configured radius; map exterior remains occupied.
        return clear & (distance>=max(0.,self.robot_radius-slack_m)+self.resolution*math.sqrt(2)-1e-9)

    def cells(self, xy: np.ndarray) -> np.ndarray:
        return np.floor((np.asarray(xy,float)-self.origin)/self.resolution).astype(int)

    def validate_path(self, xy: list | np.ndarray, max_slope_deg: float=20., slack_m: float=0.) -> dict:
        points=np.asarray(xy,float)
        if points.ndim!=2 or points.shape[1]!=2 or len(points)<1 or not np.isfinite(points).all():
            return {'accepted':False,'reason':'invalid_points'}
        samples=[points[0]]
        for a,b in zip(points[:-1],points[1:]):
            count=max(1,int(np.ceil(np.linalg.norm(b-a)/(self.resolution*.4))))
            if count>1_000_000:return {'accepted':False,'reason':'path_too_long'}
            samples.extend(np.linspace(a,b,count+1)[1:])
        cells=self.cells(np.asarray(samples));cols,rows=self.observed.shape[1],self.observed.shape[0]
        if np.any(cells<0) or np.any(cells[:,0]>=cols) or np.any(cells[:,1]>=rows):
            return {'accepted':False,'reason':'outside_map'}
        safe=self.safe_mask(max_slope_deg,slack_m)
        valid=safe[cells[:,1],cells[:,0]]
        return {'accepted':bool(valid.all()),'reason':'clear' if valid.all() else 'unknown_obstacle_clearance_or_slope',
                'length_m':float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()),
                'samples':len(cells)}

    def save(self, folder: Path) -> None:
        if self.mesh is None:raise ValueError('mesh required for site save; load full mesh first')
        import open3d as o3d
        folder=Path(folder).resolve();folder.mkdir(parents=True,exist_ok=False)
        with (folder/'grid.npz').open('wb') as stream:
            np.savez_compressed(stream,height_m=self.height_m,slope_deg=self.slope_deg,
                                observed=self.observed,blocked=self.blocked)
        if not o3d.io.write_triangle_mesh(str(folder/'mesh.ply'),self.mesh):
            raise OSError('site mesh write failed')
        result=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=False,
                              cwd=Path(__file__).resolve().parents[2])
        atomic_json(folder/'site.json',{**self.metadata,'identity':self.identity,
                    'code_commit':result.stdout.strip() or None,
                    'assets':{name:file_hash(folder/name) for name in ('grid.npz','mesh.ply')}})

    @classmethod
    def load(cls, folder: Path, *, load_mesh: bool=True) -> SiteMap:
        folder=Path(folder).resolve();data=json.loads((folder/'site.json').read_text())
        if data.get('schema_version')!=1 or data.get('frame')!='site' or data.get('units')!='m':
            raise ValueError('invalid site metadata')
        if data.get('grid_revision')!=3:raise ValueError('old grid revision; rebuild site with current surface checks')
        meta={k:v for k,v in data.items() if k not in ('identity','code_commit','assets')}
        if hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()!=data['identity']:
            raise ValueError('site metadata hash changed')
        for name in ('grid.npz','mesh.ply'):
            if name == 'mesh.ply' and not load_mesh:
                continue  # 실기 실행기는 검증된 격자만 사용하며 mesh를 읽지 않는다.
            if file_hash(folder/name)!=data['assets'][name]:raise ValueError('site asset hash changed')
        source=Path(data['source_path']) if data['source_path'] else None
        if source is not None and source.is_file() and file_hash(source)!=data['source_sha256']:
            raise ValueError('source STL changed; prepare a new site')
        self=cls();self.metadata=meta;self.identity=data['identity'];self.source=source
        self.frame='site';self.origin=np.asarray(data['origin'],float)
        self.resolution=float(data['resolution']);self.robot_radius=float(data['robot_radius'])
        self.max_step_m=float(data['max_step_m']);self.mesh=None
        if load_mesh:
            import open3d as o3d
            self.mesh=o3d.io.read_triangle_mesh(str(folder/'mesh.ply'))
        with np.load(folder/'grid.npz',allow_pickle=False) as arrays:
            for key in ('height_m','slope_deg','observed','blocked'):setattr(self,key,arrays[key].copy())
        if list(self.observed.shape)!=data['shape'] or any(getattr(self,key).shape!=self.observed.shape
                    for key in ('height_m','slope_deg','blocked')):
            raise ValueError('site grid shape mismatch')
        return self


def load_stl(path: Path, *, units: str, **kwargs) -> SiteMap:
    import open3d as o3d
    path=Path(path).expanduser().resolve()
    if path.suffix.lower()!='.stl' or not path.is_file():raise ValueError('existing STL required')
    mesh=o3d.io.read_triangle_mesh(str(path))
    return SiteMap.from_mesh(mesh,units=units,source=path,**kwargs)
