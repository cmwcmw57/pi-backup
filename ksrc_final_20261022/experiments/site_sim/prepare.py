# ================================================================
# prepare.py
# 설명: 기존 경기장 STL·모래 높이를 meter 지도와 검증된 세 경로 임무로 준비한다.
# 사용법: python -m experiments.site_sim.prepare --stl arena.stl --sand sand.npz --output NEW_FOLDER
# ================================================================
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from experiments.handheld_mapping.coverage_path import astar
from experiments.handheld_mapping.site_map import SiteMap, file_hash
from experiments.handheld_mapping.site_mission import SiteMission, dense_path
from experiments.handheld_mapping.storage import atomic_json

if TYPE_CHECKING:
    import open3d as o3d


def metric_arena_mesh(stl: Path, sand: Path) -> o3d.geometry.TriangleMesh:
    """기존 exporter의 STL×0.1과 XY 높이 배열을 같은 meter 형상으로 변환한다."""
    import open3d as o3d
    arena=o3d.io.read_triangle_mesh(str(stl))
    if not len(arena.triangles):raise ValueError('nonempty arena STL required')
    arena.scale(.1,center=[0,0,0])
    with np.load(sand,allow_pickle=False) as data:
        height=np.asarray(data['height'],float).T
        x0,y0,cell=(float(data[k]) for k in ('x0','y0','cell'))
    if height.ndim!=2 or min(height.shape)<2 or not math.isfinite(cell) or cell<=0:
        raise ValueError('finite XY height grid required')
    stride=max(1,int(.025/cell))
    # Preserve holes/walls between retained samples: no face spans any invalid source cell.
    iy=np.arange(0,height.shape[0],stride);ix=np.arange(0,height.shape[1],stride)
    xx,yy=np.meshgrid(x0+(ix+.5)*cell,y0+(iy+.5)*cell)
    hh=height[np.ix_(iy,ix)]
    points=np.c_[xx.ravel(),yy.ravel(),hh.ravel()]
    ids=np.arange(hh.size).reshape(hh.shape)
    faces=[]
    for row in range(len(iy)-1):
        for col in range(len(ix)-1):
            patch=height[iy[row]:iy[row+1]+1,ix[col]:ix[col+1]+1]
            if not np.isfinite(patch).all() or np.ptp(patch)>.02:continue
            a,b,c,d=ids[row,col],ids[row,col+1],ids[row+1,col+1],ids[row+1,col]
            faces.extend(((a,b,c),(a,c,d)))
    if faces:
        faces=np.asarray(faces,int);used,inverse=np.unique(faces,return_inverse=True)
        surface=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(points[used]),
            o3d.utility.Vector3iVector(inverse.reshape(-1,3)))
        arena+=surface
    arena.compute_triangle_normals()
    return arena


def route_through(site: SiteMap, anchors: list[list[float]], max_slope_deg: float) -> np.ndarray:
    """각 필수 anchor를 보존하며 안전 A* 경로를 검증된 선분으로만 단순화한다."""
    anchors=np.asarray(anchors,float)
    if anchors.ndim!=2 or anchors.shape[1]!=2 or len(anchors)<2 or not np.isfinite(anchors).all():
        raise ValueError('finite XY anchors required')
    safe=site.safe_mask(max_slope_deg);cells=site.cells(anchors)
    if np.any(cells<0) or np.any(cells[:,0]>=safe.shape[1]) or np.any(cells[:,1]>=safe.shape[0]):
        raise ValueError('anchor outside map')
    if not safe[cells[:,1],cells[:,0]].all():raise ValueError('unsafe anchor')
    result=[anchors[0]]
    for first,last,a,b in zip(anchors[:-1],anchors[1:],cells[:-1],cells[1:]):
        path=astar(safe,tuple(a[::-1]),tuple(b[::-1]))
        if path is None:raise ValueError('route anchor is unreachable')
        xy=site.origin+(np.asarray(path)[:,::-1]+.5)*site.resolution
        joined=np.vstack((first,xy,last));i=0
        while i<len(joined)-1:
            chosen=None
            for j in range(len(joined)-1,i,-1):
                if site.validate_path([joined[i],joined[j]],max_slope_deg)['accepted']:
                    chosen=j;break
            if chosen is None:raise ValueError('A* cell segment lacks clearance')
            if np.linalg.norm(result[-1]-joined[chosen])>1e-8:result.append(joined[chosen])
            i=chosen
    result=np.asarray(result)
    if not site.validate_path(result,max_slope_deg)['accepted']:raise ValueError('route clearance failed')
    return result


def prepare(stl: Path, sand: Path, output: Path) -> SiteMission:
    """서버 경기장 세 경사로와 공통 정상의 재현 가능한 임무 bundle을 저장한다."""
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    mesh=metric_arena_mesh(stl,sand)
    # 0.1675 m pivot radius + wheel sweep <=0.06 m, 0.02 m clearance + 0.05 m arrival margin.
    site=SiteMap.from_mesh(mesh,units='m',resolution=.025,robot_radius=.30,surface_mode='top')
    start=[2.9,1.2,0.];junction=[3.7,.35];summit=[5.35,.4]
    routes={
        '30':route_through(site,[junction,[3.8,.4],[4.5,.4],summit],35),
        '15':route_through(site,[junction,[3.7,1.2],[4.7,1.2],[6.15,1.2],[6.15,.4],summit],20),
        '10':route_through(site,[junction,[3.7,1.2],[2.9,1.2],[2.9,-1.25],[4.7,-1.25],
                                 [6.15,-1.25],[6.15,.4],summit],20)}
    max_length=max(np.linalg.norm(np.diff(p,axis=0),axis=1).sum() for p in routes.values())
    mission=SiteMission(site,start,junction,summit,routes,{
        'max_speed_m_s':.08,'arrival_tolerance_m':.05,'max_attempt_s':float(max(120.,max_length/.08*2)),
        'stuck_window_s':8.,'min_progress_m':.025},site_name='demo')
    if not mission.validate()['accepted']:raise ValueError(json.dumps(mission.validate()))
    mission.save(output)
    atomic_json(output/'simulation.json',{'simulation_only':True,'pose_source':'sim_ground_truth',
        'T_site_sim_world':np.eye(4).tolist(),'source_stl':str(Path(stl).resolve()),'stl_scale_to_m':.1,
        'source_stl_sha256':file_hash(stl),'source_sand':str(Path(sand).resolve()),
        'source_sand_sha256':file_hash(sand),'source_sand_axes':'x,y','site_axes':'row=y,col=x',
        'footprint_radius_m':.23,'clearance_m':.02,'arrival_margin_m':.05,
        'surface_model':'fixed baked sand mesh; no deformable soil'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,7))
    extent=[site.origin[0],site.origin[0]+site.observed.shape[1]*site.resolution,
            site.origin[1],site.origin[1]+site.observed.shape[0]*site.resolution]
    im=ax.imshow(np.where(site.observed,site.height_m,np.nan),origin='lower',extent=extent,cmap='terrain')
    fig.colorbar(im,ax=ax,label='Surface height (m)')
    for key,p in routes.items():ax.plot(p[:,0],p[:,1],'-o',label=key+' deg route',ms=3)
    ax.plot(start[0],start[1],'ks',label='start');ax.set(xlabel='X (m)',ylabel='Y (m)',aspect='equal')
    ax.legend();fig.tight_layout();fig.savefig(output/'preview.png',dpi=130);plt.close(fig)
    report={}
    for key,p in routes.items():
        cells=site.cells(dense_path(p,site.resolution/2))
        report[key]={'length_m':float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum()),
            'max_slope_deg':float(site.slope_deg[cells[:,1],cells[:,0]].max())}
    atomic_json(output/'route_report.json',report)
    return mission


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--stl',type=Path,required=True)
    parser.add_argument('--sand',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();mission=prepare(args.stl,args.sand,args.output)
    print(json.dumps({'mission':str(args.output),'validation':mission.validate()}))


if __name__=='__main__':main()
