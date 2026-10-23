# ================================================================
# overlap.py
# 설명: 유효 depth를 IR로 재투영해 관측 횟수를 색으로 겹치며 가까운 표면만 표시한다.
# 사용법: from experiments.handheld_mapping.overlap import overlap_image
# ================================================================
import numpy as np

from experiments.handheld_mapping.coverage import CoverageGrid
from experiments.handheld_mapping.reconstruct import transform


def overlap_image(original: np.ndarray, depth: np.ndarray, depth_info: dict, left_info: dict,
                  world_left: np.ndarray, grid: CoverageGrid) -> np.ndarray:
    """원본을 변경하지 않고 initial X-Z 관측 횟수의 참고 RGB overlay를 반환한다."""
    pose = transform(world_left).astype(np.float32)
    left_depth = transform(depth_info['T_left_depth']).astype(np.float32)
    kd = np.asarray(depth_info['k'],dtype=np.float32).reshape(3,3)
    kl = np.asarray(left_info['k'],dtype=np.float32).reshape(3,3)
    scale = float(depth_info['depth_scale_m'])
    if (original.dtype != np.uint8 or original.shape != (left_info['height'],left_info['width'])
            or depth.dtype != np.uint16 or depth.shape != (depth_info['height'],depth_info['width'])
            or not np.isfinite([*kd.ravel(),*kl.ravel(),scale]).all()
            or min(kd[0,0],kd[1,1],kl[0,0],kl[1,1]) <= 0 or not 0 < scale < .1
            or any(abs(float(v)) > 1e-8 for v in left_info.get('d',[])+depth_info.get('d',[]))):
        raise ValueError('rectified matching IR/depth calibration required for overlay')
    result = np.repeat(original[:,:,None],3,axis=2)
    rows,cols = np.nonzero((depth*scale >= .25) & (depth*scale <= 3.))
    distance = depth[rows,cols].astype(np.float32)*scale
    points = np.column_stack(((cols-kd[0,2])*distance/kd[0,0],
                              (rows-kd[1,2])*distance/kd[1,1],distance)).astype(np.float32)
    left_points = points @ left_depth[:3,:3].T+left_depth[:3,3]
    left_points = left_points[left_points[:,2] > 0]
    z = left_points[:,2]
    uu = np.rint(kl[0,0]*left_points[:,0]/z+kl[0,2]).astype(int)
    vv = np.rint(kl[1,1]*left_points[:,1]/z+kl[1,2]).astype(int)
    height,width = original.shape
    visible = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
    uu,vv,left_points = uu[visible],vv[visible],left_points[visible]
    pixels = vv*width+uu
    zbuffer = np.full(height*width,np.inf,np.float32)
    np.minimum.at(zbuffer,pixels,left_points[:,2])
    nearest = left_points[:,2] <= zbuffer[pixels]+1e-4
    pixels,left_points = pixels[nearest],left_points[nearest]
    world = left_points @ pose[:3,:3].T+pose[:3,3]
    world = world[:,list(grid.axes)]
    xmin,xmax,zmin,zmax = grid.bounds
    inside = (world[:,0] >= xmin) & (world[:,0] < xmax) & (world[:,1] >= zmin) & (world[:,1] < zmax)
    world,pixels = world[inside],pixels[inside]
    gx = np.floor((world[:,0]-xmin)/grid.resolution).astype(int)
    gz = np.floor((world[:,1]-zmin)/grid.resolution).astype(int)
    counts = grid.counts[gz,gx]
    colors = np.empty((len(pixels),3),np.float32)
    colors[:] = [222,177,63]
    colors[counts >= 3] = [53,175,133]
    flat = result.reshape(-1,3)
    flat[pixels] = np.rint(.65*flat[pixels]+.35*colors).astype(np.uint8)
    return result
