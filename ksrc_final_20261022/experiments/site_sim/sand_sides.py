# ================================================================
# sand_sides.py
# 설명: 흑백 시뮬 영상의 좌우 모래 경계를 스테레오 3D 직선으로 읽고 지도 벽과 비교한다.
# 사용법: python -m experiments.site_sim.sand_sides RUN --output NEW_FOLDER
# ================================================================
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from experiments.localization.sand_edges import sand_mask
from experiments.localization.temporal_sand import TemporalSand
from experiments.localization.wall_range import Camera, load_scene, stereo_depth, fit_lines
from experiments.site_sim.wall_replay import sgbm
from experiments.site_sim.sand_replay import SandProjector, guided_candidate, replay_priors, load_truth


def side_pixels(mask: np.ndarray) -> dict[str,np.ndarray]:
    """각 행의 좌우 윤곽에서 모래 안쪽 3px을 표본화한다. 영상 밖 경계는 제외한다."""
    result: dict[str,list] = {'left':[], 'right':[]}
    for v in range(30,mask.shape[0]-10,2):
        columns = np.flatnonzero(mask[v])
        if len(columns)<20: continue
        if columns[0]>5 and mask[v,columns[0]:columns[0]+4].all():
            result['left'].append([int(columns[0])+3,v])
        if columns[-1]<mask.shape[1]-6 and mask[v,columns[-1]-3:columns[-1]+1].all():
            result['right'].append([int(columns[-1])-3,v])
    return {key:np.array(value,dtype=int).reshape(-1,2) for key,value in result.items()}


def observe_sides(mask: np.ndarray, depth: np.ndarray, camera: Camera, prior: np.ndarray, planes: list[dict]) -> list[dict]:
    """현재 스테레오 측정만으로 측면 직선을 만든다. 대응 선택에는 지도와 추정 자세를 사용한다."""
    yaw = Rotation.from_matrix(prior[:3,:3]).as_euler('xyz')[2]
    leveling = Rotation.from_euler('z',-yaw).as_matrix()@prior[:3,:3]
    result = []
    for side,pixels in side_pixels(mask).items():
        points = []; kept = []
        for u,v in pixels:
            patch = depth[v-2:v+3,u-2:u+3]; sand = mask[v-2:v+3,u-2:u+3]>0
            values = patch[sand&np.isfinite(patch)&(patch>.2)&(patch<3.)]
            if len(values)<5: continue
            z = float(np.median(values))
            optical = np.array([(u-camera.cx)/camera.fx,(v-camera.cy)/camera.fy,1.])*z
            base = camera.base_from_camera[:3,:3]@optical+camera.base_from_camera[:3,3]
            points.append((leveling@base)[:2]); kept.append([int(u),int(v)])
        if len(points)<10: continue
        points = np.array(points)
        for line in fit_lines(points,np.random.default_rng(0)):
            # 진행 방향에 가까운 직선만 측면 후보로 분류한다.
            if abs(line['n'][1])<np.cos(np.deg2rad(45.)): continue
            center = (line['lo']+line['hi'])/2*np.array([-line['n'][1],line['n'][0]])+line['rho']*line['n']
            world_center = Rotation.from_euler('z',yaw).apply([*center,0.])[:2]+prior[:2,3]
            candidates = []
            for index,plane in enumerate(planes):
                sign = 1. if plane['d']-plane['n']@prior[:2,3]>0 else -1.
                normal = sign*plane['n']; rho = sign*(plane['d']-plane['n']@prior[:2,3])
                angle = float((line['alpha']-(np.arctan2(normal[1],normal[0])-yaw)+np.pi)%(2*np.pi)-np.pi)
                along = np.array([-plane['n'][1],plane['n'][0]])@world_center
                if abs(angle)<np.deg2rad(20.) and abs(line['rho']-rho)<.3 and plane['lo']-.15<=along<=plane['hi']+.15:
                    candidates.append((abs(line['rho']-rho)+.2*abs(angle),index,normal))
            if not candidates: continue
            _,index,normal = min(candidates,key=lambda x:x[0])
            yaw_measured = np.arctan2(normal[1],normal[0])-line['alpha']
            yaw_measured = yaw+float((yaw_measured-yaw+np.pi)%(2*np.pi)-np.pi)
            along_points=points@np.array([-line['n'][1],line['n'][0]])
            inliers=(abs(points@line['n']-line['rho'])<.03)&(along_points>=line['lo']-.01)&(along_points<=line['hi']+.01)
            physical_side='left' if line['n'][1]>0 else 'right'
            inlier_pixels=np.array(kept)[inliers]
            image_span=float(np.linalg.norm(np.ptp(inlier_pixels,axis=0))) if len(inlier_pixels) else 0.
            result.append(dict(side=physical_side,image_boundary_source=side,plane_index=index,normal_world=normal.tolist(),
                               rho_m=float(line['rho']), yaw_measured_rad=float(yaw_measured),
                               span_m=float(line['hi']-line['lo']),support_points=line['count'],
                               pixels=inlier_pixels.tolist(), normal_body=line['n'].tolist(), image_span_px=image_span,
                               sufficient_image_support=bool(line['count']>=30 and image_span>=60.)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument('run',type=Path); parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    paths = sorted((args.run/'stereo/mav0/cam0/data').glob('*.png'),key=lambda p:int(p.stem))
    priors,error = replay_priors(args.run,{int(p.stem) for p in paths})
    if error>1e-6: raise ValueError('estimator replay differs from recorded poses')
    calibration = json.loads((args.run/'stereo/calibration.json').read_text()); camera = Camera.from_sim(calibration)
    projector = SandProjector(args.run/'mission_bundle/site/mesh.ply',calibration)
    _,planes,_ = load_scene(); truth = load_truth(args.run)
    tracker = TemporalSand(); matcher = sgbm(); records = []; next_check = 0.; previews = []
    for path in paths:
        stamp = int(path.stem); t=stamp*1e-9; gray=cv2.imread(str(path),0)
        if stamp not in priors: continue
        prior=priors[stamp]['pose']; candidate=guided_candidate(gray,sand_mask(gray),projector.project_labels(prior))
        mask,tracking=tracker.update(t,gray,candidate)
        if t<next_check: continue
        next_check=t+1.
        right=args.run/'stereo/mav0/cam1/data'/path.name
        if not right.is_file(): raise FileNotFoundError(right)
        depth=stereo_depth(matcher,gray,cv2.imread(str(right),0),camera)
        observations=observe_sides(mask,depth,camera,prior,planes)
        # 여기부터 GT는 관측값 평가에만 사용한다.
        gt=truth(priors[stamp]['filter_time']); yaw_gt=Rotation.from_matrix(gt[:3,:3]).as_euler('xyz')[2]
        for observation in observations:
            plane=planes[observation['plane_index']]; n=np.array(observation['normal_world'])
            sign=float(n@plane['n']); expected=sign*(plane['d']-plane['n']@gt[:2,3])
            observation['evaluation']=dict(range_error_cm=float((observation['rho_m']-expected)*100),
                yaw_error_deg=float(np.rad2deg((observation['yaw_measured_rad']-yaw_gt+np.pi)%(2*np.pi)-np.pi)))
        records.append(dict(stamp_s=t,tracking=tracking,observations=observations))
        if observations and len(previews)<6 and (not previews or t-previews[-1][0]>4.):
            previews.append((t,gray,mask,observations))
    measurements=[o for r in records for o in r['observations']]
    summary=dict(evaluated_frames=len(records),frames_with_side_measurement=sum(bool(r['observations']) for r in records),
                 measurements=len(measurements),by_side={})
    for side in ['left','right']:
        subset=[o for o in measurements if o['side']==side]
        if not subset: continue
        distance=[abs(o['evaluation']['range_error_cm']) for o in subset]
        angle=[abs(o['evaluation']['yaw_error_deg']) for o in subset]
        summary['by_side'][side]=dict(count=len(subset),range_abs_error_median_cm=float(np.median(distance)),
                                     range_abs_error_p90_cm=float(np.percentile(distance,90)),
                                     yaw_abs_error_median_deg=float(np.median(angle)),yaw_abs_error_p90_deg=float(np.percentile(angle,90)))
    qualified=[o for o in measurements if o['sufficient_image_support']]
    summary['image_support_subset']={'criterion':'at least 30 depth points sampled every 2 rows and at least 60px image span; exploratory, not independent holdout', 'by_side':{}}
    for side in ['left','right']:
        subset=[o for o in qualified if o['side']==side]
        if not subset:continue
        distance=[abs(o['evaluation']['range_error_cm']) for o in subset]
        angle=[abs(o['evaluation']['yaw_error_deg']) for o in subset]
        summary['image_support_subset']['by_side'][side]=dict(count=len(subset),range_abs_error_median_cm=float(np.median(distance)),range_abs_error_p90_cm=float(np.percentile(distance,90)),yaw_abs_error_median_deg=float(np.median(angle)),yaw_abs_error_p90_deg=float(np.percentile(angle,90)))
    report=dict(simulation_only=True,grayscale_input=True,side_reference='rover horizontal body frame; image boundary source stored separately',depth_source='SGBM of current left/right grayscale images; no scale fitting',
                correspondence_source='prior pose and known fence planes, not GT',pose_updates_enabled=False,
                summary=summary,records=records)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    if not previews:
        print(json.dumps(summary,indent=2),flush=True)
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams['font.family']='Noto Sans CJK JP'
    fig,axes=plt.subplots(len(previews),2,figsize=(10,3.2*len(previews)),layout='constrained',squeeze=False)
    for row,(t,gray,mask,observations) in enumerate(previews):
        for ax in axes[row]:ax.imshow(gray,cmap='gray',vmin=0,vmax=255);ax.axis('off')
        overlay=np.zeros((*mask.shape,4));overlay[:,:,:3]=[.1,1.,.2];overlay[:,:,3]=mask*.25;axes[row,0].imshow(overlay)
        labels=[]
        for side,color in [('left','#00ddff'),('right','#ffbb44')]:
            selected=[o for o in observations if o['side']==side]
            if not selected:continue
            best=max(selected,key=lambda o:o['support_points']);pixels=np.array(best['pixels'])
            axes[row,1].scatter(pixels[:,0],pixels[:,1],s=3,color=color)
            labels.append(f"{'왼쪽' if side=='left' else '오른쪽'} 거리 {best['rho_m']:.2f}m")
        axes[row,0].set_title(f'흑백 모래 후보 / {t:.2f}초');axes[row,1].set_title(' / '.join(labels))
    fig.suptitle('흑백 시뮬 영상의 좌우 모래 경계: 현재 스테레오로 거리·방향 측정\n표시는 측면 후보의 깊이 표본이며 pose 보정은 아직 연결하지 않음')
    fig.savefig(args.output/'sides.png',dpi=130);plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
