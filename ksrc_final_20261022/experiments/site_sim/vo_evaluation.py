# ================================================================
# vo_evaluation.py
# 설명: 저장 VO pose와 별도 GT를 map별 단일 초기 SE(3) 정합으로 비교한다. scale fitting은 하지 않는다.
# 사용법: .venv-mapping/bin/python -m experiments.site_sim.vo_evaluation VO_RUN STEREO_SEQUENCE
# ================================================================
from pathlib import Path
import argparse
import collections
import csv
import json
import numpy as np
from scipy.spatial.transform import Rotation


def align_trajectory(estimated: np.ndarray, first_gt: np.ndarray) -> np.ndarray:
    """최초 pose 한 개로 정한 강체 변환을 전체 궤적에 적용한다. GT 전체 fitting 금지."""
    alignment=first_gt@np.linalg.inv(estimated[0])
    return alignment[None]@estimated


def evaluate(run: Path, sequence: Path) -> dict:
    """tracking 손실을 분모에 포함하고 서로 다른 map 원점을 하나로 합치지 않는다."""
    calibration=json.loads((sequence/'calibration.json').read_text());base_camera=np.asarray(calibration['T_base_left_optical'])
    gt={int(r['stamp_ns']):r for r in map(json.loads,(sequence/'gt.jsonl').read_text().splitlines())}
    rows=list(csv.DictReader((run/'live_poses.csv').open()))
    if not rows:raise ValueError('no VO rows')
    by_map=collections.defaultdict(list)
    for r in rows:
        if int(r['state'])==2 and int(r['map_id'])>=0:
            pose=np.array([float(r[f'm{i}{j}']) for i in range(4) for j in range(4)]).reshape(4,4)
            if not np.isfinite(pose).all():raise ValueError('nonfinite valid pose')
            if int(r['stamp_ns']) not in gt:raise ValueError('VO timestamp absent from GT')
            by_map[r['map_id']].append((r,pose))
    result={'frames':len(rows),'tracking_OK_frames':sum(int(r['state'])==2 for r in rows),
            'tracking_OK_fraction':sum(int(r['state'])==2 for r in rows)/len(rows),
            'state_counts':dict(collections.Counter(r['state'] for r in rows)),
            'tracked_map_count':len(by_map),'alignment':'one initial SE(3) per map; no scale fitting; gaps excluded from pose error only',
            'GT_used_for_estimation':False,'maps':{}}
    process=np.asarray([float(r['process_ms']) for r in rows])
    result['processing_median_ms']=float(np.median(process));result['processing_p95_ms']=float(np.percentile(process,95))
    lost=longest=0
    for r in rows:
        lost=lost+1 if int(r['state'])!=2 else 0
        longest=max(longest,lost)
    result['longest_nontracking_run_frames']=longest
    curves={};error_rows=[]
    for mid,items in by_map.items():
        stamps=[int(r['stamp_ns']) for r,_ in items];est=np.array([p for _,p in items]);truth=[]
        for stamp in stamps:
            g=gt[stamp];q=g['quaternion_wxyz'];p=np.eye(4)
            p[:3,:3]=Rotation.from_quat([*q[1:],q[0]]).as_matrix();p[:3,3]=g['position_m'];truth.append(p)
        truth=np.asarray(truth);aligned=align_trajectory(est,truth[0]@base_camera)@np.linalg.inv(base_camera)
        errors=np.linalg.norm(aligned[:,:3,3]-truth[:,:3,3],axis=1)
        angle=Rotation.from_matrix(np.transpose(truth[:,:3,:3],(0,2,1))@aligned[:,:3,:3]).magnitude()*180/np.pi
        duration=(stamps[-1]-stamps[0])*1e-9;length=np.linalg.norm(np.diff(truth[:,:3,3],axis=0),axis=1).sum()
        result['maps'][mid]={'frames':len(items),'start_s':stamps[0]*1e-9,'end_s':stamps[-1]*1e-9,'duration_s':duration,
            'GT_path_length_m':float(length),'position_RMSE_m':float(np.sqrt(np.mean(errors**2))),
            'position_max_m':float(errors.max()),'position_endpoint_m':float(errors[-1]),'orientation_RMSE_deg':float(np.sqrt(np.mean(angle**2)))}
        curves[mid]=(truth,aligned)
        error_rows.extend({'stamp_ns':t,'map_id':mid,'position_error_m':float(e),'orientation_error_deg':float(a)} for t,e,a in zip(stamps,errors,angle))
    result['dominant_map']=max(result['maps'],key=lambda m:result['maps'][m]['frames']) if by_map else None
    with (run/'pose_errors.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['stamp_ns','map_id','position_error_m','orientation_error_deg']);writer.writeheader();writer.writerows(error_rows)
    (run/'evaluation.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    if curves:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        mid=result['dominant_map'];truth,aligned=curves[mid];fig,ax=plt.subplots(figsize=(7,5))
        ax.plot(truth[:,0,3],truth[:,1,3],label='GT');ax.plot(aligned[:,0,3],aligned[:,1,3],label='ORB stereo (one initial SE3)')
        ax.set(xlabel='Site X (m)',ylabel='Site Y (m)',title=f'High contrast / dominant map {mid} only',aspect='equal');ax.legend();ax.grid(alpha=.3)
        fig.tight_layout();fig.savefig(run/'trajectory_comparison.png',dpi=150);plt.close(fig)
    return result


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('sequence',type=Path);args=p.parse_args()
    print(json.dumps(evaluate(args.run,args.sequence),indent=2))


if __name__=='__main__':main()
