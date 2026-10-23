# ================================================================
# sand_replay.py
# 설명: RL 시뮬 영상을 순서대로 모래 검출·추적하고 알려진 지도와 조건부 pose를 비교한다.
# 사용법: python -m experiments.site_sim.sand_replay RUN --output NEW_FOLDER
# ================================================================
from __future__ import annotations
import argparse
from itertools import product
import json
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation, Slerp
from experiments.localization.online import OnlineEstimator
from experiments.localization.sand_edges import sand_mask
from experiments.localization.temporal_sand import TemporalSand


def replay_priors(run: Path, stamps: set[int]) -> tuple[dict, float]:
    """실제 기록 IMU/ORB만 재생한다. pose 이벤트의 GT 필드는 사용하지 않는다."""
    config = json.loads((run/'estimator_calibration.json').read_text())
    options = json.loads((run/'run_options.json').read_text())
    scale = options.get('visual_turn_scale')
    schedule = None if scale is None else dict(turn_rotation_scale=scale, turn_position_scale=scale)
    est = OnlineEstimator(np.array(config['start_world_from_base']), config['calibration'],
                          lag_s=config['render_lag_s'], adaptive_visual_noise=config['adaptive_visual_noise'],
                          visual_schedule=schedule)
    result = {}; max_replay_error = 0.
    with (run/'estimator_events.jsonl').open() as stream:
        for line in stream:
            event = json.loads(line); t = event['stamp_s']
            if event['kind'] == 'imu':
                est.on_imu(t, event['dt_s'], np.array(event['gyro']), np.array(event['force']))
            elif event['kind'] == 'frame':
                est.on_frame(t, event['orb_state'], event['map_id'], np.array(event['world_from_camera']))
                stamp = round(t*1e9)
                if stamp in stamps:
                    result[stamp] = dict(pose=est.filter.pose@est.imu_from_base, filter_time=est.filter_time)
            elif 'estimated_position_m' in event:
                # 저장된 추정값과 재생의 일치 검사. GT는 읽지 않는다.
                max_replay_error = max(max_replay_error, float(np.linalg.norm(est.pose(t)[0][:3,3]-event['estimated_position_m'])))
    return result, max_replay_error


class SandProjector:
    """알려진 시뮬 mesh와 입력 pose로 보이는 완만한 표면의 가림 포함 마스크를 계산한다."""

    def __init__(self, mesh_path: Path, calibration: dict, size: tuple[int,int] = (160,120)) -> None:
        mesh = o3d.io.read_triangle_mesh(str(mesh_path)); mesh.compute_triangle_normals()
        normals = np.asarray(mesh.triangle_normals)
        self.sand = normals[:,2] >= np.cos(np.deg2rad(35.))
        grades = np.rad2deg(np.arccos(np.clip(normals[:,2],-1,1)))
        self.labels = np.where(self.sand, np.digitize(grades,[5.,12.5,22.5])+1,0).astype(np.uint8)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        self.size = size; width, height = size
        # 픽셀 중심을 원본 영상 좌표로 돌린다.
        u,v = np.meshgrid((np.arange(width)+.5)*calibration['width']/width-.5,
                          (np.arange(height)+.5)*calibration['height']/height-.5)
        self.rays = np.stack([(u-calibration['cx'])/calibration['fx'],
                              (v-calibration['cy'])/calibration['fy'], np.ones_like(u)],axis=-1).reshape(-1,3)
        self.bc = np.array(calibration['T_base_left_optical'])

    def project(self, pose: np.ndarray) -> np.ndarray:
        """GT를 받는 특별 경로 없이 임의 pose에 같은 지도 예측을 적용한다."""
        return (self.project_labels(pose)>0).astype(np.uint8)

    def project_labels(self, pose: np.ndarray) -> np.ndarray:
        """경사별 탐색 영역을 나눈다. 재료 정답을 나타내는 label이 아니다."""
        camera = pose@self.bc
        directions = self.rays@camera[:3,:3].T
        query = np.c_[np.broadcast_to(camera[:3,3], directions.shape), directions].astype(np.float32)
        hit = self.scene.cast_rays(o3d.core.Tensor(query))
        indices = hit['primitive_ids'].numpy().astype(np.int64); depth = hit['t_hit'].numpy()
        valid = (indices<len(self.sand))&np.isfinite(depth)&(depth>.02)
        result = np.zeros(len(indices),np.uint8); result[valid] = self.labels[indices[valid]]
        return result.reshape(self.size[1],self.size[0])


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """두 이진 영역의 합집합 대비 교집합 비율. 정답 여부는 호출 측에서 구분한다."""
    union = np.count_nonzero((a>0)|(b>0))
    return float(np.count_nonzero((a>0)&(b>0))/union) if union else 1.


def guided_candidate(gray: np.ndarray, baseline: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """이전 실제 영상 실험처럼 지도 경사 구역별로 현재 영상의 밝기를 따로 검사한다."""
    labels = cv2.resize(labels,(gray.shape[1],gray.shape[0]),interpolation=cv2.INTER_NEAREST)
    blurred = cv2.GaussianBlur(gray,(0,0),3); result = baseline.copy()
    for label in range(1,5):
        expected = (labels==label).astype(np.uint8)
        if expected.sum()<100: continue
        roi = cv2.dilate(expected,np.ones((41,41),np.uint8))>0
        values = blurred[roi]
        if np.percentile(values,95)-np.percentile(values,5)<12: continue
        threshold,_ = cv2.threshold(values,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        current = cv2.morphologyEx((roi&(blurred>threshold)).astype(np.uint8),cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
        n,components,stats,_ = cv2.connectedComponentsWithStats(current)
        for index in range(1,n):
            pixels = components==index
            if stats[index,cv2.CC_STAT_AREA]>=100 and (pixels&(expected>0)).sum()>=50:
                result[pixels]=1
    return result


def refine_pose(mask: np.ndarray, prior: np.ndarray, projector: SandProjector) -> tuple[np.ndarray, dict]:
    """현재 검출과 지도 투영 IoU만 비교하는 제한된 shadow 탐색. GT나 이전 정답을 사용하지 않는다."""
    observed = cv2.resize(mask, projector.size, interpolation=cv2.INTER_NEAREST)
    best = prior.copy(); original_score = iou(observed, projector.project(prior)); best_score = original_score
    # prior 근방의 제한된 세 단계 탐색. 픽셀 정합의 관측 가능성/공분산을 검증한 필터 업데이트가 아니다.
    for distance, yaw_deg in [(.10,5.),(.05,2.5),(.025,1.25)]:
        center = best.copy(); candidates = []
        for dx,dy,dyaw in product([0.,-distance,distance], [0.,-distance,distance], [0.,-yaw_deg,yaw_deg]):
            candidate = center.copy(); candidate[:2,3] += [dx,dy]
            candidate[:3,:3] = Rotation.from_euler('z',dyaw,degrees=True).as_matrix()@center[:3,:3]
            candidates.append((iou(observed,projector.project(candidate)),candidate))
        score, pose = max(candidates,key=lambda value:value[0])
        if score > best_score+1e-8:
            best_score, best = score, pose
    accepted = best_score-original_score >= .01
    return (best if accepted else prior.copy()), dict(accepted=accepted, prior_score=original_score,
                best_score=best_score, delta_xy_m=(best[:2,3]-prior[:2,3]).tolist(),
                diagnostic_only=True, observability_validated=False)


def load_truth(run: Path):
    """평가 전용으로 시간 보간한 GT를 반환한다. 검출·추정 입력에는 사용하지 않는다."""
    rows = [json.loads(line) for line in (run/'estimator_events.jsonl').read_text().splitlines()]
    rows = [r for r in rows if r['kind']=='pose']
    times = np.array([r['stamp_s'] for r in rows]); positions = np.array([r['position_m'] for r in rows])
    quaternions = np.array([r['quaternion_wxyz'] for r in rows])[:,[1,2,3,0]]
    rotations = Slerp(times, Rotation.from_quat(quaternions))
    def truth(t: float) -> np.ndarray:
        t = np.clip(t,times[0],times[-1]); result = np.eye(4)
        result[:3,:3] = rotations(t).as_matrix()
        result[:3,3] = [np.interp(t,times,positions[:,i]) for i in range(3)]
        return result
    return truth


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path,required=True); parser.add_argument('--refine-every-s',type=float,default=1.)
    args = parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    paths = sorted((args.run/'stereo/mav0/cam0/data').glob('*.png'),key=lambda p:int(p.stem))
    priors, replay_error = replay_priors(args.run,{int(p.stem) for p in paths})
    if replay_error > 1e-6:
        raise ValueError(f'replayed estimator differs from recorded estimate by {replay_error}m')
    calibration = json.loads((args.run/'stereo/calibration.json').read_text())
    projector = SandProjector(args.run/'mission_bundle/site/mesh.ply',calibration)
    truth = load_truth(args.run); tracker = TemporalSand(); records = []; next_refine = 0.
    picked = set(np.linspace(0,len(paths)-1,6).round().astype(int)); panels = []
    for index,path in enumerate(paths):
        stamp = int(path.stem); t = stamp*1e-9; gray = cv2.imread(str(path),0)
        candidate = sand_mask(gray)
        guided = guided_candidate(gray,candidate,projector.project_labels(priors[stamp]['pose'])) if stamp in priors else candidate
        tracked, info = tracker.update(t,gray,guided)
        row = dict(stamp_s=t,tracking=info,raw_fraction=float(candidate.mean()),guided_fraction=float(guided.mean()),tracked_fraction=float(tracked.mean()))
        if stamp in priors:
            prior = priors[stamp]['pose']; filter_time = priors[stamp]['filter_time']
            if t >= next_refine:
                corrected, fitting = refine_pose(tracked,prior,projector)
                row['fitting'] = fitting; next_refine = t+args.refine_every_s
                # 이 아래에서만 GT를 평가에 사용한다. 지도 mask는 재료 정답이 아닌 기하 대리 지표다.
                gt = truth(filter_time)
                row['evaluation'] = dict(prior_xy_error_cm=float(np.linalg.norm(prior[:2,3]-gt[:2,3])*100),
                    candidate_xy_error_cm=float(np.linalg.norm(corrected[:2,3]-gt[:2,3])*100),
                    raw_map_proxy_iou=iou(cv2.resize(candidate,projector.size,interpolation=cv2.INTER_NEAREST),projector.project(gt)),
                    guided_map_proxy_iou=iou(cv2.resize(guided,projector.size,interpolation=cv2.INTER_NEAREST),projector.project(gt)),
                    tracked_map_proxy_iou=iou(cv2.resize(tracked,projector.size,interpolation=cv2.INTER_NEAREST),projector.project(gt)))
        records.append(row)
        if index in picked:
            panels.append((t,cv2.imread(str(path)),candidate,tracked,row))
        if index%50==0:print(f'{index}/{len(paths)} t={t:.2f}',flush=True)
    evaluations = [r['evaluation'] for r in records if 'evaluation'in r]
    summary = dict(frames=len(paths), tracked_frames=sum(r['tracking']['tracked'] for r in records),
                   reset_or_unlinked_frames=sum(not r['tracking']['tracked'] for r in records),
                   empty_detection_frames=sum(r['tracking']['reason']=='no_current_detection' for r in records),
                   replay_max_position_difference_m=replay_error, tested_pose_frames=len(evaluations),
                   raw_map_proxy_iou_mean=float(np.mean([r['raw_map_proxy_iou'] for r in evaluations])),
                   guided_map_proxy_iou_mean=float(np.mean([r['guided_map_proxy_iou'] for r in evaluations])),
                   tracked_map_proxy_iou_mean=float(np.mean([r['tracked_map_proxy_iou'] for r in evaluations])),
                   prior_xy_mean_cm=float(np.mean([r['prior_xy_error_cm'] for r in evaluations])),
                   candidate_xy_mean_cm=float(np.mean([r['candidate_xy_error_cm'] for r in evaluations])))
    report = dict(simulation_only=True, source_run=str(args.run), runtime_fusion_enabled=False,
                  map_material_proxy_not_semantic_GT=True, render_lag_from_record=True,
                  summary=summary, records=records)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams['font.family']='Noto Sans CJK JP'
    fig,axes=plt.subplots(len(panels),3,figsize=(12,2.8*len(panels)),layout='constrained')
    for row,(t,bgr,candidate,tracked,record) in enumerate(panels):
        for ax in axes[row]:ax.imshow(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY),cmap='gray',vmin=0,vmax=255);ax.axis('off')
        for ax,mask in zip(axes[row,1:],[candidate,tracked]):
            overlay=np.zeros((*mask.shape,4));overlay[:,:,:3]=[.1,1.,.2];overlay[:,:,3]=mask*.35;ax.imshow(overlay)
        axes[row,0].set_title(f'RL 시뮬 원본 / {t:.2f}초')
        axes[row,1].set_title('현재 프레임 모래 후보')
        axes[row,2].set_title('연속 검출 / '+('추적 연결' if record['tracking']['tracked'] else '재획득'))
    fig.suptitle('시뮬 영상을 흑백으로 변환해 모래 검출: 시간 순서 처리 / GT는 검출에 사용하지 않음')
    fig.savefig(args.output/'comparison.png',dpi=130);plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
