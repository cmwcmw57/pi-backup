"""정지 구간의 보정 IMU와 카메라 자세로 map별 고정 z-up 기준을 추정한다.

이 단계는 ORB 관성 융합이 아니다. Pi 수신 시각은 정지 구간을 찾는 데만
사용하며 센서 acquisition clock 동기화로 간주하지 않는다.
"""
from collections import deque
import argparse
import hashlib
import json
import math
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from experiments.handheld_mapping.reconstruct import pose_matrix, transform
from experiments.handheld_mapping.recording import ordered_keyframes
from experiments.handheld_mapping.storage import atomic_json

WINDOW_NS = 2_000_000_000
MAX_GAP_NS = 120_000_000


class BodyIncline:
    """중력 기준 차체 자세의 표시 이벤트. 지면 경사나 estimator 전환 기준이 아니다."""

    def __init__(self) -> None:
        self.map_id = None
        self.tilted = False
        self.candidate_since = None
        self.last_stamp = None

    def update(self, map_id: int, stamp_ns: int, world_body: np.ndarray) -> dict:
        matrix = transform(world_body)
        if self.map_id != map_id:
            self.map_id,self.tilted,self.candidate_since,self.last_stamp = map_id,False,None,None
        if self.last_stamp is not None and stamp_ns <= self.last_stamp:
            raise ValueError('nonmonotonic body inclination timestamp')
        if self.last_stamp is not None and stamp_ns-self.last_stamp > 500_000_000:
            self.candidate_since = None
        self.last_stamp = stamp_ns
        tilt = math.degrees(math.acos(float(np.clip(matrix[2,2],-1.,1.))))
        roll = math.degrees(math.atan2(matrix[2,1],matrix[2,2]))
        pitch = math.degrees(math.atan2(-matrix[2,0],math.hypot(matrix[2,1],matrix[2,2])))
        candidate = tilt <= 3. if self.tilted else tilt >= 5.
        event = None
        if candidate:
            if self.candidate_since is None:self.candidate_since=stamp_ns
            if stamp_ns-self.candidate_since >= 1_000_000_000:
                self.tilted = not self.tilted
                self.candidate_since = None
                event = 'body_tilt_enter' if self.tilted else 'body_tilt_exit'
        else:
            self.candidate_since = None
        return {'map_id':map_id,'stamp_ns':stamp_ns,'tilt_deg':tilt,'roll_deg':roll,'pitch_deg':pitch,
                'body_tilted':self.tilted,'event':event,'enter_deg':5.,'exit_deg':3.,'dwell_s':1.,
                'source':'visual_pose_in_stationary_IMU_gravity_frame',
                'terrain_slope':False,'orb_fused':False}


def stationary_reference(samples: list[dict], center_ns: int) -> dict:
    """정지 상태의 specific force를 위쪽 벡터로 사용한다. 칩 축을 다시 뒤집지 않는다."""
    selected = [row for row in samples if abs(row['pi_ns']-center_ns) <= WINDOW_NS//2]
    if len(selected) < 40:
        raise ValueError('중력 기준 대기 · 3초간 정지하세요')
    stamps = np.asarray([row['pi_ns'] for row in selected],dtype=np.int64)
    gaps = np.diff(stamps)
    if (stamps[0] > center_ns-WINDOW_NS//2+MAX_GAP_NS or
            stamps[-1] < center_ns+WINDOW_NS//2-MAX_GAP_NS or
            np.any(gaps <= 0) or np.any(gaps > MAX_GAP_NS) or
            len({row['segment'] for row in selected}) != 1):
        raise ValueError('정지 IMU 구간에 시각 공백 또는 재시작이 있습니다')
    acc = np.asarray([row['acc_g'] for row in selected],dtype=float)
    gyro = np.asarray([row['gyro_dps'] for row in selected],dtype=float)
    if acc.shape != (len(selected),3) or gyro.shape != acc.shape or not np.isfinite([acc,gyro]).all():
        raise ValueError('유한한 보정 IMU 3축 값이 필요합니다')
    mean = acc.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    deviation = float(np.max(np.linalg.norm(acc-mean,axis=1)))
    gyro_max = float(np.max(np.linalg.norm(gyro,axis=1)))
    if not .95 <= norm <= 1.05 or deviation > .025 or gyro_max > 1.5:
        raise ValueError('중력 기준을 잡는 동안 로버를 정지하세요')
    return {'body_up': (mean/norm).tolist(), 'samples':len(selected),
            'start_pi_ns':int(stamps[0]), 'end_pi_ns':int(stamps[-1]),
            'segment':selected[0]['segment'], 'acc_norm_g':norm,
            'max_acc_deviation_g':deviation, 'max_gyro_dps':gyro_max}


def sample_from_record(row: dict) -> dict | None:
    if row.get('acceptance') != 'accepted' or not row.get('calibrated'):
        return None
    clock,values = row['clock'],row['calibrated']
    stamp,segment = clock.get('pi_receipt_monotonic_ns'),clock.get('segment')
    if type(stamp) is not int or stamp < 0 or type(segment) is not int or segment < 0:
        raise ValueError('invalid IMU receipt clock/segment')
    return {'pi_ns':stamp,'segment':segment,'acc_g':values['acc_g'],'gyro_dps':values['gyro_dps']}


def mounting(run: Path) -> np.ndarray:
    """보존된 URDF의 body→left optical 경로와 실측 rectification을 합성한다."""
    run = Path(run)
    graph = {}
    for joint in ElementTree.parse(run/'robot.urdf').getroot().findall('joint'):
        if joint.get('type') != 'fixed':
            continue
        parent,child = joint.find('parent').get('link'),joint.find('child').get('link')
        origin = joint.find('origin')
        xyz = [float(v) for v in origin.get('xyz','0 0 0').split()] if origin is not None else [0.]*3
        rpy = [float(v) for v in origin.get('rpy','0 0 0').split()] if origin is not None else [0.]*3
        if len(xyz) != 3 or len(rpy) != 3 or not np.isfinite([xyz,rpy]).all():
            raise ValueError('invalid fixed URDF origin')
        r,p,y = rpy
        cr,sr,cp,sp,cy,sy = math.cos(r),math.sin(r),math.cos(p),math.sin(p),math.cos(y),math.sin(y)
        edge = np.eye(4)
        edge[:3,:3] = [[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
                      [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]]
        edge[:3,3] = xyz
        graph.setdefault(parent,[]).append((child,edge))
        graph.setdefault(child,[]).append((parent,np.linalg.inv(edge)))
    queue,seen,candidates = deque([('base_link',np.eye(4))]),set(),[]
    while queue:
        frame,matrix = queue.popleft()
        if frame in seen:
            continue
        seen.add(frame)
        if frame in ('camera_left_optical_frame','camera_infra1_optical_frame'):
            candidates.append(matrix)
        for child,edge in graph.get(frame,[]):
            queue.append((child,matrix@edge))
    if len(candidates) != 1:
        raise ValueError('URDF의 좌측 optical 고정 경로가 없거나 모호합니다')
    calibration = json.loads((run/'capture_calibration.json').read_text())
    info = calibration['stereo']['source_camera_info']['left']
    rectification = np.eye(4)
    rectification[:3,:3] = np.asarray(info['r'],dtype=float).reshape(3,3)
    return transform(rectification) @ np.linalg.inv(transform(candidates[0]))


def alignment_matrix(world_left: np.ndarray, left_body: np.ndarray, body_up: list) -> np.ndarray:
    """z=중력 반대, x=기준 로버 전방의 수평 성분. 원점은 기준 카메라 위치다."""
    pose,extrinsic = transform(world_left),transform(left_body)
    up = np.asarray(body_up,dtype=float)
    if up.shape != (3,) or not np.isfinite(up).all() or not .99 < np.linalg.norm(up) < 1.01:
        raise ValueError('unit body up vector required')
    world_body_rotation = pose[:3,:3] @ extrinsic[:3,:3]
    z = world_body_rotation @ up
    z /= np.linalg.norm(z)
    x = world_body_rotation[:,0]
    x = x-z*np.dot(x,z)
    if np.linalg.norm(x) < .1:
        raise ValueError('로버 전방이 거의 수직이어서 수평 기준을 정할 수 없습니다')
    x /= np.linalg.norm(x)
    y = np.cross(z,x)
    result = np.eye(4)
    result[:3,:3] = np.stack((x,y,z))
    result[:3,3] = -result[:3,:3] @ pose[:3,3]
    return transform(result)


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_paths(run: Path, backend: str, map_id: int) -> list[Path]:
    if backend not in ('orb','orb_offline','orbline','orb_inertial') or type(map_id) is not int or map_id < 0:
        raise ValueError('invalid gravity backend/map identity')
    return [run/backend/f'map_{map_id}_keyframes.csv',run/'capture_calibration.json',
            run/'robot.urdf',run/'imu_calibration.json',run/'imu.jsonl']


def load_alignment(run: Path, backend: str, map_id: int) -> dict | None:
    """다른 run/map 또는 pose 재계산 전의 오래된 기준을 재사용하지 않는다."""
    run = Path(run).resolve()
    paths = source_paths(run,backend,map_id)
    path = run/'gravity'/backend/f'map_{map_id}.json'
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    if data.get('run_id') != run.name or data.get('backend') != backend or data.get('map_id') != map_id:
        raise ValueError(f'중력 기준의 RUN/backend/map이 다릅니다: {path}')
    for source in paths:
        if data.get('source_sha256',{}).get(str(source.relative_to(run))) != fingerprint(source):
            raise ValueError(f'중력 기준 생성 후 입력이 변경됐습니다: {source}')
    transform(data['world_from_slam'])
    return data


def left_receipts(run: Path):
    previous = -1
    with (run/'frames.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('stream') != 'left':
                continue
            stamp = row['stamp_ns']
            if type(stamp) is not int or stamp <= previous:
                raise ValueError('invalid left timestamp ordering')
            previous = stamp
            yield row


def derive_alignment(run: Path, backend: str, map_id: int) -> dict:
    """같은 Pi의 수신 시각으로 정지 IMU와 최종 keyframe을 대응시킨다."""
    run = Path(run).resolve()
    paths = source_paths(run,backend,map_id)
    left_body = mounting(run)
    manifest = json.loads((run/'imu_manifest.json').read_text())
    calibration = json.loads((run/'imu_calibration.json').read_text())
    if (manifest.get('run_id') != run.name or manifest.get('frame') != 'rover_body' or
            manifest.get('chip_to_body_actual') != calibration.get('chip_to_body') or
            manifest.get('calibration_sha256') != fingerprint(run/'imu_calibration.json') or
            manifest.get('urdf_sha256') != fingerprint(run/'robot.urdf')):
        raise ValueError('IMU 축·보정·장착 파일의 출처가 일치하지 않습니다')
    frames = left_receipts(run)
    current,previous = next(frames,None),None
    window = deque(maxlen=4096)
    with (run/'imu.jsonl').open() as stream:
        samples = (sample_from_record(json.loads(line)) for line in stream)
        samples = (sample for sample in samples if sample is not None)
        sample = next(samples,None)
        reason = '촬영과 겹치는 2초 정지 IMU 구간이 없습니다'
        for stamp,row in ordered_keyframes(paths[0],map_id):
            while current is not None and current['stamp_ns'] < stamp:
                previous,current = current,next(frames,None)
            frame = min((f for f in (previous,current) if f is not None),
                        key=lambda f:abs(f['stamp_ns']-stamp),default=None)
            if frame is None or abs(frame['stamp_ns']-stamp) > 2_000_000:
                continue
            center = frame.get('pi_receipt_monotonic_ns')
            if type(center) is not int:
                continue
            while sample is not None and sample['pi_ns'] <= center+WINDOW_NS//2:
                window.append(sample)
                sample = next(samples,None)
            while window and window[0]['pi_ns'] < center-WINDOW_NS//2:
                window.popleft()
            try:
                reference = stationary_reference(list(window),center)
            except ValueError as error:
                reason = str(error)
                continue
            matrix = alignment_matrix(pose_matrix(row),left_body,reference['body_up'])
            result = {'schema':1,'run_id':run.name,'backend':backend,'map_id':map_id,
                      'world_from_slam':matrix.tolist(),'coordinate_system':'gravity_z_up',
                      'method':'stationary_accelerometer_and_final_visual_pose',
                      'anchor_stamp_ns':stamp,'anchor_keyframe_id':int(row['keyframe_id']),
                      'stationary_reference':reference,'clock_basis':'same_Pi_receipts_stationary_window_only',
                      'imu_clock_verified':False,'orb_fused':backend == 'orb_inertial','extrinsic_verified':False,
                      'extrinsic_source':'archived_urdf_provisional_with_CameraInfo_rectification',
                      'origin':'anchor camera position; elevation is relative, not height above ground',
                      'accuracy_verified_on_real_camera':False,
                      'source_sha256':{str(path.relative_to(run)):fingerprint(path) for path in paths}}
            output = run/'gravity'/backend/f'map_{map_id}.json'
            output.parent.mkdir(parents=True,exist_ok=True)
            atomic_json(output,result)
            return result
    raise ValueError(reason)


def process_maps(run: Path, backend: str) -> dict:
    """정지 기준을 얻은 map만 처리하고 실패한 map을 명시한다."""
    results,failed = {},{}
    for path in sorted((Path(run)/backend).glob('map_*_keyframes.csv')):
        identity = int(path.name.removeprefix('map_').removesuffix('_keyframes.csv'))
        try:
            results[identity] = derive_alignment(run,backend,identity)
        except (OSError,ValueError,KeyError,ElementTree.ParseError) as error:
            failed[identity] = str(error)
    return {'status':'complete' if results and not failed else 'partial' if results else 'unavailable',
            'maps':results,'failed_maps':failed,'orb_fused':backend == 'orb_inertial'}


def terrain_maps(run: Path, backend: str, gravity: dict, depth_max: float = 3.,
                 output_root: Path | None = None) -> dict:
    """중력 기준을 얻은 map만 별도의 높이·경사·단차 결과로 저장한다."""
    from experiments.handheld_mapping.terrain import GridSpec
    from experiments.handheld_mapping.terrain_replay import replay
    run = Path(run)
    destination = Path(output_root) if output_root is not None else run
    results,failed = {},{}
    for identity,data in gravity['maps'].items():
        identity = int(identity)
        try:
            matrix = transform(data['world_from_slam'])
            positions = np.array([pose_matrix(row)[:3,3] for _stamp,row in
                                  ordered_keyframes(run/backend/f'map_{identity}_keyframes.csv',identity)])
            positions = positions @ matrix[:3,:3].T + matrix[:3,3]
            padding = max(3., depth_max)
            low,high = positions[:,:2].min(axis=0)-padding,positions[:,:2].max(axis=0)+padding
            # 1/2/3m 비교는 같은 격자 범위를 쓰며 더 먼 depth 설정도 포함한다.
            resolution = max(.05,math.ceil(math.sqrt(float(np.prod(high-low))/200_000)*100)/100)
            spec = GridSpec((float(low[0]),float(high[0]),float(low[1]),float(high[1])),resolution)
            output = destination/'terrain'/backend/f'map_{identity}'
            revision = 1
            while output.exists():
                revision += 1
                output = destination/'terrain'/backend/f'map_{identity}_v{revision}'
            results[identity] = replay(run,output,identity,spec,matrix,backend=backend,
                                       depth_max=depth_max, alignment_metadata={key:data[key] for key in
                                        ('method','extrinsic_verified','imu_clock_verified','orb_fused','origin')})
            results[identity]['output_dir'] = str(output)
        except (OSError,ValueError,KeyError) as error:
            failed[identity] = str(error)
    return {'status':'complete' if results and not failed else 'partial' if results else 'unavailable',
            'maps':results,'failed_maps':failed}


def main() -> None:
    parser = argparse.ArgumentParser(description='같은 RUN의 정지 IMU로 중력 기준과 지형 지도를 생성')
    parser.add_argument('run',type=Path)
    parser.add_argument('--backend',choices=('orb','orb_offline','orbline','orb_inertial'),default='orb_offline')
    parser.add_argument('--reconstruct',action='store_true',help='원본 depth와 pose로 중력 기준 TSDF 갱신')
    parser.add_argument('--terrain',action='store_true',help='새 폴더에 높이·경사·단차 지도 생성')
    args = parser.parse_args()
    result = process_maps(args.run,args.backend)
    if result['maps'] and args.reconstruct:
        from experiments.handheld_mapping.reconstruct import reconstruct
        result['dense'] = reconstruct(args.run,backend=args.backend,gravity_alignments=result['maps'])
    if result['maps'] and args.terrain:
        result['terrain'] = terrain_maps(args.run,args.backend,result)
    print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
    if result['status'] != 'complete' or result.get('terrain',{}).get('status','complete') != 'complete':
        parser.exit(1,'일부 map의 중력/지형 기준을 만들지 못했습니다. 원본은 보존됩니다.\n')


if __name__ == '__main__':
    main()
