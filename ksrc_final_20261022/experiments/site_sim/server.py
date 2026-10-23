# ================================================================
# server.py
# 설명: 서버 GPU1 임무 run의 제한된 upload·launch·status·tunnel·stop 명령을 실행한다.
# 사용법: bash tools/run_site_sim_server.sh upload LABEL MISSION_FOLDER
# ================================================================
from __future__ import annotations

import argparse
import hashlib
import io
import math
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tarfile
import tempfile
from xml.etree import ElementTree

ROOT='/home/sumee/gyusa/runs/site_mission'
ARENA='/home/sumee/gyusa/arena_sand_v15_m5.usd'
IMAGE='nvcr.io/nvidia/isaac-sim:6.1.0'
ORB_IMAGE='rover-orbslam3-jazzy:4452a3c4'
STREAM_ORB='/home/sumee/gyusa/runs/stream_orb'  # experiments/site_sim/stream_orb.cpp를 ORB 이미지에서 빌드한 실행 파일
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=15','hpi-lab']
REPO=Path(__file__).resolve().parents[2]


def remote(command: str, *, capture: bool=False, stdin=None) -> subprocess.CompletedProcess:
    return subprocess.run([*SSH,command],check=True,text=True,capture_output=capture,stdin=stdin)


SIM_ORB_SETTINGS={'ORBextractor.nFeatures':2000,'ORBextractor.iniThFAST':12,'ORBextractor.minThFAST':3}


def sim_orb_yaml(text: str) -> str:
    """시뮬 모래 텍스처의 저대비 구간(경사)에서 추적을 덜 잃도록 특징점 수와 FAST 임계값만 바꾼다. 실기 설정에는 쓰지 않는다."""
    lines=text.splitlines();seen=set()
    for index,line in enumerate(lines):
        key=line.split(':',1)[0].strip()
        if key in SIM_ORB_SETTINGS:lines[index]=f'{key}: {SIM_ORB_SETTINGS[key]}';seen.add(key)
    if seen!=SIM_ORB_SETTINGS.keys():raise ValueError('ORB yaml lacks '+', '.join(sorted(SIM_ORB_SETTINGS.keys()-seen)))
    return '\n'.join(lines)+'\n'


def upload(label: str, mission: Path, assets: Path, texture: Path | None=None, ksrc: Path | None=None) -> None:
    """필요한 소스와 asset만 새 run에 기록하고 hash manifest를 동봉한다."""
    if not (mission/'mission.json').is_file():raise ValueError('valid mission bundle required')
    from experiments.handheld_mapping.site_mission import SiteMission
    SiteMission.load(mission,load_mesh=False)
    urdf=assets/'rover_portable.urdf';geometry=assets/'config/swerve_geometry.json'
    if not urdf.is_file() or not geometry.is_file():raise ValueError('portable rover assets required')
    referenced={node.get('filename') for node in ElementTree.parse(urdf).findall('.//mesh')}
    for name in referenced:
        path=(assets/name).resolve()
        if not path.is_relative_to(assets.resolve()) or not path.is_file():raise ValueError('missing/outside URDF mesh')
    hashes={};run=f'{ROOT}/{label}'
    with tempfile.TemporaryFile() as archive:
        with tarfile.open(fileobj=archive,mode='w:gz') as tar:
            for package in ('handheld_mapping','site_sim','orbslam3_ros2','orbslam3_offline','localization'):
                for path in sorted((REPO/'experiments'/package).rglob('*')):
                    if '__pycache__' in path.parts:continue
                    if path.suffix not in ('.py','.html') and path.name not in ('gyusa_regions.json','gyusa_region_portals.json'):continue
                    relative=str(path.relative_to(REPO));tar.add(path,arcname=relative)
                    hashes[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
            tar.add(mission,arcname='mission')
            tar.add(urdf,arcname='rover/rover_portable.urdf');tar.add(geometry,arcname='rover/swerve_geometry.json')
            if (assets/'orb_stereo.yaml').is_file():
                tuned=sim_orb_yaml((assets/'orb_stereo.yaml').read_text()).encode()
                info=tarfile.TarInfo('rover/orb.yaml');info.size=len(tuned);tar.addfile(info,io.BytesIO(tuned))
                hashes['rover/orb.yaml']=hashlib.sha256(tuned).hexdigest()
            if texture is not None:
                if not texture.is_file():raise ValueError('sand texture file not found')
                tar.add(texture,arcname='rover/sand_texture.jpg');hashes['rover/sand_texture.jpg']=hashlib.sha256(texture.read_bytes()).hexdigest()
            for name in sorted(referenced):tar.add(assets/name,arcname='rover/'+name)
            if ksrc is not None:
                ksrc=Path(ksrc).resolve()
                for pattern in ('sim/ksrc_common.py','sim/build.sh','pi/common/*.py','pi/policy/student_runtime.py','pi/policy/res8_*','firmware/common/*.c','firmware/common/*.h',
                                'sim/isaac/ksrc_rover/source/ksrc_rover/utils/contract.py'):
                    for path in sorted(ksrc.glob(pattern)):
                        if path.is_file():
                            relative='ksrc/'+str(path.relative_to(ksrc));tar.add(path,arcname=relative);hashes[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
            commit=subprocess.run(['git','rev-parse','HEAD'],cwd=REPO,text=True,capture_output=True,check=True).stdout.strip()
            body=json.dumps({'code_commit':commit,'source_sha256':hashes,'source_assets':str(assets.resolve()),
                'stl_and_sand_provenance':json.loads((mission/'simulation.json').read_text())}).encode()
            info=tarfile.TarInfo('source_manifest.json');info.size=len(body);tar.addfile(info,io.BytesIO(body))
        archive.seek(0)
        remote(f'mkdir -p {shlex.quote(ROOT)} && mkdir {shlex.quote(run)} && tar -xzf - -C {shlex.quote(run)}',stdin=archive)
    if ksrc is not None:   # 서버 호스트의 gcc로 펌웨어 C 기구학 라이브러리를 만든다(컨테이너는 같은 폴더를 마운트한다)
        remote(f'cd {shlex.quote(run)}/ksrc/sim && bash build.sh')
    print('uploaded:',run)


def fetch_commands(label: str, destination: Path, stereo_names: list[str], every: int=3) -> list[list[str]]:
    """기록을 받는 rsync 명령들: 무거운 stereo 폴더는 빼고 `every`장마다 한 쌍만 따로 받아 뷰어 재생에 쓴다."""
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    source=f'hpi-lab:/home/sumee/gyusa/runs/site_mission/{label}/output/'
    options=['rsync','-a','--partial','-e','ssh -o ConnectTimeout=15']
    listing=destination/'.stereo_files.txt'
    listing.write_text(''.join(f'mav0/cam0/data/{n}\nmav0/cam1/data/{n}\n' for n in stereo_names[::every]))
    return [[*options,'--exclude','stereo','--exclude','front_frames','--exclude','snapshots',source,str(destination)+'/'],
            [*options,f'--files-from={listing}',source+'stereo/',str(destination/'stereo')+'/']]


def fetch(label: str, destination: Path) -> None:
    """서버 기록을 받아 `viewer`로 바로 열 수 있게 한다(stereo는 3장마다 1쌍)."""
    destination=Path(destination);first,second=fetch_commands(label,destination,[],1)
    subprocess.run(first,check=True)
    subprocess.run(['rsync','-a','-e','ssh -o ConnectTimeout=15',f'hpi-lab:/home/sumee/gyusa/runs/site_mission/{label}/mission/',str(destination/'mission_bundle')+'/'],check=False)   # 벽 거리 재생용
    listing=subprocess.run([*SSH,f'ls /home/sumee/gyusa/runs/site_mission/{label}/output/stereo/mav0/cam0/data'],capture_output=True,text=True)
    if listing.returncode==0 and listing.stdout.split():
        _,second=fetch_commands(label,destination,listing.stdout.split(),3);subprocess.run(second,check=True)
    print(f'fetched: {destination}\nview: python3 -m experiments.site_sim.viewer --output {destination} --port 8790')


def start(label: str, fault: bool, sim: float, wall: float, port: int, gpu: int=1, save_front_s: float=0., stereo_hz: float=0., start_hold_s: float=0., estimator: bool=False, adaptive_visual_noise: bool=False, save_overview_s: float=0.,
          wheel_gain_left: float=1., wheel_gain_right: float=1., slip_start_s: float=0., slip_end_s: float | None=None, stanley_gain: float=0.,
          rl_policy: str | None=None, imu_realism: bool=False, imu_seed: int=0, control_dt: float | None=None, extra_args: list[str] | None=None) -> None:
    """지정한 유휴 GPU에서만 새 물리 run과 독립 loopback viewer를 시작한다."""
    from experiments.site_sim.run import validate_limits
    validate_limits(sim,wall)
    if not math.isfinite(stereo_hz) or not 0<=stereo_hz<=30:raise ValueError("stereo_hz must be in [0,30]")
    run=f'{ROOT}/{label}';name='rover-site-'+label
    remote(f'test -d {shlex.quote(run)} && test ! -e {shlex.quote(run+"/output")}')
    if gpu not in (0,1):raise ValueError('GPU index must be 0 or 1')
    gpu_before=remote(f'nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits -i {gpu}',capture=True).stdout.strip()
    free,util=(int(v.strip()) for v in gpu_before.split(','))
    if free<12000 or util>20:raise RuntimeError(f'GPU{gpu} free memory {free}MiB / utilization {util}%: refusing launch')
    # Fail before starting either process if the requested observer port is unavailable.
    probe=f'import socket; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); s.bind(("127.0.0.1",{port})); s.close()'
    remote(shlex.join(['python3','-c',probe]))
    args=['/isaac-sim/kit/python/bin/python3','-m','experiments.site_sim.run','--mission','/workspace/mission',
        '--urdf','/workspace/rover/rover_portable.urdf','--geometry','/workspace/rover/swerve_geometry.json',
        '--scene-usd','/arena/arena.usd','--output','/workspace/output',
        '--max-sim-s',str(sim),'--max-wall-s',str(wall)]
    if fault:args.append('--fault-injection')
    if save_front_s>0:args+=['--save-front-s',str(save_front_s)]
    if save_overview_s>0:args+=['--save-overview-s',str(save_overview_s)]
    if stanley_gain>0:args+=['--stanley-gain',str(stanley_gain)]
    if wheel_gain_left!=1. or wheel_gain_right!=1.:
        args+=['--wheel-gain-left',str(wheel_gain_left),'--wheel-gain-right',str(wheel_gain_right),'--slip-start-s',str(slip_start_s)]
        if slip_end_s is not None:args+=['--slip-end-s',str(slip_end_s)]
    if stereo_hz>0:args+=['--stereo-hz',str(stereo_hz)]
    if start_hold_s>0:args+=['--start-hold-s',str(start_hold_s)]
    if estimator:
        if stereo_hz<=0:raise ValueError('estimator needs stereo capture (--stereo-hz)')
        args+=['--estimator']
    if adaptive_visual_noise:
        if not estimator:raise ValueError('adaptive visual noise needs --estimator')
        args+=['--adaptive-visual-noise']
    if imu_realism:
        if not estimator:raise ValueError('imu realism needs --estimator')
        args+=['--imu-realism','--imu-seed',str(imu_seed)]
    if rl_policy:
        if remote(f'test -f {shlex.quote(run+"/ksrc/sim/libksrc_common.so")} && echo yes || echo no',capture=True).stdout.strip()!='yes':
            raise ValueError('--rl-policy needs an upload with --ksrc (built libksrc_common.so)')
        if control_dt is None or abs(control_dt-.02)>1e-9:raise ValueError('--rl-policy needs --control-dt 0.02 (policy trained at 50 Hz)')
        args+=['--rl-policy','/workspace/ksrc/pi/policy/'+rl_policy,'--ksrc-root','/workspace/ksrc']
    if control_dt is not None:args+=['--control-dt',str(control_dt)]
    args+=list(extra_args or [])
    if remote(f'test -f {shlex.quote(run+"/rover/sand_texture.jpg")} && echo yes || echo no',capture=True).stdout.strip()=='yes':
        args+=['--sand-texture','/workspace/rover/sand_texture.jpg']
    # Match the installed python.sh environment but exec Python as PID1 for SIGTERM/finally.
    inside=('export CARB_APP_PATH=/isaac-sim/kit ISAAC_PATH=/isaac-sim EXP_PATH=/isaac-sim/apps\n'
            'source /isaac-sim/setup_python_env.sh\n'
            'export LD_PRELOAD=/isaac-sim/kit/libcarb.so${LD_PRELOAD:+:$LD_PRELOAD}\n'
            'exec '+shlex.join(args)+' > /workspace/run.log 2>&1')
    docker=['docker','run','-d','--name',name,'--user','0:0','--gpus',f'device={gpu}','--shm-size=2g',
        '-e','ACCEPT_EULA=Y','-e','PRIVACY_CONSENT=Y','-e','OMNI_KIT_ALLOW_ROOT=1',
        '-e','PYTHONPATH=/workspace','-e','PYTHONNOUSERSITE=1','-e','OMP_NUM_THREADS=4',
        '-e','OPENBLAS_NUM_THREADS=1','-e','MPLCONFIGDIR=/workspace/.mpl',
        '-v',run+':/workspace','-v',run+'/rover:/workspace/rover:ro','-v',ARENA+':/arena/arena.usd:ro',
        '-w','/workspace','--entrypoint','bash',IMAGE,'-lc',inside]
    launch={'container':name,'gpu':gpu,'observer_port':port,'fault_injection':fault,
            'max_sim_s':sim,'max_wall_s':wall,'gpu_before':gpu_before,'argv':docker}
    write='from pathlib import Path; Path('+repr(run+'/launch.json')+').write_text('+repr(json.dumps(launch,indent=2))+')'
    remote(shlex.join(['python3','-c',write]))
    remote(f'docker image inspect {shlex.quote(IMAGE)} --format "{{{{.Id}}}}" > {run}/image_id.txt')
    if estimator:
        sidecar=['docker','run','-d','--name','rover-orb-'+label,'--user','0:0','-v',run+':/workspace',
                 '-v',STREAM_ORB+':/usr/local/bin/stream_orb:ro',ORB_IMAGE,'bash','-lc',
                 'mkdir -p /workspace/orbio && exec stream_orb /opt/ORB_SLAM3/Vocabulary/ORBvoc.txt /workspace/rover/orb.yaml /workspace/orbio']
        remote(f'mkdir {shlex.quote(run+"/orbio")}')
        remote(shlex.join(sidecar))
    remote(shlex.join(docker))
    remote(f'nohup docker wait {name} > {run}/exit.txt 2> {run}/exit-wait.log < /dev/null &')
    viewer=['python3','-m','experiments.site_sim.viewer','--output',run+'/output','--port',str(port)]
    remote(f'cd {shlex.quote(run)}\nnohup {shlex.join(viewer)} > {run}/viewer.log 2>&1 < /dev/null &\necho $! > {run}/viewer.pid')
    print('started:',run,'container:',name,'observer port:',port)


def status(label: str) -> None:
    run=f'{ROOT}/{label}'
    remote(f'docker ps -a -f name=^rover-site-{label}$ --format "{{{{.Status}}}}"; '
           f'if test -f {run}/output/summary.json; then cat {run}/output/summary.json; '
           f'elif test -f {run}/output/live/state.json; then tail -c 1600 {run}/output/live/state.json; fi; '
           f'tail -4 {run}/run.log 2>/dev/null | cut -c1-250')


def tunnel(label: str, local_port: int) -> None:
    run=f'{ROOT}/{label}'
    launch=json.loads(remote('cat '+run+'/launch.json',capture=True).stdout)
    directory=Path.home()/'.cache/rover-site-sim';directory.mkdir(parents=True,exist_ok=True)
    pidfile=directory/f'{label}.pid'
    if pidfile.exists():
        try:record=json.loads(pidfile.read_text())
        except ValueError:record=None
        if isinstance(record,dict) and process_identity(record['pid'])==record:raise RuntimeError('tunnel already running')
    log=(directory/f'{label}.log').open('ab')
    command=['ssh','-o','BatchMode=yes','-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=40','-N',
        '-L',f'127.0.0.1:{local_port}:127.0.0.1:{int(launch["observer_port"])}','hpi-lab']
    process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    pidfile.write_text(json.dumps(process_identity(process.pid)));log.close()
    print(f'http://127.0.0.1:{local_port} (tunnel pid {process.pid})')


def process_identity(pid: int) -> dict | None:
    """PID 재사용을 구분하는 시작 tick과 정확한 argv를 함께 읽는다."""
    try:
        root=Path(f'/proc/{pid}')
        return {'pid':pid,'start_ticks':int((root/'stat').read_text().rsplit(')',1)[1].split()[19]),
                'argv':(root/'cmdline').read_bytes().decode().split('\0')}
    except (OSError,ValueError,IndexError):return None


def stop(label: str) -> None:
    run=f'{ROOT}/{label}'
    remote(f'docker stop --time 20 rover-site-{label}')
    remote(f'docker rm -f rover-orb-{label} >/dev/null 2>&1; true')
    # Verify the recorded PID still belongs to this exact observer before stopping it.
    terminate=f'''import os,signal
from pathlib import Path
pidfile=Path({run+'/viewer.pid'!r})
if pidfile.is_file():
    pid=int(pidfile.read_text())
    command=Path('/proc/'+str(pid)+'/cmdline')
    if command.is_file():
        argv=command.read_bytes().split(b'\\0')
        if b'experiments.site_sim.viewer' in argv and {run+'/output'!r}.encode() in argv:
            os.kill(pid,signal.SIGTERM)
'''
    remote(shlex.join(['python3','-c',terminate]))
    pidfile=Path.home()/'.cache/rover-site-sim'/f'{label}.pid'
    if pidfile.is_file():
        try:
            record=json.loads(pidfile.read_text())
            if isinstance(record,dict) and process_identity(record['pid'])==record:
                os.kill(record['pid'],signal.SIGTERM)
        except (ProcessLookupError,FileNotFoundError,ValueError,KeyError):pass
        pidfile.unlink()


def main(argv: list[str] | None=None) -> int:
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('upload','start','status','tunnel','stop','fetch'))
    parser.add_argument('label');parser.add_argument('path',nargs='?');parser.add_argument('--fault-injection',action='store_true')
    parser.add_argument('--max-sim-s',type=float,default=900.);parser.add_argument('--max-wall-s',type=float,default=3600.)
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--gpu',type=int,choices=(0,1),default=1)
    parser.add_argument('--assets',type=Path,default=REPO/'outputs/site_sim/rover_assets_new')
    parser.add_argument('--texture',type=Path,default=None)
    parser.add_argument('--save-front-s',type=float,default=0.)
    parser.add_argument('--record-images',action='store_true',help='start: 영상 원본·재생 프레임 보관(기본 OFF)')
    parser.add_argument('--stanley-gain',type=float,default=0.,help='0보다 크면 경로 추종에 Stanley식 횡방향 오차 항을 더한다')
    parser.add_argument('--follower',choices=('unicycle','blend'),default=None,help='start: 경로 추종기(blend=yaw+vy 혼합)')
    parser.add_argument('--blend-max-yaw-rate',type=float,default=None,help='start: blend 추종기의 각속도 상한 [rad/s] (기본 0.09)')
    parser.add_argument('--wheel-gain-left',type=float,default=1.,help='왼쪽 바퀴 속도 배율(슬립 주입, 0~2)')
    parser.add_argument('--wheel-gain-right',type=float,default=1.)
    parser.add_argument('--slip-start-s',type=float,default=0.);parser.add_argument('--slip-end-s',type=float,default=None)
    parser.add_argument('--save-overview-s',type=float,default=.25,help='위에서 내려다본 관찰 화면을 이 간격(sim s)으로 저장한다')
    parser.add_argument('--stereo-hz',type=float,default=0.)
    parser.add_argument('--lighting-profile',choices=('legacy','diffuse'),default=None,help='start: 환경광 중심 또는 기존 조명 재현')
    parser.add_argument('--wall-material',choices=('black','legacy'),default=None,help='start: 무광 검정 벽(기본) 또는 원본 벽 재질')
    parser.add_argument('--start-hold-s',type=float,default=0.)
    parser.add_argument('--navigation',choices=('path','region-mppi'),default=None)
    parser.add_argument('--mppi-max-speed',type=float,default=None,help='start: 구역 MPPI 최대 속도 [m/s] (run.py 기본 0.15)')
    parser.add_argument('--actuator-model',choices=('legacy','ksrc-pi'),default=None,help='start: 시뮬 하위 구동 모델')
    parser.add_argument('--soil-model',choices=('fixed','ksrc-legacy'),default=None,help='start: 고정 접촉 또는 peer legacy 모래 모델')
    parser.add_argument('--estimator',action='store_true',help='GT 대신 ORB+ESKF 추정 위치로 주행')
    parser.add_argument('--adaptive-visual-noise',action='store_true')
    parser.add_argument('--ksrc',type=Path,default=None,help='upload: ksrc 체크아웃(RL 정책·펌웨어 C 소스)을 같이 올리고 서버에서 .so를 빌드한다')
    parser.add_argument('--rl-policy',default=None,help='start: 업로드된 정책 이름(확장자 없이, 예: res8_mlp_v1_rc4000)')
    parser.add_argument('--imu-realism',action='store_true',help='start: 실기에서 잰 IMU 바이어스·잡음·진동을 더함')
    parser.add_argument('--imu-seed',type=int,default=0)
    parser.add_argument('--imu-profile',choices=('nominal','stress'),default=None,help='start: IMU 정지 잡음 프로필')
    parser.add_argument('--imu-vibration-scale',type=float,default=None,help='start: 이동 진동 배율')
    parser.add_argument('--visual-turn-scale',type=float,default=None,help='start: 회전 중 영상 관측 σ 배율')
    parser.add_argument('--filter-absolute-attitude-sigma-deg',type=float,default=None,help='start: ORB 절대 자세를 이 σ[deg]로 필터에 묶음')
    parser.add_argument('--filter-acc-noise',type=float,default=None,help='start: 필터 IMU 가속도 잡음 가정')
    parser.add_argument('--filter-gyro-bias-init-sigma-dps',type=float,default=None,help='start: 정지 구간 gyro 평균으로 필터 bias 초기화 후 z bias σ [dps]')
    parser.add_argument('--filter-gyro-noise-scale',type=float,default=None,help='start: 필터 IMU 자이로 잡음(IMU.NoiseGyro) 배율')
    parser.add_argument('--velocity-direction-sigma',type=float,default=None,help='start: 조향·바퀴 측정 이동 방향을 속도 방향 관측으로 쓰는 σ [m/s]')
    parser.add_argument('--control-dt',type=float,default=None,help='start: 제어·IMU 구간 주기 [s] (기본 0.05; RL은 0.02 필요)')
    parser.add_argument('--mu',type=float,default=None,help='start: 지면·바퀴 마찰계수')
    parser.add_argument('--wheel-effort',type=float,default=None,help='start: 바퀴 최대 토크 [N m]')
    parser.add_argument('--command-delay-ms',type=float,nargs=2,default=None,metavar=('LO','HI'),help='start: 명령→조인트 지연 범위 [ms]')
    parser.add_argument('--delay-seed',type=int,default=None)
    parser.add_argument('--imu-drop-prob',type=float,default=None,help='start: IMU 패킷 누락 확률(--imu-realism 필요)')
    parser.add_argument('--start-noise-xy-m',type=float,default=None);parser.add_argument('--start-noise-yaw-deg',type=float,default=None)
    parser.add_argument('--start-seed',type=int,default=None)
    parser.add_argument('--rtx-aa-op',type=int,default=None,help='start: 렌더러 안티앨리어싱 /rtx/post/aa/op (0 끔·1 TAA·2 FXAA·3 DLSS·4 RTXAA)')
    parser.add_argument('--dlss-exec-mode',type=int,default=None,help='start: DLSS 모드 /rtx/post/dlss/execMode (0 Performance·1 Balanced·2 Quality·3 Auto; 안 주면 run.py 기본 2=Quality, 0은 예전 Performance)')
    args=parser.parse_args(argv)
    extra=[]
    if args.record_images:extra.append('--record-images')
    for flag in ('mu','wheel_effort','delay_seed','imu_drop_prob','imu_vibration_scale','visual_turn_scale','filter_acc_noise','filter_absolute_attitude_sigma_deg','filter_gyro_noise_scale','filter_gyro_bias_init_sigma_dps','velocity_direction_sigma','imu_profile','start_noise_xy_m','start_noise_yaw_deg','start_seed','blend_max_yaw_rate','follower','rtx_aa_op','dlss_exec_mode','lighting_profile','wall_material','navigation','mppi_max_speed','actuator_model','soil_model'):
        if getattr(args,flag) is not None:extra+=['--'+flag.replace('_','-'),str(getattr(args,flag))]
    if args.command_delay_ms is not None:extra+=['--command-delay-ms',str(args.command_delay_ms[0]),str(args.command_delay_ms[1])]
    try:
        if not re.fullmatch('[A-Za-z0-9_-]+',args.label):raise ValueError('invalid label')
        if not 1024<=args.port<=65535:raise ValueError('invalid observer port')
        if args.action=='upload':
            if not args.path:raise ValueError('mission folder required')
            upload(args.label,Path(args.path).resolve(),args.assets,args.texture,args.ksrc)
        elif args.action=='start':start(args.label,args.fault_injection,args.max_sim_s,args.max_wall_s,args.port,args.gpu,args.save_front_s,args.stereo_hz,args.start_hold_s,args.estimator,args.adaptive_visual_noise,args.save_overview_s,args.wheel_gain_left,args.wheel_gain_right,args.slip_start_s,args.slip_end_s,args.stanley_gain,rl_policy=args.rl_policy,imu_realism=args.imu_realism,imu_seed=args.imu_seed,control_dt=args.control_dt,extra_args=extra)
        elif args.action=='status':status(args.label)
        elif args.action=='fetch':fetch(args.label,Path(args.path) if args.path else REPO/'outputs/site_sim'/args.label)
        elif args.action=='tunnel':tunnel(args.label,int(args.path or args.port))
        else:stop(args.label)
    except ValueError as error:print(error,file=sys.stderr);return 2
    except (OSError,RuntimeError,subprocess.CalledProcessError) as error:print(error,file=sys.stderr);return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
