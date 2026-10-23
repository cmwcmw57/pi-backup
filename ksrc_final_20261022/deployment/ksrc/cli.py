# ================================================================
# cli.py
# 설명: 인터넷 없는 AP 내부망에서 최종 f판 프로세스를 시작·감시·종료한다.
#       기본은 모터 없는 fake이고 --real을 명시한 경우만 기존 정책의 SPI를 연다.
# 사용법: ./ksrc check; ./ksrc start; ./ksrc status; ./ksrc estop; ./ksrc stop
# ================================================================
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / 'local/deps'), str(HERE / 'runtime')]
DEFAULT_STATE = HERE / 'state'


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description='KSRC f판 offline CLI (기본 fake)')
    result.add_argument('--state-dir', type=Path, default=DEFAULT_STATE)
    commands = result.add_subparsers(dest='command', required=True)
    commands.add_parser('check', help='의존성·최종 코드 fingerprint 확인; 인터넷·SPI 불필요')
    config = commands.add_parser('configure', help='실제 CameraInfo·URDF·IMU 보정으로 실기 설정 생성')
    camera_source = config.add_mutually_exclusive_group(required=True)
    camera_source.add_argument('--camera-json', type=Path)
    camera_source.add_argument('--orb-settings', type=Path, help='실측 camera intrinsics가 든 기존 ORB YAML; 외부 파라미터는 URDF에서 재생성')
    config.add_argument('--urdf', type=Path, required=True)
    config.add_argument('--imu-json', type=Path, required=True)
    config.add_argument('--base-height', type=float, default=.165)
    for name in ('start', '_supervise'):
        start = commands.add_parser(name, help='백그라운드 시작' if name == 'start' else argparse.SUPPRESS)
        start.add_argument('--real', action='store_true', help='실제 camera·ORB·SPI 정책 사용')
        plan = start.add_mutually_exclusive_group()
        plan.add_argument('--async-plan', dest='async_plan', action='store_true', help='S1: MPPI 별도 계산(기본)')
        plan.add_argument('--sync-plan', dest='async_plan', action='store_false', help='S0 비교 시험 전용: 동기 MPPI')
        start.set_defaults(async_plan=True)
        start.add_argument('--upper-cpu', default='0')
        start.add_argument('--orb-cpus', default='1,2')
        start.add_argument('--lower-cpu', default='3')
        start.add_argument('--rt', type=int, default=50, help='하위 정책 SCHED_FIFO 우선순위(기본 50)')
        start.add_argument('--ros-domain', type=int, default=76)
        start.add_argument('--policy-script', type=Path, default=Path('/home/pi/ksrc/pi/policy/run_policy.py'))
        start.add_argument('--policy', type=Path, default=Path('/home/pi/ksrc/pi/policy/res8_mlp_v3vo_rc4000'))
        start.add_argument('--seconds', type=float, default=3600.)
    for name in ('status', 'anchor', 'start1', 'start2', 'estop', 'stop', 'logs'):
        commands.add_parser(name)
    view = commands.add_parser('view', help='읽기 전용 한국어 상태 요약')
    view.add_argument('--watch', action='store_true', help='한 화면을 1초마다 갱신; Ctrl+C는 보기만 종료')
    commands.add_parser('report', help='저장된 CPU·온도·정책 주기 로그 요약')
    teleop = commands.add_parser('teleop', help='같은 gateway를 통한 CLI 속도 명령')
    teleop.add_argument('vx', type=float)
    teleop.add_argument('vy', type=float)
    teleop.add_argument('yaw', type=float)
    teleop.add_argument('--seconds', type=float, default=1.)
    return result


def fake_config() -> dict:
    """합성 입력 전용 설정. 실기 설정으로 저장하거나 재사용하지 않는다."""
    import numpy as np
    identity = np.eye(4).tolist()
    return dict(source='FAKE_ONLY', start_z_m=.455, wall=None,
                calibration={'ROS.T_c1_base': identity, 'IMU.T_b_c1': identity,
                             'IMU.NoiseGyro': .00017, 'IMU.NoiseAcc': .1,
                             'IMU.GyroWalk': 1e-12, 'IMU.AccWalk': .003},
                imu=dict(gyro_bias_dps=[0, 0, 0], acc_offset_g=[0, 0, 0], acc_scale=[1, 1, 1]))


def configure(args) -> None:
    """오래된 YAML 외부 파라미터 대신 현재 실제 URDF와 카메라 보정을 쓴다."""
    import numpy as np
    from ksrc_runtime.orbslam3_offline.calibration import CameraIntrinsics, build_orb_calibration, write_orb_yaml
    from ksrc_runtime.handheld_mapping.imu import load_calibration
    from ksrc_runtime.handheld_mapping.site_mission import SiteMission
    if args.camera_json:
        data = json.loads(args.camera_json.read_text())['stereo']
        left = data['left']
        k = left['k']
        intrinsics = CameraIntrinsics(left['width'], left['height'], k[0], k[4], k[2], k[5])
        stereo_transform = data['T_c1_c2']
    else:
        import cv2
        storage = cv2.FileStorage(str(args.orb_settings), cv2.FILE_STORAGE_READ)
        try:
            intrinsics = CameraIntrinsics(int(storage.getNode('Camera.width').real()), int(storage.getNode('Camera.height').real()),
                                          *(float(storage.getNode('Camera1.' + key).real()) for key in ('fx', 'fy', 'cx', 'cy')))
            stereo_transform = storage.getNode('Stereo.T_c1_c2').mat().tolist()
        finally:
            storage.release()
    if (intrinsics.width, intrinsics.height) != (640, 480):
        raise ValueError('IR 640x480 calibration required')
    calibration = build_orb_calibration(args.urdf, intrinsics, camera_hz=15.)
    calibration['ROS.pose_output'] = 'base'
    calibration['Stereo.T_c1_c2'] = stereo_transform
    mission = SiteMission.load(HERE / 'mission', load_mesh=False)
    cell = mission.site.cells([mission.start[:2]])[0]
    height = float(mission.site.height_m[cell[1], cell[0]])
    if not np.isfinite(height):
        raise ValueError('start region has no height')
    imu = load_calibration(args.imu_json)
    if imu['chip_to_body'] != [[0, -1], [1, -1], [2, 1]]:
        raise ValueError('IMU chip_to_body differs from deployed nucleo_link')
    # f판 추정기 잡음 설정. ORB는 stereo만 사용하므로 IMU 항은 ORB tracking에 사용되지 않는다.
    estimator_cal = dict(calibration, **{'IMU.NoiseAcc': .1, 'IMU.GyroWalk': 1e-12})
    wall = dict(fx=intrinsics.fx, fy=intrinsics.fy, cx=intrinsics.cx, cy=intrinsics.cy,
                T_base_left_optical=np.linalg.inv(np.asarray(calibration['ROS.T_c1_base'])).tolist())
    result = dict(source='REAL_CAMERA_URDF_IMU', calibration=estimator_cal, imu=imu, wall=wall,
                  start_z_m=height + args.base_height,
                  inputs={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.urdf, args.camera_json or args.orb_settings, args.imu_json)})
    local = HERE / 'local'
    local.mkdir(exist_ok=True)
    (local / 'real.json').write_text(json.dumps(result, indent=2) + '\n')
    write_orb_yaml(local / 'orb.yaml', calibration)
    print('configured', local / 'real.json', '; ORB stereo/base 640x480@15; depth OFF')


def control(op: int, run: int = 0) -> None:
    """기존 gateway KCTL로 시작·E-STOP을 전달한다."""
    seq = int(time.monotonic() * 1000) & 0xFFFFFFFF
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as output:
        for i in range(3):
            output.sendto(struct.pack('<4sIB3f', b'KCTL', (seq + i) & 0xFFFFFFFF, op, run, 0., 0.), ('127.0.0.1', 5006))
            time.sleep(.02)


def process_identity(pid: int) -> str | None:
    """PID 재사용 때 다른 프로세스를 종료하지 않도록 Linux starttime을 확인한다."""
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def stop_owned(info: dict, sig: int = signal.SIGTERM) -> None:
    """이번 세션이 기록한 프로세스 그룹만 종료한다."""
    if process_identity(info['pid']) == info['starttime']:
        try:
            os.killpg(info['pid'], sig)
        except ProcessLookupError:
            pass


def stop_session(state: Path) -> None:
    """E-STOP을 보내고 이 폴더가 소유한 세션을 종료한다. 12초 안에 안 끝나면 오류."""
    control(4)
    path = state / 'supervisor.json'
    if path.exists():
        owner = json.loads(path.read_text())
        stop_owned(owner)
        deadline = time.monotonic() + 12.
        while time.monotonic() < deadline and process_identity(owner['pid']) == owner['starttime'] and not (state / 'stopped.json').exists():
            time.sleep(.1)
        if not (state / 'stopped.json').exists() and process_identity(owner['pid']) == owner['starttime']:
            raise RuntimeError('shutdown not finished; see supervisor.log')
    print('stopped owned processes' if (state / 'stopped.json').exists() else 'supervisor not running; see logs')


def stop_other_sessions(state: Path) -> None:
    """같은 상위 폴더의 다른 살아 있는 세션을 끈다. AUTO 중이거나 상태를 모르면 끄지 않고 중단한다."""
    for path in sorted(state.parent.glob('*/supervisor.json')):
        other = path.parent
        try:
            owner = json.loads(path.read_text())
        except ValueError:
            continue
        if other == state or process_identity(owner['pid']) != owner['starttime']:
            continue
        status_path = other / 'status.json'
        try:
            status = json.loads(status_path.read_text())
            age = time.time() - status_path.stat().st_mtime
        except (OSError, ValueError):
            status, age = {}, None
        reason = 'AUTO 주행 중' if status.get('active') else '상태를 알 수 없음' if age is None or age > 5. else ''
        if reason:
            raise RuntimeError(f'다른 세션이 실행 중이라 시작하지 않는다 ({reason}): {other}\n'
                               f'먼저 ./ksrc --state-dir {other} estop 후 ./ksrc --state-dir {other} stop 으로 끈 뒤 다시 시작하세요.')
        print('다른 세션이 대기 중이라 먼저 종료한다:', other, flush=True)
        stop_session(other)


def supervise(args) -> None:
    """SSH가 끊겨도 프로세스를 유지하고 ORB만 자동 재시작한다."""
    state = args.state_dir.resolve()
    state.mkdir(parents=True, exist_ok=True)
    lock = (state / 'session.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # 서로 다른 state-dir에서도 동일한 실기 제어 스택이 두 번 기동되지 않게 한다.
    global_lock = Path('/dev/shm/ksrc-navigation.lock').open('a')
    fcntl.flock(global_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(os.environ, PYTHONNOUSERSITE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', ROS_DOMAIN_ID=str(args.ros_domain),
               PYTHONPATH=str(HERE / 'local/deps'))
    config = HERE / 'local/real.json' if args.real else state / 'fake.json'
    if args.real:
        if not config.exists() or not (HERE / 'local/orb.yaml').exists():
            raise ValueError('run configure first')
    else:
        config.write_text(json.dumps(fake_config()))
    owned, handles = {}, []
    quit_requested = False

    def quitting(signum: int, frame: object) -> None:
        nonlocal quit_requested
        quit_requested = True

    signal.signal(signal.SIGTERM, quitting)
    signal.signal(signal.SIGINT, quitting)

    def launch(name: str, command: list[str], cpu: str | None = None, ros: bool = False):
        if cpu is not None:
            command = ['taskset', '-c', cpu, *command]
        if ros:
            command = ['bash', str(HERE / 'ros_env.sh'), *command]
        log = (state / f'{name}.log').open('a', buffering=1)
        handles.append(log)
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                 stdin=subprocess.DEVNULL, start_new_session=True, cwd=HERE)
        info = dict(pid=child.pid, starttime=process_identity(child.pid), command=command)
        owned[name] = child, info
        (state / 'processes.json').write_text(json.dumps({k: v[1] for k, v in owned.items()}, indent=2))
        print('started', name, child.pid, flush=True)
        return child

    python = sys.executable
    began, generation, orb = time.monotonic(), 0, None
    supervisor = dict(pid=os.getpid(), starttime=process_identity(os.getpid()))
    (state / 'supervisor.json').write_text(json.dumps(supervisor))
    try:
        command = [python, str(HERE / 'runner.py'), '--config', str(config), '--state-dir', str(state)]
        if args.async_plan:
            command += ['--plan-cpu', args.upper_cpu or str(min(os.sched_getaffinity(0)))]
        launch('runner', command, args.upper_cpu)
        launch('gateway', [python, str(HERE / 'runtime/ksrc_runtime/handheld_mapping/control_gateway.py'), '--runner', '127.0.0.1:5010'])
        if args.real:
            camera = '/opt/ros/jazzy/lib/realsense2_camera/realsense2_camera_node'
            launch('camera', [camera, '--ros-args', '-r', '__ns:=/camera', '-r', '__node:=camera',
                             '-p', 'enable_color:=false', '-p', 'enable_depth:=false', '-p', 'enable_infra1:=true',
                             '-p', 'enable_infra2:=true', '-p', 'depth_module.infra_profile:=640,480,15',
                             '-p', 'depth_module.infra1_format:=Y8', '-p', 'depth_module.infra2_format:=Y8',
                             '-p', 'depth_module.emitter_enabled:=0', '-p', 'enable_sync:=true'], ros=True)
            launch('inputs', [python, str(HERE / 'ros_inputs.py'), '--config', str(config)], ros=True)
        else:
            launch('inputs', [python, str(HERE / 'fake_inputs.py'), '--state-dir', str(state)])
        if args.policy_script.exists():
            command = [python, str(HERE / 'policy_telemetry.py'), '--config', str(config),
                       '--policy-script', str(args.policy_script)]
            if args.real:
                command.append('--real')
            if not args.real:
                # 합성 입력 프로세스가 IMU를 보낸다. FakeLink telemetry를 중복 적분하지 않는다.
                command = [python, str(args.policy_script), '--fake']
            else:
                command.append('--')
            command += ['--policy', str(args.policy), '--cmd-udp', '5005', '--vo-port', '5600',
                        '--seconds', str(args.seconds), '--rt', str(args.rt), '--gc-off',
                        '--log', str(state / 'policy.npz')]
            launch('policy', command, args.lower_cpu)
        elif args.real:
            raise FileNotFoundError(args.policy_script)
        else:
            print('FAKE: lower policy absent; gateway output only', flush=True)
        next_orb = 0.
        while not quit_requested and time.monotonic() - began < args.seconds:
            # runner/gateway/입력/하위의 사망은 세션 전체 정지. ORB만 여기서 자동 재시작한다.
            dead = [name for name, (child, _) in owned.items() if name != 'orb' and child.poll() is not None]
            if dead:
                raise RuntimeError('process exited: ' + ', '.join(dead))
            restart = state / 'restart_orb'
            if args.real and time.monotonic() >= next_orb and (orb is None or orb.poll() is not None or restart.exists()):
                if orb is not None:
                    stop_owned(owned['orb'][1])
                    try:
                        orb.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        stop_owned(owned['orb'][1], signal.SIGKILL)
                        orb.wait(timeout=2)
                restart.unlink(missing_ok=True)
                generation += 1
                trajectory = state / f'orb_{generation:03d}'
                trajectory.mkdir()
                orb = launch('orb', ['/home/pi/handheld_mapping/ros2_ws/install/orbslam3_ros2/lib/orbslam3_ros2/orbslam3_stereo_inertial',
                                    '--ros-args', '-p', 'vocabulary_path:=/home/pi/handheld_mapping/runtime/ORBvoc.txt',
                                    '-p', f'settings_path:={HERE / "local/orb.yaml"}', '-p', f'trajectory_dir:={trajectory}',
                                    '-p', 'sensor_mode:=stereo', '-p', 'use_sim_time:=false',
                                    '-p', 'left_topic:=/camera/camera/infra1/image_rect_raw',
                                    '-p', 'right_topic:=/camera/camera/infra2/image_rect_raw'], args.orb_cpus, ros=True)
                next_orb = time.monotonic() + 8.
            time.sleep(.1)
    finally:
        control(4)
        for name in ('runner', 'inputs', 'orb', 'camera', 'policy', 'gateway'):
            if name in owned:
                # FAKE의 기존 정책은 SIGTERM handler 없이 KeyboardInterrupt에서 로그를 저장한다.
                stop_owned(owned[name][1], signal.SIGINT if name == 'policy' and not args.real else signal.SIGTERM)
        for child, info in owned.values():
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                stop_owned(info, signal.SIGKILL)
                child.wait(timeout=2)
        for handle in handles:
            handle.close()
        (state / 'stopped.json').write_text(json.dumps(dict(stopped=True, time=time.time())))
        print('stopped owned processes', flush=True)


def main() -> None:
    args = parser().parse_args()
    state = args.state_dir.resolve()
    if args.command == 'check':
        from runner import Navigation
        navigation = Navigation(HERE / 'mission', fake_config())
        origin = json.loads((HERE / 'ORIGIN.json').read_text())
        for name, expected in origin['deployed_sha256'].items():
            if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != expected:
                raise ValueError('runtime fingerprint mismatch: ' + name)
        navigation.close()
        print('OK: YW16 + YW17 / f; ESKF + RegionMission + MPPI imports; grid hash; runtime hashes')
        print('mode default=FAKE; SPI unopened; network unnecessary; Python', sys.version.split()[0])
        print('RT priority limit:', __import__('resource').getrlimit(__import__('resource').RLIMIT_RTPRIO))
    elif args.command == 'configure':
        configure(args)
    elif args.command == 'start':
        state.mkdir(parents=True, exist_ok=True)
        path = state / 'supervisor.json'
        if path.exists():
            previous = json.loads(path.read_text())
            if process_identity(previous['pid']) == previous['starttime']:
                raise RuntimeError('session already running')
        stop_other_sessions(state)
        for name in ('status.json', 'stopped.json', 'restart_orb'):
            (state / name).unlink(missing_ok=True)
        forwarded = sys.argv[1:]
        forwarded[forwarded.index('start')] = '_supervise'
        with (state / 'supervisor.log').open('a') as log:
            child = subprocess.Popen([sys.executable, str(HERE / 'cli.py'), *forwarded],
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, cwd=HERE)
        time.sleep(.8)
        if child.poll() is not None:
            raise RuntimeError((state / 'supervisor.log').read_text()[-2500:])
        print('started', 'REAL' if args.real else 'FAKE', 'supervisor', child.pid, '; logs:', state)
        print('plan:', 'S1 async' if args.async_plan else 'S0 sync',
              '; upper CPU', args.upper_cpu, '; lower CPU', args.lower_cpu, '; RT', args.rt)
        print('초기화 후 view로 확인하세요. 시작 직후 상태 파일이 없을 수 있습니다.')
    elif args.command == '_supervise':
        supervise(args)
    elif args.command in ('view', 'report'):
        from display import report_text, view_text, watch
        if args.command == 'report':
            print(report_text(state))
        elif args.watch:
            watch(state)
        else:
            print(view_text(state))
    elif args.command == 'status':
        path = state / 'status.json'
        print(path.read_text() if path.exists() else 'status unavailable; see supervisor.log')
        if (state / 'stopped.json').exists():
            print('SESSION STOPPED (above status is historical)')
    elif args.command in ('start1', 'start2', 'estop'):
        control(4 if args.command == 'estop' else 3, 1 if args.command == 'start1' else 2)
        print('sent', args.command, '; confirm with status')
    elif args.command == 'anchor':
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as output:
            output.sendto(b'{"op":"ANCHOR"}', ('127.0.0.1', 5010))
        print('preset anchor requested; only accepted while stopped with fresh sensors for 3 s; confirm status')
    elif args.command == 'stop':
        stop_session(state)
    elif args.command == 'logs':
        for path in sorted(state.glob('*.log')):
            print(path.name, '\n', '\n'.join(path.read_text(errors='replace').splitlines()[-8:]))
    elif args.command == 'teleop':
        import math
        if not all(math.isfinite(v) for v in (args.vx, args.vy, args.yaw, args.seconds)) or not 0 < args.seconds <= 60:
            raise ValueError('finite velocity and 0 < seconds <= 60 required')
        seq = int(time.monotonic() * 1000) & 0xFFFFFFFF
        end = time.monotonic() + args.seconds
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as output:
            try:
                while time.monotonic() < end:
                    seq = (seq + 1) & 0xFFFFFFFF
                    output.sendto(struct.pack('<4sIfff', b'KSRC', seq, args.vx, args.vy, args.yaw), ('127.0.0.1', 5006))
                    time.sleep(.02)
            finally:
                for _ in range(3):
                    seq = (seq + 1) & 0xFFFFFFFF
                    output.sendto(struct.pack('<4sIfff', b'KSRC', seq, 0., 0., 0.), ('127.0.0.1', 5006))


if __name__ == '__main__':
    main()
