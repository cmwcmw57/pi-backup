# ================================================================
# controller.py
# 설명: 노트북 촬영 세션과 소유 subprocess의 정상 종료·저장·TSDF를 관리한다.
# 사용법: from experiments.handheld_mapping.controller import Session
# ================================================================
from datetime import datetime
from collections import deque
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import uuid

from experiments.handheld_mapping.storage import RunWriter, atomic_json
from experiments.handheld_mapping.transport import PacketReader

ROOT = Path(__file__).resolve().parents[2]
ACTIVE = {"starting", "recording", "stopping", "processing", "reconstructing", "comparing"}


def remote_command(host: str, run_id: str, sensors_only: bool = False, queue_mib: float = 4.,
                   projector_on: bool = False, rgb_enabled: bool = False, fps: int = 15) -> list[str]:
    if not re.fullmatch(r"[A-Za-z0-9_.@:-]+", host) or host.startswith("-"):
        raise ValueError("잘못된 SSH 연결 대상")
    if not math.isfinite(queue_mib) or not .25 <= queue_mib <= 64:
        raise ValueError("invalid queue MiB limit")
    remote = f"/dev/shm/rover_mapping_{run_id}"
    code = (
        "set -e; source /opt/ros/jazzy/setup.bash >&2; "
        "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 ROS_DOMAIN_ID=67 RMW_IMPLEMENTATION=rmw_fastrtps_cpp RCUTILS_LOGGING_USE_STDOUT=0; "
        f"export ROS_LOG_DIR={shlex.quote(remote+'/logs')}; "
        "export FASTRTPS_DEFAULT_PROFILES_FILE=/home/pi/handheld_mapping/live/fastdds_stereo_local.xml; "
        "export LD_LIBRARY_PATH=/home/pi/handheld_mapping/runtime/install/lib:/home/pi/handheld_mapping/runtime/prefix/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}; "
        f"cd {shlex.quote(remote+'/code')}; "
        f"exec /usr/bin/python3 -m experiments.handheld_mapping.pi_worker --run-dir {shlex.quote(remote+'/run')}"
        f" --queue-mib {queue_mib}"+(" --sensors-only" if sensors_only else "")
        +(" --projector-on" if projector_on else "")
        +(" --rgb" if rgb_enabled else "")
        +(f" --fps {int(fps)}" if fps != 15 else "")
    )
    return ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=5",
            "-o", "ServerAliveCountMax=3", host, shlex.join(["bash", "-c", code])]


def deploy(host: str, run_id: str) -> None:
    # 명령의 host 검증을 공유하며 사용자 셸 문자열을 실행하지 않는다.
    remote_command(host, run_id)
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        for path in (ROOT / "experiments/handheld_mapping").glob("*.py"):
            tar.add(path, arcname=f"experiments/handheld_mapping/{path.name}")
        for name in ("prepare_mapping_run.py",):
            tar.add(ROOT / "ros2_ws/src/orbslam3_ros2/scripts" / name, arcname=f"helpers/{name}")
    remote = f"/dev/shm/rover_mapping_{run_id}/code"
    command = f"mkdir -p {shlex.quote(remote)} && tar -xzf - -C {shlex.quote(remote)}"
    subprocess.run(["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, command],
                   input=archive.getvalue(), check=True, timeout=20, stdout=subprocess.DEVNULL,
                   stderr=subprocess.PIPE)


class Session:
    """촬영·파일 기록·최종 지도 생성의 상태를 한 worker thread에서 관리한다."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.status = {"state": "idle", "message": "준비", "counts": {}, "run_dir": "", "dropped": 0}
        self.process: subprocess.Popen | None = None
        self.thread: threading.Thread | None = None
        self.stop_requested = threading.Event()
        self.preview = None
        self.color_preview = None
        self.started_at = 0.
        self.ended_at = 0.
        self.generation = 0
        self.left_receipts = deque(maxlen=30)
        self.live = None
        self.imu = None

    def live_snapshot(self) -> dict | None:
        with self.lock:
            mapper = self.live
            error = self.status.get('live_error')
        return mapper.snapshot() if mapper is not None else ({'error':error} if error else None)

    def snapshot(self) -> dict:
        with self.lock:
            result = {**self.status, "counts": dict(self.status.get("counts", {}))}
            result["elapsed_s"] = (self.ended_at or time.monotonic())-self.started_at if self.started_at else 0.
            receipts = self.left_receipts
            result["left_receipt_fps"] = ((len(receipts)-1)/(receipts[-1]-receipts[0])
                if len(receipts) >= 2 and result["state"] == "recording" and time.monotonic()-receipts[-1] < 1 else 0.)
            imu = self.imu
        if imu is not None:
            result["imu"] = imu.snapshot()
        return result

    def update(self, **values) -> None:
        with self.lock:
            if values.get("state") in {"complete", "partial", "failed"} and not self.ended_at:
                self.ended_at = time.monotonic()
            self.status.update(values)

    def start(self, config: dict) -> None:
        config = dict(config)
        if config.get("inertial_fusion") and not config.get("demo"):
            config["imu_enabled"] = True
        with self.lock:
            if self.status["state"] in ACTIVE or (self.thread is not None and self.thread.is_alive()):
                raise RuntimeError("이미 실행 중입니다")
            self.stop_requested.clear()
            self.generation += 1
            self.process = None
            self.preview = None
            self.color_preview = None
            self.live = None
            self.imu = None
            self.left_receipts.clear()
            self.started_at = time.monotonic()
            self.ended_at = 0.
            self.status = {"state": "starting", "message": "연결·카메라 준비 중", "counts": {}, "run_dir": "", "dropped": 0}
            self.thread = threading.Thread(target=self._run, args=(dict(config),), daemon=False)
            self.thread.start()

    def stop(self) -> None:
        with self.lock:
            if self.status["state"] not in {"starting", "recording", "stopping"}:
                return
            first = not self.stop_requested.is_set()
            self.stop_requested.set()
            self.status.update(state="stopping", message="촬영 종료·최종 경로 저장 중")
            process = self.process
            if process is not None and process.poll() is None:
                try:
                    process.stdin.write(b"stop\n")
                    process.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
            if first:
                threading.Thread(target=self._stop_timeout, args=(self.generation, time.monotonic()+75), daemon=True).start()

    def _stop_timeout(self, generation: int, deadline: float) -> None:
        while time.monotonic() < deadline:
            with self.lock:
                if generation != self.generation or not self.stop_requested.is_set():
                    return
                process = self.process
                state = self.status["state"]
            if state not in {"starting", "recording", "stopping"}:
                return
            if process is not None and process.poll() is not None:
                return
            time.sleep(.2)
        with self.lock:
            if generation != self.generation or not self.stop_requested.is_set():
                return
            process = self.process
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)

    def _run(self, config: dict) -> None:
        writer = None
        run = None
        process = None
        log = None
        live = None
        packet_reader = None
        imu_partial = False
        try:
            gravity_requested = bool(config.get('gravity_enabled'))
            if gravity_requested and (not config.get('imu_enabled') or config.get('demo')):
                raise ValueError('중력 기준 지도에는 실제 보정 IMU 기록이 필요합니다')
            root = Path(config["output_root"]).expanduser().resolve()
            root.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(root).free < 2_000_000_000:
                raise OSError("노트북 여유 공간이 2GB 미만입니다")
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S_")+uuid.uuid4().hex[:6]
            run = root / run_id
            writer = RunWriter(run)
            self.update(run_dir=str(run), source="synthetic_demo" if config.get("demo") else "D435_Pi",
                        rgb_requested=bool(config.get('rgb_enabled')),
                        global_requested=bool(config.get('global_reconstruction')), gravity_requested=gravity_requested)
            atomic_json(run / "session.json", self.snapshot())
            if config.get("demo"):
                command = [sys.executable, "-u", "-m", "experiments.handheld_mapping.demo"]
            elif config.get("worker_command"):
                command = config["worker_command"]
            else:
                deploy(config.get("host", "pi@raspberrypi.local"), run_id)
                command = remote_command(config.get("host", "pi@raspberrypi.local"), run_id,
                                         sensors_only=config.get("sensors_only", False),
                                         queue_mib=config.get("queue_mib", 4.),
                                         projector_on=config.get("projector_on", False),
                                         rgb_enabled=config.get("rgb_enabled", False),
                                         fps=config.get("camera_fps", 15))
            if config.get("imu_enabled") and not config.get("demo"):
                from experiments.handheld_mapping.imu import ImuCapture
                calibration = config.get("imu_calibration")
                if not calibration:
                    raise ValueError("IMU 보정 파일 경로가 필요합니다")
                imu = ImuCapture(config.get("host", "pi@raspberrypi.local"), run,
                                 Path(calibration).expanduser(),
                                 ROOT / "ros2_ws/src/rover_description/urdf/rocker_bogie_4wd_rover.urdf")
                with self.lock:
                    self.imu = imu
                self.update(message="보정 IMU의 최신 유효 샘플 대기 중")
                imu.start()
                self.update(imu=imu.snapshot(), message="IMU 기록 준비 완료 · 카메라 준비 중")
            if self.stop_requested.is_set():
                raise RuntimeError("시작 중 종료 요청: 촬영하지 않았습니다")
            log = (run / "worker.log").open("wb")
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=log, start_new_session=True, bufsize=0)
            with self.lock:
                self.process = process
                if self.stop_requested.is_set():
                    process.stdin.write(b"stop\n")
                    process.stdin.flush()
            finished = None
            received = 0
            laptop_capture = {"packets": 0, "read_wait_ms_total": 0., "read_wait_ms_max": 0.,
                              "writer_accept_ms_total": 0., "writer_accept_ms_max": 0.,
                              "packet_process_ms_max": 0., "packet_process_ms_max_kind": None,
                              "packet_process_ms_max_stamp_ns": None}
            packet_reader = PacketReader(process.stdout)
            while True:
                if self.imu is not None:
                    imu_status = self.imu.snapshot()
                    if imu_status.get("error"):
                        raise RuntimeError("IMU 기록 오류: "+str(imu_status["error"]))
                    if imu_status.get("stale"):
                        self.update(imu_warning="IMU 표시 갱신 대기 · 노트북 수신 지연")
                    else:
                        self.update(imu_warning=None)
                packet = packet_reader.get()
                if packet is None:
                    break
                header, payload = packet
                receipt_ns = header['laptop_receipt_monotonic_ns']
                read_ms = header['laptop_read_wait_ms']
                # Pi와 노트북 monotonic은 서로 다른 clock이다. read 대기는 latency가 아니다.
                header = {**header, "laptop_receipt_monotonic_ns": receipt_ns,
                          "laptop_read_wait_ms": read_ms}
                accept_start = time.monotonic_ns()
                image = writer.accept(header, payload)
                accept_ms = (time.monotonic_ns()-accept_start)/1e6
                if header['kind'] == 'calibration':
                    self.update(capture_mode=header.get('capture_mode', 'pi_orb'),
                                source=header.get('source', self.snapshot()['source']))
                    if config.get('live_mapping'):
                        try:
                            from experiments.handheld_mapping.live import LiveMapper
                            live_options = {'demo':bool(config.get('demo'))}
                            if gravity_requested:
                                live_options['gravity_provider'] = self.imu.gravity_samples
                            live = LiveMapper(run, **live_options)
                            with self.lock:
                                self.live = live
                        except Exception as error:
                            self.update(live_error=str(error))
                if header['kind'] == 'pose' and live is not None:
                    try:
                        live.offer_pose(header)
                    except Exception as error:
                        self.update(live_error=str(error))
                received += 1
                if received % 45 == 0 and shutil.disk_usage(run).free < 256_000_000:
                    raise OSError("노트북 공간 부족으로 촬영을 종료합니다")
                if header["kind"] == "ready" and not self.stop_requested.is_set():
                    self.update(state="recording", message=header.get("message", "촬영 중"))
                elif header["kind"] == "status":
                    self.update(tracking_state=header.get("tracking_state", -1), dropped=header.get("dropped", 0),
                                source_frames=header.get("source_frames", 0), transport_queue=header.get('queue', {}))
                    if 'capture_metrics' in header:
                        self.update(capture_metrics=header['capture_metrics'])
                elif header["kind"] == "error":
                    raise RuntimeError(header["message"])
                elif header["kind"] == "finished":
                    finished = header
                    self.update(**{target: header[source] for source, target in (
                        ('queue', 'transport_queue'), ('dropped', 'dropped'), ('source_frames', 'source_frames'),
                        ('capture_metrics', 'capture_metrics')) if source in header})
                if image is not None:
                    if live is not None and header['stream'] in ('left', 'right', 'depth'):
                        live.offer(header)
                    with self.lock:
                        self.status["counts"] = dict(writer.counts)
                        if header["stream"] == "left":
                            self.preview = image
                            self.left_receipts.append(time.monotonic())
                        elif header['stream'] == 'color':
                            self.color_preview = image
                laptop_capture['packets'] = received
                for key, value in (("read_wait_ms", read_ms), ("writer_accept_ms", accept_ms)):
                    laptop_capture[key+'_total'] += value
                    laptop_capture[key+'_max'] = max(laptop_capture[key+'_max'], value)
                processing_ms = (time.monotonic_ns()-accept_start)/1e6
                if processing_ms > laptop_capture['packet_process_ms_max']:
                    laptop_capture.update(packet_process_ms_max=processing_ms,
                        packet_process_ms_max_kind=header['kind'],
                        packet_process_ms_max_stamp_ns=header.get('stamp_ns'))
                self.update(laptop_capture={**laptop_capture,
                    'receive_buffer': packet_reader.snapshot(),
                    'images': {side: dict(stats) for side, stats in writer.image_timings.items()}})
            exitcode = process.wait(timeout=10)
            imu_status = self._finish_imu()
            if self.snapshot().get("imu_cleanup_error"):
                raise RuntimeError("IMU 기록 저장 실패: "+self.snapshot()["imu_cleanup_error"])
            if imu_status:
                if imu_status.get("error"):
                    raise RuntimeError("IMU 기록 오류: "+str(imu_status["error"]))
                imu_partial = bool(imu_status.get('stale') or any(imu_status.get(key,0) for key in
                                   ('source_gap_events','resets','clock_jumps','tee_drops')))
                if not imu_status.get('stale'):
                    self.update(imu_warning=None)
            writer.close()
            writer = None
            if live is not None:
                try:
                    live.finish()
                    details = live.snapshot()
                    self.update(live={key:details[key] for key in ('processed','dropped_pairs','no_depth',
                                'error','map_id','lag_ms','map_changes','synthetic')})
                except Exception as error:
                    self.update(live_error=str(error))
                live = None
            if exitcode != 0 or finished is None or finished.get("status") != "saved":
                raise RuntimeError(f"정상 종료·저장 확인 실패 (exit={exitcode}, finished={finished}) — worker.log 확인")
            backend = 'orb'
            if finished.get('capture_mode') == 'sensors_only':
                self.update(state='processing', message='노트북에서 원본으로 ORB 계산 중')
                from experiments.handheld_mapping.offline import run_point
                self.update(offline=run_point(run))
                backend = 'orb_offline'
            gravity = None
            if gravity_requested:
                self.update(state='reconstructing',message='정지 IMU로 지도별 중력 기준 계산 중')
                from experiments.handheld_mapping.gravity import process_maps
                gravity = process_maps(run,backend)
                self.update(gravity=gravity)
            self.update(state="reconstructing", message="최종 keyframe으로 TSDF 지도 생성 중")
            from experiments.handheld_mapping.reconstruct import reconstruct
            fusion_options = {'color': True} if config.get('rgb_enabled') and not config.get('demo') else {}
            dense_gravity = {'gravity_alignments': gravity['maps']} if gravity_requested else {}
            dense = reconstruct(run, progress=lambda message: self.update(message=message),
                                backend=backend, **fusion_options, **dense_gravity)
            self.update(dense=dense)
            if fusion_options and dense.get('color_status') in ('partial', 'unavailable'):
                self.update(color_warning='RGB가 없거나 일부만 일치함 · 형상은 보존하고 색상 상태 별도 기록')
            inertial_partial = False
            if config.get("inertial_fusion") and not config.get("demo"):
                fusion_result = None
                try:
                    self.update(state="processing", message="저장한 원본으로 IMU 융합 시험 중")
                    from experiments.handheld_mapping.inertial import run_inertial_auto
                    fusion_result = run_inertial_auto(run)
                    self.update(inertial_fusion=fusion_result)
                    if fusion_result.get("orb_fused"):
                        self.update(state="reconstructing", message="IMU 융합 경로로 별도 TSDF 지도 생성 중")
                        inertial_dense = reconstruct(run, backend="orb_inertial",
                                                     progress=lambda message: self.update(message=message),
                                                     **fusion_options)
                        self.update(inertial_dense=inertial_dense)
                        inertial_partial = inertial_dense.get("status") != "complete"
                        if inertial_partial:
                            self.update(inertial_warning="IMU 융합 지도 일부 누락 · 기본 지도는 저장됨")
                    else:
                        inertial_partial = True
                        self.update(inertial_warning=fusion_result.get("error") or "IMU 융합 초기화 미완료 · 기본 지도는 저장됨")
                except Exception as error:
                    inertial_partial = True
                    if fusion_result is None:
                        self.update(inertial_fusion={"status": "failed", "orb_fused": False, "error": str(error)})
                    self.update(inertial_warning="IMU 융합 후처리 실패 · 기본 지도는 저장됨: "+str(error))
            terrain = None
            if gravity_requested:
                self.update(message='중력 기준 높이·경사·단차 지도 생성 중')
                from experiments.handheld_mapping.gravity import terrain_maps
                terrain = terrain_maps(run, backend, gravity, depth_max=1.)
                self.update(terrain=terrain)
            line_partial = False
            if config.get("compare_lines"):
                self.update(state="comparing", message="동일 원본으로 점+선 SLAM 비교 중")
                from experiments.handheld_mapping.compare import run_orbline
                comparison = run_orbline(run)
                self.update(comparison=comparison)
                line_partial = comparison.get("dense", {}).get("status", "complete") != "complete"
            global_partial = False
            if config.get('global_reconstruction'):
                self.update(state='reconstructing', message='depth 구간 정합·전역 재구성 중')
                try:
                    reference = self._postprocess_global(run, backend, color=bool(fusion_options))
                    global_partial = reference['status'] != 'complete'
                    self.update(reference_reconstruction=reference)
                    if reference['status'] == 'failed':
                        errors = reference.get('failures', [])
                        reason = errors[0].get('error', '유효한 결과 없음') if errors else '유효한 결과 없음'
                        self.update(reference_warning='전역 재구성 실패 · 기존 지도 보존: '+reason)
                    if reference['maps'] and all(row['optimization_status'] != 'optimized_unverified'
                                                 for row in reference['maps']):
                        self.update(reference_warning='추가 depth 제약 부족 · 기존 위치를 유지한 재융합 결과')
                except Exception as error:
                    global_partial = True
                    self.update(reference_warning='전역 재구성 실패 · 기존 지도 보존: '+str(error))
            partial = (dense["status"] != "complete" or finished.get("dropped", 0) > 0
                       or line_partial or imu_partial or inertial_partial or global_partial
                       or (gravity_requested and (gravity['status'] != 'complete' or terrain['status'] != 'complete')))
            self.update(state="partial" if partial else "complete", message="일부 누락 있음 · 지도 저장 완료" if partial else "지도 저장 완료",
                        dense=dense, dropped=finished.get("dropped", 0))
        except Exception as error:
            self.update(state="failed", error=str(error), message=str(error))
        finally:
            if packet_reader is not None:
                packet_reader.close()
            self._finish_imu()
            if live is not None:
                try:
                    live.finish()
                except Exception as error:
                    self.update(live_error=str(error))
            if process is not None:
                if process.poll() is None:
                    try:
                        process.stdin.close()
                        process.wait(timeout=3)
                    except (OSError, subprocess.TimeoutExpired):
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                process.stdout.close()
                if not process.stdin.closed:
                    process.stdin.close()
            if packet_reader is not None:
                packet_reader.thread.join(timeout=3)
            if writer is not None:
                writer.close()
            if log is not None:
                log.close()
            with self.lock:
                if not self.ended_at:
                    self.ended_at = time.monotonic()
            if run is not None:
                try:
                    atomic_json(run / "session.json", self.snapshot())
                except OSError as error:
                    self.update(state="failed", error=str(error), message="상태 파일 저장 실패: "+str(error))

    def _finish_imu(self) -> dict | None:
        with self.lock:
            imu = self.imu
        if imu is None:
            return None
        try:
            summary = imu.stop()
            self.update(imu=summary)
            return summary
        except Exception as error:
            self.update(imu_cleanup_error=str(error))
            try:
                self.update(imu=imu.snapshot())
            except Exception:
                pass
            return None
        finally:
            with self.lock:
                self.imu = None

    def zero_imu_display(self) -> dict:
        with self.lock:
            imu = self.imu
        if imu is None:
            raise ValueError('IMU를 함께 기록하는 촬영 중에 평지에서 정지한 뒤 누르세요')
        result = imu.zero_display()
        self.update(message='롤·피치 참고 표시 영점 설정 완료')
        return result

    def _postprocess_global(self, run: Path, backend: str, color: bool = False) -> dict:
        """명시 backend의 map을 각각 새 폴더에 재구성하고 실패도 따로 보존한다."""
        from experiments.handheld_mapping.global_reconstruct import run_global
        run = Path(run).resolve()
        if backend not in ('orb', 'orb_offline', 'orb_inertial'):
            raise ValueError('invalid reference backend')
        files = sorted((run / backend).glob('map_*_keyframes.csv'))
        if not files:
            raise ValueError('전역 재구성에 사용할 최종 keyframe이 없습니다')
        destination = run / 'reference_reconstruction' / uuid.uuid4().hex
        destination.mkdir(parents=True, exist_ok=False)
        result = {'status': 'processing', 'kind': 'reference_reconstruction_batch',
                  'source_run': str(run), 'backend': backend, 'output': str(destination),
                  'maps': [], 'failures': []}
        atomic_json(destination / 'summary.json', result)
        for path in files:
            token = path.name.removeprefix('map_').removesuffix('_keyframes.csv')
            if not token.isdecimal():
                result['failures'].append({'file': path.name, 'error': 'invalid map identity'})
                continue
            identity = int(token)
            output = destination / f'map_{identity}'
            self.update(message=f'지도 {identity}: depth 전역 재구성 중')
            try:
                summary = run_global(run, output, backend=backend, map_id=identity,
                                     depth_max=1., color=color,
                                     progress=lambda message: self.update(message=message))
                result['maps'].append({'map_id': identity, 'output': str(output),
                    'status': summary.get('status', 'partial'),
                    'optimization_status': summary.get('optimization_status', 'unknown')})
            except Exception as error:
                result['failures'].append({'map_id': identity, 'error': str(error), 'output': str(output)})
            atomic_json(destination / 'summary.json', result)
        result['status'] = ('failed' if not result['maps'] else 'partial'
                            if result['failures'] or any(row['status'] != 'complete' for row in result['maps'])
                            else 'complete')
        atomic_json(destination / 'summary.json', result)
        return result

    def rebuild(self, run_dir: Path, global_reconstruction: bool = True, color: bool = True) -> None:
        with self.lock:
            if self.status["state"] in ACTIVE or (self.thread is not None and self.thread.is_alive()):
                raise RuntimeError("촬영·저장 중에는 다시 생성할 수 없습니다")
            self.started_at = time.monotonic()
            self.ended_at = 0.
            self.preview = self.color_preview = None
            self.live = None
            self.left_receipts.clear()
            self.status = {'state': 'reconstructing', 'run_dir': str(run_dir),
                           'message': '원본에서 TSDF 다시 생성 중', 'counts': {}, 'dropped': 0}
            self.thread = threading.Thread(target=self._rebuild,
                args=(Path(run_dir), global_reconstruction, color), daemon=False)
            self.thread.start()

    def _rebuild(self, run_dir: Path, global_reconstruction: bool = True, color: bool = True) -> None:
        try:
            from experiments.handheld_mapping.reconstruct import reconstruct
            backend = next((name for name in ('orb_offline', 'orb', 'orb_inertial')
                            if any((run_dir / name).glob('map_*_keyframes.csv'))), 'orb_offline')
            if global_reconstruction:
                result = self._postprocess_global(run_dir, backend, color=color)
                self.update(reference_reconstruction=result)
                if result['maps'] and all(row['optimization_status'] != 'optimized_unverified'
                                          for row in result['maps']):
                    self.update(reference_warning='추가 depth 제약 부족 · 기존 위치를 유지한 재융합 결과')
            else:
                output = run_dir / 'reference_reconstruction' / uuid.uuid4().hex
                result = reconstruct(run_dir, progress=lambda message: self.update(message=message),
                                     backend=backend, color=color, output_dir=output)
                self.update(dense=result, reconstruction_output=str(output))
            state = result['status'] if result['status'] in ('complete', 'partial', 'failed') else 'partial'
            message = '새 폴더에 지도 재생성 결과 저장'
            if state == 'failed':
                failures = result.get('failures', [])
                reason = failures[0].get('error', '유효한 결과 없음') if failures else '유효한 결과 없음'
                message = '모든 지도 재생성 실패: '+reason
            self.update(state=state, message=message)
        except Exception as error:
            self.update(state="failed", message=str(error), error=str(error))
        finally:
            with self.lock:
                if not self.ended_at:
                    self.ended_at = time.monotonic()
