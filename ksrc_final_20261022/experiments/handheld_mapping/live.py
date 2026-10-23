# ================================================================
# live.py
# 설명: 원본 기록을 기다리게 하지 않는 native ORB worker와 잠정 관측 지도를 관리한다.
# 사용법: from experiments.handheld_mapping.live import LiveMapper
# ================================================================
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

import cv2
import numpy as np

from experiments.handheld_mapping.compare import ROOT, runner_environment, vocabularies
from experiments.handheld_mapping.coverage import CoverageGrid, StereoMailbox
from experiments.handheld_mapping.offline import write_settings
from experiments.handheld_mapping.overlap import overlap_image
from experiments.handheld_mapping.reconstruct import pose_matrix
from experiments.handheld_mapping.storage import atomic_json


class LiveMapper:
    """bounded FIFO로 연속 IR을 추적하며 최신 시각화와 raw 기록을 분리하는 worker."""

    def __init__(self, run: Path, command: list[str] | None = None, result_timeout: float = 30.,
                 demo: bool = False, gravity_provider=None) -> None:
        self.run = Path(run).resolve()
        self.folder = self.run/'orb_live'
        self.folder.mkdir(exist_ok=False)
        calibration = json.loads((self.run/'capture_calibration.json').read_text())
        self.info = calibration['depth']
        self.left_info = calibration.get('stereo',{}).get('left')
        self.mailbox = StereoMailbox(self.run)
        self.grid = CoverageGrid()
        self.gravity_provider = gravity_provider
        self.gravity_frames = {}
        self.gravity_error = '중력 기준 대기 · 3초간 정지하세요' if gravity_provider else None
        self.left_body = None
        self.body_attitude = None
        self.tilt_log = None
        if gravity_provider is not None:
            from experiments.handheld_mapping.gravity import mounting,BodyIncline
            self.left_body = mounting(self.run)
            self.incline = BodyIncline()
        self.lock = threading.RLock()
        self.error = None
        self.processed = self.no_depth = 0
        self.tracking_state = -1
        self.pose_valid = False
        self.skipped_visualizations = 0
        self.lag_ms = self.process_ms = 0.
        self.last_result_at = 0.
        self.result_timeout = result_timeout
        self.map_changes = 0
        self.finished = False
        self.overlay = self.overlay_original = None
        self.overlay_stamp_ns = None
        self.overlay_error = None
        self.overlay_ms = 0.
        self.overlay_generated_at = 0.
        self.overlay_map_id = None
        self.demo = demo
        self.process = self.thread = None
        if demo:
            return
        if command is None:
            settings = write_settings(self.run, self.folder)
            orb_voc, line_voc = vocabularies()
            command = [str(ROOT/'outputs/handheld_mapping/orb3_comparison_build/rover_orb3_compare'),
                       str(orb_voc), str(line_voc), str(settings), '-', str(self.folder)]
        self.log = (self.folder/'runner.log').open('wb')
        try:
            self.process = subprocess.Popen(command, cwd=self.folder, env=runner_environment(True),
                                            stdin=subprocess.PIPE, stdout=self.log, stderr=self.log,
                                            text=True, bufsize=1, start_new_session=True)
        except Exception:
            self.log.close()
            raise
        self.thread = threading.Thread(target=self._work, name='live-orb', daemon=False)
        self.thread.start()

    def offer(self, header: dict) -> None:
        if not self.demo or header['stream'] == 'depth':
            self.mailbox.offer(header)

    def offer_pose(self, header: dict) -> None:
        """합성 시험의 알려진 pose만 표시하며 native 추정 결과로 취급하지 않는다."""
        if not self.demo:
            return
        frame = self.mailbox.depth_for(header['stamp_ns'])
        if frame is None:
            return
        values = [*header['position'], *header['quaternion_xyzw']]
        pose = pose_matrix(dict(zip(('x','y','z','qx','qy','qz','qw'),values)))
        depth = cv2.imread(frame['path'],cv2.IMREAD_UNCHANGED)
        with self.lock:
            self.grid.update(header['map_id'],pose,depth,self.info)
            self._set_overlay(header['stamp_ns'],str(self.run/'raw/left'/f"{header['stamp_ns']}.png"),pose,depth)
            self.processed += 1
            self.tracking_state = 2
            self.pose_valid = True
            self.last_result_at = time.monotonic()

    def snapshot(self) -> dict:
        with self.lock:
            result = {**self.grid.snapshot(), 'error':self.error, 'processed':self.processed,
                      'tracking_state':self.tracking_state, 'no_depth':self.no_depth,
                      'pose_valid':self.pose_valid, 'skipped_visualizations':self.skipped_visualizations,
                      'lag_ms':self.lag_ms, 'process_ms':self.process_ms,
                      'stale':not self.last_result_at or self.lag_ms/1000+time.monotonic()-self.last_result_at > 1.,
                      'map_changes':self.map_changes, 'finished':self.finished, 'synthetic':self.demo}
            result.update(gravity_requested=self.gravity_provider is not None,gravity_error=self.gravity_error,
                          gravity_source='stationary_IMU_URDF_provisional' if self.grid.gravity_aligned else None,
                          body_attitude=dict(self.body_attitude) if self.body_attitude else None)
            result.update(overlay=self.overlay.copy() if self.overlay is not None else None,
                          overlay_original=self.overlay_original.copy() if self.overlay_original is not None else None,
                          overlay_stamp_ns=self.overlay_stamp_ns, overlay_error=self.overlay_error,
                          overlay_ms=self.overlay_ms)
        return {**result, **self.mailbox.snapshot()}

    def _set_overlay(self, stamp: int, left_path: str, pose: np.ndarray | None, depth: np.ndarray | None) -> None:
        """같은 processed frame의 영상 두 장만 유지하며 overlay 오류를 추적 오류와 분리한다."""
        if (pose is not None and depth is not None and self.overlay is not None
                and self.overlay_map_id == self.grid.map_id
                and time.monotonic()-self.overlay_generated_at < .2):
            return
        started = time.perf_counter()
        self.overlay_generated_at = time.monotonic()
        self.overlay_map_id = self.grid.map_id
        self.overlay_original = None
        self.overlay_stamp_ns = stamp
        self.overlay = None
        self.overlay_error = None
        try:
            self.overlay_original = cv2.imread(left_path,cv2.IMREAD_GRAYSCALE)
            if self.overlay_original is None:
                raise ValueError('오버레이 IR 원본 읽기 실패')
            if pose is not None and depth is not None:
                if self.left_info is None:
                    raise ValueError('오버레이 IR 보정값 없음')
                self.overlay = overlap_image(self.overlay_original,depth,self.info,self.left_info,pose,self.grid)
        except Exception as error:
            self.overlay_error = str(error)
        self.overlay_ms = (time.perf_counter()-started)*1000

    def _line(self, stream) -> str:
        deadline = time.monotonic()+self.result_timeout
        value = ''
        while time.monotonic() < deadline:
            value += stream.readline()
            if len(value) > 65536:
                raise ValueError('live pose row too large')
            if value.endswith('\n'):
                return value
            if self.process.poll() is not None:
                raise RuntimeError(f'실시간 ORB 종료: exit={self.process.returncode}')
            time.sleep(.005)
        raise TimeoutError('실시간 ORB pose 대기 시간 초과')

    def _save_grid(self) -> None:
        if self.grid.map_id is not None:
            np.savez_compressed(self.folder/f'coverage_map_{self.grid.map_id}.npz',
                                observations=self.grid.counts, bounds=self.grid.bounds,
                                resolution=self.grid.resolution,
                                coordinate_system=self.grid.snapshot()['coordinate_system'])

    def _gravity_pose(self, pose: np.ndarray, map_id: int, receipt_ns: int | None) -> np.ndarray | None:
        if self.gravity_provider is None:
            return pose
        if map_id not in self.gravity_frames:
            from experiments.handheld_mapping.gravity import stationary_reference, alignment_matrix
            try:
                if type(receipt_ns) is not int:
                    raise ValueError('카메라의 Pi 수신 시각이 없어 중력 기준을 정할 수 없습니다')
                reference = stationary_reference(self.gravity_provider(),receipt_ns-1_000_000_000)
                matrix = alignment_matrix(pose,self.left_body,reference['body_up'])
            except ValueError as error:
                self.gravity_error = str(error)
                return None
            self.gravity_frames[map_id] = matrix
            atomic_json(self.folder/f'gravity_map_{map_id}.json',
                        {'map_id':map_id,'world_from_slam':matrix.tolist(),'stationary_reference':reference,
                         'final_map':False,'orb_fused':False,'extrinsic_verified':False,
                         'clock_basis':'same_Pi_receipts_stationary_window_only'})
        if self.grid.map_id != map_id or not self.grid.gravity_aligned:
            self.grid = CoverageGrid(gravity_aligned=True)
        self.gravity_error = None
        return self.gravity_frames[map_id] @ pose

    def _work(self) -> None:
        reader = None
        fields = None
        try:
            while True:
                pair = self.mailbox.take()
                if pair is None:
                    if self.mailbox.closed: break
                    if self.process.poll() is not None:
                        raise RuntimeError(f'실시간 ORB 종료: exit={self.process.returncode}')
                    continue
                self.process.stdin.write(f"{pair['stamp_ns']}\t{pair['left']}\t{pair['right']}\n")
                self.process.stdin.flush()
                if reader is None:
                    path = self.folder/'live_poses.csv'
                    deadline = time.monotonic()+self.result_timeout
                    while not path.exists():
                        if self.process.poll() is not None:
                            raise RuntimeError(f'실시간 ORB 종료: exit={self.process.returncode}')
                        if time.monotonic() > deadline:
                            raise TimeoutError('실시간 ORB 시작 시간 초과')
                        time.sleep(.01)
                    reader = path.open()
                    fields = next(csv.reader([self._line(reader)]))
                values = next(csv.reader([self._line(reader)]))
                if len(fields) != 20 or len(values) != 20:
                    raise ValueError('invalid live pose columns')
                row = dict(zip(fields,values))
                stamp, state, map_id = int(row['stamp_ns']), int(row['state']), int(row['map_id'])
                if stamp != pair['stamp_ns']:
                    raise ValueError('live pose/image timestamp mismatch')
                pose = np.array([float(row[f'm{i}{j}']) for i in range(4) for j in range(4)]).reshape(4,4)
                process_ms = float(row['process_ms'])
                if not np.isfinite(process_ms) or process_ms < 0:
                    raise ValueError('invalid live processing time')
                pose_received_at = time.monotonic()
                frame = self.mailbox.depth_for(stamp)
                with self.lock:
                    self.processed += 1
                    self.tracking_state = state
                    self.pose_valid = state == 2 and map_id >= 0
                    self.process_ms = process_ms
                    self.last_result_at = pose_received_at
                    self.lag_ms = (self.last_result_at-pair['offered_at'])*1000
                    if self.pose_valid:
                        map_changed = self.grid.map_id != map_id
                        if self.grid.map_id is not None and self.grid.map_id != map_id:
                            self._save_grid()
                            self.map_changes += 1
                        pose = self._gravity_pose(pose,map_id,pair.get('pi_receipt_monotonic_ns'))
                        if pose is None:
                            # 다른 map 좌표를 보여주거나 정렬 전/후 관측을 섞지 않는다.
                            self.grid = CoverageGrid()
                            self.overlay = self.overlay_original = None
                            self.body_attitude = None
                            continue
                        if self.gravity_provider is not None:
                            self.body_attitude = self.incline.update(map_id,stamp,pose@self.left_body)
                            if self.tilt_log is None:
                                self.tilt_log = (self.folder/'body_incline.jsonl').open('x',buffering=1)
                            self.tilt_log.write(json.dumps(self.body_attitude,allow_nan=False)+'\n')
                        self.grid.update_pose(map_id,pose)
                        if map_changed:
                            self.overlay = self.overlay_original = None
                            self.overlay_stamp_ns = None
                            self.overlay_error = None
                            self.overlay_generated_at = 0.
                        if frame is None:
                            self.no_depth += 1
                        if map_changed or time.monotonic()-self.overlay_generated_at >= .2:
                            depth = None
                            if frame is not None:
                                depth = cv2.imread(frame['path'], cv2.IMREAD_UNCHANGED)
                                if depth is None: raise ValueError('실시간 depth 파일 읽기 실패')
                                self.grid.update(map_id,pose,depth,self.info)
                            self._set_overlay(stamp,pair['left'],pose,depth)
                        else:
                            self.skipped_visualizations += 1
                    else:
                        # 이전 격자·경로는 남기되 손실 프레임을 유효 관측처럼 표시하지 않는다.
                        self.overlay = self.overlay_original = None
                        self.overlay_stamp_ns = stamp
                        self.overlay_error = None
                        self.overlay_generated_at = 0.
            self.process.stdin.close()
            code = self.process.wait(timeout=20)
            if code != 0:
                raise RuntimeError(f'실시간 ORB 종료: exit={code}')
        except Exception as error:
            with self.lock:
                self.error = str(error)
                self.pose_valid = False
        finally:
            if reader is not None: reader.close()
            if not self.process.stdin.closed:
                try: self.process.stdin.close()
                except OSError: pass
            # Parent가 먼저 종료해도 같은 owned group에 남은 descendant를 정리한다.
            try:
                os.killpg(self.process.pid,signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid,signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait()
            deadline = time.monotonic()+.3
            while time.monotonic() < deadline:
                try:
                    os.killpg(self.process.pid,0)
                except ProcessLookupError:
                    break
                time.sleep(.01)
            else:
                try:
                    os.killpg(self.process.pid,signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self.log.close()
            if self.tilt_log is not None:self.tilt_log.close()
            with self.lock:
                try: self._save_grid()
                except Exception as error: self.error = self.error or str(error)
                self.finished = True
            self._write_summary()

    def _write_summary(self) -> None:
        summary = self.snapshot()
        for key in ('image','path','camera','overlay','overlay_original'): summary.pop(key)
        summary.update(status='failed' if self.error else 'complete',
                       coordinate_system=self.grid.snapshot()['coordinate_system'],
                       lag_scope='laptop stereo saved to live pose received; excludes sensor/network/display',
                       final_map=False)
        try: atomic_json(self.folder/'summary.json',summary)
        except OSError as error:
            with self.lock: self.error = self.error or str(error)

    def finish(self) -> None:
        self.mailbox.close()
        if self.demo:
            with self.lock:
                self._save_grid()
                self.finished = True
            self._write_summary()
            return
        self.thread.join(timeout=20)
        if self.thread.is_alive():
            with self.lock: self.error = self.error or '실시간 ORB 종료 시간 초과'
            if self.process.poll() is None: os.killpg(self.process.pid, signal.SIGTERM)
            self.thread.join(timeout=5)
            if self.thread.is_alive() and self.process.poll() is None:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.thread.join(timeout=3)
