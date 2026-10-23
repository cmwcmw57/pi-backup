# ================================================================
# app.py
# 설명: 노트북에서 시작·종료 버튼으로 D435 기록과 TSDF 지도 생성을 제어한다.
# 사용법: bash tools/handheld_mapping.sh [--demo]
# ================================================================
import argparse
import os
from pathlib import Path
import sys
import signal
import subprocess

from PyQt5.QtCore import Qt, QTimer, QUrl
from PyQt5.QtGui import QDesktopServices, QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QGridLayout,
                            QHBoxLayout, QInputDialog, QLabel, QLineEdit, QPushButton, QSizePolicy, QVBoxLayout, QWidget)

from experiments.handheld_mapping.controller import ACTIVE, ROOT, Session
from experiments.handheld_mapping.live_preview import CoverageView


class MappingWindow(QWidget):
    """촬영 상태를 표시하고 blocking 작업을 session thread에 맡기는 노트북 제어창."""

    def __init__(self, session: Session, output_root: Path | None = None, demo: bool = False,
                 imu_calibration: Path | None = None, inertial_fusion: bool = False,
                 projector_on: bool = False, camera_fps: int = 15) -> None:
        super().__init__()
        self.session = session
        self.projector_on = projector_on
        self.camera_fps = camera_fps
        self.close_pending = False
        self.setWindowTitle("D435 · Handheld Mapping")
        self.resize(1200, 980)
        self.setStyleSheet("""
            QWidget { background:#111820; color:#e8edf2; font-size:14px; }
            QLineEdit,QComboBox { background:#1b2733; border:1px solid #344556; border-radius:6px; padding:9px; }
            QPushButton { background:#243445; border:1px solid #41566a; border-radius:7px; padding:12px 18px; }
            QPushButton:hover { background:#334d64; }
            QPushButton:disabled { color:#647382; background:#19232d; border-color:#253340; }
            QPushButton#start { background:#176951; border-color:#259b79; font-weight:bold; }
            QPushButton#stop { background:#713b33; border-color:#a46458; font-weight:bold; }
            QLabel#heading { font-size:20px; font-weight:bold; }
            QLabel#status { color:#80d5bb; font-size:18px; padding:4px 0; }
            QLabel#preview { background:#090e14; border:1px solid #2c3a48; border-radius:8px; }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, 10, 26, 10)
        heading = QLabel("D435  /  HANDHELD MAPPING")
        heading.setObjectName("heading")
        layout.addWidget(heading)
        fields = QGridLayout()
        self.host = QLineEdit("pi@raspberrypi.local")
        self.output = QLineEdit(str(output_root or ROOT / "outputs/handheld_mapping/sessions"))
        self.mode = QComboBox()
        self.mode.addItem("ORB-SLAM3 + TSDF", False)
        line_runner = ROOT / "outputs/handheld_mapping/orbline_build/rover_orbline"
        self.mode.addItem("ORB + TSDF → 점+선 비교 (촬영 후 노트북)", True)
        self.mode.model().item(1).setEnabled(line_runner.is_file())
        self.demo = QCheckBox("합성 입력으로 시험 (카메라 연결 없음)")
        self.demo.setChecked(demo)
        self.point_ready = (ROOT/'outputs/handheld_mapping/orb3_comparison_build/rover_orb3_compare').is_file()
        self.sensors_only = QCheckBox('Pi는 센서 전송만 · 종료 후 노트북에서 ORB 계산')
        self.sensors_only.setChecked(self.point_ready)
        self.sensors_only.setToolTip('Pi에 Atlas를 만들지 않습니다. 노트북 native ORB 실행기가 필요합니다.')
        self.live_mapping = QCheckBox('실시간 촬영 범위 표시 · 노트북에서 ORB 동시 실행')
        self.live_mapping.setChecked(self.point_ready)
        self.live_mapping.setToolTip('원본 저장과 별도로 최신 입력을 추적하며 종료 후 최종 지도를 다시 계산합니다.')
        calibration = imu_calibration or ROOT / "outputs/handheld_mapping/calibration/imu_calibration.json"
        self.imu_enabled = QCheckBox("보정 IMU 함께 기록")
        self.imu_enabled.setChecked(calibration.expanduser().is_file() and not demo)
        self.imu_enabled.setToolTip("raw·보정 IMU를 원본 영상과 같은 폴더에 저장합니다. ORB 융합과 시계 동기화는 검증되지 않았습니다.")
        self.imu_calibration = QLineEdit(str(calibration))
        self.inertial_fusion = QCheckBox("종료 후 IMU 융합 시험")
        self.inertial_fusion.setToolTip("실시간 기본 stereo, 종료 후 별도 관성 결과")
        self.inertial_fusion.setChecked(inertial_fusion and not demo)
        self.rgb_enabled = QCheckBox('RGB 함께 기록 · 컬러 지도')
        self.rgb_enabled.setChecked(not demo)
        self.global_reconstruction = QCheckBox('종료 후 depth 구간 정합·전역 재구성')
        self.global_reconstruction.setChecked(not demo)
        if self.inertial_fusion.isChecked():
            self.imu_enabled.setChecked(True)
        self.inertial_fusion.toggled.connect(lambda checked: self.imu_enabled.setChecked(True) if checked else None)
        self.imu_enabled.toggled.connect(lambda checked: self.inertial_fusion.setChecked(False) if not checked else None)
        self.imu_zero_button = QPushButton('평지 영점')
        self.imu_zero_button.setToolTip('알려진 평지에서 3초 정지한 뒤 누르세요. IMU 참고 각도의 표시 영점만 설정하며 지도와 센서 보정값은 바꾸지 않습니다.')
        self.imu_zero_button.clicked.connect(self.zero_imu_display)
        self.gravity_enabled = QCheckBox('중력 기준 지도와 높이·경사·단차 생성 · 시작 시 3초 정지')
        self.gravity_enabled.setChecked(False)
        self.gravity_enabled.setToolTip('정지 IMU와 카메라 장착 모델로 위쪽을 정합니다. 동적 IMU 융합과 실측 경사 검증은 별도입니다.')
        fields.addWidget(QLabel("Pi 연결"), 0, 0)
        fields.addWidget(self.host, 0, 1)
        fields.addWidget(QLabel("저장 폴더"), 1, 0)
        fields.addWidget(self.output, 1, 1)
        choose = QPushButton("선택")
        choose.clicked.connect(self.choose_output)
        fields.addWidget(choose, 1, 2)
        fields.addWidget(QLabel("처리 방식"), 2, 0)
        fields.addWidget(self.mode, 2, 1)
        options = QGridLayout()
        for index, widget in enumerate((self.demo, self.sensors_only, self.live_mapping, self.imu_enabled,
                                        self.inertial_fusion, self.rgb_enabled, self.global_reconstruction,
                                        self.gravity_enabled)):
            options.addWidget(widget, index//2, index%2)
        fields.addLayout(options, 3, 1)
        fields.addWidget(QLabel("IMU 보정 파일"), 4, 0)
        fields.addWidget(self.imu_calibration, 4, 1)
        fields.addWidget(self.imu_zero_button, 4, 2)
        self.choose_button = choose
        layout.addLayout(fields)
        self.status_label = QLabel("준비 · 시작 버튼을 누르세요")
        self.status_label.setObjectName("status")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.metrics = QLabel("촬영 시간 00:00   |   저장 프레임 0   |   추적 대기")
        layout.addWidget(self.metrics)
        self.imu_status = QLabel("IMU 기록 꺼짐 · ORB 융합 안 함 · 시계 동기화 미검증")
        self.imu_status.setWordWrap(True)
        layout.addWidget(self.imu_status)
        self.preview = QLabel("카메라 미리보기\n촬영을 시작하면 좌측 IR 영상이 표시됩니다")
        self.preview.setObjectName("preview")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(120)
        self.preview.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Ignored)
        self.overlay_preview = QLabel('촬영하면 관측 횟수 색이 영상 위에 표시됩니다')
        self.overlay_preview.setObjectName('preview')
        self.overlay_preview.setAlignment(Qt.AlignCenter)
        self.overlay_preview.setMinimumHeight(120)
        self.overlay_preview.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Ignored)
        self.overlay_note = QLabel('관측 횟수 오버레이 · 노랑 적음 / 초록 여러 시점')
        self.overlay_note.setWordWrap(True)
        cameras = QVBoxLayout()
        self.raw_caption = QLabel('RGB 수집 선택 시 위 RGB · 아래 IR 관측 오버레이')
        cameras.addWidget(self.raw_caption)
        cameras.addWidget(self.preview,1)
        cameras.addWidget(self.overlay_note)
        cameras.addWidget(self.overlay_preview,1)
        views = QHBoxLayout()
        views.addLayout(cameras,1)
        self.coverage = CoverageView()
        views.addWidget(self.coverage,1)
        layout.addLayout(views,1)
        self.path_label = QLabel("매번 새 날짜별 폴더를 만들며 기존 기록을 덮어쓰지 않습니다")
        self.path_label.setWordWrap(True)
        layout.addWidget(self.path_label)
        controls = QHBoxLayout()
        self.start_button, self.stop_button = QPushButton("매핑 시작"), QPushButton("종료 및 지도 생성")
        self.start_button.setObjectName("start")
        self.stop_button.setObjectName("stop")
        self.open_button, self.rebuild_button = QPushButton("저장 폴더 열기"), QPushButton("기록에서 다시 생성")
        self.view_button = QPushButton('3D 지도 보기')
        self.edit_button = QPushButton('지도 편집·출발점')
        self.terrain_button = QPushButton("높이·경사 보기")
        for button in (self.start_button, self.stop_button, self.view_button, self.open_button, self.rebuild_button, self.edit_button, self.terrain_button):
            controls.addWidget(button)
        layout.addLayout(controls)
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(session.stop)
        self.open_button.clicked.connect(self.open_folder)
        self.rebuild_button.clicked.connect(self.rebuild)
        self.view_button.clicked.connect(self.view_map)
        self.edit_button.clicked.connect(self.edit_map)
        layout.addWidget(QLabel(f"입력 요청{camera_fps} FPS · projector {'ON(실험)' if projector_on else 'OFF'} · depth 0.25~1m · 노트북 저장 | RGB 추가 부하는 실기 확인 필요"))
        self.terrain_button.clicked.connect(self.view_terrain)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(200)
        self.demo.toggled.connect(self.refresh)
        self.imu_enabled.toggled.connect(self.refresh)
        self.refresh()

    def choose_output(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "노트북 저장 폴더", self.output.text())
        if selected:
            self.output.setText(selected)

    def start(self) -> None:
        try:
            self.session.start({"host": self.host.text().strip(), "output_root": self.output.text(),
                                "demo": self.demo.isChecked(), "compare_lines": self.mode.currentData(),
                                "sensors_only": self.sensors_only.isChecked(),
                                "live_mapping": self.live_mapping.isChecked(),
                                "imu_enabled": self.imu_enabled.isChecked() and not self.demo.isChecked(),
                                "inertial_fusion": self.inertial_fusion.isChecked() and not self.demo.isChecked(),
                                "imu_calibration": self.imu_calibration.text().strip(),
                                "projector_on": self.projector_on and not self.demo.isChecked(),
                                "camera_fps": self.camera_fps,
                                "rgb_enabled": self.rgb_enabled.isChecked() and not self.demo.isChecked(),
                                "gravity_enabled": self.gravity_enabled.isChecked(),
                                "global_reconstruction": self.global_reconstruction.isChecked()})
        except Exception as error:
            self.status_label.setText(str(error))
        self.refresh()

    def open_folder(self) -> None:
        path = self.session.snapshot()["run_dir"]
        if path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def zero_imu_display(self) -> None:
        try:
            self.session.zero_imu_display()
        except (ValueError,OSError) as error:
            self.session.update(message=str(error))
        self.refresh()

    def rebuild(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "원본 기록 폴더 선택", self.output.text())
        if selected:
            self.session.rebuild(Path(selected), self.global_reconstruction.isChecked(), self.rgb_enabled.isChecked())
            self.refresh()

    def result_target(self) -> str | None:
        """현재 세션의 명시 결과만 고르고 취소와 아직 결과 없음을 구분한다."""
        status = self.session.snapshot()
        results = status.get('reference_reconstruction', {}).get('maps', [])
        if results:
            if len(results) > 1:
                labels = [f"지도 {row['map_id']} · {row['optimization_status']}" for row in results]
                label, accepted = QInputDialog.getItem(self, '편집할 지도', '좌표계가 다른 지도는 각각 편집합니다', labels, 0, False)
                if not accepted:
                    return None
                return results[labels.index(label)]['output']
            else:
                return results[0]['output']
        return status.get('reconstruction_output') or status.get('run_dir', '')

    def edit_map(self) -> None:
        """현재 성공한 전역 결과 또는 사용자가 고른 기록을 독립 편집 창으로 연다."""
        selected = self.result_target()
        if selected is None:
            return
        selected = selected or QFileDialog.getExistingDirectory(
            self, '지도·촬영·편집 버전 폴더 선택', self.output.text())
        if not selected:
            return
        try:
            with (Path(selected) / 'reference_editor.log').open('ab') as log:
                subprocess.Popen([sys.executable, '-m', 'experiments.handheld_mapping.reference_editor', selected],
                                 cwd=ROOT, stdout=log, stderr=log, start_new_session=True)
        except OSError as error:
            self.session.update(message=str(error))
            self.refresh()

    def view_map(self) -> None:
        from experiments.handheld_mapping.viewer import select_maps
        selected = self.result_target()
        if selected is None:
            return
        selected = selected or QFileDialog.getExistingDirectory(
            self,'3D 지도가 저장된 촬영 폴더 선택',self.output.text())
        if not selected:
            return
        try:
            select_maps(Path(selected))
            environment = dict(os.environ)
            if environment.get('DISPLAY'):
                environment.pop('WAYLAND_DISPLAY',None)
                environment['XDG_SESSION_TYPE'] = 'x11'
            with (Path(selected)/'map_viewer.log').open('ab') as log:
                subprocess.Popen([sys.executable,'-m','experiments.handheld_mapping.viewer',selected],
                                 cwd=ROOT,env=environment,stdout=log,stderr=log,start_new_session=True)
        except (ValueError,OSError) as error:
            self.session.update(message=str(error))
            self.refresh()

    def view_terrain(self) -> None:
        selected = self.session.snapshot()['run_dir'] or QFileDialog.getExistingDirectory(
            self,'지형 지도가 저장된 촬영 폴더 선택',self.output.text())
        if not selected:
            return
        terrain = self.session.snapshot().get('terrain',{})
        paths = [Path(row['output_dir'])/'preview.png' for row in terrain.get('maps',{}).values()
                 if row.get('output_dir') and (Path(row['output_dir'])/'preview.png').is_file()]
        if not paths:
            paths = sorted((Path(selected)/'terrain').glob('*/map_*/preview.png'))
        if not paths:
            self.session.update(message='저장된 지형 지도가 없습니다. 중력 기준과 정지 IMU 구간을 확인하세요.')
            self.refresh()
            return
        for path in paths:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def refresh(self) -> None:
        status = self.session.snapshot()
        active = status["state"] in ACTIVE or (self.session.thread is not None and self.session.thread.is_alive())
        self.start_button.setEnabled(not active)
        self.stop_button.setEnabled(status["state"] in {"starting", "recording"})
        self.rebuild_button.setEnabled(not active)
        self.view_button.setEnabled(not active)
        self.edit_button.setEnabled(not active)
        self.rgb_enabled.setEnabled(not active and not self.demo.isChecked())
        self.global_reconstruction.setEnabled(not active)
        self.terrain_button.setEnabled(not active)
        self.open_button.setEnabled(bool(status["run_dir"]))
        for widget in (self.host, self.output, self.mode, self.demo, self.choose_button):
            widget.setEnabled(not active)
        self.sensors_only.setEnabled(not active and self.point_ready)
        self.live_mapping.setEnabled(not active and self.point_ready)
        if self.demo.isChecked():
            self.inertial_fusion.setChecked(False)
            self.imu_enabled.setChecked(False)
        self.inertial_fusion.setEnabled(not active and not self.demo.isChecked())
        self.imu_enabled.setEnabled(not active and not self.demo.isChecked())
        self.imu_calibration.setEnabled(not active and not self.demo.isChecked())
        if not self.imu_enabled.isChecked() or self.demo.isChecked():
            self.gravity_enabled.setChecked(False)
        self.gravity_enabled.setEnabled(not active and self.imu_enabled.isChecked() and not self.demo.isChecked())
        imu = status.get("imu")
        self.imu_zero_button.setEnabled(status['state']=='recording' and bool(imu) and not imu.get('stale',True))
        imu_text = "IMU 유효 샘플 대기" if self.imu_enabled.isChecked() else "IMU 기록 꺼짐"
        if imu:
            imu_text = (f"IMU 기록 {imu.get('state', '대기')} · 유효 {imu.get('valid', 0)}"
                        f" · 고유 {imu.get('unique', 0)} / {imu.get('hz', 0):.1f} Hz")
            if imu.get("roll_deg") is not None and imu.get("pitch_deg") is not None:
                roll = imu.get('display_roll_deg',imu['roll_deg'])
                pitch = imu.get('display_pitch_deg',imu['pitch_deg'])
                imu_text += f" · 가속도 참고 roll {roll:.1f}° / pitch {pitch:.1f}°"
                if imu.get('display_zero_set'):imu_text += ' (평지 영점 기준)'
            if imu.get("stale"):
                imu_text += " · 수신 지연"
            if imu.get('source_gap_events',0):
                imu_text += f" · 기록 중 표본 공백 {imu['source_gap_events']}회"
            if imu.get("error"):
                imu_text += " · 오류: "+str(imu["error"])
        gravity = status.get('gravity')
        if gravity:
            imu_text += f" · 중력 기준 {len(gravity.get('maps',{}))}개 지도 / {gravity['status']}"
        elif self.gravity_enabled.isChecked():
            imu_text += ' · 중력 기준 준비: 촬영 시작 후 3초간 정지'
        fusion_text = "ORB 융합 안 함"
        if status.get("inertial_warning"):
            fusion_text = status["inertial_warning"]
        elif status.get("inertial_dense", {}).get("status") == "complete":
            fusion_text = "IMU 융합 지도 저장"
        elif self.inertial_fusion.isChecked():
            fusion_text = "종료 후 IMU 융합 시험"
        self.imu_status.setText(imu_text+" · "+fusion_text+" · 시계 동기화 미검증")
        live = self.session.live_snapshot()
        if live and live.get('body_attitude') and not live.get('stale') and live.get('tracking_state') == 2:
            body = live['body_attitude']
            self.imu_status.setText(self.imu_status.text()+f" · 중력 기준 차체 기울기 {body['tilt_deg']:.1f}°"
                                    + (' · 기울어진 구간' if body['body_tilted'] else ''))
        self.coverage.set_snapshot(live)
        label = status["message"]
        if status.get('reference_warning'):
            label += ' · '+status['reference_warning']
        if status.get('color_warning'):
            label += ' · '+status['color_warning']
        if self.close_pending and active:
            label = "저장이 끝나면 창을 닫습니다 · "+label
        if self.demo.isChecked():
            label = "[합성 시험] "+label
        self.status_label.setText(label)
        elapsed = int(status["elapsed_s"])
        counts = status.get("counts", {})
        recovery = '기존 지도 복구 중 · 정지해서 이전 장면 보기'
        tracking_labels = {2:'정상',3:recovery,4:'손실'}
        tracking = tracking_labels.get(status.get("tracking_state"), "대기")
        if status.get('capture_mode') == 'sensors_only':
            tracking = (tracking_labels.get(live.get('tracking_state'),'준비')
                        if live and not live.get('error') else '노트북 후처리')
        self.metrics.setText(f"시간 {elapsed//60:02d}:{elapsed%60:02d}  |  저장 IR {counts.get('left', 0)}/{counts.get('right', 0)}"
                             f" · depth {counts.get('depth', 0)} · RGB {counts.get('color', 0)}  |  수신 {status.get('left_receipt_fps', 0):.1f} FPS"
                             f"  |  추적 {tracking}  |  전송 누락 {status.get('dropped', 0)}")
        if status["run_dir"]:
            self.path_label.setText(status["run_dir"])
        with self.session.lock:
            image = self.session.preview
            if image is not None:
                image = image.copy()
            color_image = self.session.color_preview
            if color_image is not None and self.rgb_enabled.isChecked():
                color_image = color_image[:, :, ::-1].copy()
            else:
                color_image = None
        overlay = None
        note = '관측 횟수 오버레이 · 노랑 적음 / 초록 여러 시점'
        if live and not live.get('error') and (not live.get('stale') or live.get('finished')):
            if live.get('tracking_state') == 2 and live.get('overlay_original') is not None:
                image = live['overlay_original']
            if live.get('tracking_state') == 2 and not live.get('overlay_error'):
                overlay = live.get('overlay')
            if live.get('overlay_error'):
                note = '오버레이 표시 대기 · '+live['overlay_error']
            elif overlay is None:
                note = '오버레이 색 없음 · 추적 또는 depth 대기'
            elif live.get('finished'):
                note += ' · 촬영 종료 프레임'
        elif live:
            note = '오버레이 색 없음 · '+('실시간 추적 오류' if live.get('error') else '최신 위치 대기')
        if live and not live.get('error') and live.get('tracking_state') == 3:
            note = recovery+' · 이전 촬영 범위 유지'
        self.overlay_note.setText(note)
        if image is not None or color_image is not None:
            if color_image is not None:
                h, w = color_image.shape[:2]
                qimage = QImage(color_image.data, w, h, color_image.strides[0], QImage.Format_RGB888).copy()
                self.raw_caption.setText('원본 RGB · 아래 IR 관측 오버레이')
            else:
                h, w = image.shape
                qimage = QImage(image.data, w, h, image.strides[0], QImage.Format_Grayscale8).copy()
                self.raw_caption.setText('원본 IR · RGB 미수집 또는 수신 대기')
            self.preview.setPixmap(QPixmap.fromImage(qimage).scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            if overlay is None:
                overlay_qimage = (QImage(image.data, image.shape[1], image.shape[0], image.strides[0],
                                        QImage.Format_Grayscale8).copy() if image is not None else qimage)
            else:
                overlay_qimage = QImage(overlay.data,overlay.shape[1],overlay.shape[0],overlay.strides[0],QImage.Format_RGB888).copy()
            self.overlay_preview.setPixmap(QPixmap.fromImage(overlay_qimage).scaled(
                self.overlay_preview.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))
        else:
            self.preview.clear()
            self.preview.setText('카메라 수신 대기')
            self.overlay_preview.clear()
            self.overlay_preview.setText('동일 프레임의 관측 오버레이 대기')
        if self.close_pending and not active:
            self.close()

    def closeEvent(self, event) -> None:
        active = self.session.snapshot()["state"] in ACTIVE or (self.session.thread is not None and self.session.thread.is_alive())
        if active:
            self.close_pending = True
            self.session.stop()
            event.ignore()
        else:
            event.accept()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--imu-calibration", type=Path)
    parser.add_argument("--imu-fusion", action="store_true", help="촬영 종료 후 별도 IMU 융합 시험 (IMU 기록 자동 활성화)")
    parser.add_argument("--projector-on", action="store_true", help="D435 IR projector 켬 (ORB 추적 영향 비교용 실험)")
    parser.add_argument("--camera-fps", type=int, choices=(6, 15), default=15, help="IR/depth 입력 FPS, D435 640x480 지원값 (시간 지연 비교 실험용)")
    parser.add_argument("--auto-stop", type=float)
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--quit-after-save", action="store_true")
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    window = MappingWindow(Session(), args.output_root, args.demo, args.imu_calibration, args.imu_fusion, args.projector_on, args.camera_fps)
    signal.signal(signal.SIGINT, lambda *_: window.close())
    signal.signal(signal.SIGTERM, lambda *_: window.close())
    window.show()
    if args.auto_stop:
        window.start()
        QTimer.singleShot(int(args.auto_stop*1000), window.session.stop)
    def check_finished() -> None:
        if window.session.snapshot()["state"] in {"complete", "partial", "failed"} and not window.session.thread.is_alive():
            window.refresh()
            if args.screenshot:
                args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                window.grab().save(str(args.screenshot))
            if args.quit_after_save:
                window.close()
                app.quit()
    finish_timer = QTimer()
    finish_timer.timeout.connect(check_finished)
    finish_timer.start(300)
    app.exec_()


if __name__ == "__main__":
    main()
