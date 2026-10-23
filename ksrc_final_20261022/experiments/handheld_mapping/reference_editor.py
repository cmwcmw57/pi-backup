# ================================================================
# reference_editor.py
# 설명: 위 3D·아래 XY 화면에서 지도 정렬·ROI·출발 prior와 반복 병합 버전을 편집한다.
# 사용법: python -m experiments.handheld_mapping.reference_editor TARGET [--map-id 0]
# ================================================================
from __future__ import annotations

import argparse
import copy
from datetime import datetime
from pathlib import Path
import sys
from typing import Callable

import numpy as np
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from experiments.handheld_mapping.reference import (
    ReferenceMap, estimate_gravity, load_source, manual_gravity, rigid_pose, source_maps,
)


class MapWorker(QtCore.QThread):
    """긴 source 로드·정합·저장을 작업 thread에서 실행하고 UI에 결과만 전달한다."""
    result = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)

    def __init__(self, operation: Callable, parent: QtCore.QObject) -> None:
        super().__init__(parent)
        self.operation = operation

    def run(self) -> None:
        try:
            self.result.emit(self.operation())
        except Exception as error:
            self.error.emit(f'{type(error).__name__}: {error}')


class ReferenceEditor(QtWidgets.QMainWindow):
    """편집 상태와 preview를 분리하고 수용된 병합만 새 버전으로 저장하는 지도 창."""

    def __init__(self, target: ReferenceMap | Path, map_id: int | None = None,
                 backend: str = 'auto', variant: str = 'optimized', units_confirmed: bool = False) -> None:
        super().__init__()
        self.target = Path(target) if not isinstance(target, ReferenceMap) else None
        self.backend, self.variant = backend, variant
        if isinstance(target, ReferenceMap):
            self.model = target
            identities = []
        else:
            identities = source_maps(target, backend, variant)
            # 여러 지도는 서로 다른 frame이다. combo의 첫 ID를 명시 선택하여 하나만 연다.
            selected_id = map_id if map_id is not None else identities[0] if identities else None
            self.model = load_source(target, selected_id, backend, variant, units_confirmed)
        self.original = copy.deepcopy(self.model)
        self.history: list[ReferenceMap] = []
        self.preview: ReferenceMap | None = None
        self.candidate: ReferenceMap | None = None
        self.merge_result: dict | None = None
        self.merge_model: ReferenceMap | None = None
        self.registrar: Callable | None = None
        self._worker: MapWorker | None = None
        self._close_requested = False
        self.polygon: list[list[float]] = []
        self.click_mode = ''
        self.first_start: list[float] | None = None
        self.setWindowTitle('Reference 지도 편집 | 길이 m · 수동 출발 pose는 초기 prior')
        self.resize(1250, 1000)
        self._build(identities)
        self.redraw()
        if len(identities) > 1:
            self.status.setText('분리 map은 별도 좌표계입니다. 위 map 선택에서 한 지도씩 편집합니다.')

    @property
    def busy(self) -> bool:
        return self._worker is not None

    @staticmethod
    def _number(value: float, low: float = -10000., high: float = 10000.) -> QtWidgets.QDoubleSpinBox:
        field = QtWidgets.QDoubleSpinBox()
        field.setRange(low, high)
        field.setDecimals(3)
        field.setSingleStep(.05)
        field.setValue(value)
        field.setMaximumWidth(105)
        return field

    def _button(self, label: str, callback: Callable, layout: QtWidgets.QLayout) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(label)
        button.clicked.connect(callback)
        layout.addWidget(button)
        self.mutation_controls.append(button)
        return button

    def _build(self, identities: list[int]) -> None:
        container = QtWidgets.QWidget()
        self.setCentralWidget(container)
        layout = QtWidgets.QVBoxLayout(container)
        self.mutation_controls: list[QtWidgets.QWidget] = []
        select = QtWidgets.QHBoxLayout()
        layout.addLayout(select)
        self.map_combo = QtWidgets.QComboBox()
        for identity in identities:
            self.map_combo.addItem(f'Map {identity}', identity)
        self.map_combo.setCurrentIndex(max(0, self.map_combo.findData(self.model.map_id)))
        self.map_combo.setEnabled(len(identities) > 1)
        self.map_combo.currentIndexChanged.connect(self._change_map)
        select.addWidget(self.map_combo)
        self.variant_combo = QtWidgets.QComboBox()
        self.variant_combo.addItems(['optimized', 'baseline'])
        self.variant_combo.setCurrentText(self.variant)
        self.variant_combo.setEnabled(self.target is not None and self._has_variants())
        self.variant_combo.currentTextChanged.connect(self._change_variant)
        select.addWidget(self.variant_combo)
        self.source_label = QtWidgets.QLabel()
        self.source_label.setWordWrap(True)
        select.addWidget(self.source_label, 1)
        self.figure = Figure(figsize=(9, 8), layout='constrained')
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax_3d = self.figure.add_subplot(211, projection='3d')
        self.ax_xy = self.figure.add_subplot(212)
        layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(self.canvas, 1)
        self.canvas.mpl_connect('button_press_event', self._click)
        gravity = QtWidgets.QHBoxLayout()
        layout.addLayout(gravity)
        gravity.addWidget(QtWidgets.QLabel('불러온 지도 up x,y,z:'))
        self.up_edits = [self._number(v, -1., 1.) for v in (0., 0., 1.)]
        for field in self.up_edits:
            gravity.addWidget(field)
        self.manual_up_button = self._button('수동 up 정렬', self._manual_up, gravity)
        self.auto_gravity_button = self._button('IMU 중력 후보', self._auto_gravity, gravity)
        gravity.addWidget(QtWidgets.QLabel('수동 roll/pitch °:'))
        self.roll_edit, self.pitch_edit = self._number(0., -180., 180.), self._number(0., -180., 180.)
        gravity.addWidget(self.roll_edit)
        gravity.addWidget(self.pitch_edit)
        self._button('각도 적용', self._manual_angles, gravity)
        self.experimental_gravity = QtWidgets.QCheckBox('미검증 시각의 실험적 IMU 후보 허용 (자동 적용 안 함)')
        layout.addWidget(self.experimental_gravity)
        roi = QtWidgets.QHBoxLayout()
        layout.addLayout(roi)
        self.roi_button = self._button('ROI 다각형 클릭', self._roi_mode, roi)
        self._button('ROI 점 지우기', self._clear_polygon, roi)
        roi.addWidget(QtWidgets.QLabel('높이 min/max m:'))
        points = np.asarray(self.model.geometry()[0].points)
        self.height_min = self._number(float(points[:, 2].min()) - .1)
        self.height_max = self._number(float(points[:, 2].max()) + .1)
        roi.addWidget(self.height_min)
        roi.addWidget(self.height_max)
        self.crop_preview_button = self._button('Crop 미리보기', self._crop_preview, roi)
        self.apply_button = self._button('미리보기 적용', self._apply_preview, roi)
        process = QtWidgets.QHBoxLayout()
        layout.addLayout(process)
        process.addWidget(QtWidgets.QLabel('큰 형상 mesh: 최대 triangle (0=보존)'))
        self.triangle_limit = QtWidgets.QSpinBox()
        self.triangle_limit.setRange(0, 10_000_000)
        self.triangle_limit.setSingleStep(1000)
        process.addWidget(self.triangle_limit)
        process.addWidget(QtWidgets.QLabel('약한 Taubin 반복 (0=OFF, 최대10)'))
        self.smooth_iterations = QtWidgets.QSpinBox()
        self.smooth_iterations.setRange(0, 10)
        process.addWidget(self.smooth_iterations)
        self.processing_button = self._button('Mesh 처리 미리보기', self._processing_preview, process)
        process.addWidget(QtWidgets.QLabel('작은 형상 손실 가능 · 분석 원본 original/ 보존'), 1)
        start = QtWidgets.QHBoxLayout()
        layout.addLayout(start)
        self.start_button = self._button('출발 위치 → heading 두 클릭', self._start_mode, start)
        start.addWidget(QtWidgets.QLabel('출발 높이 m:'))
        self.start_height = self._number(0.)
        start.addWidget(self.start_height)
        start.addWidget(QtWidgets.QLabel('녹색: 다음 주행 initial prior · 빨강: 원본 촬영 시작점'), 1)
        merge = QtWidgets.QHBoxLayout()
        layout.addLayout(merge)
        self.candidate_button = self._button('후보 지도 열기', self._choose_candidate, merge)
        self.candidate_label = QtWidgets.QLabel('후보 없음')
        merge.addWidget(self.candidate_label, 1)
        self.merge_preview_button = self._button('정합 미리보기', self._merge_preview, merge)
        self.merge_save_button = self._button('통과 병합을 새 버전 저장', self._merge_save, merge)
        pose = QtWidgets.QHBoxLayout()
        layout.addLayout(pose)
        pose.addWidget(QtWidgets.QLabel('후보 → 기준 초기 pose:'))
        self.pose_edits = []
        for label in ('x m', 'y m', 'z m', 'yaw °', 'roll °', 'pitch °'):
            pose.addWidget(QtWidgets.QLabel(label))
            field = self._number(0.)
            field.valueChanged.connect(self._candidate_pose_changed)
            pose.addWidget(field)
            self.pose_edits.append(field)
        self.mutation_controls.extend(self.pose_edits)
        save = QtWidgets.QHBoxLayout()
        layout.addLayout(save)
        self.undo_button = self._button('Undo', self._undo, save)
        self.reset_button = self._button('원래 상태', self._reset, save)
        self.output_edit = QtWidgets.QLineEdit(str(self._default_output()))
        save.addWidget(self.output_edit, 1)
        self._button('저장 폴더 선택', self._choose_output, save)
        self.save_button = self._button('새 버전 저장', self._save, save)
        self.status = QtWidgets.QLabel('원본은 보존합니다. up은 창에 불러온 지도 좌표계 벡터이며 지형 평면을 자동 수평화하지 않습니다.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self._update_controls()

    def _has_variants(self) -> bool:
        import json
        if self.target is None:
            return False
        path = self.target if self.target.name == 'summary.json' else self.target/'summary.json'
        try:
            data = json.loads(path.read_text())
            return 'baseline_summary' in data and 'optimized_summary' in data
        except (OSError, ValueError):
            return False

    def _default_output(self) -> Path:
        base = self.target if self.target is not None and self.target.is_dir() else self.target.parent if self.target is not None else Path.cwd()
        return base/'reference_versions'/datetime.now().strftime('reference_%Y%m%d_%H%M%S_%f')

    def _update_controls(self) -> None:
        for control in self.mutation_controls:
            control.setEnabled(not self.busy)
        self.map_combo.setEnabled(not self.busy and self.map_combo.count() > 1)
        self.variant_combo.setEnabled(not self.busy and self._has_variants())
        self.apply_button.setEnabled(not self.busy and self.preview is not None)
        self.processing_button.setEnabled(not self.busy and self.model.mesh is not None)
        self.undo_button.setEnabled(not self.busy and bool(self.history))
        self.merge_preview_button.setEnabled(not self.busy and self.candidate is not None)
        self.merge_save_button.setEnabled(not self.busy and self.preview is None and self.merge_model is not None
                                          and bool(self.merge_result and self.merge_result.get('accepted')))

    def _run(self, operation: Callable, callback: Callable, message: str) -> None:
        if self.busy:
            return
        self.status.setText(message)
        worker = MapWorker(operation, self)
        self._worker = worker
        worker.result.connect(callback)
        worker.error.connect(self.status.setText)
        worker.finished.connect(self._finished)
        self._update_controls()
        worker.start()

    def _finished(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()
        self._update_controls()
        if self._close_requested:
            self.close()

    def _push(self) -> None:
        self.history.append(copy.deepcopy(self.model))
        # undo에 원본 geometry 복사가 쌓이지 않도록 편집 10회로 제한한다.
        self.history = self.history[-10:]
        self.preview = None
        self._invalidate_merge()

    def _invalidate_merge(self, *_args) -> None:
        self.merge_result = None
        self.merge_model = None
        if hasattr(self, 'merge_save_button'):
            self._update_controls()

    def _candidate_pose_changed(self, *_args) -> None:
        self._invalidate_merge()
        self.redraw()

    def redraw(self) -> None:
        value = self.preview or self.model
        cloud, mesh = value.geometry()
        points = np.asarray(cloud.points)
        stride = max(1, int(np.ceil(len(points) / 20000)))
        display = points[::stride]
        self.ax_3d.clear()
        self.ax_xy.clear()
        if len(display):
            colors = np.asarray(cloud.colors)[::stride] if cloud.has_colors() else display[:, 2]
            self.ax_3d.scatter(*display.T, c=colors, s=1, cmap=None if cloud.has_colors() else 'terrain')
            self.ax_xy.scatter(display[:, 0], display[:, 1], c=colors, s=2, cmap=None if cloud.has_colors() else 'terrain')
            low, high = display.min(axis=0), display.max(axis=0)
            center, size = (low+high)/2, max(float((high-low).max()), .1)*.6
            self.ax_3d.set_xlim(center[0]-size, center[0]+size)
            self.ax_3d.set_ylim(center[1]-size, center[1]+size)
            self.ax_3d.set_zlim(center[2]-size, center[2]+size)
            self.ax_3d.set_box_aspect([1, 1, 1])
            margin = max(float((high[:2]-low[:2]).max())*.12, .1)
            self.ax_xy.set_xlim(low[0]-margin, high[0]+margin)
            self.ax_xy.set_ylim(low[1]-margin, high[1]+margin)
        if mesh is not None and len(mesh.triangles):
            triangles = np.asarray(mesh.triangles)
            triangles = triangles[::max(1, int(np.ceil(len(triangles)/8000)))]
            surface = Poly3DCollection(np.asarray(mesh.vertices)[triangles], alpha=.15, linewidths=0)
            surface.set_facecolor('#7baac4')
            self.ax_3d.add_collection3d(surface)
        for segment in value.trajectory_segments():
            path = np.asarray(segment)
            self.ax_3d.plot(*path.T, color='#ff9f1c', linewidth=1.5, marker='.', markersize=2)
            self.ax_xy.plot(path[:, 0], path[:, 1], color='#ff9f1c', linewidth=1.5, marker='.', markersize=2)
        # 병합 preview에서도 기준 geometry/pose는 고정되고 후보만 source→reference 변환한다.
        if self.candidate is not None:
            candidate_cloud, _ = self.candidate.geometry()
            matrix = (np.asarray(self.merge_result['transformation']) if self.merge_result and self.merge_result.get('accepted')
                      else rigid_pose(*[field.value() for field in self.pose_edits]))
            candidate_points = np.asarray(candidate_cloud.transform(matrix).points)
            candidate_points = candidate_points[::max(1, int(np.ceil(len(candidate_points)/8000)))]
            self.ax_3d.scatter(*candidate_points.T, color='#c27cfa', s=1, alpha=.4)
            self.ax_xy.scatter(candidate_points[:, 0], candidate_points[:, 1], color='#c27cfa', s=2, alpha=.4)
        if value.original_start is not None:
            original = (value.local_transform @ np.asarray(value.original_start))[:3, 3]
            self.ax_3d.scatter(*original, color='red', s=40)
            self.ax_xy.scatter(*original[:2], color='red', s=40, label='original camera start')
        if value.start_pose is not None:
            start = np.asarray(value.start_pose['matrix'])
            p, direction = start[:3, 3], start[:3, 0]*.2
            self.ax_3d.quiver(*p, *direction, color='green')
            self.ax_xy.arrow(*p[:2], *direction[:2], color='green', width=.006, length_includes_head=True)
            self.ax_xy.scatter(*p[:2], color='green', s=25)
        if self.polygon:
            polygon = np.asarray(self.polygon + ([self.polygon[0]] if len(self.polygon) >= 3 else []))
            self.ax_xy.plot(polygon[:, 0], polygon[:, 1], 'c.-', linewidth=1.5)
        self.ax_3d.scatter(0, 0, 0, marker='+', color='cyan', s=50)
        self.ax_xy.scatter(0, 0, marker='+', color='cyan', s=50)
        self.ax_3d.set(xlabel='X m', ylabel='Y m', zlabel='Z m', title='3D reference (+Z declared up)')
        self.ax_xy.set(xlabel='X m', ylabel='Y m', title=f'XY | {value.path_kind} trajectory | prior is not measured pose')
        self.ax_xy.set_aspect('equal', adjustable='box')
        self.ax_xy.grid(alpha=.2)
        self.source_label.setText(f"Map {value.map_id} | {len(cloud.points):,} points | 경로: {value.path_kind} | 중력: {value.gravity.get('source')} (실측 검증 안 됨)")
        self.canvas.draw_idle()
        self._update_controls()

    def _guard(self, operation: Callable) -> None:
        try:
            operation()
        except (ValueError, OSError) as error:
            self.status.setText(str(error))
        self._update_controls()

    def _manual_up(self) -> None:
        def apply() -> None:
            declared = np.asarray([field.value() for field in self.up_edits])
            current_up = self.model.local_transform[:3, :3] @ declared
            gravity = manual_gravity(current_up)
            gravity.update(up_input=declared.tolist(), input_frame='loaded_geometry')
            self._push()
            if np.allclose(gravity['transformation'], np.eye(4), atol=1e-9):
                self.model.gravity = gravity
                self.status.setText('지정한 up이 이미 +Z입니다. 현재 ROI를 유지합니다. 실측 중력 검증은 아닙니다.')
            else:
                self.model.apply_transform(gravity['transformation'], gravity)
                self.polygon = []
                self.status.setText('사용자 up 벡터 정렬 적용. 이전 ROI는 새 좌표계에서 다시 지정하세요. 실측 중력 검증은 아닙니다.')
            self.redraw()
        self._guard(apply)

    def _manual_angles(self) -> None:
        self._push()
        matrix = rigid_pose(0, 0, 0, 0, self.roll_edit.value(), self.pitch_edit.value())
        self.model.apply_transform(matrix, {'source': 'manual_roll_pitch', 'verified': False,
                                           'quality': 'user_declared_not_measured', 'transformation': matrix.tolist()})
        self.polygon = []
        self.status.setText('사용자 roll/pitch 적용. 실제 지형 경사는 rigid 변환으로 유지됩니다.')
        self.redraw()

    def _auto_gravity(self) -> None:
        run = self.model.source.get('run')
        if not run:
            self.status.setText('원본 run/IMU provenance가 없습니다. 수동 up 벡터를 지정하세요.')
            return
        snapshot = copy.deepcopy(self.model)
        allow_unverified = self.experimental_gravity.isChecked()
        def finish(result: dict) -> None:
            if result.get('accepted'):
                self._invalidate_merge()
                self.preview = copy.deepcopy(self.model)
                self.preview.apply_transform(result['transformation'], result)
                self.status.setText('정지 IMU+camera+URDF 중력 후보 preview. 시각/외부보정/물리적 정확도는 미검증일 수 있습니다. 미리보기 적용을 눌러 확정하세요.')
                self.redraw()
            else:
                self.status.setText(f"자동 중력 거부: {result.get('reason')} · 수동 up 벡터를 지정하세요.")
        self._run(lambda: estimate_gravity(Path(run), snapshot, allow_unverified_clock=allow_unverified), finish,
                  '중력 provenance와 정지 구간 검사 중…')

    def _roi_mode(self) -> None:
        self.click_mode = 'roi'
        self.first_start = None
        self.status.setText('아래 XY 화면에서 polygon 꼭짓점을 클릭한 뒤 Crop 미리보기를 누르세요. mesh는 모든 vertex가 내부인 triangle만 유지합니다.')

    def _clear_polygon(self) -> None:
        self.polygon = []
        self.click_mode = ''
        self.redraw()

    def _start_mode(self) -> None:
        self.click_mode = 'start'
        self.first_start = None
        self.status.setText('아래 XY에서 출발 위치를 클릭한 뒤 진행 방향을 클릭하세요. 높이는 입력한 m 값입니다.')

    def _click(self, event) -> None:
        if self.busy or event.inaxes is not self.ax_xy or event.button != 1 or event.xdata is None or event.ydata is None:
            return
        if self.canvas.toolbar is not None and self.canvas.toolbar.mode:
            return
        if self.click_mode == 'roi':
            self.polygon.append([float(event.xdata), float(event.ydata)])
            self.redraw()
        elif self.click_mode == 'start':
            point = [float(event.xdata), float(event.ydata), self.start_height.value()]
            if self.first_start is None:
                self.first_start = point
                self.status.setText('출발 위치 선택. 두 번째 클릭으로 heading을 지정하세요.')
            else:
                def apply() -> None:
                    self._push()
                    self.model.set_start(self.first_start, point)
                    self.first_start, self.click_mode = None, ''
                    self.status.setText('출발 initial prior를 지정했습니다. 실측 pose/localization 성공을 뜻하지 않습니다.')
                    self.redraw()
                self._guard(apply)

    def _crop_preview(self) -> None:
        def preview() -> None:
            value = copy.deepcopy(self.model)
            value.set_crop(self.polygon, self.height_min.value(), self.height_max.value())
            if value.geometry()[0].is_empty():
                raise ValueError('ROI에 남는 geometry가 없습니다. 범위를 다시 지정하세요.')
            self._invalidate_merge()
            self.preview = value
            self.click_mode = ''
            self.status.setText('Crop preview입니다. 미리보기 적용 후 새 버전으로 저장하세요.')
            self.redraw()
        self._guard(preview)

    def _apply_preview(self) -> None:
        if self.preview is None:
            return
        value = self.preview
        self._push()
        self.model = value
        self.status.setText('미리보기 적용. Undo/원래 상태로 복구할 수 있습니다.')
        self.redraw()

    def _processing_preview(self) -> None:
        snapshot = copy.deepcopy(self.model)
        limit, iterations = self.triangle_limit.value(), self.smooth_iterations.value()
        def operation() -> ReferenceMap:
            snapshot.set_processing(limit, iterations)
            snapshot.geometry()
            return snapshot
        def finish(value: ReferenceMap) -> None:
            self._invalidate_merge()
            self.preview = value
            self.status.setText('Mesh 처리 preview. 작은 형상이 손실될 수 있습니다. 적용 후 새 버전에만 저장합니다.')
            self.redraw()
        self._run(operation, finish, 'Mesh 간소화/Taubin preview 계산 중…')

    def _undo(self) -> None:
        if self.history:
            self.model = self.history.pop()
            self.preview = None
            self.polygon = []
            self._invalidate_merge()
            self.redraw()

    def _reset(self) -> None:
        self._push()
        self.model = copy.deepcopy(self.original)
        self.preview = None
        self.polygon = []
        self.first_start, self.click_mode = None, ''
        self.redraw()

    def _choose_output(self) -> None:
        selected = QtWidgets.QFileDialog.getExistingDirectory(self, '새 버전을 둘 부모 폴더 선택')
        if selected:
            self.output_edit.setText(str(Path(selected)/datetime.now().strftime('reference_%Y%m%d_%H%M%S_%f')))

    def _save(self) -> None:
        if self.preview is not None:
            self.status.setText('Crop preview를 먼저 적용하거나 Undo/원래 상태로 취소하세요.')
            return
        destination = Path(self.output_edit.text()).expanduser()
        snapshot = copy.deepcopy(self.model)
        def finish(_result: dict) -> None:
            self.model.parent_version = str(destination.resolve())
            self.status.setText(f'새 reference 버전: {destination} · 원본 보존, 실측 정확도 미검증')
            self.output_edit.setText(str(self._default_output()))
        self._run(lambda: snapshot.save(destination), finish, '새 버전 geometry·metadata 저장 중…')

    def set_candidate(self, candidate: ReferenceMap) -> None:
        self.candidate = candidate
        self._invalidate_merge()
        self.candidate_label.setText(f'후보 Map {candidate.map_id} | 초기 pose를 명시하세요')
        self.redraw()

    def _choose_candidate(self) -> None:
        selected, _ = QtWidgets.QFileDialog.getOpenFileName(self, '후보 summary.json / reference.json / PLY / STL 선택', '',
                                                          '지도 (*.json *.ply *.stl)')
        if not selected:
            return
        target, identity, confirmed = Path(selected), None, False
        try:
            identities = source_maps(target)
            if identities:
                chosen, okay = QtWidgets.QInputDialog.getItem(self, '후보 map 선택', '각 map은 별도 좌표계입니다.',
                                                             [str(v) for v in identities], 0, False)
                if not okay:
                    return
                identity = int(chosen)
            if target.suffix.lower() == '.stl':
                confirmed = QtWidgets.QMessageBox.question(self, 'STL 단위 확인', '이 STL의 좌표 단위가 meter입니까?') == QtWidgets.QMessageBox.Yes
                if not confirmed:
                    return
        except (ValueError, OSError) as error:
            self.status.setText(str(error))
            return
        self._run(lambda: load_source(target, identity, units_confirmed=confirmed), self.set_candidate, '후보 geometry 로드 중…')

    def _merge_preview(self) -> None:
        if self.candidate is None:
            return
        self.preview = None
        self._invalidate_merge()
        reference, candidate = copy.deepcopy(self.model), copy.deepcopy(self.candidate)
        initial = rigid_pose(*[field.value() for field in self.pose_edits])
        registrar = self.registrar
        def operation() -> tuple[dict, ReferenceMap | None]:
            result = reference.preview_merge(candidate, initial, registrar=registrar)
            return result, reference.merged(candidate, result) if result.get('accepted') else None
        def finish(value: tuple[dict, ReferenceMap | None]) -> None:
            self.merge_result, self.merge_model = value
            if self.merge_model is None:
                self.status.setText(f"정합 거부: {self.merge_result.get('reason')} · 기준 버전은 그대로입니다.")
            else:
                self.status.setText('정합 gate 통과 preview. 원래 지도 위치는 고정됩니다. 저장은 geometry union이며 joint TSDF가 아닙니다.')
            self.redraw()
        self._run(operation, finish, '양방향 정합·관측성 gate 검사 중…')

    def _merge_save(self) -> None:
        if self.preview is not None or self.merge_model is None or not self.merge_result or not self.merge_result.get('accepted'):
            return
        destination, snapshot = Path(self.output_edit.text()).expanduser(), copy.deepcopy(self.merge_model)
        def finish(_result: dict) -> None:
            self._push()
            self.model = snapshot
            self.model.parent_version = str(destination.resolve())
            self.candidate = None
            self.candidate_label.setText('후보 없음')
            self.output_edit.setText(str(self._default_output()))
            self.status.setText(f'통과 geometry 병합 새 버전: {destination} · 기존 버전은 보존됩니다.')
            self.redraw()
        self._run(lambda: snapshot.save(destination), finish, '통과 병합 버전을 저장 중…')

    def _change_map(self, _index: int) -> None:
        if self.target is not None and not self.busy:
            identity = self.map_combo.currentData()
            self._run(lambda: load_source(self.target, identity, self.backend, self.variant), self._replace_source, '선택한 map 로드 중…')

    def _change_variant(self, variant: str) -> None:
        if self.target is not None and not self.busy:
            self.variant = variant
            identity = self.map_combo.currentData()
            self._run(lambda: load_source(self.target, identity, self.backend, variant), self._replace_source, '선택한 global 결과 로드 중…')

    def _replace_source(self, value: ReferenceMap) -> None:
        self.model, self.original = value, copy.deepcopy(value)
        self.history, self.polygon = [], []
        self.preview, self.candidate = None, None
        self._invalidate_merge()
        self.status.setText('선택한 map을 별도 좌표계로 열었습니다.')
        self.redraw()

    def closeEvent(self, event) -> None:
        if self.busy:
            self._close_requested = True
            self.status.setText('진행 중인 정합/저장을 안전하게 마친 뒤 창을 닫습니다.')
            event.ignore()
            return
        event.accept()


def main() -> None:
    parser = argparse.ArgumentParser(description='지도 ROI·중력·출발 prior·반복 정합 편집')
    parser.add_argument('target', type=Path)
    parser.add_argument('--map-id', type=int)
    parser.add_argument('--backend', choices=['auto', 'orb', 'orbline', 'orb_offline', 'orb_inertial'], default='auto')
    parser.add_argument('--variant', choices=['baseline', 'optimized'], default='optimized')
    parser.add_argument('--stl-meters', action='store_true', help='STL의 좌표 단위가 meter임을 명시 확인')
    args = parser.parse_args()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    confirmed = args.stl_meters
    if args.target.suffix.lower() == '.stl' and not confirmed:
        confirmed = QtWidgets.QMessageBox.question(None, 'STL 단위 확인', 'STL 좌표 단위가 meter입니까? 경로 정보는 없습니다.') == QtWidgets.QMessageBox.Yes
        if not confirmed:
            raise SystemExit('STL meter 단위 확인 필요')
    try:
        editor = ReferenceEditor(args.target, args.map_id, args.backend, args.variant, confirmed)
    except (OSError, ValueError) as error:
        raise SystemExit(f'지도 편집 입력 오류: {error}') from error
    editor.show()
    print('REFERENCE_EDITOR_READY', args.target, flush=True)
    raise SystemExit(app.exec_())


if __name__ == '__main__':
    main()
