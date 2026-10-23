# ================================================================
# live_preview.py
# 설명: 초기 카메라 기준 관측 범위·경로·시야 방향을 Qt의 참고 지도로 그린다.
# 사용법: from experiments.handheld_mapping.live_preview import CoverageView
# ================================================================
import math

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPolygonF
from PyQt5.QtWidgets import QWidget


class CoverageView(QWidget):
    """미관측·시점 변화 관측 색과 추적 손실을 구분하는 잠정 X-Z 투영 화면."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.state = None
        self.setMinimumSize(350,300)

    def set_snapshot(self, state: dict | None) -> None:
        self.state = state
        self.setToolTip(str(state.get('error') or '') if state else '')
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(),QColor('#090e14'))
        painter.setPen(QColor('#e8edf2'))
        painter.drawText(14,24,'촬영 범위 · 위에서 본 참고 지도')
        painter.setPen(QColor('#a6b8c8'))
        state = self.state
        painter.drawText(14,45,'중력 기준 X-Y 투영 · 정지 IMU/장착 모델 기반'
                         if state and state.get('gravity_aligned') else '초기 카메라 X-Z 투영 · 중력 미정렬')
        if not state or 'image' not in state:
            painter.drawText(self.rect(),Qt.AlignCenter,
                             '실시간 안내 오류\n'+str(state['error']) if state and state.get('error')
                             else '촬영하면 관측 범위가 표시됩니다')
            return
        error = state.get('error')
        label = ('오류 · 원본 기록은 별도 유지' if error else state.get('gravity_error')
                 if state.get('gravity_requested') and state.get('gravity_error') else '촬영 종료 · 잠정 지도'
                 if state.get('finished') else 'ORB 준비 · 아직 위치 없음'
                 if not state.get('processed') else '위치 오래됨 · 갱신 대기'
                 if state.get('stale') else '추적 정상' if state.get('tracking_state') == 2
                 else '추적 손실 · 지도 갱신 중지')
        painter.setPen(QColor('#efad67') if error or state.get('stale') or state.get('tracking_state') != 2
                       else QColor('#80d5bb'))
        painter.drawText(14,66,f"Map {state['map_id']}  |  {label}")
        image = state['image']
        xmin,xmax,zmin,zmax = state['bounds']
        resolution = state['resolution']
        rows,cols = np.nonzero(image[:,:,0] != 45)
        samples = list(state['path'])
        if len(rows):
            samples += [(xmin+cols.min()*resolution,zmin+rows.min()*resolution),
                        (xmin+(cols.max()+1)*resolution,zmin+(rows.max()+1)*resolution)]
        if samples:
            points = np.asarray(samples)
            center = (points.min(axis=0)+points.max(axis=0))/2
            extent = max(4.,float(np.ptp(points,axis=0).max())+1.)
        else:
            center, extent = np.array([0.,1.]),4.
        extent = min(extent,xmax-xmin,zmax-zmin)
        left = float(np.clip(center[0]-extent/2,xmin,xmax-extent))
        bottom = float(np.clip(center[1]-extent/2,zmin,zmax-extent))
        size = max(1,min(self.width()-28,self.height()-148))
        area = QRectF((self.width()-size)/2,80,size,size)
        qimage = QImage(image.data,image.shape[1],image.shape[0],image.strides[0],QImage.Format_RGB888).copy().mirrored(False,True)
        source = QRectF((left-xmin)/resolution,(zmax-bottom-extent)/resolution,
                        extent/resolution,extent/resolution)
        painter.drawImage(area,qimage,source)
        def pixel(x: float,z: float) -> QPointF:
            return QPointF(area.left()+(x-left)/extent*size,area.bottom()-(z-bottom)/extent*size)
        painter.save()
        painter.setClipRect(area)
        painter.setPen(QPen(QColor('#74b9ed'),2))
        path = state['path']
        for previous,current in zip(path,path[1:]):
            painter.drawLine(pixel(*previous),pixel(*current))
        if state['camera'] is not None:
            x,z,yaw = state['camera']
            position = pixel(x,z)
            # 부채꼴은 시야 방향 설명이며 관측되지 않은 격자를 채우지 않는다.
            fan = [position]+[pixel(x+.6*math.sin(a),z+.6*math.cos(a))
                              for a in np.linspace(yaw-.5,yaw+.5,12)]
            painter.setPen(QPen(QColor('#e9c46a'),1))
            painter.setBrush(QColor(233,196,106,40))
            painter.drawPolygon(QPolygonF(fan))
            painter.setBrush(QColor('#f5d179'))
            painter.drawEllipse(position,4,4)
            painter.drawLine(position,pixel(x+.35*math.sin(yaw),z+.35*math.cos(yaw)))
        painter.restore()
        painter.setPen(QColor('#a6b8c8'))
        y = int(area.bottom())+21
        painter.drawText(14,y,f"가로 {extent:.1f} m  |  관측 {state['observed_cells']}셀")
        painter.drawText(14,y+20,f"수신 후 {state.get('lag_ms',0):.0f} ms  |  live 건너뜀 {state.get('dropped_pairs',0)}쌍")
        painter.drawText(14,y+40,'회색 미관측 · 노랑 1–2회 · 초록 3회 이상')
        if state.get('synthetic'):
            painter.drawText(14,y+59,'합성 known pose · native 추정/실기 성능 아님')
