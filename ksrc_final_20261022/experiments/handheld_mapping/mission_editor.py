# ================================================================
# mission_editor.py
# 설명: 팀원 STL의 출발·갈림길·공동 정상·세 경로를 클릭해 임무를 저장한다.
# 사용법: bash tools/plan_site_mission.sh --stl TEAM.stl --units m
# ================================================================
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
from PyQt5 import QtCore,QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg,NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib import font_manager
from matplotlib.text import Text
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from experiments.handheld_mapping.reference_editor import MapWorker
from experiments.handheld_mapping.site_map import SiteMap,load_stl
from experiments.handheld_mapping.site_mission import SiteMission,DEFAULT_SETTINGS
from experiments.handheld_mapping.mission_replay import demo_mission


class MissionEditor(QtWidgets.QMainWindow):
    """metric 지도 위 클릭·검사·Undo를 실제 저장 임무 모델에 연결하는 편집 창."""

    def __init__(self, site: SiteMap | None=None, mission: SiteMission | None=None) -> None:
        super().__init__()
        self.site=site;self.start=self.junction=self.summit=None
        self.interior={name:[] for name in ('10','15','30')};self.settings=DEFAULT_SETTINGS.copy()
        self.mode='';self.first_start=None;self.history=[];self.validation=None
        self._worker=None;self._close_requested=False
        available={font.name for font in font_manager.fontManager.ttflist}
        self.plot_font=next((name for name in ('Noto Sans CJK JP','NanumGothic') if name in available),'DejaVu Sans')
        self.setWindowTitle('대표점 방문 | 팀원 STL → 세 경로 → 정상·복귀 | 모터 송신 없음')
        self.resize(1250,950);self._build()
        if mission is not None:self._adopt(mission)
        else:self.redraw()

    def _build(self) -> None:
        central=QtWidgets.QWidget();self.setCentralWidget(central);layout=QtWidgets.QVBoxLayout(central)
        source=QtWidgets.QHBoxLayout();layout.addLayout(source)
        self.units=QtWidgets.QComboBox();self.units.addItems(['STL 단위 선택','m','mm']);source.addWidget(self.units)
        self.up_edit=QtWidgets.QLineEdit('0,0,1');self.up_edit.setMaximumWidth(110)
        source.addWidget(QtWidgets.QLabel('원본 up'));source.addWidget(self.up_edit)
        self.resolution=self._spin(.05,.005,.5,.01);self.radius=self._spin(.15,0.,1.,.01)
        source.addWidget(QtWidgets.QLabel('격자 m'));source.addWidget(self.resolution)
        source.addWidget(QtWidgets.QLabel('로버 여유 반경 m'));source.addWidget(self.radius)
        self._button(source,'STL 열기',self._choose_stl)
        self._button(source,'저장 임무 열기',self._choose_mission)
        self._button(source,'합성 Demo',lambda:self._adopt(demo_mission()))
        self.site_name=QtWidgets.QComboBox();self.site_name.addItems(['yeouido','sacheon','demo']);source.addWidget(self.site_name)
        self.surface_mode=QtWidgets.QComboBox();self.surface_mode.addItem('다층 보수 검사','conservative')
        self.surface_mode.addItem('상부 표면 명시 선택','top');source.addWidget(self.surface_mode)
        self.figure=Figure(figsize=(10,7),layout='constrained')
        self.canvas=FigureCanvasQTAgg(self.figure);self.ax_3d=self.figure.add_subplot(211,projection='3d')
        self.ax_xy=self.figure.add_subplot(212)
        layout.addWidget(NavigationToolbar2QT(self.canvas,self));layout.addWidget(self.canvas,1)
        self.canvas.mpl_connect('button_press_event',self._click)
        modes=QtWidgets.QHBoxLayout();layout.addLayout(modes)
        for mode,label in [('start','출발 위치→방향 두 클릭'),('junction','갈림길'),('summit','정상 대표점'),
                           ('route10','10° 경유점'),('route15','15° 경유점'),('route30','30° 진입·경유점')]:
            self._button(modes,label,lambda checked=False,key=mode:self.set_mode(key))
        self._button(modes,'선택 경로 지우기',self._clear_route)
        settings=QtWidgets.QHBoxLayout();layout.addLayout(settings)
        self.settings_fields={}
        for key,label,maximum in [('normal_slope_deg','일반 경사 °',89.),('trial_slope_deg','시험 경사 °',89.),
            ('arrival_tolerance_m','도착 허용 m',2.),('max_speed_m_s','요청 속도 m/s',.5),
            ('max_attempt_s','등판 시도 상한 s',600.),('stuck_window_s','진행 부족 창 s',60.),
            ('min_progress_m','최소 진행 m',1.)]:
            settings.addWidget(QtWidgets.QLabel(label))
            field=self._spin(self.settings[key],.001,maximum,.01);settings.addWidget(field);self.settings_fields[key]=field
        save=QtWidgets.QHBoxLayout();layout.addLayout(save)
        self.undo_button=self._button(save,'Undo',self.undo)
        self.validate_button=self._button(save,'경로 검사',self._validate)
        self.output_edit=QtWidgets.QLineEdit(str(self._default_output()));save.addWidget(self.output_edit,1)
        self.save_button=self._button(save,'새 임무 저장',self._save)
        self.status=QtWidgets.QLabel('STL 단위/up을 확인하고 출발·갈림길·정상·경유점을 설정하세요. 실제 모터 송신 없음.')
        self.status.setWordWrap(True);layout.addWidget(self.status)

    @staticmethod
    def _spin(value: float, low: float, high: float, step: float) -> QtWidgets.QDoubleSpinBox:
        field=QtWidgets.QDoubleSpinBox();field.setRange(low,high);field.setDecimals(3)
        field.setSingleStep(step);field.setValue(value);field.setMaximumWidth(85);return field

    @staticmethod
    def _button(layout, label: str, action) -> QtWidgets.QPushButton:
        button=QtWidgets.QPushButton(label);button.clicked.connect(action);layout.addWidget(button);return button

    @staticmethod
    def _default_output() -> Path:
        root=Path(__file__).resolve().parents[2]
        return root/'outputs/site_missions'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')

    def _adopt(self, mission: SiteMission) -> None:
        self.site=mission.site;self.start=mission.start.copy();self.junction=mission.junction.copy()
        self.summit=mission.summit.copy();self.settings=mission.settings.copy()
        self.interior={name:points[1:-1].tolist() for name,points in mission.routes.items()}
        self.history=[];self.mode='';self.first_start=None;self.validation=None
        self.site_name.setCurrentText(mission.site_name)
        for key,field in self.settings_fields.items():field.setValue(self.settings[key])
        self.resolution.setValue(self.site.resolution);self.radius.setValue(self.site.robot_radius)
        self.surface_mode.setCurrentIndex(self.surface_mode.findData(self.site.metadata['surface_mode']))
        self.output_edit.setText(str(self._default_output()))
        self.status.setText('임무 불러옴 · 경로 검사 후 새 버전으로 저장하세요.');self.redraw()

    def load_mission(self, folder: Path) -> None:
        self._adopt(SiteMission.load(folder))

    def _choose_mission(self) -> None:
        folder=QtWidgets.QFileDialog.getExistingDirectory(self,'mission.json이 있는 폴더')
        if not folder:return
        try:self.load_mission(Path(folder))
        except Exception as error:self.status.setText(str(error))

    def _choose_stl(self) -> None:
        if self.units.currentText() not in ('m','mm'):
            self.status.setText('STL의 실제 단위를 먼저 선택하세요: m 또는 mm.');return
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'팀원 STL 선택','','STL (*.stl)')
        if not path:return
        try:up=[float(v.strip()) for v in self.up_edit.text().split(',')]
        except ValueError:self.status.setText('up은 x,y,z 세 숫자입니다.');return
        config={'units':self.units.currentText(),'up':up,'resolution':self.resolution.value(),
                'robot_radius':self.radius.value(),'surface_mode':self.surface_mode.currentData()}
        self.status.setText('STL과 지형 grid 계산 중…');self.centralWidget().setEnabled(False)
        self._worker=MapWorker(lambda:load_stl(Path(path),**config),self)
        self._worker.result.connect(self._loaded_site)
        self._worker.error.connect(self.status.setText)
        self._worker.finished.connect(self._worker_done);self._worker.start()

    def _loaded_site(self, site: SiteMap) -> None:
        self.site=site;self.start=self.junction=self.summit=None
        self.interior={name:[] for name in ('10','15','30')};self.history=[];self.validation=None
        self.first_start=None;self.mode='';self.status.setText('STL 로드 완료 · 출발 위치와 heading부터 지정하세요.')
        self.redraw()

    def _worker_done(self) -> None:
        worker=self._worker;self._worker=None
        self.centralWidget().setEnabled(True)
        if worker is not None:worker.deleteLater()
        if self._close_requested:self.close()

    def closeEvent(self,event) -> None:
        if self._worker is not None:
            self._close_requested=True;event.ignore();return
        super().closeEvent(event)

    def set_mode(self, mode: str) -> None:
        self.mode=mode;self.first_start=None
        self.status.setText('XY 지도 클릭: '+mode+' · 경로는 갈림길→중간점→정상 순서입니다.')

    def _remember(self) -> None:
        self.history.append(copy.deepcopy((self.start,self.junction,self.summit,self.interior)))
        self.history=self.history[-100:];self.validation=None

    def undo(self) -> None:
        if self.history:
            self.start,self.junction,self.summit,self.interior=self.history.pop()
            self.validation=None;self.first_start=None;self.redraw()

    def _clear_route(self) -> None:
        if self.mode.startswith('route'):
            self._remember();self.interior[self.mode[5:]]=[];self.redraw()

    def _click(self,event) -> None:
        if (self.site is None or event.inaxes!=self.ax_xy or event.button!=1 or not self.mode
                or event.xdata is None or event.ydata is None or self._worker is not None):return
        if self.canvas.toolbar is not None and self.canvas.toolbar.mode:return
        xy=np.array([event.xdata,event.ydata],float)
        if not np.isfinite(xy).all():return
        if self.mode=='start':
            if self.first_start is None:
                self.first_start=xy;self.status.setText('다음 클릭으로 출발 heading을 지정하세요.');return
            delta=xy-self.first_start
            if np.linalg.norm(delta)<1e-3:self.status.setText('heading을 위한 다른 점을 클릭하세요.');return
            self._remember();self.start=np.r_[self.first_start,np.arctan2(delta[1],delta[0])]
            self.first_start=None
        elif self.mode in ('junction','summit'):
            self._remember();setattr(self,self.mode,xy)
        elif self.mode.startswith('route'):
            self._remember();self.interior[self.mode[5:]].append(xy.tolist())
        self.redraw()

    def build_mission(self) -> SiteMission:
        if self.site is None or any(value is None for value in (self.start,self.junction,self.summit)):
            raise ValueError('STL과 출발 position/heading·갈림길·정상 대표점을 먼저 지정하세요.')
        routes={name:np.asarray([self.junction,*self.interior[name],self.summit]) for name in ('10','15','30')}
        settings={**self.settings,**{key:field.value() for key,field in self.settings_fields.items()}}
        return SiteMission(self.site,self.start,self.junction,self.summit,routes,settings,self.site_name.currentText())

    def _validate(self) -> None:
        try:
            self.validation=self.build_mission().validate()
            self.status.setText(('경로 검사 통과' if self.validation['accepted'] else '경로 거부')+' · '+json.dumps(self.validation,ensure_ascii=False))
        except Exception as error:self.validation=None;self.status.setText(str(error))
        self.redraw()

    def _save(self) -> None:
        try:
            mission=self.build_mission();folder=Path(self.output_edit.text()).expanduser()
            mission.save(folder);save_preview(mission,folder/'preview.png')
            self.status.setText('저장: '+str(folder)+' · 임무/경로 준비 완료, 실기 주행 검증 전')
        except Exception as error:self.status.setText(str(error))

    def redraw(self) -> None:
        self.ax_xy.clear();self.ax_3d.clear()
        if self.site is not None:
            draw_site(self.site,self.ax_xy,self.ax_3d)
            colors={'10':'#16834a','15':'#2e6cce','30':'#dc7040'}
            for name,inside in self.interior.items():
                points=([self.junction] if self.junction is not None else [])+inside+([self.summit] if self.summit is not None else [])
                if points:
                    path=np.asarray(points);self.ax_xy.plot(*path.T,'o-',color=colors[name],label=name+'° 경로')
            for name,xy,color in [('출발',None if self.start is None else self.start[:2],'green'),
                                  ('갈림길',self.junction,'blue'),('정상 대표점',self.summit,'red')]:
                if xy is not None:
                    self.ax_xy.scatter(*xy,s=65,color=color,zorder=10);self.ax_xy.annotate(name,xy)
            if self.start is not None:
                self.ax_xy.arrow(*self.start[:2],.2*np.cos(self.start[2]),.2*np.sin(self.start[2]),width=.012,color='green')
            self.ax_xy.legend(loc='upper right') if any(self.interior.values()) else None
        self.ax_xy.set(xlabel='site X (m)',ylabel='site Y (m)',title='XY 설정 · 빈/회색=unknown, 빨강=장애물',aspect='equal')
        self.ax_3d.set(xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)',title='원본 형상·실제 경사 유지')
        for text in self.figure.findobj(match=Text):text.set_fontfamily(self.plot_font)
        self.canvas.draw_idle()


def draw_site(site: SiteMap, ax_xy, ax_3d=None) -> None:
    """저장 geometry 좌표를 그대로 3D와 metric XY 격자에 표시한다."""
    rows,cols=site.observed.shape;x,y=site.origin;res=site.resolution
    extent=[x,x+cols*res,y,y+rows*res]
    ax_xy.set_facecolor('#d9d9d9')
    ax_xy.imshow(np.ma.masked_where(~site.observed,site.slope_deg),origin='lower',extent=extent,cmap='terrain',vmin=0,vmax=35)
    ax_xy.imshow(np.ma.masked_where(~site.blocked,site.blocked),origin='lower',extent=extent,cmap='Reds',vmin=0,vmax=1,alpha=.8)
    if ax_3d is not None:
        vertices=np.asarray(site.mesh.vertices);triangles=np.asarray(site.mesh.triangles)
        faces=vertices[triangles[::max(1,len(triangles)//10000)]]
        ax_3d.add_collection3d(Poly3DCollection(faces,facecolor='#bdbdbd',edgecolor='none',alpha=.7))
        for axis,values in zip(('x','y','z'),vertices.T):
            lo,hi=float(values.min()),float(values.max());getattr(ax_3d,'set_'+axis+'lim')(lo,hi if hi>lo else lo+.1)


def save_preview(mission: SiteMission, path: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure=Figure(figsize=(9,7));FigureCanvasAgg(figure);axis=figure.subplots();draw_site(mission.site,axis)
    for name,points in mission.routes.items():axis.plot(*points.T,'o-',label=name+' deg named route')
    axis.scatter(*mission.start[:2],c='green',label='start');axis.scatter(*mission.summit,c='red',label='summit')
    axis.set(xlabel='site X (m)',ylabel='site Y (m)',aspect='equal',title='Mission routes | no motor transmission')
    axis.legend();figure.savefig(path)


def main() -> None:
    parser=argparse.ArgumentParser(description='팀원 STL 이후 대표점·세 경로 설정')
    source=parser.add_mutually_exclusive_group();source.add_argument('--stl',type=Path)
    source.add_argument('--mission',type=Path);source.add_argument('--demo',action='store_true')
    parser.add_argument('--units',choices=('m','mm'));parser.add_argument('--up',nargs=3,type=float,default=[0,0,1])
    parser.add_argument('--resolution',type=float,default=.05);parser.add_argument('--robot-radius',type=float,default=.15)
    parser.add_argument('--surface-mode',choices=('conservative','top'),default='conservative')
    parser.add_argument('--output',type=Path);parser.add_argument('--save-only',action='store_true')
    args=parser.parse_args()
    if args.stl and not args.units:parser.error('--stl requires explicit --units m|mm')
    mission=demo_mission() if args.demo else SiteMission.load(args.mission) if args.mission else None
    site=mission.site if mission else load_stl(args.stl,units=args.units,up=args.up,resolution=args.resolution,
                                            robot_radius=args.robot_radius,surface_mode=args.surface_mode) if args.stl else None
    if args.save_only:
        if mission is None or args.output is None:parser.error('--save-only requires --demo/--mission and --output')
        mission.save(args.output);save_preview(mission,args.output/'preview.png')
        print(json.dumps({'mission':str(args.output),'simulation_only':args.demo,'hardware_commands_enabled':False}));return
    app=QtWidgets.QApplication(sys.argv);window=MissionEditor(site,mission)
    if args.output:window.output_edit.setText(str(args.output))
    if args.units:window.units.setCurrentText(args.units)
    if args.stl:window.surface_mode.setCurrentIndex(window.surface_mode.findData(args.surface_mode))
    window.up_edit.setText(','.join(map(str,args.up)));window.show();sys.exit(app.exec_())


if __name__=='__main__':main()
