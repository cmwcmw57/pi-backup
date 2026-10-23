# ================================================================
# viewer.py
# 설명: 저장된 map별 TSDF mesh/point cloud를 독립된 Open3D 창으로 연다.
# 사용법: bash tools/view_mapping.sh [RUN_OR_PLY] [--map-id 0] [--cloud]
# ================================================================
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any
import re
import time

ROOT = Path(__file__).resolve().parents[2]
FOLDERS = {'orb_inertial':'dense_inertial','orb_offline':'dense_offline','orb':'dense','orbline':'dense_orbline'}


def resolve_target(target: Path | None, sessions: Path | None = None) -> Path:
    if target is None:
        root = sessions or ROOT/'outputs/handheld_mapping/sessions'
        candidates = [path for path in root.iterdir() if path.is_dir()] if root.is_dir() else []
        if not candidates:
            raise FileNotFoundError('촬영 기록이 없습니다. RUN 또는 PLY를 지정하세요.')
        target = max(candidates, key=lambda path: path.name)
    return Path(target).expanduser().resolve()


def select_maps(target: Path | None, backend: str = 'auto', map_id: int | None = None,
                cloud: bool = False, sessions: Path | None = None) -> list[tuple[str,Path]]:
    """최신 촬영의 저장 결과만 선택하며 좌표계가 다른 map을 별도로 반환한다."""
    if backend not in ('auto',*FOLDERS):
        raise ValueError('unknown viewer backend')
    if target is None:
        root = sessions or ROOT/'outputs/handheld_mapping/sessions'
        candidates = [path for path in root.iterdir() if path.is_dir()] if root.is_dir() else []
        if not candidates:
            raise FileNotFoundError('촬영 기록이 없습니다. RUN 또는 PLY를 지정하세요.')
        target = max(candidates,key=lambda path:path.name)
    target = Path(target).expanduser().resolve()
    if target.is_file():
        if target.suffix.lower() != '.ply' or map_id is not None or backend != 'auto':
            raise ValueError('직접 PLY를 열 때는 backend/map-id를 지정하지 않습니다.')
        return [(target.name,target)]
    if not target.is_dir():
        raise FileNotFoundError(str(target))
    explicit = backend == 'auto' and (target / 'summary.json').is_file()
    folders = list(FOLDERS.values()) if backend == 'auto' else [FOLDERS[backend]]
    folder = target if explicit else next((target/name for name in folders if (target/name/'summary.json').is_file()),None)
    if folder is None:
        raise FileNotFoundError(f'아직 저장된 3D 지도가 없습니다. 촬영 종료·지도 생성 후 열어주세요: {target}')
    summary = json.loads((folder/'summary.json').read_text())
    if explicit and isinstance(summary, dict) and isinstance(summary.get('optimized_summary'), str):
        child = (target / summary['optimized_summary']).resolve()
        if not child.is_relative_to(target) or child.name != 'summary.json':
            raise ValueError('invalid optimized summary path')
        return select_maps(child.parent, map_id=map_id, cloud=cloud)
    if not isinstance(summary,dict) or not isinstance(summary.get('maps'),list):
        raise ValueError('저장 지도 summary의 maps 형식이 올바르지 않습니다.')
    selected = []
    for row in summary['maps']:
        if not isinstance(row,dict):
            raise ValueError('저장 지도 map 항목 형식이 올바르지 않습니다.')
        identity = row.get('map_id')
        if not isinstance(identity,int) or isinstance(identity,bool) or identity < 0:
            raise ValueError('invalid map identity')
        if map_id is not None and identity != map_id:
            continue
        geometry = row.get('cloud' if cloud else 'mesh')
        if not isinstance(geometry,str):
            raise ValueError('저장 지도 summary에 유효한 PLY 경로가 없습니다.')
        base = folder if summary.get('paths_relative_to') == 'summary' else target
        path = (base/geometry).resolve()
        if not path.is_relative_to(folder.resolve()) or path.suffix.lower() != '.ply':
            raise ValueError('invalid saved map path')
        if not path.is_file():
            raise FileNotFoundError(str(path))
        selected.append((f'Map {identity}',path))
    if not selected:
        raise ValueError(f'요청한 map {map_id}의 저장 결과가 없습니다.')
    return selected


def load_trajectory(geometry_path: Path) -> list[tuple[float,float,float]]:
    """같은 backend/map의 최종 keyframe 위치만 시간순으로 읽는다.

    실시간 pose는 최종 지도와 최적화 상태가 다를 수 있어 대체 입력으로 쓰지 않는다.
    위치는 카메라 optical 원점이며 로봇 중심이나 지면으로 변환하지 않는다.
    """
    geometry_path = Path(geometry_path).resolve()
    backend = next((key for key,value in FOLDERS.items() if value == geometry_path.parent.name),None)
    match = re.fullmatch(r'map_(\d+)_(?:mesh|cloud)\.ply',geometry_path.name)
    if backend is None or match is None:
        return []
    identity = int(match[1])
    source = geometry_path.parent.parent/backend/f'map_{identity}_keyframes.csv'
    if not source.is_file():
        return []
    rows = []
    with source.open(newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'map_id','keyframe_id','source_stamp','x','y','z'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'이동 경로 CSV 열이 올바르지 않습니다: {source}')
        for row in reader:
            try:
                map_id, keyframe = int(row['map_id']), int(row['keyframe_id'])
                stamp = float(row['source_stamp'])
                point = tuple(float(row[key]) for key in ('x','y','z'))
            except (TypeError,ValueError) as error:
                raise ValueError(f'이동 경로 CSV 값이 올바르지 않습니다: {source}') from error
            if map_id != identity or keyframe < 0 or not all(math.isfinite(v) for v in (stamp,*point)):
                raise ValueError(f'이동 경로의 map ID 또는 위치·시각이 올바르지 않습니다: {source}')
            rows.append((stamp,keyframe,point))
    rows.sort(key=lambda row:(row[0],row[1]))
    points = [row[2] for row in rows]
    matrix,gravity = map_frame(geometry_path)
    if gravity and points:
        import numpy as np
        positions = np.asarray(points) @ matrix[:3,:3].T + matrix[:3,3]
        return [tuple(row) for row in positions]
    return points


def map_frame(geometry_path: Path):
    """PLY가 저장된 좌표계와 동일한 변환을 경로에도 적용한다."""
    import numpy as np
    from experiments.handheld_mapping.reconstruct import transform
    match = re.fullmatch(r'map_(\d+)_(?:mesh|cloud)\.ply',geometry_path.name)
    summary = geometry_path.parent/'summary.json'
    if match is None or geometry_path.parent.name not in FOLDERS.values() or not summary.is_file():
        return np.eye(4),False
    data = json.loads(summary.read_text())
    row = next((row for row in data['maps'] if row['map_id'] == int(match[1])),None)
    if row is None or row.get('coordinate_system') != 'gravity_z_up':
        return np.eye(4),False
    return transform(row['world_from_slam']),True


def trajectory_geometries(points: list[tuple[float,float,float]]) -> list:
    """3D 위치를 그대로 연결하고 시작/끝을 구분한다. 원본 PLY는 수정하지 않는다."""
    import open3d as o3d
    if not points:
        return []
    geometries = []
    if len(points) > 1:
        line = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector([(i,i+1) for i in range(len(points)-1)]))
        line.paint_uniform_color([.12,.5,1.])
        geometries.append(line)
    endpoints = [(points[0],[.15,.9,.3])]
    if len(points) > 1:
        endpoints.append((points[-1],[1.,.2,.15]))
    for point,color in endpoints:
        marker = o3d.geometry.TriangleMesh.create_sphere(radius=.04)
        marker.translate(point)
        marker.compute_vertex_normals()
        marker.paint_uniform_color(color)
        geometries.append(marker)
    return geometries


def display_geometry(geometry: Any) -> Any:
    """저장된 RGB를 유지하고 색이 없는 형상에만 기본 표시색을 적용한다."""
    colored = geometry.has_colors() if hasattr(geometry, 'has_colors') else geometry.has_vertex_colors()
    if not colored:
        geometry.paint_uniform_color([.65, .78, .85])
    return geometry


def show_maps(selected: list[tuple[str,Path]], cloud: bool = False, trajectory: bool = True,
              terrain_colors: bool = True) -> None:
    import open3d as o3d
    windows = []
    try:
        for label,path in selected:
            _matrix,gravity_aligned = map_frame(path)
            geometry = o3d.io.read_point_cloud(str(path)) if cloud else o3d.io.read_triangle_mesh(str(path))
            if cloud or len(geometry.triangles) == 0:
                if not cloud: geometry = o3d.io.read_point_cloud(str(path))
                if len(geometry.points) == 0: raise ValueError(f'빈 point cloud: {path}')
                if not geometry.has_normals(): geometry.estimate_normals()
            else:
                geometry.compute_vertex_normals()
            display_geometry(geometry)
            color_legend = ''
            if terrain_colors:
                from experiments.handheld_mapping.surface_colors import apply_surface_colors, LEGEND
                try:
                    classified = apply_surface_colors(geometry, path)
                    if classified is not None:
                        color_legend = LEGEND+' | '
                        print(f'{label}: {LEGEND} · 지면 {classified["ground"]:,} / '
                              f'장애물 {classified["obstacle"]:,} / 미분류 {classified["unknown"]:,} '
                              f'· {classified["terrain"]}', flush=True)
                    else:
                        print(f'{label}: 대응 지형 분류가 없어 기본 색으로 표시합니다.', flush=True)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    print(f'{label}: 지면 색상 표시 생략: {error}', flush=True)
            overlays = []
            if trajectory:
                try:
                    points = load_trajectory(path)
                    overlays = trajectory_geometries(points)
                    if points:
                        print(f'{label}: 최종 ORB 키프레임 {len(points)}개로 카메라 추정 경로 표시 '
                              '(파랑 경로 / 초록 시작 / 빨강 끝; 로봇 중심·실측 경로 아님)',flush=True)
                    else:
                        print(f'{label}: 대응하는 최종 키프레임이 없어 경로를 표시하지 않습니다.',flush=True)
                except (ValueError,OSError,csv.Error) as error:
                    print(f'{label}: 이동 경로 표시 생략: {error}',flush=True)
            viewer = o3d.visualization.Visualizer()
            suffix = (' | Gravity Z-up' if gravity_aligned else '') + (' | ORB camera trajectory' if overlays else '')
            if not viewer.create_window(window_name=f'Rover 3D | {color_legend}{path.parent.parent.name} | {label}{suffix}',
                                        width=1100,height=780):
                raise RuntimeError('3D 창 초기화 실패. tools/view_mapping.sh로 DISPLAY/X11 환경에서 실행하세요.')
            windows.append((viewer,[geometry,*overlays]))
            if not viewer.add_geometry(geometry): raise RuntimeError(f'3D geometry 표시 실패: {path}')
            for overlay in overlays:
                if not viewer.add_geometry(overlay): raise RuntimeError(f'이동 경로 표시 실패: {path}')
            viewer.get_render_option().background_color=[.07,.09,.12]
            viewer.get_render_option().mesh_show_back_face=True
            viewer.get_render_option().line_width=3.
            viewer.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Color
            viewer.get_render_option().point_color_option = o3d.visualization.PointColorOption.Color
            control=viewer.get_view_control()
            control.set_lookat(geometry.get_center()); control.set_up([0,0,1] if gravity_aligned else [0,-1,0])
            control.set_front([1,-1,.7] if gravity_aligned else [0,0,-1]); control.set_zoom(.7)
            viewer.poll_events(); viewer.update_renderer()
            print('VIEWER_READY',label,path,flush=True)
        while windows:
            for viewer,geometry in windows[:]:
                if viewer.poll_events():
                    viewer.update_renderer()
                else:
                    viewer.destroy_window()
                    windows.remove((viewer,geometry))
            time.sleep(.02)
    finally:
        for viewer,_geometry in windows: viewer.destroy_window()


def main() -> None:
    parser=argparse.ArgumentParser(description='최신 촬영 또는 지정 RUN/PLY의 저장된 3D 지도 보기')
    parser.add_argument('target',nargs='?',type=Path)
    parser.add_argument('--backend',choices=['auto',*FOLDERS],default='auto')
    parser.add_argument('--map-id',type=int)
    parser.add_argument('--cloud',action='store_true',help='mesh 대신 point cloud 표시')
    parser.add_argument('--no-trajectory',action='store_true',help='최종 ORB 카메라 추정 경로 숨기기')
    parser.add_argument('--no-terrain-colors',action='store_true',help='지면·장애물 분류 색을 끄고 단색으로 표시')
    parser.add_argument('--depth-max', type=float, help='최대 depth [m]; 거리별 결과를 생성/재사용해 표시 (예:1,2,3)')
    parser.add_argument('--build-only', action='store_true', help='--depth-max 결과만 생성하고 3D 창은 열지 않음')
    args=parser.parse_args()
    if args.build_only and args.depth_max is None:
        parser.error('--build-only에는 --depth-max가 필요합니다')
    try:
        target = args.target
        if args.depth_max is not None:
            from experiments.handheld_mapping.depth_variants import ensure_depth_variant
            target = ensure_depth_variant(resolve_target(target), args.depth_max, args.backend,
                                           progress=lambda message: print(message, flush=True))
        selected=select_maps(target,args.backend,args.map_id,args.cloud)
        if args.build_only:
            print(f'결과 준비 완료: {target}', flush=True)
            return
        if len(selected)>1: print('여러 map은 각각 별도 좌표계·창으로 표시합니다.',flush=True)
        print('왼쪽 드래그 회전 | 휠 확대 | Shift+왼쪽 드래그 이동',flush=True)
        show_maps(selected,args.cloud,trajectory=not args.no_trajectory,
                  terrain_colors=not args.no_terrain_colors)
    except (ValueError,FileNotFoundError,RuntimeError) as error:
        parser.exit(1,str(error)+'\n')


if __name__ == '__main__':
    main()
