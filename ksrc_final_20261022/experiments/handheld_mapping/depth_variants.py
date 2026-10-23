"""한 기록의 pose/raw를 유지하며 거리별 TSDF·지형 결과를 별도 폴더에 만든다."""
import json
import math
from pathlib import Path
import re
import shutil

from experiments.handheld_mapping.gravity import fingerprint, load_alignment, terrain_maps
from experiments.handheld_mapping.reconstruct import reconstruct
from experiments.handheld_mapping.storage import atomic_json


def variant_source(output_root: Path, backend: str) -> Path:
    """거리별 결과는 원래 촬영을 가리키고 pose/calibration의 작은 사본을 보존한다."""
    manifest = output_root/'depth_variant.json'
    if not manifest.is_file():
        return output_root
    data = json.loads(manifest.read_text())
    if (data.get('schema') != 1 or data.get('status') != 'complete'
            or data.get('backend') != backend or not isinstance(data.get('source_run'), str)):
        raise ValueError('거리별 결과가 완성되지 않았거나 backend가 다릅니다')
    return Path(data['source_run']).resolve()


def ensure_depth_variant(run: Path, depth_max: float, backend: str = 'auto', progress=None) -> Path:
    """같은 입력/설정의 완성본은 재사용하고 변경 시 새 revision으로 보존한다."""
    from experiments.handheld_mapping.viewer import FOLDERS, select_maps
    run = Path(run).expanduser().resolve()
    if not math.isfinite(depth_max) or not .3 < depth_max <= 10:
        raise ValueError('--depth-max는 0.3보다 크고 10 이하인 유한한 m 값이어야 합니다')
    if not run.is_dir() or (run/'depth_variant.json').exists():
        raise ValueError('--depth-max에는 원본 촬영 RUN 폴더를 지정하세요')
    if backend == 'auto':
        backend = next((name for name in FOLDERS if any((run/name).glob('map_*_keyframes.csv'))), None)
    if backend not in FOLDERS:
        raise ValueError('거리별 재생성에 필요한 최종 ORB 키프레임이 없습니다')
    keyfiles = sorted((run/backend).glob('map_*_keyframes.csv'))
    if not keyfiles:
        raise ValueError(f'{backend}의 최종 키프레임이 없습니다')
    identities = []
    for path in keyfiles:
        match = re.fullmatch(r'map_(\d+)_keyframes.csv', path.name)
        if match is None:
            raise ValueError('invalid keyframe map identity')
        identities.append(int(match[1]))
    sources = [run/'capture_calibration.json', run/'frames.jsonl', *keyfiles]
    sources += [path for identity in identities
                if (path := run/'gravity'/backend/f'map_{identity}.json').is_file()]
    code_root = Path(__file__).parent
    pipeline_files = ('depth_variants.py', 'reconstruct.py', 'depth_filter.py', 'terrain_replay.py',
                      'ground.py', 'terrain.py', 'costmap.py', 'gravity.py', 'recording.py')
    source_hashes = {str(path.relative_to(run)): fingerprint(path) for path in sources}
    cache_key = {'source_run': str(run), 'backend': backend, 'depth_max_m': float(depth_max),
                 'voxel_size_m': .01, 'source_sha256': source_hashes,
                 'pipeline_sha256': {name: fingerprint(code_root/name) for name in pipeline_files}}
    tag = str(int(depth_max)) if float(depth_max).is_integer() else str(float(depth_max)).replace('.', 'p')
    name = f'depth_{tag}m'
    parent = run/'depth_variants'/backend
    candidates = []
    if parent.is_dir():
        for path in parent.iterdir():
            match = re.fullmatch(re.escape(name)+r'(?:_v([2-9]\d*|1\d+))?', path.name)
            if path.is_dir() and match:
                candidates.append((int(match[1] or 1), path))
    for _, path in sorted(candidates, reverse=True):
        try:
            saved = json.loads((path/'depth_variant.json').read_text())
            if saved.get('status') != 'complete' or saved.get('cache_key') != cache_key:
                continue
            for source in [run/'capture_calibration.json', *keyfiles]:
                name_in_run = str(source.relative_to(run))
                if fingerprint(path/name_in_run) != source_hashes[name_in_run]:
                    raise ValueError('changed variant pose/calibration snapshot')
            select_maps(path, backend=backend)
            for row in saved.get('terrain', {}).get('maps', {}).values():
                terrain_dir = Path(row['output_dir'])
                if not (terrain_dir/'terrain.npz').is_file() or not (terrain_dir/'summary.json').is_file():
                    raise FileNotFoundError('missing terrain result')
            if progress:
                progress(f'{depth_max:g}m 기존 결과 사용: {path}')
            return path
        except (OSError, ValueError, KeyError, TypeError):
            continue
    parent.mkdir(parents=True, exist_ok=True)
    revision = max((value for value, _ in candidates), default=0)+1
    while True:
        output = parent/(name if revision == 1 else f'{name}_v{revision}')
        try:
            output.mkdir()
            break
        except FileExistsError:
            revision += 1
    manifest = {'schema': 1, 'status': 'building', 'source_run': str(run), 'backend': backend,
                'depth_max_m': float(depth_max), 'range_kind': 'camera optical Z, not map radius',
                'raw_and_orb_poses_unchanged': True, 'cache_key': cache_key}
    atomic_json(output/'depth_variant.json', manifest)
    try:
        # These snapshots keep the old variant's path/colours valid if original poses later change.
        for source in [run/'capture_calibration.json', *keyfiles]:
            target = output/source.relative_to(run)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        gravity = {}
        for identity in identities:
            alignment = load_alignment(run, backend, identity)
            if alignment is not None:
                gravity[identity] = alignment
        if progress:
            progress(f'{depth_max:g}m: 원본 depth + 최종 ORB pose로 3D 지도 생성 중')
        dense = reconstruct(run, backend=backend, depth_max=depth_max,
                            gravity_alignments=gravity, output_root=output)
        integrated_maps = {row['map_id'] for row in dense['maps']}
        gravity = {key: value for key, value in gravity.items() if key in integrated_maps}
        if gravity:
            if progress:
                progress(f'{depth_max:g}m: 같은 거리 제한으로 지면·장애물 분류 중')
            terrain = terrain_maps(run, backend, {'maps': gravity}, depth_max=depth_max, output_root=output)
            if terrain['failed_maps']:
                raise ValueError(f'지형 결과 생성 실패: {terrain["failed_maps"]}')
        else:
            terrain = {'status': 'unavailable', 'maps': {}, 'reason': '이 기록에 고정 중력 기준이 없습니다'}
        if source_hashes != {str(path.relative_to(run)): fingerprint(path) for path in sources}:
            raise ValueError('재처리 중 원본 입력이 바뀌었습니다. 다시 실행하세요')
        manifest.update(status='complete', dense_status=dense['status'], terrain=terrain,
                        maps_without_terrain=sorted(integrated_maps-set(gravity)))
        atomic_json(output/'depth_variant.json', manifest)
        if progress:
            progress(f'{depth_max:g}m 결과 저장 완료: {output}')
        return output
    except Exception as error:
        manifest.update(status='failed', error=str(error))
        atomic_json(output/'depth_variant.json', manifest)
        raise
