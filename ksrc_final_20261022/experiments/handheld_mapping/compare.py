# ================================================================
# compare.py
# 설명: 저장한 동일 stereo로 ORB-LINE을 재실행하고 추적·처리시간·TSDF 결과를 별도 보존한다.
# 사용법: .venv-mapping/bin/python -m experiments.handheld_mapping.compare RUN
# ================================================================
import argparse
import csv
import json
import hashlib
import os
from pathlib import Path
import math
import subprocess
import tarfile

import numpy as np

from experiments.handheld_mapping.storage import atomic_json
from experiments.handheld_mapping.recording import disk_index

ROOT = Path(__file__).resolve().parents[2]


def runner_environment(point_baseline: bool) -> dict:
    environment = dict(os.environ)
    libraries = [ROOT / "outputs/handheld_mapping/native_dependencies/pangolin/lib"]
    if point_baseline:
        libraries.insert(0, ROOT / "outputs/handheld_mapping/orb3_comparison_build/runtime")
    else:
        libraries.insert(0, ROOT / "outputs/handheld_mapping/orbline_build")
        libraries.insert(1, ROOT / "external/ORB-LINE-SLAM/Thirdparty/g2o/lib")
    environment["LD_LIBRARY_PATH"] = ":".join(str(path) for path in libraries)
    return environment


def summarize_timings(path: Path) -> dict:
    count = okay = 0
    maps = set()
    with disk_index(path.parent) as database:
        database.execute('CREATE TABLE times(value REAL NOT NULL)')
        with path.open() as stream:
            for row in csv.DictReader(stream):
                value = float(row['process_ms'])
                if not math.isfinite(value) or value < 0:
                    raise ValueError('invalid processing times')
                database.execute('INSERT INTO times VALUES (?)', (value,))
                count += 1
                okay += int(row['state']) == 2
                if int(row['map_id']) >= 0:
                    maps.add(int(row['map_id']))
        if not count:
            raise ValueError('no tracking timing samples')
        database.execute('CREATE INDEX sorted_times ON times(value)')
        def percentile(probability: float) -> float:
            rank = (count-1)*probability
            lower = math.floor(rank)
            values = [row[0] for row in database.execute('SELECT value FROM times ORDER BY value LIMIT 2 OFFSET ?', (lower,))]
            return values[0] if len(values) == 1 else values[0]+(rank-lower)*(values[1]-values[0])
        return {'frames': count, 'tracking_ok_fraction': okay/count, 'map_ids': sorted(maps),
                'process_ms_median': percentile(.5), 'process_ms_p95': percentile(.95),
                'scope': 'offline tracking-call elapsed time on executing machine; not Pi FPS or display latency',
                'gt_accuracy_evaluated': False}


def stereo_pairs(run: Path) -> list[tuple[int, str, str]]:
    from experiments.handheld_mapping.offline import iter_pairs
    return list(iter_pairs(run))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(65536), b''):
            digest.update(chunk)
    return digest.hexdigest()


def line_settings(calibration: dict) -> str:
    left = calibration["left"]
    right = calibration["right"]
    k = np.asarray(left["k"], dtype=float)
    baseline = float(calibration["T_c1_c2"][0][3])
    if not (baseline > 0 and np.isfinite(k).all() and k[0] > 0 and k[4] > 0
            and np.allclose(k, right["k"]) and all(abs(v) < 1e-8 for v in left["d"]+right["d"])):
        raise ValueError("rectified stereo calibration required for ORB-LINE")
    values = {"Camera.type": "PinHole", "Camera.fx": float(k[0]), "Camera.fy": float(k[4]),
              "Camera.cx": float(k[2]), "Camera.cy": float(k[5]), "Camera.k1": 0., "Camera.k2": 0.,
              "Camera.p1": 0., "Camera.p2": 0., "Camera.bFishEye": 0, "Camera.width": left["width"],
              "Camera.height": left["height"], "Camera.fps": float(calibration["camera_profile"].split("x")[-1].split(":")[0]),
              "Camera.bf": baseline*float(k[0]), "Camera.RGB": 0, "ThDepth": 40.,
              "ORBextractor.nFeatures": 1200, "ORBextractor.scaleFactor": 1.2, "ORBextractor.nLevels": 8,
              "ORBextractor.iniThFAST": 20, "ORBextractor.minThFAST": 7,
              "SLAM": 0, "extractor": 0, "nfeatures": 400, "lsd_refine": 0, "lsd_scale": .5,
              "levels": 1, "scale": 2., "Viewer.KeyFrameSize": .05, "Viewer.KeyFrameLineWidth": 1.,
              "Viewer.GraphLineWidth": .9, "Viewer.PointSize": 2., "Viewer.CameraSize": .08,
              "Viewer.CameraLineWidth": 3., "Viewer.ViewpointX": 0., "Viewer.ViewpointY": -.7,
              "Viewer.ViewpointZ": -1.8, "Viewer.ViewpointF": 500.}
    return "%YAML:1.0\n---\n"+"".join(f"{key}: {json.dumps(value)}\n" for key, value in values.items())


def vocabularies() -> tuple[Path, Path]:
    folder = ROOT / "external/ORB-LINE-SLAM/Vocabulary"
    for archive, name in (("ORBvoc.txt.tar.gz", "ORBvoc.txt"), ("LSvoc.txt.tar.gz", "LSDvoc.txt")):
        target = folder / name
        if not target.exists():
            with tarfile.open(folder / archive) as tar:
                member = tar.getmember(name)
                with tar.extractfile(member) as source, target.open("xb") as output:
                    import shutil
                    shutil.copyfileobj(source, output)
    return folder / "ORBvoc.txt", folder / "LSDvoc.txt"


def run_orbline(run: Path) -> dict:
    run = Path(run).resolve()
    binary = ROOT / "outputs/handheld_mapping/orbline_build/rover_orbline"
    if not binary.is_file():
        raise FileNotFoundError("ORB-LINE 실행기가 없습니다: bash tools/build_orbline.sh")
    folder = run / "orbline"
    folder.mkdir(exist_ok=False)
    calibration = json.loads((run / "capture_calibration.json").read_text())["stereo"]
    settings = folder / "settings.yaml"
    settings.write_text(line_settings(calibration))
    index = folder / "pairs.tsv"
    from experiments.handheld_mapping.offline import iter_pairs
    paired = 0
    with index.open('w') as stream:
        for stamp, left, right in iter_pairs(run):
            stream.write(f'{stamp}\t{left}\t{right}\n')
            paired += 1
    if not paired:
        raise ValueError("동일 timestamp의 좌우 IR 원본이 없습니다")
    orb_voc, line_voc = vocabularies()
    with (folder / "runner.log").open("wb") as log:
        subprocess.run([str(binary), str(orb_voc), str(line_voc), str(settings), str(index), str(folder)],
                       cwd=folder, env=runner_environment(False), stdout=log, stderr=log, check=True, timeout=3600)
    if json.loads((folder / "runner_complete.json").read_text())["status"] != "complete":
        raise ValueError("ORB-LINE did not complete cleanly")
    summary = summarize_timings(folder / "timings.csv")
    summary["backend"] = "ORB-LINE-SLAM point+line; LSD .5 scale/1 level"
    summary["build_note"] = "line core uses O1/native and existing point core O3/native; timing difference includes compiler optimization differences"
    summary["source"] = json.loads((run / "capture_calibration.json").read_text()).get("source")
    point_binary = ROOT / "outputs/handheld_mapping/orb3_comparison_build/rover_orb3_compare"
    if point_binary.is_file():
        point_folder = run / "orb_offline"
        if point_folder.exists():
            marker = json.loads((point_folder/'runner_complete.json').read_text())
            point_metadata = json.loads((point_folder/'summary.json').read_text())
            calibration_hash = hashlib.sha256((run/'capture_calibration.json').read_bytes()).hexdigest()
            if (marker.get('status') != 'complete'
                    or point_metadata.get('input_calibration_sha256') != calibration_hash
                    or file_hash(point_folder/'pairs.tsv') != file_hash(index)):
                raise ValueError('existing point baseline input/completion mismatch; preserved without overwrite')
        else:
            from experiments.handheld_mapping.offline import run_point
            run_point(run)
        summary["same_input_point_baseline"] = summarize_timings(point_folder / "timings.csv")
    baseline = run / "orb/tracking.json"
    if baseline.exists():
        values = json.loads(baseline.read_text())
        summary["baseline_tracking"] = values
        summary["comparison_limit"] = "Only offline point baseline and offline line used identical laptop input; live Pi timing is not comparable. No ground-truth accuracy or Pi point-line speed verified."
    if (run / "raw/depth").is_dir():
        from experiments.handheld_mapping.reconstruct import reconstruct
        summary["dense"] = reconstruct(run, backend="orbline")
    atomic_json(folder / "comparison.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_orbline(args.run_dir), indent=2))


if __name__ == "__main__":
    main()
