# ================================================================
# stereo_capture.py
# 설명: 좌우 영상을 ORB용 임시 파일로 전달하거나 선택적으로 원본·timestamp·별도 GT를 기록한다.
# 사용법: from experiments.site_sim.stereo_capture import StereoCapture
# ================================================================
from pathlib import Path
import csv
import json
import numpy as np
from PIL import Image


class StereoCapture:
    """좌우 쌍을 ORB 임시 파일로 전달하거나 EuRoC 원본으로 보관한다."""

    def __init__(self, output: Path, calibration: dict, *, archive: bool=True, transport_dir: Path | None=None) -> None:
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
        self.archive=archive
        if not archive and transport_dir is None:raise ValueError('temporary stereo needs a shared transport directory')
        self.transport=Path(transport_dir) if transport_dir is not None else self.output
        if not archive:self.transport.mkdir(parents=True,exist_ok=True)
        self.width,self.height=int(calibration['width']),int(calibration['height'])
        self.last=-1
        (self.output/'calibration.json').write_text(json.dumps(calibration,indent=2,allow_nan=False))
        for side in ('cam0','cam1') if archive else ():
            root=self.output/'mav0'/side;(root/'data').mkdir(parents=True)
            (root/'data.csv').write_text('#timestamp [ns],filename\n')

    def write(self, stamp_ns: int, left: np.ndarray, right: np.ndarray, position: list, quaternion: list) -> tuple[Path,Path]:
        """같은 시각의 pair를 전달/보관하며 잘못된 입력은 쓰기 전에 거부한다."""
        if type(stamp_ns) is not int or stamp_ns<=self.last:raise ValueError('strictly increasing integer timestamp required')
        images=[np.asarray(a) for a in (left,right)]
        if any(a.dtype!=np.uint8 or a.shape!=(self.height,self.width,3) for a in images):raise ValueError('stereo RGB image shape/type mismatch')
        if np.asarray(position).shape!=(3,) or np.asarray(quaternion).shape!=(4,) or not np.isfinite([*position,*quaternion]).all():raise ValueError('finite GT required')
        filename=f'{stamp_ns}.png'
        paths=[]
        for side,im in zip(('cam0','cam1'),images):
            target=(self.output/'mav0'/side/'data'/filename if self.archive else self.transport/f'{side}.png')
            tmp=target.with_suffix('.tmp')
            Image.fromarray(im).save(tmp,format='PNG',compress_level=1 if self.archive else 0);tmp.replace(target)
            paths.append(target)
        if not self.archive:
            self.last=stamp_ns
            return tuple(paths)
        for side in ('cam0','cam1'):
            with (self.output/'mav0'/side/'data.csv').open('a') as f:csv.writer(f).writerow([stamp_ns,filename])
        with (self.output/'gt.jsonl').open('a') as f:
            f.write(json.dumps({'stamp_ns':stamp_ns,'position_m':list(position),'quaternion_wxyz':list(quaternion),'pose_source':'sim_ground_truth'},allow_nan=False)+'\n')
        self.last=stamp_ns
        return tuple(paths)

    def close(self) -> None:
        """보관하지 않는 ORB 전달용 두 장만 종료 시 지운다."""
        if not self.archive:
            for side in ('cam0','cam1'):(self.transport/f'{side}.png').unlink(missing_ok=True)
