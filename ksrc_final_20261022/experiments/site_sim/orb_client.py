# ================================================================
# orb_client.py
# 설명: 시뮬이 stereo 프레임마다 stream_orb 보조 프로세스에 파일로 요청하고 camera pose 응답을 lock-step으로 기다린다.
# 사용법: client = OrbClient(Path('/workspace/orbio')); result = client.track(stamp_ns, left_png, right_png)
# ================================================================
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation


class OrbClient:
    """요청 파일(<stamp_ns>.req)을 원자적으로 쓰고 응답(<stamp_ns>.pose)이 생길 때까지 기다린다."""

    def __init__(self, io: Path, ready_timeout_s: float=180., timeout_s: float=30.) -> None:
        self.io, self.timeout_s = Path(io), float(timeout_s)
        deadline = time.monotonic()+ready_timeout_s
        while not (self.io/'READY').exists():
            if (self.io/'FAILED').exists() or time.monotonic() > deadline:
                raise RuntimeError('ORB sidecar did not become ready')
            time.sleep(.2)

    def track(self, stamp_ns: int, left: Path, right: Path) -> dict:
        """한 쌍을 처리시키고 {'state','map_id','keypoints','world_from_camera'}를 반환한다."""
        request, response = self.io/f'{stamp_ns}.req', self.io/f'{stamp_ns}.pose'
        temporary = self.io/f'{stamp_ns}.req.tmp'
        temporary.write_text(f'{left}\n{right}\n')
        temporary.rename(request)
        deadline = time.monotonic()+self.timeout_s
        while not response.exists():
            if (self.io/'FAILED').exists():
                raise RuntimeError('ORB sidecar failed: '+(self.io/'FAILED').read_text().strip())
            if time.monotonic() > deadline:
                raise TimeoutError(f'ORB response timeout for {stamp_ns}')
            time.sleep(.002)
        values = response.read_text().split()
        response.unlink()
        request.unlink(missing_ok=True)
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat([float(v) for v in values[6:10]]).as_matrix()
        pose[:3, 3] = [float(v) for v in values[3:6]]
        return {'state': int(values[0]), 'map_id': values[1], 'keypoints': int(values[2]), 'world_from_camera': pose}
