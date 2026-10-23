# ================================================================
# stl_to_usda.py
# 설명: binary STL(mm)을 미터 단위 단일 Mesh USDA로 변환한다. pxr 없이 텍스트로 쓰므로 어디서든 실행된다.
# 사용법:
#   python3 -m experiments.orbslam3_ros2.stl_to_usda 모래경사_데모.stl yeouido_ramps.usda
# ================================================================
import struct
import sys
from pathlib import Path

import numpy as np


def convert(stl: Path, usda: Path, scale: float = 0.001) -> tuple[int, int]:
    """STL 삼각형을 정점 병합 후 USDA Mesh로 쓰고 (정점 수, 삼각형 수)를 반환한다."""
    data = stl.read_bytes()
    count = struct.unpack("<I", data[80:84])[0]
    record = np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")])
    triangles = np.frombuffer(data[84:], dtype=record, count=count)["v"].reshape(-1, 3) * scale
    points, inverse = np.unique(np.round(triangles, 6), axis=0, return_inverse=True)
    lo, hi = points.min(0), points.max(0)
    text = (
        '#usda 1.0\n(\n    defaultPrim = "Scene"\n    metersPerUnit = 1\n    upAxis = "Z"\n)\n\n'
        'def Xform "Scene"\n{\n    def Mesh "Ramps"\n    {\n'
        f"        float3[] extent = [({lo[0]}, {lo[1]}, {lo[2]}), ({hi[0]}, {hi[1]}, {hi[2]})]\n"
        f"        int[] faceVertexCounts = [{', '.join(['3'] * count)}]\n"
        f"        int[] faceVertexIndices = [{', '.join(map(str, inverse.ravel()))}]\n"
        f"        point3f[] points = [{', '.join(f'({p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f})' for p in points)}]\n"
        '        uniform token subdivisionScheme = "none"\n    }\n}\n'
    )
    usda.write_text(text)
    return len(points), count


if __name__ == "__main__":
    print("vertices, triangles =", convert(Path(sys.argv[1]), Path(sys.argv[2])))
