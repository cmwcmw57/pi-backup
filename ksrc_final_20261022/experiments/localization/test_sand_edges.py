# ================================================================
# test_sand_edges.py
# 설명: sand_edges.py 의 기하(ψ·지면 수직거리 부호와 크기)를 합성 영상으로, 그리고 10-18 49e304 실제 프레임 1장의 고정값으로 확인한다.
# 사용법:
#   cd /home/su/rover-mapping && QT_QPA_PLATFORM=offscreen PYTHONNOUSERSITE=1 .venv-mapping/bin/python -m pytest experiments/localization/test_sand_edges.py -q
# ================================================================
from pathlib import Path

import cv2
import numpy as np
import pytest

from experiments.localization.sand_edges import K, edge_signal, heading_sensitivity, line_ground_offset, line_heading_deg, sand_mask

FRAME = Path('/home/su/rover-mapping/outputs/pi_1018/zip_shoot/촬영데이터_2026-10-18/20261018_123653_49e304/raw/left/1792294645297662109.png')
H = 0.16


@pytest.mark.parametrize('corner', ['top_left', 'top_right', 'bottom_left', 'bottom_right'])
def test_border_touching_sand_does_not_fill_exterior(corner: str) -> None:
    """모래가 어느 영상 모서리에 닿더라도 외부 배경은 남고 내부의 어두운 구멍만 메워야 한다."""
    img = np.zeros((160, 200), np.uint8)
    img[:120, :140] = 200
    img[40:60, 40:60] = 0
    if 'bottom' in corner:
        img = np.flipud(img)
    if 'right' in corner:
        img = np.fliplr(img)
    mask = sand_mask(img)
    assert mask[80, 80] == 1
    assert mask.mean() == pytest.approx(120 * 140 / img.size, abs=0.01)
    assert mask[img == 0].mean() < 0.1


def test_uniform_mask_has_no_fabricated_foreground() -> None:
    """빈 입력은 빈 마스크, 전체 밝은 입력은 전체 마스크여야 한다."""
    assert not sand_mask(np.zeros((80, 100), np.uint8)).any()
    assert sand_mask(np.full((80, 100), 200, np.uint8)).all()


def test_sand_strip_keeps_both_disconnected_exterior_regions() -> None:
    """영상 위아래에 닿는 띠 양쪽은 모두 외부 배경이며 한쪽을 구멍으로 메우면 안 된다."""
    img = np.zeros((160, 200), np.uint8)
    img[:, 60:140] = 200
    mask = sand_mask(img)
    assert mask[80, 100] == 1
    assert mask[80, 20] == mask[80, 180] == 0


def _basis(pitch_deg: float, roll_deg: float = 0.) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """피치(아래)·롤 카메라의 광학 좌표 up, 수평 앞 f, 수평 오른쪽 r 단위벡터."""
    p, r = np.radians(pitch_deg), np.radians(roll_deg)
    up = np.array([np.sin(r)*np.cos(p), -np.cos(r)*np.cos(p), -np.sin(p)])
    up /= np.linalg.norm(up)
    f = np.array([0, 0, 1.])-up*up[2]
    f /= np.linalg.norm(f)
    return up, f, np.cross(f, up)


def _render(left: float, right: float, psi_deg: float, pitch_deg: float = 20., roll_deg: float = 0., s0: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """높이 H 지면 위에서 길 축 방향 psi 인 띠(좌 left<0, 우 right>0 수직거리)를 640x480 밝은 띠로 그린다."""
    up, f, r = _basis(pitch_deg, roll_deg)
    a = np.radians(psi_deg)
    pts = []
    for s, side in ((s0, left), (6., left), (6., right), (s0, right)):
        fw = s*np.cos(a)-side*np.sin(a)
        rt = s*np.sin(a)+side*np.cos(a)
        X = -H*up+fw*f+rt*r
        pts.append(K@X/X[2] if X[2] > 0.02 else None)
    assert all(p is not None for p in pts)
    img = np.full((480, 640), 25, np.uint8)
    cv2.fillPoly(img, [np.array([p[:2] for p in pts]).round().astype(np.int32)], 200)
    return img, up


@pytest.mark.parametrize('left,right,psi,roll', [(-0.40, 0.30, 5., 0.), (-0.35, 0.35, -4., 3.), (-0.45, 0.25, 0., -3.)])
def test_synthetic_strip_recovers_heading_and_offsets(left: float, right: float, psi: float, roll: float) -> None:
    img, up = _render(left, right, psi, roll_deg=roll)
    sig = edge_signal(img, up, H)
    assert sig is not None
    assert sig.off_right == pytest.approx(right, abs=0.03)
    assert sig.off_left == pytest.approx(left, abs=0.03)
    assert sig.axis_deg == pytest.approx(psi, abs=1.5)
    assert sig.width == pytest.approx(right-left, abs=0.04)
    assert sig.center_m == pytest.approx(-(right+left)/2, abs=0.03)


def test_heading_shifts_one_to_one_with_yaw() -> None:
    """카메라가 반시계로 d 도 돌면(길이 시계 방향으로 d 도 더 보이면) ψ 가 d 만큼 늘어야 한다."""
    img0, up = _render(-0.4, 0.3, 0.)
    img1, _ = _render(-0.4, 0.3, 8.)
    a, b = edge_signal(img0, up, H), edge_signal(img1, up, H)
    assert b.axis_deg-a.axis_deg == pytest.approx(8., abs=1.5)


def test_horizon_parallel_line_is_flagged_unstable() -> None:
    up, _, _ = _basis(20.)
    horizon_row = K[1, 2]-K[0, 0]*np.tan(np.radians(20.))
    assert heading_sensitivity(np.array([0., horizon_row+8]), np.array([639., horizon_row+9]), up) > 3.
    assert heading_sensitivity(np.array([100., 470.]), np.array([260., 130.]), up) < 3.


def test_line_through_camera_foot_has_zero_offset_and_sign() -> None:
    up, f, r = _basis(20.)
    def pix(fw: float, rt: float) -> np.ndarray:
        X = -H*up+fw*f+rt*r
        return (K@X/X[2])[:2]
    assert abs(line_ground_offset(pix(0.5, 0.), pix(3., 0.), up, H)) < 1e-6
    assert line_ground_offset(pix(0.5, 0.3), pix(3., 0.3), up, H) == pytest.approx(0.3, abs=1e-6)
    assert line_ground_offset(pix(0.5, -0.3), pix(3., -0.3), up, H) == pytest.approx(-0.3, abs=1e-6)
    assert line_heading_deg(pix(0.5, 0.), pix(3., 0.), up) % 180. == pytest.approx(0., abs=1e-6) or line_heading_deg(pix(0.5, 0.), pix(3., 0.), up) == pytest.approx(180., abs=1e-6)


def test_axis_beyond_gate_is_rejected() -> None:
    """축 방향이 max_axis 를 넘으면 None(49e304 끝벽 프레임에서 ψ≈90° 선 두 개가 가장자리로 잡힌 실제 오검출을 막는 게이트)."""
    img, up = _render(-0.4, 0.3, 20.)
    assert edge_signal(img, up, H) is not None
    assert edge_signal(img, up, H, max_axis=10.) is None


@pytest.mark.skipif(not FRAME.exists(), reason='10-18 49e304 프레임 없음')
def test_real_frame_49e304_fixed_values() -> None:
    """실제 프레임 1장 고정값(회귀 방지용, 정답이 아님): h=0.16, up 은 acc+장착 오차 보정값."""
    up = np.array([-0.00047, -0.94317, -0.33231])
    sig = edge_signal(cv2.imread(str(FRAME), 0), up, H)
    assert sig is not None
    assert sig.off_right == pytest.approx(0.273, abs=0.01)
    assert sig.off_left == pytest.approx(-0.436, abs=0.01)
    assert sig.width == pytest.approx(0.709, abs=0.01)
    assert sig.axis_deg == pytest.approx(2.8, abs=0.5)
