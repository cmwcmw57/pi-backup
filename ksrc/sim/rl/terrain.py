"""절차적 지형 생성 + 난이도 커리큘럼.

학습은 **랜덤 지형**으로 한다.  실제 경기장 STL 하나로 학습하면 그 지형을
외워버려서 일반화가 안 되고, 대회 규정에도 "경기장 구성은 변경될 수 있음"
이라고 명시돼 있다.  실제 지형(assets/arena_hfield.npz)은 **검증셋**으로만 쓴다.

커리큘럼이 필수인 이유: 30도 모래 경사에서 시작하면 정책이 단 한 번도 성공을
못 해서 보상 신호가 0이고 영원히 학습이 안 된다.  난이도를 0 -> 1 로 서서히
올리면서 성공률이 임계를 넘을 때만 다음 단계로 간다.

MuJoCo hfield 데이터는 [0,1] 정규화 값이고 실제 높이 = data * size[2] 이다.
모델을 다시 컴파일하지 않고 m.hfield_data 만 덮어쓰면 리셋이 싸진다.
"""
from dataclasses import dataclass
import numpy as np

# 학습 맵은 **일부러 크게** 잡는다 (8x8m).  대회장 크기(2.5x5.5m)로 학습하면
# 20초 에피소드에 6m 를 주행하는 로버가 계속 경계에 부딪히고, 경계를 벽으로
# 막으면 이번엔 벽에 박혀서 고착률이 100% 가 된다.  경계 자체를 없애는 게 맞다.
# 지형 통계(분화구 밀도/경사 분포)는 그대로이므로 학습 대상은 달라지지 않는다.
# 평가만 실제 대회장 지형(4x5m)에서 한다.
ARENA_X, ARENA_Y = 8.0, 8.0     # 학습용. 실제 규사 지형(3.2x4.0m)보다 넉넉하되
                                # 램프가 누적돼 기복이 터지지 않을 만큼만
EVAL_X, EVAL_Y = 4.0, 5.0       # 암석 지형(착륙지) STL 범위 -- 후순위
SAND_X, SAND_Y = 3.2, 4.0       # 규사 경사지형 STL 범위 -- **주 평가 대상**

# 평가용 실측 높이맵.  규사가 기본값이다. 암석지형은 아주 짧게 통과할 구간이라
# 후순위로 두고, 주행 성능은 규사 경사지형에서 판정한다.
import pathlib as _pl

# **모듈 기준 절대경로.**  상대경로("assets/...") 로 두면 다른 디렉토리(ros/ 등)에서
# import 할 때 파일을 못 찾는다.
_ASSETS = str(_pl.Path(__file__).resolve().parent / "assets")

ARENA_FILES = {"sand": (_ASSETS + "/sand_hfield.npz", SAND_X, SAND_Y),
               "rock": (_ASSETS + "/arena_hfield.npz", EVAL_X, EVAL_Y)}
HF_N = 384                      # 격자 (학습 31mm / 평가 10~13mm 해상도)


# 실제 규사 STL 을 10mm 격자로 재보면 경사가 **이산 집합**으로 딱 떨어진다.
#   0~3도 53.8% | 3~10도 29.8% | 10~20도 10.5% | 28~35도 4.7% | >35도 1.2%(벽)
# 대회측이 말한 "0/5/7/15/30도" 와 일치한다.  절차 생성도 같은 집합에서 뽑는다.
# (예전에는 uniform(0.2,1.0)*30도 로 뽑아서 3~10도 구간이 1.2% 밖에 안 나왔다.)
SLOPE_SET = np.array([0.0, 5.0, 7.0, 15.0, 30.0])
SLOPE_W1 = np.array([0.540, 0.150, 0.150, 0.105, 0.055])   # d=1: 실측 분포
SLOPE_W0 = np.array([0.850, 0.150, 0.000, 0.000, 0.000])   # d=0: 거의 평지


def slope_weights(d: float):
    w = (1.0 - d) * SLOPE_W0 + d * SLOPE_W1
    return w / w.sum()


@dataclass
class TerrainSpec:
    """실제 규사 지형의 구조를 따른다 -- 분화구 밭이 아니라 **계단식 평면**이다.

    대회 STL 실측 (3.20 x 4.00 m, 기복 450mm):
        0도  6.99 m^2 (55%)   5도 3.20 (25%)   7도 0.64 (5%)
       15도  1.34 m^2 (10%)  30도 0.62 (5%)   <- 최대 30도
    평지가 절반이고, 승부는 5% 짜리 30도 램프 하나에서 갈린다.
    """
    max_slope:   tuple = (2.0, 30.0)     # 난이도가 올리는 최대 경사각. 30도가 상한
    n_band:      tuple = (3, 9)          # (미사용) 띠 개수는 폭 누적으로 결정된다
    rough_amp:   tuple = (0.001, 0.010)  # 모래 표면 잔거칠기 [m]
    relief_max:  tuple = (0.08, 0.50)    # 가장 높은 고원의 높이 상한 [m].
                                         # 실측 규사 지형 기복이 0.45m

    def at(self, d: float) -> dict:
        d = float(np.clip(d, 0.0, 1.0))
        lerp = lambda t: t[0] + d * (t[1] - t[0])
        return dict(max_slope=np.radians(lerp(self.max_slope)),
                    n_band=int(round(lerp(self.n_band))),
                    rough=lerp(self.rough_amp), relief=lerp(self.relief_max))


def _value_noise(n, rng, octaves=4, persistence=0.5):
    """scipy 없이 만드는 프랙탈 노이즈 (격자 랜덤 + 쌍선형 업샘플의 합)."""
    out = np.zeros((n, n)); amp = 1.0; tot = 0.0
    for o in range(octaves):
        g = max(2, 2 ** (o + 2))
        base = rng.standard_normal((g, g))
        yi = np.linspace(0, g - 1, n); xi = np.linspace(0, g - 1, n)
        y0 = np.floor(yi).astype(int).clip(0, g - 2); x0 = np.floor(xi).astype(int).clip(0, g - 2)
        fy = (yi - y0)[:, None]; fx = (xi - x0)[None, :]
        b = base[np.ix_(y0, x0)]; br = base[np.ix_(y0, x0 + 1)]
        bb = base[np.ix_(y0 + 1, x0)]; bbr = base[np.ix_(y0 + 1, x0 + 1)]
        out += amp * ((b * (1 - fx) + br * fx) * (1 - fy) + (bb * (1 - fx) + bbr * fx) * fy)
        tot += amp; amp *= persistence
    return out / tot


def _clamp_slope(Z, cell, max_slope, iters=40):
    """로버 스케일 경사가 max_slope 를 넘는 곳을 반복 평활해 상한을 강제한다.

    대회 경기장은 "최대 30도, 그보다 가파른 건 없음" 이다. 띠 경계나 잔거칠기
    때문에 그걸 넘는 지점이 생기면 실제로 존재하지 않는 지형에서 학습하게 된다.
    """
    lim = np.tan(max_slope)
    k = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]) / 4.0
    for _ in range(iters):
        gy, gx = np.gradient(Z, cell, cell)
        bad = np.hypot(gx, gy) > lim
        if not bad.any():
            break
        sm = (np.roll(Z, 1, 0) + np.roll(Z, -1, 0) +
              np.roll(Z, 1, 1) + np.roll(Z, -1, 1)) / 4.0
        grow = bad | np.roll(bad, 1, 0) | np.roll(bad, -1, 0) | \
               np.roll(bad, 1, 1) | np.roll(bad, -1, 1)
        Z = np.where(grow, 0.5 * Z + 0.5 * sm, Z)
    return Z


def generate(difficulty: float, rng, spec: TerrainSpec = TerrainSpec(), n: int = HF_N):
    """계단식 램프 지형. 실제 규사 지형과 같은 구조를 랜덤 생성한다.

    한 축을 따라 폭이 랜덤한 띠로 나누고, 띠마다 경사를 뽑아 높이를 누적한다.
    누적 높이는 relief 상한 안에 가두고(안 그러면 8m 맵에서 기복이 수 m 로
    터진다), 마지막에 경사 상한을 강제한다.
    """
    p = spec.at(difficulty)
    ys = np.linspace(-ARENA_Y / 2, ARENA_Y / 2, n)
    xs = np.linspace(-ARENA_X / 2, ARENA_X / 2, n)
    X, Y = np.meshgrid(xs, ys)
    cell = ARENA_X / n

    # --- 고원 + 둘레 램프 --------------------------------------------------
    # 한 축에 투영해 띠를 쌓는 방식(예전)은 경사 분포는 맞출 수 있어도 기하가
    # 틀렸다: 모든 램프가 평행한 줄무늬가 되어, 로버가 등고선과 나란히 가면
    # 경사를 한 번도 안 올라간다.  실측 STL 은 **ㄷ자 고원** 이고 30도 면이
    # 닫힌 테두리를 그린다 -- 넘지 않고는 못 지나간다.
    #
    # 그래서 사각형 고원을 몇 개 쌓고, 각 고원의 **거리변환**으로 둘레 전체에
    # 램프를 두른다.  거리 D 에 대해 높이 h*clip(1-D/W, 0, 1) 을 더하면 램프
    # 경사는 정확히 atan(h/W) 이고, 그 경사가 고원 둘레를 한 바퀴 감싼다.
    from scipy import ndimage
    w_sl = slope_weights(float(np.clip(difficulty, 0.0, 1.0)))
    steep = SLOPE_SET[SLOPE_SET > 0]
    w_steep = w_sl[SLOPE_SET > 0] / w_sl[SLOPE_SET > 0].sum()

    # 고원을 **합이 아니라 최댓값**으로 쌓는다.  더하면 겹친 램프의 경사가 서로
    # 더해져 설계한 이산 경사가 뭉개지고(급사면 0.5%, 평지 30%), 기복도 고원
    # 개수만큼 터진다.  최댓값으로 쌓으면 겹침이 그대로 계단식 단이 되고, 각
    # 램프는 제 경사를 유지하며, 전체 기복은 가장 높은 고원 하나로 묶인다.
    W_MAX = 2.0                                    # 램프가 맵을 통째로 먹지 않게
    Z = np.zeros((n, n))
    for pi in range(int(rng.integers(3, 6))):
        # 높이를 먼저 뽑고, 그 높이에서 **램프 폭이 맵에 들어가는 경사만** 고른다.
        # 경사를 먼저 뽑으면 완경사(5도)+큰 높이에서 램프 폭이 7m 가 되고, 반대로
        # 폭을 먼저 맞추면 완경사 고원이 전부 낮아져 기복이 136mm 로 주저앉는다.
        # 실제 지형도 큰 단차는 30도 면이 받고 완경사는 낮고 넓은 둔덕이다.
        # 첫 고원 하나가 전체 기복을 정하고(높고 급함), 나머지는 낮게 깔아
        # 완경사를 만든다.  전부 높게 뽑으면 폭 제한(W_MAX) 때문에 5/7도 램프가
        # 전멸해 3~10도 구간이 1% 로 내려간다 -- 실측은 29.8% 다.
        h = (float(rng.uniform(0.75, 1.0)) if pi == 0
             else float(rng.uniform(0.08, 0.45))) * p["relief"]
        fit = steep[h / np.tan(np.radians(steep)) <= W_MAX]
        if len(fit) == 0:
            s_deg = float(steep.max())
        else:
            wf = w_steep[np.isin(steep, fit)]
            s_deg = float(rng.choice(fit, p=wf / wf.sum()))
        W = h / np.tan(np.radians(s_deg))          # 램프 폭 -> 경사 = s_deg
        half_x = float(rng.uniform(min(W, 0.25 * ARENA_X), 0.30 * ARENA_X))
        half_y = float(rng.uniform(min(W, 0.25 * ARENA_Y), 0.30 * ARENA_Y))
        cx = float(rng.uniform(-0.3 * ARENA_X, 0.3 * ARENA_X))
        cy = float(rng.uniform(-0.3 * ARENA_Y, 0.3 * ARENA_Y))
        inside = (np.abs(X - cx) <= half_x) & (np.abs(Y - cy) <= half_y)
        if not inside.any():
            continue
        D = ndimage.distance_transform_edt(~inside, sampling=(cell, cell))
        Z = np.maximum(Z, h * np.clip(1.0 - D / max(W, 1e-3), 0.0, 1.0))

    Z -= Z.mean()

    Z += p["rough"] * _value_noise(n, rng)
    # 요철 노이즈가 만든 첨두만 깎는다. 램프(최대 30도) 자체는 남겨야 한다.
    Z = _clamp_slope(Z, cell, p["max_slope"] + np.radians(6.0))

    flat = np.hypot(X, Y) < 0.45                      # 출발 지점은 평평하게
    if flat.any():
        Z -= Z[flat].mean()
        Z = np.where(flat, Z * (np.hypot(X, Y) / 0.45) ** 2, Z)
    return Z, p


def eval_extent(kind: str = "sand"):
    """평가 지형의 (가로, 세로) [m]."""
    return ARENA_FILES[kind][1], ARENA_FILES[kind][2]


def load_arena(kind: str = "sand", n: int = HF_N):
    """실제 경기장 STL 에서 뽑은 높이맵 (검증 전용)."""
    z = np.load(ARENA_FILES[kind][0])
    H = np.nan_to_num(z["H"], nan=float(np.nanmedian(z["H"])))
    yi = np.linspace(0, H.shape[0] - 1, n).astype(int)
    xi = np.linspace(0, H.shape[1] - 1, n).astype(int)
    return H[np.ix_(yi, xi)] - np.median(H)


def to_hfield(model, Z, hf_id=0):
    """높이맵(m)을 MuJoCo hfield 로 써넣는다. 모델 재컴파일 불필요."""
    zmin, zmax = float(Z.min()), float(Z.max())
    span = max(zmax - zmin, 1e-3)
    model.hfield_size[hf_id, 2] = span                  # elevation
    model.hfield_data[:] = ((Z - zmin) / span).astype(np.float32).ravel()
    return -zmin                                        # 지면 0 을 맞추는 오프셋


class Curriculum:
    """성공률이 임계를 넘으면 난이도를 올린다."""

    def __init__(self, start=0.0, step=0.05, thresh=0.75, window=40):
        self.d = start; self.step = step; self.thresh = thresh
        self.window = window; self.hist = []

    def report(self, success: bool):
        self.hist.append(bool(success))
        if len(self.hist) >= self.window:
            if np.mean(self.hist) >= self.thresh:
                self.d = min(1.0, self.d + self.step)
            self.hist.clear()
        return self.d


# ---------------------------------------------------------------- 주행 가능 영역
# 스폰과 목표를 맵 중앙/전역에서 그냥 뽑으면 **절벽 위에 스폰**한다.  실제 규사
# STL 의 중앙은 87mm -> -144mm 로 한 셀에 231mm 꺾이는 단차라, 로버가 리셋 직후
# 그대로 전복했다 (전복률 90%).  nav2 도 그런 곳으로 경로를 내지 않는다.
def export_costmap(kind, out_dir, max_slope_deg=None, inflate_m=0.0):
    """주행가능 마스크를 **Nav2 정적 맵**(PGM + YAML)으로 내보낸다.

    `drivable_mask` 는 경사·단차의 통과 가능성만 판정한다. 실제 로버의 외접
    반경은 Nav2 costmap 의 footprint/inflation 에서 별도로 반영한다.

    값 규약 (nav2_map_server, `trinary` 모드):
        0   = 점유(lethal)      PGM 픽셀 0
        254 = 자유              PGM 픽셀 254
    PGM 은 위에서 아래로 쓰므로 y 축을 뒤집는다 (origin 이 좌하단).
    """
    from scipy import ndimage
    import pathlib as _pl
    Z = load_arena(kind)
    ex, ey = eval_extent(kind)
    if max_slope_deg is None:
        from config import RoverCfg
        max_slope_deg = RoverCfg().max_slope_deg
    drive, _ = drivable_mask(Z, ex, ey, max_slope_deg=max_slope_deg)
    ny, nx = Z.shape
    # OccupancyGrid 는 x/y 에 해상도 하나만 쓴다. 원본 hfield 는 정사각 배열이지만
    # 실제 경기장은 직사각형이므로 y 를 같은 물리 해상도로 재표본화한다.
    res = ex / nx
    occ = np.where(drive, 254, 0).astype(np.uint8)
    if inflate_m > 0:                               # 추가 여유 (Nav2 inflation 과 별개)
        k = int(round(inflate_m / res))
        if k > 0:
            occ = np.where(ndimage.binary_erosion(drive, iterations=k), 254, 0
                           ).astype(np.uint8)
    ny_out = int(round(ey / res))
    rows = np.clip(((np.arange(ny_out) + 0.5) * ny / ny_out).astype(int), 0, ny - 1)
    occ = occ[rows[::-1], :]                         # PGM 은 위에서 아래
    d = _pl.Path(out_dir); d.mkdir(parents=True, exist_ok=True)
    pgm = d / f"arena_{kind}.pgm"
    with open(pgm, "wb") as f:
        f.write(b"P5\n# KSRC drivable mask (max_slope %.1f deg)\n%d %d\n255\n"
                % (max_slope_deg, nx, ny_out))
        f.write(occ.tobytes())
    yml = d / f"arena_{kind}.yaml"
    yml.write_text(
        "image: %s\n"
        "mode: trinary\n"
        "resolution: %.6f\n"
        "origin: [%.6f, %.6f, 0.0]\n"
        "negate: 0\n"
        "occupied_thresh: 0.65\n"
        "free_thresh: 0.196\n" % (pgm.name, res, -ex / 2, -ey / 2))
    return str(yml), dict(nx=nx, ny=ny_out, res=res, free=float(np.mean(occ == 254)),
                          ex=ex, ey=ey)


def perturb(Z, ex, ey, rng, amp=0.025, lam=0.20):
    """대회맵 위에 **미세 요철**만 얹는다 (거시 구조는 건드리지 않는다).

    왜: 학습을 대회맵 하나에 고정하면 그 한 장의 요철까지 외운다.  그런데
    모래는 움직인다 -- 스캔과 대회 당일 사이에 다른 팀이 밟고, 갈퀴질하고,
    습도가 변한다.  변하지 않는 것은 램프/평지 배치(건축물)이고 변하는 것은
    바퀴자국 규모(수 cm)이므로, 그 스케일만 흔든다.

    amp 는 진폭 [m], lam 은 상관길이 [m].  lam 은 로버 풋프린트(0.16m)와
    같은 규모로 둔다 -- 그보다 짧으면 바퀴가 타고 넘어가 자세에 영향이 없고,
    그보다 길면 거시 구조를 바꿔버린다.
    """
    from scipy import ndimage
    if amp <= 0:
        return Z
    ny, nx = Z.shape
    sx = max(lam / (ex / nx), 1.0)
    sy = max(lam / (ey / ny), 1.0)
    n = ndimage.gaussian_filter(rng.standard_normal((ny, nx)), (sy, sx))
    sd = float(n.std())
    if sd < 1e-9:
        return Z
    return Z + n * (amp / sd)


def soil_field(ex, ey, rng, n=128, amp=0.0, lam=0.20):
    """**바퀴별 흙 차이**를 만드는 mu 배율 장(場).  평균 1.0.

    왜 필요한가: `terramech` 의 흙 파라미터가 에피소드 내내 4바퀴 공통이면
    **좌우 비대칭 슬립이 아예 발생하지 않는다.**  그런데 이 RL 의 목적이
    "한쪽 바퀴 슬립으로 yaw 가 틀어지면 되잡기" 다 -- 교란 기구가 없는 세계에서
    학습시키면 다른 과제를 배운 것이 된다.  로버스트성 추가가 아니라 과제 본체다.

    `lam` (상관길이) 은 트랙폭(265mm) 보다 조금 작게 둔다.  그래야 좌우 바퀴가
    서로 다른 값을 밟고, 주행 중 패치를 넘어갈 때 **과도 외란**이 생긴다.
    `amp` 는 상대 표준편차이고 난이도에 비례시킨다 (d=0 이면 균일).
    """
    from scipy import ndimage
    if amp <= 0:
        return None
    sx = max(lam / (ex / n), 1.0)
    sy = max(lam / (ey / n), 1.0)
    f = ndimage.gaussian_filter(rng.standard_normal((n, n)), (sy, sx))
    sd = float(f.std())
    if sd < 1e-9:
        return None
    return 1.0 + f * (amp / sd)


def sample_field(F, ex, ey, xy):
    """장(場)을 월드좌표 xy (N,2) 에서 샘플링.  범위를 벗어나면 가장자리 값."""
    if F is None:
        return np.ones(len(xy))
    n = F.shape[0]
    ix = np.clip(((xy[:, 0] + ex / 2) / ex * (n - 1)).astype(int), 0, n - 1)
    iy = np.clip(((xy[:, 1] + ey / 2) / ey * (n - 1)).astype(int), 0, n - 1)
    return F[iy, ix]


def drivable_mask(Z, ex, ey, max_slope_deg=35.0, max_step=0.04, foot=0.16):
    """주행 가능 셀 마스크와 **풋프린트 규모 경사**를 돌려준다.

    두 번 틀렸던 부분이라 근거를 남긴다:
      1) 처음엔 "풋프린트 안 높이차 <= 50mm" 로 잡았다.  그러면 30도 램프도
         (0.16*tan30 = 92mm) 통째로 배제된다.  램프는 넘어야 하는 대상이므로,
         국소 경사로 설명되는 높이차는 빼고 **단차만** 본다.
      2) 그 다음엔 경사에 maximum_filter 를 씌웠다.  그런데 STL 을 10mm 격자로
         래스터화하면 매끈한 29도 사면에도 ~6mm 양자화 계단이 생겨 국소 경사가
         22셀마다 한 번씩 40도로 튄다.  그 한 셀이 19x19 이웃을 전부 배제해서
         30도 램프 전체가 고립된 섬이 됐다 (평지->급사면 직선 175쌍 중 통과
         가능 0쌍, 로버가 경험한 최대 기울기 13도).
    로버 자세를 정하는 것은 한 셀이 아니라 풋프린트 규모의 평면이므로, 경사는
    풋프린트로 평활한 표면에서 재고, 그 평면을 넘는 요철만 단차로 센다.
    """
    from scipy import ndimage
    ny, nx = Z.shape
    dx, dy = ex / nx, ey / ny
    kx = max(int(round(foot / dx)) | 1, 3)
    ky = max(int(round(foot / dy)) | 1, 3)
    Zs = ndimage.uniform_filter(Z, size=(ky, kx))        # 풋프린트 규모 표면
    gy, gx = np.gradient(Zs, dy, dx)
    slope = np.arctan(np.hypot(gx, gy))                  # 로버가 실제로 겪는 경사
    relief = (ndimage.maximum_filter(Z, size=(ky, kx)) -
              ndimage.minimum_filter(Z, size=(ky, kx)))
    step = relief - foot * np.tan(slope)                 # 평면으로 설명 안 되는 몫
    ok = (slope <= np.radians(max_slope_deg)) & (step <= max_step)
    ok[:ky, :] = ok[-ky:, :] = ok[:, :kx] = ok[:, -kx:] = False   # 경계 여유
    return ok, np.degrees(slope)


def center_clearance_mask(drivable, ex, ey, radius):
    """지형 통과가능 셀로부터 로버 중심이 `radius` 이상 떨어진 경로 마스크.

    `drivable_mask` 의 0.16m 창은 경사·단차를 측정하는 공간 척도이지 로버의
    실제 충돌 반경이 아니다. 경로 생성에는 별도 중심 이격을 적용한다.
    Nav2 에는 원본 지형 마스크를 주고 costmap 의 robot_radius 로 이격한다.
    """
    if radius < 0 or not np.isfinite(radius):
        raise ValueError("radius 는 0 이상의 유한한 값이어야 합니다")
    mask = np.asarray(drivable, dtype=bool)
    if radius == 0:
        return mask.copy()
    from scipy.ndimage import distance_transform_edt
    return mask & (distance_transform_edt(mask, sampling=(ey / mask.shape[0],
                                                          ex / mask.shape[1])) >= radius)


def flat_spawn_mask(drivable, slope_deg, ex, ey, max_slope_deg, clearance_m):
    """주행 가능 구역 중 로버 풋프린트 전체가 평탄한 스폰 중심만 남긴다."""
    from scipy.ndimage import distance_transform_edt

    flat = drivable & (slope_deg <= max_slope_deg)
    ny, nx = flat.shape
    distance = distance_transform_edt(flat, sampling=(ey / ny, ex / nx))
    return flat & (distance >= clearance_m)


def cell_centers(Z, ex, ey):
    """셀 인덱스 -> 월드 좌표 (맵은 원점 중심)."""
    ny, nx = Z.shape
    xs = (np.arange(nx) + 0.5) / nx * ex - ex / 2
    ys = (np.arange(ny) + 0.5) / ny * ey - ey / 2
    return xs, ys


def sample_drivable(mask, Z, ex, ey, rng, k=1, weight=None):
    """주행 가능 셀에서 월드 좌표를 k개 뽑는다.  없으면 원점을 돌려준다.

    weight 를 주면 그 값에 비례해 뽑는다.  30도 사면은 맵 면적의 1% 라, 균등
    샘플로는 후보 64개 중 1개도 안 걸리고 로버가 램프를 영영 안 올라간다
    (실측: 경험한 최대 기울기 평균 13.1도, >20도 시간 0.4%).
    """
    xs, ys = cell_centers(Z, ex, ey)
    iy, ix = np.nonzero(mask)
    if len(iy) == 0:
        return np.zeros((k, 2))
    if weight is None:
        sel = rng.integers(0, len(iy), size=k)
    else:
        w = np.maximum(weight[iy, ix], 0.0) + 1e-9
        sel = rng.choice(len(iy), size=k, p=w / w.sum())
    return np.stack([xs[ix[sel]], ys[iy[sel]]], axis=1)


def local_top(Z, ex, ey, xy, foot=0.16):
    """(x,y) 주변 풋프린트 안의 최대 높이.  스폰 시 지형에 파묻히지 않게."""
    ny, nx = Z.shape
    xs, ys = cell_centers(Z, ex, ey)
    j = int(np.clip(np.searchsorted(xs, xy[0]), 0, nx - 1))
    i = int(np.clip(np.searchsorted(ys, xy[1]), 0, ny - 1))
    kx = max(int(round(foot / (ex / nx))), 1)
    ky = max(int(round(foot / (ey / ny))), 1)
    return float(Z[max(i - ky, 0):i + ky + 1, max(j - kx, 0):j + kx + 1].max())


def line_clear(mask, ex, ey, p0, p1, step=0.04):
    """p0->p1 직선이 전부 주행 가능한가.

    목표만 주행 가능 셀에서 뽑아도 그 사이 직선이 절벽을 지나면 로버가 그대로
    떨어진다.  nav2 는 웨이포인트 사이를 통과 가능하게 보장하므로, 명령 생성도
    같은 조건을 지켜야 지형 난이도와 목표 운이 섞이지 않는다.
    """
    ny, nx = mask.shape
    d = np.linalg.norm(np.asarray(p1) - np.asarray(p0))
    # 4cm 간격은 규사맵의 8.3mm 셀 여러 개를 건너뛴다. 최소 셀 폭의
    # 절반 이하로 샘플링해 가는 장애물을 지나치지 않도록 한다.
    step = min(step, 0.5 * min(ex / nx, ey / ny))
    n = max(int(np.ceil(d / step)) + 1, 2)
    t = np.linspace(0.0, 1.0, n)[:, None]
    pts = np.asarray(p0)[None, :] * (1 - t) + np.asarray(p1)[None, :] * t
    jx = np.clip(((pts[:, 0] + ex / 2) / ex * nx).astype(int), 0, nx - 1)
    iy = np.clip(((pts[:, 1] + ey / 2) / ey * ny).astype(int), 0, ny - 1)
    return bool(mask[iy, jx].all())


def slope_map(Z, ex, ey):
    """셀별 경사 [deg]."""
    ny, nx = Z.shape
    gy, gx = np.gradient(Z, ey / ny, ex / nx)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def line_mean(field, ex, ey, p0, p1, step=0.04):
    """p0->p1 직선 위 field 의 평균값.  목표의 '험함' 을 재는 데 쓴다."""
    ny, nx = field.shape
    d = np.linalg.norm(np.asarray(p1) - np.asarray(p0))
    n = max(int(d / step), 2)
    t = np.linspace(0.0, 1.0, n)[:, None]
    pts = np.asarray(p0)[None, :] * (1 - t) + np.asarray(p1)[None, :] * t
    jx = np.clip(((pts[:, 0] + ex / 2) / ex * nx).astype(int), 0, nx - 1)
    iy = np.clip(((pts[:, 1] + ey / 2) / ey * ny).astype(int), 0, ny - 1)
    return float(field[iy, jx].mean())
