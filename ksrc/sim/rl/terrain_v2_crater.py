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
ARENA_X, ARENA_Y = 12.0, 12.0   # 학습용 -- 경계를 아예 만나지 않게 넉넉히
EVAL_X, EVAL_Y = 4.0, 5.0       # 실제 경기장 STL 범위
HF_N = 384                      # 격자 (학습 31mm / 평가 10~13mm 해상도)


@dataclass
class TerrainSpec:
    """난이도 0~1 이 각 항목을 아래 범위로 보간한다."""
    # v2: 실제 대회장 STL 의 **공간 구조**에 맞춘 값.
    #
    # v1 은 경사 히스토그램만 맞추려고 분화구를 작고(R 0.16~0.38m) 촘촘하게
    # (3.7개/m^2) 뿌렸다. 통계는 맞았지만 겹친 림들이 솟아올라 "그릇" 이 아니라
    # "자갈밭" 이 됐다. 실제 경기장의 특징 크기는 698mm 인데 v1 은 1.5~6.0m 로
    # 2~8배 컸고(대부분 맵 절반을 덮는 램프 탓), 주행 경험이 전혀 다르다:
    #   작고 촘촘한 혹  -> 계속 타넘음, 배 접촉은 잦지만 얕고 탈출 쉬움
    #   크고 완만한 그릇 -> 한번 들어가면 나와야 함, 갇히면 오래 갇힘
    #
    # v2 는 분화구를 **적고 크게**(실제 육안 추정 약 1.5개/m^2, R 0.25~0.80m)
    # 바꾸고, 램프를 반평면이 아니라 국소 고원으로 제한한다.
    ramp_deg:    tuple = (0.0, 32.0)     # 국소 고원의 사면 경사각
    ramp_h:      tuple = (0.0, 0.25)     # 고원 높이 [m]
    ramp_r:      tuple = (0.6, 1.6)      # 고원 반경 [m] -- 맵 절반을 덮지 않게
    rough_amp:   tuple = (0.002, 0.030)  # 잔거칠기 진폭 [m]
    crater_den:  tuple = (0.0, 1.5)      # 분화구 **밀도** [개/m^2]
    crater_dep:  tuple = (0.02, 0.17)    # 분화구 깊이 [m]
    crater_rad:  tuple = (0.25, 0.80)    # 분화구 반경 [m]  (림 경사 = 2D/R)
    step_h:      tuple = (0.0, 0.10)     # 단차 높이 [m]

    def at(self, d: float) -> dict:
        d = float(np.clip(d, 0.0, 1.0))
        lerp = lambda t: t[0] + d * (t[1] - t[0])
        return dict(ramp_a=np.radians(lerp(self.ramp_deg)),
                    ramp_h=lerp(self.ramp_h), ramp_r=lerp(self.ramp_r),
                    rough=lerp(self.rough_amp),
                    crater_den=lerp(self.crater_den),
                    dep=lerp(self.crater_dep),
                    rad=lerp(self.crater_rad),
                    step=lerp(self.step_h))


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


def generate(difficulty: float, rng, spec: TerrainSpec = TerrainSpec(), n: int = HF_N):
    """난이도에 맞는 높이맵(미터)과 메타데이터를 만든다."""
    p = spec.at(difficulty)
    ys = np.linspace(-ARENA_Y / 2, ARENA_Y / 2, n)
    xs = np.linspace(-ARENA_X / 2, ARENA_X / 2, n)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros((n, n))

    # 1) 경사구조물 -- **국소 고원**. 실제 경기장 가운데의 융기부에 해당한다.
    #    v1 은 반평면 램프라 맵 절반을 덮어버려 특징 크기를 왜곡했다.
    if p["ramp_h"] > 1e-3 and p["ramp_a"] > 1e-3:
        cx = rng.uniform(-ARENA_X / 4, ARENA_X / 4)
        cy = rng.uniform(-ARENA_Y / 4, ARENA_Y / 4)
        R0 = p["ramp_r"]                                   # 평탄한 정상부 반경
        L = p["ramp_h"] / max(np.tan(p["ramp_a"]), 1e-3)   # 사면 수평 길이
        r = np.hypot(X - cx, Y - cy)
        Z += p["ramp_h"] * np.clip((R0 + L - r) / max(L, 1e-3), 0.0, 1.0)

    # 2) 잔거칠기
    Z += p["rough"] * _value_noise(n, rng)

    # 3) 분화구 -- 실제 경기장이 이 모양이다. 바닥은 패이고 림은 솟는다.
    n_cr = int(round(p["crater_den"] * ARENA_X * ARENA_Y))
    for _ in range(n_cr):
        cx, cy = rng.uniform(-ARENA_X / 2, ARENA_X / 2), rng.uniform(-ARENA_Y / 2, ARENA_Y / 2)
        R = p["rad"] * rng.uniform(0.6, 1.5)
        D = p["dep"] * rng.uniform(0.4, 1.2)
        r = np.hypot(X - cx, Y - cy) / max(R, 1e-3)
        bowl = -D * (1.0 - r ** 2) * (r < 1.0)                   # 그릇
        rim = 0.35 * D * np.exp(-((r - 1.0) / 0.25) ** 2)        # 림
        Z += bowl + rim

    # 4) 단차 (인공 암반) -- 사각 융기 한두 개
    n_step = int(round(rng.integers(0, 3) * ARENA_X * ARENA_Y / 15.0)) if p["step"] > 1e-3 else 0
    for _ in range(n_step):
        cx, cy = rng.uniform(-ARENA_X / 2, ARENA_X / 2), rng.uniform(-ARENA_Y / 2, ARENA_Y / 2)
        w, l = rng.uniform(0.15, 0.5), rng.uniform(0.15, 0.5)
        Z += p["step"] * rng.uniform(0.5, 1.0) * ((np.abs(X - cx) < w) & (np.abs(Y - cy) < l))

    # 출발 지점은 평평하게 -- 리셋 직후 전복/고착을 피한다
    flat = np.hypot(X, Y) < 0.35
    Z -= Z[flat].mean() if flat.any() else 0.0
    Z = np.where(flat, Z * (np.hypot(X, Y) / 0.35) ** 2, Z)
    return Z, p


def load_arena(path="assets/arena_hfield.npz", n: int = HF_N):
    """실제 경기장 STL 에서 뽑은 높이맵 (검증 전용)."""
    z = np.load(path)
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
