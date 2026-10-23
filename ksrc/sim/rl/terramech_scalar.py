"""휠-토양 상호작용 (슬립-싱키지).

MuJoCo 의 쿨롱 마찰은 "붙거나 미끄러지거나"뿐이라, 미끄러지면 견인력이 mu*N
으로 고정된다.  실제 모래는 견인력이 슬립비에 대해 **피크를 갖고 다시 떨어진다**:

    슬립 ↑ → 침하 ↑ → 주행저항 ↑ → 더 미끄러짐 → ...      (양의 되먹임)

이 되먹임이 곧 "고착"이고, 쿨롱 마찰에는 그 항이 없어서 시뮬 로버는 절대
자기 구덩이를 파지 못한다.  학습으로 막으려는 현상을 정책이 한 번도 못 보게 된다.

구현 방식 -- MuJoCo 접촉과 싸우지 않고 두 갈래로 얹는다:

  (1) 전단 곡선은 **마찰계수를 매 스텝 갱신**해서 준다 (Janosi-Hanamoto).
          mu(i) = mu_max * (1 - exp(-|i| / K))
      미끄러지는 동안 접촉력이 정확히 mu(i)*N 이 되므로 곡선이 그대로 나온다.
      가로방향 저항도 자동으로 따라와서 별도 처리가 필요 없다.

  (2) 주행저항(압밀/불도징)은 접촉점에 **수평력**으로 인가한다.
          R = N * (c_r + k_z * z)
      침하 z 는 파이썬 쪽 상태로 적분한다:
          z' = alpha*|i|*|v_wheel| - beta*z        (슬립이 파고, 시간이 회복)

  DP = mu(i)*N - R  이 슬립에 대해 피크를 갖는 이유: mu 는 i~3K 에서 포화하는데
  R 은 z 를 통해 계속 커지기 때문.  피크 위치 i* 와 높이가 곧 실측으로 맞출 값이다.
"""
from dataclasses import dataclass
import numpy as np
import mujoco

WHEELS = ("fl", "rl", "fr", "rr")


@dataclass
class TerrainParams:
    mu_max: float          # 최대 전단 마찰계수
    K:      float          # 전단변형 계수 (작을수록 빨리 포화)
    alpha:  float          # 슬립-싱키지 계수 (암반=0)
    beta:   float          # 침하 회복률 [1/s]
    c_r:    float          # 기본 구름저항 (N 대비)
    k_z:    float          # 침하 1m 당 저항 증가 (N 대비)
    z_max:  float = 0.050  # 침하 상한 = **모래 두께**. 구조물 위에 규사가
                           # 깔린 형태라 침하가 무한정 진행되지 않는다
                           # (제안서: 미고결 모래 깊이 50mm).
    # 횡방향 접지력 하한 (mu_max 대비 비율).
    # MuJoCo 마찰은 접선면에서 등방성이라, 전단곡선 mu(i) 를 그대로 쓰면
    # 종슬립이 0 일 때 **횡방향 접지력까지 0** 이 되어 로버가 옆으로 자유낙하한다
    # (10도 횡경사에서 8초에 70m 미끄러지는 게 실측됨).
    # 물리적으로 종방향으로 안 미끄러지는 바퀴도 옆으로는 버티므로, 마찰계수에
    # 하한을 둔다. 이 하한은 저슬립 구간만 끌어올리고 견인력 피크(i*~0.4)와
    # 그 이후 하강은 건드리지 않는다 -- 고착 재현은 침하 저항 R 이 담당하므로.
    mu_lat: float = 0.60


# 10/6~8 연습장 실측으로 교체할 값들. 지금은 문헌 수준의 대표값.
PRESETS = {
    # 인공 암반: 강체. 침하 없음 -> DP 가 단조증가 (피크 없음). 쿨롱과 유사.
    "rock":  TerrainParams(mu_max=0.90, K=0.08, alpha=0.00, beta=1.0, c_r=0.02, k_z=0.0),
    # 다져진 모래
    "firm":  TerrainParams(mu_max=0.70, K=0.12, alpha=0.030, beta=0.6, c_r=0.03, k_z=6.0),
    # 규사 (대회 지형 가정)
    "sand":  TerrainParams(mu_max=0.55, K=0.15, alpha=0.058, beta=0.5, c_r=0.05, k_z=8.0),
    # 느슨한 모래 / 경사면
    "loose": TerrainParams(mu_max=0.45, K=0.20, alpha=0.090, beta=0.4, c_r=0.08, k_z=11.0),
}

# 규사 전용 랜덤화.
#
# 대회 규사 지형은 **나무 판자 프레임 위에 규사를 얇게 깐** 구조다. 따라서
# 암반/firm 같은 다른 흙을 섞을 이유가 없다 (암석지형은 별도 구역이고 후순위).
#
# 실측으로 확인한 것: 30도 램프에서 최적 스로틀을 가르는 건 **두께가 아니라
# 다짐 상태(mu_max)** 였다.
#     두께 10 -> 55mm   최적 duty 거의 불변 (얇으면 침하가 판자에서 멈춤)
#     mu_max 0.70 -> 0.78   최적 duty 0.5 -> 1.0 으로 급변
# 그리고 어떤 고정 스로틀을 써도 평균 19.4% / 최악 49.2% 손해다.
# 그 격차가 적응 제어(고전이든 RL이든)가 가져갈 수 있는 전부다.
#
# 다짐 상태는 대회 당일에 실제로 변한다: 앞 팀들의 주행, 경사면 모래 흘러내림,
# 이틀간의 습도 변화. 직접 측정할 수단이 없으므로 주행 이력에서 추론해야 한다.
SAND_MU   = (0.60, 0.82)   # 다짐/습도. 난이도가 높을수록 무름 (주된 변수)
SAND_Z    = (0.008, 0.060) # 뿌린 두께 [m]. 난이도와 무관하게 랜덤
JITTER    = 0.20


def sample_terrain(rng, difficulty: float = 0.5) -> TerrainParams:
    """난이도 d: 0 이면 잘 다져진 규사, 1 이면 무른 규사."""
    d = float(np.clip(difficulty, 0.0, 1.0))
    mu = SAND_MU[1] + d * (SAND_MU[0] - SAND_MU[1])      # 쉬움 0.82 -> 어려움 0.60
    mu *= rng.uniform(1 - JITTER * 0.5, 1 + JITTER * 0.5)
    z_max = float(rng.uniform(*SAND_Z))
    base = PRESETS["sand"]
    j = lambda v: v * rng.uniform(1 - JITTER, 1 + JITTER)
    return TerrainParams(
        mu_max=float(np.clip(mu, 0.50, 0.90)),
        K=float(np.clip(j(base.K), 0.08, 0.28)),
        alpha=float(np.clip(j(base.alpha), 0.02, 0.12)),
        beta=float(np.clip(j(base.beta), 0.25, 0.9)),
        c_r=float(np.clip(j(base.c_r), 0.02, 0.10)),
        k_z=float(np.clip(j(base.k_z), 4.0, 13.0)),
        z_max=z_max,
        mu_lat=float(np.clip(j(base.mu_lat), 0.45, 0.78)))


class Terramechanics:
    """mj_step 직전에 apply() 를 호출한다."""

    def __init__(self, model, wheel_radius: float, params: TerrainParams):
        self.m = model
        self.r = wheel_radius
        self.p = params
        self.wb = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"wheel_{w}") for w in WHEELS]
        # 전진방향은 **조향 바디**에서 가져온다. 바퀴 바디 프레임은 바퀴와 함께
        # 회전하므로 (7.96 rad/s) 그 x축에 속도를 투영하면 슬립이 전부 쓰레기가 된다.
        self.sb = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"steer_{w}") for w in WHEELS]
        self.wj = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"wh_{w}") for w in WHEELS]
        self.wdof = [model.jnt_dofadr[j] for j in self.wj]
        # 타이어 geom: 바퀴 바디에는 원통이 둘(타이어, 모터) 있으므로 큰 쪽.
        # 반경값으로 찾으면 wheel_r 랜덤화 + XML 의 %.5f 반올림 때문에 못 찾는다.
        self.wg = []
        for b in self.wb:
            gs = [g for g in range(model.ngeom) if model.geom_bodyid[g] == b
                  and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER]
            if not gs:
                raise RuntimeError(f"wheel body {b} 에 cylinder geom 이 없음")
            self.wg.append(max(gs, key=lambda g: model.geom_size[g][0]))
        # MuJoCo 는 접촉 마찰을 두 geom 의 element-wise max 로 정한다.
        # 지면을 낮춰두지 않으면 바퀴 mu 를 아무리 낮춰도 지면값이 이긴다.
        for g in range(model.ngeom):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
            if nm in ("ground", "bump"):
                model.geom_friction[g, 0] = 1e-3
        self.reset()

    def reset(self):
        self.z = np.zeros(4)            # 침하 [m]
        self.slip = np.zeros(4)
        self.Fn = np.zeros(4)
        self.R = np.zeros(4)

    # ---- 내부 ------------------------------------------------------------
    def _wheel_state(self, d, k):
        """(전진방향 단위벡터, 대지 종속도, 바퀴 원주속도)"""
        R = d.xmat[self.sb[k]].reshape(3, 3)
        fwd = R[:, 0].copy(); fwd[2] = 0.0
        n = np.linalg.norm(fwd)
        fwd = fwd / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
        vel = np.zeros(6)               # [ang(3); lin(3)] 월드
        mujoco.mj_objectVelocity(self.m, d, mujoco.mjtObj.mjOBJ_BODY,
                                 self.wb[k], vel, 0)
        return fwd, float(vel[3:6] @ fwd), float(self.r * d.qvel[self.wdof[k]])

    def _normal_forces(self, d):
        Fn = np.zeros(4); f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            for g in (c.geom1, c.geom2):
                if g in self.wg:
                    mujoco.mj_contactForce(self.m, d, i, f6)
                    Fn[self.wg.index(g)] += abs(f6[0])
        return Fn

    # ---- 공개 ------------------------------------------------------------
    def apply(self, d, dt):
        p = self.p
        self.Fn = self._normal_forces(d)
        d.xfrc_applied[:] = 0.0
        for k in range(4):
            fwd, v, rw = self._wheel_state(d, k)
            den = max(abs(rw), abs(v), 1e-3)
            i = np.clip((rw - v) / den, -1.0, 1.0)
            self.slip[k] = i

            # 침하 적분: 슬립이 파고, 시간이 회복
            zdot = p.alpha * abs(i) * max(abs(rw), abs(v)) - p.beta * self.z[k]
            self.z[k] = float(np.clip(self.z[k] + dt * zdot, 0.0, p.z_max))

            # (1) 전단 곡선을 마찰계수로 -- 미끄러지는 동안 F = mu(i)*N 이 된다
            mu = p.mu_max * (1.0 - np.exp(-abs(i) / p.K))
            mu = max(mu, p.mu_lat * p.mu_max)      # 횡방향 접지력 하한 (위 설명)
            self.m.geom_friction[self.wg[k], 0] = mu

            # (2) 주행저항: 접촉점에 수평력. CoM 인가분의 모멘트를 보정한다.
            Rn = self.Fn[k] * (p.c_r + p.k_z * self.z[k])
            self.R[k] = Rn
            if Rn > 1e-9 and abs(v) > 1e-4:
                F = -Rn * np.sign(v) * fwd
                com = d.xipos[self.wb[k]]
                cp = com - np.array([0.0, 0.0, self.r])     # 접촉점 근사
                d.xfrc_applied[self.wb[k], :3] += F
                d.xfrc_applied[self.wb[k], 3:] += np.cross(cp - com, F)

    @property
    def drawbar(self):
        """정규화 견인력 DP/N (진단용)."""
        mu = np.maximum(self.p.mu_max * (1.0 - np.exp(-np.abs(self.slip) / self.p.K)),
                        self.p.mu_lat * self.p.mu_max)
        return mu - (self.p.c_r + self.p.k_z * self.z)
