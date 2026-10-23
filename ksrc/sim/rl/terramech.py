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

import terrain as _terr
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
        self.wdof = np.asarray(self.wdof)
        self.wg = np.asarray(self.wg)
        self.wb = np.asarray(self.wb)
        self.sb = np.asarray(self.sb)
        self._g2k = {int(g): k for k, g in enumerate(self.wg)}
        self.z = np.zeros(4)            # 침하 [m]
        self.mu_field = None            # 바퀴별 mu 배율 장 (env 가 넣어준다)
        self.f_ex = self.f_ey = 1.0
        self.mu_w = np.full(4, self.p.mu_max)
        self.slip = np.zeros(4)
        self.Fn = np.zeros(4)
        self.R = np.zeros(4)

    # ---- 내부 ------------------------------------------------------------
    def _wheel_states(self, d):
        """바퀴 4개를 한 번에: (전진방향 (4,3), 대지 종속도 (4,), 원주속도 (4,)).

        바퀴마다 따로 계산하면 numpy 호출 오버헤드를 4번 낸다.  np.exp 같은 함수는
        호출당 ~0.4us 가 크기와 무관하게 들고 원소 4개의 실제 계산은 ~4ns 라,
        시간의 99% 가 파이썬<->C 경계를 넘는 비용이다.  배열로 묶으면 그 비용을
        한 번만 낸다 (연산이 빨라지는 게 아니라 호출이 줄어드는 것).
        """
        R = d.xmat[self.sb].reshape(4, 3, 3)      # 조향 바디의 회전행렬 4개
        fwd = R[:, :, 0].copy()                   # 각 행렬의 x축
        fwd[:, 2] = 0.0
        n = np.linalg.norm(fwd, axis=1, keepdims=True)
        fwd = np.where(n > 1e-9, fwd / np.maximum(n, 1e-12),
                       np.array([1.0, 0.0, 0.0]))
        # mj_objectVelocity 는 배치 API 가 없어 4번 부른다 (C 호출이라 상대적으로 쌈)
        v = np.empty(4)
        vel = np.zeros(6)
        for k in range(4):
            mujoco.mj_objectVelocity(self.m, d, mujoco.mjtObj.mjOBJ_BODY,
                                     self.wb[k], vel, 0)
            v[k] = vel[3:6] @ fwd[k]
        rw = self.r * d.qvel[self.wdof]
        return fwd, v, rw

    def _normal_forces(self, d):
        # wg 가 ndarray 가 된 뒤로 list.index 를 못 쓴다. geom id -> 0..3 사전을 쓴다.
        Fn = np.zeros(4); f6 = np.zeros(6)
        g2k = self._g2k
        for i in range(d.ncon):
            c = d.contact[i]
            for g in (c.geom1, c.geom2):
                k = g2k.get(int(g))
                if k is not None:
                    mujoco.mj_contactForce(self.m, d, i, f6)
                    Fn[k] += abs(f6[0])
        return Fn

    # ---- 공개 ------------------------------------------------------------
    def apply(self, d, dt):
        p = self.p
        self.Fn = self._normal_forces(d)
        d.xfrc_applied[:] = 0.0

        fwd, v, rw = self._wheel_states(d)
        a_rw, a_v = np.abs(rw), np.abs(v)
        den = np.maximum(np.maximum(a_rw, a_v), 1e-3)
        i = np.clip((rw - v) / den, -1.0, 1.0)
        self.slip = i

        # 침하 적분: 슬립이 파고, 시간이 회복.  여기 max 에는 1e-3 하한이 없다
        # (den 과 다른 양이다 -- 섞으면 정지 상태에서 침하가 생긴다).
        # **접촉한 바퀴만 판다.**  예전엔 Fn 게이트가 없어서 공중에 뜬 바퀴도
        # 회전하면 침하가 쌓였고, 그 상태가 착지 후에 저항으로 작용했다.
        # 배걸림으로 바퀴가 뜨는 시간이 44% 였으니 무시할 수 없는 양이다.
        # 회복(-beta*z)은 접촉과 무관하게 진행시킨다 (흙이 되메워지는 항).
        on = (self.Fn > 1e-6).astype(float)
        zdot = on * p.alpha * np.abs(i) * np.maximum(a_rw, a_v) - p.beta * self.z
        self.z = np.clip(self.z + dt * zdot, 0.0, p.z_max)

        # (1) 전단 곡선을 마찰계수로 -- 미끄러지는 동안 F = mu(i)*N 이 된다
        # **바퀴별 mu 배율.**  흙 장(場) 을 각 바퀴 접지 위치에서 샘플링한다.
        # 이게 없으면 4바퀴가 같은 흙을 밟아 좌우 비대칭 슬립이 발생하지 않는다
        # (terrain.soil_field 주석 참고).  이 RL 의 목적이 "한쪽 바퀴 슬립으로
        # yaw 가 틀어지면 되잡기" 이므로, 교란 기구가 없으면 과제 자체가 없다.
        if self.mu_field is not None:
            k = _terr.sample_field(self.mu_field, self.f_ex, self.f_ey,
                                   d.xpos[self.wb, :2])
            mu_w = np.clip(p.mu_max * k, 0.15, 1.20)
        else:
            mu_w = np.full(4, p.mu_max)
        self.mu_w = mu_w
        mu = mu_w * (1.0 - np.exp(-np.abs(i) / p.K))
        mu = np.maximum(mu, p.mu_lat * mu_w)       # 횡방향 접지력 하한 (위 설명)
        self.m.geom_friction[self.wg, 0] = mu

        # (2) 주행저항(압밀·불도징): **바퀴 CoM 에 병진력만** 인가한다.
        #
        # 예전에는 "접촉점에 작용하는 힘" 으로 만들려고 CoM 인가분에 모멘트
        # cross(arm, F) 를 더했다.  그런데 그 모멘트가 **접촉이 만드는 차축 부하를
        # 거의 정확히 상쇄**해서 구동 모터가 무부하로 돌았다 (duty 0.558 에서
        # 순토크 -0.000009 N*m, 무부하 평형 4.44 rad/s 와 일치).
        #
        # 관례와 무관한 판정은 **에너지 수지**다.  xfrc_applied 가 넣는 총 일률을
        # 재보면 평지에서:
        #     모멘트 포함  +0.2406 W   <- 저항 모델이 에너지를 **공급**한다
        #     병진만       -0.1280 W   <- 소산 (정상)
        # 평지 정속의 해석적 형태도 같은 얘기다:  P = -R*v + R*r*omega = R(r*omega - v)
        # 로 굴림에서는 0, 구동 슬립에서는 양수가 된다.
        #
        # 병진력만 주면 이중계상이 없다.  MuJoCo 의 xfrc_applied 는 body CoM
        # (xipos) 에 작용하고 힘을 접촉점으로 옮기지 않으므로, 차축 부하는
        # **지면 접촉이 한 번만** 만든다 (R*r = 0.2800 x 0.03995 = 0.01119 N*m,
        # 실측 모터 부하 0.01118 N*m).
        #
        # 회전 저항(베어링·기어 손실)이 별도로 필요하면 실측 근거를 갖춘 별도 항으로
        # 넣어야 한다.  같은 R 을 힘과 제동토크로 둘 다 주면 중복이다.
        Rn = self.Fn * (p.c_r + p.k_z * self.z)
        self.R = Rn
        act = (Rn > 1e-9) & (a_v > 1e-4)
        if act.any():
            F = -(Rn * np.sign(v) * act)[:, None] * fwd
            d.xfrc_applied[self.wb, :3] += F

    @property
    def drawbar(self):
        """정규화 견인력 DP/N (진단용)."""
        mu = np.maximum(self.p.mu_max * (1.0 - np.exp(-np.abs(self.slip) / self.p.K)),
                        self.p.mu_lat * self.p.mu_max)
        return mu - (self.p.c_r + self.p.k_z * self.z)
