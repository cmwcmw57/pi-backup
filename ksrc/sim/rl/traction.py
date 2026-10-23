"""고전 트랙션 제어 (RL 의 비교 기준선).

**엔코더만 사용한다.** 실기체에는 힘센서도 전류센서도 없고, IMU 적분으로는
차체 속도를 못 얻는다(자세 오차 1도 -> 1초에 0.17 m/s, 최고속도가 0.32 m/s
인데). 그래서 슬립비 i = (rw - v)/rw 를 직접 계산하는 고전 트랙션 제어는
애초에 구현이 불가능하다.

우회로: **바퀴 각가속도**만 본다.
    차체가 낼 수 있는 최대 가속도는 지면 견인력이 정한다:  a_max = mu*g
    바퀴 원주 가속도가 그보다 빠르면 접지를 잃은 것 외에 설명이 없다.
        dw/dt * r  >  mu_expect * g     ->  슬립 중
자동차 TCS 와 같은 원리이고, 엔코더만으로 성립한다.

측정된 물리(30도 규사, 풀스로틀 대비 throttle 0.4 가 2.4배 빠름)를 근거로,
슬립이 감지되면 duty 를 빠르게 낮추고 안정되면 천천히 복구한다.
"""
import numpy as np

WHEELS = ("fl", "rl", "fr", "rr")


class SlopeScheduledLimit:
    """경사각으로 duty 상한을 거는 개루프 제어.

    실측 근거: 규사에서 15~25도는 풀스로틀이 최적이지만 30도에서는 throttle 0.4
    가 1.0 보다 2.4배 빠르다 (견인력 피크를 넘기 때문). 그래서 경사가 임계를
    넘으면 duty 상한을 낮춘다.

    왜 개루프인가 -- 폐루프로 할 신호가 없다:
      * 슬립비 i=(rw-v)/rw 는 차체속도 v 가 필요한데 실기체에서 측정 불가
        (IMU 적분은 자세 1도 오차에 1초 0.17 m/s, 최고속도가 0.32 m/s)
      * 바퀴속도로 부하를 보려 해도 정상주행 96% vs 고착 94% 로 2%p 차이뿐.
        모터가 지면 전달 한계보다 과토크라 슬립해도 속도가 거의 안 변한다
      * 각가속도(자동차 TCS 원리)는 과도현상만 잡고 정상상태 슬립엔 무력
      * 바퀴 간 합의 잔차는 **균일 슬립**(경사 등판)에서 0 이 된다
    따라서 관측 가능한 건 IMU 자세뿐이고, 이게 고전 제어의 현실적 상한이다.
    """

    def __init__(self, lo_deg=24.0, hi_deg=30.0, duty_hi=1.0, duty_lo=0.42,
                 tau=0.25, ctrl_hz=50.0):
        self.lo = np.radians(lo_deg); self.hi = np.radians(hi_deg)
        self.d_hi = duty_hi; self.d_lo = duty_lo
        self.alpha = (1.0 / ctrl_hz) / max(tau, 1e-3)
        self.reset()

    def reset(self):
        self.lim = self.d_hi

    def __call__(self, duty_cmd, pitch_rad):
        """duty_cmd: IK duty 4개.  pitch_rad: 차체 피치(오르막이 +)."""
        t = np.clip((abs(pitch_rad) - self.lo) / max(self.hi - self.lo, 1e-6), 0.0, 1.0)
        target = self.d_hi + t * (self.d_lo - self.d_hi)
        self.lim += self.alpha * (target - self.lim)        # 급변 방지
        return np.clip(duty_cmd, -self.lim, self.lim)


class TractionControl:
    def __init__(self, wheel_r, ctrl_hz,
                 mu_expect=0.55,      # 기대 견인계수. 보수적으로 잡을수록 빨리 개입
                 cut=0.22,            # 슬립 감지 시 duty 감소량 (스텝당)
                 recover=0.035,       # 안정 시 복구량 (스텝당). 감소보다 훨씬 느리게
                 floor=0.30,          # duty 하한. 너무 낮추면 아예 못 감
                 tau=0.12):           # 각가속도 저역통과 시정수 [s]
        self.r = wheel_r; self.dt = 1.0 / ctrl_hz
        self.a_thr = mu_expect * 9.81 / wheel_r      # rad/s^2 임계
        self.cut = cut; self.recover = recover; self.floor = floor
        self.alpha = self.dt / max(tau, 1e-3)
        self.reset()

    def reset(self):
        self.w_prev = np.zeros(4)
        self.acc = np.zeros(4)          # 저역통과된 각가속도
        self.scale = np.ones(4)         # 바퀴별 duty 배율 [floor, 1]
        self.n_cut = 0

    def __call__(self, duty_cmd, w_meas):
        """duty_cmd: IK 가 낸 [-1,1] duty 4개.  w_meas: 바퀴 각속도 4개."""
        a = (w_meas - self.w_prev) / self.dt
        self.w_prev = np.asarray(w_meas, float).copy()
        self.acc += self.alpha * (a - self.acc)          # 노이즈 억제

        # 구동 방향으로 가속 중인 바퀴만 본다 (감속/제동은 슬립이 아니다)
        spinning = (np.sign(self.acc) == np.sign(duty_cmd)) & \
                   (np.abs(self.acc) > self.a_thr)
        self.scale = np.where(spinning, self.scale - self.cut,
                              self.scale + self.recover)
        self.scale = np.clip(self.scale, self.floor, 1.0)
        self.n_cut += int(spinning.any())
        return np.clip(duty_cmd * self.scale, -1.0, 1.0)
