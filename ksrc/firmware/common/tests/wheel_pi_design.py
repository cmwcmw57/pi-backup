"""바퀴 속도 PI 설계 시뮬 (firmware/MOTOR_MODELING_PLAN.md §8, common/wheel_pi.c 의 이득 근거).
플랜트 = 2026-10-13 실측 (sysid_logs/20261013_102224_drive).

    python3 firmware/common/tests/wheel_pi_design.py      # 후보별 표 (~1분, numpy)


duty 단위 모델:  tau * dw/dt = k*(duty - load - fric(w)) - w
  k = 0.0097 rad/s/duty (D1 기울기), tau = 0.040 s
  fric: 운동마찰 5 duty + 저속 Stribeck 41*exp(-|w|/0.13)  (램프 정지 duty ~33, D1 40duty -> 0.25-0.35)
  정지마찰: 멈춘 상태에서 |duty-load| < 47 (기동 35-52, 마찰 최대 46 보다 커야) 이면 안 움직임
엔코더 8384 tick/rev, 1 kHz 적분.
"""
import numpy as np

TPR = 8384
K = 0.0097
TAU = 0.040
F_KIN, F_STR, W_STR, F_STATIC = 5.0, 41.0, 0.13, 47.0
PWM_MAX = 799


def fric(w):
    return F_KIN + F_STR * np.exp(-abs(w) / W_STR)


class Plant:
    def __init__(self, kscale=1.0, static=F_STATIC, tau=TAU):
        self.w = 0.0; self.th = 0.0; self.k = K * kscale; self.static = static; self.tau = tau
        self.stuck = True

    def step(self, duty, load, dt):
        net = duty - load
        if self.stuck or self.w == 0.0:
            self.stuck = True
            if abs(net) <= self.static:
                self.w = 0.0
                return
            self.stuck = False
            s = np.sign(net)
        else:
            s = np.sign(self.w) if self.w != 0 else np.sign(net)
        dw = (self.k * (net - s * fric(self.w)) - self.w) / self.tau * dt
        w_new = self.w + dw
        if self.w != 0 and np.sign(w_new) != np.sign(self.w):   # 0 통과 -> 멈춤 판정
            w_new = 0.0
            if abs(net) <= self.static:
                self.stuck = True
        self.w = w_new
        self.th += self.w * dt

    @property
    def ticks(self):
        return int(np.floor(self.th / (2 * np.pi) * TPR))


class PI:
    """duty = FF(w*) + Kp*e + I.  FF = w*/K + sgn*F_ff (+ 기동 킥)."""

    def __init__(self, ts, win, kp, ki, ff=True, f_ff=5.0, kick=40.0, i_max=150.0, tau_leak=0.0,
                 kick_ticks=1, w_min=0.0, tau_m=0.0):
        self.ts, self.kp, self.ki, self.ff, self.f_ff, self.kick = ts, kp, ki, ff, f_ff, kick
        self.i_max, self.tau_leak, self.kick_ticks, self.w_min = i_max, tau_leak, kick_ticks, w_min
        self.n = max(1, int(round(win / ts)))
        self.hist = [0] * (self.n + 1)
        self.i = 0.0
        self.w_meas = 0.0
        self.tau_m = tau_m
        self.r_m = 0.0

    def update(self, ticks, w_ref):
        self.hist = self.hist[1:] + [ticks]
        d = self.hist[-1] - self.hist[0]
        self.w_meas = d / TPR * 2 * np.pi / (self.n * self.ts)
        moving = abs(self.hist[-1] - self.hist[-2]) >= self.kick_ticks
        if abs(w_ref) < self.w_min:
            self.i = 0.0
            self.r_m = 0.0
            return 0.0
        # 기준 모델: FF 만으로 바퀴가 따라갈 궤적 (1차, tau_m). PI 는 그 궤적과의 차이만 고친다
        if self.tau_m > 0:
            self.r_m += (w_ref - self.r_m) * self.ts / (self.tau_m + self.ts)
        else:
            self.r_m = w_ref
        e = self.r_m - self.w_meas
        ff = 0.0
        if self.ff and w_ref != 0:
            ff = w_ref / K + np.sign(w_ref) * self.f_ff
            if not moving:
                ff += np.sign(w_ref) * self.kick
        u_pre = ff + self.kp * e + self.i
        sat = abs(u_pre) >= PWM_MAX and np.sign(u_pre) == np.sign(e)
        if not sat:                                   # 조건부 적분
            self.i += self.ki * e * self.ts
        if self.tau_leak > 0:
            self.i -= self.i * self.ts / self.tau_leak
        self.i = float(np.clip(self.i, -self.i_max, self.i_max))
        if w_ref == 0:
            self.i = 0.0
        u = ff + self.kp * e + self.i
        return float(np.clip(u, -PWM_MAX, PWM_MAX))


def run(ctrl, ref_fn, load_fn=lambda t: 0.0, T=3.0, plant=None, dt=1e-3):
    p = plant or Plant()
    n = int(T / dt)
    every = int(round(ctrl.ts / dt)) if ctrl else 20
    u = 0.0
    t_log, w_log, u_log, r_log = [], [], [], []
    for i in range(n):
        t = i * dt
        if i % every == 0:
            r = ref_fn(t)
            u = ctrl.update(p.ticks, r) if ctrl else float(np.clip(r / (7.96 / PWM_MAX), -PWM_MAX, PWM_MAX))
        p.step(u, load_fn(t), dt)
        t_log.append(t); w_log.append(p.w); u_log.append(u); r_log.append(ref_fn(t))
    return np.array(t_log), np.array(w_log), np.array(u_log), np.array(r_log)


def metrics(t, w, u, r, t0, t1):
    s = (t >= t0) & (t < t1)
    e = r[s] - w[s]
    return dict(mean_err=e.mean(), rms=np.sqrt((e ** 2).mean()), du=np.abs(np.diff(u[s])).sum() / (t1 - t0))


def rise(t, w, target, t0):
    s = t >= t0
    k = np.argmax(w[s] >= 0.9 * target) if (w[s] >= 0.9 * target).any() else -1
    return (t[s][k] - t0) * 1e3 if k >= 0 else np.nan


# ---- 시나리오 ----

def step_ref(a, b=0.0, t1=0.2, t2=None, c=None):
    return lambda t: (a if t >= t1 else b) if (t2 is None or t < t2) else c

def scen(mk):
    out = {}
    t,w,u,r = run(mk(), step_ref(2.0), T=2.0)
    out["rise2"] = rise(t,w,2.0,0.2); out["ovs"] = (w.max()-2.0)/2.0*100
    m = metrics(t,w,u,r,1.0,2.0); out["e2"] = m["mean_err"]; out["du"] = m["du"]
    t,w,u,r = run(mk(), step_ref(0.3), T=3.0)
    m = metrics(t,w,u,r,1.0,3.0); out["e03"] = m["mean_err"]; out["rms03"] = m["rms"]
    out["rise03"] = rise(t,w,0.3,0.2)
    for ks in (0.85, 1.05):
        t,w,u,r = run(mk(), step_ref(2.0), T=2.0, plant=Plant(kscale=ks))
        out[f"eV{ks}"] = metrics(t,w,u,r,1.0,2.0)["mean_err"]
    t,w,u,r = run(mk(), step_ref(2.0), load_fn=lambda t: 80.0 if t >= 1.0 else 0.0, T=3.0)
    s = t >= 1.0
    out["load_drop"] = 2.0 - w[s].min()
    out["load_e"] = metrics(t,w,u,r,2.0,3.0)["mean_err"]
    ok = np.abs(w[s]-2.0) < 0.1
    out["load_rec"] = np.nan
    for k in range(len(ok)):
        if ok[k:].all(): out["load_rec"] = k; break
    t,w,u,r = run(mk(), step_ref(2.0, t2=1.0, c=-2.0), T=2.0)
    out["rev"] = rise(t, -w, 2.0, 1.0)
    return out

keys = ["rise2","ovs","e2","du","rise03","e03","rms03","eV0.85","eV1.05","load_drop","load_rec","load_e","rev"]


def _pi(ts, win, wc, **kw):
    kp = wc * TAU / K
    return lambda: PI(ts, win, kp, kp / TAU, **kw)


if __name__ == "__main__":
    C = {"개루프 (현행 duty=w/7.96*799)": lambda: None}
    for ts, win in ((0.02, 0.02), (0.005, 0.02)):
        for wc in (15, 20, 30):
            C[f"PI {1 / ts:.0f}Hz wc{wc} 기준모델 없음"] = _pi(ts, win, wc)
            C[f"PI {1 / ts:.0f}Hz wc{wc} τm 40ms"] = _pi(ts, win, wc, tau_m=0.04)
    C["PI 50Hz wc20 τm40 킥 없음 (채택-킥)"] = _pi(0.02, 0.02, 20, tau_m=0.04, kick=0.0)
    print("rise=90% 도달 ms, ovs=오버슈트 %, e=정상오차 rad/s, du=duty 변화량/s, eV=전압 배율 정상오차,")
    print("load_*=80 duty 부하 계단 (처짐 rad/s, 회복 ms, 정상오차), rev=2->-2 역전 90% ms.  채택: PI 50Hz wc20 τm 40ms")
    print(f"{'':36s}" + "".join(f"{k:>10s}" for k in keys))
    for name, mk in C.items():
        o = scen(mk)
        print(f"{name:36s}" + "".join(f"{o[k]:10.3f}" if abs(o[k]) < 100 else f"{o[k]:10.0f}" for k in keys))
