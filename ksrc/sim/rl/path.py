"""기준경로와 pure pursuit.

**왜 Nav2 를 직접 안 쓰는가**
학습 루프는 16 env 병렬이라 ROS2 노드를 띄울 수가 없다.  그리고 필요한 것은
플래너가 아니라 **플래너의 출력 인터페이스**다 -- 차체좌표 lookahead 점과
cmd_vel.  그래서 같은 인터페이스를 파이썬으로 미러링한다.  나중에 이 자리에
Nav2(`nav2_mppi_controller`, motion_model=Omni)를 꽂아도 정책은 그대로 쓴다.

**왜 RPP 가 아니라 홀로노믹 추종인가**
`nav2_regulated_pure_pursuit_controller` 는 differential/ackermann 용이라
(v, omega) 만 낸다.  스워브의 게걸음(vy)을 통째로 버리는 셈이라, 여기서는
lookahead 점을 향한 속도벡터를 그대로 낸다 (vx, vy, omega).

**왜 경로를 매번 새로 뽑는가**
경로 하나에 고정하면 그 경로에 과적합된다.  그리고 직선만 주면 나중에 실제
플래너가 내는 곡선 경로에서 무너지므로, 중간점을 꺾어 곡률을 일부러 섞는다.
"""
import math

import numpy as np

import terrain as terr


def _wrap(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


class RefPath:
    """꺾인 선분열 + 호길이 파라미터화.

    투영은 **단조 진행**으로 한다.  전역 최근접점을 쓰면 경로가 자기 근처로
    되돌아오는 구간에서 진행도가 뒤로 점프하고, 그러면 진행 보상이 음수로
    튀어 학습이 망가진다.
    """

    def __init__(self, pts):
        self.P = np.asarray(pts, float)
        seg = np.diff(self.P, axis=0)
        L = np.linalg.norm(seg, axis=1)
        keep = L > 1e-6
        self.P = np.vstack([self.P[:1], self.P[1:][keep]])
        seg = np.diff(self.P, axis=0)
        self.L = np.linalg.norm(seg, axis=1)
        self.U = seg / self.L[:, None]
        self.S = np.concatenate([[0.0], np.cumsum(self.L)])
        self.total = float(self.S[-1])
        self.i = 0                      # 진행 중인 구간 (되돌아가지 않는다)
        self.s_last = 0.0               # 직전 진행도 (단조·상한용)
        self.p_last = self.P[0].copy()   # 직전 질의 위치 (변위 상한용)

    def seed(self, s, p):
        """경로 중간에서 시작할 때 단조 투영의 기준도 같은 위치로 맞춘다."""
        s = float(np.clip(s, 0.0, self.total))
        self.i = int(np.clip(np.searchsorted(self.S, s, side="right") - 1,
                             0, len(self.L) - 1))
        self.s_last = s
        self.p_last = np.asarray(p, dtype=float).copy()

    def distance(self, p):
        """전체 경로 선분까지의 실거리. 횡복귀 보상용으로 연속적이다.

        단조 진행 투영의 국소 선분 전환은 e_y 를 점프시킬 수 있다. 이 전역
        실거리는 위치 변위보다 빠르게 변할 수 없어 가짜 복귀 보상을 막는다.
        진행도 계산에는 사용하지 않는다.
        """
        d = np.asarray(p, dtype=float) - self.P[:-1]
        t = np.clip(np.sum(d * self.U, axis=1), 0.0, self.L)
        q = self.P[:-1] + self.U * t[:, None]
        return float(np.min(np.linalg.norm(np.asarray(p, dtype=float) - q, axis=1)))

    # ------------------------------------------------------------------
    def project(self, p, limit=True):
        """(호길이 s, 부호있는 횡이탈 e_y, 접선 단위벡터) 를 돌려준다.

        **선분 위 최근접점까지의 실제 거리**로 후보를 고른다.  예전엔 선분을
        무한히 연장한 직선까지의 수직거리로 골랐는데, 그러면 두 가지가 깨졌다:

          - 선분 끝을 지나면 연장선 위에 있어 `e_y = 0` 이 나온다.  직선 1m
            경로의 끝에서 1.02m 를 지난 점이 `e_y=0`, 남은거리 0 으로 계산돼
            **완주 판정까지 True** 였다.
          - 꺾인 경로에서 아직 도달하지 않은 뒤쪽 선분의 연장선이 더 가까우면
            거기로 넘어간다.  ㄱ자 경로의 (0.6, 0.5) 에서 실제 진행은 0.60m
            인데 `s = 1.500` 이 나왔다 -- 코너를 건너뛰면 0.9m 의 유령 진행
            보상이 붙는다.

        `e_y` 는 **최근접점까지의 부호있는 거리**다.  선분 내부에 떨어지면
        직선까지의 수직거리와 같고, 끝을 벗어나면 끝점까지의 거리가 된다
        (부호는 진행방향 기준 좌/우를 유지).

        최근접점 기준만으로는 **코너 건너뛰기**가 안 막힌다.  ㄱ자 경로의
        (0.6, 0.5) 는 선분1 까지가 0.40m, 선분0 까지가 0.50m 로 선분1 이 실제로
        더 가깝다 -- 기하적으로는 맞지만 진행도가 0.6 -> 1.5 로 튄다.

        그래서 진행도 증가를 **실제 변위 `|p - p_last|` 로 상한**한다.  경로를 따라
        간 거리가 실제로 움직인 거리를 넘을 수 없다는 물리적 사실이다.

        상한을 고정 상수(`max_adv`)로 두면 안 된다 -- 그러면 코너를 자른 위치에
        **정지해 있어도** 매 스텝 상한만큼 진행도가 올라간다 (실측: s 가 0.030씩
        증가, prog = 6.8 -> clip 1.0 = 최대 보상).  코너 자르기 악용을 정지 악용으로
        바꾸는 셈이다.
        """
        best = None
        for j in range(self.i, min(self.i + 3, len(self.L))):
            d = p - self.P[j]
            t = float(np.clip(np.dot(d, self.U[j]), 0.0, self.L[j]))
            q = self.P[j] + self.U[j] * t              # 선분 위 최근접점
            v = p - q
            dist = float(np.linalg.norm(v))
            sgn = np.sign(np.cross(self.U[j], v))
            if sgn == 0.0:
                sgn = np.sign(np.cross(self.U[j], d)) or 1.0
            if best is None or dist < best[0]:
                best = (dist, self.S[j] + t, float(sgn) * dist, j)
        _, s, e, j = best
        if limit:
            # 실제 변위가 상한.  정지하면 0 이므로 진행도가 안 올라간다.
            cap = self.s_last + float(np.linalg.norm(p - self.p_last))
            if s > cap:                       # 건너뛰기 -- 진행도를 묶는다
                s = cap
                j = int(np.clip(np.searchsorted(self.S, s, side="right") - 1,
                                0, len(self.L) - 1))
                d = p - self.P[j]
                t = float(np.clip(np.dot(d, self.U[j]), 0.0, self.L[j]))
                q = self.P[j] + self.U[j] * t
                v = p - q
                dist = float(np.linalg.norm(v))
                sg = np.sign(np.cross(self.U[j], v)) or 1.0
                e = float(sg) * dist
            s = max(s, self.s_last)           # 단조
        self.s_last = s
        self.p_last = np.asarray(p, float).copy()
        self.i = j
        return s, e, self.U[j]

    def at(self, s):
        """호길이 s 지점의 좌표 (양끝은 잘라낸다)."""
        s = float(np.clip(s, 0.0, self.total))
        j = int(np.searchsorted(self.S, s, side="right") - 1)
        j = int(np.clip(j, 0, len(self.L) - 1))
        return self.P[j] + self.U[j] * (s - self.S[j])


def _grad(Z, ex, ey):
    """지형 기울기 (gx, gy) [무차원].  make_path 가 매 선분마다 쓰므로 캐시한다."""
    gy, gx = np.gradient(Z)
    return gx / (ex / Z.shape[1]), gy / (ey / Z.shape[0])


def _seg_cross_slope(gx, gy, Z, ex, ey, a, b):
    """선분 a->b 의 **측면경사(side-hill) 성분** [deg].

    경사를 오르내리면 pitch 만 생기고 roll 은 0 이다.  내리막 횡드리프트는 roll
    이 만들므로, 경사 **크기**가 아니라 진행방향에 **수직한 성분**을 봐야 한다.
    """
    u = np.asarray(b, float) - np.asarray(a, float)
    n = float(np.linalg.norm(u))
    if n < 1e-9:
        return 0.0
    u /= n
    m = 0.5 * (np.asarray(a, float) + np.asarray(b, float))
    jx = int(np.clip((m[0] + ex / 2) / ex * Z.shape[1], 0, Z.shape[1] - 1))
    iy = int(np.clip((m[1] + ey / 2) / ey * Z.shape[0], 0, Z.shape[0] - 1))
    cross = abs(float(gx[iy, jx] * u[1] - gy[iy, jx] * u[0]))
    return math.degrees(math.atan(cross))


def make_path(drive, Z, ex, ey, rng, start, min_len=2.0, max_len=3.5,
              step=(0.7, 1.4), turn_deg=70.0, tries=40,
              slope_deg=None, d_ep=0.0, n_cand=8, turn_deg_max=90.0,
              cross_slope=False):
    """주행 가능 셀만 밟는 꺾인 경로.

    > [!important] 2026-10-04: fallback 상한을 170도 -> 90도 로 낮췄다
    > `turn_deg_max` 전의 하드코딩 170도 때문에 실측 꺾임각이 **p90 148~166도,
    > 최대 168도** 였다 (학습 seed 1234 평균 69.1 / 평가 4321 평균 82.9).
    > 168도는 경로가 되돌아가는 **헤어핀**이고 로버가 멈춰서 제자리 회전을 해야
    > 통과한다 -- "어려운 과제" 가 아니라 "불가능한 과제" 다.
    > Nav2 격자 planner 는 한 스텝 최대 45도이고 smoother 를 거치면 더 완만하다.
    >
    > 90도로 정한 근거: 경기장 맵의 꺾임이 직각이고, 직각은 로버가 게걸음·회전으로
    > 실제로 통과할 수 있다.  Nav2 `/plan` 실측 분포가 나오면 그 p99 로 다시 맞춘다
    > (`ros/plan_stats.py`).
    >
    > **이 변경 이전(2026-10-03까지)의 모든 숫자는 헤어핀 경로 위의 값이다** --
    > 순수 IK 기준선·s9 기각·s10 판정·기구선호 접기 기각 전부
    > ([[RL 주행보조 진행상황]] 2026-10-03 §6).

    한 번 막혔을 때 바로 포기하면 30% 가 0.5m 직선으로 퇴화했다 (막힌 방향을
    계속 고집하기 때문).  그래서 좁은 선회각으로 먼저 시도하고, 실패하면
    **선회각 제한을 풀어** 되돌아가는 방향까지 허용한다.  초기 방향도 경기장
    안쪽으로 편향시킨다 -- 벽을 향해 시작하면 첫 구간부터 막힌다.

    `d_ep` 에 비례해 **급사면을 지나는 구간**을 고른다.  line_clear 만 보면
    경로가 험한 지형을 전부 피해가서, 난이도를 올려도 과제가 안 어려워진다
    (그 상태로 재보니 순수 IK 가 d=1.0 에서 93% 를 완주했다).  30도 사면은
    맵 면적의 1% 라 무작위로는 거의 안 지나간다.
    """
    start = np.asarray(start, float)
    gx, gy = _grad(Z, ex, ey) if cross_slope else (None, None)
    pts = [start]
    head = float(np.arctan2(-start[1], -start[0]))          # 중심 방향
    head += float(rng.uniform(-np.pi / 2, np.pi / 2))
    total = 0.0
    while total < min_len:
        got = False
        for lim in (turn_deg, turn_deg_max):                # 2단: 좁게 -> 넓게
            cand = []
            for _ in range(tries):
                r = float(rng.uniform(*step))
                a = head + float(np.radians(rng.uniform(-lim, lim)))
                nxt = pts[-1] + r * np.array([np.cos(a), np.sin(a)])
                if abs(nxt[0]) > ex / 2 - 0.3 or abs(nxt[1]) > ey / 2 - 0.3:
                    continue
                if not terr.line_clear(drive, ex, ey, pts[-1], nxt):
                    continue
                cand.append((nxt, a, r))
                if len(cand) >= n_cand:
                    break
            if not cand:
                continue
            if slope_deg is not None and rng.random() < d_ep:
                if cross_slope:
                    # **측면경사 최대**.  line_mean(slope) 최대화는 가장 급한
                    # 방향으로 곧장 오르는 후보를 골라서 roll 이 0 이 된다
                    # (2026-10-04 측정: pitch 11.89도 / roll 1.10도).
                    j = int(np.argmax([_seg_cross_slope(gx, gy, Z, ex, ey,
                                                        pts[-1], cc[0])
                                       for cc in cand]))
                else:
                    j = int(np.argmax([terr.line_mean(slope_deg, ex, ey,
                                                      pts[-1], cc[0])
                                       for cc in cand]))
            else:
                j = 0
            nxt, a, r = cand[j]
            pts.append(nxt); head = a; total += r; got = True
            break
        if not got:
            break
        if total > max_len:
            break
    if len(pts) < 2:
        # 기존 무조건 0.5m 직선 fallback 은 통행 불가 셀을 뚫을 수 있었다.
        # 짧은 대체 경로도 반드시 같은 마스크로 검증한다.
        for da in np.linspace(0.0, 2 * np.pi, 32, endpoint=False):
            a = head + da
            nxt = pts[0] + 0.5 * np.array([np.cos(a), np.sin(a)])
            if (abs(nxt[0]) <= ex / 2 - 0.3 and
                    abs(nxt[1]) <= ey / 2 - 0.3 and
                    terr.line_clear(drive, ex, ey, pts[0], nxt)):
                pts.append(nxt)
                break
        else:
            raise ValueError("시작점에서 주행 가능한 경로를 찾지 못했습니다")
    return RefPath(pts)


def pursuit_distance(v_cruise, k_v=2.0, L_min=0.25, L_max=0.80):
    """기본 제어기와 정책 관측이 공유하는 경로 선행 거리 [m]."""
    return float(np.clip(k_v * v_cruise, L_min, L_max))


def pursue(path, s, pos, yaw, v_cruise, k_v=2.0, L_min=0.25, L_max=0.80,
           k_psi=1.2, om_lim=0.8):
    """차체좌표 속도명령 (vx, vy, omega).

    lookahead 거리를 속도에 비례시키는 것(regulated)은 Nav2 RPP 와 같다.
    요는 **경로 방향에 정렬**시킨다 -- 스워브는 요가 자유도지만, 명령을 주지
    않으면 슬립으로 틀어진 요를 되잡을 학습 신호가 생기지 않는다.

    요 기준을 **선분 접선이 아니라 [s, s+L] 구간의 현(chord)** 으로 잡는다.
    접선은 꼭짓점에서 크게 튀어서 omega 명령에 계단이 생기고(측정: 중앙
    0.584 rad/s, 13% 가 ±0.8 포화), 정책이 물리적으로 못 따라갈 것을 벌한다.
    **실측 꺾임각 p90 148~166도, 최대 168도** (2026-10-03). 이전 주석의 '최대 70도'
    는 turn_deg 기본값이고, make_path 가 막히면 fallback 으로 170도까지 허용한다.
    현 방향은 s 에 대해 연속이고 직선 구간에서는 접선과 같다.
    또한 현은 **경로만의 함수**라 횡이탈과 커플링되지 않는다 -- 로버에서 본
    lookahead 방향을 요 기준으로 쓰면, 경로를 벗어날수록 몸을 틀어버려서
    vy(게걸음)로 복귀하는 경로가 막힌다.
    """
    L = pursuit_distance(v_cruise, k_v, L_min, L_max)
    tgt = path.at(s + L)
    d = tgt - pos
    n = float(np.linalg.norm(d))
    rem = path.total - s
    v_w = d / max(n, 1e-6) * min(v_cruise, 1.5 * max(rem, 0.0))
    c, sn = np.cos(yaw), np.sin(yaw)
    v_b = np.array([c * v_w[0] + sn * v_w[1], -sn * v_w[0] + c * v_w[1]])
    om = float(np.clip(k_psi * _wrap(course_at(path, s, L) - yaw),
                       -om_lim, om_lim))
    return np.array([v_b[0], v_b[1], om])


def lookahead_body(path, s, pos, yaw, dists=(0.30, 0.80)):
    """관측용: 앞쪽 lookahead 점들을 차체좌표로.

    실기에서는 Nav2 의 local plan + TF 에서 나온다.  즉 **측위가 있어야**
    얻을 수 있는 값이므로, env 에서 측위 오차를 노이즈로 섞어 준다.
    """
    c, sn = np.cos(yaw), np.sin(yaw)
    out = []
    for L in dists:
        d = path.at(s + L) - pos
        out.append([c * d[0] + sn * d[1], -sn * d[0] + c * d[1]])
    return np.asarray(out, float)


def start_heading(path, L=0.5):
    """경로 시작부의 진행 방향.  스폰 자세를 여기에 대충 맞춘다."""
    d = path.at(min(L, path.total)) - path.P[0]
    return float(np.arctan2(d[1], d[0]))


def heading_at(path, s, eps=0.05):
    """호길이 s 지점의 **경로 방향**.  중앙 현(central chord) 으로 정의한다.

    선분 접선을 그대로 쓰면 꼭짓점에서 각도가 순간 점프한다 (**실측 꺾임각은
    p90 148~166도, 최대 168도** -- make_path 의 fallback 170도 때문이다.
    주석에 '최대 70도' 로 적혀 있었는데 틀렸다, 2026-10-03).  그러면 스폰 자세·관측 e_psi·보상이 같은 지점에서 불연속을 겪는다.
    중앙 현은 s 에 대해 연속이고 직선 구간에서는 접선과 같다.

    스폰 자세·관측 e_psi·실제 이동방향 보상이 같은 접선 정의를 공유한다.
    """
    a = path.at(max(s - eps, 0.0))
    b = path.at(min(s + eps, path.total))
    d = b - a
    if float(np.linalg.norm(d)) < 1e-9:          # 경로가 eps 보다 짧은 극단
        d = path.U[min(path.i, len(path.U) - 1)]
    return float(np.arctan2(d[1], d[0]))


def course_at(path, s, L):
    """제어기가 요 기준으로 삼는 **[s, s+L] 구간의 현(chord)** 방향.

    `pursue` 의 요 명령과 `e_psi` 평가가 **같은 기준**을 쓰게 하려고 뽑아냈다.
    평가에 `heading_at` (앞뒤 5cm 접선) 을 쓰면, 꼭짓점 앞에서 제어기가 미리
    트는 올바른 선행 조향이 그대로 기수오차로 기록된다.  보상의 `head` 벌점과
    align 게이트 `max(0, cos e_psi)` 가 그 값을 쓰므로, 추종이 가장 중요한
    지점에서 정책이 제어기에 복종한 것을 벌하게 된다.
    (holdout 30경로 측정: 두 기준의 차이 평균 8.28도, p90 25.3도, 중앙값 0도.
     70도 꼭짓점 5cm 앞에서는 58.5도까지 벌어진다.)

    경로 끝에서 현의 길이가 0 이 되면 접선으로 되돌아간다.
    """
    d = path.at(min(s + L, path.total)) - path.at(s)
    if float(np.linalg.norm(d)) < 1e-9:
        return heading_at(path, s)
    return float(np.arctan2(d[1], d[0]))


def path_mean_slope_deg(path, slope, ex, ey, step=0.05):
    """경로가 지나는 지형 경사의 평균 [deg]."""
    q = [path.at(s) for s in np.arange(0.0, path.total, step)]
    if not q:
        return 0.0
    return float(np.mean([terr.line_mean(slope, ex, ey, x, x) for x in q]))


def path_mean_cross_slope_deg(path, Z, ex, ey, step=0.05):
    """경로의 **측면경사(side-hill) 성분** 평균 [deg].

    경사가 있어도 그것을 **오르내리면** pitch 만 생기고 roll 은 0 이다.  내리막
    횡드리프트(논문이 경사 잔디에서 관찰한 현상)는 **roll** 이 만들고, roll 은
    경사를 **가로지를 때** 생긴다.

    2026-10-04 측정: 평균경사 하한 8도만 걸면 |pitch| 평균 11.89도인데
    **|roll| 은 1.10도** 뿐이다 (roll>5도 비율 2.9%).  경로가 사면을 오르내리고
    있었다.  그래서 경사 크기가 아니라 **진행방향에 수직한 성분**을 재야 한다.

    지형 기울기 g 와 진행방향 u 에 대해 측면성분 = |g| * |sin(angle(g, u))|
    를 각도로 환산한다 (u 가 g 에 수직 = 완전한 side-hill, 평행 = 직등반).
    """
    gy, gx = np.gradient(Z)
    sy, sx = (ey / Z.shape[0]), (ex / Z.shape[1])
    gx = gx / sx
    gy = gy / sy
    vals = []
    ss = np.arange(0.0, max(path.total - step, step), step)
    for s0 in ss:
        a = path.at(s0)
        b = path.at(min(s0 + step, path.total))
        u = b - a
        n = float(np.linalg.norm(u))
        if n < 1e-9:
            continue
        u = u / n
        jx = int(np.clip((a[0] + ex / 2) / ex * Z.shape[1], 0, Z.shape[1] - 1))
        iy = int(np.clip((a[1] + ey / 2) / ey * Z.shape[0], 0, Z.shape[0] - 1))
        g = np.array([gx[iy, jx], gy[iy, jx]])
        gm = float(np.linalg.norm(g))
        if gm < 1e-9:
            vals.append(0.0)
            continue
        # u 에 수직한 성분 = 외적 크기 / |u|  (|u|=1)
        cross = abs(float(g[0] * u[1] - g[1] * u[0]))
        vals.append(math.degrees(math.atan(cross)))
    return float(np.mean(vals)) if vals else 0.0


def make_path_set(n, drive, Z, ex, ey, seed, slope=None,
                  min_mean_slope_deg=0.0, min_cross_slope_deg=0.0, **kw):
    """고정 경로 세트.  **학습용과 평가용을 반드시 다른 seed 로 만든다.**

    같은 세트로 학습·평가하면 `s0` 를 랜덤화해도 "그 geometry 를 외운 건지"
    구분할 수 없다.

    `min_mean_slope_deg` 로 **경로가 지나는 평균 경사의 하한**을 둘 수 있다.
    왜 필요한가 (2026-10-04 측정): 이 하한이 없으면 뽑힌 경로의 경사 중앙값이
    **0도**(평균 3.98) 이고 주행 중 차체 |roll| 이 평균 0.44도 뿐이다.  그러면
    견인력 수요가 R/N = 0.057 로 가용 mu(i) 0.17~0.39 의 1/3~1/7 이라,
    바퀴별 mu 를 +-16% 흔들어도(soil_amp 0.25) 횡오차·기수오차가 0.03mm /
    0.04deg 밖에 안 변한다 -- **목적 현상(비대칭 슬립으로 yaw 틀어짐)이
    물리적으로 발생하지 않는다.**  경사가 있어야 roll 이 생겨 내리막 드리프트가
    나오고, 그때 비로소 mu 비대칭이 의미를 갖는다.
    """
    rng = np.random.default_rng(seed)
    if min_mean_slope_deg > 0.0 and slope is None:
        raise ValueError("min_mean_slope_deg 를 쓰려면 slope 배열이 필요합니다")
    out = []
    attempts = 0
    while len(out) < n:
        attempts += 1
        if attempts > 400 * n:
            raise RuntimeError(
                f"경로를 충분히 생성하지 못했습니다 (평균경사 하한 "
                f"{min_mean_slope_deg}도가 이 지형에서 너무 높을 수 있습니다)")
        st = terr.sample_drivable(drive, Z, ex, ey, rng, 1)[0]
        try:
            pth = make_path(drive, Z, ex, ey, rng, st, **kw)
        except ValueError:
            continue
        if pth.total < 2.0:                      # 퇴화 경로 제외
            continue
        if min_mean_slope_deg > 0.0 and \
                path_mean_slope_deg(pth, slope, ex, ey) < min_mean_slope_deg:
            continue
        # 측면경사(roll 을 만드는 성분).  경사 크기만 보면 사면을 오르내리는
        # 경로가 뽑혀서 roll 이 1도밖에 안 생긴다 (위 함수 docstring 참고).
        if min_cross_slope_deg > 0.0 and \
                path_mean_cross_slope_deg(pth, Z, ex, ey) < min_cross_slope_deg:
            continue
        out.append(pth)
    return out


def clone(path):
    """같은 꼭짓점의 새 RefPath.  투영 상태(s_last, p_last, i)가 독립이어야
    보상용(참값)과 컨트롤러용(추정치)을 분리할 수 있다."""
    return RefPath(path.P)
