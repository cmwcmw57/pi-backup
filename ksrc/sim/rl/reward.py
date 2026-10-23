"""논문식 진행·횡오차·헤딩·매끄러움을 스워브 잔차제어에 맞춘 보상."""
from dataclasses import dataclass
import math

import numpy as np


@dataclass
class RewardCfg:
    """진행·횡오차·횡복귀·차체 헤딩·행동 크기/변화로 경로추종을 평가한다.

    2026년 4WIS 논문의 항목은 참고하지만 진행량과 행동의 단위가 달라 계수를
    복사하지 않는다. 논문처럼 차체 헤딩 정렬 게이트를 쓰되 이동 방향에는
    게이트를 걸지 않는다.
    """
    w_prog:    float = 0.20    # + 횡오차가 커도 경로 진행에 남는 바닥값
    w_align:   float = 0.80    # + 횡오차 게이트가 걸린 경로 진행
    w_lat:     float = 0.04    # - 이동 중 횡오차; 먼 곳에서 복귀 보상보다 크지 않게
    # 2026-10-04: 0.25 -> 0.60.  **w_recover 가 w_lat/w_resid 의 상한을 정한다.**
    # test_sideways_recovery_beats_stopping_when_heading_aligned 이 "300mm 이탈에서
    # 전체 잔차로 복귀하는 것이 정지보다 이득" 을 요구하고, 그 조건이
    #     4*w_lat + w_resid < w_recover
    # 다 (lat 은 min(., 4) 상한에 걸리므로 4*w_lat).  깨면 정책이 **멈추는 것을
    # 배운다**.  0.25 에서는 4(0.04)+0.06 = 0.22 로 여유가 0.03 뿐이라 s9~s12 에서
    # 확인된 잔차 포화(결정론적 p95 100%)를 벌할 수가 없었다.
    w_recover: float = 0.60    # 할인 일치 경로 실거리 잠재함수 가중치
    w_course:  float = 0.0     # 대각선 횡복귀를 벌주지 않는다
    w_head:    float = 0.02    # - 헤딩 게이트와 중복되는 이차 벌점은 약하게
    w_smooth:  float = 0.01    # - 행동 2차 차분; PPO 탐색을 압도하지 않게
    # 2026-10-04: 0.06 -> 0.20.  s9/s10/s11/s12 네 런에서 **결정론적 잔차 p95 가
    # 항상 100%** 였고 cmd_sat 이 기준선의 2~4.3배였다.  메커니즘: Delta vx=+0.03 은
    # v_cruise(0.14~0.32)의 +9~21% 라 포화시키면 prog/align 보상이 늘고, 대가는
    # w_resid x 1 = 0.06/스텝 뿐이다.  실측 교환비가 "진행 +0.3%p 대 횡오차 +6.4mm"
    # 로 잘못돼 있다 (align 대비 세금이 resid 7.7% + smooth 3.4% = 11% 뿐).
    w_resid:   float = 0.20    # - 지속적인 큰 차체속도 잔차
    r_goal:    float = 20.0
    r_tip:     float = -10.0
    r_stuck:   float = -5.0
    r_oob:     float = -10.0   # 예전엔 벌점이 **없어서** 경계 밖 탈출이 공짜였다
    sigma_y:   float = 0.08    # m.  20~30mm 횡오차 차이를 더 분명히 구별
    sigma_heading: float = 0.35  # rad. 논문과 같은 헤딩오차 정규화 폭
    sigma_course: float = 0.35  # rad.  이동 방향오차 정규화 폭
    course_gate: bool = False  # 경로 중앙으로 돌아오는 대각 이동 허용
    heading_gate: bool = True   # 논문의 max(0, cos(e_psi)) 정렬 게이트


def recovery_potential(prev_dist, next_dist, den, gamma, terminal=False):
    """경로 실거리 잠재함수의 할인 일치 변화량 (가중치 적용 전).

    Phi(s)=-d_route/den, F=gamma*Phi(s')-Phi(s). 종료 상태의 Phi 는 0이다.
    같은 시간 동안 같은 상태로 되돌아오면 왕복과 정지의 할인 보상은 같다.
    """
    if den <= 0 or not math.isfinite(den) or not 0.0 < gamma <= 1.0:
        raise ValueError("복귀 보상의 den/gamma 가 유효하지 않습니다")
    if min(prev_dist, next_dist) < 0 or not all(map(math.isfinite, (prev_dist, next_dist))):
        raise ValueError("경로 실거리는 음수가 아닌 유한한 값이어야 합니다")
    return (prev_dist - (0.0 if terminal else gamma * next_dist)) / den


def tracking_terms(prog, e_y, e_psi, action, prev_action, prev2_action, cfg,
                   *, course_error=0.0, movement=0.0, lateral_recovery=0.0):
    """참값 경로오차로 논문을 참고한 스워브 보상 항을 계산한다."""
    if cfg.sigma_y <= 0 or not math.isfinite(cfg.sigma_y):
        raise ValueError("sigma_y 는 양의 유한한 숫자여야 합니다")
    if cfg.sigma_course <= 0 or not math.isfinite(cfg.sigma_course):
        raise ValueError("sigma_course 는 양의 유한한 숫자여야 합니다")
    if cfg.sigma_heading <= 0 or not math.isfinite(cfg.sigma_heading):
        raise ValueError("sigma_heading 은 양의 유한한 숫자여야 합니다")
    lateral_error_sq = (e_y / cfg.sigma_y) ** 2
    lateral_gate = math.exp(-lateral_error_sq)
    gate = lateral_gate
    if cfg.course_gate:
        gate *= max(0.0, math.cos(course_error))
    if cfg.heading_gate:
        gate *= max(0.0, math.cos(e_psi))
    moving = float(np.clip(movement, 0.0, 1.0))
    # 정지 중에는 이동 방향이 정의되지 않는다. 오차 벌점도 정지 중
    # 누적되지 않게 하여, 일찍 실패해서 비용을 끊는 해를 만들지 않는다.
    course_error_sq = min((course_error / cfg.sigma_course) ** 2, 4.0)
    a = np.asarray(action, dtype=float)
    d2 = a - 2.0 * np.asarray(prev_action) + np.asarray(prev2_action)
    terms = {
        "prog": cfg.w_prog * prog,
        "align": cfg.w_align * prog * gate,
        "lat": -cfg.w_lat * moving * min(lateral_error_sq, 4.0),
        # 할인 일치 잠재함수 변화량. 각 스텝 clip 을 하면 서로 다른 크기의
        # 외측/내측 이동을 조합해 왕복 보상을 만들 수 있어 clip 하지 않는다.
        "recover": cfg.w_recover * float(lateral_recovery),
        "course": -cfg.w_course * moving * lateral_gate * course_error_sq,
        # 논문식 이차 헤딩오차. 큰 각도에서 PPO 보상이 폭주하지 않게 상한을 둔다.
        "head": -cfg.w_head * moving * min((e_psi / cfg.sigma_heading) ** 2, 4.0),
        "smooth": -cfg.w_smooth * float(np.mean(d2 ** 2)),
        "resid": -cfg.w_resid * float(np.mean(a ** 2)),
    }
    return gate, terms


@dataclass
class VelRewardCfg:
    """cmd_vel 추종(밑단) 보상.  경로·lookahead·진행량이 들어가지 않는다.

    설계 근거 (2026-10-05 측정, cmdvel_probe.py):
      순수 IK 는 랜덤 twist 를 상대오차 27~38% 로 틀린다.  그 오차는 노이즈가
      아니라 **슬립의 거의 선형 함수**다 (슬립 0.04 -> vx 오차 -0.015,
      슬립 0.34 -> -0.049).  따라서 추종오차를 직접 벌하면 **슬립 보정이
      별도 항 없이 자동으로 목표가 된다** -- 슬립은 속도 손실로 나타나므로.
      경로추종 보상에서 슬립 항이 없어 "슬립 보정" 을 한 번도 측정하지
      못했던 문제가 구조적으로 해결된다.

    `exp(-(e/sigma)^2)` 형태는 legged_gym 의 track_lin_vel_xy_exp 와 같다.
    유계라서 보상이 폭주하지 않지만, 오차 0 과 큰 오차에서는 기울기가
    작아진다 (최대 기울기는 e=sigma/sqrt(2)). 따라서 p90 개선은 보상
    상승만으로 판단하지 않고 별도로 측정한다.
    오차는 **무차원화**해서 넣는다 (v_max, omega_max 로 나눔) -- 로버 치수가
    바뀌어도 가중치를 다시 안 잡게.
    """
    w_v:      float = 0.60   # + 병진속도 추종 (vx, vy 합쳐 하나)
    w_om:     float = 0.40   # + 요속도 추종
    # sigma 는 **운용점 기울기와 원거리 학습가능성의 교환**으로 정한다.
    # 최초 설정 당시 무차원 오차: ||v_xy|| 0.039/0.318 = 0.124,
    # |om| 0.138/1.847 = 0.075. 현재 v_max=0.378, om_max=2.258 로
    # 바뀌었으므로 아래 수치 예시는 당시 스케일이며 실험 지표는 새 스케일로
    # 다시 계산했다.
    #   sig_v  운용점보상  운용점기울기   e=2배   e=3배
    #    0.12    0.344      5.92       0.014   0.000   <- 2배에서 기울기 死
    #    0.16    0.548      5.31       0.090   0.004   <- 채택
    #    0.25    0.782      3.10       0.374   0.109   <- 기울기 48% 손실
    # 0.12 -> 0.16 은 기울기를 10% 만 잃고 원거리를 6배 살린다.
    # 더 넓히지 않는 대신 **초기 탐색 std 를 낮춘다** (cmdvel 기본
    # --log-std-init -2.0 -> std 0.135). 현재 action scale 은
    # 0.11/0.085/0.45 이므로 std 1.0
    # 이면 초기 오차가 운용점의 3배까지 벌어져 보상이 평평해진다.
    sigma_v:  float = 0.16
    sigma_om: float = 0.13
    # 행동 벌점은 **작게**.  이 과제에는 전진량 보상이 없으므로 잔차를
    # 포화시켜서 얻을 것이 없다 (경로추종에서 포화가 난 원인은 align/prog 였다).
    # 남겨두는 이유는 실기 액추에이터 떨림 억제뿐이다.
    w_resid:  float = 0.01   # - 평균 a^2
    w_smooth: float = 0.01   # - 2차 차분
    w_tail:   float = 0.0    # 실험 노브: 큰 속도오차에 계속 기울기를 주는 이차 비용
    r_tip:    float = -10.0
    r_stuck:  float = -5.0
    r_oob:    float = -10.0


def velocity_terms(v_err_n, om_err_n, action, prev_action, prev2_action, cfg):
    """무차원 추종오차 -> 보상 항.

    v_err_n  : ||v_xy 실제 - v_xy 명령|| / v_max          (>= 0)
    om_err_n : |omega 실제 - omega 명령| / omega_max      (>= 0)
    """
    for nm, sg in (("sigma_v", cfg.sigma_v), ("sigma_om", cfg.sigma_om)):
        if sg <= 0 or not math.isfinite(sg):
            raise ValueError(f"{nm} 은 양의 유한한 숫자여야 합니다")
    if v_err_n < 0 or om_err_n < 0 or not all(map(math.isfinite, (v_err_n, om_err_n))):
        raise ValueError("추종오차는 음수가 아닌 유한한 값이어야 합니다")
    a = np.asarray(action, dtype=float)
    d2 = a - 2.0 * np.asarray(prev_action) + np.asarray(prev2_action)
    return {
        "track_v":  cfg.w_v * math.exp(-(v_err_n / cfg.sigma_v) ** 2),
        "track_om": cfg.w_om * math.exp(-(om_err_n / cfg.sigma_om) ** 2),
        "tail": -cfg.w_tail * ((v_err_n / cfg.sigma_v) ** 2 +
                               (om_err_n / cfg.sigma_om) ** 2),
        "resid":   -cfg.w_resid * float(np.mean(a ** 2)),
        "smooth":  -cfg.w_smooth * float(np.mean(d2 ** 2)),
    }
