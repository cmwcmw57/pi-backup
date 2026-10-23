"""접기(fold) 선택 기준을 **구동모터가 차체 중심에 가까운 쪽**으로 바꾼 env.

배경.  조향각 theta 와 theta+-180도 는 바퀴 속도 부호를 뒤집으면 **차체 운동이
완전히 같다** (접지점 속도벡터가 동일).  그래서 가동범위 안에 등가각이 둘 있으면
어느 쪽을 쓸지는 자유이고, 그 선택이 **모터 몸통이 어디로 튀어나오는지**를 정한다.

구동모터는 바퀴 축과 동축이라 (mjcf.py: zaxis="0 1 0") 조향각 +-90도 에서
차체 **전방 또는 후방**을 향한다.  앞 모듈은 뒤로, 뒤 모듈은 앞으로 접으면
네 모터가 차체 가운데로 모인다:

    순수 횡걸음에서  FL -90  FR +90  RL +90  RR -90
    모터 외접반경    217.3mm (전부 같은 부호)  ->  118.4mm

기본 `swerve_fold_to_range` 는 기준이 "직전 각에 가장 가까운 등가각"(이력)이라
네 바퀴가 같은 분기를 고르고, 위 조합이 나오지 않는다.

측정 결과 (2026-10-03).  이 모듈은 **비교·계측용이고 채택안이 아니다.**

명령열 재생 (60 에피소드, 24,999 스텝):
    모터 외접반경 최대   225.2 -> 208.0 mm   (-17.2)
    서보 총 회전량      80,490 -> 92,321 도  (+14.7%)
    180도 반전 스텝        179 -> 261        (+46%)
등가각이 2개였던 바퀴.스텝 5.74%, 기구선호가 다른 선택을 한 경우 0.13%.
그 0.13% 가 반전 82회를 만들고 82 x 180도 = 14,760도 가 추가 회전량의 거의 전부다
-- 이력이 없어서 |원시각| 80~100도 밴드에서 두 분기 사이를 떤다.

in-loop 짝비교 (100 에피소드, 순수 IK, holdout 4321):
    완주율      96.0 -> 71.0 %   (-25.0%p)
    고착         3.0 -> 28.0 %   (+25.0%p)
    경로 진행률  82.9 -> 72.3 %   (-10.6%p)
    조향 대기    9.78 -> 17.90 %  (+8.1%p)
    기수오차    25.55 -> 31.15 도
    슬립        0.178 -> 0.226

> [!caution] 위 in-loop 결과는 이 규칙을 **기각하는 증거가 아니다**
> 실패 연쇄가 "경로 꼭짓점에서 pure pursuit 명령 급변 -> 분기 추가 전환 -> 게이트
> -> 슬립 -> 고착" 인데, 쓴 경로 세트의 꺾임각이 p90 148~166도, 최대 168도(헤어핀)
> 였다 (`path.py:166` 의 fallback 170도).  Nav2 planner 는 그런 경로를 내지 않는다.
> **현실적인 경로에서 재측정해야 하며 현재 상태는 "미정" 이다.**

매끄러운 입력(조이스틱)에서는 이야기가 다르다: 방향 360도 스윕 당 180도 반전이
이력 우선 2회(100도, 280도) vs 기구 선호 4회(81, 100, 260, 280도)이고, 반전 1회가
약 0.6초다.  게이트 트리거는 둘 다 `driving` 을 요구하므로(`drive_align_gate.c:45,76`)
속도 0 으로 지나면 공짜다.  즉 한 바퀴당 추가 1.2초 vs 외접반경 217.3 -> 118.4mm 로,
**수동 조작에서는 받아들일 만한 거래다.**

살릴 방법은 `slow_mps` 조건과 결합하는 것 -- "구동 중이 아닐 때만 분기 전환".
펌웨어 `swerve_fold_to_range` 의 `unwind_rad`/`slow_mps` 가 그 구조이고 지금
`main.c:447`/`env.py:265` 에서 둘 다 0.0 으로 꺼져 있다.
"""
import math

import numpy as np

from config import RoverCfg
from env import RoverEnv, C2MJ   # env 가 sim/ 을 sys.path 에 넣는다
import ksrc_common as kc         # 그래서 env 보다 뒤에 import 해야 한다


def _motor_tip_radius(cfg: RoverCfg, ci: int, theta: float) -> float:
    """C 모듈 순서 ci 의 바퀴가 조향각 theta 일 때, 모터 끝단의 차체중심 거리."""
    d = cfg.wheel_w / 2.0 + cfg.motor_len
    # C 순서 = (fl, fr, rl, rr).  sy 는 mjcf 의 좌(+1)/우(-1) 부호.
    x0 = cfg.axle_x if ci < 2 else -cfg.axle_x
    sy = 1.0 if ci % 2 == 0 else -1.0
    y0 = sy * cfg.track / 2.0
    ly = sy * cfg.motor_y_dir * d
    return math.hypot(x0 - math.sin(theta) * ly, y0 + math.cos(theta) * ly)


class MechFoldEnv(RoverEnv):
    """`_ik_of` 의 접기 기준만 바꾼다.  나머지 물리·보상·관측은 동일하다."""

    def _ik_of(self, cmd):
        out = kc.swerve_ik_compute(float(cmd[0]), float(cmd[1]), float(cmd[2]),
                                   self.c_modules, self.v_max, self.c_state)
        th = np.zeros(4)
        v = np.zeros(4)
        for ci in range(4):
            base = math.atan2(math.sin(out[ci].angle_rad),
                              math.cos(out[ci].angle_rad))
            lo, hi = self.st_lo_c[ci], self.st_hi_c[ci]
            cand = [base + math.pi * n for n in range(-3, 4)
                    if lo - 1e-9 <= base + math.pi * n <= hi + 1e-9]
            if not cand:                      # 폭 < pi 인 극단: 경계에 붙인다
                cand = [min(max(base, lo), hi)]
            pick = min(cand, key=lambda t: _motor_tip_radius(self.cfg, ci, t))
            # 180도 등가로 넘어갔으면 바퀴 속도 부호를 뒤집어야 운동이 같다
            flipped = round((pick - base) / math.pi) % 2 != 0
            mi = C2MJ[ci]
            th[mi] = pick
            v[mi] = -out[ci].speed_mps if flipped else out[ci].speed_mps
            self.c_state[ci].last_cmd_rad = pick
        self.th_last = th.copy()
        return th, v
