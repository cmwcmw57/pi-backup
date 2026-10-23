# ================================================================
# blend_follower.py
# 설명: 몸 헤딩 기준 ψ_ref를 느리게 따라가고 병진은 몸 좌표 (vx, vy)로 내는 swerve 혼합 추종 법칙(순수 계산, 지도·임무 의존 없음).
# 사용법:
#   f = BlendFollower(BlendParams()); out = f.step(yaw=yaw, travel_heading=phi, speed=.08, slope_deg=0., dt=.02)
#   정지·일시정지 때는 f.invalidate()로 다음 step에서 ψ_ref와 병진 방향을 다시 잡는다.
# ================================================================
from __future__ import annotations

import math
from dataclasses import dataclass


def _wrap(value: float) -> float:
    return math.atan2(math.sin(value),math.cos(value))


@dataclass(frozen=True)
class BlendParams:
    """혼합 추종기의 각속도 상한·이득·램프 게이트·병진 방향 슬루 설정."""
    max_yaw_rate: float = .09
    yaw_gain: float = 2.
    ramp_slope_deg: float = 8.
    resume_deg: float = 15.
    stop_deg: float = 25.
    dir_rate: float = 1.
    min_dir_speed: float = .01
    heading_hold: bool = False
    realign_stop_deg: float = 90.     # 몸 좌표 병진 방향 오차가 이를 넘으면 정지하고 방향부터 재정렬
    realign_resume_deg: float = 45.   # 재정렬 중 오차가 이하가 되면 출발


@dataclass(frozen=True)
class BlendOutput:
    """한 제어 주기의 몸 좌표 명령과 텔레메트리용 헤딩·게이트 값."""
    vx: float
    vy: float
    yaw_rate: float
    psi_ref: float
    heading_gap_rad: float
    travel_gap_rad: float
    gated: bool
    realigning: bool = False


class BlendFollower:
    """ψ_ref 슬루, 램프 게이트, 병진 방향 슬루를 상태로 들고 매 주기 (vx, vy, yaw_rate)를 계산한다."""

    def __init__(self, params: BlendParams=BlendParams()) -> None:
        self.params=params;self.invalidate()

    def invalidate(self) -> None:
        """다음 step에서 ψ_ref=현재 yaw, 병진 방향=목표 방향으로 다시 시작한다."""
        self.initialized=False;self.psi_ref=0.;self.theta_app=0.;self.gated=False;self.realigning=False

    def step(self, *, yaw: float, travel_heading: float, speed: float, slope_deg: float, dt: float,
             retreat: bool=False) -> BlendOutput:
        """세계 방향 travel_heading으로 speed만큼 병진하는 명령을 yaw 기준 몸 좌표로 만든다."""
        p=self.params;theta=_wrap(travel_heading-yaw)
        if not self.initialized:
            self.initialized=True;self.psi_ref=yaw;self.theta_app=theta;fresh=True
        else:fresh=False
        if retreat:self.psi_ref=yaw
        elif not fresh and not p.heading_hold:
            limit=p.max_yaw_rate*dt;self.psi_ref=_wrap(self.psi_ref+max(-limit,min(limit,_wrap(travel_heading-self.psi_ref))))
        heading_gap=_wrap(self.psi_ref-yaw)
        yaw_rate=0. if retreat else max(-p.max_yaw_rate,min(p.max_yaw_rate,p.yaw_gain*heading_gap))
        gap=abs(theta)
        if retreat or slope_deg<p.ramp_slope_deg:self.gated=False
        else:self.gated=gap>(p.resume_deg if self.gated else p.stop_deg)*math.pi/180.
        error=abs(_wrap(theta-self.theta_app))   # 병진 방향 오차가 크면(180° 전환·후퇴) 옆걸음 대신 정지-재정렬-출발
        self.realigning=not fresh and speed>=p.min_dir_speed and error>(p.realign_resume_deg if self.realigning else p.realign_stop_deg)*math.pi/180.
        speed_eff=0. if self.gated or self.realigning else max(0.,speed)
        if self.realigning or (speed_eff>=p.min_dir_speed and not fresh):
            limit=p.dir_rate*dt;self.theta_app=_wrap(self.theta_app+max(-limit,min(limit,_wrap(theta-self.theta_app))))
        return BlendOutput(vx=speed_eff*math.cos(self.theta_app),vy=speed_eff*math.sin(self.theta_app),yaw_rate=yaw_rate,
                           psi_ref=self.psi_ref,heading_gap_rad=abs(heading_gap),travel_gap_rad=gap,gated=self.gated,realigning=self.realigning)
