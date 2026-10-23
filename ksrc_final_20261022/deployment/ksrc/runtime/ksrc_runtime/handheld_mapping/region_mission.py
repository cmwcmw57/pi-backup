# ================================================================
# region_mission.py
# 설명: 구역/통로 MPPI로30° 시도·실측 정체 후퇴·15° 우회·정상·외곽 복귀를 판단한다.
#       추정 pose와 stereo 관측만 받으며 GT·고정 경로 CSV·실기 명령은 사용하지 않는다.
# 사용법: RegionMission(mission).update(sample, now_s)
# ================================================================
from __future__ import annotations

from collections import deque
import json
import math
from pathlib import Path

import cv2
import numpy as np

from ksrc_runtime.localization.region_navigation import RegionGraph, RegionBelief

ASSETS = Path(__file__).parents[1]/'localization/assets'


class RegionMission:
    """추정·관측 기반 임무 전이와 지역 MPPI를 기존 속도 명령 인터페이스에 연결한다."""

    def __init__(self, mission, *, planner=None, safety=None, hold_s: float = 0., scenario: dict | None = None) -> None:
        self.mission, self.settings, self.safety = mission, mission.settings, safety
        self.scenario = scenario or {}   # 경기장별 구역·통로·단계 좌표(scenario.json). 없으면 옛 gyusa 경기장 값을 쓴다
        self.regions = self.scenario['regions'] if 'regions' in self.scenario else json.loads((ASSETS/'gyusa_regions.json').read_text())['regions']
        self.portals = self.scenario['portals'] if 'portals' in self.scenario else json.loads((ASSETS/'gyusa_region_portals.json').read_text())['portals']
        self.start_region = int(self.scenario.get('start_region', 1))
        self.graph = RegionGraph(self.regions, self.portals);self.belief = RegionBelief(self.graph, self.start_region)
        if self.graph.distance(self.start_region, mission.start[:2]) > 0:
            raise ValueError('region-mppi requires the start inside the start region of the region map')
        if planner is None:
            from ksrc_runtime.handheld_mapping.terrain_mppi import TerrainMppi
            planner = TerrainMppi(mission.site, max_speed=min(.15, self.settings['max_speed_m_s']),
                                  max_yaw_rate=min(.3, self.settings['max_yaw_rate_rad_s']),
                                  start_tolerance_m=self.settings.get('position_uncertainty_m', .15))
        self.planner = planner
        self.state, self.route, self.direction = 'trial_entry', '30', 'up'
        self.stage = 'entry30';self.hold_s = hold_s
        self.visits = dict(entry30='active', entry15='pending', entry10='pending', summit='pending', **{'return': 'pending'})
        self.route_status = {k: dict(up='unknown', down='unknown') for k in ('30', '15', '10')}
        self.events: list[dict] = [];self.blocked_edges: set[tuple[int, int]] = set()
        self.last_stamp = None;self.last_position = None;self.last_belief_stamp = -math.inf
        self.phase_elapsed = 0.;self.effort = 0.;self.progress = deque();self.breadcrumbs = []
        self.phase_start_s=0.;self.trial30_forward_s=0.;self.turn_target=0.;self.turn_settle=0.
        self.progress_anchor = self.progress_stamp = None;self.progress_effort = 0.
        self.plan_profile = None
        self.command = dict(vx_m_s=0., yaw_rate_rad_s=0.)
        self.observation = dict(stamp_s=-1., valid=False, reason='depth_pending', visible_regions=[], obstacles_xy=[])
        self.failure_evidence = {};self.last_plan = {};self.next_plan_s = 0.;self._metadata_sent = False
        self.sensor_status = None;self.paused = False;self.pose_epoch = None;self.blocked_since = None;self.obstacles_ignored = False
        self.plan_offset = np.zeros(2)   # 벽 측정(--wall-plan)으로 구한 계획용 위치 보정(월드 XY, 추정기는 그대로)
        self.corner_gate = False;self.front_obs = None;self.corner_hold = False;self.corner_wait = None;self.corner_logged = set()   # 모서리 전용 넘어가는 조건(--corner-gate)
        self.CORNER_GATE = ({k: dict(v) for k, v in self.scenario['corner_gate'].items()} if 'corner_gate' in self.scenario
                            else {'return9': dict(prev=(6.1, -.4), target=(6.1, -1.2)), 'return7': dict(prev=(5.25, -1.2), target=(2.9, -1.2))})
        s = mission.site;yy, xx = np.indices(s.observed.shape)
        xy = np.stack([s.origin[0]+(xx+.5)*s.resolution, s.origin[1]+(yy+.5)*s.resolution], axis=-1)
        self.region_masks = {}
        for k, polygon in self.graph.polygons.items():
            # polygon 경계는 통로 단면을 공유한다. 실제 collision 여유는 MPPI가 grid에서 검사한다.
            mask = np.zeros(s.observed.shape, np.uint8)
            cells = np.round((polygon-s.origin)/s.resolution-.5).astype(np.int32)
            cv2.fillPoly(mask, [cells], 1);self.region_masks[k] = mask.astype(bool)
        self._belief_output = dict(current_region=self.start_region, candidates=[self.start_region], probabilities={str(k): float(k == self.start_region) for k in self.graph.regions})

    def observe_depth(self, observation: dict) -> None:
        """독립 stereo 관측을 저장하며 전방 구역을 현재 구역으로 대입하지 않는다."""
        if float(observation['stamp_s']) >= self.observation['stamp_s']:
            self.observation = observation

    def observe_imu(self, stamp_s: float, now_s: float, valid: bool = True) -> None:
        """기존 IMU freshness 정책에 전달한다."""
        if self.safety is not None:
            self.safety.observe_imu(stamp_s, now_s, valid)

    def _front_ray(self, origin: np.ndarray, direction: np.ndarray, max_m: float = 3.) -> float | None:
        """origin에서 direction으로 처음 만나는 blocked 셀까지의 거리(1 cm 간격)."""
        s = self.mission.site
        for d in np.arange(0., max_m, .01):
            c = s.cells([origin+direction*d])[0]
            if not (0 <= c[0] < s.blocked.shape[1] and 0 <= c[1] < s.blocked.shape[0]):return None
            if s.blocked[c[1], c[0]]:return float(d)
        return None

    def _corner_arrived(self, stamp: float, distance: float, est_arrived: bool) -> tuple[bool, bool]:
        """모서리 전용 넘어가는 조건. (도착 여부, 직진 유지 여부)를 돌려준다."""
        info = self.CORNER_GATE[self.stage]
        if 'dir' not in info:
            target, prev = np.array(info['target'], float), np.array(info['prev'], float)
            direction = (target-prev)/np.linalg.norm(target-prev)
            expected = self._front_ray(target, direction)
            info.update(dir=direction, expected=expected if expected is not None else .4)
            info['extend'] = float(min(.2, max(0., info['expected']-.30)))
        expected, obs = info['expected'], self.front_obs
        fresh = obs is not None and 0. <= stamp-obs['stamp'] <= .6
        def note(action: str, **details) -> None:
            if (self.stage, action) not in self.corner_logged:
                self.corner_logged.add((self.stage, action));self._event('corner_gate', stamp, stage=self.stage, action=action, expected_m=expected, **details)
        if fresh and obs['d'] <= expected-.08 and distance <= .5:
            note('already_close', front_m=obs['d']);self.corner_wait = None;return True, False
        if not est_arrived:
            self.corner_wait = None;return False, False
        if fresh:
            if obs['d'] <= expected+.05:
                note('accepted', front_m=obs['d']);self.corner_wait = None;return True, False
            note('hold_far', front_m=obs['d'])
            return False, True
        if self.corner_wait is None:self.corner_wait = stamp
        if stamp-self.corner_wait >= 6.:
            note('timeout_accept');self.corner_wait = None;return True, False
        note('hold_unmeasured')
        return False, True

    def _event(self, kind: str, stamp: float, **details) -> None:
        """기존 recorder가 읽는 임무 이벤트 형식이다."""
        self.events.append(dict(type=kind, stamp_s=stamp, state=self.state, route=self.route, **details))

    def _phase(self, stage: str, state: str, stamp: float) -> None:
        """단계 전환 때 정체 창과 이전 궤적을 지운다."""
        self.stage, self.state = stage, state;self.phase_elapsed = self.effort = 0.
        self.phase_start_s=float(stamp);self.trial30_forward_s=0.
        self.progress.clear();self.next_plan_s = stamp;self.last_plan = {}
        self.progress_anchor = self.progress_stamp = None
        self._event('region_phase', stamp, stage=stage)

    def _begin_descent(self, stamp: float) -> None:
        """정상에서 바깥 고리 하행(return10…return1)을 시작한다."""
        self.visits['return'] = 'active';self.visits['entry10'] = 'active'
        self.route, self.direction = '10', 'down';self._phase('return10', 'descending', stamp)

    @staticmethod
    def _bearing(xy: np.ndarray, point) -> float:
        """xy에서 point를 향하는 방위각(rad)."""
        return math.atan2(float(point[1])-float(xy[1]), float(point[0])-float(xy[0]))

    def _start_turn(self, stage: str, heading: float, stamp: float) -> None:
        """제자리 회전 단계를 시작한다."""
        self._phase(stage, 'turning', stamp);self.turn_target, self.turn_settle = heading, 0.

    def _turn_hold(self, stamp: float, yaw: float, dt: float) -> dict:
        """정상 관찰 단계: 목표 방향으로 제자리 회전(느리게)하고, 도착하면 정지해 hold_s 동안 서 있다가 하행 방향으로 돌아선다."""
        look = self.scenario['look']
        if self.state == 'holding':
            self.command = dict(vx_m_s=0., yaw_rate_rad_s=0.)
            if self.phase_elapsed >= float(look.get('hold_s', 4.)):
                back = look.get('back_heading_rad')   # 관찰 뒤 돌아설 방향(없으면 하행 첫 목표 방향)
                self._start_turn('look_back', float(back) if back is not None else self._bearing(self.last_position, self.scenario['goals']['return10']['target']), stamp)
        else:
            error = (self.turn_target-yaw+math.pi) % (2*math.pi)-math.pi
            rate = float(look.get('turn_rate_rad_s', .1))
            if abs(error) <= math.radians(float(look.get('tol_deg', 3.))):
                self.turn_settle += dt;self.command = dict(vx_m_s=0., yaw_rate_rad_s=0.)
                if self.turn_settle >= .5:
                    if self.stage == 'look_face':self._phase('look_hold', 'holding', stamp)
                    elif 'look_exit' in self.scenario.get('goals', {}):self._phase('look_exit', 'to_look_exit', stamp)   # 복도를 똑바로 나온 뒤 하행
                    else:self._begin_descent(stamp)
            else:
                self.turn_settle = 0.
                self.command = dict(vx_m_s=0., yaw_rate_rad_s=math.copysign(min(rate, max(.03, 1.5*abs(error))), error))
        return self._output('active', dict(next_region=None, portal_segment_xy_m=None, target_xy_m=self.last_position.tolist(), guidance_valid=True,
                                          obstacles_ignored=False, selected_trajectory=[], candidate_trajectories=[]))

    def _fail_ascent(self, stamp: float, reason: str) -> None:
        """시도한 통로만 이번 임무에서 막고 지나온 구역으로 후퇴한다."""
        policy_stop=reason=='trial_budget'
        self.route_status[self.route]['up'] = 'aborted' if policy_stop else 'failed'
        edge = (1, 2) if self.route == '30' else (1, 3)
        self.blocked_edges.add(edge)
        self._event('ascent_aborted' if policy_stop else 'ascent_failed', stamp, reason=reason, evidence=self.failure_evidence.copy())
        self._phase('retreat'+self.route, 'retreating', stamp)

    def _portal_normal(self, source: int, destination: int) -> np.ndarray:
        """통로 선분에 수직이고 목적 구역을 향하는 단위 방향을 반환한다."""
        segment = self.graph.portals[tuple(sorted((source, destination)))]
        tangent = segment[1]-segment[0]
        normal = np.array([tangent[1], -tangent[0]])/np.linalg.norm(tangent)
        centers = self.graph.polygons[destination].mean(axis=0)-self.graph.polygons[source].mean(axis=0)
        return normal if normal@centers >= 0 else -normal

    def _ascent_exit(self) -> tuple[np.ndarray, np.ndarray, float]:
        """차체 여유까지 정상 통로를 지나도록 출구 목표와 도착 조건을 계산한다."""
        source = 2 if self.route == '30' else 3
        center = self.graph.portals[(source, 4)].mean(axis=0)
        normal = self._portal_normal(source, 4)
        clearance = self.mission.site.robot_radius+self.mission.site.resolution*math.sqrt(2)
        return center, normal, clearance

    def _goal(self) -> tuple[np.ndarray, list[int], tuple[int, int] | None, bool]:
        """목표는 통로/구역 영역이다. 기존 route CSV를 읽지 않는다."""
        goal = self.scenario.get('goals', {}).get(self.stage)
        if goal is not None:return np.array(goal['target'], float), list(goal['regions']), (tuple(goal['edge']) if goal.get('edge') else None), bool(goal.get('reverse', False))
        if self.stage == 'entry30':return np.array([3.8, .4]), [1], (1, 2), False
        if self.stage == 'ascent30':
            center, normal, clearance = self._ascent_exit()
            edge = (1, 2) if self._belief_output['current_region'] == 1 else (2, 4)
            return center+normal*(clearance+.14), [1, 2, 4], edge, False
        if self.stage == 'retreat30':return np.array([3.7, .4]), [1, 2], (2, 1), True
        if self.stage == 'entry15':return np.array([3.85, 1.2]), [1], (1, 3), False
        if self.stage == 'ascent15':
            center, normal, clearance = self._ascent_exit()
            edge = (1, 3) if self._belief_output['current_region'] == 1 else (3, 4)
            return center+normal*(clearance+.14), [1, 3, 4], edge, False
        if self.stage == 'retreat15':return np.array([3.85, 1.2]), [1, 3, 4], (3, 1), True
        if self.stage == 'summit':return np.asarray(self.mission.summit), [2, 4] if self.route == '30' else [3, 4], None, False
        goals = {'return10': ([6.1, -.4], [4, 10], (4, 10)),
                 'return9': ([6.1, -1.2], [10, 9], (10, 9)),
                 'return8': ([5.25, -1.2], [9, 8], (9, 8)),
                 'return7': ([2.9, -1.2], [8, 7], (8, 7)),
                 'return6': ([2.9, -.45], [7, 6], (7, 6)),
                 'return1': (self.mission.start[:2], [6, 1], (6, 1))}
        target, regions, edge = goals[self.stage]
        return np.array(target), regions, edge, False

    def _output(self, reason: str, navigation: dict | None = None) -> dict:
        """기존 viewer/recorder 임무 schema에 navigation 상태를 더한다."""
        nav = dict(mode='region-mppi', **self._belief_output, visible_regions=self.observation.get('visible_regions', []),
                   stage=self.stage, observation_reason=self.observation['reason'], failure_evidence=self.failure_evidence.copy(),
                   blocked_edges=[list(e) for e in sorted(self.blocked_edges)], **(navigation or {}))
        if not self._metadata_sent:
            nav.update(regions=self.regions, portals=self.portals);self._metadata_sent = True
        return dict(state=self.state, route=self.route, direction=self.direction, reason=reason,
                    visits=self.visits.copy(), route_status={k: v.copy() for k, v in self.route_status.items()},
                    command=self.command.copy(), frame='site', site_identity=self.mission.site.identity,
                    command_stamp_s=self.last_stamp, valid_until_s=None if self.last_stamp is None else self.last_stamp+.5,
                    hardware_commands_enabled=False, navigation=nav,
                    **({} if self.sensor_status is None else {'sensor_safety': self.sensor_status.copy()}))

    def _stop(self, reason: str, blocked: bool = False) -> dict:
        """센서/계획 실패는 등판 실패와 구분하고 명령을0으로 한다."""
        local_reason = self.last_plan.get('reason')
        if not blocked:self.blocked_since = None
        self.command = dict(vx_m_s=0., yaw_rate_rad_s=0.);self.progress.clear();self.paused = True
        self.next_plan_s = 0.;self.last_plan = {};self.plan_profile = None
        self.progress_anchor = self.progress_stamp = None
        return self._output(reason, dict(guidance_valid=False, local_plan_reason=local_reason))

    def update(self, sample: dict, now_s: float) -> dict:
        """추정 pose·기울기와 실제 명령 노력으로 단계를 판단한다."""
        if self.state in ('complete', 'failed'):return self._stop(self.state)
        try:
            stamp = float(sample['stamp_s']);xy = np.asarray(sample['position_m'], float)[:2];yaw = float(sample['yaw_rad'])
            pitch, roll = float(sample.get('pitch_rad', 0.)), float(sample.get('roll_rad', 0.))
            if not np.isfinite(np.r_[xy, yaw, stamp, now_s, pitch, roll]).all() or xy.shape != (2,):raise ValueError()
        except (KeyError, TypeError, ValueError, OverflowError):return self._stop('invalid_pose')
        xy = xy+self.plan_offset
        if sample.get('frame') != 'site' or sample.get('site_identity') != self.mission.site.identity:return self._stop('unaligned_frame')
        if stamp > now_s+.02 or now_s-stamp > .5:return self._stop('stale_pose')
        if self.last_stamp is not None and stamp <= self.last_stamp:return self._stop('nonmonotonic_pose')
        dt = 0. if self.last_stamp is None or self.paused else min(stamp-self.last_stamp, .1)
        self.last_stamp = stamp
        if sample.get('tracking_valid') is not True:return self._stop('tracking_lost')
        epoch = sample.get('pose_epoch', 0)
        if self.pose_epoch is None:self.pose_epoch = epoch
        if epoch != self.pose_epoch:return self._stop('pose_epoch_changed')
        if self.last_position is not None and np.linalg.norm(xy-self.last_position) > .5:return self._stop('pose_jump')
        self.last_position = xy.copy()
        if self.safety is not None:
            self.safety.observe_pose(stamp, now_s, True, sample.get('tracking_quality', 'good'))
            self.safety.observe_filter(sample.get('filter_status', 'ok'), stamp)
            self.sensor_status = self.safety.status(now_s)
            if self.sensor_status['mode'] == 'stop':return self._stop(self.sensor_status['reason'])
        if stamp < self.hold_s:return self._stop('start_hold')
        self.paused = False;self.phase_elapsed += dt
        if stamp-self.last_belief_stamp >= .2:
            likelihood = {k: math.exp(-.5*(self.graph.distance(k, xy)/.15)**2) for k in self.graph.regions}
            self._belief_output = self.belief.update(stamp, likelihood, xy);self.last_belief_stamp = stamp
        if not self.breadcrumbs or np.linalg.norm(xy-np.array(self.breadcrumbs[-1])) >= .025:
            self.breadcrumbs.append(xy.tolist())
        if self.state in ('turning', 'holding'):return self._turn_hold(stamp, yaw, dt)
        target, regions, edge, reverse = self._goal();distance = float(np.linalg.norm(xy-target))
        self.effort += max(0., self.command['vx_m_s'])*dt
        self.progress.append((stamp, xy.copy(), self.effort))
        window = max(3., float(self.settings.get('stuck_window_s', 8.)))
        while len(self.progress) >= 2 and stamp-self.progress[1][0] >= window:self.progress.popleft()
        if self.state == 'ascending':
            oldest, first_xy, first_effort = self.progress[0]
            normal = self._portal_normal(1, 2 if self.route == '30' else 3)
            # 옆 미끄러짐은 통로를 통과하는 등판 진행으로 세지 않는다.
            progress = max(0., max(float((p-first_xy)@normal) for _, p, _ in self.progress))
            motion = max(float(np.linalg.norm(p-first_xy)) for _, p, _ in self.progress)
            command_distance = self.effort-first_effort
            self.failure_evidence = dict(window_s=stamp-oldest, observed_progress_m=progress, observed_motion_m=motion,
                command_distance_m=command_distance, pitch_deg=math.degrees(pitch), roll_deg=math.degrees(roll),
                tracking_quality=sample.get('tracking_quality', 'good'))
            along = float(xy@normal)
            if (self.progress_anchor is None or self.command['vx_m_s'] <= 0.
                    or along-self.progress_anchor >= self.settings.get('min_progress_m', .025)):
                self.progress_anchor, self.progress_stamp, self.progress_effort = along, stamp, self.effort
            patience = max(window, float(self.settings.get('ascent15_no_progress_s', 30.))) if self.route == '15' else window
            no_progress = stamp-self.progress_stamp
            effort_since_progress = self.effort-self.progress_effort
            self.failure_evidence.update(no_progress_s=no_progress, no_progress_limit_s=patience,
                                         command_since_progress_m=effort_since_progress)
            stalled = no_progress >= patience-1e-6 and effort_since_progress >= .2
            tilted = abs(roll) > math.radians(30.) or abs(pitch) > math.radians(40.)
            policy_stop=False
            if self.route=='30':
                if self.command['vx_m_s']>0:self.trial30_forward_s+=dt
                forward_limit=float(self.settings.get('trial30_forward_limit_s',5.))
                elapsed_limit=float(self.settings.get('trial30_elapsed_limit_s',10.))
                elapsed=stamp-self.phase_start_s
                policy_stop=((forward_limit>0 and self.trial30_forward_s>=forward_limit-1e-6)
                             or (elapsed_limit>0 and elapsed>=elapsed_limit-1e-6))
                self.failure_evidence.update(trial30_forward_s=self.trial30_forward_s,trial30_elapsed_s=elapsed,
                                             trial30_forward_limit_s=forward_limit,trial30_elapsed_limit_s=elapsed_limit,
                                             policy_stop=bool(policy_stop and not tilted))
            if stalled or tilted or policy_stop:
                self._fail_ascent(stamp,'tilt_limit' if tilted else 'trial_budget' if policy_stop else 'no_progress')
                target, regions, edge, reverse = self._goal()
                distance = float(np.linalg.norm(xy-target))
        arrived = distance <= float(self.scenario.get('goals', {}).get(self.stage, {}).get('arrive_m', .14))   # 단계별 도착 반경(scenario goals의 arrive_m), 기본 0.14 m
        if self.state == 'ascending':
            center, normal, clearance = self._ascent_exit()
            arrived = arrived and self.graph.distance(4, xy) <= 1e-7 and (xy-center)@normal >= clearance-1e-6
        if self.corner_gate and self.state == 'descending' and self.stage in self.CORNER_GATE:
            arrived, self.corner_hold = self._corner_arrived(stamp, distance, arrived)
        else:self.corner_hold = False
        if arrived:
            if self.stage == 'entry30':
                self.visits['entry30'] = 'reached';self._phase('ascent30', 'ascending', stamp)
            elif self.stage == 'ascent30':
                self.route_status['30']['up'] = 'succeeded';self._phase('summit', 'to_summit', stamp)
            elif self.stage == 'retreat30':
                self.route = '15';self.visits['entry15'] = 'active';self._phase('entry15', 'to_entry15', stamp)
            elif self.stage == 'entry15':
                self.visits['entry15'] = 'reached';self._phase('ascent15', 'ascending', stamp)
            elif self.stage == 'ascent15':
                self.route_status['15']['up'] = 'succeeded';self._phase('summit', 'to_summit', stamp)
            elif self.stage == 'retreat15':
                self.state = 'failed';self._event('mission_failed', stamp, reason='ascent_routes_exhausted')
                return self._stop('ascent_routes_exhausted')
            elif self.stage == 'summit':
                self.visits['summit'] = 'reached'
                if self.settings.get('summit_look') and 'look' in self.scenario:self._phase('look_go', 'to_look', stamp)
                else:self._begin_descent(stamp)
            elif self.stage == 'look_go':
                self._start_turn('look_face', self._bearing(xy, self.scenario['look']['face']), stamp)
            elif self.stage == 'look_exit':
                self._begin_descent(stamp)
            elif self.stage == 'return1':
                self.visits['return'] = self.visits['entry10'] = 'reached';self.route_status['10']['down'] = 'succeeded'
                self.state = 'complete';self._event('mission_complete', stamp);return self._stop('complete')
            else:
                stages = ['return10', 'return9', 'return8', 'return7', 'return6', 'return1']
                self._phase(stages[stages.index(self.stage)+1], 'descending', stamp)
            if self.state not in ('turning', 'holding'):target, regions, edge, reverse = self._goal()
        if self.state in ('turning', 'holding'):return self._turn_hold(stamp, yaw, dt)
        if self.corner_hold:
            info = self.CORNER_GATE[self.stage];target = target+info['dir']*info['extend']   # 앞벽이 아직 멀면 목표를 조금 연장해 직진을 이어간다
        scale = 1. if self.sensor_status is None else self.sensor_status['scale']
        if sample.get('tracking_quality', 'good') != 'good':scale = min(scale, .3)
        if self.state == 'descending':scale *= float(self.settings.get('descent_speed_scale', 1.))
        minimum_speed = 0.  # 평지·등판 모두 MPPI가 정지부터 최대속도까지 선택한다.
        profile = (minimum_speed, scale)
        if stamp >= self.next_plan_s or profile != self.plan_profile:
            allowed = np.logical_or.reduce([self.region_masks[k] for k in regions])
            observation_fresh = stamp-self.observation['stamp_s'] <= 1.
            obstacles = (np.asarray(self.observation['obstacles_xy'], float).reshape(-1, 2)+self.plan_offset) if observation_fresh else None
            self.last_plan = self.planner.plan(xy, yaw, target, allowed,
                self.settings.get('trial_slope_deg', 35.) if self.route == '30' else self.settings.get('normal_slope_deg', 20.),
                reverse=reverse, obstacles_xy=obstacles, min_speed_m_s=minimum_speed, command_scale=scale)
            self.obstacles_ignored = False
            if not self.last_plan.get('valid') and obstacles is not None and len(obstacles):
                # 깊이 장애물은 추정 pose 오차로 경사를 장애물로 오인할 수 있다(v4·v9). 지도만으로 갈 수 있으면 지도를 따르고 기록한다.
                retry = self.planner.plan(xy, yaw, target, allowed, self.settings.get('trial_slope_deg', 35.) if self.route == '30'
                                          else self.settings.get('normal_slope_deg', 20.), reverse=reverse, obstacles_xy=None,
                                          min_speed_m_s=minimum_speed, command_scale=scale)
                if retry.get('valid'):
                    self.last_plan, self.obstacles_ignored = retry, True
            self.next_plan_s = stamp+.1;self.plan_profile = profile
        if not self.last_plan.get('valid'):
            # 계획 불가가 계속되면 영원히 정지하지 않고 원인과 함께 임무를 실패로 끝낸다.
            if self.blocked_since is None:self.blocked_since = stamp
            if stamp-self.blocked_since >= self.settings.get('blocked_timeout_s', 20.):
                self.state = 'failed';self._event('mission_failed', stamp, reason='local_plan_blocked_timeout',
                                                  local_reason=self.last_plan.get('reason'))
                return self._stop('local_plan_blocked_timeout', True)
            return self._stop('local_plan_blocked', True)
        self.blocked_since = None
        self.command = self.last_plan['command'].copy()
        if 'steep_pitch_deg' in self.scenario and self.state in ('ascending', 'retreating') and abs(pitch) > math.radians(float(self.scenario['steep_pitch_deg'])):
            self.command['yaw_rate_rad_s'] = 0.   # 급경사 위에서 조향을 돌리면 미끄러져 뒤집힌다(새 맵 시뮬 시드 4·9 전복). 직진·직후진만 한다
        portal = None if edge is None else self.graph.portals[tuple(sorted(edge))].tolist()
        return self._output('active', dict(next_region=None if edge is None else edge[1],
            portal_segment_xy_m=portal, target_xy_m=target.tolist(), guidance_valid=True,
            obstacles_ignored=self.obstacles_ignored, selected_trajectory=self.last_plan.get('selected_trajectory', []),
            candidate_trajectories=self.last_plan.get('candidate_trajectories', [])))
