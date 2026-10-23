"""KSRC 스워브 로버 Gymnasium 환경.

액션은 기본 차체 명령 위의 제한된 3차원 잔차다 (raw 바퀴 명령이 아님):
    cmd_applied = cmd_nom + action * (d_vx_max, d_vy_max, d_om_max)
    (theta_i, wheel_speed_i) = IK(cmd_applied)
IK 는 펌웨어에 실제로 올라가는 C 코드(firmware/common/swerve_kinematics.c)를
ctypes 로 호출한다.  파이썬으로 재구현하면 언젠가 반드시 어긋나고, 그러면
잔차 정책이 통째로 무의미해진다.

현재 시뮬의 저수준 구동은 wheel_speed_i 를 개루프 duty 로 변환한다.
"""
import sys, pathlib
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # sim/
import ksrc_common as kc

from config import RoverCfg, NoiseCfg, sample
from mjcf import build
from terramech import Terramechanics, PRESETS as TERRAIN_PRESETS, sample_terrain
from reward import (RewardCfg, VelRewardCfg, tracking_terms,
                    recovery_potential, velocity_terms)
import terrain as terr
import path as pth


def _wrap(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))

# MJCF 의 바퀴 순서와 펌웨어 C 의 모듈 순서가 다르다. 여기서 한 번만 맞춘다.
MJ_WHEELS = ("fl", "rl", "fr", "rr")          # mjcf.CORNERS 순서
C_ORDER   = ("fl", "fr", "rl", "rr")          # ksrc_common._make_modules 순서
C2MJ = [MJ_WHEELS.index(w) for w in C_ORDER]  # C 인덱스 -> MJCF 인덱스

# cmd_nom 3 + wheel_ref_applied_prev 4 + wheel_meas 4 + steer 4
# + roll/pitch 2 + gyro 3 + acc 3 + lookahead 2점 4 + e_y 1 + cos/sin(e_psi) 2
# + 선행 목표점 2 + 선행 거리 1 + prev_action 3  =  36
#
# e_y 와 cos/sin(e_psi) 를 **직접** 넣는다.  lookahead 점만 주고 네트워크가 유도하길
# 기대했는데, 2026 4WIS 논문은 추종오차를 명시적으로 넣는다.  lookahead 는 경로
# 모양을, e_y/e_psi 는 현재 오차를 알려주므로 중복이 아니라 보완이다.
# e_psi 는 차체 자세 관측·진단이고, 주행 보상은 실제 이동 방향을 평가한다.
# cos/sin 으로 넣는 것은 2*pi 불연속 제거 목적 (논문과 동일).
#
# prev_action 은 보상의 2차 차분(Delta^2 a) 때문에 필요하다.  정책이 자기 직전
# 행동을 모르면 매끄러움을 맞출 수가 없다.
OBS_PER_FRAME = 36
# 경로를 관측에 넣지 않으면 횡이탈을 **원리적으로** 관측할 수 없어 경로추종
# 보상이 학습 불가능하다 (기존 23차원에는 자기 위치도 경로도 없었다).
# 실기에서는 Nav2 local plan + TF 로 얻는다 -> 측위 오차를 노이즈로 섞는다.
# 특권 관측 (teacher 전용).  실기체에서는 절대 못 얻는 값들이다.
#   흙 8 + 로버 3 + 차체속도 3 + 슬립 4 + 침하 4 + 수직항력 4 + 배걸림 2 + 지형스캔 15
PRIV_DIM = 43
SCAN_X = (0.00, 0.25, 0.50, 0.75, 1.00)   # 전방 [m] (차체 기준)
SCAN_Y = (-0.35, 0.0, 0.35)               # 좌우 [m]
HISTORY = 50          # 1.0초. 흙 다짐 상태를 이력에서 추정하려면 0.2초로는
                      # 과도응답도 안 끝난다 (Lee 2020 은 2.0초, RMA 는 ~1.0초)
CTRL_HZ = 50


class RoverEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: RoverCfg = None, rew: RewardCfg = None,
                 difficulty: float = 0.0, episode_s: float = 20.0,
                 difficulty_min_frac: float = 0.6,
                 randomize: bool = True, arena_eval: bool = False,
                 eval_kind: str = "sand", privileged: bool = False,
                 noise: NoiseCfg = None, seed: int = None,
                 perturb: float = 0.0, paths=None, s0_frac=None,
                 spawn_max_slope_deg: float = None,
                 drive_align_gate_deg: float = 10.0,
                 soil_amp: float = None,
                 task: str = "path",
                 action_mode: str = "body",
                 velocity_obs: bool = False,
                 history_frames: int = HISTORY,
                 push_force_max: float = 0.0,
                 # 시각 전용 메시(CAD STL 13개).  **기본은 끈다** -- 켜면 워커당
                 # RSS 56MB -> 512MB 라 16워커에서 8.0GB(가용 11GB)를 먹고
                 # SubprocVecEnv 워커가 죽는다.  뷰어(view.py)만 켠다.
                 # 끄면 남는 geom 은 전부 group=3 / alpha=0.2 인 충돌체이고
                 # MuJoCo 뷰어는 group 3 을 기본으로 **안 그리므로** 로버가
                 # 화면에서 사라진다 (바닥만 보인다).
                 visual: bool = False,
                 # reset 마다 MjModel/MjData 를 **새로 만들지** 여부.
                 # 기본 True = 에피소드마다 로버 파라미터를 다시 뽑는다
                 # (도메인 랜덤화).  뷰어는 False 로 둬야 한다 -- passive
                 # viewer 는 처음 넘긴 MjData 객체를 계속 들고 있어서, 재생성
                 # 하면 화면의 로버가 **영원히 얼어붙는다** (env 는 새 d 를
                 # 돌리는데 뷰어는 옛 d 를 그린다).  지형·잡음·흙은 어차피
                 # self.m / self.d 에 **제자리로** 다시 쓰므로 False 여도
                 # 에피소드마다 바뀐다.  고정되는 건 cfg 샘플 하나뿐이다.
                 rebuild_on_reset: bool = True,
                 cmd_filter_tau: float = 0.28,
                 discount_gamma: float = 0.998,
                 vel_rew: VelRewardCfg = None):
        self.base_cfg = cfg or RoverCfg()
        self.rew = rew or RewardCfg()
        # "path"  : pure pursuit 를 따라가는 경로추종 (align/prog/gate 보상)
        # "cmdvel": 주어진 차체속도를 변형지형에서 달성하는 **밑단** 과제.
        #           경로 항은 계산하지 않는다 (코드는 아래 _reward 에 남겨둠).
        if task not in ("path", "cmdvel"):
            raise ValueError("task 는 'path' 또는 'cmdvel' 이어야 합니다")
        self.task = task
        # "body"      : [dvx, dvy, domega]              (3D, 기존)
        # "wheel"     : [dv_1..4]                        (4D, 트랙션 컨트롤만)
        # "body+wheel": [dvx, dvy, domega, dv_1..4]      (7D)
        # 바퀴별이 필요한 이유는 config.d_wheel_frac 주석 참고.
        # 액션 공간 3계열을 Isaac 과 공유한다 (sim2sim 계약).
        #   body / wheel / body+wheel : 기존 차체·바퀴속도 잔차
        #   res8   : IK 출력 8개(조향4+속도4)에 각각 잔차 (residual 8D)
        #   direct8: IK 를 쓰지 않고 조향4+속도4 를 정책이 직접 (pure 8D)
        _MODES = ("body", "wheel", "body+wheel", "res8", "direct8")
        if action_mode not in _MODES:
            raise ValueError(f"action_mode 는 {_MODES} 중 하나여야 합니다")
        self.action_mode = action_mode
        self.n_body_act = 3 if action_mode in ("body", "body+wheel") else 0
        self.n_wheel_act = 4 if action_mode in ("wheel", "body+wheel") else 0
        # res8/direct8 은 조향4+속도4 를 별도 블록으로 둔다 (잔차 또는 직접).
        self.n_steer_act = 4 if action_mode in ("res8", "direct8") else 0
        self.n_wspd_act  = 4 if action_mode in ("res8", "direct8") else 0
        # 진단 전용: MuJoCo 참 차체속도를 정책에 넣어 속도 관측 부족의 상한을 잰다.
        # 실기에는 독립적인 대지속도 추정기 검증 전까지 켜서는 안 된다.
        self.velocity_obs = bool(velocity_obs)
        if not isinstance(history_frames, int) or not 1 <= history_frames <= HISTORY:
            raise ValueError(f"history_frames 는 1..{HISTORY} 정수여야 합니다")
        self.history_frames = history_frames
        if not np.isfinite(push_force_max) or push_force_max < 0:
            raise ValueError("push_force_max 는 0 이상의 유한한 힘 [N] 이어야 합니다")
        self.push_force_max = float(push_force_max)
        self.visual = bool(visual)
        self.rebuild_on_reset = bool(rebuild_on_reset)
        self.push_body = None
        self.push_force = np.zeros(2)
        self.push_start = float("inf")
        self.push_end = float("inf")
        self.vel_rew = vel_rew or VelRewardCfg()
        # cmd_vel 스케줄: 2초마다 새 목표를 뽑고 1차 필터로 수렴시킨다.
        # 계단 명령을 그대로 주면 오차의 대부분이 플랜트 지연(잔차로 못 고침)이
        # 되어 학습 신호가 흐려진다.
        # ROS mujoco_bridge.filter_cmd 와 같은 alpha=dt/(tau+dt) 식이다.
        # 기본 tau=0.28 은 구 코드 alpha=1/(0.3*50) 와 정확히 같다.
        if not np.isfinite(cmd_filter_tau) or cmd_filter_tau < 0:
            raise ValueError("cmd_filter_tau 는 0 이상의 유한한 초 단위 값이어야 합니다")
        self.cmd_filter_tau = float(cmd_filter_tau)
        self.cmd_hold = int(2.0 * CTRL_HZ)
        self.cmd_a = (1.0 / CTRL_HZ) / (self.cmd_filter_tau + 1.0 / CTRL_HZ)
        self.cmd_tgt = np.zeros(3)
        self.cmd_cur = np.zeros(3)
        self._cmd_k = 0
        self.vel_err_hist = []
        if not 0.0 < discount_gamma <= 1.0 or not np.isfinite(discount_gamma):
            raise ValueError("discount_gamma 는 (0, 1] 범위여야 합니다")
        self.discount_gamma = float(discount_gamma)
        self.difficulty = difficulty
        if (not np.isfinite(difficulty_min_frac) or
                not 0 <= difficulty_min_frac <= 1):
            raise ValueError("difficulty_min_frac 는 [0,1] 범위여야 합니다")
        self.difficulty_min_frac = float(difficulty_min_frac)
        self.randomize = randomize
        # 모래 비대칭 장(soil_field)은 **이 RL 의 과제 본체**다 (terrain.soil_field
        # docstring 참고).  그런데 `--stage1` 은 randomize 를 통째로 끄면서 이것까지
        # 꺼버려서, s9/s10 은 "교란이 없는 세계" 에서 학습·평가됐다 (외란이 없으면
        # 최적 잔차가 a=0, 즉 순수 IK 가 정답이라 정책이 이길 수 없다).
        # soil_amp 를 명시하면 randomize 와 무관하게 이 항만 켠다.
        if soil_amp is not None and (not np.isfinite(soil_amp) or soil_amp < 0):
            raise ValueError("soil_amp 는 0 이상의 유한한 숫자여야 합니다")
        self.soil_amp_override = None if soil_amp is None else float(soil_amp)
        self.arena_eval = arena_eval          # True 면 실제 경기장 지형으로 평가
        self.eval_kind = eval_kind            # "sand"(규사 경사지형) / "rock"(착륙지)
        # 대회맵으로 **학습**할 때 쓴다: 거시 구조는 그대로 두고 미세 요철만
        # 매 에피소드 교란한다 (terrain.perturb 주석 참고).  평가 스크립트는
        # 0 으로 둬서 맵 원본 그대로 본다.
        self.perturb = float(perturb)
        # privileged=True 면 관측 뒤에 특권정보를 붙인다 (teacher 학습용).
        # student 는 앞의 OBS_PER_FRAME*HISTORY 만 본다 -- 잘라 쓰면 그대로 호환된다.
        self.privileged = privileged
        # 구동·센싱 노이즈. randomize=False 면 끈다 (검증 스크립트용).
        self.nz = noise or NoiseCfg()
        self.rng = np.random.default_rng(seed)
        # 외력은 별도 난수열이다. 스트레스 시험을 켠 것만으로 명령·흙·노이즈
        # 샘플까지 바뀌면 같은 시드 비교에서 외력 효과를 분리할 수 없다.
        self.push_rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), 0x4B535243]) if seed is not None else None)
        self.episode_steps = int(episode_s * CTRL_HZ)

        # 36D 기본 프레임에는 직전 3D 행동이 포함된다. 바퀴 잔차 모드에서는
        # 직전 행동이 4D/7D 이므로 프레임과 관측 공간도 함께 늘어야 한다.
        self.obs_per_frame = (OBS_PER_FRAME - 3 + self.n_body_act +
                              self.n_wheel_act + self.n_steer_act +
                              self.n_wspd_act + 3 * self.velocity_obs)
        n_obs = self.obs_per_frame * self.history_frames + (PRIV_DIM if privileged else 0)
        self.observation_space = spaces.Box(-np.inf, np.inf, (n_obs,), np.float32)
        # 3D 차체 잔차 [Delta_vx, Delta_vy, Delta_omega].  config 주석 참고.
        self.action_space = spaces.Box(
            -1.0, 1.0, (self.n_body_act + self.n_wheel_act +
                         self.n_steer_act + self.n_wspd_act,), np.float32)
        # 고정 경로 세트 (Stage 1).  None 이면 에피소드마다 새로 생성 (Stage 2).
        # 학습용과 평가용은 **반드시 다른 seed 로 만든 세트**여야 한다.
        self.path_set = paths
        # RViz/텔레옵 스폰 전용. 학습·평가의 경로 위 랜덤 시작 분포는 유지한다.
        # 고정 경로에서 시작하면 임의의 평탄 지점으로 옮길 수 없으므로 함께 쓰지 않는다.
        if spawn_max_slope_deg is not None and paths is not None:
            raise ValueError("spawn_max_slope_deg 와 paths 를 함께 사용할 수 없습니다")
        self.spawn_max_slope_deg = spawn_max_slope_deg
        # IK 아래의 공통 저수준 제어: 목표각이 한 주기에 60도 넘게 바뀌는
        # 큰 방향 전환에만 정렬 대기를 시작한다. 평소 작은 조향 오차마다
        # 멈추면 경로 완주율이 떨어진다. 학습·평가·ROS 가 같은 규칙을 쓴다.
        # drive_align_gate_deg 는 대기 해제 허용 오차, 0 은 기능 OFF.
        if drive_align_gate_deg is not None and (not np.isfinite(drive_align_gate_deg) or
                                                 drive_align_gate_deg < 0):
            raise ValueError("drive_align_gate_deg 는 0 이상의 유한한 숫자여야 합니다")
        self.drive_align_gate_rad = (None if not drive_align_gate_deg else
                                     float(np.radians(drive_align_gate_deg)))
        self.drive_align_trigger_rad = float(np.radians(60.0))
        # 경로 위 시작점.  None 이면 U(0, 0.6*total) 로 랜덤 (학습용 커버리지),
        # 숫자면 그 비율로 고정 (수동 재생·재현용).
        self.s0_frac = s0_frac
        self._build()

    # ------------------------------------------------------------------ 모델
    def _build(self):
        self.cfg = sample(self.base_cfg, self.rng) if self.randomize else self.base_cfg
        c = self.cfg
        xml = build(c, terrain="hfield", hfield_n=terr.HF_N,
                    hfield_size=(terr.ARENA_X / 2, terr.ARENA_Y / 2),
                    visual=self.visual)
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        self.n_sub = int(round((1.0 / CTRL_HZ) / self.m.opt.timestep))

        aid = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        jid = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.a_st = [aid(f"a_st_{w}") for w in MJ_WHEELS]
        self.a_wh = [aid(f"a_wh_{w}") for w in MJ_WHEELS]
        self.j_st = [self.m.jnt_qposadr[jid(f"st_{w}")] for w in MJ_WHEELS]
        self.j_wh = [self.m.jnt_dofadr[jid(f"wh_{w}")] for w in MJ_WHEELS]
        self.sid = {n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, n)
                    for n in ("s_quat", "s_gyro", "s_acc")}
        self.sadr = {k: self.m.sensor_adr[v] for k, v in self.sid.items()}
        self.ground_gid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self.push_body = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "chassis")

        # 펌웨어 C 에 넘길 모듈 좌표 (C 순서)
        self.c_modules = kc._make_modules(half_w=c.track / 2, half_l=c.axle_x)
        # 제자리 회전의 기하 한계.  요 오차를 무차원화할 기준이다
        # (omega_max = v_max / 모듈 최대반경).  v_max 는 아래에서 정해지므로
        # 여기서는 반경만 잡아둔다.
        self.mod_r = max(float(np.hypot(mm.x, mm.y)) for mm in self.c_modules)
        # 바퀴별 조향 가동범위 (config.steer_*_deg). C 순서는 fold 용, MJCF 순서는 clip 용.
        rng_c = [c.steer_range_rad(w) for w in C_ORDER]
        self.st_lo_c = [r[0] for r in rng_c]
        self.st_hi_c = [r[1] for r in rng_c]
        self.st_lo_mj = np.array([c.steer_range_rad(w)[0] for w in MJ_WHEELS])
        self.st_hi_mj = np.array([c.steer_range_rad(w)[1] for w in MJ_WHEELS])
        self.c_state = (kc.SwerveModuleState * 4)()
        self.v_max = c.mot_w_noload * c.wheel_r
        self.om_max = self.v_max / max(self.mod_r, 1e-9)

    # ------------------------------------------------------------------ 명령
    def _new_path(self):
        """에피소드용 기준경로와 **경로 위 시작점**을 정한다.

        `path_set` 이 주어지면 그 중 하나를 고르고, 아니면 새로 생성한다.

        시작점을 경로 위에서 랜덤화하는 이유: 스폰을 고정하면 경로 앞부분만 매
        에피소드 학습되고 뒷부분은 거기까지 도달한 에피소드에서만 학습된다
        (커버리지 불균형).  2026 4WIS 논문도 "random positions along the
        generated path" 로 초기화한다.
        """
        if self.path_set:
            # reset() 이 스폰 위치를 정할 때 이미 고른 경로를 그대로 쓴다.
            self.path = getattr(self, "_pending", None) or pth.clone(
                self.path_set[int(self.rng.integers(len(self.path_set)))])
            self.s0 = float(getattr(self, "_pending_s0", 0.0))
        else:
            self.path = self._pending_generated
            self.s0 = 0.0
        self.path_c = pth.clone(self.path)
        start_xy = self.d.qpos[:2].copy()
        self.path.seed(self.s0, start_xy)
        self.path_c.seed(self.s0, start_xy)
        self.s_path = self.s0
        self.s_est = self.s0
        self.e_y = 0.0
        self.e_psi = 0.0
        self.goal_dist = float(self.path.total - self.s0)

    def _yaw(self):
        w, x, y, z = self.d.qpos[3:7]
        return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def _project_truth(self):
        """**참값** 위치로 경로 진행도·횡이탈·완주를 갱신한다.

        보상과 종료 판정은 전부 여기서 나온 값을 쓴다.  관측만 센서로 얻을 수
        있으면 되고 보상은 특권정보를 써도 된다.  컨트롤러(추정치)와 섞어 쓰다가
        `prog` 에 측위 잡음의 시간미분이 들어간 적이 있다.
        """
        pos = self.d.qpos[:2].copy()
        # 진행도 증가는 **실제 변위로 상한**된다 (path.project 주석 참고).
        self.s_path, self.e_y, _ = self.path.project(pos)
        # 경로 기준 참 차체 기수오차. 보상·진단에 쓰고 관측은 추정치를 쓴다.
        # 기준은 **제어기가 요 명령을 내는 방향**(course_at) 이다. 접선을 쓰면
        # 꼭짓점 앞 선행 조향이 오차로 기록되어 보상이 옳은 행동을 벌한다.
        self.e_psi = _wrap(self._yaw()
                           - pth.course_at(self.path, self.s_path, self.pp_L))
        # 완주 = 종방향 잔여 + 목표점 실거리 둘 다.  잔여거리만 보면 경로를 크게
        # 벗어난 상태에서도 완주로 잡힌다 (직선 끝에서 1.02m 벗어난 점이 완주였다).
        self.goal_dist = float(np.linalg.norm(pos - self.path.P[-1]))
        if (self.path.total - self.s_path < 0.25) and (self.goal_dist < 0.25):
            self.goals += 1
            self.goal_hit = True

    def _update_cmd(self):
        """Nav2 컨트롤러가 하는 일을 흉내낸다: 기준경로 pure pursuit -> cmd_vel.

        **컨트롤러는 측위 추정치만 본다.**  실기에서는 map->odom TF 로 얻으므로
        측위 정확도가 그대로 실린다.  시뮬에서 참 위치를 쓰면 lookahead 를 극단적으로
        짧게 잡는 게 항상 유리하다고 나오는데(e_y 중앙 38.8 -> 4.0mm) 실기에서는
        잡음 30mm 에 lookahead 0.08m 면 발진한다.
        잡음은 저역통과(tau=0.2s) -- 50Hz 백색이면 조향이 지터를 따라 떨어
        scrub 이 0.012 -> 0.21 로 17배가 된다.
        """
        pos = self.d.qpos[:2].copy()
        yaw = self._yaw()
        if self.randomize:
            n, r = self.nz, self.rng
            self.pos_bias = np.clip(
                self.pos_bias + r.normal(0, n.wp_drift / CTRL_HZ, 2),
                -n.wp_drift_max, n.wp_drift_max)
            a = self.nz_a
            g = np.sqrt((2.0 - a) / a)
            self.pos_lp += a * (r.normal(0, n.wp_noise * g, 2) - self.pos_lp)
            self.yaw_lp += a * (float(r.normal(0, n.wp_yaw_noise * g)) - self.yaw_lp)
            pos_est = pos + self.pos_bias + self.pos_lp
            yaw_est = yaw + self.yaw_lp
        else:
            pos_est, yaw_est = pos, yaw
        self.s_est, self.e_y_est, _ = self.path_c.project(pos_est)
        self.e_psi_est = _wrap(
            yaw_est - pth.course_at(self.path_c, self.s_est, self.pp_L))
        cmd = pth.pursue(self.path_c, self.s_est, pos_est, yaw_est, self.v_cruise,
                         k_v=self.cfg.pp_k_v, L_min=self.cfg.pp_l_min,
                         L_max=self.cfg.pp_l_max)
        # 실현 가능하도록 스케일 -- IK 가 포화하면 완벽한 정책도 추종을 못 한다
        need = max(np.hypot(cmd[0] - cmd[2] * mm.y, cmd[1] + cmd[2] * mm.x)
                   for mm in self.c_modules)
        if need > self.v_max:
            cmd *= self.v_max / need
        # **nominal** 명령.  보상·판정의 기준이고 정책이 수정하지 않는다.
        self.cmd_nom = cmd
        # 관측용 lookahead 도 같은 추정치에서 나온다 (같은 map->odom TF).
        self.wp_b = pth.lookahead_body(self.path_c, self.s_est, pos_est, yaw_est)
        # 기존 0.30/0.80m 샘플은 경로 모양을, 이 점은 제어기의 실제
        # 선행 목표점을 나타낸다. 속도에 따라 L 이 달라도 정책이 구분한다.
        self.target_L = self.pp_L
        self.target_b = pth.lookahead_body(
            self.path_c, self.s_est, pos_est, yaw_est, dists=(self.target_L,))[0]

        if self.task == "cmdvel":
            # pure pursuit 의 출력을 **버리고** 랜덤 twist 로 교체한다.  위
            # 경로 계산은 관측 차원(36D)을 유지하기 위해 남겨두지만 아래에서
            # 0 으로 덮으므로 보상·명령 어디에도 들어가지 않는다.
            if self._cmd_k % self.cmd_hold == 0:
                self.cmd_tgt = self.sample_cmd()
            self._cmd_k += 1
            self.cmd_cur = self.cmd_cur + self.cmd_a * (self.cmd_tgt - self.cmd_cur)
            self.cmd_nom = self.cmd_cur.copy()
            # 경로 관측을 0 으로 막는다.  측위에 의존하는 특징을 남기면 밑단
            # 정책이 경로/측위에 결합되어 "상위 제어기와 분리" 가 깨진다.
            self.wp_b = np.zeros_like(self.wp_b)
            self.e_y_est = 0.0
            self.e_psi_est = 0.0
            self.target_b = np.zeros_like(self.target_b)
            self.target_L = 0.0

    # ------------------------------------------------------- cmdvel 과제
    def body_twist(self):
        """차체좌표 실제 (vx, vy, omega).  추종오차의 '실제' 쪽이다."""
        q = self.d.qpos[3:7]
        Rm = np.zeros(9); mujoco.mju_quat2Mat(Rm, q); Rm = Rm.reshape(3, 3)
        v_b = Rm.T @ self.d.qvel[:3]
        om_b = Rm.T @ self.d.qvel[3:6]
        return np.array([v_b[0], v_b[1], om_b[2]])

    def sample_cmd(self):
        """도달 가능한 랜덤 twist.

        IK 포화는 "명령 자체가 불가능" 한 경우를 만들어 추종오차 측정을
        오염시킨다 (정책도 못 고친다).  바퀴 요구속도를 v_max 의 90% 로
        공통 스케일해서 **항상 도달 가능한 명령만** 준다.
        """
        r = self.rng
        v = self.v_max
        c = np.array([r.uniform(-0.9, 0.9) * v,
                      r.uniform(-0.5, 0.5) * v,
                      r.uniform(-1.0, 1.0) * 0.8])
        need = max(np.hypot(c[0] - c[2] * mm.y, c[1] + c[2] * mm.x)
                   for mm in self.c_modules)
        if need > 0.9 * v:
            c *= 0.9 * v / need
        return c

    def _cmdvel_terms(self, a):
        """속도추종 보상.  오차는 v_max / omega_max 로 무차원화한다."""
        tw = self.body_twist()
        cmd = self.cmd_nom
        v_err = float(np.linalg.norm(tw[:2] - cmd[:2])) / max(self.v_max, 1e-9)
        om_err = abs(float(tw[2] - cmd[2])) / max(self.om_max, 1e-9)
        self.vel_err_hist.append((v_err, om_err))
        # 복구 지표는 힘이 작용하는 0.3초를 빼고, 힘이 끝난 뒤 1초만 센다.
        if self.push_force_max > 0 and self.push_end < self.k / CTRL_HZ <= self.push_end + 1.0:
            self.push_err_hist.append((v_err, om_err))
        return velocity_terms(v_err, om_err, a, self.a_prev, self.a_prev2,
                              self.vel_rew)

    def _ik_of(self, cmd):
        """cmd_vel -> (조향각[4], 바퀴속도 기준[4] m/s).  **펌웨어와 같은 C 코드.**

        duty 는 여기서 계산하지 않는다.  IK 의 출력은 `(theta_ref, v_wheel_ref)`
        까지이고 duty 는 그 아래 계층(지금은 개루프 비례, 나중에 속도 PI)의 일이다.
        모터·기어비·전압이 바뀌어도 기구학을 건드리지 않게 하려는 분리다.
        """
        out = kc.swerve_ik_compute(float(cmd[0]), float(cmd[1]), float(cmd[2]),
                                   self.c_modules, self.v_max, self.c_state)
        kc.swerve_fold_to_range(self.st_lo_c, self.st_hi_c, self.c_state, out,
                                unwind_rad=0.0, slow_mps=0.0)
        th = np.zeros(4); v = np.zeros(4)
        for ci, mi in enumerate(C2MJ):
            th[mi] = out[ci].angle_rad
            v[mi] = out[ci].speed_mps
        self.th_last = th.copy()
        return th, v

    def set_difficulty(self, d: float):
        """커리큘럼 콜백이 SubprocVecEnv.env_method 로 호출한다."""
        self.difficulty = float(np.clip(d, 0.0, 1.0))
        return self.difficulty

    # ------------------------------------------------------------ 구동/센싱 노이즈
    def _sample_noise(self):
        """에피소드마다 고정되는 것(바이어스·전압·지연)과 매 스텝 바뀌는 것을 나눈다."""
        n, r = self.nz, self.rng
        if not self.randomize:                    # 검증 스크립트는 무잡음
            self.volt = 1.0; self.gyro_b = np.zeros(3); self.acc_b = np.zeros(3)
            self.servo_a = 1.0; self.delay = 0
            self.act_buf = []
            self.wp_bias = np.zeros(2); self.pos_bias = np.zeros(2)
            self.pos_lp = np.zeros(2); self.yaw_lp = 0.0; self.nz_a = 1.0
            return
        self.volt = float(r.uniform(*n.volt_scale))
        self.gyro_b = r.uniform(*n.gyro_bias, size=3)
        self.acc_b = r.uniform(*n.acc_bias, size=3)
        tau = float(r.uniform(*n.servo_lag))
        self.servo_a = float(np.clip((1.0 / CTRL_HZ) / max(tau, 1e-3), 0.0, 1.0))
        self.delay = int(r.integers(n.act_delay[0], n.act_delay[1] + 1))
        self.act_buf = []
        self.wp_bias = np.zeros(2)          # (미사용, 호환용)
        self.pos_bias = np.zeros(2)         # 측위 드리프트 (랜덤워크)
        self.pos_lp = np.zeros(2); self.yaw_lp = 0.0      # 저역통과 잡음 상태
        self.nz_a = float(np.clip((1.0 / CTRL_HZ) / max(n.wp_tau, 1e-3), 0.0, 1.0))

    def _motor_scale(self, duty):
        """배터리 전압 강하.  DC 모터는 정지토크와 무부하속도가 **둘 다** 전압에
        비례한다.  부하가 클수록 더 떨어진다 (3S 11.1V 가 78A 피크에서 셀당
        3.2V 까지 간다)."""
        if not self.randomize:
            return 1.0
        load = float(np.mean(np.abs(duty)))
        return float(np.clip(self.volt * (1.0 - self.nz.volt_sag_k * load), 0.3, 1.0))


    def _wheel_speed_meas(self):
        """제어기가 보는 바퀴속도 [m/s].  **펌웨어가 엔코더로 읽는 것과 같다.**

        _frame() 의 관측용 계산과 같은 양자화·잡음을 쓴다 (실기에서 제어기와
        관측이 같은 엔코더를 본다).  단 호출마다 난수를 소비하므로 폐루프가
        켜졌을 때만 부른다 -- 개루프 경로의 시드 재현성을 지키기 위해서다.
        """
        c = self.cfg
        v = np.array([self.d.qvel[j] for j in self.j_wh]) * c.wheel_r
        if self.randomize:
            n, r = self.nz, self.rng
            tick = 2 * np.pi / c.ticks_per_rev * CTRL_HZ * c.wheel_r
            if n.enc_quant > 0:
                v = np.round(v / (tick * n.enc_quant)) * tick * n.enc_quant
            v = v * (1.0 + r.normal(0, n.enc_noise, 4))
        return v

    def _wheel_duty(self, v_ref):
        """바퀴속도 기준 -> duty.  전향보상 + PI (config.wheel_closed_loop 주석).

        kp=ki=0 이면 기존 개루프 du = v_ref/v_max 와 **정확히** 같다.
        """
        c = self.cfg
        ff = v_ref / self.v_max
        if not c.wheel_closed_loop:
            return np.clip(ff, -1.0, 1.0)
        e = (v_ref - self._wheel_speed_meas()) / self.v_max
        # 적분: 누설 + 상한.  포화 중에는 같은 방향으로 더 쌓지 않는다.
        leak = np.exp(-1.0 / (CTRL_HZ * max(c.wheel_pi_tau_i, 1e-6)))
        cand = leak * self.wheel_pi_i + e / CTRL_HZ
        raw = ff + c.wheel_pi_kp * e + c.wheel_pi_ki * cand
        sat = np.abs(raw) > 1.0
        self.wheel_pi_i = np.where(sat & (np.sign(e) == np.sign(raw)),
                                   self.wheel_pi_i, cand)
        lim = c.wheel_pi_i_max / max(c.wheel_pi_ki, 1e-9)
        self.wheel_pi_i = np.clip(self.wheel_pi_i, -lim, lim)
        du = ff + c.wheel_pi_kp * e + c.wheel_pi_ki * self.wheel_pi_i
        self.wheel_v_err += float(np.mean(np.abs(e)))
        return np.clip(du, -1.0, 1.0)

    # ------------------------------------------------------------------ 관측
    def _frame(self):
        c = self.cfg
        q = self.d.sensordata[self.sadr["s_quat"]:self.sadr["s_quat"] + 4]
        w, x, y, z = q
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        gyro = self.d.sensordata[self.sadr["s_gyro"]:self.sadr["s_gyro"] + 3]
        acc = self.d.sensordata[self.sadr["s_acc"]:self.sadr["s_acc"] + 3]
        wheel_meas = np.array([self.d.qvel[j] for j in self.j_wh]) * c.wheel_r
        steer = np.array([self.d.qpos[j] for j in self.j_st])
        # --- 센싱 노이즈 -----------------------------------------------------
        # 관측이 완벽하게 깨끗하면 이력만으로 다 풀려서 정책이 특권정보를 안 쓴다.
        # 실기체에서 못 없애는 것들만 넣는다.
        if self.randomize:
            n, r = self.nz, self.rng
            # 엔코더: 50Hz 에서는 양자화가 지배적이다 (8384 tick/rev).
            tick = 2 * np.pi / c.ticks_per_rev * CTRL_HZ * c.wheel_r
            if n.enc_quant > 0:
                wheel_meas = np.round(wheel_meas / (tick * n.enc_quant)) * tick * n.enc_quant
            wheel_meas = wheel_meas * (1.0 + r.normal(0, n.enc_noise, 4))
            roll += float(r.normal(0, n.rp_noise)); pitch += float(r.normal(0, n.rp_noise))
            gyro = gyro + self.gyro_b + r.normal(0, n.gyro_noise, 3)
            acc = acc + self.acc_b + r.normal(0, n.acc_noise, 3)
        # 고전 피드백 기준선이 **정책과 똑같은 센서값**을 쓸 수 있게 남긴다.
        # 이게 없으면 비교 대상이 참값을 쓰는 oracle 밖에 안 되고, "고전
        # 제어기로 충분한가" 라는 질문에 배포 가능한 답을 낼 수 없다.
        self.gyro_meas = np.asarray(gyro, dtype=float).copy()
        self.wheel_meas = np.asarray(wheel_meas, dtype=float).copy()
        # 물리 스케일로 정규화 -- 파라미터가 바뀌어도 분포가 안 흔들리게
        # wheel_ref_applied 는 **직전 제어주기에 실제 저수준으로 들어간** 값이다
        # (요청값이 아니라 지연 후 값).  그래야 v_ref(t-1) -> v_meas(t) 라는
        # 명령->반응 관계가 이력에 남아 슬립·동특성 추론에 쓸 수 있다.
        return np.concatenate([
            self.cmd_nom / np.array([self.v_max, self.v_max, 3.0]),
            self.wheel_ref_applied / self.v_max,
            wheel_meas / self.v_max,
            steer / np.pi,
            [roll / (np.pi / 2), pitch / (np.pi / 2)],
            gyro / 5.0, acc / 9.81,
            np.clip(self.wp_b.ravel() / 1.0, -2.0, 2.0),   # lookahead 2점 (m)
            [np.clip(self.e_y_est / 0.3, -3.0, 3.0)],      # 측위 추정치의 횡이탈
            [np.cos(self.e_psi_est), np.sin(self.e_psi_est)],  # 측위 추정 기수오차
            np.clip(self.target_b, -2.0, 2.0),           # 제어기 선행 목표점 (차체좌표)
            [self.target_L],                             # 경로상 선행 거리 (m)
            self.a_prev,                   # 직전 행동 (3/4/7D, action_mode 에 따라)
            *( [self.body_twist() / np.array([self.v_max, self.v_max, self.om_max])]
               if self.velocity_obs else [] ),
        ]).astype(np.float32)

    def _obs(self):
        o = np.concatenate(self.hist).astype(np.float32)
        if self.privileged:
            o = np.concatenate([o, self._privileged()]).astype(np.float32)
        return o

    # -------------------------------------------------------------- 특권 관측
    def _terrain_scan(self):
        """차체 기준 격자점의 지형 높이 (현재 차체 높이 대비).  D435 depth 로
        언젠가 student 에 줄 수도 있지만, 지금은 teacher 전용으로 둔다."""
        Z, ex, ey = self.hf_Z, self.hf_ex, self.hf_ey
        ny, nx = Z.shape
        yaw = self._yaw(); c, s_ = np.cos(yaw), np.sin(yaw)
        px, py, pz = self.d.qpos[0], self.d.qpos[1], self.d.qpos[2]
        gx = np.array([[c * a - s_ * b + px for b in SCAN_Y] for a in SCAN_X]).ravel()
        gy = np.array([[s_ * a + c * b + py for b in SCAN_Y] for a in SCAN_X]).ravel()
        j = np.clip(((gx + ex / 2) / ex * nx).astype(int), 0, nx - 1)
        i = np.clip(((gy + ey / 2) / ey * ny).astype(int), 0, ny - 1)
        h = Z[i, j] + self.hf_off - pz          # 차체보다 얼마나 높은가 [m]
        return np.clip(h / 0.30, -1.0, 1.0)     # ±300mm 로 정규화

    def _privileged(self):
        """실기체에서 못 얻는 값 전부.  teacher 만 본다."""
        p, c = self.terr_p, self.cfg
        soil = np.array([p.mu_max / 0.9, p.K / 0.3, p.alpha / 0.12, p.beta / 0.9,
                         p.c_r / 0.1, p.k_z / 13.0, p.z_max / 0.06, p.mu_lat])
        rover = np.array([c.m_base / 1.0, c.wheel_r / 0.045, c.pivot_z / 0.06])
        v_w = self.d.qvel[:3].copy()
        q = self.d.qpos[3:7]
        Rm = np.zeros(9); mujoco.mju_quat2Mat(Rm, q); Rm = Rm.reshape(3, 3)
        v_b = (Rm.T @ v_w) / self.v_max                    # 차체좌표 실속도
        slip = self.tm.slip.copy()                          # 바퀴별 슬립비
        sink = self.tm.z / max(self.tm.p.z_max, 1e-6)       # 바퀴별 침하
        nominal = self.m.body_mass.sum() * 9.81 / 4.0
        fn = np.clip(self.tm.Fn / max(nominal, 1e-6), 0.0, 3.0) / 3.0
        belly = np.array([float(self.n_belly > 0 and self.k > 0),
                          float(np.sum(self.tm.Fn < 0.25 * nominal)) / 4.0])
        return np.concatenate([soil, rover, v_b, slip, sink, fn, belly,
                               self._terrain_scan()]).astype(np.float32)

    # ------------------------------------------------------------------ 리셋
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.push_rng = np.random.default_rng(
                np.random.SeedSequence([int(seed), 0x4B535243]))
        if self.randomize and (self.rebuild_on_reset or not hasattr(self, "m")):
            self._build()
        # 난이도를 기본 U(0.6*d_max, d_max) 로 뽑는다. 평가에서는 하한을
        # 올려 최상위 난도만 따로 시험할 수 있다.
        # 고정하면 정책이 그 난이도에만 맞춰지고, 커리큘럼이 정체하면 어려운
        # 지형을 영영 못 본다.  범위로 주면 항상 일부는 어렵다.
        # arena_eval 이어도 흙 난이도는 **학습과 같은 분포**에서 뽑는다.
        # 예전엔 d_ep=difficulty 로 고정했는데, 그러면 흙이 제일 무른 쪽으로
        # 고정되어 지형 기하의 차이와 흙의 차이가 뒤섞인다 (실제로 실제 경기장
        # 평가에서 세 변형이 전부 무너져 비교가 불가능했다).
        # arena_eval 에서 d_ep 는 흙에만 영향을 준다 -- 지형은 항상 실제 STL.
        self.d_ep = float(self.rng.uniform(
            self.difficulty_min_frac * self.difficulty, self.difficulty))
        # 흙도 난이도에 연동한다. firm -> sand 로 보간되므로 고난이도 에피소드는
        # 실제로 모래가 된다 (예전 거부 샘플링은 모래만 골라서 버렸다).
        self.terr_p = (sample_terrain(self.rng, self.d_ep)
                       if self.randomize else TERRAIN_PRESETS["sand"])
        self._sample_noise()
        Z = (terr.load_arena(self.eval_kind) if self.arena_eval
             else terr.generate(self.d_ep, self.rng)[0])
        if self.arena_eval and self.perturb > 0:
            Z = terr.perturb(Z, *terr.eval_extent(self.eval_kind), self.rng,
                             amp=self.perturb * self.d_ep)
        # to_hfield 는 -Z.min() 을 돌려준다. MuJoCo 표면높이 = Z - Z.min() 이므로
        # 스폰 높이에 이 오프셋을 반드시 더해야 한다 (안 더하면 지형에 파묻힌다).
        # hfield 의 물리 범위는 학습(8x8)과 평가(실제 4x5)가 다르다. 런타임에 맞춘다.
        ex, ey = terr.eval_extent(self.eval_kind)
        # 높이맵의 격자 간격과 물리 크기는 반드시 일치해야 한다.  이전에는
        # cmdvel 평가에서 3.2x4.0m 규사맵을 8x8m 로 늘렸는데 경사각과
        # 스폰/경로 좌표가 모두 달라졌다. 학습은 8x8m 절차 규사지형을 쓰고,
        # arena_eval 은 원본 크기 그대로 별도 검증한다.
        use_arena = self.arena_eval
        hx, hy = ((ex / 2, ey / 2) if use_arena
                  else (terr.ARENA_X / 2, terr.ARENA_Y / 2))
        self.m.hfield_size[0, 0] = hx; self.m.hfield_size[0, 1] = hy
        z_off = terr.to_hfield(self.m, Z)

        # 스폰/목표는 **주행 가능 영역**에서만 뽑는다.  맵 중앙 고정 스폰은 실제
        # 규사 STL 에서 절벽 단차(한 셀에 231mm) 위에 로버를 놓아 리셋 직후
        # 전복시켰다 (전복률 90%). 주행 마스크는 cfg.max_slope_deg 이하의
        # 풋프린트 경사와 경사로 설명되지 않는 40mm 이하 단차로 정의한다.
        self.hf_Z = Z; self.hf_ex = 2 * hx; self.hf_ey = 2 * hy; self.hf_off = z_off
        # 경사 상한은 로버 견인력 예산에서 유도된다 (config.max_slope_deg 주석).
        self.drive, self.slope_deg = terr.drivable_mask(
            Z, self.hf_ex, self.hf_ey, max_slope_deg=self.cfg.max_slope_deg,
            max_step=self.cfg.max_step)
        self.route_drive = terr.center_clearance_mask(
            self.drive, self.hf_ex, self.hf_ey, self.cfg.route_clearance)
        if not self.route_drive.any():
            raise RuntimeError("로버 크기를 반영한 주행 가능 경로가 없습니다")
        spawn_drive = self.route_drive
        if self.spawn_max_slope_deg is not None:
            # 중심만 평탄하면 바퀴가 경사에 걸친 채 시작할 수 있다. 차체 방향에
            # 상관없이 바퀴 바깥쪽까지 들어가는 반경만큼 경사 경계를 비운다.
            clearance = float(np.hypot(self.cfg.axle_x + self.cfg.wheel_r,
                                        self.cfg.track / 2 + self.cfg.wheel_r))
            spawn_drive = terr.flat_spawn_mask(
                self.drive, self.slope_deg, self.hf_ex, self.hf_ey,
                self.spawn_max_slope_deg, clearance) & self.route_drive
            if not spawn_drive.any():
                raise RuntimeError("평탄한 스폰 후보가 없습니다: 지형 또는 경사 상한을 확인하세요")
        if self.task == "cmdvel" and not self.arena_eval:
            # 이 과제는 경로가 아닌 랜덤 차체속도를 추종한다. 지형 가장자리에서
            # 시작하면 잘못된 속도추종과 무관하게 8초 안에 경계를 벗어난다.
            # 8x8m 절차지형의 중앙에서 출발시켜 모든 방향의 명령에 공간을 준다.
            ny, nx = spawn_drive.shape
            xs = ((np.arange(nx) + 0.5) / nx - 0.5) * self.hf_ex
            ys = ((np.arange(ny) + 0.5) / ny - 0.5) * self.hf_ey
            for half_width in (0.9, 1.5):
                central = spawn_drive & (np.abs(xs)[None, :] <= half_width) & \
                          (np.abs(ys)[:, None] <= half_width)
                if central.any():
                    spawn_drive = central
                    break
        if self.path_set:
            # 경로 위 s0 에서 시작한다.  경로를 먼저 고르고 그 지점으로 스폰.
            self._pending_generated = None
            base = self.path_set[int(self.rng.integers(len(self.path_set)))]
            self._pending = pth.clone(base)
            s0 = (float(self.rng.uniform(0.0, 0.6 * self._pending.total))
                  if self.s0_frac is None
                  else float(self.s0_frac) * self._pending.total)
            spawn = self._pending.at(s0)
            self._pending_s0 = s0
        else:
            self._pending = None
            # 안전 마스크의 아주 작은 포켓은 출발점만 유효하고 0.5m 앞도
            # 막힐 수 있다(특히 rock). 실제로 갈 수 있는 경로를 찾은 뒤 스폰한다.
            for _ in range(100):
                spawn = terr.sample_drivable(spawn_drive, Z, self.hf_ex, self.hf_ey,
                                             self.rng, 1)[0]
                try:
                    self._pending_generated = pth.make_path(
                        self.route_drive, Z, self.hf_ex, self.hf_ey, self.rng, spawn,
                        slope_deg=self.slope_deg, d_ep=self.d_ep)
                    break
                except ValueError:
                    continue
            else:
                raise RuntimeError("안전한 스폰·경로를 찾지 못했습니다: 지형과 로버 크기를 확인하세요")
        z_top = terr.local_top(Z, self.hf_ex, self.hf_ey, spawn) + z_off

        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[0] = spawn[0]; self.d.qpos[1] = spawn[1]
        self.d.qpos[2] += z_top + 0.01
        # 자세는 뒤에서 경로 시작방향에 맞춰 넣는다 (_new_path 이후).
        self.d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.tm = Terramechanics(self.m, self.cfg.wheel_r, self.terr_p)
        self.c_state = (kc.SwerveModuleState * 4)()

        # 명령은 **목표점 추종**으로 만든다.  고정 명령을 쓰면 20초 에피소드에서
        # 6.4m 를 직진해 3x5m 지형 밖으로 나가버린다 (평지에서도 성공률이 15%
        # 밖에 안 나온 원인).  목표점 방식은 그걸 막으면서 동시에 nav2 가 실제로
        # 주는 "계속 변하는 속도 명령"을 재현한다.
        self.goals = 0
        self.v_cruise = self.rng.uniform(0.45, 1.0) * self.v_max
        # 제어기 선행거리.  에피소드 안에서 상수이고, 요 기준(course_at)·관측·
        # 보상·스폰 자세가 **전부 이 하나**를 공유한다.
        self.pp_L = pth.pursuit_distance(
            self.v_cruise, self.cfg.pp_k_v, self.cfg.pp_l_min, self.cfg.pp_l_max)
        self._new_path()
        # 스폰 자세는 **s0 지점에서 제어기가 요구할 방향** 기준이다 (경로
        # 시작방향이 아니다 -- 중간에서 스폰하면 그건 틀린 기준).  pursue 의 요
        # 명령과 같은 course_at 을 쓴다.  접선(heading_at)으로 스폰하면 꼭짓점
        # 근처에서 시작할 때 제어기 기준으로 이미 틀어진 자세가 된다.
        # 실기에서도 nav2 는 rotation_shim_controller 로 큰 초기 요 오차를 먼저
        # 없앤 뒤 경로추종에 들어가므로 +-30도 로 좁힌다.
        yaw0 = (float(self.rng.uniform(-np.pi, np.pi)) if self.task == "cmdvel" else
                pth.course_at(self.path, self.s0, self.pp_L) + float(
                    self.rng.uniform(-np.pi / 6, np.pi / 6)))
        self.d.qpos[3:7] = [np.cos(yaw0 / 2), 0.0, 0.0, np.sin(yaw0 / 2)]
        self.wp_b = np.zeros((2, 2))
        self.cmd = np.zeros(3)
        self.duty_ik = np.zeros(4)
        # 액션 차원은 action_mode 에 따라 3 / 4 / 7 이다 (보상의 2차 차분에 쓴다)
        n_act = (self.n_body_act + self.n_wheel_act +
                 self.n_steer_act + self.n_wspd_act)
        self.a_prev = np.zeros(n_act)
        self.a_prev2 = np.zeros(n_act)
        self.st_prev = np.array([self.d.qpos[j] for j in self.j_st])
        self.st_cmd = self.st_prev.copy()        # 서보 1차 지연 상태
        self.th_last = self.st_prev.copy()       # 등가각 선택의 기준 (직전 명령각)
        self.k = 0
        self.prog_hist = []
        self.goal_hit = False
        self.path_len = 0.0
        self.rsum = {}                           # 항별 누적 보상 (에피소드 단위)
        self.n_belly = 0; self.n_lifted = 0; self.n_blocked = 0
        self.e_sum = 0.0                     # 누적 전기에너지 [J]
        # **에피소드 전 구간 집계.**  예전엔 eval/train 이 종료 스텝의 순간값을
        # 모았다 (slip/sink/e_y/yaw/energy).  f_belly 같은 누적 비율과 의미가
        # 달라 한 표에 섞여 있었다.  평균·p90·최대를 전부 남긴다.
        self.acc = {k: [] for k in ("slip", "sink", "e_y", "yaw", "pw", "prog",
                                    "e_psi", "course",
                                    # cmdvel 과제의 **본 지표**.  무차원
                                    # (v_max / om_max 로 나눈 값) 이다.
                                    "v_err", "om_err")}
        self.n_duty_sat = 0.0; self.duty_lost = 0.0; self.duty_use = 0.0
        self.n_drive_gated = 0.0
        self.steer_err = 0.0
        self.wheel_pi_i = np.zeros(4)
        self.wheel_v_err = 0.0
        self.n_steer_sat = 0.0; self.steer_use = 0.0
        # 3D 잔차 진단: 명령 포화율 / 바퀴 desaturation 율 / 권한 사용률
        self.n_cmd_sat = 0.0; self.n_wheel_desat = 0.0
        self.a_use = 0.0; self.a_p95 = []
        self.wheel_use = 0.0
        self._cmd_k = 0
        self.cmd_tgt = np.zeros(3); self.cmd_cur = np.zeros(3)
        if self.push_force_max > 0:
            direction = float(self.push_rng.uniform(-np.pi, np.pi))
            magnitude = float(self.push_rng.uniform(0.5, 1.0) * self.push_force_max)
            self.push_force = magnitude * np.array([np.cos(direction), np.sin(direction)])
            latest = max(0.0, self.episode_steps / CTRL_HZ - 1.3)
            self.push_start = float(self.push_rng.uniform(min(0.5, latest), latest))
            self.push_end = self.push_start + 0.3
        else:
            self.push_force[:] = 0.0
            self.push_start = self.push_end = float("inf")
        self.vel_err_hist = []
        self.push_err_hist = []
        self.wheel_ref_applied = np.zeros(4)
        self.drive_alignment_pending = False
        self.cmd_nom = np.zeros(3)
        self.stuck_cause = "-"
        self.last_xy = self.d.qpos[:2].copy()
        self.x0 = self.d.qpos[:2].copy()

        for _ in range(int(0.3 / self.m.opt.timestep)):     # 정착
            self.tm.apply(self.d, self.m.opt.timestep); mujoco.mj_step(self.m, self.d)
        self.tm.reset()
        # **바퀴별 흙 차이.**  난이도에 비례해 비대칭을 키운다 (d=0 이면 균일).
        # 상관길이는 트랙폭(265mm)보다 조금 작게 둬서 좌우 바퀴가 서로 다른 값을
        # 밟고, 패치를 넘어갈 때 과도 외란이 생긴다.  **tm.reset() 이 mu_field 를
        # 지우므로 반드시 그 뒤에 넣어야 한다** (처음에 앞에 뒀다가 안 먹었다).
        if self.soil_amp_override is not None:
            soil_amp = self.soil_amp_override       # 명시값이 randomize 를 이긴다
        else:
            soil_amp = self.cfg.soil_amp if self.randomize else 0.0
        if soil_amp > 0:
            self.tm.mu_field = terr.soil_field(
                self.hf_ex, self.hf_ey, self.rng,
                amp=soil_amp * self.d_ep, lam=self.cfg.soil_lam)
            self.tm.f_ex, self.tm.f_ey = self.hf_ex, self.hf_ey
        # 정착으로 위치가 조금 움직이지만 project 가 단조로 흡수한다.
        # 여기서 _new_path 를 다시 부르면 위에서 자세를 맞춘 경로와 **다른**
        # 경로가 되어 초기 요 오차 정렬이 무의미해진다.
        self._project_truth()            # 참값 경로 상태
        self.route_dist_prev = self.path.distance(self.d.qpos[:2])
        self.last_xy = self.d.qpos[:2].copy()  # 정착 이동을 첫 행동의 이동으로 세지 않는다
        self.goal_hit = False            # 정착 중 완주로 잡히지 않게
        self._update_cmd()               # 첫 nominal 명령
        self.cmd = self.cmd_nom.copy()
        # 실제 서보 각에서 setpoint 궤적을 시작한다. IK 목표각이 크게 바뀌어도
        # 한 제어주기에 물리적인 서보 속도보다 멀리 명령하지 않는다.
        self.st_applied = np.array([self.d.qpos[j] for j in self.j_st])
        self.th_goal_prev = self.st_applied.copy()
        # 리셋 직후에는 아직 바퀴 명령을 인가하지 않았다. 첫 관측에 nominal
        # 속도를 넣으면 게이트가 실제로 막고 있는 동안 관측값이 거짓이 된다.
        self.wheel_ref_applied = np.zeros(4)
        # 리셋 시 물리 시간이 흐르지 않는다. 같은 정착 상태의 센서값을 한 번만
        # 읽어 패딩해야 이력이 가짜 독립 노이즈 표본으로 채워지지 않고, 이력 길이를
        # 바꿔도 이후 환경 난수열이 바뀌지 않는다.
        first_frame = self._frame()
        self.hist = [first_frame.copy() for _ in range(self.history_frames)]
        return self._obs(), {}

    # ------------------------------------------------------------------ 스텝
    def step(self, a):
        """제어 한 주기.  순서가 중요하다.

        예전 순서는 `진행도 저장 -> 진행도 갱신 -> 행동 -> 물리 -> 보상` 이어서
        행동 a_t 에 대해 **a_(t-1) 의 이동**을 보상했다 (실측: 스텝0 의 실제 진행
        0.204 가 스텝1 에 반환됐다).  게다가 e_y 는 물리 전, yaw 는 물리 후 값을
        써서 한 보상 안에 두 시점이 섞였다.

        바른 순서:
          1) 정책이 관측한 그 명령에 잔차를 적용한다 (th_ik/duty_ik 는 직전 스텝
             끝에서 계산된 것이고, 관측에 들어간 것과 **같은 값**이다)
          2) 물리 진행
          3) 물리 후 **참값**으로 경로 상태·보상·종료 계산
          4) 다음 명령과 관측 생성
        """
        a = np.clip(np.asarray(a, np.float64), -1, 1)
        c = self.cfg
        # **변화율 제한.**  크기 상한만으로는 정책이 한 스텝에 -1 -> +1 로
        # 튀는 것을 못 막는다.  실기에서 그게 깨진 사례가 arXiv:2504.02420
        # 이다 (config.d_rate_max 주석).  관측에 들어가는 a_prev 와 보상에
        # 쓰는 a 가 **실제로 적용된 값**이 되도록 자른 뒤에 쓴다.
        if c.d_rate_max is not None:
            r = float(c.d_rate_max)
            a = np.clip(a, self.a_prev - r, self.a_prev + r)
        self.route_dist_prev = self.path.distance(self.d.qpos[:2])

        # 1) **차체 수준 잔차**를 nominal 명령에 더한다.
        #    cmd_nom 은 pure pursuit 이 낸 것이고 보상·판정의 기준이다.
        #    cmd_applied 는 정책이 수정한 것이고 IK 로만 들어간다.
        # res8/direct8 은 조향4+속도4 블록만 쓴다 (차체·바퀴변조 잔차 없음).
        a_steer8 = a_wspd8 = None
        if self.n_steer_act:
            a_steer8 = a[:4]; a_wspd8 = a[4:8]
            a_body = a[:0]; a_wheel = a[:0]
        else:
            a_body = a[:self.n_body_act]
            a_wheel = a[self.n_body_act:]
        lim = np.array([c.d_vx_max, c.d_vy_max, c.d_om_max])
        cmd_app = (self.cmd_nom + a_body * lim if self.n_body_act
                   else self.cmd_nom.copy())
        # 실현 가능하도록 **공통 비율** 스케일.  바퀴별로 따로 자르면 body twist
        # 모양이 깨진다 (원래 요청한 vx:vy:omega 비율이 바뀐다).
        need = max(np.hypot(cmd_app[0] - cmd_app[2] * mm.y,
                            cmd_app[1] + cmd_app[2] * mm.x) for mm in self.c_modules)
        self.n_cmd_sat += float(need > self.v_max)
        if need > self.v_max:
            cmd_app = cmd_app * (self.v_max / need)
        self.cmd = cmd_app
        self.a_use += float(np.mean(np.abs(a)))           # 잔차 권한 사용률
        self.a_p95.append(float(np.max(np.abs(a))))

        # 2) IK -> 바퀴 조향각 / 속도 기준.
        #    direct8 은 IK 를 쓰지 않지만, cmd_nom(보상 기준)과 IK 출력을
        #    관측·베이스 비교용으로 계속 계산해 둔다 (정책 입력엔 안 들어감).
        th_ik, v_ref = self._ik_of(cmd_app)
        if self.action_mode == "direct8":
            # **pure 8D**: 정책이 조향각·바퀴속도를 직접 명령한다.
            #   조향 = a_steer8 * steer_hi (전체 가동범위)
            #   속도 = a_wspd8 * v_max
            th_ik = a_steer8 * self.st_hi_mj
            v_ref = a_wspd8 * self.v_max
            self.steer_use += float(np.mean(np.abs(a_steer8)))
        elif self.action_mode == "res8":
            # **residual 8D**: IK 출력에 잔차를 더한다 (a=0 이면 순수 IK).
            th_ik = th_ik + a_steer8 * c.d_steer_rad
            v_ref = v_ref * (1.0 + a_wspd8 * c.d_wheel_frac)
            self.steer_use += float(np.mean(np.abs(a_steer8)))
        # 2b) **바퀴별 잔차** (트랙션 컨트롤).  IK 출력에 비율로 곱한다.
        #     조향각은 건드리지 않는다 -- 조향은 위치제어로 이미 잘 따라가고,
        #     오차의 원인은 구동 슬립이다.
        #     c.d_wheel_cut_only 면 cut = max(0, a_i) 로 접어 |v_ref_i| 가
        #     커질 수 없게 한다.  기본은 양방향이다 -- 합 0 투영을 쓰는
        #     train_wheel_residual.py 와 "공통 증속이 가장 큰 레버" 라는
        #     사후 최선 측정 때문이다 (config.d_wheel_frac 주석 참고).
        if self.n_wheel_act:
            if c.d_wheel_cut_only:
                mod = -np.maximum(a_wheel, 0.0)
            else:
                mod = a_wheel
            v_ref = v_ref * (1.0 + mod * c.d_wheel_frac)
            self.wheel_use += float(np.mean(np.abs(mod)))
        du = self._wheel_duty(v_ref)        # 개루프 or 속도 PI (config 참고)
        self.n_wheel_desat += float(np.any(np.abs(v_ref) > self.v_max * 1.001))
        th = np.clip(th_ik, self.st_lo_mj, self.st_hi_mj)

        # 3) 구동 노이즈.  통신 지연은 **차체 명령**에 걸어야 맞지만 Stage 1 은 OFF.
        self.act_buf.append((th.copy(), du.copy()))
        th_a, du_a = self.act_buf[0] if len(self.act_buf) > self.delay else (th, du)
        if len(self.act_buf) > self.delay:
            self.act_buf.pop(0)
        # 지연된 바퀴속도와 짝이 되는 최종 목표각. 서보 지연·속도제한으로
        # 움직이는 중간 setpoint 와 비교하면 구동을 너무 일찍 허용한다.
        th_goal = th_a.copy()
        if self.drive_align_gate_rad is not None:
            # 조향 서보는 ±180도 기계적 제한이 있다. +179와 -179는 방향만
            # 보면 2도 차이지만 실제 서보 이동은 358도라 wrap 하면 위험하다.
            delta = th_goal - self.th_goal_prev
            if (np.max(np.abs(delta)) > self.drive_align_trigger_rad and
                    np.any(np.abs(du_a) > 1e-6)):
                self.drive_alignment_pending = True
            self.th_goal_prev = th_goal.copy()
        if self.randomize:
            err = th_a - self.st_cmd
            err = np.where(np.abs(err) < self.nz.servo_dead, 0.0, err)
            self.st_cmd = self.st_cmd + self.servo_a * err
            th_a = self.st_cmd
        # 기존 위치 액추에이터는 목표각을 즉시 바꾸고 감쇠가 거의 없어,
        # -90도 step 에서 -132도까지 지나치고 35.6 rad/s 로 튀었다.
        # setpoint 를 서보 정격속도만큼만 전진시켜 명령 자체도 연속으로 만든다.
        max_dth = c.steer_vmax / CTRL_HZ
        th_a = self.st_applied + np.clip(th_a - self.st_applied, -max_dth, max_dth)
        self.st_applied = th_a.copy()
        # **조향 정렬 오차.**  실기에서 조향 엔코더로 바로 재는 값이고,
        # 펌웨어의 drive_align_gate 가 이미 같은 양을 쓴다.  "지금이 조향
        # 재정렬 중인가" 를 시계 없이 판정할 수 있는 유일한 배포 가능 신호다
        # (cmdvel 시뮬은 명령 전환 시점을 알지만 실기 Nav2 명령은 연속이라
        # '명령이 바뀐 뒤 몇 초' 라는 게 정의되지 않는다).
        self.steer_err = float(np.max(np.abs(
            th_goal - np.array([self.d.qpos[j] for j in self.j_st]))))
        if self.drive_alignment_pending:
            # 큰 방향 전환 중 한 바퀴만 먼저 구동하면 의도하지 않은 yaw 가
            # 생긴다. 지연된 목표각에 네 바퀴가 모두 들어올 때까지 보류한다.
            actual = np.array([self.d.qpos[j] for j in self.j_st])
            err = th_goal - actual
            if np.max(np.abs(err)) <= self.drive_align_gate_rad:
                self.drive_alignment_pending = False
            else:
                self.n_drive_gated += float(np.any(np.abs(du_a) > 1e-6))
                du_a = np.zeros_like(du_a)
        vs = self._motor_scale(du_a)
        for i in range(4):
            self.m.dof_damping[self.j_wh[i]] = c.wheel_damping
        for i in range(4):
            self.d.ctrl[self.a_st[i]] = th_a[i]
            self.d.ctrl[self.a_wh[i]] = du_a[i] * vs
        # 실제로 저수준에 들어간 wheel ref (지연 후).  다음 관측에 쓴다.
        self.wheel_ref_applied = du_a * self.v_max

        # 2) 물리
        for _ in range(self.n_sub):
            self.tm.apply(self.d, self.m.opt.timestep)
            if self.push_start <= self.k / CTRL_HZ < self.push_end:
                self.d.xfrc_applied[self.push_body, :2] += self.push_force
            mujoco.mj_step(self.m, self.d)
        self.k += 1

        # 3) 물리 후 참값으로 경로 상태 -> 보상 -> 종료
        self.s_prev = self.s_path
        self.goal_hit = False
        self._project_truth()
        r, term, info = self._reward(a, du_a, vs)

        # 4) 다음 nominal 명령과 관측
        self.a_prev2 = self.a_prev
        self.a_prev = a
        self._update_cmd()
        self.hist.pop(0); self.hist.append(self._frame())
        trunc = self.k >= self.episode_steps
        return self._obs(), r, term, trunc, info

    # ------------------------------------------------------------- 고착 원인 분해
    # 셋 다 "바퀴는 도는데 안 나간다" 로 보이지만 원인도 대응도 다르다.
    #   배 걸림  차체가 지형에 얹혀 바퀴 하중 상실 -> 토크 조절로는 해결 안 됨
    #   막힘     바위에 물려 바퀴가 안 돔          -> 토크를 더 줘야 함
    #   모래고착  슬립-침하 되먹임                  -> 토크를 빼야 함
    # 실기체에선 수직항력/전류/침하를 못 재서 구분이 불가능하지만, 시뮬에서는
    # 전부 관측된다. **정책 입력에는 넣지 않고** 평가 리포트에만 쓴다.
    def _diagnose(self, w_ang, du):
        """(배 접촉, 뜬 바퀴 수, 막힘) 을 돌려준다.

        배 접촉과 '진짜 배 걸림' 은 다르다. 차체가 지형을 스치기만 하면 바퀴
        하중은 그대로라 주행에 지장이 없다. 문제는 차체가 **하중을 받아 바퀴가
        뜨는** 경우다. 그래서 접촉 여부와 별개로 수직항력이 떨어진 바퀴 수를 센다.
        """
        c = self.cfg
        belly = False
        for i in range(self.d.ncon):
            con = self.d.contact[i]
            if self.ground_gid not in (con.geom1, con.geom2):
                continue
            other = con.geom2 if con.geom1 == self.ground_gid else con.geom1
            if other not in self.tm.wg:          # 타이어가 아닌 것이 지면에 닿음
                belly = True; break
        nominal = self.m.body_mass.sum() * 9.81 / 4.0
        lifted = int(np.sum(self.tm.Fn < 0.25 * nominal))
        v_meas = np.abs(w_ang) * c.wheel_r
        v_cmd = np.abs(du) * self.v_max
        act = v_cmd > 0.15 * self.v_max          # 의미있는 명령을 받은 바퀴만
        blocked = bool(act.any() and
                       np.mean(v_meas[act] < 0.15 * v_cmd[act]) > 0.5)
        return belly, lifted, blocked

    # ------------------------------------------------------------------ 보상
    def _reward(self, a, du, vs=1.0):
        R, c = self.rew, self.cfg
        dt_ctrl = 1.0 / CTRL_HZ
        den = max(self.v_cruise * dt_ctrl, 1e-6)

        # --- 경로추종: 논문의 진행·정렬·횡오차·방향·매끄러움 항 ------------
        # task="cmdvel" 에서는 **이 블록 전체를 건너뛴다**.  지우지 않고 남기는
        # 이유: 경로추종으로 되돌아올 수 있고, s13/s19 등 기존 체크포인트의
        # 보상 공식을 재현해야 할 수 있다.
        # prog: 경로 호길이 진행.  실제 변위로 상한돼 정지 상태에서 0 이다.
        #       음수는 자른다.
        prog = float(np.clip((self.s_path - self.s_prev) / den, 0.0, 1.0))
        # 실제 이동 방향오차는 진단/옛 보상 비교용이다. 현재 보상은
        # 논문처럼 차체 기수오차를 정렬 게이트에 쓰며 이동 방향을 벌주지 않는다.
        delta_xy = self.d.qpos[:2] - self.last_xy
        travel = float(np.linalg.norm(delta_xy))
        # 2 cm/s 이하에서는 수치적 위치 떨림의 방향이 실제 진행 방향을
        # 대표하지 않으므로 각도 평가를 건너뛴다.
        if travel > 0.02 * dt_ctrl:
            # 여기만 접선(heading_at)을 쓴다.  "한 제어주기 변위(~4mm)가 경로
            # 방향과 맞나" 를 묻는 국소 질문이라 26cm 현으로 채점하면 안 된다.
            # 진단 전용이다 (w_course = 0).
            path_heading = pth.heading_at(self.path, 0.5 * (self.s_prev + self.s_path))
            course_error = _wrap(float(np.arctan2(delta_xy[1], delta_xy[0]))
                                 - path_heading)
        else:
            course_error = 0.0
        movement = float(np.clip(travel / den, 0.0, 1.0))
        route_dist_now = self.path.distance(self.d.qpos[:2])
        gate, terms = tracking_terms(
            prog, self.e_y, self.e_psi, a, self.a_prev, self.a_prev2, R,
            course_error=course_error, movement=movement,
            lateral_recovery=recovery_potential(
                self.route_dist_prev, route_dist_now, den, self.discount_gamma))
        align = prog * gate
        goal_bonus = R.r_goal if self.goal_hit else 0.0

        if self.task == "cmdvel":
            # 경로 항을 **전부 버리고** 속도추종으로 교체한다.
            #   보상 = w_v*exp(-(e_v/sigma_v)^2)
            #        + w_om*exp(-(e_om/sigma_om)^2) - 행동벌점
            # 경로·lookahead·align/prog/gate 가 보상에서 사라지므로, pure
            # pursuit 튜닝(lookahead)과 경로 생성기 기하(꺾임각 분포)가 더 이상
            # 결과를 교란하지 않는다.
            terms = self._cmdvel_terms(a)
            prog = align = gate = 0.0
            goal_bonus = 0.0

        # --- 진단만 (가중치 0.  Stage 2 에서 켠다) --------------------------
        w_ang = np.array([self.d.qvel[j] for j in self.j_wh])
        slip = float(np.abs(self.tm.slip).mean())
        sink = float(self.tm.z.mean() / self.tm.p.z_max)
        tau = du * vs * c.mot_tau_stall - c.wheel_damping * w_ang
        p_mech = np.maximum(tau * w_ang, 0.0)
        p_cu = tau ** 2 * c.mot_w_noload / c.mot_tau_stall
        self.p_elec = float(np.sum(p_mech + p_cu))
        P_REF = c.mot_tau_stall * c.mot_w_noload
        energy = float(np.clip(self.p_elec / P_REF, 0.0, 1.0))
        TAU_RATED = 0.49
        over = float(np.mean(np.clip(np.abs(tau) - TAU_RATED, 0, None)) /
                     (c.mot_tau_stall - TAU_RATED))
        st = np.array([self.d.qpos[j] for j in self.j_st])
        dst = np.abs(st - self.st_prev) * CTRL_HZ
        self.st_prev = st
        stopped = np.clip(1 - np.abs(w_ang * c.wheel_r) / self.v_max, 0, 1)
        scrub = float(np.mean(np.clip(dst / c.steer_vmax, 0, 1) * stopped))
        belly, lifted, blocked = self._diagnose(w_ang, du)
        self.n_belly += belly
        self.n_lifted += (belly and lifted >= 1)
        self.n_blocked += blocked
        self.e_sum += self.p_elec / CTRL_HZ

        terms["goal"] = goal_bonus
        rew = float(sum(terms.values()))

        # 종료 조건
        # hist[-1] 은 행동 전 관측이라 한 제어주기 늦다. 물리 직후 센서 참값을 쓴다.
        w, x, y, z = self.d.sensordata[self.sadr["s_quat"]:self.sadr["s_quat"] + 4]
        roll = float(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
        pitch = float(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
        tip = abs(roll) > np.radians(60) or abs(pitch) > np.radians(60)
        # 누적 경로길이. 출발점으로부터의 거리로 재면 목표를 찍고 되돌아올 때
        # 값이 줄어들어 "정상 주행 중인데 고착" 으로 오판한다.
        self.path_len += float(np.linalg.norm(self.d.qpos[:2] - self.last_xy))
        self.last_xy = self.d.qpos[:2].copy()
        self.prog_hist.append(self.path_len)
        stuck = False
        if len(self.prog_hist) > 2 * CTRL_HZ:                  # 최근 2초
            stuck = (self.prog_hist[-1] - self.prog_hist[-2 * CTRL_HZ]) < 0.02 and slip > 0.5
        # 벽이 물리적으로 막으므로 이건 안전망일 뿐 (벽을 넘었거나 떨어진 경우)
        # hfield 물리 범위와 **같은 기준**을 쓴다 (reset 의 use_arena 참고).
        # 예전엔 여기서 eval_extent 를 다시 읽어서, cmdvel 처럼 물리 범위를
        # 넓힌 경우 판정 범위가 안 맞았다.
        oob = bool(abs(self.d.qpos[0]) > self.hf_ex / 2 - 0.1
                   or abs(self.d.qpos[1]) > self.hf_ey / 2 - 0.1
                   or self.d.qpos[2] < -0.5)
        # 종료/시간만료의 다음 상태 잠재값은 0. 이 보정이 없으면 목표 도착 시점에
        # 남은 경로 실거리에 따라 shaping 총점이 달라지고 조기 종료를 악용할 수 있다.
        if self.task == "path" and (tip or stuck or oob or self.goal_hit
                                    or self.k >= self.episode_steps):
            correction = R.w_recover * self.discount_gamma * route_dist_now / den
            terms["recover"] += correction
            rew += correction
        if stuck:
            # 귀속 순서가 곧 우선순위다. 배가 하중을 받아 바퀴가 뜬 상태가
            # 가장 치명적이고(토크 조절로 해결 불가), 그다음이 바퀴 스톨,
            # 나머지는 전부 슬립 고착이다 -- 고착 판정 자체가 slip>0.5 를
            # 요구하므로 "셋 중 아무것도 아님" 은 정의상 슬립 고착이다.
            # (예전엔 침하 임계로 갈랐는데, 침하는 시정수 2초로 쌓이고 고착
            #  판정도 2초라 대부분이 "기타" 로 빠졌다 -- 전체의 30%.)
            self.stuck_cause = ("배걸림-하중상실" if (belly and lifted >= 1) else
                                "배걸림-접촉"     if belly else
                                "막힘"            if blocked else
                                "슬립고착")
        if tip:   rew += R.r_tip;   terms["tip"] = R.r_tip
        if stuck: rew += R.r_stuck; terms["stuck"] = R.r_stuck
        # oob 에 벌점이 **없었다** -> 경계 밖으로 나가는 게 공짜 탈출이었다.
        if oob:   rew += R.r_oob;   terms["oob"] = R.r_oob
        for kk, vv in terms.items():
            self.rsum[kk] = self.rsum.get(kk, 0.0) + vv
        A = self.acc
        A["slip"].append(slip); A["sink"].append(sink); A["e_y"].append(abs(self.e_y))
        if self.vel_err_hist:
            ve, oe = self.vel_err_hist[-1]
            A["v_err"].append(ve); A["om_err"].append(oe)
        A["yaw"].append(max(0.0, float(np.cos(self.e_psi))))
        A["pw"].append(self.p_elec); A["prog"].append(prog)
        A["e_psi"].append(abs(np.degrees(self.e_psi)))
        if movement > 0.1:
            A["course"].append(abs(np.degrees(course_error)))
        # 경로를 완주하면 거기서 끝난다 -- 다음 목표를 이어붙이면 "경로를
        # 따라갔나" 라는 판정이 흐려진다.
        # cmdvel 에서는 경로 완주가 과제가 아니므로 goal_hit 으로 끝내지 않는다
        # (랜덤 twist 로 돌다가 우연히 목표 근처를 지나면 에피소드가 잘린다).
        goal_term = self.goal_hit and self.task == "path"
        term = bool(tip or stuck or oob or goal_term)
        # 성공 = **지정된 경로를 끝까지 따라갔나**.  예전 기준(간 거리 / 명령속도로
        # 갈 수 있었던 거리 >= 0.35)은 두 군데가 틀렸다:
        #   - 0.35 는 너무 관대했다 (3분의 1만 가도 성공)
        #   - path_len 이 누적 경로길이라 **헤매는 것에 보상**이 됐다
        # 경로 호길이 진행률은 둘 다 없다.  헤매면 s_path 가 안 늘어난다.
        frac = float(np.clip((self.s_path - self.s0) /
                             max(self.path.total - self.s0, 1e-6), 0.0, 1.0))
        # 완주 판정은 goal_hit 을 그대로 쓴다.  frac >= 0.9 로 재면 goal_hit 이
        # "남은 거리 0.25m" 에서 뜨므로 frac = 1 - 0.25/total 이 되어 **경로 길이가
        # 성공을 결정**한다 (2.15m 경로 -> 0.885 실패, 3.13m -> 0.920 성공).
        completed = bool(self.goal_hit and not (tip or stuck or oob))
        k = max(self.k, 1)
        info = dict(prog=prog, align=align, gate=gate,
                    course_error_deg=float(np.degrees(abs(course_error))),
                    e_y=float(abs(self.e_y)), e_psi=float(np.degrees(abs(self.e_psi))),
                    slip=slip, sink=sink, energy=energy,
                    over=over, scrub=scrub,
                    f_belly=self.n_belly / k, f_lifted=self.n_lifted / k,
                    f_blocked=self.n_blocked / k, cause=self.stuck_cause,
                    duty_sat=self.n_duty_sat / k, duty_lost=self.duty_lost / k,
                    duty_use=self.duty_use / k,
                    drive_gate=self.n_drive_gated / k,
                    # 3D 잔차 진단.  계속 제한에 박혀 있으면 action bound 가 부족하다는 증거
                    cmd_sat=self.n_cmd_sat / k, wheel_desat=self.n_wheel_desat / k,
                    a_use=self.a_use / k, wheel_use=self.wheel_use / k,
                    a_p95=float(np.percentile(self.a_p95, 95)) if self.a_p95 else 0.0,
                    t_elapsed=float(self.k / CTRL_HZ),
                    steer_sat=self.n_steer_sat / k, steer_use=self.steer_use / k,
                    tip=tip, stuck=stuck, oob=oob,
                    push_v_err_m=(float(np.mean([x[0] for x in self.push_err_hist]))
                                  if self.push_err_hist else float("nan")),
                    push_om_err_m=(float(np.mean([x[1] for x in self.push_err_hist]))
                                   if self.push_err_hist else float("nan")),
                    # 에피소드 전 구간 집계 (평균 / 90분위 / 최대)
                    **{f"{k}_{st}": float(f(v)) for k, v in self.acc.items()
                       if v for st, f in (("m", np.mean),
                                          ("p90", lambda x: np.percentile(x, 90)),
                                          ("max", np.max))},
                    goals=self.goals, dist=self.path_len, frac=float(frac),
                    # 배터리에 중요한 것은 순간 전력이 아니라 **이동거리당 소비**다.
                    # 느리게 가면 순간 전력은 낮아도 미터당은 더 나쁠 수 있다.
                    # 물리 단위.  e_J 는 누적 전기에너지, e_per_m 은 **완주한 경로
                    # 호길이당** 소비다 (헤맨 거리를 분모에 넣으면 헤매는 게 유리해진다).
                    # e_J 는 **구동 모터의 기계일 + 구리손** 추정치다.  무부하 전류,
                    # 기어 손실, 드라이버 손실, 조향 서보, 전장 소비가 빠져 있어
                    # 배터리 지속시간 예측에는 못 쓴다 (모델 내부 비교용).
                    e_J=float(self.e_sum),
                    e_per_m=(float(self.e_sum / (self.s_path - self.s0))
                             if self.s_path - self.s0 > 0.3 else float("nan")),
                    goal_dist=float(self.goal_dist),
                    **{f"rterm_{kk}": vv for kk, vv in self.rsum.items()},
                    completed=completed, success=completed,
                    route_m=float(self.path.total))
        return float(rew), term, info
