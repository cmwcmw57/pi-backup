"""Isaac Lab manager-based velocity-tracking task for the KSRC rover.

This is the first flat-ground bring-up configuration. Terrain, actuator, and
Isaac↔MuJoCo baseline calibration must pass before comparative RL training.
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as lab_mdp

from ksrc_rover import mdp
from ksrc_rover.assets.rover_cfg import ROVER_CFG
from ksrc_rover.utils.contract import MODULE_NAMES

STEER_JOINTS = [f"{name}_steering_joint" for name in MODULE_NAMES]
WHEEL_JOINTS = [f"{name}_wheel_joint" for name in MODULE_NAMES]

# ============================ 실험 설정 ============================
# 여기만 고치고 저장 → `bash deploy.sh` → 학습 재실행. (환경변수 안 씀)
TERRAIN_KIND = "rough_slope"   # plane/flat/slope/rough_flat/rough_slope/mixed
TERRAIN_MODE = "wide"          # off/narrow/wide  (wide: 경사 0–20°, 요철 ≤15mm). 2026-10-11 S6 채택
# 2026-10-11 S6 액션비교 설정: WHEEL_MU custom, MU_RANGE (0.1,0.6), MU_PER wheel, MU_RESAMPLE_S 1.0,
#   MASS_SCALE (0.8,1.5), COM 없음, 명령 8s 고정·에피소드 8s. 재현하려면 아래를 그 값으로.
WHEEL_MU     = "patch"         # off/narrow/wide (동료 헬퍼: 좌우페어·μ0.2–0.6·리셋만) / custom (아래 MU_*) / patch (MU_PATCH)
WHEEL_SEED   = 4321
MU_RANGE     = (0.1, 0.6)      # custom: 바퀴 정지마찰 μ 균등 범위
MU_PER       = "wheel"         # custom: global(4바퀴 같음)/pair(좌우 독립)/wheel(4바퀴 독립)
MU_RESAMPLE_S = 1.0            # custom: >0 이면 에피소드 중 이 주기[s]로 재추첨 (0=리셋 때만)
# patch: 에피소드마다 지형 기준 μ, 바퀴별 배율, 1–5s(환경별 무작위)마다 배율 재추첨 + 미끄러운 패치.
MU_PATCH = dict(base_range=(0.25, 0.8),     # 지형 기준 μ (리셋 때만)
                factor_range=(0.75, 1.25),  # 바퀴별 배율
                uniform_prob=0.2,           # 이 확률로 그 에피소드는 4바퀴 같은 μ
                slick_prob=0.5,             # 재추첨마다 이 확률로 한 바퀴·한쪽·대각·앞뒤·전부(10-12 추가)가 (10-11: 0.25->0.5, 보정 학습 신호↑)
                slick_range=(0.1, 0.2),     #   이 μ 로 (헐거운 모래·미끄러운 패치 대용)
                mu_clip=(0.08, 1.0))
MU_PATCH_INTERVAL_S = (1.0, 5.0)
MASS_SCALE   = (0.95, 1.2)     # base_link 질량 배율 범위 (startup, 관성 같이 배율). (1,1)=끔
COM_RANGE_M  = (0.02, 0.02, 0.01)  # base_link 무게중심 이동 ±x,y,z [m] (startup). 0=끔
# 명령: Nav2 처럼 주행 중 바뀌는 명령을 겪게 한다. (8,8)/8.0 이면 예전 "에피소드당 1회".
# 주의: Isaac Lab error_vel_* 지표는 (최대 재샘플 시간)으로 나눠 에피소드 내내 누적하므로
#   재샘플 < 에피소드면 m/s 평균이 아니다(여기선 ×16/6≈2.7). 정확한 값은 eval_analyze 로.
CMD_RESAMPLE_S = (2.0, 6.0)
EPISODE_S    = 16.0
# DR 단계. 1 = μ·지형·질량·명령(위). 2 = + 모터·서보·지연·센서노이즈(아래 DR2). 관측 크기는 같아서
# 1단계 teacher 체크포인트를 2단계로 이어 학습(--resume) 할 수 있다.
DR_STAGE     = 2             # 2026-10-11: 본학습 teacher 3종은 처음부터 DR 1–8 전부 (사용자 결정)
DR2 = dict(
    voltage_scale=(0.85, 1.05),   # 배터리: 정지토크·무부하속도 같이 배율
    motor_spread=0.05,            # 모터 개체차 ±
    damping_scale=(0.8, 1.2),     # 개루프 duty 기울기(정지토크/무부하속도 ≈0.25 N·m/(rad/s))
    armature_range=(8.3e-3, 11.3e-3),  # 바퀴 반사관성 [kg m²]. 10-13 실측 τ 40 ms → 9.8e-3 ±15% (전 1–3e-3)
    drive_friction_scale=(0.75, 1.25), # 바퀴별 기어 마찰 배율 (DRIVE_MODEL="pi" 만). 정지마찰 기동 35–52 duty
    servo_rate_scale=(0.85, 1.05),     # STS3215 프로파일(가속 7.7 rad/s²·최고속) 배율. 띄운 상태 개체차 <1%, 지면(S4) 미측정
    servo_stiffness_scale=(0.8, 1.2),
    servo_offset_deg=0.6,              # 바퀴별 조향각 오차 ± (조향 링크 놀음 ~1 mm, 10-13)
    servo_deadband_deg=(0.15, 0.35),   # 이 안의 목표 변화는 무시 (실측 최소 명령 0.3–0.5°, 히스테리시스 0.37°)
    servo_delay_s=(0.030, 0.038),      # 목표 쓰기 → 움직임 시작 (실측 33–34.5 ms). SPI 지연(action_delay)과 별개
    action_delay=(0, 1),          # [정책 스텝] 명령은 다음 SPI 교환·Nucleo 틱부터
    obs_delay=(1, 2),             # [정책 스텝] 텔레메트리는 직전 50Hz 주기에 만든 것
    gyro_bias=0.01,               # [rad/s] 에피소드마다 고정 바이어스 ±
)
# 넓힌 DR (2026-10-13): --set DR_WIDE=True 이면 위 MU_PATCH·MASS_SCALE·COM_RANGE_M·DR2 대신 이 값.
# 실측 중심값은 그대로 두고 폭만 넓힌다 (지면 조향·배터리·적재·SPI 주기는 아직 미측정).
DR_WIDE = False
MU_PATCH_WIDE = dict(MU_PATCH, base_range=(0.2, 0.9), factor_range=(0.65, 1.35),
                     slick_range=(0.06, 0.25), mu_clip=(0.05, 1.0))
MASS_SCALE_WIDE = (0.9, 1.35)
COM_RANGE_M_WIDE = (0.03, 0.03, 0.015)
DR2_WIDE = dict(DR2, voltage_scale=(0.8, 1.1), motor_spread=0.10, damping_scale=(0.7, 1.3),
                armature_range=(7.4e-3, 12.3e-3),        # 9.8e-3 ±25%
                drive_friction_scale=(0.6, 1.4),
                servo_rate_scale=(0.75, 1.1), servo_stiffness_scale=(0.7, 1.3),
                servo_offset_deg=1.0, servo_deadband_deg=(0.1, 0.5), servo_delay_s=(0.026, 0.045),
                gyro_bias=0.02)
# 센서 노이즈 σ (DR_STAGE>=2 에서만). 엔코더 8384 tick/rev @50Hz → 1 tick = 0.037 rad/s.
NOISE = dict(gyro=0.01, gravity=0.02, steer_pos=0.01, steer_vel=0.08, wheel_vel=0.04)
# student 는 위 대신 실기와 같은 계산 (2026-10-13): 바퀴 속도 = 엔코더 틱 20 ms 차분, 조향 = 서보 4096 격자,
# 중력 = 20 ms 평균 IMU(자세·속도 변화로 계산, mdp/dr.py _GravityImuSrc)를 Pi 와 같은 보완 필터로.  아래는 그 입력에 넣는 잡음 (DR_STAGE>=2).
STUDENT_NOISE = dict(imu_gyro=0.01, imu_acc=0.05, steer_pos=0.003, steer_vel=0.08, wheel_vel=0.0)
# 중력 관측 오차 DR (2026-10-13, mdp/dr.py randomize_gravity_error). 실기 필터 오차가 시뮬(평균 0.8°)보다 클 수 있어서:
# 에피소드마다 수평축 무작위로 고정 기울기 0..tilt_deg (exact 면 정확히 tilt_deg) + 느린 표류 0..drift_deg_s, 합 ≤ max_deg.
# 실측(IMU 장착각·주행 중 필터 오차, Pi–STM 측정일) 후 다시 정한다.
GRAV_ERR = dict(tilt_deg=3.0, drift_deg_s=0.3, max_deg=5.0, exact=False)
GRAV_ERR_STUDENT = True       # student 중력 관측에 (DR_STAGE>=2)
GRAV_ERR_POLICY = False       # teacher 'policy' 그룹 중력에도 -- 민감도 평가용 (--set GRAV_ERR_POLICY=True)
# student 입력 기록 길이 [정책 스텝]. MLP student 5(0.1s)/25(0.5s), GRU student 는 1.
STUDENT_HISTORY = 5
# 리셋 생성 높이 [m] (base_link, 타일 중심 지면 기준). URDF: 바퀴 바닥 = base_link −0.1618.
# 0.15 였을 땐 평지서도 1.2cm 박혀 리셋 직후 튕겨 ~50% 전복(2026-10-10 진단).
# 0.21 = 0.1618 + 경사여유 0.175·tan10°(0.031) + 0.01 + 요철(narrow). wide(20°) 는 0.24.
SPAWN_Z      = 0.24
TERRAIN_OVERRIDES = {}         # 동료 로더 geometry override (예: {"slope_deg": 30.0, "slope_heading_deg": 0.0}). 진단용
# 보상 실험: tight std(목표수렴) + 선형항 ON/OFF 2암 비교. LINEAR만 바꿔 재배포.
STD_V        = 0.05           # track_v exp std [m/s] (보상 B: 도달가능 스케일)
STD_OM       = 0.20           # track_om exp std [rad/s]
LINEAR       = False          # 덴스 선형항. 보상 B = exp-only (선형 불필요)
# 양수 선형 추종 보상 (가중치, e_max). None=끔. 10-12: 미끄러운 곳(오차>0.1)에서 exp 신호가 0 이라
#   teacher 가 거기서 IK 보다 나빠짐 → 오차 e_max 까지 기울기 일정. 음수 벌점형(LINEAR)은 넘어짐 종료를 부추김.
#   10-12 결과: 끔. 짝비교(seed 0·1·2)에서 넓은 Res8 보다 조향반전 2배·미끄러운 바퀴 회전오차 악화.
#   0.1 넘는 오차의 2/3 는 명령 직후 과도·네 바퀴 미끄러움이라 정책이 못 줄이는 구간이었음.
LIN_TRI_V    = None           # 시험값 (0.3, 0.5) 직선 [m/s]
LIN_TRI_OM   = None           # 시험값 (0.2, 1.5) 회전 [rad/s]
SLIP_W       = 0.5            # 슬립(바퀴 헛돎) 페널티 가중치 (0=끔). 모래 파묻힘·좌우불균형 억제
# 바퀴별 슬립 속도² 벌점 (가중치, 데드존 m/s). None=끔. 10-12: SLIP_W(비율 평균)는 미끄러운 바퀴를
#   거의 구분 못 함(IK 0.265 vs 0.157) → 넓은 Res8 이 μ 를 보면서도 바퀴별 보정을 안 함. 이 항은 0.071 vs 0.017.
#   켤 때 SLIP_W=0 으로 같이 끈다 (예: --set SLIP_SQ="(2.5, 0.03)" --set SLIP_W=0).
SLIP_SQ      = None
# 구동 damping 덮어쓰기 (None = rover_cfg 기본 0.246). 10-12 전 런(0.20 으로 학습)을 같은 조건으로 평가할 때
#   --set DRIVE_DAMPING=0.2
DRIVE_MODEL  = "isaac"         # 바퀴 구동: isaac (DCMotor 가 속도 목표 추종, 마찰 없음) / pi (펌웨어 WPI) /
                               #   openloop (펌웨어 WPI OFF duty). pi·openloop 는 duty 모터 + 기어 마찰 (10-13)
DRIVE_GATE   = True           # 펌웨어 구동 정렬 게이트 (조향 60° 넘게 바뀌면 10° 안 들 때까지 바퀴 0). 실기와 같게 켬 (10-13 사용자).
                               #   10-13 이전 런(n1499 등)과 같은 조건으로 비교·평가할 때만 --set DRIVE_GATE=False
DRIVE_DAMPING = None
# 매끄러움(SLAM 카메라 보호). 0=끔. 세 액션공간 공통: 액션이 만든 관절 명령(조향각/한계, 바퀴속도/최고속)에 건다.
# RATE_* 중 하나라도 켜면 기존 통짜 action_rate(-0.01) 는 끄고 조향/바퀴 분리판으로 대체.
# 가중치 근거(1500iter Direct8 eval): 조향 1차 0.64 / 2차 2.45, 바퀴 1차 0.11 / 2차 0.34 (스텝당 sum²)
#   → 현 정책 기준 페널티 합 ≈0.11 (추종 보상 최대 1.0 대비 ~10%). 통짜 action_rate 는 0.0075 뿐이었음.
# 10-11: ×30 (0.05/0.02/0.03/0.01 → 아래). ±10°/±30% Res8 teacher iter1000 이 IK 보다 나쁨
# (xy +18%, 조향떨림 4.3배, 조향반전 6/s/바퀴): 차분이 ±90°·최고속으로 정규화돼 벌점 ≈ −0.001 로 무시됐음.
RATE_STEER_W = 1.5            # 조향 명령 1차 차분² (변화량)
RATE_WHEEL_W = 0.6            # 바퀴 명령 1차 차분²
JERK_STEER_W = 0.9            # 조향 명령 2차 차분² (떨림. 등속 램프는 0)
JERK_WHEEL_W = 0.3            # 바퀴 명령 2차 차분²
YAW_ACC_W    = 0.001          # 몸체 yaw 각가속도². 버그수정 env 서 Direct8 6.9 vs IK 2.4 rad/s² → 정책 기인
# ===================================================================

# scripts/train.py --set KEY=VALUE 덮어쓰기 (KSRC_SET 환경변수로 전달됨, 파이썬 리터럴).
import ast as _ast  # noqa: E402
import os as _os  # noqa: E402

for _kv in filter(None, _os.environ.get("KSRC_SET", "").split("\n")):
    _k, _v = _kv.split("=", 1)
    if _k not in globals():
        raise KeyError(f"--set: rover_env_cfg 에 {_k} 없음")
    globals()[_k] = _ast.literal_eval(_v)
    print(f"[set] {_k} = {globals()[_k]!r}", flush=True)


@configclass
class RoverSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )
    robot: ArticulationCfg = ROVER_CFG
    sky_light = AssetBaseCfg(prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=750.0))


@configclass
class CommandsCfg:
    # Same as Isaac Lab's UniformVelocityCommand + raw error metrics (mdp/commands.py).
    base_velocity = mdp.RoverVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 8.0),
        rel_standing_envs=0.05,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.RoverVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.25, 0.25), lin_vel_y=(-0.20, 0.20), ang_vel_z=(-1.0, 1.0), heading=(-3.14, 3.14)
        ),
    )


@configclass
class Direct8ActionsCfg:
    rover = mdp.Direct8ActionCfg()


@configclass
class Residual3ActionsCfg:
    rover = mdp.Residual3ActionCfg()


@configclass
class Res8ActionsCfg:
    rover = mdp.Res8ActionCfg()


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # 29D common actor input, no true translational velocity or terrain state.
        # Sensor terms go through DelayedObs: plain signal in stage 1; in stage 2 they are
        # env._obs_delay steps old (+ gyro bias) and get NOISE added (enable_corruption).
        command = ObsTerm(func=lab_mdp.generated_commands, params={"command_name": "base_velocity"})
        gyro = ObsTerm(func=mdp.DelayedObs, params={"source": "gyro"},
                       noise=GaussianNoiseCfg(std=NOISE["gyro"]))
        gravity = ObsTerm(func=mdp.DelayedObs, params={"source": "gravity", "grav_err": GRAV_ERR_POLICY},
                          noise=GaussianNoiseCfg(std=NOISE["gravity"]))
        steering = ObsTerm(func=mdp.DelayedObs, noise=GaussianNoiseCfg(std=NOISE["steer_pos"]),
                           params={"source": "steer_pos", "asset_cfg": SceneEntityCfg(
                               "robot", joint_names=STEER_JOINTS, preserve_order=True)})
        steering_rate = ObsTerm(func=mdp.DelayedObs, noise=GaussianNoiseCfg(std=NOISE["steer_vel"]),
                                params={"source": "steer_vel", "asset_cfg": SceneEntityCfg(
                                    "robot", joint_names=STEER_JOINTS, preserve_order=True)})
        wheel_rate = ObsTerm(func=mdp.DelayedObs, noise=GaussianNoiseCfg(std=NOISE["wheel_vel"]),
                             params={"source": "wheel_vel", "asset_cfg": SceneEntityCfg(
                                 "robot", joint_names=WHEEL_JOINTS, preserve_order=True)})
        # What the Pi itself last commanded: known exactly, so no delay or noise.
        applied_targets = ObsTerm(func=mdp.observations.applied_targets)

        def __post_init__(self):
            self.enable_corruption = DR_STAGE >= 2
            self.concatenate_terms = True
            # One-step history (current + previous frame): lets the policy infer
            # accelerations/slip that a single frame cannot show. 29D -> 58D.
            self.history_length = 2
            self.flatten_history_dim = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        # Teacher-only ground truth (22D). Student never sees this group; it must infer
        # these from its history. No noise: they are the answers, not sensor readings.
        lin_vel = ObsTerm(func=lab_mdp.base_lin_vel)      # true body velocity (3)
        mu = ObsTerm(func=mdp.wheel_mu)                   # static μ per wheel (4)
        slip = ObsTerm(func=mdp.wheel_slip)               # signed slip per wheel (4)
        mass_com = ObsTerm(func=mdp.base_mass_com)        # mass scale, CoM offset (4)
        dr = ObsTerm(func=mdp.dr_params)                  # stage-2 DR values, nominal in stage 1 (7)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class StudentCfg(PolicyCfg):
        # The teacher's "policy" group, but every sensor computed as on the real rover (2026-10-13)
        # and a longer history (STUDENT_HISTORY). Separate group so the teacher's input stays fixed.
        #   gravity  : 20 ms-averaged IMU -> complementary filter (pi/common/sensor_fusion.py), not true gravity
        #   steering : servo position register (shaft side, linkage play not seen), 4096 grid
        #   wheel    : encoder tick difference over 20 ms (1 tick = 0.0375 rad/s), no added noise
        gravity = ObsTerm(func=mdp.DelayedObs, params={    # filter-input noise only in DR stage 2
            "source": "gravity_imu", "noise_gyro": STUDENT_NOISE["imu_gyro"] if DR_STAGE >= 2 else 0.0,
            "noise_acc": STUDENT_NOISE["imu_acc"] if DR_STAGE >= 2 else 0.0,
            "grav_err": GRAV_ERR_STUDENT and DR_STAGE >= 2})
        steering = ObsTerm(func=mdp.DelayedObs, noise=GaussianNoiseCfg(std=STUDENT_NOISE["steer_pos"]),
                           params={"source": "steer_pos_servo", "asset_cfg": SceneEntityCfg(
                               "robot", joint_names=STEER_JOINTS, preserve_order=True)})
        steering_rate = ObsTerm(func=mdp.DelayedObs, noise=GaussianNoiseCfg(std=STUDENT_NOISE["steer_vel"]),
                                params={"source": "steer_vel_servo", "asset_cfg": SceneEntityCfg(
                                    "robot", joint_names=STEER_JOINTS, preserve_order=True)})
        wheel_rate = ObsTerm(func=mdp.DelayedObs, params={"source": "wheel_vel_enc", "asset_cfg": SceneEntityCfg(
            "robot", joint_names=WHEEL_JOINTS, preserve_order=True)})

        def __post_init__(self):
            super().__post_init__()
            self.history_length = STUDENT_HISTORY

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()
    student: StudentCfg = StudentCfg()


@configclass
class RewardsCfg:
    track_v = RewTerm(func=mdp.track_linear_velocity, weight=0.60, params={"std": 0.05})
    track_om = RewTerm(func=mdp.track_angular_velocity, weight=0.40, params={"std": 0.20})
    # Direct8 raw steering chatters without this; keep it small so it only
    # suppresses jitter and does not fight velocity tracking. Tune per mode.
    action_rate = RewTerm(func=lab_mdp.action_rate_l2, weight=-0.01)
    # Dense linear terms (enabled when LINEAR=True in CONFIG).
    lin_dense = None
    ang_dense = None
    track_v_lin = None
    track_om_lin = None
    slip = None
    slip_sq = None
    # Smoothness terms (enabled from CONFIG RATE_*/JERK_*/YAW_ACC_W).
    rate_steer = None
    rate_wheel = None
    jerk_steer = None
    jerk_wheel = None
    yaw_acc = None


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=lab_mdp.time_out, time_out=True)
    excessive_tilt = DoneTerm(func=mdp.excessive_tilt)
    # 8x16 m map (2x4 tiles of 4 m, no border). With 16 s episodes ~1.3% of runs reach the
    # edge (2026-10-11 Monte Carlo). Leaving the map is not the policy's failure, so end it
    # as a truncation (time_out=True: PPO bootstraps the value) right at the edge.
    off_map = DoneTerm(func=lab_mdp.terrain_out_of_bounds, time_out=True,
                       params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 0.2})


@configclass
class EventsCfg:
    # Asymmetric left/right wheel μ (teammate helper); enabled when WHEEL_MU != 'off' (CONFIG).
    wheel_friction = None
    wheel_friction_interval = None
    base_mass = None
    # Stage-2 DR (DR_STAGE >= 2): drive motor, steering servo, latency, IMU bias.
    drive_motor = None
    steer_servo = None
    latency = None
    imu_bias = None
    grav_err = None              # gravity-estimate error (GRAV_ERR), student and/or policy group
    # This DR tests friction robustness; it does not simulate granular sinkage.
    physics_material = EventTerm(
        func=lab_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    reset_base = EventTerm(
        func=lab_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        },
    )
    reset_joints = EventTerm(
        func=lab_mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )


@configclass
class RoverBaseEnvCfg(ManagerBasedRLEnvCfg):
    scene: RoverSceneCfg = RoverSceneCfg(num_envs=4096, env_spacing=2.5)
    commands: CommandsCfg = CommandsCfg()
    actions: object = MISSING
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = EPISODE_S
        self.commands.base_velocity.resampling_time_range = CMD_RESAMPLE_S
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        # rough_slope(318k tri) 는 여러 PhysX GPU 버퍼를 넘긴다(contact 드롭=물리 부정확).
        # 모두 넉넉히 키운다. 4096env 에서 PhysX 가 요구한 값(collision stack ≥6.9e8,
        # patch ≥2.6e5, 2026-10-09 로그)보다 크게 잡음. 학습은 1개씩, 기본 4096env.
        self.sim.physx.gpu_collision_stack_size = 2**30
        self.sim.physx.gpu_max_rigid_patch_count = 2**20
        self.sim.physx.gpu_max_rigid_contact_count = 2**24
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**23
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**25
        if TERRAIN_KIND != "plane":
            from experiments.rover_training_terrains.isaaclab_loader import make_terrain_cfg

            self.scene.terrain = make_terrain_cfg(
                TERRAIN_KIND, mode=TERRAIN_MODE,
                seed=self.seed or 0, num_rows=2, num_cols=4, overrides=TERRAIN_OVERRIDES or None,
            )
        self.sim.physics_material = self.scene.terrain.physics_material
        mass, com = (MASS_SCALE_WIDE, COM_RANGE_M_WIDE) if DR_WIDE else (MASS_SCALE, COM_RANGE_M)
        if mass != (1.0, 1.0) or any(com):
            self.events.base_mass = EventTerm(
                func=mdp.randomize_base_mass_com, mode="startup",
                params={"mass_scale": mass, "com_range_m": com})
        if DR_STAGE >= 2:
            d = DR2_WIDE if DR_WIDE else DR2
            self.events.drive_motor = EventTerm(func=mdp.randomize_drive_motor, mode="reset", params={
                **{k: d[k] for k in ("voltage_scale", "motor_spread", "damping_scale", "armature_range")},
                "friction_scale": d["drive_friction_scale"]})
            self.events.steer_servo = EventTerm(func=mdp.randomize_steer_servo, mode="reset", params={
                "rate_scale": d["servo_rate_scale"], "stiffness_scale": d["servo_stiffness_scale"],
                "offset_deg": d["servo_offset_deg"], "deadband_deg": d["servo_deadband_deg"],
                "delay_s": d["servo_delay_s"]})
            self.events.latency = EventTerm(func=mdp.randomize_latency, mode="reset", params={
                "action_delay": d["action_delay"], "obs_delay": d["obs_delay"]})
            self.events.imu_bias = EventTerm(func=mdp.randomize_imu_bias, mode="reset",
                                             params={"gyro_bias": d["gyro_bias"]})
        if (GRAV_ERR_STUDENT and DR_STAGE >= 2) or GRAV_ERR_POLICY:
            self.events.grav_err = EventTerm(func=mdp.randomize_gravity_error, mode="reset", params=dict(GRAV_ERR))
        self.scene.robot = self.scene.robot.replace(
            init_state=self.scene.robot.init_state.replace(pos=(0.0, 0.0, SPAWN_Z)))
        self.actions.rover.drive = DRIVE_MODEL
        self.actions.rover.drive_gate = DRIVE_GATE
        if DRIVE_DAMPING is not None:
            acts = dict(self.scene.robot.actuators)
            acts["drive"] = acts["drive"].replace(damping=DRIVE_DAMPING)
            self.scene.robot = self.scene.robot.replace(actuators=acts)
        if WHEEL_MU == "patch":
            p = dict(MU_PATCH_WIDE if DR_WIDE else MU_PATCH, master_seed=WHEEL_SEED)
            self.events.wheel_friction = EventTerm(func=mdp.randomize_wheel_mu_patch, mode="reset",
                                                   params=dict(p, new_base=True))
            self.events.wheel_friction_interval = EventTerm(
                func=mdp.randomize_wheel_mu_patch, mode="interval",
                interval_range_s=MU_PATCH_INTERVAL_S, params=dict(p, new_base=False))
            self.events.physics_material = None
        elif WHEEL_MU == "custom":
            mu_params = {"mu_range": MU_RANGE, "per": MU_PER, "master_seed": WHEEL_SEED}
            self.events.wheel_friction = EventTerm(func=mdp.randomize_wheel_mu, mode="reset", params=mu_params)
            if MU_RESAMPLE_S > 0:
                self.events.wheel_friction_interval = EventTerm(
                    func=mdp.randomize_wheel_mu, mode="interval",
                    interval_range_s=(MU_RESAMPLE_S, MU_RESAMPLE_S), params=mu_params)
            self.events.physics_material = None
        elif WHEEL_MU != "off":
            # Target phenomenon ON: asymmetric wheel μ at reset, and drop the
            # symmetric all-body friction DR so the two do not fight over the wheels.
            self.events.wheel_friction = EventTerm(
                func=mdp.randomize_wheel_friction, mode="reset",
                params={"mode": WHEEL_MU, "master_seed": WHEEL_SEED, "num_buckets": 64},
            )
            self.events.physics_material = None
        # Tight "terminal-convergence" std from CONFIG. At very tight std the exp
        # gradient vanishes far from target, so the dense linear term may be what
        # carries learning -- the A/B question re-run on Res8 at target precision.
        # Judge by error_vel_xy/yaw (NOT reward -- scales differ between arms).
        self.rewards.track_v.params["std"] = STD_V
        self.rewards.track_om.params["std"] = STD_OM
        if LINEAR:
            self.rewards.lin_dense = RewTerm(func=mdp.lin_vel_error_abs, weight=1.0)
            self.rewards.ang_dense = RewTerm(func=mdp.ang_vel_error_abs, weight=0.3)
        if LIN_TRI_V:
            self.rewards.track_v_lin = RewTerm(func=mdp.track_lin_vel_linear, weight=LIN_TRI_V[0],
                                               params={"e_max": LIN_TRI_V[1]})
        if LIN_TRI_OM:
            self.rewards.track_om_lin = RewTerm(func=mdp.track_ang_vel_linear, weight=LIN_TRI_OM[0],
                                                params={"e_max": LIN_TRI_OM[1]})
        if SLIP_W > 0:
            self.rewards.slip = RewTerm(func=mdp.slip_magnitude, weight=-SLIP_W)
        if SLIP_SQ:
            self.rewards.slip_sq = RewTerm(func=mdp.slip_speed_sq, weight=-SLIP_SQ[0],
                                           params={"deadzone": SLIP_SQ[1]})
        if RATE_STEER_W > 0 or RATE_WHEEL_W > 0:
            self.rewards.action_rate = None
        for name, w, part, order in (("rate_steer", RATE_STEER_W, "steer", 1),
                                     ("rate_wheel", RATE_WHEEL_W, "wheel", 1),
                                     ("jerk_steer", JERK_STEER_W, "steer", 2),
                                     ("jerk_wheel", JERK_WHEEL_W, "wheel", 2)):
            if w > 0:
                setattr(self.rewards, name, RewTerm(func=mdp.ActionDiffPenalty, weight=-w,
                                                    params={"part": part, "order": order}))
        if YAW_ACC_W > 0:
            self.rewards.yaw_acc = RewTerm(func=mdp.YawAccelPenalty, weight=-YAW_ACC_W)


@configclass
class RoverDirect8EnvCfg(RoverBaseEnvCfg):
    actions: Direct8ActionsCfg = Direct8ActionsCfg()


@configclass
class RoverResidual3EnvCfg(RoverBaseEnvCfg):
    actions: Residual3ActionsCfg = Residual3ActionsCfg()


@configclass
class RoverRes8EnvCfg(RoverBaseEnvCfg):
    actions: Res8ActionsCfg = Res8ActionsCfg()


class _PlayMixin:
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
        self.events.physics_material = None


@configclass
class RoverDirect8EnvCfg_PLAY(_PlayMixin, RoverDirect8EnvCfg):
    pass


@configclass
class RoverResidual3EnvCfg_PLAY(_PlayMixin, RoverResidual3EnvCfg):
    pass


@configclass
class RoverRes8EnvCfg_PLAY(_PlayMixin, RoverRes8EnvCfg):
    pass
