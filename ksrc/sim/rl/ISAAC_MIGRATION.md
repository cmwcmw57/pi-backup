# Isaac Lab 이주 계획 (KSRC 4WIS 주행보조 RL)

> 작성 2026-10-08 · 대상 서버 hpi-lab(166.104.224.241, `sumee`, 공용 GPU)
> 목표: **마찰 DR 만으로** cmd_vel 추종을 Isaac Lab 에서 학습하고(B-1),
> 그 정책을 **MuJoCo(전체 terramech)** 에서 sim2sim 전이 검증한다.
> 작업 로그·전체 맥락은 옵시디언 `[[Isaac Lab 전이 계획]]` / `[[RL 주행보조 진행상황]]`.
> 이 파일은 **코드와 함께 가는 공식 계획서**(커밋·서버 scp 대상). 옵시디언 노트는 작업일지.

---

## 0. 한 줄 요약

Isaac 은 **마찰만 있는 빠른 GPU 시뮬**이라 학습 처리량에만 쓰고, **침하·슬립·고착 같은 토질 물리는
MuJoCo terramech 에서만** 검증한다. 세 가지 액션공간(3D residual / res8 / direct8)을 **같은 조건에서 같은 시간**
학습시켜 한 번에 비교한다. 성패의 관문은 Phase 3 (순수 IK 기준선이 Isaac↔MuJoCo 에서 일치하는가)이다.

---

## 1. 왜 Isaac 인가 / 왜 B-1 인가

- 서버 `rover_training_terrains` 지형 패키지 = **고정 mesh + PhysX 쿨롱 마찰**뿐. 침하·전단곡선·공간 토질 **없음**
  (동료 문서가 "물리적 한계"로 명시). → Isaac 에서 토질을 학습시킬 길이 없다.
- **B-1**: 마찰계수 DR 만으로 학습 → MuJoCo terramech 로 전이 검증. 싼 경로.
  깨지면 그때 **B-2**(terramech 를 Isaac custom force 로 이식, 병목·고비용).
- 선례: Lee & Joe 2026 (4WIS) 가 정확히 이 구성 — 슬립 물리 없이 마찰 DR [0.3,1.0] + 높이 0~10cm 로
  학습해 실제 경사 잔디밭 횡오차 −21.54%. B-1 이 현실적임을 뒷받침.
- **노트북 RTX 3050 4GB 로는 Isaac 불가**(최소 8GB) → 서버 GPU 필수. 그래서 이주.

---

## 2. 확정된 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| 물리 | B-1 PhysX 마찰만 | 지형 패키지에 슬립·침하 없음 |
| 학습기 | **rsl_rl** (SB3 아님) | GPU 텐서 유지. SB3 는 CPU 왕복 병목 |
| env 방식 | **ManagerBasedRLEnv** | 사용자 경험 보유 |
| 바퀴 액추에이터 | **속도 폐루프**(DCMotorCfg / PI) | 실기가 엔코더 피드백. 2026-10-08 폐루프 병진 −37% |
| Actor 관측 | 자기수용만 (26D × 이력) | 서버 규칙 + 특권가림 측정 |
| Critic/teacher | 특권 43D 허용 | 규칙 허용. s51 사고는 **Actor 쪽**에서만 방지 |
| 구조 | teacher–student | 보행 문헌 주류 + s38 작동 확인 |
| 조향 속도 | **데이터시트 물리상한 고정**(DR 아님) | 모래 위 실측 불가 → 상한 가정이 안전 |
| 잔차 상한 | d_rate_max 0.10, scale [0.11,0.085,0.45] | 2026-10-07 변화율 제한 (건드리지 않음) |

### 2.1 세 가지 액션공간 — 같은 구조, 액션 정의만 교체

> 사용자 방침: "셋 다 준비 → 같은 시간 학습 → 한 번에 비교". MDP 에서 액션 term 만 바꾼다.

| 모드 | 차원 | 정의 | IK 벡터화 |
|---|---|---|---|
| **3D residual** | 3 | `cmd_applied = cmd_nom + a·[Δvx,Δvy,Δω]`, a=0 ⇒ 순수 IK | 필요 |
| **res8** | 8 | IK 출력(조향4+속도4) 각각에 잔차: `th_ik+a·d_steer`, `v_ref·(1+a·d_wheel)` | 필요 |
| **direct8** | 8 | 순수 8D(조향4+속도4), IK 안 거침: `th=a_steer·st_hi`, `v=a_wspd·v_max` | **불필요** |

- res8/3D 는 펌웨어 IK 를 거치므로 **GPU 상에서 IK 를 벡터화**해야 한다(§5.4).
- direct8 은 IK 를 안 쓰므로 벡터화 대상 아님. 비교의 바닥선(순수 RL).

---

## 3. 액추에이터 수치 (데이터시트 확정, 2026-10-08)

> 실기 effort limit · kp/kd 를 시뮬에 그대로 박아야 sim2real 이 성립. 로봇핸드 때와 동일 원칙.

### 3.1 구동 모터 SPG30E-GR131 (DC) — `DCMotorCfg`
- stall torque 20 kg·cm = **1.96 N·m** → `effort_limit = 1.96`
- no-load 76 rpm = **7.96 rad/s** → `velocity_limit = 7.96`
- DCMotorCfg 가 토크–속도 포화선을 자동 모델링(속도↑ → 가용토크↓). 이게 "최대로 쓴다" 가정의 안전한 표현.

### 3.2 조향 서보 STS3215 — `ImplicitActuatorCfg`
- stall 19.5 kg·cm = **1.912 N·m** → `effort_limit = 1.912`
- no-load 52 rpm = **5.45 rad/s** → `velocity_limit = 5.45`
- `kp = 12`, `kd = 0.30` (위치 폐루프; 서보 내부 PID 게인 비공개 → 실기 정합되는 값으로 설정)
- `joint_pos`(range) = **±90° = ±1.5708 rad** (조향 가동범위, URDF 한계. 예전 ±100°)
- 주의: 실기 `S` 명령은 내부 속도캡. **재부팅 시 0(=최대)로 리셋**됨. 실무도 첫 1회만 S1500,
  이후는 켤 때마다 리셋된 최대. 그래서 시뮬은 **데이터시트 물리상한 5.45 를 고정**(S캡 가정 안 함).

### 3.3 로커 디퍼렌셜 — 소프트 구속 (Phase 1b)
- USD 에 좌우 로커 1:1 역방향 구속 **없음**(revolute 10 + fixed 2 뿐).
- 하드 기어조인트(`PhysxPhysicsGearJoint`)는 GPU 지원 불확실 → **소프트 구속** 채택:
  `τ = −k(θ_L+θ_R) − d(ω_L+ω_R)`. 스파이크에서 SUM→0, θ_L=+27°/θ_R=−30° 확인(대립 성립).
- GPU-safe, 텐서화 가능. **이게 없으면 서스펜션 거동이 달라져 다른 기계가 된다.**

---

## 4. 관측 / 액션 계약 (`utils/contract.py` 로 숫자 못박기)

> Isaac·MuJoCo 양쪽이 **같은 파일을 import**. 정규화 스케일·차원·순서가 한 글자라도 다르면 전이가 깨진다.

- Actor 관측 **26D × 이력**(history 길이는 Phase 5 변수: 5/25/50/100).
- **Actor 금지 목록(특권)**: 로커 각도, 참 body velocity, 바퀴 하중(wheel load), 높이맵.
  → s51 사고(특권 teacher 가 참 속도 보고 병진이득 과대평가)를 Actor 단계에서 원천 차단.
- Critic/teacher 는 특권 43D 허용(규칙상 OK).
- pitch 25.5° 는 base_link **프레임 오프셋**이지 물리 자세 아님 — 계약에서 보정할 것.

---

## 5. 구현 뼈대 (설치 · 폴더 · 등록 · 실행)

> 여기까지는 전략이었고, 아래가 실제로 손으로 짓는 부분이다. 전부 서버 실측(2026-10-08)에 근거.

### 5.1 소프트웨어 스택 — 이미지에 든 것 / 없는 것

**이미지 `ksrc-rover:lab2.3.2` 에 이미 있는 것** (Phase 0 검증 완료):

| 구성 | 경로 / 버전 |
|---|---|
| Isaac Sim kit python | `/isaac-sim/kit/python` (컨테이너 기본 인터프리터) |
| Isaac Lab 본체 | `/opt/IsaacLab/source/isaaclab` (0.54.2) |
| Isaac Lab tasks | `/opt/IsaacLab/source/isaaclab_tasks` (velocity 템플릿 포함) |
| RL 래퍼 | `isaaclab_rl` (RslRlOnPolicyRunnerCfg 등) |
| 학습기 | `rsl_rl` |
| 시뮬 커널 | `warp 1.8.2` |
| 스톡 학습 스크립트 | `/opt/IsaacLab/scripts/reinforcement_learning/rsl_rl/{train,play}.py` |

→ **B-1 에 추가로 pip install 할 것은 없다.** 우리가 "설치"할 것은 **우리 task 패키지 하나**뿐(PYTHONPATH 등록).

**이미지에 없는 것 — 주의**:

- **gcc/빌드 툴체인 없음.** 따라서 **ctypes `.so` 를 컨테이너 안에서 컴파일할 수 없다.**
  결론(§5.4): 컨테이너용 IK 는 **PyTorch 로 작성**하고, 펌웨어 C(ctypes)는 **노트북(gcc 있음)에서 검증 오라클로만** 쓴다.
  (build-essential 을 Dockerfile 에 넣어 재빌드하는 길도 있지만, GPU IK 는 어차피 PyTorch 가 필요하므로 불필요.)

### 5.2 폴더 구조 (타깃)

현재 서버 `~/gyusa/ksrc_rover/` 는 **스파이크·진단 스크립트 더미**다(`scripts/phase1_*`, `rocker_*`, `convert_urdf`, `assets/`).
이걸 Isaac Lab velocity task 레이아웃(`isaaclab_tasks/manager_based/locomotion/velocity`)을 본떠 **정식 패키지**로 승격한다:

```
~/gyusa/ksrc_rover/                     # 마운트 → /workspace
├── run.sh  run_view.sh                 # docker 래퍼 (source/와 지형 패키지 마운트)
├── assets/                             # (존재) usd/  rover_portable.urdf  meshes/
├── scripts/
│   ├── train.py                        # 버전 맞는 스톡 train.py 호출 + task 등록 (§5.3)
│   ├── play.py                         # 버전 맞는 스톡 play.py 호출 + task 등록
│   ├── smoke.py                        # reset·소수 스텝 로딩 검사
│   ├── export_jit.py                   # actor → TorchScript (Phase 6 MuJoCo 전이용)
│   └── _spikes/                        # 기존 진단 스크립트 격리 자리
└── source/ksrc_rover/                  # 파이썬 패키지  (PYTHONPATH=/workspace/source)
    ├── __init__.py                     # gym.register ×(3 액션모드 × train/play)
    ├── rover_env_cfg.py                # ManagerBasedRLEnvCfg: Scene/Obs/Action/Reward/Term/Event/Command
    ├── assets/
    │   └── rover_cfg.py                # ArticulationCfg: USD, DCMotorCfg/ImplicitActuatorCfg, 로커 소프트구속(§3)
    ├── mdp/
    │   ├── __init__.py
    │   ├── actions.py                  # ActionTerm 3종: residual3 / res8 / direct8 (§2.1)
    │   ├── observations.py             # 자기수용 26D×이력 / 특권 43D(critic 전용) (§4)
    │   ├── rewards.py                  # track_v 0.60 + track_om 0.40 − resid − smooth
    │   ├── terminations.py             # 전복·경계이탈
    │   ├── events.py                   # 마찰 DR, 리셋, push
    │   └── commands.py                 # cmd_vel 샘플링 (180° 반전 빈도 제한)
    ├── agents/
    │   └── rsl_rl_ppo_cfg.py           # RslRlOnPolicyRunnerCfg + PPO + teacher/student
    └── utils/
        ├── contract.py                 # 관측/액션/정규화 스케일 숫자 못박기 (Isaac·MuJoCo 양쪽 import)
        └── ik_torch.py                 # 펌웨어 IK 의 PyTorch 벡터화 (컨테이너용, §5.4)
```

노트북(이 repo) 쪽 검증 자산:

```
sim/rl/ik_validate.py                   # ik_torch ↔ ctypes C 수치 대조 (gcc 있는 노트북에서)
firmware/common/swerve_kinematics.c     # → .so, 검증 오라클
```

> 왜 `source/ksrc_rover/` 를 따로 두나: Isaac Lab 의 외부 task 관례(패키지를 PYTHONPATH 로 얹고 gym.register).
> `scripts/` 는 실행 진입점, `source/` 는 import 대상. 섞으면 상대 import 가 깨진다.
> 기존 서버 `phase1_*`/`rocker_*`/`convert_urdf.py`는 이전 CLI 호환을 위해 현재 경로에 유지한다.
> 새 task는 이 스파이크를 import하지 않는다.

### 5.3 task 등록 & 실행

Isaac Lab velocity task 의 등록 방식(`config/anymal_c/__init__.py`)을 그대로 따른다 —
`gym.register(id=..., entry_point="isaaclab.envs:ManagerBasedRLEnv", kwargs={env_cfg_entry_point, rsl_rl_cfg_entry_point})`.

`source/ksrc_rover/__init__.py` 에서 **액션공간 3종 × (train/play)** 를 등록:

| Task ID | env_cfg | 액션 |
|---|---|---|
| `Isaac-KSRC-Rover-Residual3-v0` | `RoverResidual3EnvCfg` | 3D 잔차 |
| `Isaac-KSRC-Rover-Res8-v0` | `RoverRes8EnvCfg` | IK+8D 잔차 |
| `Isaac-KSRC-Rover-Direct8-v0` | `RoverDirect8EnvCfg` | 순수 8D |
| `...-Play-v0` (각 3개) | `..._PLAY` | 평가용(소수 env, 커리큘럼 OFF) |

세 cfg 는 **같은 base** 를 상속하고 `actions` term 만 교체한다(사용자 방침: "액션만 바꿔서 같은 시간 학습").

**외부 task 등록 주의**: 스톡 `train.py` 는 `import isaaclab_tasks` 만 한다 → 우리 패키지는 **자동 등록 안 됨**.
그래서 `scripts/train.py`/`play.py` 는 스톡 스크립트를 호출하기 전에 `import ksrc_rover`로 등록한다.
컨테이너 래퍼는 `/workspace/source`를 `PYTHONPATH`에 넣고 작업 디렉터리를 `/workspace`로 지정한다.
그러면 `gym.make("Isaac-KSRC-Rover-Direct8-v0")` 가 동작하고 로그도 마운트된 `logs/rsl_rl/`에 남는다.

**실행 (사용자 몫 — GPU 학습)**:

```bash
# 서버에서. GPU 확인 후:
KSRC_TERRAIN_KIND=rough_slope KSRC_TERRAIN_MODE=narrow \
  bash run.sh /workspace/scripts/train.py \
    --task Isaac-KSRC-Rover-Direct8-v0 --headless --num_envs 4096 \
    --seed 100   # 평가는 --seed 900
```

Claude 는 **등록·cfg·스크립트 작성과 수초짜리 import/make 검증**까지, **학습 실행은 사용자**.

### 5.4 IK: 컨테이너는 PyTorch, 검증은 노트북 ctypes

- `utils/ik_torch.py` — 펌웨어 `swerve_kinematics.c` 의 스워브 IK 를 **배치 텐서**로 재구현(GPU, grad 불필요).
  residual3 / res8 가 이걸 거친다. direct8 은 IK 를 안 쓰므로 **대상 아님**.
- `sim/rl/ik_validate.py`(노트북) — 같은 `cmd_nom` 연속 시퀀스를 ik_torch 와 ctypes `.so` 양쪽에 넣어 **수치 일치** 확인.
  2026-10-08 1000개 명령에서 최대 절대오차 `6e-8` 통과. 컨테이너엔 gcc 가 없어 .so 를 못 만든다.

## 6. 진행 단계

- [x] **Phase 0 — 학습 이미지** (2026-10-08 완료)
  - `~/gyusa/ksrc_rover_build/Dockerfile`, 이미지 `ksrc-rover:lab2.3.2` (24.3GB)
  - 베이스(`rover-training-terrains:lab2.3.2-sim5.1`) + `isaaclab_rl` + `rsl_rl`.
    베이스는 preview 전용이라 학습 의존성이 없었음(동료 문서 명시).
  - 검증: AppLauncher 경유 `warp 1.8.2 / isaaclab 0.54.2 / rsl_rl / ManagerBasedRLEnv /
    RslRlOnPolicyRunnerCfg` 전부 import OK (GPU1, 수초).

- [x] **Task 골격 및 등록 smoke** (2026-10-08)
  - `sim/isaac/ksrc_rover/`에 3개 action mode × train/play Gym task, 공통 ManagerBasedRLEnvCfg,
    액추에이터/로커 소프트 구속, rsl_rl 설정, train/play/export 진입점 작성.
  - 서버 flat terrain에서 Direct8 1스텝, Residual3·Res8 2스텝 reset/step 통과. IK C ↔ torch
    1000개 연속 명령 최대 절대오차 `6e-8`.
  - **이는 task 로딩 검증이지 Phase 1~4 완료가 아니다.** 지형 로더, 접지·동역학, 모터 응답,
    관측 계약(현재 Isaac 29D, 계획 26D, MuJoCo 36D), 기준선은 후속 검증이 필요.

- [ ] **Phase 1 — 로버를 지형 위에 세우기**
  - USD: 서버 `rover_portable.usda`(md5 일치, 2.3263kg). `make_terrain_cfg` 위 스폰.
  - 100스텝 물리 → 자세·질량·접지 정상(4바퀴 z≈47.5mm 접지).

- [ ] **Phase 1b — 로커 디퍼렌셜** ★난제 (§3.3)
  - 소프트 구속 텐서화. 없으면 다른 서스펜션 = 다른 기계.

- [ ] **Phase 2 — 계약 + IK + 액추에이터** ★핵심
  - `utils/contract.py`(§4), `utils/ik_torch.py`(GPU), 펌웨어 C/ctypes는 로컬 검증 오라클.
  - 액추에이터 §3 수치로. 조향 속도제한·지연·데드밴드, 제어지연 0~2스텝, 정렬게이트.
  - **통과 조건**: 같은 `cmd_nom` 시퀀스 → Isaac·MuJoCo 양쪽 th_ik/du **수치 일치**.

- [ ] **Phase 3 — 기준선 대조** ★관문
  - 순수 IK(a=0) Isaac ↔ MuJoCo, 지형 맞추고(평지·μ고정·잡음 OFF). 병진·요 오차 **10~20% 이내**.
  - **안 맞으면 지형 탓으로 넘기지 말고 계약·액추에이터·접촉/구속 차이를 분리 진단. Phase 4 진입 금지.** 늦어도 5일차 판단.
  - 2026-10-09 평지 실측: [원자료·절차·결과](../isaac/ksrc_rover/benchmarks/2026-10-09_pure_ik/RESULTS.md). IK 목표각/휠 목표속도는 일치. 단독 좌·우 곡선은 모든 성분 20% 이내이나, 직진→횡이동→회전 뒤 곡선의 정상구간 횡속도 차이는 **29.9%**(Isaac 0.042, MuJoCo 0.031 m/s). 따라서 Phase 3 **부분 통과, 전체 관문 미통과**. 토질 힘을 끈 CAD MuJoCo 대조이므로 모래 전이 증거가 아님.

- [ ] **Phase 4 — task (mdp term)**
  - 보상 `track_v 0.60 + track_om 0.40 − resid − smooth [+tail]`.
  - 명령 샘플링(180° 반전 빈도 주의). 종료: 전복·경계. 고착은 PhysX 에 기구 없음(판정만).
  - 타일 4×4m, 경계여유 = 외접반경 226mm + 제동거리.

- [ ] **Phase 5 — 학습 (세 액션공간 동일 조건)**
  - 단일변수 스윕: **history 5/25/50/100**(병진속도 복원 핵심 미검증 변수), DR narrow/wide, teacher→student.
  - 세 모드(3D/res8/direct8) 같은 스텝·같은 시드로 걸고 **한 번에 비교**.

- [ ] **Phase 6 — sim2sim 전이**
  - rsl_rl actor → **TorchScript** → MuJoCo 평가 래퍼.
  - 단계: ① μ상수(계약·액추에이터만) ② μ(i) 침하 OFF(슬립곡선) ③ 전체(침하).

### 5.1 Phase 6 판정표 (나중에 합리화 방지 — 미리 못박음)

| 결과 | 다음 |
|---|---|
| 병진·요 평균 둘 다 유의 개선 | student 증류 → 실기 |
| 평균 개선, 침하·슬립·고착 악화 | 보상에 슬립/침하 항 추가 후 Isaac 재학습 |
| 차이 없거나 악화 | B-2 (terramech 이식) |

**예측**: 침하 없이 학습 → "슬립은 공짜"로 배움 → 전이에서 추종 OK, 침하·고착 ↑. **가운데 칸 유력.**

---

## 7. ⚠️ 주의사항 (반드시 지킬 것)

### 6.1 운영·안전 (절대 규칙)
1. **`pkill -f` 금지.** 공용 계정이라 남의 프로세스를 죽인다. 항상 **PID 기반 kill**.
2. **GPU 학습 실행·play 는 사용자 몫.** Claude 는 코드/빌드/파일/docker build/**수초 import·load 검증**까지만.
   "gpu 써서 학습 돌리는 것만 안 하면 됨."
3. 공용 `sumee` — **GPU 확인 후 `GPU=1`**(runner 자동확보 안 함). label 에 식별자.
4. **기존 `runs/` 폴더 덮어쓰기 금지.** `~/gyusa/runs/<실험>/<YYYYMMDD-HHMM>_<라벨>/` 신규 폴더.
5. **코드는 이미지에 굽지 않고 마운트.** `docker run ... -v ~/gyusa/ksrc_rover:/workspace ...`.
6. 실행 패턴: `docker run --gpus '"device=1"' -v ~/gyusa/ksrc_rover:/workspace ksrc-rover:lab2.3.2 <script> --headless`.
7. 학습 `--split train --seed 100` / 평가 `--split eval --seed 900`.

### 6.2 정확성 (전이가 조용히 깨지는 지점)
8. **Actor 특권관측 금지**(§4). 로커각·참속도·바퀴하중·높이맵을 Actor 가 보면 s51 재현(시뮬에서만 이기는 정책).
9. **계약 불일치 = Phase 3 관문.** 거기서 안 맞으면 **terrain 탓 하지 말 것** — 계약·액추에이터·접촉/구속을 먼저 분리 진단한다.
10. **IK 벡터화는 res8·3D 만 필요, direct8 은 불필요.** 벡터화 IK 는 ctypes C 와 **수치 대조로 검증**한 뒤 쓸 것.
11. **소프트 로커 구속 필수**(§3.3). 빠지면 서스펜션 거동이 달라 다른 기계로 학습됨.
12. **조향은 데이터시트 상한 5.45 고정**(DR 아님). 모래 위 실제 각속도 실측 전까지 S캡 가정하지 말 것.
13. **명령 샘플링 180° 반전 빈도 주의.** 급반전이 너무 잦으면 학습이 평균적으로 멈추는 걸 배운다.

### 6.3 범위·해석 (오해 방지)
14. **soil 비대칭 μ 변경은 OOD(분포 밖)이지 검증이 아니다.** 마찰 DR 로 학습한 정책을 토질로 테스트하는 건
    "일반화 확인"이지 "같은 분포 성능"이 아님 — 결과 해석 시 명시.
15. **지금까지의 학습 결과가 정답이 아니다.** 실기 실측값·Isaac 재학습으로만 알 수 있는 값이 남아 있다.
    (사용자 명시 방침)
16. **Chrono CRM(입상물리)은 실시간 1/577 → 학습 불가, 측정 장비.** F6(슬립–견인 곡선)를 시뮬에서 답할 수단일 뿐.
17. **노트북 메모리 15GB(가용 ~5GB)로 빠듯.** MuJoCo 측정·스모크는 **한 번에 하나씩**(동시 실행 OOM 전례).

---

## 8. 역할 분담

| 일 | 주체 |
|---|---|
| 코드·빌드·파일 편집·docker build·수초 import/load 검증 | Claude |
| GPU 학습 실행·play·train.py 검토 | 사용자 |

---

## 9. 서버 자산 경로

- 학습 이미지: `ksrc-rover:lab2.3.2` / 빌드 `~/gyusa/ksrc_rover_build/`
- 작업 디렉터리: `~/gyusa/ksrc_rover/`
- 로버 USD: `~/gyusa/runs/fast_sensor_replay/20261008/reference/assets/rover_portable/`
- 지형 패키지: `~/gyusa/runs/rover_training_terrains/npzcheck20261007c/experiments/rover_training_terrains/`
- 참고 Chrono CRM: `~/gyusa/runs/chrono_rover/`, `wheel_probe/`(82회)

---

## 10. MuJoCo 기준선 (config 정정 후 재측정, 2026-10-08)

> scrub 9.53→0, 조향 5.2→5.45 정정이 기준선을 흔들지 않음을 확인(sim2real 숨은 불일치 제거).
> 경로추종 순수 IK, sand, holdout seed=4321, 60ep, d=0.8, 조향±100°, 게이트 10°, soil-amp 0.25, 잡음 ON:

| 지표 | 값 |
|---|---|
| 완주율 | 100% |
| 횡오차 평균 | 15.90mm (정정 전 15.99mm — 거의 동일) |
| 기수오차 평균 | 8.19° (정정 전 ~10.36° — 개선) |
| 슬립 평균 | 0.136 |
| 침하 평균 | 37.8mm |
| 전복·고착·경계이탈 | 0% |

cmdvel(요 P-loop) 순수 IK 기준선 (같은 조건, 40ep, 정렬구간 ≤10°):

| 지표 | 값 |
|---|---|
| 병진 | 32.3 mm/s |
| 요 | 5.39 deg/s |
| 병진오차/요오차(전체) | 12.41 %vmax / 6.01 %ommax |
| 슬립 평균 | 0.18 |

**발견**: 요 P-feedback(Kp 0.1~0.2) 이득이 정정 후 **사라짐**(요 차이 +0.00~+0.02, CI 0 포함 = 무효).
정정 전(scrub 9.53)엔 요 −0.37~−0.60 deg/s 유의 개선이었다. 가설: scrub 이 만들던 요 외란을
P-loop 이 보정하던 것 → scrub=0 으로 외란이 사라져 순수 IK 요가 이미 맞음.
→ **고전 피드백이 메울 모델오차(scrub)가 줄었고, 남은 외란은 토질 쪽.** RL/폐루프의 실제 레버는 거기다.

---

## 관련
- 옵시디언: `[[Isaac Lab 전이 계획]]`, `[[RL 주행보조 진행상황]]`, `[[시뮬레이션 구조]]`, `[[실기 확인 목록]]`
- 코드: `sim/rl/config.py`, `sim/rl/env.py`, `sim/rl/pfb_baseline.py`, `firmware/common/swerve_kinematics.c`
