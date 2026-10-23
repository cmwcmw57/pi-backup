# KSRC 2026 — 4륜 독립조향 우주로버

2026 대한민국 우주로버 챌린지(우주항공청) 출전 로버의 **저수준 제어기 · 시뮬레이터 ·
강화학습** 코드베이스임.

바퀴 4개를 각각 독립 **조향** + **구동** 하는 스워브 방식임. 중앙 베벨 디퍼렌셜이
좌우 로커를 1:1 역방향으로 묶어 울퉁불퉁한 지형에서 네 바퀴 접지를 유지함.

| 항목 | 값 |
|---|---|
| 총질량 | 2.26 kg (규정 3 kg 이하) |
| 크기 | 284 × 263 × 173 mm (규정 300×300×200 이하) |
| 휠베이스 / 트랙 | 210 / 265 mm (조향축 = 바퀴 중심, 2026-10-03 실측. 이전 STL 183.6 / 236.7) |
| 바퀴 | 지름 95.0 mm, 폭 37.2 mm, 그라우저 약 28개 (URDF `rover_description` 충돌 실린더 r=47.503 mm. 이전 79.9 mm 는 구 조립본 rev_2) |
| 지상고 | 9.6 mm (다리 하우징 최저점 기준) |
| 구동 | SPG30E-GR131 ×4 (12V, 131:1, 76 rpm 무부하, 20 kg·cm 정지토크) |
| 조향 | FEETECH STS3215 ×4 (반이중 UART 버스, 1 Mbaud). 가동범위: 바퀴별 ±90° (URDF 관절 한계 = 펌웨어 `joint.c` = 시뮬 `config.steer_*_deg` = Isaac `STEER_LIMIT_RAD`, 조립 후 실측으로 확정) |
| 최고 속도 | 0.378 m/s (76 rpm × 바퀴 반경 47.5 mm) |
| 배터리 | 3S 650mAh 120C 리포 |
| MCU / SBC | STM32 Nucleo-F446RE / Raspberry Pi 5 + AI HAT |

---

## 시스템 구조

```
        조작자 / 자율
             │
    ┌────────┴────────┐
    │  Raspberry Pi 5 │   pi/teleop        조이스틱 → cmd_vel
    │                 │   Nav2 (MPPI/Omni) 경로 → cmd_vel   ← 시뮬 검증 완료
    │                 │   (계획) D435 ICP  대회장 맵 정합 → 절대 위치
    │                 │   RL 정책          cmd_vel 위의 잔차 보정
    └────────┬────────┘
             │  아스키/바이너리 시리얼 (115200)
    ┌────────┴────────┐
    │ Nucleo-F446RE   │   swerve_kinematics.c   cmd_vel → 바퀴 4개 각·속도
    │                 │   swerve_fold_to_range  바퀴별 가동범위 안 등가해 선택
    │                 │   drive_align_gate.c    큰 조향 전환 중 구동 보류
    │                 │   spi_link.c            Pi SPI2 명령/텔레메트리 프레임
    │                 │   mdd3a_driver.c        PWM
    │                 │   encoder.c             엔코더 4채널
    └────────┬────────┘
             │
      구동모터 ×4      조향서보 ×4
```

명령 인터페이스는 `cmd_vel = (vx, vy, ω)` 한 지점으로 모임. 조이스틱 · Nav2 ·
RL 잔차가 모두 같은 지점에 붙으므로 서로 교체 가능함.

**시뮬레이터가 펌웨어 C 코드를 그대로 호출함.** `sim/build.sh` 가
`firmware/common/*.c` 를 `libksrc_common.so` 로 빌드하고 파이썬이 ctypes 로 부름.
기구학을 파이썬으로 재구현하면 언젠가 반드시 어긋나는데, 정책이 기구학 위의
**잔차**를 학습하는 구조라 어긋나는 순간 정책이 통째로 무의미해짐.

---

## 저장소 구조

```
firmware/
  common/                   MCU 비의존 순수 C + 호스트 유닛테스트
  src2026-rover-firmware/   보드에 올라가는 CubeIDE 프로젝트 (별도 git 저장소)
pi/teleop/                  조이스틱 텔레옵 (Python)
ros/                        ROS2 / Nav2 통합
  make_maps.py              주행가능 마스크 -> Nav2 정적 맵
  maps/                     arena_{sand,rock}.{pgm,yaml}
  mujoco_bridge.py          MuJoCo <-> ROS2 (/odom /tf 발행, /cmd_vel 구독)
  nav2_params.yaml          Nav2 파라미터 (MPPI motion_model=Omni)
  nav2_ksrc.launch.py       최소 구성 런치
  rviz/ksrc.rviz            RViz 설정
sim/
  ksrc_common.py            펌웨어 C 를 ctypes 로 호출
  rl/                       MuJoCo 물리 + 강화학습
CLI.md                      전체 실행 명령 모음
```

### 폴더별 문서

| 폴더 | 문서 | 내용 |
|---|---|---|
| `firmware/` | [README](firmware/README.md) · [CLI](firmware/CLI.md) | STM32 저수준 제어기: 핀, 50Hz 제어주기, 설정값, 조향 영점·구동 세팅, 터미널 명령 |
| `sim/` | [README](sim/README.md) · [CLI](sim/CLI.md) | MuJoCo 물리, 강화학습 환경·보상, 학습·평가·재생, 설계상 선택, 학습 결과 |
| `ros/` | 이 문서 아래 "ROS2 / Nav2" | Nav2·RViz 클릭 주행, 대회맵 텔레옵 |
| `pi/` | [firmware/CLI.md §5·6](firmware/CLI.md) | 조이스틱 텔레옵, Pi ↔ Nucleo SPI 링크 |

---

## ROS2 / Nav2 실행

전체 인자는 **[CLI.md](CLI.md)** 에 있다. 학습·평가 명령은 [sim/CLI.md](sim/CLI.md).

### Nav2 + RViz 로 클릭 주행

```bash
# 브리지 + Nav2 + RViz 를 한 번에 실행한다 (기존 별도 브리지는 먼저 종료).
cd /home/eomyunbeen/ksrc/ros && source /opt/ros/humble/setup.bash
python3 make_maps.py                      # 맵이 없으면 한 번
# 아래 둘 중 하나만 실행
ros2 launch nav2_ksrc.launch.py terrain:=sand           # 순수 IK
ros2 launch nav2_ksrc.launch.py terrain:=sand rl:=true  # 구형 s2 정책 적용 확인용
```

브리지는 `/cmd_vel`을 0.10초 시정수로 필터링해 `/cmd_vel_filtered`로 내고,
`/rl_action`(정규화 행동), `/rl_delta`(요청 보정), `/cmd_vel_applied`(포화 후 IK 입력)를
발행한다. 보정 적용은 `/cmd_vel_filtered`와 `/cmd_vel_applied`를 비교하면 된다.
`ros2 service call /set_rl std_srvs/srv/SetBool '{data: false}'`로 주행 중 끄고
`'{data: true}'`로 다시 켤 수 있다. 정책 모델은 `rl_model`, `rl_vecnorm` 런치
인자로 바꿀 수 있다. `cmd_filter_tau:=0`은 명령 필터 비교용이다.
기본 `rl:=true`가 불러오는 `runs/s2`는 이번 경로·관측 수정 이전 모델이므로
성능 개선 판정에는 쓰지 않는다. 새 모델은 재학습 후 `rl_model`과 `rl_vecnorm`을
명시한다.

RViz 상단 **"2D Goal Pose"** 로 지도를 클릭하면 로버가 그쪽으로 감.
회색이 주행가능, 검정이 통과 불가(경사 21° 초과)임.
목표 허용오차가 0.15m 이므로 로버 바로 옆을 찍으면 이미 도착한 것으로 판정될 수 있다.
이 런치는 시뮬 전용 ROS 도메인 77과 localhost 통신을 사용해 다른 ROS 노드의
`/odom`·`/map_server`와 섞이지 않게 한다. 진단 명령도 같은 도메인에서 실행한다:

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1
ros2 lifecycle get /bt_navigator         # active 여야 목표를 받는다
ros2 topic hz /odom                       # 브리지 위치가 계속 발행되는지 확인
```

전복 시 브리지는 구동을 멈추되 물리는 계속 진행하며, 자동으로 스폰으로 순간이동하지 않는다. 다시 시작하려면 별도 터미널에서
위 두 환경변수를 설정하고 `ros2 service call /reset_sim std_srvs/srv/Trigger '{}'` 를 실행한다. 같은 초기 스폰으로 돌아간다.
RViz 의 "2D Pose Estimate" 는 시뮬 스폰을 바꾸지 않는다. RViz 주행에서 RL 잔차는
런치 인자 `rl:=true`를 준 경우에만 적용된다.

컨트롤러는 **MPPI (`motion_model: Omni`)** 를 씀. `nav2_regulated_pure_pursuit_controller`
는 differential/ackermann 용이라 `(v, ω)` 만 내고 스워브의 게걸음(`vy`)을 버림.
실측으로 MPPI 가 `vy = 0.057` 을 명령하는 것을 확인함.

측위는 브리지가 `map→odom` 을 **항등으로** 발행함 (참값). 그래서 "Nav2 가 우리 로버에
맞게 도는가" 만 남고 측위 문제가 섞이지 않음. 실제 측위(D435 지형 ICP)가 되면 그
항등 변환만 갈아끼우면 됨.

Nav2 정적 맵은 경사·단차의 통행 가능성만 표시한다. `drivable_mask`의 0.16m 창은
지형 기울기를 측정하는 규모이며 차체 충돌 반경이 아니다. Nav2에는 바퀴 위치와
반경에서 계산한 약 0.21m 풋프린트와 0.24m inflation을 적용한다 (2026-10-03 실측 치수, 이전 0.19/0.22). 목표점이
턱에 너무 가까우면 반경 밖의 도달 가능한 지점으로 경로가 끝날 수 있다
(플래너 허용오차 0.15m). 순수 IK 경로도 같은 중심 이격 기준으로 만든다.

격리된 Nav2 검증에서 턱을 사이에 둔 `(-1.1, 0.2) → (0.4, 0.2)` 경로의
통행 불가 영역 최소 이격은 기존 약 3cm에서 수정 후 22.1cm로 늘었다 (이전 치수 기준 기록).
기본 스폰 `(0.121, 0.661)`에서 `(-1.1, 0.2)`로 실제 MuJoCo 주행도 성공했고,
로버 중심 최소 이격은 23.5cm였다. 이 검증은 순수 IK, 참값 측위 조건이다.

### 대회맵 + 기존 조이스틱 텔레옵

첫 터미널에서 대회맵, MuJoCo 로버, RViz를 띄운다. 텔레옵 모드에는 Nav2
컨트롤러를 띄우지 않아 `/cmd_vel` 명령이 충돌하지 않는다.

```bash
cd ~/ksrc/ros
source /opt/ros/humble/setup.bash
ros2 launch nav2_ksrc.launch.py terrain:=sand mode:=teleop
```

두 번째 터미널에서 **기존** 마우스 조이스틱을 ROS 모드로 실행한다. 왼쪽 패드로
전후·게걸음, 오른쪽 패드 또는 Q/E로 회전하고 스페이스바로 멈춘다.
급격히 주행 방향을 바꾸면 텔레옵 브리지가 먼저 감속한다. 네 바퀴가 새 방향으로
조향된 뒤 구동하는 규칙은 학습·평가·RViz 주행에도 공통 적용된다.
`teleop_guard:=false`는 사전 감속만 끄고,
`drive_align_gate_deg:=0`은 공통 조향각 구동 게이트를 끈다.

```bash
cd ~/ksrc
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1
python3 pi/teleop/teleop_joystick.py --ros --max-lin 0.25 --max-ang 0.8
```

`terrain:=rock`은 암석 대회맵으로 바꾼다. 이 모드의 RViz는 맵과 위치를 보여주는
화면이고 목표 주행 명령은 받지 않는다. 기존 `--port /dev/ttyACM0` 방식은 실기
Nucleo용 직렬 경로로 그대로 유지된다. 규사맵에서는 최초 스폰과 `/reset_sim`
위치를 3° 이하 평탄 구역으로 제한한다. [세 모드의 전체 CLI](CLI.md#6-대회맵-주행-rviz-목표-또는-텔레옵).

### 펌웨어 · 실기 텔레옵

빌드·플래시·터미널 명령·조향 영점·구동모터 확인·조이스틱 주행은
**[firmware/CLI.md](firmware/CLI.md)**, 설정값은 [firmware/README.md](firmware/README.md).

```bash
cd firmware/src2026-rover-firmware && ./build.sh --flash
python3 pi/teleop/teleop_joystick.py --port /dev/ttyACM0 --max-lin 0.15 --max-ang 0.8   # 저장소 루트에서
```

---

## 설계상 중요한 선택

시뮬·학습 쪽 설계 결정(잔차 정책 구조, 흙 모델, 보상, 랜덤화 등)은
[sim/README.md](sim/README.md#설계상-중요한-선택)로 옮겼다.

---

## 현재 상태

| 항목 | 상태 |
|---|---|
| 펌웨어 | 플래시·터미널·기구학 주행·워치독은 **이전 구조**에서 실기 검증. 2026-09-29 이후 50Hz 제어주기·정렬 게이트·SPI2 링크·IMU·서보 피드백으로 재구성 — **미플래시, 실기 미검증**. 조향 영점·바퀴별 범위·구동모터 부호는 조립 후 확인 |
| 시뮬 물리 | 검증 13종 통과. 슬립모델이 고착 재현 확인 |
| 지형 | 실측 규사 경사지형(기복 450 mm, 최대 30°) 통계에 맞춘 절차 생성 확인 |
| 경로추종 | `s4_route_fix` 400만 스텝 모델은 미학습 규사 경로에서 순수 IK보다 횡오차·슬립·잔차 포화가 커졌다. 현재 보상 `balanced`는 수정 후 **아직 재학습·성능 검증 전** |
| 측위 | **미해결.** 엔코더+IMU 데드레커닝은 1.7 m 주행에 845 mm 오차(슬립 때문). D435 + 대회장 맵 ICP 가 기하적으로 가능함을 확인, 미구현 |
| Nav2 | **시뮬에서 동작 확인.** MPPI(Omni) 주행과 선택적 `rl:=true` 잔차 적용을 `/goal_pose` 종단 시험으로 확인. 정책 적용은 성능 개선 검증과 별개이며 실기는 측위 선행 |
| 텔레옵 + 대회맵 | 기존 pygame 조이스틱의 ROS 모드에서 `/cmd_vel`→MuJoCo 이동·패드 해제 후 정지, `/map` 발행 종단 시험 통과 |
| teacher-student | teacher 가 특권 잠재변수를 거의 안 씀(ablation 노이즈 수준) → 증류 보류 |

학습 결과 상세(`s3_path_reward` 검증 등)와 알려진 제약은 [sim/README.md](sim/README.md#학습-현황).
