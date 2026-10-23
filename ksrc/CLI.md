# CLI

이 문서는 **ROS2 / Nav2 / 대회맵 주행** 명령만 담는다. 런치는 `ros/`, 조이스틱은 저장소 루트에서 실행한다.

| 다른 명령 | 문서 |
|---|---|
| 학습·증류·평가·재생 (`sim/rl`) | [sim/CLI.md](sim/CLI.md) |
| 펌웨어 빌드·플래시·터미널·세팅·조이스틱 직결·Pi SPI | [firmware/CLI.md](firmware/CLI.md) |

---

## 0. 네 가지 조합 — 어느 명령을 쓸지

**목표를 어떻게 주느냐**(RViz 클릭 / 텔레옵) × **차체명령을 누가 내느냐**(순수 IK / RL 잔차)
의 조합이다. 어느 경우든 그 아래 **펌웨어와 같은 공유 C 기구학**
(`swerve_ik_compute` → `swerve_fold_to_range`)을 지나 MuJoCo 액추에이터로 간다.

| | 목표 | 차체명령 | 명령 | 상태 |
|---|---|---|---|---|
| **1** | RViz 2D Goal Pose | Nav2 MPPI (순수 IK) | `mode:=nav2` (기본) | ✅ |
| **2** | RViz 2D Goal Pose | Nav2 MPPI **+ RL 잔차** | `mode:=nav2 rl:=true` | ✅ |
| **3** | 텔레옵 (조이스틱) | 사람 명령 (순수 IK) | `mode:=teleop` | ✅ |
| **4** | 텔레옵 (조이스틱) | 사람 명령 **+ RL 잔차** | — | ❌ **불가** |

```bash
cd ~/ksrc/ros
source /opt/ros/humble/setup.bash
python3 make_maps.py    # 맵이 없거나 지형 생성 코드를 바꿨을 때만

# 1. 클릭으로 목표 -> 순수 IK
ros2 launch nav2_ksrc.launch.py terrain:=sand

# 2. 클릭으로 목표 -> RL 잔차 (**검증된 기본 모델 없음** -- 경로를 직접 줘야 한다)
ros2 launch nav2_ksrc.launch.py terrain:=sand rl:=true \
    rl_model:=<...>.zip rl_vecnorm:=<...>.pkl

# 3. 텔레옵 -> 순수 IK   (두 번째 터미널에서 조이스틱, 아래)
ros2 launch nav2_ksrc.launch.py terrain:=sand mode:=teleop
```

**한 번에 하나의 런치만** 띄운다. 암석 대회맵은 전부 `terrain:=rock`.

### 4번(텔레옵 + RL)이 왜 불가인가

런치가 **거부한다** (`nav2_ksrc.launch.py`):

```python
if mode == "teleop" and use_rl:
    raise RuntimeError("teleop 모드에는 /plan 이 없으므로 rl:=true 를 쓸 수 없습니다")
```

정책 관측에 **기준경로가 들어간다** — lookahead 2점, 횡이탈 `e_y`,
`cos/sin(e_psi)`, 선행 목표점, 선행거리. 이 값들은 Nav2 `/plan` 에서 나온다
(`ros/policy_observation.py`). 텔레옵에는 경로가 없으므로 `e_y`·`e_psi` 가
**정의되지 않는다.** 0 을 채우면 정책이 학습 때 본 적 없는 분포 밖 입력을 받는다.

구조적인 이유다 — 이 정책은 "주행 보조" 가 아니라 **경로추종 보정**이고, 보정할
경로가 있어야 존재할 수 있다.

> 하고 싶다면: 텔레옵 명령 방향으로 **가상 직선 경로를 합성**해서 넣으면 된다
> (`e_y = 0`, `e_psi` = 기수와 명령방향 차, lookahead 는 그 직선 위). 학습 분포에
> 직선·`e_y≈0` 구간이 있으므로 완전히 밖은 아니다. 약 30행 작업이고 아직 안 했다.

> [!danger] 2번의 기본 모델 — **없다** (2026-10-07)
> 런치의 `rl_model` 기본값을 **빈 값**으로 바꿨다. `rl:=true` 를 주면
> `rl_model`·`rl_vecnorm` 을 명시하라는 오류가 난다. 검증 완료된 배포 정책이
> 아직 없기 때문이다.
>
> **`s13` 채택은 철회됐다.** 이전에 여기 적혀 있던 "순수 IK 를 유의하게 이긴
> 유일한 체크포인트" 는 틀린 판정이었다:
>
> | 재판정 | 결과 |
> |---|---|
> | lookahead 를 튜닝하면 | 횡오차 이득이 `L=0.264` −1.88mm → `0.20` −0.85 → `0.16` −0.27 → `0.12` **+0.18mm** 로 단조 소멸 |
> | 경사 경로집합 (이전 33개 비교에 0건) | **13개 지표 전부 유의하게 악화, 완주율 −6%p** |
> | 비용 | 포화 +13~15%p, 침하 +4~13, 진행률 −0.3%p — 줄지 않았다 |
>
> 즉 `s13` 의 이득은 잘못 맞춰진 `lookahead` 상수 하나를 때워준 것이었다.
> 기준선을 고친 뒤(`L` 0.18~0.45 변동 → **0.20 고정**) 남는 기여는 기수오차
> −0.5~−0.9° 뿐이고, 그조차 **고전 P 제어기 게인 하나가 −1.78° 로 더 잘한다**
> (`sim/rl/pfb_baseline.py --task path --signal psi --gains 6`).
>
> **현재 가장 나은 cmdvel 모델 `s51` 도 여기 넣으면 안 된다.** 223D **특권
> 관측**(시뮬 참속도 포함)을 보는 teacher 라서 실기 센서로는 관측을 만들 수
> 없다. 실기 배포에는 자기수용감각만 쓰는 student 증류와 그 student 의 독립
> 재평가가 먼저다.
>
> 전체 기록은 `sim/rl/TEACHER_CMDVEL_REPORT.md` (s36~s54) 와
> [[RL 주행보조 진행상황]] 2026-10-05/06 에 있다.

---

## 1. 대회맵 주행: RViz 목표 또는 텔레옵

실행 명령은 **0절의 표**에 있다. 이 절은 각 모드가 무엇을 띄우고 무엇을 확인하는지,
그리고 겪은 문제들을 설명한다.

**텔레옵 모드(3번)**는 런치를 켜 둔 채 두 번째 터미널에서 기존 마우스 조이스틱을
ROS 모드로 실행한다. 왼쪽 패드는 전후·게걸음, 오른쪽 패드 또는 Q/E는 회전,
스페이스바는 정지다. 패드에서 손을 떼거나 창 포커스를 잃으면 0 명령을 보낸다.
텔레옵에서 큰 방향 전환 중에는 차체가 거의 멈출 때까지 최대 0.5초 감속한다.
조향 목표가 한 주기에 60° 넘게 바뀌면 네 바퀴가 새 조향각의 10° 이내에
들어온 뒤 구동하는 규칙은 모든 모드의 `RoverEnv`에 공통 적용된다.

```bash
cd ~/ksrc
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1
python3 pi/teleop/teleop_joystick.py --ros --max-lin 0.25 --max-ang 0.8
```

텔레옵 런치는 MuJoCo 뷰어와 RViz의 대회맵·로버 위치를 함께 띄우며,
`map_server`만 활성화한다. Nav2 컨트롤러가 없어 `/cmd_vel` 발행자가 충돌하지
않는다. 이 모드에서 RViz의 2D Goal Pose는 주행 명령이 아니다. 조이스틱이
직접 `/cmd_vel`을 낸다. MuJoCo 창이 부담되면 `view:=false`를 런치에 추가해
RViz 맵만 볼 수 있다. 종료 후에도 명령이 끊기면 브리지의 0.5초 워치독이 정지한다.
규사맵의 최초 스폰과 `/reset_sim` 복귀 위치는 3° 이하 평탄 구역으로 제한하며,
경사 경계에서 로버 바퀴가 걸치지 않도록 풋프린트만큼 여유를 둔다. 같은 시드로
리셋하므로 복귀 위치는 매번 같다. 학습·평가 에피소드의 시작 위치 분포는 바꾸지 않았다.
텔레옵의 사전 감속만 끄려면 `teleop_guard:=false`, 조향각 구동 게이트만
끄려면 `drive_align_gate_deg:=0`을 런치에 추가한다. 이전 두 기능 모두 없던
동작과 비교하려면 둘 다 지정한다.
ROS 발행 옵션은 현재 `teleop_joystick.py`에 추가했다. `teleop_keyboard.py`와
`teleop_terminal.py`는 기존 Nucleo 직렬 프로토콜용이다.

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1
ros2 topic echo /cmd_vel --once          # 조이스틱을 움직인 상태에서 실행
ros2 topic echo /cmd_vel_applied --once  # 실제 IK 입력
ros2 topic echo /odom --once             # 맵 위 로버 위치
```

현재 저장소의 ROS 런치 파일은 `ros/nav2_ksrc.launch.py` 하나다. 기본 `nav2`
모드에서는 MuJoCo 브리지, `map_server`, `planner_server`, `controller_server`
(MPPI Omni), `behavior_server`, `bt_navigator`, `lifecycle_manager_navigation`,
RViz를 실행한다. `teleop` 모드에서는 브리지·맵 서버·RViz만 실행한다.
`amcl`과 `velocity_smoother`는 파라미터 파일에 항목이 있지만 현재 런치에서는 실행하지
않는다. 측위 대신 브리지가 `map → odom` 항등 변환과 참값 `odom → base_link`를 낸다.
Nav2 컨트롤러의 `/cmd_vel`은 브리지가 받아 공유 스워브 IK를 거쳐 **MuJoCo 조향·구동
액추에이터**에 적용한다. 브리지는 기본적으로 IK 전에 차체 명령을 0.10초 시정수로
저역통과하고 그 결과를 `/cmd_vel_filtered`에도 발행한다. 0 명령은 지연 없이 정지한다.
기본은 순수 IK(잔차 0)다. `rl:=true`로 띄우면 모델과 짝인 `vecnorm.pkl`을 로드하고
Nav2 `/plan` 기준 관측으로 3D 잔차를 계산한다. **`rl_model`/`rl_vecnorm` 을 반드시
명시한다** — 기본값이 `runs/s2/final.zip` 인데 `s2` 는 조향범위 ±180°, 옛 보상,
옛 `e_psi` 기준으로 학습해 지금 환경과 맞지 않는다 (0절 주의 참고).
`/rl_action`은 정규화 행동, `/rl_delta`는 요청된 차체속도 보정,
`/cmd_vel_applied`는 포화 처리 후 실제 IK 입력이다.
이 런치만으로 실제 STS/Cytron 모터에 명령이 나가지는 않는다.
STM32의 현재 `Swerve_Drive()`는 조향 명령과 구동 PWM을 바로 보내므로,
실기에서도 같은 정렬 대기를 쓰려면 서보 각 피드백·캘리브레이션을 확인한 뒤
펌웨어 구동 제어에 별도로 반영해야 한다.

```bash
cd ~/ksrc/ros
source /opt/ros/humble/setup.bash
ros2 launch nav2_ksrc.launch.py terrain:=sand rl:=true
# 다른 모델: rl_model:=... rl_vecnorm:=...
# 필요하면 잔차만 절반으로 줄여 확인: rl_scale:=0.5
# 주행 중 켜고 끄기: ros2 service call /set_rl std_srvs/srv/SetBool '{data: false}'
```

RViz의 **2D Goal Pose**로 목표를 준 뒤 아래 토픽을 보면 정책이 적용됐는지
확인할 수 있다. `topic echo`는 계속 실행되므로 관심 있는 토픽마다 별도 터미널을
쓴다. 목표에 도착해 `/cmd_vel`이 0이면 정책 잔차도 0이다.

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1
ros2 topic echo /plan --once             # Nav2 경로 수신
ros2 topic echo /rl_action                # [Delta vx, Delta vy, Delta omega] 정규화 행동
ros2 topic echo /rl_delta                 # 요청 보정: m/s, m/s, rad/s
ros2 topic echo /cmd_vel_filtered         # 보정 전 Nav2 명령
ros2 topic echo /cmd_vel_applied          # 보정 후 IK 입력
ros2 service call /set_rl std_srvs/srv/SetBool '{data: false}'
ros2 service call /set_rl std_srvs/srv/SetBool '{data: true}'
```

필터 명령과 최종 명령은 포화 처리 때문에 `/rl_delta`만큼 정확히 차이 나지 않을
수 있다. 정책은 pure pursuit 출력으로 학습됐으므로 Nav2 MPPI에서 성능 향상을
보장하는 실험은 아니다. 실제 주행 결과는 순수 IK와 같은 목표에서 비교한다.

### 왜 한 런치로 합쳤나

브리지와 Nav2를 각각 실행하던 구성에서는 브리지 누락, 서로 다른 지형·맵, 서로 다른
ROS 도메인 때문에 Nav2가 `/odom`·TF를 못 받거나 다른 실행의 노드와 섞일 수 있었다.
실패 당시 로그에는 `map_server` 활성화 실패, 오래된 `/odom` 타임스탬프, 현재 맵과
크기가 다른 맵이 함께 보였다. **다른 ROS 실행과의 충돌은 유력한 원인이지만 로그만으로
단정할 수는 없다.** 통합 런치는 같은 프로세스 묶음에서 지형에 맞는 맵을 고르고,
모든 자식에 `ROS_DOMAIN_ID=77`, `ROS_LOCALHOST_ONLY=1`을 설정해 이 문제를 피한다.
ROS 자체가 한 런치를 요구하는 것은 아니다. 아래처럼 분리 실행해도 같은 환경·맵·
라이프사이클 설정을 맞추면 된다. 기존 맵 파일 누락은 `make_maps.py`로 생성한다.

### 왜 2D Goal Pose 뒤에 조향이 튀었나

RViz의 **2D Goal Pose는 차체 속도나 바퀴각을 직접 보내지 않는다.** 클릭 위치가 목표
`x, y`, 드래그 방향이 최종 yaw이며, Nav2가 경로와 `/cmd_vel`을 계산한다. 이전 시뮬
조향 액추에이터는 큰 목표각을 즉시 받는 위치 제어(`kp=12`, `kv=0`)였다. 고정된
−90° 목표각 시험에서도 실제 바퀴가 약 −132°까지 지나치고 최대 35.6 rad/s로 움직였다.
따라서 MPPI 출력이 변하기 전에 조향 모델 자체가 과도하게 반응했다. 현재 코드는
`steer_kv=0.30` 감쇠를 넣고 목표각을 제어 주기마다 최대 `5.2 rad/s ÷ 50 Hz`만큼
진행시킨다. 같은 스텝 시험에서 초과각은 0.14°, 최대 속도는 약 4.7 rad/s였다.
빠르게 변하는 명령 시험에서는 최대 5.18 rad/s였다. 이는 시뮬 단위 시험 결과이며
RViz GUI에서 연속 주행한 결과를 뜻하지 않는다. 저장된 RL 정책의 성능도 변경된
조향 물리에서 다시 평가해야 한다.

RViz에서 **차체 위치가 순간이동하던 문제**도 따로 있었다. 전복 후 브리지가 자동으로
환경을 리셋해 스폰 위치를 다시 발행했기 때문이다. 지금은 전복 시 구동을 멈추고
현재 위치를 유지하며, `/reset_sim`을 호출할 때만 원래 스폰으로 리셋한다.

목표를 눌러도 움직이지 않는 현상은 별개로 확인한다. Nav2 노드가 `active`가 아니면
목표가 수행되지 않는다. 또 목표가 시작 위치에서 0.15 m 이내이고 방향 차이가 0.50 rad
이내면 현재 목표 허용오차 안이라 즉시 `Reached the goal!`이 된다. 실제 시험 중 한
목표는 시작 위치에서 약 0.04 m 떨어져 있었다. 먼저 충분히 떨어진 통과 가능 지점을
클릭하고 드래그한다. RViz의 **2D Pose Estimate**는 이 브리지의 스폰 위치를 바꾸지
않으며, `Publish Point`는 현재 웨이포인트 명령으로 연결되지 않았다.

### 통합 실행 (권장)

```bash
source /opt/ros/humble/setup.bash
cd ~/ksrc/ros
python3 make_maps.py                         # 맵이 없거나 지형/맵 생성 코드를 바꿨을 때
ros2 launch nav2_ksrc.launch.py terrain:=sand
```

`terrain:=rock`으로 암석 맵·지형을 선택한다. `rviz:=false`는 RViz만,
`view:=false`는 MuJoCo 창만 끈다. `cmd_filter_tau:=0`이면 이번 차체 명령 필터를
꺼서 기존 동작과 비교할 수 있다. 이 필터는 **브리지 안에만** 있으며 Nav2의
`velocity_smoother` 노드를 실행하는 것은 아니다. `domain_id:=78`처럼 도메인을 바꾸면 별도
터미널의 `ROS_DOMAIN_ID`도 같은 값으로 맞춘다. **통합 런치의 도메인 설정은 그
런치가 띄운 자식 프로세스에만 적용된다.** 세 번째 진단 터미널에는 자동으로
전파되지 않는다. 종료는 런치를 실행한 터미널에서
Ctrl-C를 누른다. 동일한 Nav2나 브리지를 중복 실행하지 않는다.

### 브리지와 Nav2 런치를 분리 실행

두 터미널 **각각** 먼저 다음 환경을 설정한다. 별도 `ros2` 진단 터미널도 같다.

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=77
export ROS_LOCALHOST_ONLY=1
cd ~/ksrc/ros
```

터미널 A: 브리지. `--view`를 빼면 MuJoCo 창 없이 실행된다. 기본 `--seed 0`,
`--d 0.5`, `--cmd-filter-tau 0.10`이며, 지형은 Nav2 런치와 같아야 한다.
필터 비교 시 `--cmd-filter-tau 0`을 준다. 정책 시험은 아래 둘째 명령처럼
모델과 vecnorm을 함께 지정한다.

```bash
python3 mujoco_bridge.py --terrain sand --view
python3 mujoco_bridge.py --terrain sand --view \
  --model ../sim/rl/runs/s2/final.zip --vecnorm ../sim/rl/runs/s2/vecnorm.pkl
```

터미널 B: 같은 지형의 Nav2와 RViz. 이미 브리지를 띄웠으므로 `bridge:=false`를 준다.
맵이 없다면 먼저 `python3 make_maps.py`를 실행한다.

```bash
ros2 launch nav2_ksrc.launch.py terrain:=sand bridge:=false
```

### 런치 없이 노드를 하나씩 실행

각 명령을 **서로 다른 터미널**에서 실행한다. 모든 터미널에 위의 `source`,
`ROS_DOMAIN_ID`, `ROS_LOCALHOST_ONLY`, `cd`를 반복한다. 맵 파일이 없다면 먼저
`python3 make_maps.py`를 실행한다. 실행 순서는 브리지 → Nav2 서버 5개 →
라이프사이클 매니저 → RViz다. `map_server`의 `yaml_filename`은 런치가 덮어쓰던
값이므로 수동 실행에서는 반드시 실제 맵 파일을 준다. 암석 지형이면 브리지의
`sand`와 맵 파일명의 `sand`를 함께 `rock`으로 바꾼다.

```bash
# 터미널 1: MuJoCo 브리지
python3 mujoco_bridge.py --terrain sand --view

# 터미널 2: 정적 맵 서버
ros2 run nav2_map_server map_server --ros-args \
  --params-file "$PWD/nav2_params.yaml" \
  -p yaml_filename:="$PWD/maps/arena_sand.yaml"

# 터미널 3: 전역 경로 계획
ros2 run nav2_planner planner_server --ros-args \
  --params-file "$PWD/nav2_params.yaml"

# 터미널 4: MPPI Omni 경로 추종 → /cmd_vel
ros2 run nav2_controller controller_server --ros-args \
  --params-file "$PWD/nav2_params.yaml"

# 터미널 5: 복구 행동
ros2 run nav2_behaviors behavior_server --ros-args \
  --params-file "$PWD/nav2_params.yaml"

# 터미널 6: 목표 행동 트리
ros2 run nav2_bt_navigator bt_navigator --ros-args \
  --params-file "$PWD/nav2_params.yaml"

# 터미널 7: 위 5개 Nav2 서버를 configure → activate
ros2 run nav2_lifecycle_manager lifecycle_manager --ros-args \
  -r __node:=lifecycle_manager_navigation \
  -p use_sim_time:=false -p autostart:=true \
  -p 'node_names:=[map_server,planner_server,controller_server,behavior_server,bt_navigator]'

# 터미널 8: RViz
ros2 run rviz2 rviz2 -d "$PWD/rviz/ksrc.rviz" --ros-args -p use_sim_time:=false
```

라이프사이클 매니저 없이 서버 실행 명령만 입력하면 Nav2 서버는 `unconfigured`에
머물러 주행할 수 없다. 현재 런치와 동일하게 `use_sim_time:=false`를 유지한다.

### 목표를 보내기 전 확인

통합 런치를 썼더라도 진단용 터미널에서 다음 환경을 다시 설정한다.
`domain_id:=`를 바꿨다면 `77` 대신 그 값을 쓴다.

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=77
export ROS_LOCALHOST_ONLY=1
ros2 node list --no-daemon
ros2 topic info /cmd_vel -v --no-daemon
```

`node list`가 비어 있으면 도메인이 다르거나 런치가 종료된 것이다. 노드는
보이는데 `/cmd_vel` 발행자가 없다면 `controller_server`가 실행·활성화됐는지
확인한다. 발행자가 있어도 목표 주행 중이 아니면 `topic echo`는 새 메시지가
올 때까지 기다린다. 다음 명령을 한 번에 하나씩 실행한다.
`topic hz`와 끝나지 않는 `topic echo`는 Ctrl-C로 종료한다. `map_server`와
`bt_navigator`가 `active`이고 `/odom`이 계속 갱신돼야 한다. RViz에서
**2D Goal Pose**를 눌러 목표점에서 클릭·드래그한다. 전복으로 구동이 잠겼을 때는
`/reset_sim`을 호출하면 처음 `--seed`의 스폰으로 돌아간다.

**클릭한 지점이 목적지이고, 드래그는 목적지에서의 최종 방향만 정한다.** 로버 위를
클릭한 뒤 멀리 드래그해도 목적지는 로버 근처다. 최근 로그에서도 현재 위치
`(1.36, 0.14)`에 대해 목표가 `(1.34, 0.14)`여서 15 cm 목표 허용오차 안이었다.
구동 확인은 로버에서 0.5 m 이상 떨어진 주행 가능 지점을 먼저 클릭한다.

```bash
ros2 lifecycle get /map_server
ros2 lifecycle get /bt_navigator
ros2 topic hz /odom
ros2 topic echo /goal_pose --once             # 실행한 다음 RViz에서 목표를 찍는다
ros2 topic echo /odom --once                  # 현재 로버 좌표와 비교
ros2 topic echo /cmd_vel --no-daemon         # Nav2 원본 명령. 목표를 찍기 전에 켠다
ros2 topic echo /cmd_vel_filtered --no-daemon  # RL 전 필터 명령
ros2 topic echo /cmd_vel_applied --no-daemon   # IK 에 들어간 최종 명령
ros2 topic echo /rl_action --no-daemon         # 정책 행동 (rl:=true 시)
ros2 service call /reset_sim std_srvs/srv/Trigger '{}'
```

---

## 2. 진단·비교용 런치 옵션 (2026-10-03 추가)

### `/plan` 꺾임각 분포 재기 — **지금 가장 급한 측정**

학습·평가 경로를 만드는 `sim/rl/path.py` `make_path` 가 막혔을 때 선회각 제한을
170° 까지 풀어서, 실측 꺾임각이 **p90 148~166°, 최대 168°** 다 (헤어핀 — 로버가
멈춰서 제자리 회전을 해야 통과한다). Nav2 planner 는 그런 경로를 내지 않는다.
그 차이가 순수 IK 기준선·`s9`·`s10` 판정 전부에 실려 있다.

```bash
# 터미널 1: Nav2 (1번과 동일)
cd ~/ksrc/ros && source /opt/ros/humble/setup.bash
ros2 launch nav2_ksrc.launch.py terrain:=sand

# 터미널 2: /plan 집계
cd ~/ksrc/ros && source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1
python3 plan_stats.py
#  -> RViz 의 2D Goal Pose 로 목표를 10~20개 찍고 Ctrl-C
#  -> 누적 꺾임각 분포 + make_path 와의 비교가 출력된다
```

`--resample` 기본값 0.264 m 는 **pure pursuit 의 선행거리 L** 과 같다. Nav2 경로는
점 간격이 코스트맵 해상도(0.02 m 급)라 점마다 재면 양자화 잡음이 지배한다
(직각 90° 1회를 0.02 m 로 재면 평균 0.9°, 0.264 m 로 재면 15.0°).

### 조향 가동범위 바꿔 보기

```bash
ros2 launch nav2_ksrc.launch.py terrain:=sand mode:=teleop steer_limit_deg:=135
```

| 한계 | 이력 | 게이트 | 슬립 | `\|조향각\|>110°` (모터가 타이어 밖) |
|---|---|---|---|---|
| ±100° (기본) | 20° | 10.85% | 0.198 | **0.00%** |
| ±135° | 90° | **7.85%** | **0.175** | **3.25%** |

±135° 는 게이트·슬립을 개선하지만 주행 중 3.25% 의 시간에 **모터가 로버의 최외곽**이
된다 (최대 돌출 36.5 mm, 좌우 전폭 328→401 mm). 그 최외곽이 굴러가는 타이어가 아니라
고정 하우징이라 턱에 걸린다. 180° 반전 횟수는 **범위와 무관하게 360° 스윕 당 2회로
고정**이고 위치만 밀린다 (±100 → 100°/280°, ±135 → 136°/315°).

### 접기 선택을 "모터가 차체 안쪽" 으로 바꿔 보기

```bash
ros2 launch nav2_ksrc.launch.py terrain:=sand mode:=teleop mech_fold:=true
```

조향각 θ 와 θ±180° 는 바퀴 속도 부호를 뒤집으면 **차체 운동이 완전히 동일**하다.
구동모터가 바퀴 축과 동축이라 θ=±90° 에서 차체 전방/후방을 향하므로, **앞 모듈은
뒤로, 뒤 모듈은 앞으로** 접으면 네 모터가 가운데로 모인다:

| 횡걸음 조합 | 모터 외접반경 | 전후 치수 |
|---|---|---|
| `+90 +90 +90 +90` (기본 규칙이 내는 것) | 217.3 mm | 364.6 mm |
| `FL−90 FR+90 RL+90 RR−90` (`mech_fold:=true`) | **118.4 mm** | **2.6 mm** |

> [!warning] **채택안이 아니다.** 파이썬 시뮬 전용이고 공유 C 에 없다
> in-loop 짝비교(100 에피소드, 순수 IK): 완주율 **96.0 → 71.0%**, 고착 **3 → 28%**,
> 조향 대기 9.78 → 17.90%. 이력이 없어서 명령 급변 지점마다 분기를 추가 전환한다.
>
> 단 그 측정은 **헤어핀 경로 위에서 나온 것**이라 기각 증거가 아니다 — 현실적인
> 경로에서 재측정해야 한다. **텔레옵(매끄러운 입력)에서는 유효한 거래**다:
> 360° 스윕 당 반전 2 → 4회, 반전 1회가 약 0.6초, 그리고 게이트 트리거는
> `driving` 을 요구하므로 속도 0 으로 지나면 공짜다.
> 자세한 것은 [[조향 회전 제한 설계]].

### 게이트 끄고 비교

```bash
ros2 launch nav2_ksrc.launch.py terrain:=sand mode:=teleop drive_align_gate_deg:=0
ros2 launch nav2_ksrc.launch.py terrain:=sand mode:=teleop teleop_guard:=false
```

조합도 된다: `mech_fold:=true steer_limit_deg:=135 drive_align_gate_deg:=0`
