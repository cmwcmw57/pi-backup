# KSRC 로버 펌웨어 (STM32 Nucleo-F446RE)

4륜 독립조향 스워브 로버의 저수준 제어기. 차체 속도 명령 `(vx, vy, ω)` 를 받아
바퀴 4개의 조향각(STS3215 서보)과 구동(MDD3A + 기어드 DC 모터)으로 바꾸고,
엔코더·서보·IMU 값을 Raspberry Pi 로 올려 보낸다.

> 실행 명령만 필요하면 **[CLI.md](CLI.md)**. 이 문서는 설정값과 구조의 정본이다.
> 상태 기준일: 2026-10-03. 2026-09-29 이후 변경(50Hz 제어주기, 정렬 게이트, SPI2 링크,
> IMU, 서보 피드백)은 **빌드는 되지만 실기 검증 전**이다.

---

## 1. 폴더 구조

```
firmware/
├─ README.md                  이 문서 (설정값 정본)
├─ CLI.md                     실행 명령 모음
├─ common/                    MCU 비의존 순수 C. 호스트 gcc 테스트 + 시뮬이 ctypes 로 같은 코드 호출
│   swerve_kinematics.c         역기구학 + 가동범위 안 등가해 선택 (swerve_fold_to_range)
│   drive_align_gate.c          큰 조향 전환 중 구동 보류
│   wheel_pi.c                  바퀴 속도 FF + 기준 모델 PI (WPI ON 일 때 구동 duty)
│   spi_link.c                  Pi <-> Nucleo SPI 144바이트 프레임 (CRC16)
│   mdd3a_driver.c              m/s -> 정규화 duty
│   sts3215_protocol.c          (시뮬 전용) 서보 패킷
│   teleop_protocol.c           (시뮬 전용) UART 바이너리 프레임
│   tests/                      호스트 유닛테스트
└─ src2026-rover-firmware/    보드에 올라가는 프로젝트 (Makefile + build.sh)
    Core/Src/main.c             50Hz 제어주기, 터미널 명령, 텔레메트리 조립
    Core/Src/joint.c            서보 각도 <-> 위치값, 바퀴별 가동범위, 피드백 읽기
    Core/Src/sts3215.c          STS3215 버스 (USART3 반이중)
    Core/Src/motor.c/encoder.c  MDD3A PWM 8채널, 엔코더 타이머 4개
    Core/Src/pi_spi.c           SPI2 슬레이브 (Pi 링크, DMA)
    Core/Src/imu_ism330.c       ISM330DHCX (SPI3 마스터)
    Core/Src/{swerve_kinematics,drive_align_gate,spi_link,mdd3a_driver,wheel_pi}.c
                                -> ../../../common/ 심볼릭 링크
```

`common/` 의 C 를 고치면 시뮬도 같이 바뀐다 (`sim/build.sh` 로 `.so` 재빌드).

---

## 2. 하드웨어와 핀

MCU 클럭: **HSI 16 MHz, PLL 없음** (APB1 = APB2 = 16 MHz).
USART2/USART3/SPI2/SPI3 는 CubeMX 가 아니라 **레지스터로 직접** 설정한다.
`.ioc` 로 Generate Code 를 하면 `MX_SPI2_Init()`/`MX_SPI3_Init()` 이 생기는데 호출하지
않는다 (HAL SPI 가 Makefile 에 없어 링크도 깨진다).

| 기능 | 주변장치 | 핀 | 설정 |
|---|---|---|---|
| 터미널 / 조이스틱 | USART2 (ST-Link VCP) | PA2 TX, PA3 RX | 115200 8N1, RX 인터럽트 |
| 조향 서보 버스 | USART3 반이중 | PB10 (오픈드레인+풀업) | 1 Mbps |
| Pi 링크 | SPI2 슬레이브 + DMA1 S3/S4 | PB12 CS (GPIO+EXTI, 소프트웨어 NSS), PB13 SCK, PC2 MISO, PC3 MOSI | 모드 0, 8비트, Pi 쪽 1 MHz 권장 (상한 8 MHz) |
| IMU | SPI3 마스터 | PC10 SCK, PC11 MISO, PC12 MOSI, PC9 CS | 모드 0, 1 MHz |
| 구동 MOTOR1 (LF) | TIM2 CH1/CH2 | PA5 / PB3 | MDD3A #1 M1 (A/B) |
| 구동 MOTOR2 (LB) | TIM12 CH1/CH2 | PB14 / PB15 | MDD3A #1 M2 |
| 구동 MOTOR3 (RF) | TIM3 CH1/CH2 | PA6 / PA7 | MDD3A #2 M1 |
| 구동 MOTOR4 (RB) | TIM3 CH3/CH4 | PB0 / PB1 | MDD3A #2 M2 |
| 엔코더 1~4 (LF, LB, RF, RB) | TIM1 / TIM4 / TIM5 / TIM8 | PA8·PA9 / PB6·PB7 / PA0·PA1 / PC6·PC7 | x4 쿼드러처 (TIM5 만 32비트) |

PWM: ARR 799 → 약 20 kHz, duty -799 ~ +799. MDD3A 는 PWM+PWM 방식 (정방향 A, 역방향 B).
서보 VCC 는 STM32 에 연결하지 않는다 (12V 외부 전원, GND 만 공통).

---

## 3. 제어 흐름 (50 Hz)

```
UART "V vx vy ω"  ─┐
SPI2 명령 프레임   ─┴─> 목표 (vx, vy, ω) 만 갱신
                          │
20 ms 마다 (main.c while 루프)
  1. 엔코더 속도 추정 (dt 는 DWT 사이클 카운터)
  2. 센서: 서보 4개 피드백 (위치·속도·부하·전압·온도·상태), IMU 1샘플
  3. 워치독: 500 ms 무명령 -> 목표 0 (조향각은 유지)
  4. swerve_ik_compute         -> 모듈별 (각, 속도), 바퀴 속도 상한 초과 시 전체 비율 축소
  5. swerve_fold_to_range      -> 바퀴별 가동범위 안의 등가각 (θ 또는 θ±180° + 역회전)
  6. 조향 SYNC_WRITE           -> 모듈 순서를 서보 순서로 재배열
  7. drive_align_gate          -> 큰 조향 전환 중이면 구동 0
  8. 구동 PWM                  -> WPI ON: wheel_pi_step (목표 바퀴 속도 vs 1 의 엔코더 속도)
                                  WPI OFF: duty = v/0.378×799 (개루프).  MOTOR_DRIVE_SIGN 적용
  9. 텔레메트리 프레임          -> 다음 SPI 전송에 실림
```

3~9 는 텔레옵(첫 구동 명령)이 시작된 뒤에만 돈다. 그 전에는 `T`, `MOT`, `MTEST` 같은
수동 명령이 액추에이터를 쓴다.

---

## 4. 설정값

**확정** = 실측·결정됨, **임시** = 조립 후 확인 필요, **가정** = 데이터시트/추정.

### 4.1 서보 배치와 조향

| 설정 | 위치 | 현재 값 | 상태 |
|---|---|---|---|
| 서보 ID ↔ 바퀴 | `joint.c` `joint_config[]` | 서보1 LF, 서보2 LB, 서보3 RF, 서보4 RB | 확정 |
| 모듈 → 서보 | `main.c` `SWERVE_MODULE_SERVO` | `{0, 2, 1, 3}` (FL→서보1, FR→서보3, RL→서보2, RR→서보4) | 확정 |
| 조향 영점 | 서보 EEPROM (`ZERO`) | 미기록 | **임시** — 조립 후 |
| 영점 위치값 | `joint.c` `.offset` | 2048 (`ZERO` 가 현재 자세를 2048 로 기록) | 확정 |
| 조향 방향 | `joint.c` `.dir` | 전부 +1 | **임시** — `T 30` 이 위에서 반시계면 +1. 뒤쪽 서보(LB, RB)는 앞쪽과 **월드 z축 기준 180°** 돌려 장착(둘 다 아래를 봄) → 회전 방향은 같으므로 네 서보 `dir` 부호는 같아야 함. 영점 차이는 `ZERO ALL` 이 흡수 |
| 가동범위 | `joint.c` `.min_deg/.max_deg` | 전부 -90° ~ +90° (URDF 관절 한계와 같게, 2026-10-11) | 조립 후 실측. 폭 ≥ 180° 유지 |
| 시뮬 가동범위 | `sim/rl/config.py` `steer_lo_deg/steer_hi_deg`, Isaac `contract.STEER_LIMIT_RAD` | 전부 ±90° (순서 FL, FR, RL, RR) | 펌웨어와 **같이** 고칠 것 |
| 조향 0° 의 뜻 | — | 바퀴 정면, 구동모터 몸체 안쪽, + 는 위에서 반시계 | 확정 (시뮬과 동일) |
| 서보 최고속 | `joint.c` `joint_goal_speed` | 0 = 서보 최대 (제한 없음). 터미널 `S` 로 변경 | 단위 step/s, 4096 step = 360° |
| 서보 가속도 | `joint.c` `joint_init()` | 50 | 가정 (단위 미실측) |
| 서보 수신 타임아웃 | `sts3215.c` `STS_RX_TIMEOUT_MS` | 3 ms | 확정 |
| 서보 읽기 실패 재시도 | `main.c` `SERVO_RETRY_TICKS` | 25 주기 (0.5 s), 한 주기에 1개만 재시도 | 확정 |

±90° 에서 모든 방향이 나온다: 한 방향의 해는 θ(정회전)와 θ±180°(역회전) 두 개이고
가동 폭이 정확히 180° 라 항상 하나는 범위 안이다. 다만 두 해가 겹치는 구간이 없어서
(예전 ±100° 때는 ±80°~±100° 가 겹쳐 히스테리시스가 있었다) 바퀴 방향이 ±90° 를
넘나들면(순수 옆걸음 근처) 180° 반전이 일어나고, 그동안 구동은 정렬 게이트가 멈춘다.

### 4.2 구동과 엔코더

| 설정 | 위치 | 현재 값 | 상태 |
|---|---|---|---|
| 모터 번호 ↔ 바퀴 | — | MOTOR1 LF, MOTOR2 LB, MOTOR3 RF, MOTOR4 RB (조향 서보와 같은 배치) | **확정 2026-10-03** |
| 모듈 → 구동모터 | `main.c` `SWERVE_MODULE_MOTOR` | `{MOTOR1, MOTOR3, MOTOR2, MOTOR4}` (FL, FR, RL, RR) | 확정 |
| 구동 부호 | `main.c` `MOTOR_DRIVE_SIGN[MOTOR1..4]` | −1, −1, +1, +1 (LF, LB, RF, RB) | 2026-10-03. 모터를 넷 다 같은 방식으로 달아 **좌우 대칭** (L −1, R +1). RF 는 −1 로 관찰돼 확인 필요 |
| 엔코더 부호 | `main.c` `ENCODER_SIGN[MOTOR1..4]` | −1, −1, +1, +1 | MOTOR1 이 duty + 에서 tick + (실측) → 구동 부호와 같음. `V 0.05 0 0` 에서 m/s 가 넷 다 + 인지 확인 |
| 기어비 | `main.c` `GEAR_RATIO` | 131 (SPG30E-GR131) | 확정 |
| 엔코더 PPR | `main.c` `ENCODER_PPR` | 16 (x4 → 64, 바퀴 1회전 8384 틱) | **확정 2026-10-03** (무부하 77.8 rpm) |
| 바퀴 반경 | `main.c` `WHEEL_RADIUS_M` | 0.047503 m (URDF rover_wheel_rev4) | 2026-10-12 URDF·Isaac 과 맞춤 (옛 0.03995 는 rev_2, 19% 작았음) |
| 바퀴 최고속 | `main.c` `SWERVE_MAX_WHEEL_MPS` | 0.378 m/s (= duty 100%, 7.96 rad/s × 반경) | 시뮬과 일치시킨 가정. 실측 무부하 77.8 rpm → 0.387 m/s (+2%). D1 로 확정 |
| 모듈 위치 | `main.c` `SWERVE_MODULES` | (±0.105, ±0.1325) m (조향축 = 바퀴 중심) | **실측 2026-10-03**. 시뮬 `axle_x=0.105`, `track=0.265` 와 동일 |

조립 전 날것 방향 기록 (`MOT`, 부호 보정 없음, 샤프트 끝에서 볼 때):

| 모터 | duty + | 엔코더 | 날짜 |
|---|---|---|---|
| MOTOR1 | 반시계 | + | 2026-10-03 |
| MOTOR2~4 | | | |

### 4.3 제어주기와 안전

| 설정 | 위치 | 현재 값 |
|---|---|---|
| 제어주기 | `main.c` `CTRL_PERIOD_MS` | 20 ms (50 Hz, 시뮬 `CTRL_HZ` 와 동일) |
| 명령 워치독 | `main.c` `TELEOP_TIMEOUT_MS` | 500 ms → 목표 0 |
| 정렬 게이트 트리거 | `main.c` `DRIVE_GATE_TRIGGER_DEG` | 60° (목표 급변, 또는 목표-실측 차이) |
| 정렬 게이트 해제 | `main.c` `DRIVE_GATE_RELEASE_DEG` | 10° (네 바퀴 실측각 기준) |
| 정렬 게이트 타임아웃 | `main.c` `DRIVE_GATE_MAX_HOLD_MS` | 1500 ms → 강제 해제 + `GATE_TIMEOUT` 플래그 |
| 모터 자가시험 | `main.c` `s_mtest_enabled` | 기본 OFF (`MTEST` 로만) |
| 로그 | `main.c` `s_log_enabled` | 부팅 시 ON, 텔레옵 시작 시 자동 OFF |

안전장치 요약:
- 손 떼면(속도 0) 조향각을 유지한다. 0° 로 튀지 않는다.
- 조향 해는 항상 가동범위 안. `joint.c` 가 한 번 더 클램프한다 (`T` 포함).
- 서보는 위치값 사이를 직선으로 움직여, 범위 안 두 각 사이의 이동 경로도 범위 안이다.
- 큰 조향 전환 중에는 구동 0 (정렬 게이트). 최대 1.5 초.
- 명령이 끊기면 500 ms 뒤 정지. 조이스틱 프로그램은 창 포커스 잃음·스페이스·종료 시 0 전송,
  패드 중심 10% 데드존.

### 4.4 IMU (ISM330DHCX)

| 설정 | 값 |
|---|---|
| 가속도 | 208 Hz, ±4 g (0.122 mg/LSB) |
| 자이로 | 208 Hz, ±500 dps (17.5 mdps/LSB) |
| 기타 | BDU=1, 주소 자동증가. 부팅 시·1초마다 WHO_AM_I(0x6B) 확인, 끊기면 재초기화 |
| 좌표계 | **센서 좌표계 원시값**. 차체 좌표계 회전은 Pi 에서 (장착 방향 확인 필요) |

---

## 5. 통신

### 5.1 USART2 아스키 (터미널 · 조이스틱)
한 줄 명령. 구동은 `V vx vy ω` (m/s, m/s, rad/s, x=전진 y=좌측 ω=반시계 +).
전체 명령은 [CLI.md](CLI.md#3-터미널-명령).

### 5.2 SPI2 프레임 (Pi)
정본: `common/spi_link.h`. Pi 쪽: `pi/common/nucleo_link.py` (`SpiLink`).

- 양방향 **144 바이트 고정**, 동기 `A5 5A`, 버전 2, CRC-16/CCITT-FALSE.
- Pi → Nucleo: 속도 명령 (`0x01`) 또는 폴링 (`0x00`, 워치독 갱신 안 함).
- Nucleo → Pi: 한 제어주기의 값 전부 + Nucleo 시각 `t_ms`.
  구동(적용 명령, 엔코더 틱, 바퀴 속도, duty), 조향(목표각, 실측 각·속도·부하·전압·온도·상태,
  유효 비트), IMU(원시 6축), 상태 플래그(워치독, 게이트 대기, 게이트 타임아웃, 서보 읽기 실패,
  조향 클램프, IMU 유효, 명령 출처). 배열은 모듈 순서 FL, FR, RL, RR.
- 받는 텔레메트리는 **직전 CS 해제까지 만든 것** (0~40 ms 묵음). 같은 `tick` 을 두 번
  받을 수 있다. Nucleo 가 없으면 CRC 실패로 `None`.

---

## 6. 세팅 CLI 요약

자세한 순서와 설명은 [CLI.md](CLI.md).

```bash
# 빌드 + 플래시 (보드 USB 연결)
cd ~/ksrc/firmware/src2026-rover-firmware
./build.sh --flash

# 터미널
python3 -m serial.tools.miniterm /dev/ttyACM0 115200
```

```
LOG                       로그 끄기
SCAN                      서보 ID 1~4 확인
OFF                       토크 해제 -> 네 바퀴 정면 정렬 (모터 안쪽)
ZERO ALL                  영점 기록 (EEPROM)
ON
T 0 0 0 0                 네 바퀴 정면 확인
T 30 / T 0 30 / T 0 0 30 / T 0 0 0 30
                          서보별 방향 확인 (위에서 반시계가 정상, 아니면 joint.c .dir = -1)
S 1500                    (선택) 조향 최고속 제한 ~132°/s
MOT 1 300 1000            구동모터 1 을 duty +300 으로 1초 (바퀴 띄우고)
MOT 1 799 2000            무부하 rpm 확인 (2초 이하로, 16비트 엔코더 한계)
```

```bash
# 조이스틱 (miniterm 닫고, 처음엔 바퀴 띄우고 저속)
cd ~/ksrc
python3 pi/teleop/teleop_joystick.py --port /dev/ttyACM0 --max-lin 0.15 --max-ang 0.8
```

```bash
# 라파에서 (ssh pi@raspberrypi.local, SPI 로 Nucleo 와 통신)
cd ~/ksrc
python3 pi/common/spi_monitor.py         # 텔레메트리 확인 (명령 안 보냄)
python3 pi/common/imu_check.py           # IMU 축·부호 확인 (로버 좌표)
python3 pi/common/udp_spi_bridge.py      # PC 조이스틱 -> SPI 중계 (켜 둔다)
```

```bash
# PC 에서 (조이스틱 창은 PC, 라파가 받아 SPI 로 전달)
python3 pi/teleop/teleop_joystick.py --udp raspberrypi.local --rate 50 --max-lin 0.15 --max-ang 0.8 --monitor
```

---

## 7. 조립 후 반드시 할 것 (주행 전)

| # | 항목 | 고칠 곳 | 방법 |
|---|---|---|---|
| Z1 | 조향 영점 | 서보 EEPROM | `OFF` → 정면 정렬 → `ZERO ALL` → `ON` → `T 0 0 0 0` |
| Z2 | 조향 방향 | `joint.c` `.dir` | `T 30` 을 서보별로, 위에서 반시계면 +1 |
| Z3 | 가동범위 | `joint.c` `.min/.max_deg` **+** `sim/rl/config.py` | `T` 로 천천히 늘려 간섭각 확인 |
| Z4 | 구동모터 ↔ 바퀴 | `main.c` `SWERVE_MODULE_MOTOR` | 바퀴 띄우고 `MOT n 300 1000` |
| Z5 | 구동 방향 | `main.c` `MOTOR_DRIVE_SIGN` | `V 0.05 0 0` 에서 뒤로 도는 바퀴의 모터 → -1 |
| Z6 | 엔코더 방향 | `main.c` `ENCODER_SIGN` | Z5 후 `LOG` m/s 가 음수인 모터 → -1 |

Z1 → Z2 → Z3 순서대로. 같은 목록이 옵시디언 `실기 확인 목록` 0절에 있다.

---

## 8. 테스트

```bash
# 공용 C (호스트)
cd ~/ksrc/firmware/common/tests
for t in swerve_kinematics drive_align_gate spi_link mdd3a_driver wheel_pi; do
  gcc -std=c99 -Wall -Wextra -I.. -o /tmp/t_$t test_$t.c ../$t.c -lm && /tmp/t_$t | tail -1
done

# Python <-> C 프레임 교차검증
cd ~/ksrc && bash sim/build.sh && python3 pi/common/test_spi_link.py
```
