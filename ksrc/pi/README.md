# pi/ — 라즈베리파이 쪽 코드 (2026-10-16)

라파(`ssh pi@raspberrypi.local`, Ubuntu 24.04, 코드 `~/ksrc` = GitHub `YunBeeeen/ksrc` main)에서 돈다.
Nucleo(STM32)와는 **SPI** 로 이야기한다. USB(UART)는 노트북에서 펌웨어 굽기·설정 명령·sysid 측정할 때만 쓴다.

```
노트북 ──USB(ST-Link)── Nucleo ──SPI 50 Hz── 라파 ── (Nav2 / 조이스틱 UDP / RL 정책)
         굽기, WPI ON,          펌웨어: IK·게이트·바퀴 PI·서보   pi/common, pi/policy
         sysid_uart.py
```

## 처음 받거나 갱신할 때 (라파)

```bash
cd ~/ksrc && git pull
bash sim/build.sh          # firmware/common C 를 라파용 .so 로 (bringup·정책 실행기가 펌웨어와 같은 C IK 를 씀)
```
필요한 것: python3, numpy, spidev (있음), pyserial (있음). RL 정책 실행에는 torch 가 더 필요 → venv `~/venv` 에 설치해 `~/venv/bin/python` 으로 실행 (`MEASURE_1016.md` 준비 단계).
실측 절차: **[MEASURE_1016.md](MEASURE_1016.md)** (바닥·IMU·루프·Pi 부하), **[MEASURE_1017.md](MEASURE_1017.md)** (조향 떨림 S0, 모래 평지·경사 A/B, 유효 견인 계수, VO).
펌웨어는 **노트북**에서 USB 로 연결해 `firmware/src2026-rover-firmware/build.sh --flash`.

## 규칙

- **SPI 는 한 번에 한 프로그램만.** 조이스틱 브리지와 bringup·정책 실행기를 같이 켜면 명령이 섞인다.
- 라파 SPI 시험 중에 USB 터미널에서 `V` 같은 주행 명령을 치지 않는다 (`WPI`, `WPI ON` 같은 설정은 괜찮음).
- `sysid_uart.py` (USB) 의 `WCAP`·`CAP` 은 텔레옵 중엔 거부된다. 라파가 SPI 명령을 보낸 뒤라면 보드 리셋 후.
- 명령이 끊기면 펌웨어가 멈춘다 (속도 500 ms, RL 잔차 100 ms). 라파 프로그램을 그냥 꺼도 안전.

## 파일

### `pi/common/` — 링크·점검·측정

| 파일 | 어디서 | 하는 일 |
|---|---|---|
| `nucleo_link.py` | (라이브러리) | SPI 프레임 인코딩·텔레메트리 해석 (`SpiLink`: `exchange` 속도 명령, `exchange_res` RL 잔차, `config` 바퀴 PI 설정). 펌웨어 `firmware/common/spi_link.c` 와 바이트 단위로 같음. 모든 SPI 도구가 이걸 씀 |
| `bringup.py` | 라파 | **실물 점검 묶음**: `link` `encdir` `latency` `res` `gate` `duty` `log` `wpi`(PI 켜기/끄기·corr, SPI) `sand`(모래 A/B), `all-lifted`. PASS/확인 필요 + `bringup_logs/*.npz`. 순서는 `firmware/CLI.md` "실물 점검·측정 순서" |
| `fake_nucleo.py` | 노트북 | 가짜 Nucleo (펌웨어 C 그대로). `bringup.py ... --fake` 로 실물 없이 시험 |
| `spi_monitor.py` | 라파 | 50 Hz 로 주고받으며 텔레메트리를 한 줄씩 (`--vx 0.05` 등으로 명령도) |
| `spi_timing.py` | 라파 | SPI 교환 주기·왕복 (폴링만) |
| `imu_check.py` | 라파 | IMU 값 사람이 보기 좋게 (기울기, 누적각) |
| `udp_spi_bridge.py` | 라파 | PC 조이스틱(UDP 5005) → SPI 중계 |
| `imu_noise.py` | 노트북 | `run_policy --log` npz 들에서 자이로 바이어스·잡음, 가속도 잡음, 정지 기울기, 진동 주파수 → 시뮬 `STUDENT_NOISE`·`GRAV_ERR` 제안값 (`--out imu_noise.json`) |
| `sensor_fusion.py` | (라이브러리) | student 관측 계산: 중력 보완 필터, 자이로 바이어스, 바퀴 rad/s. 시뮬 `utils/sensors_torch.py` 와 같은 식 |
| `sysid_uart.py` | 노트북 (USB) | 모터·서보 측정: `drive` `steer` `steer-small` `pi`(바퀴 PI 벤치) `wcap` `imu`(ICAP 208 Hz) `cap` `raw` → `sysid_logs/` |
| `sysid_analyze.py` | 노트북 | sysid 결과 분석·판정 (`firmware/MOTOR_MODELING_PLAN.md` §0) |
| `test_spi_link.py` `test_sysid.py` | 노트북 | SPI 프레임 C 교차검증 / sysid 측정·분석을 가짜 펌웨어로 끝까지 |

### `pi/policy/` — RL 정책 실행 (구조 B: 펌웨어 IK + 바퀴별 잔차)

| 파일 | 하는 일 |
|---|---|
| `run_policy.py` | 내보낸 student(`<p>.pt` + `<p>.json`, `sim/isaac/ksrc_rover/scripts/export_student.py`)를 50 Hz 로. 시작 2 s 정지해 자이로 바이어스 → 텔레메트리 → 관측 → 정책 → SPI 0x02. 시작 때 Nucleo 가 들고 있는 조향각으로 IK 상태 동기화 (`FirmwareIK.sync`). **정지 명령(\|v\|<0.01 m/s, \|ω\|<0.02 rad/s)이면 잔차 0** (서 있을 때 조향 떨림 방지). json 에 `res8_rate_limit` 가 있으면 잔차 변화율 제한도 학습과 같게. `--residual-off` (경로만), `--fake`, `--log` (관측·행동 + 원시 텔레메트리: `host_ns` `cmd` `gyro_dps` `acc_g` `imu_n` `steer_cmd`/`steer_meas`/`steer_load`/`steer_volt` `duty` `wheel_mps` …), `--rt 50 --cpu 3 --gc-off` (루프 지터) |
| `student_runtime.py` | 관측 조립 (Isaac student 와 같은 항목·순서·이력, student v3 의 `vo` 항 포함 — VO 스트림이 없으면 `[0, 0, 무효, 1.0 s]`), 펌웨어 C IK (`FirmwareIK`), 정책 출력 → 잔차 |
| `res8_mlp_v1_rc4000.{pt,json}` | **지금 실물에 쓰는 정책** = student v1 (`sim/rl/runs/isaac/2026-10-15_13-06-32_student_mlp_v1_rc4000/model_1000.pt` 를 내보낸 것, teacher rc4000). 시뮬 IK 대비 큰헛돎 −39% |
| `test_student_runtime.py` | 노트북: C IK == 시뮬 IK, 관측 배치, `--fake` 루프 |

### `pi/teleop/` — 사람이 모는 텔레옵

| 파일 | 통로 | 하는 일 |
|---|---|---|
| `teleop_joystick.py` | PC 창 → UDP → 라파 브리지 (또는 직접 시리얼) | 화면 마우스 조이스틱. `--udp raspberrypi.local --monitor` |
| `teleop_keyboard.py` | 시리얼 | WASD/QE |
| `teleop_terminal.py` | 시리얼 | "vx vy w" 숫자 입력 |
| `teleop_ascii.py` | USB UART | 펌웨어 `V` 명령 직접 |
| `joystick_math.py` (+ 테스트) | — | 조이스틱 데드존·스케일 계산 |

## 자주 쓰는 것

```bash
# 링크·부호 점검 (바퀴 띄움)
python3 pi/common/bringup.py all-lifted
# 조이스틱 주행: 라파 / PC
python3 pi/common/udp_spi_bridge.py
python3 pi/teleop/teleop_joystick.py --udp raspberrypi.local --rate 50 --max-lin 0.15 --max-ang 0.8 --monitor
# 바퀴 PI 끄기/켜기 (USB 없이)
python3 pi/common/bringup.py wpi --wpi off
python3 pi/common/bringup.py wpi --wpi on --corr 250
# 모래 A/B (조건 자동 전환, 매 회 줄자 입력)
python3 pi/common/bringup.py sand --vx 0.1 --sec 8 --conds off,on,c100 --sand-reps 2
# 경사 정지 등 기록
python3 pi/common/bringup.py log --vx 0 --sec 10 --tag hold12_down
# RL 정책 (torch venv). 잔차 끄고 경로만 먼저, 그다음 잔차 켬
RUN="$HOME/venv/bin/python pi/policy/run_policy.py --policy pi/policy/res8_mlp_v1_rc4000 --rt 50 --cpu 3 --gc-off"
$RUN --residual-off --cmd 0.10 0 0 --seconds 12 --log logs/1016/ik_v010.npz
$RUN --cmd 0.10 0 0 --seconds 12 --log logs/1016/rl_v010.npz
# 로그 분석 (노트북)
python3 pi/common/imu_noise.py sysid_logs/1016/*.npz --out sysid_logs/1016/imu_noise.json
```

## 10-14 실측 (bringup)

- link: 제어주기 20 ms (텔레옵 중. 텔레옵 전엔 10 Hz UART 로그로 40 ms 가 섞임), CRC 0, imu_n 4–5, 서보 100 %
- encdir: 네 바퀴 부호 OK (개루프 0.05 m/s 명령에 0.038–0.043 m/s)
- latency: 명령 → 적용 17.6–46.5 ms (평균 30.2 = 1.5 스텝), 관측 나이 평균 14 ms (최대 29), Nucleo 시계 +0.28 %
- res: 잔차 경로 6항목 PASS / gate: 90° 전환 0.74 s 대기, 40°×3 실측 트리거 PASS

관련 문서: `firmware/CLI.md` (펌웨어 명령, 점검 순서), `firmware/MOTOR_MODELING_PLAN.md` (측정 계획),
`firmware/common/spi_link.h` (SPI 프레임), `sim/isaac/ksrc_rover/CLI.md` (시뮬).
