# 펌웨어 CLI

빌드·플래시, 터미널 명령, 세팅 절차를 모은 문서. 설정값의 의미와 위치는
[README.md](README.md) 가 정본이다.

---

## 1. 빌드 / 플래시 (PC)

```bash
cd ~/ksrc/firmware/src2026-rover-firmware

./build.sh              # 빌드만
./build.sh --flash      # 빌드 후 ST-Link 로 플래시 (보드 USB 연결)
./build.sh --clean      # build/ 지우고 전체 재빌드
```

- `arm-none-eabi-gcc` 가 PATH 에 없으면 CubeIDE 에 들어 있는 툴체인을 자동으로 찾는다.
- **코드를 고쳤으면 `--flash` 까지** 해야 보드에 반영된다. `./build.sh` 만으로는 안 바뀐다.
- `common/*.c` 를 고쳤으면 시뮬 라이브러리도 다시 빌드: `bash ~/ksrc/sim/build.sh`
- 플래시와 터미널은 같은 USB 포트(`/dev/ttyACM0`)를 쓴다. 터미널을 열어 둔 채로는
  플래시가 실패하므로 먼저 닫는다.

---

## 2. 시리얼 터미널 (PC)

```bash
python3 -m serial.tools.miniterm /dev/ttyACM0 115200
```

나가기: `Ctrl+]`. 대소문자 구분 없음, 한 줄 입력 후 엔터.

**시작하면 먼저 `LOG`** 를 친다. 기본 상태에서는 엔코더 로그가 0.1초마다 쏟아져서
명령 응답이 묻힌다. 텔레옵(첫 `V`)이 시작되면 로그와 에코는 자동으로 꺼진다.

USB 만 꽂으면 STM32 만 켜진다. 모터·서보는 12V 별도 전원이 있어야 움직인다.

---

## 3. 터미널 명령

### 화면

| 명령 | 설명 |
|---|---|
| `LOG` | 주기 로그 on/off 토글. 부팅 시 ON, 텔레옵 시작 시 자동 OFF |
| `ECHO` | 입력 에코 on/off 토글 (백스페이스 수정 가능) |
| `HELP` | 명령 목록 |

### 조향 서보

| 명령 | 설명 |
|---|---|
| `PING` | 서보 ID 1~4 응답 확인 |
| `SCAN` | ID 1~253 전체 스캔 (약 1초) |
| `ID <현재> <새>` | 서보 ID 변경. **서보 1개만 연결한 상태에서**. `254` = 브로드캐스트 |
| `ZERO <id\|ALL>` | 현재 자세를 0° 로 서보 EEPROM 에 기록 |
| `ON` / `OFF` | 토크 켜기 / 끄기 (끄면 손으로 돌릴 수 있음) |
| `T <a1> <a2> <a3> <a4>` | 조향각 직접 지정 (도). **서보 번호 순서 = LF, LB, RF, RB**. 준 개수만큼만 보냄. 가동범위(±90°)로 클램프 |
| `P` | 현재 조향각 읽기 |
| `ACC <0-254>` | 조향 서보 4개 가속도 레지스터. 부팅 기본 50 (`joint_init`), `0` = 가감속 제한 없음. 모델링 S1 에서 `ACC 50` / `ACC 0` 으로 `STEP` 비교. 서보 전원 껐다 켜면 50 |
| `S <step/s>` | 서보 최고속 제한. `0` = 최대(기본). 4096 step = 360° → `S 1500` ≈ 132°/s. 재부팅 시 0 |

### 구동 / 주행

| 명령 | 설명 |
|---|---|
| `V <vx> <vy> <ω>` | 차체 속도 (m/s, m/s, rad/s). x=전진, y=좌측, ω 반시계 +. **첫 `V` 에서 텔레옵 시작** (되돌리려면 리셋) |
| `MOT <1-4\|ALL> <duty> <ms>` | 구동모터 하나를 날 부호로 돌리고 엔코더 변화·rpm 출력. duty -799~+799, ms 50~3000 (**2000 이하 권장**, 16비트 엔코더 한계). 텔레옵 전에만. 도는 동안 로그 멈춤 |
| `CAP <1-4\|ALL\|234..> <d0> <d1> <t_sw> <t_end> [RAMP] [DEC n]` | **모터 모델링용 고속 캡처.** duty d0 → d1 계단(RAMP 면 0..t_sw 직선)을 넣으며 (모터는 숫자를 붙여 여러 개: `CAP 234` = 2·3·4) 엔코더를 1 kHz(DEC n 이면 n ms)로 RAM 에 최대 2000 샘플 기록한 뒤 `t_us,dp1..dp4` 로 덤프. **부호 보정 적용 (+ = 바퀴 전진)**, 열 = MOTOR1..4. d0≠0 이면 0.8 s 먼저 돌려 안정. 텔레옵 전에만, 바퀴 띄우고. 끝 줄 `[CAP] END late_max_us=.. vs_dv=..` (샘플 최대 지연, 끝 시점 서보1 전압 0.1 V). 노트북 `pi/common/sysid_uart.py` 가 파싱·저장, `sysid_analyze.py` 가 판정 |
| `MTEST` | 모터 자가시험 시퀀스 on/off (기본 OFF). 1→4번 정/역 2초씩. 텔레옵 전에만 |
| `WPI [ON\|OFF\|SET <이름> <값>]` | **바퀴 속도 PI** (`common/wheel_pi.c`, MOTOR_MODELING_PLAN §8). ON 이면 텔레옵 구동 duty = FF + 기준 모델 PI (엔코더 속도를 목표 바퀴 속도에 맞춤), OFF 면 예전 개루프 `duty = v/0.378×799`. **기본 ON** (10-14 벤치·지면 확인 뒤 `main.c WHEEL_PI_DEFAULT_ON 1`. 개루프로 돌리려면 `WPI OFF`). 인자 없으면 상태·이득 출력. `SET` 은 RAM 값 (재부팅하면 설계값): `kp` 82.5 `ki` 2062 `imax` 200 `kick` 40 `fkin` 5 `kduty` 0.0097 `taum` 0.04 `corr` 250 `stallt` 0.5. 텔레메트리 flags 비트 8 (`WHEEL_PI`). **보호 (10-13)**: ① 보정량 \|kp·e+i\| ≤ `corr` (duty ≤ FF+250) ② 스톨: 보정이 상한에 붙고 \|w\|<0.05 rad/s 가 0.5 s → 그 바퀴 FF 만 (바퀴가 목표 절반 이상으로 돌거나 정지 명령이면 풀림), flags 비트 9 `WHEEL_STALL` ③ 엔코더 방향 불일치 (A/B·모터 +/−·`MOTOR_DRIVE_SIGN`·커넥터 실수 → PI 양의 되먹임 폭주 방지): \|duty\|≥150 인데 바퀴가 반대로 ≥0.3 rad/s 가 0.3 s → 1 s 동안 FF 만, 그 뒤 PI 재시도. 10 s 안에 3 번이면 잠금 (FF 만, flags 비트 10 `ENC_DIR`, `WPI ON` 으로 해제). 조립 후 바퀴 띄우고 모터 방향·엔코더 부호 확인이 먼저. 인자 없는 `WPI` 가 모듈별 상태 출력 |
| `ICAP [ms]` / `ICAP DUMP` | **IMU 208 Hz 원시 기록** (MOTOR_MODELING_PLAN §11). 제어 루프를 막지 않아 주행 중에도 기록 (최대 2000 샘플 ≈ 9.6 s). 덤프는 ~7 s 루프를 막으므로 주행 명령이 0 일 때만. 줄 `t_us,gx,gy,gz,ax,ay,az` (칩 좌표 원시값). 노트북 `sysid_uart.py imu [--ref-deg 경사계값] --tag ..` |
| `WCAP <1-4\|ALL\|234..> <w0> <w1> <t_sw> <t_end> [RAMP] [DEC n]` | **PI 벤치 검증 캡처.** `CAP` 과 같지만 duty 대신 바퀴 속도 rad/s (\|w\|≤8, + = 전진) 를 주고 20 ms 마다 PI 가 duty 를 정한다 (`WPI ON/OFF` 와 무관하게 PI). 엔코더 줄은 `CAP` 과 같고, 이어서 PI 주기마다 `D,t_ms,d1..d4` (건 duty). 텔레옵 전에만, 바퀴 띄우고. 노트북 `sysid_uart.py pi` (계단·저속·역전 묶음) / `wcap` (한 번) |
| `GATE` | 구동 정렬 게이트 on/off (기본 ON). 서보 없이 구동만 시험할 때 끈다 |
| `CTRL [ON\|OFF]` | 텔레옵 중 5Hz 상태줄: `[CTRL] v vx vy ω \| deg 조향각×4 \| duty ×4` (+ `GATE`, `WDOG`). 받은 명령 확인용. 인자 없으면 토글. `LOG` 와 별개라 텔레옵 시작 후에도 유지 |
| `SPI` | Pi SPI 링크 진단: `CS에지`(프레임 끝으로 인정한 CS 상승) `잡음`(무시한 CS 스파이크) `완전/짧음` 프레임 수, 마지막 수신 바이트 수·첫 두 바이트(정상 `A5 5A`), 파싱오류, 현재 PB12/PB13 레벨(Pi 유휴면 1/0). Pi 에서 보내기 전·후로 한 번씩 쳐서 비교 |
| `STEP <n> <도>` | 조향 서보 n(1~4)을 목표각으로 보내며 피드백을 ~0.7ms 간격으로 읽어 **10→90% 시간·평균/최고 속도(도/s)·도달 시간(±1° 100ms 유지)·오버슈트**를 출력. 나머지 서보는 제자리. `S` 로 속도를 바꿔 비교 (예: `S 0` → `STEP 1 90` → `STEP 1 0`). 텔레옵 중엔 거부. 첫 줄 다음에 `시작 지연(0.5도 움직일 때까지)` 도 찍는다 |
| `STEP <n> <도> RAW [MS <ms>]` | 위 + **1 ms 궤적 덤프** (`[STEPRAW] BEGIN ... acc= dz_cw= dz_ccw=` / `t_us,각도0.01도,속도도/s,부하` / `[STEPRAW] END`, 최대 2000 샘플). **2도 미만 작은 계단은 RAW 로만** 허용 (데드밴드·히스테리시스, 기본 800 ms 고정 기록). MS 를 주면 큰 계단도 ±1° 도달 판정 없이 고정 시간 기록. `pi/common/sysid_uart.py steer / steer-small` 이 쓴다 |

`V` 동작:
- 50Hz 제어주기가 출력한다 (명령은 목표값만 바꾼다).
- 500 ms 동안 명령이 없으면 정지 (조향각은 유지).
- 조향이 60° 넘게 바뀌면 바퀴가 10° 안에 정렬될 때까지 구동 0 (최대 1.5초).

---

## 4. 세팅 절차

### 4.1 처음 배선 후 (서보 ID)

```
LOG
SCAN                      버스에 몇 번으로 잡히는지
```

ID 를 새로 줄 때는 서보를 **하나씩만** 연결한다.
```
ID 254 4                  연결된 1개를 4번으로 -> 분리하고 다음 서보
ID 254 3
ID 254 2
ID 254 1
SCAN                      4개 전부 연결 후 1~4 확인
```

### 4.2 조향 영점 (조립 완료 후, Z1)

```
OFF                       토크 해제
                          -> 네 바퀴를 손으로 정면 정렬 (자/직선 기준, 구동모터는 몸체 안쪽)
ZERO ALL                  지금 자세를 0° 로 EEPROM 기록
ON
T 0 0 0 0                 네 바퀴 모두 정면이면 성공
```

### 4.3 조향 방향 (Z2)

```
T 30                      서보1 (LF)
T 0 30                    서보2 (LB)
T 0 0 30                  서보3 (RF)
T 0 0 0 30                서보4 (RB)
T 0 0 0 0
```
각각 **위에서 볼 때 반시계**면 정상. 시계로 돈 서보는
`src2026-rover-firmware/Core/Src/joint.c` 의 그 서보 `.dir = -1` → `./build.sh --flash` → 다시 확인.

### 4.4 가동범위 (Z3)

```
T 60   / T -60            서보1 을 천천히 키워 가며
T 90   / T -90
T 100  / T -100           구동모터·배선이 몸체에 닿는지
T 0
```
다른 서보는 `T 0 90`, `T 0 0 90`, `T 0 0 0 90` 처럼 자리만 바꾼다.
간섭이 있으면 `joint.c` `.min_deg/.max_deg` **와** `sim/rl/config.py`
`steer_lo_deg/steer_hi_deg` (순서 FL, FR, RL, RR) 를 같이 고친다. 폭은 180° 이상 유지.

### 4.5 구동모터 (바퀴 띄우고, Z4~Z6)

```
LOG                       (로그가 켜져 있으면 끄기 -- MOT 결과 줄이 묻힌다)
MOT 1 300 1000            모터1: 어느 바퀴가, 어느 방향으로 도는지 + 엔코더 부호
MOT 2 300 1000
MOT 3 300 1000
MOT 4 300 1000
MOT 1 799 2000            무부하 rpm (정격 약 76 rpm)
```
- 엉뚱한 바퀴가 돌면 `main.c` `SWERVE_MODULE_MOTOR` 수정
- 그다음 `V 0.05 0 0` 으로 확인: 뒤로 도는 바퀴의 모터 → `MOTOR_DRIVE_SIGN = -1`
- 전진하는데 `LOG` m/s 가 음수인 모터 → `ENCODER_SIGN = -1`
- 고친 뒤 `./build.sh --flash`

### 4.6 조향 속도 (선택)

```
S 1500                    ~132°/s 로 제한 (S 0 = 최대)
```
마음에 드는 값을 찾으면 `joint.c` 의 `joint_goal_speed` 기본값으로 넣는다.

---

## 5. 조이스틱 주행 (PC, USB 직결)

```bash
# miniterm 은 먼저 닫는다 (포트는 한 프로그램만 연다)
cd ~/ksrc
python3 pi/teleop/teleop_joystick.py --port /dev/ttyACM0 --max-lin 0.15 --max-ang 0.8
```

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--port` | `/dev/ttyACM0` | STM32 USB 포트 |
| `--rate` | 20 | 전송 Hz (30 이상이면 파서가 바이트를 흘림) |
| `--max-lin` | 0.5 | 패드 끝 = 이 속도 (m/s). 실제 최고 ~0.38 이라 처음엔 0.15 권장 |
| `--max-ang` | 2.0 | 패드 끝 = 이 각속도 (rad/s) |
| `--monitor` | 끔 | STM 에 `CTRL ON` 을 보내고, 보낸 값(`PC >`)과 STM 이 받은 값(`STM>`)을 터미널에 나란히 출력 |
| `--deadzone` | 0.1 | 패드 중심 데드존 (반경 비율). 중심 손떨림이 바퀴를 홱홱 돌리지 않게 |
| `--snap-deg` | 10 | 앞·뒤·좌·우에서 이 각도 이내면 축으로 맞춤. 횡걸음 중 손떨림으로 조향이 87°↔93° 떠는 것 제거 |
| `--hold-mag` | 0.35 | 이 크기 미만에서는 바퀴 축을 `--hold-deg` 넘게 바꾸는 방향을 0 으로 보냄 (조향 유지) |
| `--hold-deg` | 20 | 위 저속 구간에서 허용하는 바퀴 축 변화. **정반대 방향(우↔좌, 전↔후)은 항상 허용** |

조작: 왼쪽 패드 = 병진, 오른쪽 패드 또는 `Q`/`E` = 회전, `Space` = 정지, `Esc` = 종료.

STM 이 vx, vy, ω 를 제대로 받는지 확인 (케이블 하나로, miniterm 없이):
```bash
python3 pi/teleop/teleop_joystick.py --port /dev/ttyACM0 --max-lin 0.15 --max-ang 0.8 --monitor
```
```
PC > v +0.150 +0.000 +0.000
STM> [CTRL] v +0.150 +0.000 +0.000 | deg 0 0 0 0 | duty 376 376 376 376
```
`PC >` 와 `STM>` 의 v 가 같으면 정상. 조이스틱은 패드를 안 움직여도 0 을 20Hz 로 보내므로
`STM>` 줄은 켜자마자 나온다. 안 나오면 펌웨어가 `CTRL` 명령이 없는 이전 버전이다
(`./build.sh --flash`). 정렬 대기 중이면 줄 끝에 `GATE`, 명령이 끊기면 `WDOG` 가 붙는다.

**저속 방향 고정이 하는 일:** 우 횡걸음 → 좌 횡걸음처럼 패드를 반대편으로 옮길 때, 마우스가
중앙을 위·아래로 스쳐도 그 사이의 "전진/후진" 성분이 나가지 않는다. 그래서 바퀴는 90° 를 유지하고
구동만 반전된다 (이전에는 경로에 따라 네 바퀴가 0° 쪽으로 돌았다 오며 최대 약 300° 를 돌았다).
대신 횡걸음 → 전진처럼 **바퀴 축을 바꾸는 방향 전환은 패드를 35% 넘게 밀어야** 시작된다.
작게만 밀면 바퀴는 제자리에 서 있는다.
처음에는 **바퀴를 띄운 상태**로 네 바퀴 방향부터 확인한다 (4.5).

---

## 6. Pi 에서 SPI 링크 확인

Pi 접속: `ssh pi@raspberrypi.local` (같은 Wi-Fi). Pi 는 Ubuntu 24.04 호스트이고,
ROS 2 는 `docker exec -it humble_rover bash` 안에 있다.

처음 한 번 (sudo 비밀번호 필요):
```bash
sudo apt install -y python3-spidev
sudo usermod -aG dialout pi      # /dev/spidev0.0 이 dialout 그룹. 재접속 후 적용
```

배선: Nucleo PB12 (CN10-16) → Pi 핀24 (CE0), PB13 (CN10-30) → 핀23 (SCLK),
PC2 (CN7-35) → 핀21 (MISO), PC3 (CN7-37) → 핀19 (MOSI),
그리고 **Pi 핀25 (GND) → Nucleo GND 직결선** (모터 GND 묶음과 별도).
**SCLK 선은 MOSI·MISO 와 떼어서** (가능하면 GND 선과 나란히) 보낸다. 묶어 두면 크로스토크로
가짜 클럭이 생겨 프레임이 전부 깨진다 (2026-10-05 실측, 속도를 낮춰도 안 없어짐). MISO·MOSI 는 묶어도 됨.

```bash
cd ~/ksrc
python3 pi/common/spi_monitor.py               # 수신만 (명령 안 보냄)
python3 pi/common/spi_monitor.py --vx 0.05     # 전진 명령
python3 pi/common/spi_monitor.py --vy 0.05     # 게걸음
python3 pi/common/spi_monitor.py --w 0.5       # 제자리 회전
```
`[BAD]` 만 나오면 배선·CE0·GND 확인. 출력 태그 `W` = 워치독, `G` = 게이트 대기,
`T` = 게이트 타임아웃. `enc[...]` = 엔코더 누적 틱 (FL FR RL RR, 바퀴 1회전 8384).

**`[BAD]` 원인 좁히기** (시리얼 `SPI` 명령과 같이):
```bash
# Pi: 정확히 10바이트(클럭 80번)만 보낸다
python3 -c "import spidev;s=spidev.SpiDev();s.open(0,0);s.mode=0;s.max_speed_hz=10000;s.xfer2([0xA5,0x5A,1,2,3,4,5,6,7,8])"
```
그다음 miniterm 에서 `SPI` → `마지막 N바이트 첫 XX XX`:
`10바이트 A5 5A` 정상 / `11바이트 이상` = SCLK 링잉(선 분리·GND·직렬저항) /
`CS에지 0` = CE0 선 / `0바이트` = SCLK 선 / `10바이트인데 다른 값` = MOSI 선.

### IMU 확인
```bash
python3 pi/common/imu_check.py      # 2초 정지 보정 후 10Hz: acc, |acc|, roll/pitch, gyro, 자이로 적분각
```
값은 **로버 좌표** (x 전진, y 좌측, z 위). 칩→로버 변환은 `pi/common/nucleo_link.py`
`IMU_CHIP_TO_BODY` (현재 칩이 z 기준 180° 장착 → x, y 반대). 정상: 정지 시 acc ≈ 0 0 +1,
반시계 90° → 셋째 누적 ≈ +90, 앞을 숙이면 pitch +. Enter = 누적각 리셋.

### 조이스틱 주행 (PC 창 → 라파 → SPI)
라파는 화면이 없고, ssh X 포워딩(`-X`/`-Y`)으로는 pygame 창이 MIT-SHM 오류로 안 뜬다.
그래서 **창은 PC**, **Nucleo 통신은 라파**가 맡는다. PC 와 라파는 같은 와이파이.
```bash
# 1) 라파 (ssh pi@raspberrypi.local)
cd ~/ksrc && git pull
python3 pi/common/udp_spi_bridge.py          # UDP 5005 -> SPI 50Hz 중계

# 2) PC (새 터미널)
cd ~/ksrc
python3 pi/teleop/teleop_joystick.py --udp raspberrypi.local --rate 50 --max-lin 0.15 --max-ang 0.8 --monitor
```
조작·인자는 5 와 같다. `--monitor` 의 `STM>` 줄과 창 아래 초록 줄은 라파가 돌려주는 텔레메트리
(STM 이 실제 적용한 v, 조향각, duty). 라파 화면에는 `[PC주소] 보냄 v ... rx= drop= crc_bad=` 가 0.5초마다.
안전: PC 명령이 0.3초 끊기면 라파가 0 을 보내고(`끊김->0`), 라파까지 끊기면 STM 워치독(500ms)이 정지.
라파는 받은 값을 `--max-lin 0.32 --max-ang 2.0` 으로 한 번 더 자른다.

### 실물 점검·측정 순서 (2026-10-14, `pi/common/bringup.py`)
펌웨어는 10-14 빌드 (PI 보호 3개·IMU 평균·CAD 기하·잔차 0x02·게이트 매 주기·옆방향 접기 수정) 를 먼저 굽는다.
라파: `cd ~/ksrc && git pull && bash sim/build.sh` (bringup·정책 실행기가 펌웨어와 같은 C 를 씀, numpy 필요).
각 항목은 PASS / 확인 필요 를 찍고 `bringup_logs/<시각>_<항목>.npz` 에 교환마다 텔레메트리를 남긴다.
노트북에서 미리 보기: 같은 명령에 `--fake` (가짜 Nucleo, `pi/common/fake_nucleo.py`).

| 순서 | 상태 | 명령 | 보는 것 |
|---|---|---|---|
| 1 | 정지 | `bringup.py link` | CRC 실패 < 0.1%, 제어주기 20 ms, `imu_n` 3–5 (IMU 평균 펌웨어), \|acc\| 1 g, 서보 4개 |
| 2 | 띄움, `WPI OFF` | `bringup.py encdir` | 네 바퀴 duty·엔코더가 명령 방향. **실패면 WPI 켜지 말 것** |
| 3 | 띄움 | `bringup.py latency` | 명령 → 적용 틱 수 (시뮬 action_delay 0–1), 관측 나이 흔들림 (obs_delay) |
| 4 | 띄움 | `bringup.py res` | 잔차 0x02: 0 이면 텔레옵과 같음, +10°·×1.5 반영, 100 ms 타임아웃, NaN, 텔레옵 우선 |
| 5 | 띄움 | `bringup.py gate` | 90° 전환 대기·해제, 40°×3 실측 트리거 |
| 6 | 띄움, USB | `sysid_uart.py pi` + `WPI ON` 후 손으로 바퀴 잡기 | PI 계단 응답 (설계 90% 37 ms), 0.6 s 쯤 스톨 (비트 9) |
| 7 | 정지·기울임 | `sysid_uart.py imu --ref-deg 0/10/20` | 기울기 정확도, 자이로 바이어스·표류 |
| 8 | 지면, WPI OFF→ON | `bringup.py duty --vx 0.15` | 바퀴끼리 미는지 (duty 차이·부호), 보호 플래그 |
| 9 | 지면 주행 | `sysid_uart.py imu --tag bumps` | 실제 진동 → 시뮬 GRAV_ERR |
| 10 | 경사, duty 0 | `bringup.py log --vx 0 --sec 10 --tag slope_hold` + 바퀴를 손으로 돌려 역구동 | 밀려 내려가는지, 역구동 뻑뻑함 (시뮬 기어 마찰) |
| 11 | 모래 A/B | `WPI OFF` / `WPI ON` / `WPI SET corr 100` 마다 `bringup.py log --vx 0.1 --sec 8 --tag sand_<조건>` | 줄자 거리 ÷ 엔코더 거리 = 진행률, duty 포화, 스톨 |

### RL 정책 주행 (구조 B, 2026-10-13): 펌웨어 IK + 바퀴별 잔차
SPI 타입 `0x02` = (vx, vy, w) + 조향 잔차 4 [rad] + 바퀴 속도 배율 4. 펌웨어는 **지금과 같은 IK·접기** 뒤에
조향각 += 잔차 (±30°, 가동범위로 자름), 바퀴 속도 ×= 배율 (0.4–1.6, 최고속으로 자름) 을 하고 게이트·PI 로 간다.
잔차가 100 ms 넘게 안 오면 순수 IK 로 계속 (속도 워치독 500 ms 는 그대로). 텔레메트리 flags 비트 11 `RES_ACTIVE`.
**텔레옵은 그대로**: UART `V`, SPI `0x01` (조이스틱 브리지) 가 오면 잔차는 즉시 꺼지고 예전과 같은 순수 IK.
```bash
# 라파: 내보낸 student (sim/isaac/ksrc_rover/scripts/export_student.py 의 <p>.pt + <p>.json) 로
cd ~/ksrc && bash sim/build.sh                       # 펌웨어와 같은 C IK 를 라파에서도 (관측용)
python3 pi/policy/run_policy.py --policy ~/policies/res8_mlp_h5 --cmd 0.10 0 0 --seconds 5 --residual-off   # 경로 확인
python3 pi/policy/run_policy.py --policy ~/policies/res8_mlp_h5 --cmd 0.10 0 0 --seconds 5 --log run1.npz
```
처음엔 바퀴 띄우고. 시작 전 2 초 정지해서 자이로 바이어스를 잡는다. 끝나면 속도 0 (`0x01`) 을 보내 잔차도 끈다.
노트북 확인: `python3 pi/policy/test_student_runtime.py` (C IK == 시뮬 IK, 관측 순서, `--fake` 루프).

---

## 7. 자주 겪는 것

| 증상 | 원인 / 해결 |
|---|---|
| 플래시 실패, 포트 열기 실패 | 다른 프로그램(miniterm, 조이스틱)이 `/dev/ttyACM0` 을 잡고 있음 |
| `No debug probe detected` | ST-Link 인식 안 됨. `lsusb \| grep 0483`. 충전 전용 USB 케이블 의심 |
| 명령을 쳐도 안 움직임 | 12V 전원 확인. USB 는 STM32 만 켠다 |
| `MOT` 가 거부됨 | 이미 `V` 로 텔레옵이 시작됨 → 보드 리셋 |
| `MOT` 결과가 안 보임 | `LOG` 로 로그 끄기. 도는 동안은 로그가 멈추는 게 정상 |
| `MOT` 엔코더 0 | 12V·엔코더 배선 확인, 또는 duty 가 너무 작아 기동 못 함 |
| 바퀴가 안 돌고 1.5초씩 멈춤 | 정렬 게이트가 서보 각을 못 읽음 (서보 전원/배선). 서보 없이 시험할 땐 `GATE` |
| `PING` 전부 FAIL | 서보 12V·PB10 배선·ID 확인 |
| `IMU ... 응답 없음` (부팅 메시지) | SPI3 배선, CS = PC9 확인. 1초마다 재시도함 |
| Pi `[BAD]` 계속 / `SPI` 파싱오류 증가 | SCLK 선을 MOSI·MISO 와 분리, Pi 핀25 GND 직결 (6 참고) |
| IMU 세 축이 같은 값 / 가끔 튐 | SPI3 잡음. 펌웨어가 즉시 무효 처리·재설정. 잦으면 PC10(SCK) 선 분리 |
| 조이스틱 `STM>` 가 안 나옴 (--udp) | 라파에서 `udp_spi_bridge.py` 실행 중인지, 같은 와이파이인지 |
| 조향이 너무 빠름 | `S 1500` 등으로 제한 |
