# 10-16 저녁 최종 점검 (17일 모래 연습장 준비)

목적: 17일 `MEASURE_1017.md` 를 현장에서 막힘 없이 돌릴 수 있게, 오늘 바닥에서 **펌웨어·링크·정책·새 옵션·로그**를 한 번씩 다 통과시켜 둔다.
낮의 `MEASURE_1016.md` (IMU·진동 측정) 를 못 했으면 C 단계에 핵심만 넣어 두었다.

총 약 1시간: A 10분 → B 10분 → C 15분 → D 20분 → E 5분.

## A. 노트북 → Pi (10분)

```bash
# 노트북
cd ~/ksrc
scp pi/policy/{run_policy.py,student_runtime.py,vo_link.py,vo_relay.py,test_vo_link.py,res8_mlp_v3vo_rc4000.pt,res8_mlp_v3vo_rc4000.json} pi@raspberrypi.local:~/ksrc/pi/policy/
wc -c pi/policy/{run_policy.py,student_runtime.py,vo_link.py,vo_relay.py,res8_mlp_v3vo_rc4000.pt}
ssh pi@raspberrypi.local 'cd ~/ksrc && wc -c pi/policy/{run_policy.py,student_runtime.py,vo_link.py,vo_relay.py,res8_mlp_v3vo_rc4000.pt} && ls -la pi/policy/res8_mlp_v1_rc4000.*'
```
- 두 파일 크기가 노트북과 같아야 한다. 정책 파일 `.pt`, `.json` 둘 다 있어야 한다.

```bash
# Pi (ssh pi@raspberrypi.local)
cd ~/ksrc && mkdir -p logs/1016 logs/1017
RUN="$HOME/venv/bin/python pi/policy/run_policy.py --policy pi/policy/res8_mlp_v1_rc4000 --rt 50 --cpu 3 --gc-off"
$RUN --fake --cmd 0.15 0 0 --seconds 1 --bias-s 0.2
$RUN --fake --cmd 0.15 0 0 --seconds 1 --bias-s 0.2 --res-ema 0.7 --res-steer-deg 15
```
| 확인 | 정상 |
|---|---|
| 첫 줄 | `EMA α 끔, 조향 잔차 폭 30°` |
| 둘째 줄 | `EMA α 0.7, 조향 잔차 폭 15.0°` |
| `[run] 루프 주기` | 평균 20 ms 근처 |
| 에러 | 없음 (`ModuleNotFoundError` 면 venv 경로 확인) |

VO 연결(17일 S4 용) 도 가짜 자세로 확인 (ORB 없이):
```bash
# Pi 노트북에서 받은 파일 3개 더: vo_link.py, vo_relay.py, res8_mlp_v3vo_rc4000.{pt,json}
python3 pi/policy/test_vo_link.py                                   # ALL PASS
timeout 10 python3 pi/policy/vo_relay.py --fake &
RUN3="$HOME/venv/bin/python pi/policy/run_policy.py --policy pi/policy/res8_mlp_v3vo_rc4000 --rt 50 --cpu 3 --gc-off"
$RUN3 --fake --cmd 0.15 0 0 --seconds 4 --bias-s 0.5 --vo-port 5600   # 출력 끝에 "VO (+0.15, +0.00) ~100 ms"
```

`--rt` 권한 경고가 나오면 17일에도 같으니 `sudo setcap cap_sys_nice+ep $(readlink -f ~/venv/bin/python)` 한 번.

### A2. Pi ORB 설치 확인 (5분, 17일 S4 가능 여부)
```bash
ls /home/pi/handheld_mapping/ros2_ws/install/orbslam3_ros2/lib/orbslam3_ros2/ /home/pi/handheld_mapping/runtime/ORBvoc.txt
ls ~/ksrc/pi/orb/rover.urdf; lsusb | grep -i intel     # D435
```
- 셋 다 있으면 시간 될 때 `ORB_VO_RUN.md` 2절대로 한 번 띄워 릴레이 숫자(자세 Hz, ORB 지연, 나이)를 메모.
- 없으면 17일 S4 는 생략 (동료에게 빌드 요청만).

## B. 펌웨어·서보 (바퀴 띄움, 10분)

**굽지 않는다 (10-16 저녁 확인)**: 지금 Nucleo 에는 동료 IMU 시각 펌웨어(flags bit12, 바퀴 PI ON)가 있다. 노트북 펌웨어로 구우면 그 기능이 지워진다. 잔차·게이트가 되는지는 C 의 bringup res/gate 로 확인.
```bash
# 노트북, Nucleo USB 연결
cd ~/ksrc/firmware/src2026-rover-firmware && ./build.sh --flash      # 필요할 때만
python3 -m serial.tools.miniterm /dev/ttyACM0 115200
```
```
LOG              로그 끄기
SCAN             서보 ID 1~4 모두 보임
T 0 0 0 0        네 바퀴 정면 (눈으로, 특히 뒤왼쪽)
T 30             앞왼쪽만 30°, 위에서 반시계
T 0 30 / T 0 0 30 / T 0 0 0 30    나머지도 반시계
T 0 0 0 0
WPI ON           바퀴 PI 켬 (17일 기본)
```
- **뒤왼쪽 영점이 틀어져 있으면** 17일의 "뒤왼쪽 바퀴 꺾임" 관찰이 정책 탓인지 영점 탓인지 구분이 안 된다. `T 0 0 0 0` 에서 네 바퀴가 몸체와 평행한지 자로 확인하고 사진 1장.
- 영점이 틀어져 있으면: `OFF` → 손으로 정면 정렬 → `ZERO ALL` → `ON` → `T 0 0 0 0`.
- miniterm 닫는다 (Ctrl+]). 이후는 SPI 로만.

## C. Pi 링크·정책 (바퀴 띄움 → 바닥, 15분)

```bash
# Pi, 바퀴 띄움
python3 pi/common/bringup.py link        # 텔레메트리·IMU 정상 -> PASS
python3 pi/common/bringup.py res         # 잔차 경로 0x02 -> PASS
python3 pi/common/bringup.py gate        # 정렬 게이트 -> PASS
$RUN --cmd 0.10 0 0 --seconds 5 --log logs/1016/air_rl.npz
$RUN --cmd 0.10 0 0 --seconds 5 --res-steer-deg 15 --res-ema 0.7 --log logs/1016/air_rl15.npz
```
| 확인 | 정상 |
|---|---|
| bringup 세 항목 | 모두 PASS |
| `air_rl` 출력 `잔차 ON` | ON, 바퀴 네 개 모두 앞으로 돎 |
| `air_rl` 뒤왼쪽 바퀴 | 눈에 띄게 꺾임 (시뮬·fake 와 같으면 약 +16°) |
| `air_rl15` 뒤왼쪽 | 그 절반쯤 (약 +7°) |
| 끝난 뒤 | 바퀴 멈춤 (0 명령 5번) |

바닥에 내려서, IMU 기준값(낮에 못 했으면):
```bash
$RUN --residual-off --cmd 0 0 0 --seconds 30 --log logs/1016/still.npz     # 수평, 모터 전원 ON, 손 대지 않기
```

## D. 바닥 주행 리허설 (17일 S0·S1 축소판, 20분)

직선 2 m, 출발선·1.5 m 테이프. 휴대폰 옆·뒤 촬영 연습 겸.
```bash
$RUN --residual-off --cmd 0.15 0 0 --seconds 10 --log logs/1016/floor_ik.npz          # IK
$RUN                --cmd 0.15 0 0 --seconds 10 --log logs/1016/floor_rl.npz          # RL 30°
$RUN --res-steer-deg 15 --cmd 0.15 0 0 --seconds 10 --log logs/1016/floor_rl15.npz    # RL 15°
$RUN --res-ema 0.7  --cmd 0.15 0 0 --seconds 10 --log logs/1016/floor_rlema.npz       # RL + EMA
$RUN --residual-off --cmd 0 0 0.6 --seconds 8 --log logs/1016/floor_ik_w06.npz        # 제자리 회전 IK
$RUN                --cmd 0 0 0.6 --seconds 8 --log logs/1016/floor_rl_w06.npz        # 제자리 회전 RL
```
보는 것
- **직진성**: IK 와 RL 이 1.5 m 에서 옆으로 얼마나 벗어나는지 (cm). RL 은 바퀴를 비스듬히 두고 다른 바퀴로 상쇄하는 구조라, 바닥에서 크게 휘면 17일 전에 알려줄 것.
- **뒤왼쪽 꺾임**: 뒤 영상으로 RL 30° / 15° 차이가 보이는지 (17일 바퀴 관찰 연습).
- **떨림**: RL 에서 서보 웅웅 소리·발열, EMA 에서 줄어드는지 (17일 S0 예행).
- **루프**: 각 주행 끝 `[run] 루프 주기` p99 ≤ 25 ms, `40 ms 넘음` 0회.
- **바퀴 소리·미끄러짐**: 바닥에서 바퀴가 끌리며 끼익 소리가 나는지 (비스듬한 바퀴).

이상하면 즉시 멈춤: 조이스틱/스페이스.

## E. 로그 회수 + 17일 짐 (5분)

```bash
# 노트북
mkdir -p ~/ksrc/sysid_logs/1016
scp pi@raspberrypi.local:~/ksrc/logs/1016/*.npz ~/ksrc/sysid_logs/1016/
```
npz 경로를 Claude 에게 주면 오늘 밤 안에 직진성·꺾임 각·떨림·루프를 정리한다 (17일 판단 기준선).

### 17일 가져갈 것
| 분류 | 물건 |
|---|---|
| 로버 | 배터리 완충 + 예비 배터리·충전기, Pi 전원(5 V 5 A PD), 육각렌치·드라이버, 예비 나사, 케이블타이, 절연테이프 |
| 통신·조작 | 노트북(충전기), Nucleo USB 케이블(펌웨어·UART 비상용), 조이스틱 PC 창 또는 키보드 비상정지 확인, 핫스팟(Pi ssh 용) |
| 측정 | 휴대폰 2대 + 거치대(옆·뒤), 줄자·자, 마스킹테이프, 경사계 앱, 메모장 |
| 모래 | 바퀴 자국 고르는 판(자·종이판), 모래 털 붓, 지퍼백(로버 덮개) |
| 문서 | `pi/MEASURE_1017.md` (휴대폰에 띄워 두기) |

### 17일 아침 출발 전 (2분)
```bash
ssh pi@raspberrypi.local 'cd ~/ksrc && ls pi/policy/res8_mlp_v1_rc4000.* && df -h ~ | tail -1 && vcgencmd measure_temp'
```
디스크 여유(로그 수백 MB), 정책 파일 확인.
