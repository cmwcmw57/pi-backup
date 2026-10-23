# 여의도 현장 실행 절차 (2026-10-18)

경로: `docs/yeouido_route_plan.json` (A 10° 오르기 → 180° 회전 → 내려오기 → B 30° 시도(막히면 후퇴) → C 15° 오르기). 속도 0.15 m/s. RL: Pi `res8_mlp_v1_rc4000` 잔차 ON. 시작 자세: `base_link` 중심 (0.2933, 0.3202) m, yaw +x(램프 방향), STL 좌표.

## 실행 방식 3가지 (위에서부터 준비가 쉽다)
| 방식 | 필요한 것 | 위치 | 상태 |
|---|---|---|---|
| **1. 열린 루프** | Pi `run_policy --cmd-udp`만 | 사후 추정(ORB+ESKF, 바퀴 로그, 줄자) | 테스트 통과(합성), 실기 미검증 |
| **2. 닫힌 루프 `--pose dr`** | 1 + Pi 텔레메트리 패치(`pi_patch/`) | 바퀴·gyro dead-reckoning | 합성 입력만 검증 |
| **3. 닫힌 루프 `--pose eskf`** | 2 + 라파 ORB `/orbslam3/base_pose`(visual-only) + `ROVER_MAPPING_ROOT` | ORB+ESKF (repeat1018a 설정) | 합성 입력만 검증, **실 ORB 미연결** |

스테레오 IR 좌우·IMU는 기록만 하고 오프라인 평가에만 쓴다.

## 현장 전 확인 (모든 방식)
1. 노트북 `git pull`한 저장소에서 `python3 -m pytest -q` (ESKF 테스트는 `ROVER_MAPPING_ROOT` 없으면 skip). 통과 수를 확인한다.
2. Pi: `~/ksrc`는 `main`보다 옛 커밋(`40499bc`)이고 작업 트리에 펌웨어 수정이 있다. **건드리지 않는다.** `run_policy`가 `--cmd-udp`, `--max-lin`, `--max-ang`, `--log`를 지원하는 것은 확인했다.
3. 정책 파일 확인: Pi `~/ksrc/pi/policy/res8_mlp_v1_rc4000.{pt,json}` (md5가 서버 export와 같음).
4. 로버를 **바퀴를 띄운 상태**에서 먼저 `run_policy --fake`가 아니라 실제 SPI로 짧게(0.05 m/s, 3초) 돌려 링크·부호를 확인한다 (기존 bringup 절차).
5. 시작 자세: 로버 `base_link` 중심을 STL 좌표 (293.28, 320.21) mm에 놓고 앞을 +x(A 램프 쪽)로 맞춘다. 줄자로 확인하고 사진을 찍는다.
6. 사람: 비상정지(전원/USB) 담당 1명, B 구간 안전줄 담당. B(30°)는 정책 학습 경사(0~20°) 밖이고 시뮬에서 입구에서 막혔다.

## 방식 1. 열린 루프
Pi:
```bash
cd ~/ksrc && python3 pi/policy/run_policy.py --policy pi/policy/res8_mlp_v1_rc4000 --cmd-udp 5005 --max-lin 0.15 --max-ang 0.8 --rt 50 --cpu 3 --gc-off --seconds 400 --log run_open1.npz
```
(시작하면 2초 정지해 자이로 바이어스를 잰다. "[run] 텔레옵 명령 UDP" 메시지가 뜬 뒤에 보낸다.)

노트북:
```bash
python -m experiments.yeouido_real.open_loop send docs/yeouido_open_loop.csv --to raspberrypi.local:5005
```
- 명령표는 110.6초(`make`로 다시 만들 수 있음). Ctrl-C로 즉시 정지(0 명령 5회). 명령이 0.3초 끊기면 Pi가 0, 500 ms 끊기면 펌웨어가 정지.
- 체크포인트 정지(0.5 m 간격 3초)에서 줄자로 위치를 잰다.

## 방식 2·3. 닫힌 루프
Pi에 텔레메트리 패치를 적용한다 (팀원 ksrc 코드 수정 — **사용자가 직접 적용**, 코드는 `pi_patch/`):
```bash
cd ~/ksrc && cp /path/to/yeouido-rover-route/pi_patch/telemetry_tee.py pi/policy/ && git apply --check /path/to/yeouido-rover-route/pi_patch/run_policy_tlm_tee.patch
```
`--check`가 통과하면 같은 명령에서 `--check`를 빼고 적용한다. 되돌리기: `git checkout pi/policy/run_policy.py && rm pi/policy/telemetry_tee.py` (Pi 작업 트리의 다른 수정은 건드리지 않는다).

Pi 터미널 1 (`run_policy`, SPI 독점):
```bash
cd ~/ksrc && python3 pi/policy/run_policy.py --policy pi/policy/res8_mlp_v1_rc4000 --cmd-udp 5005 --max-lin 0.15 --max-ang 0.8 --rt 50 --cpu 3 --gc-off --seconds 400 --tlm-socket /tmp/ksrc-run.sock --log run_closed1.npz
```
Pi 터미널 2 (`run_real`, 같은 Pi에서 실행해야 시계가 같다):
```bash
python -m experiments.yeouido_real.run_real --pose dr --tlm-socket /tmp/ksrc-run.sock --cmd-to 127.0.0.1:5005
```
`--pose eskf`는 `ROVER_MAPPING_ROOT=<rover-mapping 경로>`와 ROS 소싱, `/orbslam3/base_pose` 발행이 필요하다(`docs/yeouido_eskf_real_io.md`). 키보드: `c` = 체크포인트에서 줄자 위치로 재기준, `q` = 정지·종료.

## 기록·사후 분석
- `run_policy --log`의 npz, `run_real`의 `yeouido_trace_*.npz`(컬럼은 `docs/yeouido_run_sheet.md`), 줄자 체크포인트 CSV(`checkpoint_id, measured_x_m, measured_y_m`), IR 좌우+IMU 기록.
- 슬립 / 시뮬-실기 비교 / 시각차 스윕 / ORB 유지율: `experiments/yeouido_analysis/`. 시뮬 수치는 모래 텍스처 0.25 m 반복 인공물 때문에 실측으로 해석하지 않는다.

## 아직 안 된 것 (현장에서 막힐 수 있는 지점)
- 실기 모터·SPI 경로, 모래 램프 등판, 실제 ORB `base_pose`로는 한 번도 돌려 보지 않았다.
- 방식 3은 라파 ORB가 `base_pose`를 내고 visual-only여야 한다(`orbslam3_stereo_inertial` 노드가 IMU를 쓰는지 미확인).
- `lag_s`(영상 지연)와 `imu-offset-s`(−47 ms)는 실측값이 아니다.
- 열린 루프는 위치를 보지 않아 슬립이 쌓이면 계획 경로와 벌어지고, 기울기 중단도 못 쓴다.
