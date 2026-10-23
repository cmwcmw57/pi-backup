# KSRC f판 — 현장 시험 절차서

이 문서는 **무엇을 어떤 순서로 시험하고 무엇을 합격으로 볼지** 정리한다. 현장 명령어는 별도 **[CLI.md](CLI.md)**, 배포·초기 배치는 **[AP 내부망 안내](deployment/ksrc/README.md)** 를 참고한다.

출발 전 사전 점검·부하 기록은 **[TESTS_BEFORE_FIELD.md](TESTS_BEFORE_FIELD.md)** 를 먼저 진행한다.

명령은 **[CLI.md](CLI.md)의 터미널 A(Pi SSH)** 에서 실행한다. 로그 관찰은 터미널 B(Pi SSH), `rsync` 다운로드는 터미널 C(노트북 로컬)에서 한다. 이 문서의 `./ksrc`·Python ORB 종료 명령은 모두 Pi에서 실행한다.

## 준비와 초기 배치

![초기 배치](deployment/ksrc/figures/initial_pose.png)

로버 `base_link` 중심을 구역 1의 **(3.60, 2.82) m**에 놓고, 정면(body +x)을 **site −y(yaw −90°)** 로 맞춘다. 15° 통로의 진입 방향이며 그림에서는 아래쪽이다. 위치 5 cm·방향 5° 이내를 목표로 바닥에 중심·정면 표시를 붙인다. **카메라나 앞바퀴 기준 좌표가 아니다.**

각 시험은 `check` 후 별도 `--state-dir`에 기록한다. 실제 구동은 시뮬·FAKE → 바퀴를 띄운 시험 → 짧은 평지 → 전체 코스 순서로 넓힌다. 아래 `state/T01`은 예시이고 매 시험마다 번호를 바꾼다. 안전 담당자는 물리 E-stop에 접근할 수 있어야 한다.

```bash
cd /home/pi/ksrc_final_20261022/deployment/ksrc
./ksrc check
./ksrc --state-dir state/T01 start --seconds 180  # FAKE부터
./ksrc --state-dir state/T01 status
```

실기 시험 때만 위 `start`에 `--real --upper-cpu 0 --lower-cpu 3 --orb-cpus 1,2 --rt 50`을 추가한다. 선택한 S1 운용에는 `--async-plan`도 추가한다. 한 번에 세션 하나만 실행한다.

## 시험 항목과 합격 기준

| 순서 | 시험 | 조작·관찰 | 합격 기준 | 이번 확인 |
|---|---|---|---|---|
| 1 | 부팅·의존성 | AP SSH 후 `check`, `start`, `status` | 인터넷 없이 import·hash 통과, 시작 전 `active=false`, 명령 0 | Pi FAKE 확인 |
| 2 | TELEOP → MPPI | TELEOP 짧게 → 초기 위치 정렬 → 3초 정지 → `anchor` → `start1` | 정상 센서·정렬 때만 AUTO. GUI 조작을 계속하지 않아도 MPPI 명령 발생 | CLI FAKE 확인; 실제 GUI·실기 남음 |
| 3 | 준비 미달 시작 거부 | 움직이는 중, 3초 미만 정지, ORB 불량, 정렬 불량에서 START | AUTO에 진입하지 않고 0 자동 명령. status로 원인 확인 | 자동 테스트 확인; 현장 남음 |
| 4 | AUTO → E-STOP | AUTO 중 `estop` 또는 GUI E-STOP | 즉시 AUTO 해제, 다음 상태 갱신에서 명령 0. 이후 센서 정상이어도 자동 재개 없음 | Pi CLI FAKE 확인; GUI·정지거리 남음 |
| 5 | E-STOP → TELEOP → AUTO | E-STOP 후 TELEOP 조작 → 초기 위치 정렬·정지·anchor → START | TELEOP로만 움직이다 명시 START 뒤 AUTO. 과거 MPPI 명령이 새어 나오지 않음 | 기본 상태 자동 테스트; 실기 남음 |
| 6 | TELEOP 명령 끊김 | TELEOP 명령 송신을 중단 | gateway 300 ms stale 후 0. 실제 정지 시간·거리 별도 기록 | gateway 회귀 테스트; 실기 남음 |
| 7 | AUTO에서 SSH·노트북 단절 | AUTO 시작 후 SSH 창 종료/노트북 AP 해제, 나중에 재접속 | Pi supervisor·AUTO 유지. 재접속 후 동일 세션·임무 단계·로그 확인 | 설계/프로세스 분리 확인; 실제 AP 단절 남음 |
| 8 | 짧은 ORB 손실 | 바퀴를 띄우거나 평지 정지 상태에서 잠시 시야 차단 후 복원 | `recovering=true`, 명령 0 → 안정 추적 확인 후 기존 단계 재개. pose가 ORB 원점으로 튀지 않음 | core 자동 테스트; live ORB 남음 |
| 9 | ORB 크래시·자동 재시작 | 아래 소유 ORB 종료 명령 실행 | ORB PID/`orb_00N` 폴더 증가, 복구 중 0, 마지막 site pose·단계 보존, 안정화 뒤 재개 | core 재닻 테스트; 실제 프로세스 복구 남음 |
| 10 | 복구 중 두 번째 손실 | 복구가 안정되기 전에 다시 시야 차단 | 이전 안정화 시간을 재사용하지 않고 재닻·1초/5개 관측부터 다시 확인 | 회귀 테스트 확인 |
| 11 | 복구 중 E-STOP | `recovering=true`일 때 `estop`, 이후 시야 복원 | 추적만 복구하고 AUTO는 해제 상태 유지 | 상태 자동 테스트; live 조합 남음 |
| 12 | 전체 stop | `stop`, `stopped.json`, `processes.json` 확인 | 소유 프로세스 종료·0 명령. 관련 없는 ROS/SSH 프로세스 유지 | 호스트·Pi FAKE 확인 |
| 13 | 1회차 | 초기 배치 후 `start1`, 전체 구간 관찰 | 30° 직진 시도·후퇴, 15° 우회, 정상 관찰·직진 출구, 7° 하행·구역1 복귀 | 기존 +x 출발 f 시뮬 4/4; −y 출발 재검증·실기 남음 |
| 14 | 2회차 | 복귀 후 정렬·3초 정지·anchor → `start2` | 30°·정상 관찰 생략, 15° 등판·7° 하행, 하행 0.7배 | 설정 테스트·호스트 async FAKE 진입 확인; 실기 남음 |
| 15 | plan S0/S1 | 같은 배치·시드·시작 조건에서 `--async-plan`만 변경 | 완주·오차·정지 동작 우선 비교. 뚜렷한 차이 없으면 S1 채택 | S1 채택; 기존 비교는 +x 출발 조건 |

**2번 전환을 처음 확인할 때:** 먼저 FAKE에서 CLI로 확인하고, 다음에 기존 조이스틱 GUI의 `--gateway` 경로로 같은 시험을 한다. 실제 GUI 버튼·스페이스 키 동작은 GUI에서 직접 확인해야 한다. CLI 성공을 GUI 버튼 시험 통과로 표시하지 않는다.

**4번 정지 측정:** status 갱신은 5 Hz다. status가 0이 된 것과 실제 바퀴·차체가 정지한 시각을 따로 기록한다. UDP 소프트웨어 E-STOP과 물리 전원 차단 E-stop은 별개 시험이다. 물리 E-stop 동작은 이 코드가 구현한 기능이 아니다.

**7번 단절 시험:** 소프트웨어 E-STOP 패킷을 보낼 수 없는 상황이므로 처음에는 바퀴를 띄운 상태에서 시험한다. 실제 주행에서는 물리 E-stop 담당자가 필요하다. 노트북 단절 시 AUTO 계속 운용은 사용자 합의한 동작이다.

## ORB 자동 재시작 시험의 명령

실기 스택에서 바퀴를 띄우거나 정지 상태로 먼저 진행한다. `state/T09`의 현재 소유 ORB만 대상으로 하며, PID starttime을 확인해 다른 프로세스를 종료하지 않는다.

```bash
./ksrc --state-dir state/T09 status
# 이 명령은 ORB만 정상 종료시켜 supervisor의 자동 재시작을 시험한다.
# policy·SPI·카메라를 직접 종료하지 않는다.
/home/pi/venv/bin/python - <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, '.')
from cli import stop_owned
owned = json.loads(Path('state/T09/processes.json').read_text())
stop_owned(owned['orb'])
print('sent SIGTERM to this session ORB only')
PY
./ksrc --state-dir state/T09 status
tail -f state/T09/navigation.jsonl
```

관찰할 이벤트는 `recovery_start`, `recovery_done`이다. `recovery_done`의 `checkpoint`, `gap_s`, `stage`와 상태 pose를 비교한다. 새 `orb_002` 등의 디렉터리와 `supervisor.log`도 확인한다. 복구 중 로버를 옮기면 마지막 pose 연결의 전제가 깨지므로 그 시험은 별도 위치 재초기화 시험으로 분류한다.

## 기록할 값

시험표를 복사해 실제 결과를 채운다. **명령 0 / 바퀴 정지 / 차체 정지**를 구분한다. 실제 위치 오차는 바닥 마커나 별도 기준 측정으로 확인하며 추정 pose 자체를 정답으로 삼지 않는다.

| 항목 | 기록 |
|---|---|
| 날짜·담당자·Git commit | |
| 세션 경로·S0/S1·코어·RT | |
| 초기 위치 오차 / 방향 오차 | cm / deg |
| 시작 전 SoC 온도 / 전압 / throttle | |
| TELEOP → AUTO 결과·전환 지연 | |
| E-STOP → 명령0 / 바퀴정지 / 차체정지 | ms / ms / ms |
| ORB 손실 길이 / 재시작 횟수 / 복구 시간 | |
| 복구 전후 실제 위치·추정 위치 차이 | cm / cm |
| SSH 단절 후 AUTO 유지 / 재접속 | |
| 최종 결과·실패 stage·reason | |
| 보관 로그 경로·영상 | |

온도·전원 참고 명령(Pi):

```bash
vcgencmd measure_temp
vcgencmd get_throttled
vcgencmd pmic_read_adc EXT5V_V
```

## 이번 확인 기록

2026-10-22, 실제 모터·SPI를 열지 않은 확인이다.

- 호스트 최종 전체 회귀: **`947 passed, 24 skipped`**(144.26 s). 새 navigation 테스트 15개 포함. GPU·실기 의존 skip을 실제 주행 통과로 해석하지 않는다.
- Pi `./ksrc check`: Python `3.12.3`, runtime/grid hash·import OK, RT limit `(50, 50)`.
- Pi `configure`: 현재 URDF + 실측 640×480 camera intrinsics + IMU 보정으로 `local/real.json`, `local/orb.yaml` 생성.
- Pi ROS import: `ROS imports OK; no node or hardware opened`.
- Pi 기본 FAKE 세션: `runner`, `gateway`, `inputs`, 실제 `run_policy --fake` 프로세스 기동.
- 시작 전 status 예: `orb_ok=true`, `stationary_s=14.60`, `active=false`, `command=[0,0,0]`.
- `start1` 후 예: `active=true`, `run=1`, `stage=entry30`, MPPI 명령 발생.
- `estop` 후 예: `active=false`, `reason=operator`, `command=[0,0,0]`.
- `stop` 후 `stopped.json`에 `stopped=true` 확인.
- policy telemetry wrapper를 별도로 **기본 fake**로 실행해 SI IMU UDP 수신, `SIGTERM` → 기존 정책의 정지·저장 경로·종료 코드 0 확인. 짧은 샘플의 하위 p99 20.02 ms는 부하 실험 결론으로 쓰지 않는다.
- 호스트 `--async-plan` + `start2`: `run=2`, `stage=ascent15`, AUTO 명령 확인. 합성 평지 입력이라 실제 등판 성공을 의미하지 않는다.
- Pi `--async-plan --upper-cpu 0 --lower-cpu 3 --rt 50`도 FAKE로 확인: SCHED_FIFO 50 적용, `start2` 후 `ascent15`·AUTO 명령, `estop` 후 0·AUTO 해제, 소유 프로세스 전체 종료 확인.
- 새 자동 테스트: 차가운 시작 정지, START/STOP, 중복 IMU freshness, map 원점 100 m 변경에도 site pose·단계 연속성, IMU 공백, 복구 중 재공백 안정화 초기화, run2 규칙, KVO1 패킷, CLI fake 기본, MCU 중복 패킷 처리.
- 종료 경계 시험도 추가해 최신 navigation 테스트 **15개 통과**: `stop`이 소유 프로세스 종료를 확인한 뒤 반환하며, 부모가 SIGTERM handler를 등록했어도 plan worker가 종료된다. Pi의 최신 파일로 async 세션 기동→종료와 `stopped.json` 생성을 다시 확인했다.
- 최종 f판 배포 runtime으로 기록 재생: pose **13,504개**, 최대 차이 **1.4874e-7 m**, flag 불일치 0, 임무 단계 **16개**·코너 이벤트 **6개** 모두 원본 일치. 원시 요약은 [fidelity.json](deployment/ksrc/validation/fidelity.json). 이는 snapshot 계산 재현성 검증이며 live 입력·지연·실기 동작의 재현성 검증은 아니다.

실제 GUI 전환, AP 연결 단절, live ORB 프로세스 재시작, 정지 거리, 전체 실기 주행은 **현장 확인 대상**으로 남긴다.
