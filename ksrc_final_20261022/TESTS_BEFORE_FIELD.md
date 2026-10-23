# 가기 전 테스트 설명서 — 터미널별 순서

위에서부터 순서대로 진행한다. 실행 명령은 [CLI.md](CLI.md), 현장 합격 기준은 [FIELD_TESTS.md](FIELD_TESTS.md)에 있다. S1 plan 분리가 기본이다. `--async-plan`은 생략해도 된다. S0 비교는 `--sync-plan`을 명시한다.

## 먼저 알아둘 출력과 다음 단계 조건

| 단계 | 예상 출력·화면 | 다음으로 넘어갈 조건 | 안 맞으면 |
|---|---|---|---|
| 1 점검 | `OK: YW16 + YW17 / f`, `RT priority limit: (50, 50)` | import·hash 통과, RT 한도 50 이상 | `check.log` 확인 |
| 2 기동 | `started FAKE supervisor ...`, `plan: S1 async ...` | B에서 FAKE / 대기 / 명령0 / 센서 정상 | 상태 생성 전이면 2초 후 다시 view. 계속 없으면 supervisor.log·runner.log |
| 3 1회차 | 요청 `sent start1` → 화면 AUTO, run=1 | 실제로 AUTO 전환·명령 발생 | run=0은 시작 전. 3분 기다리지 말고 준비 조건 확인 |
| 3 E-stop | 요청 `sent estop` → 대기 / TELEOP, 명령0 | AUTO 해제·0 유지 | 다음 단계 중단, logs 확인 |
| 4 anchor·2회차 | `operator_anchor` → AUTO, run=2, ascent15 | 적용 이벤트·2회차 수락 확인 | 정지 3초·센서·추정 출발 정렬 확인 |
| 5 SSH 재접속 | 새 start 없이 기존 상태 조회 | 같은 세션·로그 시각 계속 증가 | supervisor.log·stopped.json 확인 |
| 6 종료 | `stopped owned processes`, NPZ 저장 | report에 주기 요약 | 정상 stop 후에도 NPZ 없으면 policy.log |
| 7 실제 정지 | REAL / run=0 / 대기 / 명령0 | 2~3분 센서·부하 관찰, 움직이지 않음 | 복구 반복·예상치 않은 명령이면 종료하고 로그 확인 |
| 8 회수 | rsync 종료 코드 0, 로컬 시험 폴더 | 노트북에 summary.txt·NPZ·load 존재 | C가 Pi SSH 창이 아닌지 확인 |

**부하 3분/10분과 START의 정지 3초는 다른 시간이다.** `orb_ok`·출발 좌표가 정상이어도 실제 장소가 맵과 일치한다는 뜻은 아니다. FAKE는 합성 출발 pose를 넣는다. REAL도 사람이 배치를 확인해야 한다.

## 0. 노트북에서 터미널 3개 열기

| 터미널 | 명령이 실행되는 곳 | 역할 |
|---|---|---|
| **A — 제어** | 노트북에서 SSH 접속한 **Pi** | 시작·전환·E-stop·종료 |
| **B — 관찰** | 노트북에서 별도로 SSH 접속한 **Pi** | 상태·부하·로그 관찰 |
| **C — 회수** | **노트북 로컬**, SSH 접속하지 않음 | 마지막에 로그 다운로드 |

현재 확인한 구성은 노트북 LAN `192.168.0.100` → Archer C80 LAN, Pi Wi-Fi `home` 2.4 GHz → `192.168.0.101`이다. 노트북 Wi-Fi 핫스팟은 인터넷용이다. DHCP 주소가 바뀌면 공유기에서 확인한 IP로 수정한다. 이 절차에 서버·VPN·인터넷은 필요 없다.

**[터미널 A · 노트북 로컬]**

```bash
PI_AP_IP='192.168.0.101'
ssh -o HostKeyAlias=raspberrypi.local pi@"$PI_AP_IP"
```

**[터미널 B · 노트북 로컬]** — 별도 터미널이므로 IP 변수도 다시 지정한다.

```bash
PI_AP_IP='192.168.0.101'
ssh -o HostKeyAlias=raspberrypi.local pi@"$PI_AP_IP"
```

A와 B는 이제 Pi에서 명령을 실행한다. `hostname`으로 Pi인지 확인할 수 있다. C는 SSH하지 않고 노트북에 둔다. 변수는 터미널 간 공유되지 않으므로 아래의 폴더 복원 명령도 각 터미널에서 실행한다.

## 1. 점검하고 FAKE 시험 폴더 만들기

**[터미널 A · Pi SSH]**

```bash
hostname
cd /home/pi/ksrc_final_20261022/deployment/ksrc
TEST_DIR="state/fake_precheck_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_DIR"
printf '%s\n' "$TEST_DIR" > state/latest_test.txt
date -Is | tee "$TEST_DIR/actions.log"
./ksrc check 2>&1 | tee "$TEST_DIR/check.log"
df -h .
```

**확인:** `check`의 import·hash가 통과해야 한다. RT limit은 50 이상인지 확인한다. 오류가 있으면 여기서 멈추고 `check.log`를 확인한다. 이후 세션은 한 번에 하나만 실행한다.

## 2. 모터 없는 S1 실행과 부하 기록 시작

**[터미널 A · Pi SSH]**

```bash
./ksrc --state-dir "$TEST_DIR" start --seconds 600
nohup bash record_load.sh "$TEST_DIR" 180 > "$TEST_DIR/load_collector.log" 2>&1 < /dev/null &
echo "collector_pid=$!" | tee -a "$TEST_DIR/actions.log"
./ksrc --state-dir "$TEST_DIR" view | tee -a "$TEST_DIR/actions.log"
```

`--real`이 없으므로 모터·SPI를 열지 않는다. 부하 기록은 3분, 실행 세션은 최대 10분이다. 수집기를 같은 폴더에서 다시 실행하면 부하 파일을 덮어쓰므로 중복 실행하지 않는다.

**[터미널 B · Pi SSH]** — A에서 만든 시험 폴더를 불러온다.

```bash
cd /home/pi/ksrc_final_20261022/deployment/ksrc
TEST_DIR="$(cat state/latest_test.txt)"
printf '관찰할 폴더: %s\n' "$TEST_DIR"
./ksrc --state-dir "$TEST_DIR" view --watch
```

**다음 단계 조건:** 시작 전 `대기 / TELEOP`, 명령 세 값 모두 0. `orb_ok=true`, `stationary_s>=3`, `in_region1=true`, `start_aligned=true`가 될 때까지 기다린다. B의 요약 화면을 계속 켜 둔다. 같은 화면을 1초마다 갱신하므로 긴 JSON·행렬을 읽지 않아도 된다.

## 3. TELEOP → MPPI → E-stop 확인

**[터미널 A · Pi SSH]** — 아래 블록은 하나씩 실행하고 B에서 결과를 본다.

```bash
date -Is | tee -a "$TEST_DIR/actions.log"
./ksrc --state-dir "$TEST_DIR" start1 | tee -a "$TEST_DIR/actions.log"
```

**다음 단계 조건:** B에서 `상태: AUTO | run=1 | 단계=entry30`와 MPPI 명령이 나와야 한다. `sent start1`만으로 수락된 것은 아니다. `run=0`이면 시작 전/거부 상태다. `start_rejected`이면 START 성공으로 기록하지 말고 A에서 `status`로 준비 미달 이유를 확인한다.

**[터미널 A · Pi SSH]**

```bash
date -Is | tee -a "$TEST_DIR/actions.log"
./ksrc --state-dir "$TEST_DIR" estop | tee -a "$TEST_DIR/actions.log"
```

**다음 단계 조건:** B의 다음 상태 갱신에서 `대기 / TELEOP`, 명령 세 값 모두 0. 센서가 정상이어도 AUTO가 저절로 다시 켜지면 안 된다. 상태 로그는 5 Hz이므로 실제 정지 지연의 정밀 측정값은 아니다.

**[터미널 A · Pi SSH]** — E-stop 후 최소 1초 기다린 뒤 실행한다.

```bash
./ksrc --state-dir "$TEST_DIR" teleop 0.03 0 0 --seconds 1
```

FAKE라 실제 바퀴는 움직이지 않는다. 이 명령은 gateway에 1초간 전진 입력을 보내고 끝에 0을 보낸다. 실제 GUI 버튼과 바퀴 정지는 별도 실기 시험이다.

## 4. 정지 후 2회차 시작 확인

**[터미널 A · Pi SSH]** — 3초 이상 정지·정상 센서 조건을 확인한 뒤 진행한다.

```bash
./ksrc --state-dir "$TEST_DIR" anchor
```

**다음 단계 조건:** B의 최근 이벤트에 `operator_anchor`가 나타나야 한다. `anchor`는 로버를 움직이는 명령이 아니다.

**[터미널 A · Pi SSH]**

```bash
./ksrc --state-dir "$TEST_DIR" view | tee -a "$TEST_DIR/actions.log"
./ksrc --state-dir "$TEST_DIR" start2
```

**다음 단계 조건:** B에서 `AUTO | run=2 | 단계=ascent15`. 정지 3초는 필요하지만 부하 수집 종료까지 3분 기다릴 필요는 없다. 합성 입력이므로 실제 등판·완주 성공을 의미하지 않는다.

## 5. FAKE에서 SSH 연결 끊었다가 재접속

**[터미널 A · Pi SSH]** — AUTO 상태에서 실행한다. 이때 `stop`은 실행하지 않는다.

```bash
date -Is | tee -a "$TEST_DIR/actions.log"
./ksrc --state-dir "$TEST_DIR" view | tee -a "$TEST_DIR/actions.log"
exit
```

A만 노트북 로컬로 돌아온다. B는 그대로 Pi의 로그가 계속 증가하는지 관찰한다. 이것은 **SSH 한 세션 종료 시험**이며 AP 무선 단절 시험과 다르다.

**[터미널 A · 노트북 로컬]**

```bash
PI_AP_IP='192.168.0.101'
ssh -o HostKeyAlias=raspberrypi.local pi@"$PI_AP_IP"
```

**[터미널 A · Pi SSH, 재접속 후]**

```bash
cd /home/pi/ksrc_final_20261022/deployment/ksrc
TEST_DIR="$(cat state/latest_test.txt)"
date -Is | tee -a "$TEST_DIR/actions.log"
./ksrc --state-dir "$TEST_DIR" view | tee -a "$TEST_DIR/actions.log"
cat "$TEST_DIR/supervisor.json"
```

**확인:** 같은 시험 폴더·supervisor PID, 로그 시각 증가, AUTO 상태 유지. 최대 실행 시간 600초가 지난 뒤 확인하면 이미 자동 종료했을 수 있다.

## 6. 부하 로그 확인 후 FAKE 종료

**[터미널 B · Pi SSH]** — Ctrl+C로 요약 화면만 끝낸 뒤 실행한다.

```bash
./ksrc --state-dir "$TEST_DIR" report
```

**예상 출력:** `CPU0: 평균 ...% / 최대 ...%`, 온도·throttle·복구 횟수. 아직 정책이 실행 중이면 NPZ 없음은 정상이다. 부하 수집 시작 후 3분이 지나면 아래 파일이 생긴다. 원시 `top` 로그는 문제가 있을 때만 열며 `less`의 `:` 화면에서는 q로 나온다.

**[터미널 B · Pi SSH]**

```bash
cat "$TEST_DIR/load/finished.txt"
```

파일이 없으면 아직 수집 중일 수 있다. 3분이 지나도 없으면 `cat "$TEST_DIR/load_collector.log"`로 오류를 확인한다.

**[터미널 A · Pi SSH]**

```bash
date -Is | tee -a "$TEST_DIR/actions.log"
./ksrc --state-dir "$TEST_DIR" estop
./ksrc --state-dir "$TEST_DIR" stop
cat "$TEST_DIR/stopped.json" | tee -a "$TEST_DIR/actions.log"
tail -n 20 "$TEST_DIR/policy.log"
ls -lh "$TEST_DIR/policy.npz"
./ksrc --state-dir "$TEST_DIR" report | tee "$TEST_DIR/summary.txt"
```

**다음 단계 조건:** `stopped owned processes`, `stopped=true`, 정책 NPZ 존재. `report`에 하위 주기의 중앙/p99/최대와 25 ms 초과 횟수가 나와야 한다. 강제 종료·전원 차단이면 NPZ 저장이 빠질 수 있으므로 정상 `stop`을 사용한다. 정책 파일이 없는 노트북 단독 FAKE에서는 NPZ가 생성되지 않는다.

## 7. 선택: 실제 센서 정지 부하 확인 — 기본 3분, 열 안정성은 10분

이 단계는 `--real`로 **실제 SPI·모터 제어가 열린다.** 시뮬 검증 후 로버를 안정적으로 고정하고 바퀴를 띄운 상태에서 진행한다. 기존 직접 SPI 제어 프로세스를 종료한다. FAKE 종료를 확인한 뒤 새 폴더를 만든다. 이 단계는 **START 없이 정지 관찰만** 한다.

**[터미널 A · Pi SSH]**

```bash
TEST_DIR="state/real_idle_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_DIR"
printf '%s\n' "$TEST_DIR" > state/latest_test.txt
date -Is | tee "$TEST_DIR/actions.log"
./ksrc check 2>&1 | tee "$TEST_DIR/check.log"
./ksrc --state-dir "$TEST_DIR" start --real --seconds 240
nohup bash record_load.sh "$TEST_DIR" 180 > "$TEST_DIR/load_collector.log" 2>&1 < /dev/null &
echo "collector_pid=$!" | tee -a "$TEST_DIR/actions.log"
./ksrc --state-dir "$TEST_DIR" view | tee -a "$TEST_DIR/actions.log"
```

실측 보정 `local/real.json`·`local/orb.yaml`이 필요하다([설정 안내](deployment/ksrc/README.md)).

**[터미널 B · Pi SSH]** — 새 폴더를 다시 불러온다. 이전 FAKE 폴더를 그대로 보지 않는다.

```bash
TEST_DIR="$(cat state/latest_test.txt)"
./ksrc --state-dir "$TEST_DIR" view --watch
```

**다음 단계 조건:** 화면 `REAL · 실기 스택`, `run=0`, `대기 / TELEOP`, 명령 0 유지. 센서 정상·복구 횟수 안정 여부를 관찰한다. **이 단계에서는 `teleop`, `anchor`, `start1`, `start2`, GUI START를 실행하지 않는다. 움직이는 시험이 아니다.** 3분 뒤 B에서 Ctrl+C 후 `load/finished.txt`를 확인하고, A에서 **6단계의 종료 명령을 다시 실행**한다. 길게 보려면 시작 전 `--seconds 240`을 `--seconds 660`, 수집 `180`을 `600`으로 바꾼다. 10분은 필수가 아니며 온도가 오른 뒤 안정성 확인용이다. 실행 중 일찍 stop해도 부하 수집기는 원래 시간까지 계속되므로 `finished.txt`가 없어도 저장 실패로 단정하지 않는다. `report`는 종료 표시 이후 CPU·온도 표본을 제외한다. 이 정지 시험은 실제 영상 입력 부하는 포함하지만 주행 중 영상·MPPI 계산 부하를 대표하지 않는다.

실제 ORB 크래시·추적 손실·바퀴 전환 시험은 [FIELD_TESTS.md](FIELD_TESTS.md)의 별도 절차를 따른다. 집/실험실에서 임의 위치를 `anchor`해 AUTO를 시작하지 않는다. 의도한 출발은 **base_link 중심 (3.60, 2.82) m / site −y, yaw −90°**다.

## 8. 노트북으로 로그 회수

**[터미널 C · 노트북 로컬, SSH 접속하지 않은 창]**

```bash
hostname
PI_AP_IP='192.168.0.101'
mkdir -p ~/ksrc-precheck-logs
rsync -a -e 'ssh -o HostKeyAlias=raspberrypi.local' pi@"$PI_AP_IP":/home/pi/ksrc_final_20261022/deployment/ksrc/state/ ~/ksrc-precheck-logs/
ls -lt ~/ksrc-precheck-logs/
```

**확인:** 노트북의 `~/ksrc-precheck-logs/` 아래에 시험 폴더가 있어야 한다. Pi SSH 안에서 실행하면 다운로드 대상도 Pi가 되므로 C에서만 실행한다.

## 저장되는 값과 읽는 법

| 파일 | 기록·해석 |
|---|---|
| `check.log`, `actions.log` | 환경 점검, 조작 시각·status. 실제 바퀴 정지 시각은 영상으로 별도 기록 |
| `navigation.jsonl` | 5 Hz 상태·pose/IMU age·단계·복구 이벤트. 50 Hz 상위 tick p99·MPPI 계산 시간 전체를 측정하는 로그는 아님 |
| `policy.log`, `policy.npz` | 정상 종료 시 하위 `period_ms`, `compute_ms`, 관측·행동·telemetry. 중앙값·p99·최대·25 ms 초과 횟수 확인 |
| `load/top_threads.log` | 1초 간격 코어별·스레드별 CPU·메모리. 첫 프레임 제외 |
| `load/processes.log` | 5초 간격 PID·PPID·스레드·현재 CPU·RT 우선순위·RSS. `ps`의 CPU%는 생존 기간 평균 |
| `load/vmstat.log` | 메모리·swap·실행 대기·CPU idle·I/O wait. 첫 데이터 행은 부팅 이후 평균이므로 제외 |
| `load/thermal.log` | 온도(m°C)·throttle 비트·ARM clock. `throttled=0x0`인지 전후 비교 |
| `load/meta.txt`, `load/finished.txt` | 환경·실행 파일 hash·소유 PID, 수집 정상 완료 시각 |

CPU는 스레드 100%=코어 하나 기준으로 보고 코어별 idle도 확인한다. S1은 runner·MPPI worker가 CPU 0, ORB가 1·2, 정책이 3이다. runner·worker는 PID/PPID로 구분한다. 전체 평균 CPU가 낮아도 특정 코어는 포화될 수 있다. 수집기 부하도 포함되므로 비교 시 동일하게 켠다. FAKE 부하를 실제 ORB 여유로 해석하지 않는다.

## 시험 후 적을 결과

| 항목 | 결과 |
|---|---|
| 날짜·배포 commit·FAKE/REAL·시험 폴더 | |
| 시작·E-stop·재시작·SSH 재접속 결과 | |
| CPU 0/1/2/3·프로세스별 평균/최대·측정 구간 | |
| 메모리·swap·I/O wait 변화 | |
| 시작/최대/종료 온도·throttle 전후 | |
| 하위 주기 중앙/p99/최대·25 ms 초과·샘플 수 | |
| 센서 공백·복구 횟수·실패 stage/reason | |
| 노트북 로그 경로·실제 바퀴 정지 영상 | |

E-stop 후 AUTO가 되살아나거나 복구 중 자동 명령이 0이 아니면 다음 단계로 진행하지 않는다. GUI 조작·실제 바퀴 정지는 CLI FAKE 성공과 구분한다.

## 직접 확인한 기록 — 2026-10-22

Pi FAKE S1·CPU 0/3·RT 50에서 정상 `stop` 후 `policy.npz` 44 KB 생성. 하위 주기 934개에서 p99 **20.01 ms**, 최대 **20.02 ms**, 25 ms 초과 **0회**였다. 짧은 FAKE 표본으로 실기 부하 결과가 아니다. 부하 로그·완료 파일도 Pi에서 확인했다. 12초 수집 온도는 61.15→58.40→57.85°C, throttle은 `0x0`. FAKE 저장 종료 경로를 포함한 navigation 테스트 **16개 통과**. 원시 기록은 Pi `state/precheck_logverify_20261022/`, `state/precheck_logverify2_20261022/`이며 세션은 종료했다.

기존 S0/S1 시뮬 비교는 실제 +x 출발이었다. 의도한 −y 출발 검증과 구분한다.

## 실제 정지 부하 확인 결과 — 2026-10-22

`real_idle_20261022_215101`: 실제 camera·ORB·IMU·정책 실행, AUTO 미시작. 하위 주기 9,769개에서 중앙 **20.00 ms**, p99 **20.07 ms**, 최대 **20.14 ms**, 25 ms 초과 **0회**. CPU 평균은 코어 0/1/2/3 각각 **18.4 / 55.1 / 50.2 / 22.5%**(종료 표시 이전 197개 프레임, 첫 프레임 제외). 온도 시작 **63.9°C**, 최대 **69.4°C**, 마지막 **68.3°C**, throttle 표본 `0x0`. 복구 시작/완료 각각 1회. 정지 관찰 결과이며 주행·plan 계산 부하·완주 검증과 구분한다.
