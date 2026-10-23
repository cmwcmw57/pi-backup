# 여의도 주행 실행 시트 (2026-10-18)

## 확정한 구성
| 항목 | 값 | 이유 |
|---|---|---|
| RL 정책 | `res8_mlp_v1_rc4000`, 잔차 ON | VO(스테레오) 입력이 없는 정책이라 "스테레오는 측정 후 평가만" 방침과 충돌하지 않음. v3vo는 시뮬 성능이 더 좋지만(−56% vs −39%) VO UDP 5600 스트림이 필요 |
| 속도 | 0.15 m/s | 사용자 기존 RL 주행(10°, 15°) 값. 학습 범위(±0.25) 안 |
| 경로 | `docs/yeouido_route_plan.json` | A 오르기 → U턴 → B 시도(막히면 후퇴) → C 오르기 |
| 실기 pose | 휠 텔레메트리 + gyro dead-reckoning | 스테레오를 온라인 제어에 쓰지 않음 |
| 스테레오 | IR 좌우 + IMU를 기록만, 오프라인 평가 | 지난 실험에서 벽 융합이 바퀴와 같이 쓰면 yaw가 틀어짐 |

## 주의
- B(30°)는 정책 학습 경사(0~20°) 밖이다. 시뮬(강체)에서는 B 입구에서 막혔다.
- 명령 패킷은 Pi `udp_spi_bridge`와 같은 KSRC 20바이트(`<4sIfff`, UDP 5005). 노트북 송신기(`experiments/yeouido_real/command_sender.py`)가 이 형식을 쓴다.
- Pi 쪽 `run_policy.py`는 GitHub에 없는 새 버전(UDP 5005 수신, 로그 인자 `cmd_udp`, `max_lin`, `max_ang`)이라 내용은 미확인이다. 실행 전 Pi에서 `--help`로 옵션 이름을 확인한다.

## 남은 틈 (실기 closed-loop에 필요)
`run_real.py`는 Pi 텔레메트리를 UDP 5007로 받아야 pose를 추정한다. Pi의 `run_policy`가 텔레메트리를 UDP로 내보내는지 미확인이다. 내보내지 않으면 팀원 쪽에서 텔레메트리 송출 몇 줄이 필요하다(SPI는 한 프로그램만 쓰므로 별도 프로세스로는 못 읽는다).

## 시뮬 검증
- Isaac 서버 서버 실행 폴더 (sim4): 경로 완주 130 s (강체, 모래 침하·슬립 없음).

## 실기 로그 형식
`run_real`은 종료할 때(완주·중단·`q`·시간초과) trace를 npz 한 파일로 저장한다. 경로는 `--log`(기본 `yeouido_trace_YYYYmmdd_HHMMSS.npz`, 실행 폴더). 한 행 = 텔레메트리 한 프레임(약 50 Hz, orb 모드는 ORB pose 약 15 Hz)이고 모든 컬럼의 길이가 같다. pose 소스와 무관하게 같은 컬럼이다(`--pose dr|orb|eskf`).

| 컬럼 | 타입 | 뜻 |
|---|---|---|
| `t_ms` | int | 그 프레임의 시각 [ms]. dr·eskf는 Nucleo `t_ms`, orb는 ORB 헤더 stamp (소스마다 시계 기준이 다르므로 같은 실행 안에서만 비교) |
| `t_s` | float | 첫 프레임 기준 경과 시간 [s] (`RouteRunner`에 넘기는 `elapsed_s`) |
| `x_m`, `y_m`, `yaw_rad` | float | 추정 pose (STL 경로 좌표, yaw는 [-pi, pi]). 정렬 전 orb는 NaN |
| `pitch_rad`, `roll_rad` | float | 추정 기울기 (ROS 규약: pitch + = 코가 아래). 중단 판정(40°/25°)에 쓴 값 |
| `cmd_vx`, `cmd_vy`, `cmd_wz` | float | 이 프레임에 `CommandSender`로 보낸 body twist [m/s, m/s, rad/s], 클램프 전. `ready=0`이면 0 |
| `ready` | bool | pose를 믿고 `RouteRunner`를 돌렸는지 (False면 정지 명령). eskf: 시작 정지·ORB 보정 대기·IMU 끊김·ORB 상실, orb: 정렬 전·추적 실패·지도 변경·pose 튐 |
| `step_index` | int | `RouteRunner.index` (경로 step 번호, 완주 후 step 수와 같음) |
| `label` | str | 현재 segment 이름 (`done`이면 완주) |
| `status` | str | pose 추정기 상태. eskf: `ok`·`recovering`·`imu_gap`·`init`, 그 외 소스는 빈 문자열 |
| `meta` | str(JSON) | `pose_source`, `fake`, `route`, `result`(`complete`/`aborted`/`quit`/`timeout`) |

진짜(GT) pose는 없다. 오프라인 평가는 같은 시각의 ORB·줄자 체크포인트·영상과 `t_s`/`t_ms`로 맞춘다. 읽기: `np.load(path)["x_m"]`, 문자열 컬럼은 `d["label"].astype(str)`.
