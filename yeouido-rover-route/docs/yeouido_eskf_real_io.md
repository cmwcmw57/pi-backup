# 여의도 `--pose eskf` 실기 입력 명세

`experiments/yeouido_real/run_real.py --pose eskf`가 실기에서 기대하는 입력과 아직 확인하지 못한 것. 시뮬 합성 입력(`--fake --pose eskf`)으로만 끝까지 검증했고 실제 ORB·IMU로는 돌려 보지 않았다.

## 구성
```
라파 run_policy (SPI 독점) ──텔레메트리(UNIX datagram/UDP)──▶ run_real ──KSRC UDP 5005──▶ run_policy --cmd-udp
라파 ORB(orbslam3) ──/orbslam3/base_pose──▶ run_real                         (rclpy, ROS 소싱 필요)
```
`run_real`은 라파에서 돌린다 (ORB 헤더 stamp와 텔레메트리 수신 시각이 같은 시계여야 한다). 필터는 rover-mapping의 `OnlineEstimator`를 읽기 전용으로 import한다: 환경변수 `ROVER_MAPPING_ROOT`(체크아웃 경로, 필수)와 scipy가 필요하다. rover-mapping은 수정하지 않는다. 라파에는 그 repo의 `experiments/localization/{online,fusion,inertial_filter}.py` 3개와 scipy 설치가 필요하다 (미확인: 라파에 scipy가 있는지).

## 1. ORB 입력 (실 ORB 연결에서 필요한 것)
| 항목 | 값 |
|---|---|
| 토픽 | `/orbslam3/base_pose` (`geometry_msgs/PoseStamped`), `/orbslam3/tracking_state` (`std_msgs/Int32`, 2=OK). `pi/policy/vo_relay.py`가 구독하는 것과 같다 |
| pose | ORB 지도 좌표의 **base_link** 자세 (`pose_output:=base`: x 전진, y 좌측, z 위). 카메라 pitch 20° 보정이 반영된 base여야 한다 (`eskf_calibration.json`의 `ROS.T_c1_base`와 같은 것) |
| `frame_id` | `orb_map_<n>`. 끝의 정수를 map id로 쓴다. 바뀌면 필터가 영상 기준을 다시 잡는다 |
| stamp | 영상 촬영 시각, 라파 `time.time()` 시계 |
| 주기 | 약 15 Hz (필터는 0.1 s마다만 관측으로 쓴다) |
| ORB 종류 | **영상만 쓰는(visual-only) ORB**여야 한다. `fusion.py`는 IMU가 이미 융합된 ORB 입력을 거부한다. 라파에서 도는 노드 이름이 `orbslam3_stereo_inertial`인데 실제로 IMU를 쓰는지 **미확인** (`headless-pi-manual-mapping.md`는 기본 launch에 외부 IMU를 연결하지 않는다고 적음) |
| 지연 | `--lag-s`(기본 0.25 s): IMU를 이만큼 늦춰 영상과 맞춘다. ORB pose가 도착하는 지연(라파 `vo_relay` 기록상 약 0.23 s)보다 같거나 커야 한다. 실측해서 지정 (미측정) |

ORB 좌표계와 경로 좌표계의 정렬은 필요 없다. 필터는 ORB의 **상대 이동**만 쓰고, 시작 pose(STL `0.293, 0.320, z 0.463, yaw 0`)는 필터가 직접 갖는다.

## 2. IMU 입력
IMU는 Nucleo SPI 텔레메트리 하나뿐이다 (ORB 쪽에는 IMU 경로가 없다). SPI는 `run_policy`가 독점하므로 **`run_policy`가 텔레메트리를 내보내야** 한다. 현재 라파의 `run_policy.py`(10-18 버전)는 내보내지 않는다 (명령 보낸 쪽에 5 Hz JSON 요약만 회신, wheel_mps/gyro/acc 없음).

필요한 변경 (라파 `run_policy.py`, 이 작업에서는 수정하지 않음): SPI 교환마다 `decode_spi_telemetry` 결과를 `pi/common/telemetry_tee.py`의 `TelemetryTee(path).emit({"host_receipt_ns": time.time_ns(), "telemetry": tlm})`로 내보낸다. `udp_spi_bridge.py`용 `ksrc_telemetry_tee.patch`가 쓰는 payload와 같은 모양이며, `run_policy`에 같은 3~4줄이 필요하다 (`--telemetry-socket` 인자). `run_real --tlm-socket PATH`가 그 소켓을 열어 받는다. UNIX datagram은 받는 쪽이 하나뿐이라 매핑 기록기(`ksrc_telemetry_capture.py`)와 같은 경로를 쓰면 안 된다. 대안: UDP JSON(`run_real.telemetry_packet(tlm)`, `--tlm-port 5007`).

| 필요한 텔레메트리 키 | 뜻 |
|---|---|
| `t_ms`, `tick` | Nucleo 시각 [ms], 8비트 tick (같은 tick 중복은 버림) |
| `flags` | bit4(`IMU_VALID`)가 0이면 그 프레임 IMU는 쓰지 않는다 |
| `gyro_dps`, `acc_g` | 로버 몸체 좌표 (x 전진, y 좌측, z 위). 단위는 `decode_spi_telemetry` 그대로(dps, g). 이미 칩→몸체 변환이 끝난 값이라 어댑터에서 다시 뒤집지 않는다 |
| `steer_meas_rad`, `wheel_mps` | `--pose eskf`는 쓰지 않는다 (dr 모드 전용). 키는 있어야 소스가 패킷을 받아들인다 |

시각: `end_s = t_ms/1000 + (최근 10 s 최소 수신 지연) + imu_offset_s`. `--imu-offset-s` 기본 −0.047 s는 legacy SPI의 카메라 대비 IMU 지연(−45~−50 ms)이다. 펌웨어를 `imu_mean_read_offset_us` 버전으로 바꿨다면 값을 다시 재야 한다 (미확인). IMU 쪽을 옮기고 영상은 옮기지 않는다.

## 3. IMU 끊김 방어 (테스트로 확인)
- IMU 프레임 간격이 0.1 s를 넘거나, 유효 IMU가 0.1 s 넘게 없거나(`flags` invalid 연속), 텔레메트리가 끊기면: 정지 명령(`ready=False`), 필터는 전진하지 않는다. `OnlineEstimator.propagate`는 dt가 (0, 0.1] 밖이면 예외라 그런 구간은 필터에 넘기지 않는다.
- 정상 IMU가 1 s 이어지면 마지막 유효 pose에서 필터를 새로 만든다 (속도 0, gyro bias 유지). 이후 ORB 보정이 한 번 받아들여질 때까지 정지.
- 한계: 끊김 감지 후 정지까지 로버가 더 간 거리(수 cm)는 필터에 반영되지 않는다. 재시작 뒤 체크포인트에서 `c`로 줄자 값을 넣는다.
- ORB 보정이 4 s 넘게 받아들여지지 않으면(`invalid_s`) 정지한다. 복귀하면 재개한다.

## 4. 시뮬과 다른 점·미확인
- 시뮬 `repeat1018a` 설정과 같은 것: `adaptive_visual_noise=True`, `good_s=1`, `invalid_s=4`, 보정행렬 `eskf_calibration.json`(= `docs/repeat1018a/estimator_calibration.json`의 calibration), IMU noise 기본값.
- 다른 것: `lag_s` 0.25 (시뮬 렌더 지연 0.1은 실기 값이 아니다). 시작 정지 구간(2 s, 100 프레임)에서 gyro bias와 시작 roll/pitch(중력)를 잡는다 (필터 코드의 `fusion.run`과 같은 방식).
- 미확인: IMU 몸체축이 시뮬 `base_from_imu`(보정행렬 안)와 같은 방식으로 base에 놓이는지 (텔레메트리를 base 좌표로 보고 `base_from_imu`의 회전만 역으로 적용). 실기 IMU noise(정지 밀도 gyro 8.2e-5, acc 1e-3 측정값이 있으나 필터에는 시뮬 기본값 사용). 실제 녹화 IMU+ORB로 이 어댑터를 재생해 본 적 없음.
