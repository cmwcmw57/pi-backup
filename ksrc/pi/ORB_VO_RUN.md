# Pi 에서 ORB 띄우고 VO 속도를 정책에 넣기 (2026-10-16)

동료 저장소 `lee-robot-lab/rover-mapping` (노트북 `~/ksrc/ORB`) 의 **headless Pi 경로**를 쓴다
(원문: `docs/experiments/headless-pi-manual-mapping.md`). Qt 매핑 앱은 쓰지 않는다:
- 앱의 "Pi 는 센서 전송만" 모드 → ORB 가 노트북에서 돌아 Pi 에 자세 토픽이 없다.
- 앱의 Pi ORB 모드 → URDF 없이 준비해서 `/orbslam3/camera_pose` 만 나오고 `base_pose` 가 없다.
- headless + `--urdf` 로 준비해야 `/orbslam3/base_pose` 가 나온다 → `vo_relay.py` 가 이걸 받는다.

Pi 에서 ORB 는 빌드·재생 시험까지 되어 있다 (동료는 녹화 재생으로만 시험, 아래 0). 라이브 카메라로는 처음.

## 0. 확인 (Pi, 10-16 저녁 확인 완료)

- ORB 실행 파일·스크립트: `/home/pi/handheld_mapping/ros2_ws/install/orbslam3_ros2/` 있음, 라이브러리 누락 0.
- vocabulary: `/home/pi/handheld_mapping/runtime/ORBvoc.txt` (145 MB) 있음.
- URDF: `~/ksrc/pi/orb/rover.urdf` (동료 저장소 `rover.urdf`, `camera_infra1_optical_frame` 포함. `rocker_bogie_4wd_rover.urdf` 는 이 프레임이 없어 안 됨).
  **카메라 장착: base_link 앞 0.117 m, 아래로 35° (pitch 0.611 rad)**. 실물 각이 다르면 몸체 속도가 섞이니 경사계로 확인.
- 동료 10-11 Pi 재생 시험 (`~/handheld_mapping/20261011_recorded_replay01`): 30 fps 녹화 912 프레임, 추적 OK 912/913,
  **ORB 처리 평균 33 ms / 최대 39 ms**, CPU 1.75 코어. 15 Hz 면 약 0.9 코어 예상.
- `lsusb | grep -i intel` 에 D435 가 보여야 한다 (10-16 저녁: 안 보임 → 꽂거나 USB3 포트 확인).

## 1. 공통 셸 설정 (tmux 창마다)

```bash
H=/home/pi/handheld_mapping
source /opt/ros/jazzy/setup.bash
source $H/ros2_ws/install/orbslam3_ros2/share/orbslam3_ros2/local_setup.bash   # install/setup.bash 는 없음
export LD_LIBRARY_PATH="$H/runtime/install/lib:$H/runtime/prefix/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"
export ROS_DOMAIN_ID=65 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=$H/ros2_ws/install/orbslam3_ros2/share/orbslam3_ros2/config/fastdds_stereo_local.xml
URDF=$HOME/ksrc/pi/orb/rover.urdf
VOC=$H/runtime/ORBvoc.txt
```
카메라·ORB·확인·릴레이 **모두 이 설정으로** 띄운다 (도메인이 다르면 릴레이가 자세를 못 받음, 공유 메모리 설정은 publisher 쪽에 필요).

## 2. 실행 순서 (tmux 창 5개)

```bash
# 창 1: D435 (IR 좌우만, 15 Hz, 프로젝터 끔)  -- 코어 0-2 에 둠 (정책은 3)
taskset -c 0-2 ros2 run realsense2_camera realsense2_camera_node --ros-args \
  -p enable_infra1:=true -p enable_infra2:=true -p enable_depth:=false -p enable_color:=false \
  -p depth_module.infra_profile:=640,480,15 -p depth_module.infra1_format:=Y8 -p depth_module.infra2_format:=Y8 \
  -p depth_module.emitter_enabled:=0 -p enable_sync:=true
```
```bash
# 창 2: 정지 상태 입력 확인 30 s -> 새 run 준비 (매번 새 RUN 이름)
ros2 run orbslam3_ros2 mapping_probe.py --run-dir ~/mapping_probe/p1017 \
  --left /camera/camera/infra1/image_rect_raw --right /camera/camera/infra2/image_rect_raw \
  --left-info /camera/camera/infra1/camera_info --right-info /camera/camera/infra2/camera_info --duration 30
RUN=~/mapping_runs/vo1017_01
ros2 run orbslam3_ros2 prepare_mapping_run.py --from-probe ~/mapping_probe/p1017/probe.json --rectified-input \
  --camera-profile 640x480x15:Y8 --urdf "$URDF" --vocabulary "$VOC" --run-dir "$RUN"
# ORB 시작 (vocabulary 로딩 수십 초)
taskset -c 0-2 ros2 launch orbslam3_ros2 manual_mapping.launch.py run_dir:="$RUN" vocabulary_path:="$VOC" image_preprocessing:=raw
```
```bash
# 창 3: 확인
ros2 topic hz /orbslam3/base_pose          # 15 Hz 근처면 OK (카메라를 손으로 조금 움직여 추적 시작)
ros2 topic echo /orbslam3/tracking_state   # 2 = OK
```
```bash
# 창 4: VO 릴레이 -> UDP 5600
cd ~/ksrc && python3 pi/policy/vo_relay.py      # 공통 설정(도메인 65) 한 창에서
# 2초마다: "자세 15 Hz  보냄 15 Hz  추적 OK  v=(+0.000, +0.000)  나이 ~150 ms  ORB 지연 평균 50 / 최대 90 ms"
```
```bash
# 창 5: 정책 (SPI 주인은 이것 하나. udp_spi_bridge 와 동시에 띄우지 않는다)
cd ~/ksrc
RUN3="$HOME/venv/bin/python pi/policy/run_policy.py --policy pi/policy/res8_mlp_v3vo_rc4000 --rt 50 --cpu 3 --gc-off"
$RUN3 --vo-port 5600 --cmd 0.15 0 0 --seconds 10 --log logs/1017/flat_v3_vo.npz
# 화면 끝에 "VO (+0.15, +0.00) 120 ms" 가 찍히면 연결됨. v1 으로 기록만: --policy pi/policy/res8_mlp_v1_rc4000 --vo-port 5600
# 끝나면: "VO 패킷 N 개, 유효 14.8 Hz" / "VO 나이 [ms]: 평균 / p95 / 최대, VO 없음 x% 스텝" / 루프 주기 p99
```

### 지연·Hz 판정 (주행마다 메모)
| 숫자 | 어디 | 좋음 | 문제 → 원인 |
|---|---|---|---|
| 자세 Hz | 릴레이 | 14–15 | < 12 → ORB 가 프레임을 놓침 (Pi 부하) |
| ORB 지연 평균/최대 | 릴레이 | < 80 / < 150 ms | 크면 ORB 처리 느림 (코어 겹침, 스로틀링) |
| 나이 = ORB 지연 + 창 절반(100 ms) + 전송 | 릴레이·정책 | 평균 < 250, p95 < 350 ms | ORB 지연은 작은데 나이만 크면 `--window` 줄임 |
| 유효 Hz | 정책 끝 | ≈ 자세 Hz | 많이 낮으면 추적 끊김 (VO 없음 % 같이 봄) |
| VO 없음 % | 정책 끝 | < 5 % | 크면 추적 잃음 / 튐 버림 |
| 루프 p99 | 정책 끝 | ≤ 25 ms | ORB 가 정책 코어를 뺏음 |

학습(v3, `VO_DR`) 범위: 측정 지연 60–300 ms + 다음 측정까지 보유 → 나이 약 60–500 ms, **500 ms 넘으면 VO 없음**으로 취급.
실물 나이가 이 안이면 학습 범위 안. 300 ms 근처를 자주 넘으면 학습 밖이라 v3 판단을 보류한다. 사후 비교는 npz `vo` 열(vx, vy, age).

## 3. 끝내기
1. 정책이 끝나 로버가 섰는지 확인.
2. 창 2 ORB: **Ctrl+C 한 번** 후 기다림 (지도 저장 최대 30 s). 강제 종료하지 않는다.
3. 창 4 릴레이, 창 1 카메라 Ctrl+C.
4. (선택) `ros2 run orbslam3_ros2 finalize_mapping_run.py --run-dir "$RUN"`

## 문제별
| 증상 | 확인 |
|---|---|
| `base_pose` 가 안 나옴, `camera_pose` 만 | run 을 `--urdf` 로 준비했는지 (`$RUN/calibration.json` 의 `pose_output` = base) |
| 추적 계속 OK 아님 | 정지 상태에서 시작, 처음엔 카메라 앞에 특징 있는 물체. 모래만 보이면 특징 부족 가능 |
| 릴레이 "튐 버림" 이 계속 늘어남 | 지도 전환·재위치 잦음. `ros2 topic echo /orbslam3/diagnostics` |
| 정책 루프 p99 > 25 ms | ORB 와 코어 겹침. `taskset -c 0-2` 확인, `vcgencmd get_throttled` |
| 나이 > 500 ms (정책은 VO 없음으로 취급) | Pi 부하·카메라 Hz. 릴레이 `--window 0.15` |
