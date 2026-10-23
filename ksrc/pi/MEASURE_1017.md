# 10-17 (또는 18) 실측 절차 (모래 연습장)

목적
1. 실물 조향이 정책 명령처럼 떨리는지, 떨리면 EMA 필터로 잡히는지 (S0)
2. 모래 평지·경사에서 IK 대비 RL 이득이 있는지 (S1, S2)
3. **비스듬한 바퀴(조향 잔차)가 모래에서 손해인지** (S1·S2 바퀴 관찰, S5)
4. 연습장 모래의 유효 견인 계수 (S3)

정책은 **student v1** `res8_mlp_v1_rc4000` 하나. 조향 잔차 폭(`--res-steer-deg`)과 EMA(`--res-ema`)만 바꿔 가며 본다.

## 왜 S5 를 넣었나 (10-16 시뮬 결과)

- 정책은 직진 명령에서도 바퀴를 IK 와 다르게 비스듬히 둔다. MuJoCo 평지에서 v1 은 **뒤왼쪽 +16°**, 앞왼쪽 −3°, 뒤오른쪽 −4°, 앞오른쪽 바퀴 +25% 속도. Pi `--fake` 에서도 뒤왼쪽 +16.2° 로 같다. Isaac 평가 평균은 뒤왼쪽 +6°, 뒤오른쪽 −5°.
- Isaac 에서 조향 잔차 폭을 줄이면 이득이 크게 준다 (큰헛돎, IK 대비): 30° **−57%**, 15° **−27%**, 0° −9%. 조향 잔차가 이득의 대부분이다.
- 그런데 시뮬에는 비스듬한 바퀴가 모래를 밀어내는 저항(쟁기 효과)이 없다. 모래에서 손해인지 이득인지는 **실물로만 알 수 있다**.
- EMA α 0.7 은 시뮬에서 이득 손해 없이 정책 출력 흔들림 −1/3, 추가 지연 9 ms.

## 준비

### 노트북 → Pi (10-16 에 바뀐·새 파일, GitHub 에 아직 안 올렸으면. `CHECK_1016.md` A 에서 했으면 생략)
```bash
cd ~/ksrc
scp pi/policy/{run_policy.py,student_runtime.py,vo_link.py,vo_relay.py,test_vo_link.py,res8_mlp_v3vo_rc4000.pt,res8_mlp_v3vo_rc4000.json} pi@raspberrypi.local:~/ksrc/pi/policy/
ssh pi@raspberrypi.local 'cd ~/ksrc && wc -c pi/policy/*.py pi/policy/*.pt && ls pi/policy/res8_mlp_v1_rc4000.*'
```
`wc -c` 크기가 노트북과 같은지 확인. 정책 파일 `res8_mlp_v1_rc4000.{pt,json}` 이 있어야 한다.

### Pi (현장에서 한 번)
```bash
cd ~/ksrc && mkdir -p logs/1017
RUN="$HOME/venv/bin/python pi/policy/run_policy.py --policy pi/policy/res8_mlp_v1_rc4000 --rt 50 --cpu 3 --gc-off"
$RUN --fake --cmd 0.15 0 0 --seconds 1 --res-ema 0.7 --res-steer-deg 15   # 옵션 확인: "EMA α 0.7, 조향 잔차 폭 15.0°"
```
- 배터리 완충, 조이스틱(비상 정지) 연결, 휴대폰 2대 (옆·뒤), 자, 경사계 앱, 테이프.
- 평지 1.5 m 구간과 경사 구간에 출발선 테이프. 매 회 시작 전 바퀴 자국을 고르게 편다.
- `RUN` 시작 후 2초는 자이로 바이어스 측정이라 로버를 건드리지 않는다.

## 측정 순서 (우선순위 S0 → S1 → S3 → S2 → S5, 시간 없으면 S4 생략)

### S0 조향 떨림 (5–10분)
```bash
$RUN --cmd 0.15 0 0 --seconds 8 --log logs/1017/jit_air.npz                  # ① 바퀴 띄움, 잔차 켬
$RUN --cmd 0.15 0 0 --seconds 8 --log logs/1017/jit_ground.npz               # ② 모래 평지, 잔차 켬
$RUN --cmd 0.15 0 0 --seconds 8 --residual-off --log logs/1017/jit_ground_ik.npz   # ③ 모래 평지, IK
# 떨리면 추가
$RUN --cmd 0.15 0 0 --seconds 8 --res-ema 0.7 --log logs/1017/jit_ground_ema.npz  # ④ 모래 평지, EMA 0.7
```
보는 것: 서보 소리(웅웅)·손으로 느끼는 떨림·발열, 로그 `steer_cmd` vs `steer_meas` 흔들림, `steer_load`.

### S1 모래 평지 직진 A/B (핵심, 20분)
IK 와 RL 을 **번갈아** 각 3회 (IK1 → RL1 → IK2 → RL2 → IK3 → RL3).
```bash
$RUN --residual-off --cmd 0.15 0 0 --seconds 10 --log logs/1017/flat_ik_1.npz
$RUN                --cmd 0.15 0 0 --seconds 10 --log logs/1017/flat_rl_1.npz
# _2, _3 반복
```
S0 에서 떨림이 있었으면 RL 줄에 `--res-ema 0.7` 을 붙이고 파일명은 그대로.
보는 것: 이동 거리(테이프·영상), 방향 틀어짐, 헛돎(`wheel_mps` vs 실제 속도), **바퀴 관찰**(아래).

### S3 유효 견인 계수 (10분)
```bash
$RUN --residual-off --cmd 0.10 0 0 --seconds 15 --log logs/1017/mu_ik.npz
```
IK 로 경사를 천천히 오르다 헛돌기 시작하는 지점의 경사각 θ 를 경사계로 잰다. μ_eff ≈ tan θ.
**≤ 0.3 이면 RL 이 의미 있는 영역** (시뮬: μ ≤ 0.3 큰 이득, ≥ 0.5 이득 없음).

### S2 경사 A/B (20분)
같은 출발점에서 IK, RL 번갈아 각 3회. 경사각 기록.
```bash
$RUN --residual-off --cmd 0.15 0 0 --seconds 15 --log logs/1017/slope_ik_1.npz
$RUN                --cmd 0.15 0 0 --seconds 15 --log logs/1017/slope_rl_1.npz
# _2, _3 반복
```
보는 것: 끝까지 올랐는지, 멈춘 위치(경사→평지 전환부 포함), 헛돎·요 틀어짐, 바퀴 관찰.

### S5 조향 잔차 폭 15° (15분)
S1 과 S2 중 **RL 과 IK 차이가 컸던 곳**에서 15° 로 3회.
```bash
$RUN --res-steer-deg 15 --cmd 0.15 0 0 --seconds 10 --log logs/1017/flat_rl15_1.npz     # 평지면
$RUN --res-steer-deg 15 --cmd 0.15 0 0 --seconds 15 --log logs/1017/slope_rl15_1.npz    # 경사면
```
보는 것: 30°(S1·S2 의 RL) 와 IK 사이 어디쯤인지, 바퀴 앞 모래 쌓임이 줄었는지.

### S4 VO 잡음·지연 + v3 (선택, 15분)
ORB 띄우는 법은 `pi/ORB_VO_RUN.md` (Pi headless 경로, base_pose 필요). 띄운 뒤:
```bash
# Pi 터미널 1: ORB 자세 -> 몸체 속도 -> UDP 5600
source /opt/ros/jazzy/setup.bash && python3 pi/policy/vo_relay.py       # 2초마다 "자세 Hz, 보냄 Hz, v, 나이, ORB 지연"
# Pi 터미널 2: v1 주행하면서 VO 만 같이 기록 (v1 은 VO 를 안 씀)
$RUN --vo-port 5600 --cmd 0.15 0 0 --seconds 10 --log logs/1017/flat_rl_vo.npz
# (VO 가 잘 나오면) v3 로 같은 주행: VO 켬 / 끔
RUN3="$HOME/venv/bin/python pi/policy/run_policy.py --policy pi/policy/res8_mlp_v3vo_rc4000 --rt 50 --cpu 3 --gc-off"
$RUN3 --vo-port 5600 --cmd 0.15 0 0 --seconds 10 --log logs/1017/flat_v3_vo.npz
$RUN3                --cmd 0.15 0 0 --seconds 10 --log logs/1017/flat_v3_novo.npz
```
보는 것: 릴레이 출력의 자세 Hz(학습 5–17 Hz), ORB 지연, 정책 끝 요약의 VO 나이 평균·p95·VO 없음 %(판정표는 `ORB_VO_RUN.md`), 직진 중 v 가 0.15 근처인지·튀는지, 추적 끊김.
로그 `vo` 열 = (vx, vy, age) 로 바퀴·실제 속도와 비교 -> 시뮬 VO 모델(VO_DR) 범위를 좁힌다.

### 바퀴 관찰 (S1·S2·S5 의 RL 주행마다, 30초)
1. **주행 중**: 뒤에서 찍은 영상으로 뒤왼쪽 바퀴가 진행 방향에서 꺾여 있는지.
2. **주행 후 자국**을 바로 옆 IK 자국과 비교:
   - 자국이 진행 방향과 비스듬한지, 옆으로 밀린 모래 둑이 있는지
   - 뒤왼쪽 바퀴 앞·옆에 모래가 쌓였는지 (높이 대략 mm)
   - 자국 깊이(침하)가 IK 보다 깊은지
3. 자를 옆에 두고 사진 1장.

- 이상하면 즉시 멈춤: 조이스틱/스페이스. 정책은 끝나면 0 명령 5번 보내고 멈춘다.

## 판정

**S0 (떨림)**
- ② 에서 `steer_meas`·`steer_load` 가 조용하고 서보 소리가 ③ 과 비슷 → 그대로.
- ② 가 떨리고 ④ 가 조용 → 이후 RL 주행은 전부 `--res-ema 0.7`.
- ④ 도 떨림 → smooth 증류본 필요 (학습 몇 시간). 오늘은 그대로 진행하고 메모.

**S1·S2 (A/B) + 바퀴 관찰 → 조향 잔차의 운명**

| 결과 | 결정 |
|---|---|
| RL 이 더 멀리/끝까지 가고, 모래 쌓임이 IK 와 비슷 | 지금 정책 유지. 옆미끄러짐 벌점 넣지 않음 |
| RL 이 이기지만 뒤왼쪽 앞에 모래가 눈에 띄게 쌓임 | S5 결과로: 15° 도 비슷하게 이기면 배포는 15°, 재학습에 약한 옆 끌림 벌점 |
| RL 이 비기거나 지고, 모래 쌓임·깊은 자국 | 조향 잔차가 모래에서 손해 → 재학습: 옆 끌림 벌점 + Isaac 에 쟁기 저항 모델 |
| RL 이 지는데 모래 쌓임 없음 | 다른 원인(침하·지연·관측). 로그로 분석 |

- 3회 중 다수결.
- 시뮬 MuJoCo 모래(μ0.4) 에서는 IK 초반 실패, RL 은 경사→평지 전환부에서만 실패했다. 실물에서 같은 지점에서 멈추는지 본다.

## 끝나고 (노트북)
```bash
mkdir -p ~/ksrc/sysid_logs/1017
scp pi@raspberrypi.local:~/ksrc/logs/1017/*.npz ~/ksrc/sysid_logs/1017/
```
npz 경로와 경사각·영상·사진 메모를 Claude 에게 주면 S0 떨림 비교, A/B 지표, 조향 잔차 각도, μ_eff 를 정리한다.
로그에는 `res_ema`, `res_steer_deg` 가 같이 저장된다.
결과로 정하는 것: EMA·조향 폭 15° 적용 여부, MuJoCo sand 계수(μ_max 등) 교체, teacher 재학습 방향(옆 끌림 벌점·쟁기 저항·침하 모델).

## 미리 MuJoCo 로 보기 (선택, 노트북)
```bash
cd ~/ksrc && source /opt/ros/humble/setup.bash
python3 ros/isaac_bridge.py --terrain flat --soil sand --mu const:0.4 --view \
  --policy sim/rl/runs/isaac/2026-10-15_13-06-32_student_mlp_v1_rc4000/model_1000.pt --res-ema 0.7 --res-steer-deg 15
# 다른 터미널: ros2 topic pub -r 20 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.15}}"
# Nav2 까지: ros2 launch ros/nav2_isaac.launch.py soil:=sand mu:=const:0.4 policy:=... res_ema:=0.7 res_steer_deg:=15
```
