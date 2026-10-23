# KSRC Rover (Isaac) CLI 정리

서버(hpi-lab) `~/gyusa/ksrc_rover` 기준. 모든 실행은 `run.sh` 래퍼를 통함.

## 0. 배포 (노트북 → 서버)
```bash
bash sim/isaac/ksrc_rover/deploy.sh      # 노트북에서. 코드 바뀌면 매번.
```
- tar+ssh로 `~/gyusa/ksrc_rover`에 덮어씀(logs/·모델은 안 건드림). **서버 직접수정 금지**(다음 deploy에 덮임).

## 1. run.sh (docker 래퍼)
```bash
bash run.sh <컨테이너내_스크립트경로> <인자...>
```
- `docker run --rm --gpus "device=1" -v ~/gyusa/ksrc_rover:/workspace ... python.sh <스크립트> <인자>`
- **GPU=1 하드코딩**(그 줄만 수정). 엔트리포인트 python.sh → **`run.sh -c "..."` 는 bash 아니라 파이썬 실행**.
- `--rm` **포그라운드**라 끝날 때까지 블록(체인에 유용).

## 2. 학습 (train)
```bash
bash run.sh /workspace/scripts/train.py \
  --task Isaac-KSRC-Rover-{Direct8|Res8|Residual3}-v0 \
  --headless --num_envs 4096 --max_iterations 4000 --seed 42
```
- 스톡 rsl_rl train.py 래퍼. 인자: `--headless`(필수) `--num_envs` `--max_iterations` `--seed` `--resume --load_run <dir> --checkpoint <n>`(이어학습).
- **env/보상 설정은 CLI 아님** → 아래 CONFIG 상수로.
- 산출: `logs/rsl_rl/ksrc_rover_<mode>/<타임스탬프>/model_<iter>.pt` (50마다 저장, 끝에 `model_<N-1>.pt`). **타임스탬프 KST**(10-10 22시 이전 런은 UTC — 이미지에 tzdata 없음, `TZ=KST-9`로 해결).

## 3. 실험 설정 = 코드 상수 (환경변수 안 씀)
`source/ksrc_rover/rover_env_cfg.py` 상단 CONFIG 블록을 **고쳐 저장 → deploy → 재실행**:
| 상수 | 뜻 | 예 |
|---|---|---|
| `TERRAIN_KIND` | 지형 종류 | `rough_slope`/`plane`/… |
| `TERRAIN_MODE` | 경사·요철 범위 | `narrow`(0–10°)/`wide`(0–20°)/`off` |
| `WHEEL_MU` | 좌우 비대칭 μ | `wide`(±0.2)/`narrow`(±0.1)/`off` |
| `WHEEL_SEED` | μ 랜덤 시드(학습 4321) | |
| `STD_V`/`STD_OM` | 속도추종 exp std(보상B=0.05/0.20) | |
| `LINEAR` | 덴스 선형보상항 on/off | `False` |
| `SLIP_W` | 슬립 페널티 가중(0=끔) | `0.5` |
| `RATE_STEER_W`/`RATE_WHEEL_W` | 조향/바퀴 액션 1차 차분² (켜면 통짜 action_rate 끔) | `0.0` |
| `JERK_STEER_W`/`JERK_WHEEL_W` | 조향/바퀴 액션 2차 차분²(떨림) | `0.0` |
| `YAW_ACC_W` | 몸체 yaw 각가속도² | `0.0` |
| `SLIP_SQ` | 바퀴별 헛돎 속도 제곱 벌점 (가중, 데드존 m/s). 켜면 `SLIP_W=0` | `(2.5, 0.03)` |
| `DR_WIDE` | 넓힌 DR (`MU_PATCH_WIDE`·`MASS_SCALE_WIDE`·`COM_RANGE_M_WIDE`·`DR2_WIDE`) | `True` |
| `DRIVE_GATE` | 펌웨어 구동 정렬 게이트 (조향 60° 넘게 바뀌면 10° 안 들 때까지 바퀴 0, 1.5 s 타임아웃). 기본 켬 = 실기 | `True` |
| `DRIVE_MODEL` | 바퀴 구동: `isaac`(DCMotor 속도추종, 마찰 없음) / `pi`(펌웨어 WPI + duty 모터 + 기어마찰) / `openloop`(펌웨어 WPI OFF duty + 같은 플랜트) | `"pi"` |

`--set KEY=VALUE` 로 상수를 실행 때 덮어쓸 수 있다 (train.py·eval_log.py, 파이썬 리터럴). 문자열은 `--set DRIVE_MODEL='"pi"'`.
구동 확인: `bash run.sh /workspace/scripts/drive_check.py --headless --set DRIVE_MODEL='"pi"'` (바퀴 추종오차·duty 포화·몸체 속도오차).

## 4. 평가 (eval)
```bash
# 순수-IK 베이스라인 (checkpoint 없음 = zero-action; Res8/Residual3만 의미)
bash run.sh /workspace/scripts/eval_log.py --task Isaac-KSRC-Rover-Res8-v0 \
  --num_envs 64 --steps 1500 --wheel-seed 777 --out /workspace/logs/eval/baseline_ik.npz --headless
# 정책 평가
bash run.sh /workspace/scripts/eval_log.py --task Isaac-KSRC-Rover-Res8-v0 \
  --checkpoint /workspace/logs/rsl_rl/.../model_1499.pt \
  --num_envs 64 --steps 1500 --wheel-seed 777 --out /workspace/logs/eval/res8.npz --headless
```
- `--wheel-seed 777`(홀드아웃, 학습 4321과 다름) → 베이스라인·정책이 같은 μ 열.
- 분석(순수 numpy):
```bash
bash run.sh /workspace/scripts/eval_analyze.py <baseline.npz> <policy.npz>   # 짝비교
bash run.sh /workspace/scripts/compare_all.py                               # logs/eval/*.npz 통합표
```

## 5. detached 실행 + 순차 체인
```bash
# 길게 돌릴 땐 detached (VPN 끊겨도 서버서 계속)
cd ~/gyusa/ksrc_rover && setsid nohup bash run.sh /workspace/scripts/train.py ... \
  </dev/null > logs/mylog.log 2>&1 & disown
# 여러 개 순차 = 체인 스크립트 (run.sh가 블록하므로 자연 순차)
setsid nohup bash chain3.sh </dev/null > logs/chain.log 2>&1 & disown
```

## 6. 텐서보드
```bash
# 서버: (이미 ksrc_tb 떠있으면 생략. 런 바꿀 땐 docker stop/rm <ksrc_tb ID> 후 재실행,
#        특정 런만: --logdir 대신 --logdir_spec 이름:/logs/<dir>/<run>,이름2:...)
docker run -d --name ksrc_tb -p 6006:6006 -v ~/gyusa/ksrc_rover/logs/rsl_rl:/logs:ro \
  --entrypoint /isaac-sim/python.sh ksrc-rover:lab2.3.2-rsl312 -m tensorboard.main --logdir /logs --port 6006 --bind_all
# 노트북:
ssh -L 6006:localhost:6006 hpi-lab    # → http://localhost:6006
```

## 7. 모니터링
```bash
grep -aE "Learning iteration" logs/<log> | tail -1        # 현재 iter
docker ps --format '{{.ID}} {{.Names}} {{.Status}}'        # 컨테이너
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

## 8. ⚠ Docker 주의사항
- **stop은 컨테이너 ID로.** `docker stop --filter ancestor=ksrc-rover...` 하면 **TB(ksrc_tb)도 같은 이미지라 같이 꺼짐.**
- **띄우기 전 개수 확인(가드)**: `docker ps ...|grep -vc ksrc_tb` → 기대값일 때만. (ssh 끊겨도 서버선 실행돼 **중복 OOM** 잘 남.)
- **`timeout`으로 `docker run` 감싸지 말 것** — 클라이언트만 죽고 컨테이너 생존 → 겹침.
- **GPU(임시 규칙): `run.sh` GPU=1 그대로, 실행 전 항상 `nvidia-smi`로 GPU 1이 비었는지 확인하고 띄움.** 학습은 1개씩, 기본 4096env.
- **동료 컨테이너(oclat-dev 등) 건드리지 말 것.**

## 9. ⚠ AppLauncher 인자 함정
- `--kit_args`에 `--`로 시작하는 값은 **반드시 등호형**: `--kit_args="--/renderer/multiGpu/enabled=false"` (공백형은 argparse가 옵션으로 오인).
- **렌더(`--enable_cameras`/`--video`)는 현재 서버 RTX 렌더러 고장으로 세그폴트** — 헤드리스 학습/eval만 가능. (호스트 드라이버 이슈, 별도 해결 필요)

## 10. Sim2sim (MuJoCo, 노트북 CPU — 서버·GPU 불필요)
Isaac 에서 학습한 정책을 MuJoCo(다른 접촉 솔버, 모래 침하 모델 terramech)에서 돌려 본다.
관측 구성·액션 종류·명령 주기·μ 방식은 체크포인트 옆 `params/env.yaml`·`agent.yaml` 에서 자동으로 읽는다
(단일단계 29×2, teacher 29×2+특권 22, student 29×H, GRU 모두). 저장소 루트에서 실행.
```bash
# 0) 체크포인트 + params 를 노트북으로 (params 없으면 실행 거부)
mkdir -p sim/isaac/ksrc_rover/logs/sim2sim/ckpt/<이름>
ssh hpi-lab "cd ~/gyusa/ksrc_rover/logs/rsl_rl/<실험폴더>/<런> && tar cf - model_1499.pt params" \
  | tar -C sim/isaac/ksrc_rover/logs/sim2sim/ckpt/<이름> -xf -

# 1) 한 번 실행 → npz (eval_log.py 와 같은 형식)
python3 sim/rl/sim2sim_isaac.py --policy sim/isaac/ksrc_rover/logs/sim2sim/ckpt/<이름>/model_1499.pt \
  --scenario rough_sand --episodes 32 --out sim/isaac/ksrc_rover/logs/sim2sim/<시나리오>_<이름>.npz
python3 sim/rl/sim2sim_isaac.py --action Res8 --policy ik --scenario rough_sand --episodes 32 \
  --out sim/isaac/ksrc_rover/logs/sim2sim/<시나리오>_ik.npz          # IK 기준 (같은 seed = 같은 명령·μ·지형)

# 2) 분석 (Isaac eval 과 같은 도구)
python3 sim/isaac/ksrc_rover/scripts/eval_analyze.py <ik.npz> <정책.npz>          # 지표 + 짝비교
python3 sim/isaac/ksrc_rover/scripts/compare_all.py sim/isaac/ksrc_rover/logs/sim2sim IK=<stem> A=<stem> ...

# 3) 체인: 체크포인트 받기 → 모델 × 4 시나리오 → 표. 모델은 스크립트 맨 위 MODELS 에 한 줄씩
bash sim/isaac/ksrc_rover/scripts/chain_sim2sim.sh 32 | tee sim/isaac/ksrc_rover/logs/sim2sim/chain.log
```
| 옵션 | 기본 | 뜻 |
|---|---|---|
| `--policy` | `ik` | `model_XXXX.pt` (옆에 `params/`), 또는 `ik`(zero action → `--action Res8` 와 같이) |
| `--action` | 런에서 | Direct8 / Res8 / Residual3. 정책이면 env.yaml 의 액션 클래스 |
| `--scenario` | `flat_mu` | `flat_mu` 평지+Coulomb μ / `flat_sand` 평지+terramech 규사 / `rough_mu` 계단식 램프 지형+μ / `rough_sand` 램프+규사 |
| `--drive` | `isaac` | `isaac` = 학습과 같은 DCMotor 속도 목표 / `pi` = 펌웨어 바퀴 PI (`wheel_pi.c` 와 같은 torch 판, 20 ms, 엔코더 속도) → duty 모터 |
| `--wheel-friction-duty` | `0` | 바퀴 관절 쿨롱 마찰, duty 환산 (실측 운동 ~5, 기동 35–52). 0 = 학습과 같음 |
| `--sensors` | `ideal` | `real` = 바퀴 속도를 엔코더 틱 20 ms 차분으로 (1틱 0.0375 rad/s), 조향각을 서보 4096 격자로 |
| `--mu` | `auto` | `auto`(런의 μ 이벤트) / `s6`(바퀴별 0.1–0.6) / `patch`(본학습 방식) / `uniform:lo,hi` / `const:μ`. 규사면 terramech μ_max 에 바퀴별 배율로 들어감 |
| `--mu-resample-s` | 런에서 | μ 재추첨 간격 최소 최대 [s] |
| `--episodes` | 32 | 에피소드 수 (= npz 의 N 축) |
| `--episode-s`, `--cmd-resample-s` | 런에서 | 에피소드 길이, 명령 교체 간격 최소 최대 [s] |
| `--obs-delay`, `--act-delay` | 0 | 센서 관측·명령 지연 [정책 스텝] (DR 2단계 정책 시험용) |
| `--noise` | 꺼짐 | 런이 `enable_corruption` 이면 그 가우시안 σ 를 센서 항목에 적용 |
| `--mj-dt` | 0.005 | MuJoCo 물리 dt (Isaac 과 같게. 0.002 로 바꿔 수치 민감도 확인 가능) |
| `--seed` | 777 | 명령·μ·지형·스폰 순서. IK 와 정책이 같은 값이면 짝비교 |
| `--view` | 꺼짐 | MuJoCo 뷰어로 실시간 재생. 카메라가 로버를 따라감. 창을 닫으면 종료. `--out` 생략 가능 |
| `--speed` | 1.0 | `--view` 재생 배속 (2 = 2배 빠르게) |

- 속도: 평지 에피소드(8 s) 당 약 0.1 s, 규사/험지 약 1 s. 4 시나리오 × 3 모델 × 32 에피소드 ≈ 3 분.
- 관측 항목 이름이 `TERM_SOURCES`(sim2sim_isaac.py) 에 없으면 그 이름을 대며 멈춘다 → 새 관측을 추가했으면 거기에 MuJoCo 대응을 한 줄 추가.
- 미러 안 된 것: 로커(MuJoCo 등식 구속 사용), 조향 관절 속도 상한 5.45 rad/s, 특권 질량/CoM·DR 값은 공칭값.
- 전복(기울기 60° 초과)은 에피소드를 끝내고 `(전복)` 표시. 그 뒤 에피소드는 IK 와 seed 열이 어긋날 수 있음.

## 11. 정책 재생 보기 (Isaac 5.1 녹화 → 6.1 WebRTC, GPU1 만)
5.1 은 이 서버에서 RTX 렌더가 죽고(세그폴트) 6.1 은 된다. 그래서 5.1 에서 헤드리스로 정책을 돌려 **모든 링크의 월드 자세 + 지형**을 기록하고, 6.1 에서 그대로 재생(물리 없음)한다.
```bash
# 1) 서버: 녹화 (env 1개, 30초). 체크포인트 없으면 IK(zero action)
bash run.sh /workspace/scripts/record_play.py --task Isaac-KSRC-Rover-Teacher-Res8-v0 \
  --checkpoint /workspace/logs/rsl_rl/<실험폴더>/<런>/model_XXXX.pt --num_envs 1 --robots 1 --steps 1500 \
  --out /workspace/logs/play/<이름>.npz --headless

# 2) 노트북 (저장소 sim/isaac/ksrc_rover 에서)
bash tools/webrtc.sh replay                      # 서버에 6.1 재생 컨테이너 (최신 녹화, 카메라 로버 추적)
bash tools/webrtc.sh replay /workspace/logs/play/<이름>.npz -1   # 특정 녹화, 자유 시점(마우스로 조작)
REPLAY_LIGHT=0.5 bash tools/webrtc.sh replay                   # 화면이 하얗게 뜨면 조명 배율 낮추기 (기본 1.0)
REPLAY_ARROW=0 bash tools/webrtc.sh replay                     # 속도 화살표 끄기 (기본 2.0 m per m/s; 초록=명령, 파랑=실측, 초록 호=명령 yaw rate)
bash tools/webrtc.sh up                          # 터널+브리지+클라이언트 → 창에서 127.0.0.1 Connect
bash tools/webrtc.sh status                      # 각 구성 + 브리지 패킷 통계 (영상이 흐르면 숫자가 늘어남)
bash tools/webrtc.sh down                        # 전부 끄고 GPU1 메모리 반납
```
- 클라이언트: NVIDIA Isaac Sim WebRTC Streaming Client 2.0.0 (`~/.local/opt/isaacsim-webrtc-client`, sudo 없이 .deb 압축만 풂).
- VPN 이 끊기면 터널·노트북 브리지가 죽는다 → `down` 후 `up`.
- 재생 컨테이너는 GPU1 에서 약 3 GB 를 더 쓴다(학습과 같이 돌아감). 다 보면 `down`.

## 태스크 ID
`Isaac-KSRC-Rover-{Direct8|Res8|Residual3}-v0` (학습), `...-Play-v0` (평가용 소수 env).
