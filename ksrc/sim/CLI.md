# 시뮬 · 학습 CLI

0~5절의 학습·평가 명령은 `sim/rl/`에서 실행한다. 설명과 학습 결과는 [README.md](README.md),
ROS/Nav2 런치는 [루트 CLI.md](../CLI.md), 펌웨어는 [firmware/CLI.md](../firmware/CLI.md).

```bash
cd ~/ksrc/sim/rl
```

---

## 0. 빌드 / 검증

펌웨어 C 코드를 공유 라이브러리로 빌드한다. **기구학을 수정했으면 반드시 먼저 실행한다** —
시뮬과 STM32 가 같은 `firmware/common/*.c` 를 쓰기 때문에, 안 하면 정책이 학습한 기구학과
실기체 기구학이 어긋난다.

```bash
bash ../build.sh
```

모델 물리 검증 9종. 인자 없음.

```bash
python3 check_model.py
```

| | 검증 |
|---|---|
| [1] | 평지 정착 — 차체 높이, 로커각, 접촉력 합 = 무게 |
| [2] | 디퍼렌셜 — `rock_L + rock_R = 0` |
| [3] | 풀스로틀 — 차체속도가 이론 무부하와 일치 |
| [4] | 비틀림 접지 — 40mm 턱에서 네 바퀴 유지 |
| [5] | 횡경사 유지 — 슬립모델 ON, 횡방향 접지력 |
| [6] | 방향별 추종 — 전진/후진/게걸음/대각/제자리회전 |
| [7] | 요 외란 저항 |
| [8] | 조향 스텝 응답 |
| [9] | 누적 조향 드리프트 — 180° 등가 접기 검증 |

견인력 곡선과 고착 재현 검증. 인자 없음.

```bash
python3 test_terramech.py
```

---

## 1. 학습

조향 목표각이 한 주기에 60° 넘게 바뀌는 큰 방향 전환에서는 네 바퀴가 목표의
10° 이내에 들어올 때까지 구동을 함께 보류하는 규칙이 `RoverEnv`의 기본값이다.
학습·평가·RViz 순수 IK·RViz RL·텔레옵에 동일하게 적용된다. 평소 작은
조향 오차마다 멈추지 않도록 큰 전환에만 대기를 시작한다.
기존 `runs/s2`와 `runs/s3_path_reward`는 경로 시작 투영·로버 크기 이격을
고치기 전에 학습했다. `runs/s4_route_fix`는 수정 전 보상으로 학습했다.
현재 보상 함수의 성능 기준으로 간주하지 않는다.
새 정책은 별도 디렉터리에 학습하고 순수 IK와 짝비교한다.
보상 함수를 수정해 재학습할 때는 `reward.py`를 바꾸고 실행마다 다른
`--out` 디렉터리를 지정한다. 학습 시작 시 사용한 `reward.py`와 `env.py`가
그 디렉터리의 `src/`에 복사된다.

```bash
cd ~/ksrc/sim/rl
python3 train.py --steps 1000000 --envs 16 --terrain sand --stage1 --out runs/s5_reward
python3 compare_policy.py --model runs/s5_reward/ppo_400000_steps.zip \
  --vecnorm runs/s5_reward/ppo_vecnormalize_400000_steps.pkl --episodes 30
```

### 밑단 목표 속도 추종 (`cmdvel`, 2026-10-06)

이 과제는 경로 진행·헤딩 보상을 쓰지 않고, `(vx, vy, ω)` 명령과 **실제 차체속도**의
오차를 보상한다. 학습은 8×8m 절차 규사지형에서 하며 `--terrain sand`를 줘도
원본 대회 높이맵을 늘려 쓰지 않는다. 대회맵은 `--eval-geometry arena`로 별도
평가한다. `cmdvel`의 기본 난이도는 0.8로 고정한다. 경로 완주율로 난이도를
올리는 콜백은 이 과제에서 사용하지 않는다.

```bash
cd ~/ksrc/sim/rl
python3 train.py --task cmdvel --action-mode body --terrain sand \
  --steps 1000000 --envs 16 --randomize --log-std-init -2.0 \
  --episode-s 8 --out runs/cmdvel_body_01
python3 compare_policy.py --task cmdvel --action-mode body --terrain sand \
  --model runs/cmdvel_body_01/final.zip \
  --vecnorm runs/cmdvel_body_01/vecnorm.pkl \
  --episodes 100 --episode-s 8 --difficulty 0.8 --randomize
```

`--action-mode wheel`은 IK 바퀴속도에 바퀴별 ±30% 잔차를, `body+wheel`은
차체 3D와 바퀴 4D를 함께 준다. 두 모드의 관측은 각각 1850/2000차원이고,
현재 펌웨어 SPI 명령과 ROS 정책 브리지는 **3D 차체 명령만** 받으므로 바퀴
모드는 연구용이다. `--velocity-obs`는 시뮬 **참 차체속도** 3D를 관측에 더해
속도 센서가 있을 때의 상한을 재는 진단용이다. 실기에 연결하려면 독립적인
대지속도 추정기와 지연·오차 검증이 먼저 필요하다.

`--history-frames`는 기본 50(1초)이다. 5프레임 참속도 진단 모델은 다음처럼
학습·평가한다. 짧은 이력과 참속도는 실기 정책 설정이 아니다.

```bash
python3 train.py --task cmdvel --terrain sand --velocity-obs \
  --history-frames 5 --steps 200000 --envs 8 --randomize \
  --episode-s 8 --checkpoint-every 50000 --out runs/cmdvel_truth_probe
python3 compare_policy.py --task cmdvel --terrain sand --velocity-obs \
  --history-frames 5 --randomize --difficulty 0.8 --episode-s 8 \
  --episodes 60 --seed0 6000 \
  --model runs/cmdvel_truth_probe/final.zip \
  --vecnorm runs/cmdvel_truth_probe/vecnorm.pkl
```

참속도 비례 보정의 상한은 `compare_policy.py`에 `--oracle-gain 0.3`을 더해
같은 시드에서 측정한다. `behavior_clone_cmdvel.py --gain 0.3 --rounds 3`은
정책이 방문한 상태를 다시 모아 모방한다. 이것은 오라클 진단이며 강화학습
결과나 실기 속도 센서가 아니다. `--push-force 1.0`은 매 에피소드 0.3초
평면 외력을 추가하고, 외력 후 1초의 오차를 별도로 출력한다.
`compare_policy.py --assist-cap 0.25`는 평가 때 보조 행동을 제한하고,
`--oracle-ki 0.1 --oracle-i-tau 1.0`은 참속도 비례 제어기에 누설 누적
보정을 더하는 진단 옵션이다. 둘 다 현재 시험에서는 p90 또는 외력 복구를
개선하지 못했다. `train.py --w-tail 0.1`은 속도 오차의 이차 비용을
추가하는 실험 옵션(기본 0)이다. 모방 정책에서 PPO를 이어 학습할 때는
`--resume-lr`, `--resume-target-kl`, `--reset-optimizer-on-resume`,
`--reset-reward-norm-on-resume`으로 갱신 폭과 보상 통계를 제어할 수 있으나,
10만 스텝 실험은 학습 전 `s33b`보다 나빴다. 수치와 시드는
[README](README.md#2026-10-06-밑단-속도추종-과제)에 기록했다.

```bash
python3 traction_gate.py --terrain loose --slope-deg 10 --speed 0.2 --episodes 12
```

이 오라클은 시뮬의 참 바퀴 슬립을 읽는다. 바퀴별 제어에 이득이 있는지
학습 전에 검사하는 용도이며 실기 정책이 아니다.

완료된 s8을 **추가 200만 스텝** 이어 학습할 때는 모델과 관측 정규화 통계를
함께 복원한다. 새 디렉터리를 지정해야 기존 s8 결과와 당시 코드가 보존된다.

```bash
python3 train.py --resume runs/s8_paper_heading --steps 2000000 --envs 16 \
  --terrain sand --stage1 --out runs/s8_resume_2m
```

중간 체크포인트에서도 재개할 수 있다. `--resume runs/<이름>/ppo_4800000_steps.zip`
처럼 지정하면 같은 스텝의 `ppo_vecnormalize_4800000_steps.pkl`을 자동으로 읽는다.
`--steps`는 기존 스텝을 포함한 최종값이 아니라 **추가 스텝 수**다. 기존
`reward.py`·`env.py`와 현재 코드가 다르면 기본적으로 중단한다. 보상이나 환경을
의도적으로 바꾸며 이어 학습할 때만 `--allow-source-change`를 붙인다.
Stage 1 이외에는 커리큘럼 진행 상태가 저장되지 않아 `--d0`로 재개 난이도를
직접 지정해야 한다. 학습은 PPO rollout 크기(`512 × --envs`) 단위로 끝나므로
요청한 추가 스텝을 조금 넘을 수 있다. 모델·옵티마이저·정규화 통계·누적 스텝은
복원하지만 실행 중이던 에피소드와 난수 상태는 재현하지 않는다.

학습 중 20만 스텝마다 모델과 그 시점의 `vecnormalize`가 함께 저장된다.
40만 스텝은 예시일 뿐이며 20만·40만·60만·80만·100만 스텝을 같은 미학습
경로에서 비교해 횡오차·완주율·잔차 포화가 가장 나은 체크포인트를 고른다.
이전 `s4_route_fix`는 40만→400만 스텝 동안 횡오차가 악화됐으므로 `final`을
자동으로 최선으로 간주하지 않는다.
잔차 벌점의 영향만 확인할 때는 `reward.py`의 `w_resid`만 바꾸고 같은
`--seed`, `--path-seed`, 스텝 수로 별도 실행한 뒤 두 런의 같은 스텝
체크포인트를 비교한다.

두 명령의 `--drive-align-gate-deg` 기본값은 대기 해제 오차 10°다. 이전 동작과 비교할 때만
둘 다 `--drive-align-gate-deg 0`으로 맞춘다.

### teacher (특권 관측)

특권정보 43차원(흙 8 / 로버 3 / 차체속도 3 / 슬립 4 / 침하 4 / 수직항력 4 / 배걸림 2 /
지형스캔 15)을 인코더로 `z(8)` 에 압축한다. 실기체에서는 못 쓰고, 증류의 교사로만 쓴다.

```bash
python3 train.py --teacher --steps 4000000 --envs 16 --out runs/teacher
```

### blind (대조군)

같은 관측 길이인데 특권정보가 없다. teacher-student 구조가 **실제로 값어치가 있는지**
가르는 기준선이다.

```bash
python3 train.py --steps 4000000 --envs 16 --out runs/blind
```

### 인자

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--steps` | 4000000 | 병렬 환경 전체의 정책 스텝 합계. 16환경에서는 PPO 반복마다 512×16=8192스텝씩 진행 |
| `--envs` | CPU 코어 수 | 병렬 환경. 16 코어면 16 |
| `--d0` | 0.0 | 시작 난이도 |
| `--episode-s` | 20.0 | 에피소드 최대 길이 [초]. 기본 50 Hz에서 최대 1000 정책 스텝, 10000 물리 스텝 |
| `--seed` | 0 | |
| `--out` | `runs/ppo` | 저장 경로 |
| `--resume` | 없음 | 기존 run 또는 모델 체크포인트와 짝인 정규화 통계를 복원 |
| `--allow-source-change` | off | 저장 당시와 보상·환경 코드가 다른 경우 명시적으로 허용 |
| `--teacher` | off | 특권 관측 + 인코더 |

기존 `--clip-steer-deg`/`--clip-duty`는 3D 잔차에 적용되지 않는 무효 옵션이라 제거했다.
순수 IK는 `compare_policy.py`에서 잔차 `a=0`으로 평가한다.

### 산출물

```
runs/<이름>/
  final.zip            최종 정책
  vecnorm.pkl          관측 정규화 통계  ← 평가할 때 반드시 같이 준다
  ppo_*_steps.zip      200k 마다 체크포인트
  ppo_vecnormalize_*.pkl
  tb/                  텐서보드 로그
```

> `vecnorm.pkl` 없이 정책만 돌리면 스케일이 어긋난 입력을 받아 결과가 무의미하다.

---

## 2. 증류 (teacher → student)

student 는 고유수용감각 이력만으로 `ẑ` 를 맞히도록 배운다. 정책 머리는 **동결**이고
`AdaptationModule` 만 학습한다. 데이터는 student 가 직접 굴러다니며 모은다 (DAgger).

```bash
python3 distill.py --teacher runs/teacher/final --vecnorm runs/teacher/vecnorm.pkl
```

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--out` | `runs/teacher/student.pt` | |
| `--iters` | 30 | DAgger 반복 |
| `--steps-per-iter` | 6000 | 반복당 수집 스텝 |
| `--epochs` | 4 | 반복당 회귀 에폭 |
| `--batch` | 512 | |
| `--lr` | 1e-3 | |
| `--w-act` | 0.0 | 행동 손실 가중치. 0 = RMA 설정, >0 = Lee 2020 설정 |
| `--difficulty` | 1.0 | |
| `--buffer` | 200000 | 집계 버퍼 |

---

## 3. 진단 — 증류를 시작해도 되는지

**성공률만 보면 안 된다.** PPO 가 인코더를 무시하고 고유수용감각만으로 풀어버리면
`z` 가 죽은 변수가 되고, 성공률은 멀쩡한데 증류할 내용이 없다.

```bash
python3 probe_teacher.py --model runs/teacher/final --vecnorm runs/teacher/vecnorm.pkl
```

| | 재는 것 | 판정 |
|---|---|---|
| [1] | `z` 의 에피소드 간 표준편차 | max < 0.02 면 인코더가 상수 → 여기서 멈춤 |
| [2] | `z → 흙 파라미터` 선형 R² | 진단용. 판정용 아님 |
| [3] | `z` 를 **다른 에피소드 것으로 바꿔치기** 했을 때 낙폭 | 주 판정 |

[3] 은 같은 씨드(= 같은 지형·흙)로 쌍대 비교하고, 목표 도달 수·이동거리 같은
연속 지표를 주로 본다. 이진 성공률은 24 에피소드에서 표준오차가 10%p 라 검정력이 낮다.
`z=0` 은 정책이 본 적 없는 분포 밖 입력이라 보조로만 쓴다.

`--episodes` (기본 24), `--difficulty` (기본 1.0) 로 조절한다.

---

## 4. 평가

**학습 지형이 아니라 실제 경기장 STL 에서 재는 게 핵심이다.**

```bash
# 실제 규사 경사지형 (주 평가 대상)
python3 eval.py --arena --terrain sand --episodes 30 \
                --model runs/teacher/final --vecnorm runs/teacher/vecnorm.pkl

# 기준선만 (모델 없이)
python3 eval.py --arena --terrain sand --episodes 30

# 절차생성 랜덤 지형
python3 eval.py --difficulty 1.0 --episodes 30

# s2 정책을 순수 IK와 같은 시드·미학습 경로에서 직접 짝비교
python3 compare_policy.py --model runs/s2/final --vecnorm runs/s2/vecnorm.pkl \
                          --episodes 100
```

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--model` / `--vecnorm` | 없음 | 없으면 기준선만 |
| `--arena` | off | 실측 STL 로 평가 |
| `--terrain` | `sand` | `sand`(규사 경사지형, 주) / `rock`(암석 착륙지, 후순위) |
| `--episodes` | 20 | |
| `--difficulty` | 0.8 | `--arena` 에서는 **흙에만** 영향 (지형은 항상 실측 STL) |
| `--episode-s` | 20.0 | |

### 비교군

| 이름 | 무엇 |
|---|---|
| PP+IK | 잔차 0 |
| 일률감속 ×0.8 / ×0.6 | 기본 차체 명령의 속도를 낮춤 |
| 경사스케줄 | IMU 피치로 속도 상한 조정 |
| RL 3-D | 차체속도 `(Δvx, Δvy, Δω)` 잔차 |

`eval.py`는 성능·고착 원인·경로오차·잔차 사용량을 출력한다. 정책 대 순수 IK의
직접 판정은 `compare_policy.py`의 짝비교와 95% 신뢰구간을 사용한다.

---

## 5. 보기

### MuJoCo 뷰어

창이 떠야 하므로 그래픽 세션이 있는 터미널에서 실행한다.

```bash
cd ~/ksrc/sim/rl
python3 view.py
```

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--terrain` | `sand` | `sand` / `rock` / `proc`(절차생성) |
| `--d` | 1.0 | 난이도 |
| `--model` / `--vecnorm` | 없음 | 없으면 순수 IK |
| `--watch-run` | 없음 | 학습 run의 최신 체크포인트와 정규화 통계를 에피소드마다 다시 읽음 (`--model`과 동시 사용 불가) |
| `--seed` | 0 | 바꾸면 다른 맵 |
| `--static` | off | 물리 끄고 지형만 (제일 가벼움) |
| `--realtime` | 1.0 | 재생 배속. 0.3 이면 슬립 관찰용 |

마우스: 좌드래그 회전 / 우드래그 이동 / 휠 줌. `p` 일시정지, `r` 새 경로,
`o` 정책 보정 ON/OFF.
에피소드가 끝나면 결과를 찍고 자동 리셋한다.

학습을 보면서 실행하려면 터미널 둘을 쓴다. 첫 20만 스텝 체크포인트 전에는
뷰어가 순수 IK를 보여주고, 이후 에피소드 경계에서 새 정책을 자동으로 적용한다.

```bash
# 터미널 1: 학습
cd ~/ksrc/sim/rl
python3 train.py --steps 5000000 --envs 16 --terrain sand --stage1 \
  --out runs/s9_watch

# 터미널 2: 별도 MuJoCo 환경에서 최신 정책 재생
cd ~/ksrc/sim/rl
python3 view.py --terrain sand --stage1 --d 0 --watch-run runs/s9_watch
```

화면은 학습 워커의 실제 내부 상태가 아니라 같은 경로 세트에서 최신 정책을
따로 실행한 것이다. `--path-seed 4321`을 터미널 2에 붙이면 학습 경로와 다른
미학습 경로를 볼 수 있다. 뷰어도 CPU를 사용하므로 학습 속도는 줄어들 수 있다.

### 텐서보드

```bash
cd ~/ksrc/sim/rl
tensorboard --logdir runs/teacher/tb
# http://localhost:6006
```

여러 실행을 한 번에 비교하려면 상위 폴더를 준다:

```bash
cd ~/ksrc/sim/rl
tensorboard --logdir runs
```

올라가는 스칼라:

| 그룹 | 내용 |
|---|---|
| `rollout/` | success, stuck, tip, oob, slip, sink, goals, dist, frac, f_belly, f_lifted, duty_sat, duty_use, steer_use |
| `stuck_cause/` | belly, lost_load, blocked, slip_stuck (멈춘 원인 비율) |
| `curriculum/` | difficulty, success_rate |
| `train/` | SB3 기본 (loss, entropy, explained_variance, approx_kl …) |

### 텐서보드 없이 돌린 학습

stdout 로그를 사후에 그림으로 만든다.

```bash
python3 plotlog.py train.log -o trainlog.png
```

### 모델 2D 단면

```bash
python3 draw.py
```

---

## 전형적인 흐름

```bash
bash ../build.sh                                    # 기구학 C 빌드
python3 check_model.py                              # 물리 9종 통과 확인

python3 train.py --teacher --envs 16 --out runs/teacher      # ~50분
python3 probe_teacher.py --model runs/teacher/final \
        --vecnorm runs/teacher/vecnorm.pkl          # z 를 쓰는가? 아니면 여기서 멈춤

python3 distill.py --teacher runs/teacher/final \
        --vecnorm runs/teacher/vecnorm.pkl          # student 증류
python3 train.py --envs 16 --out runs/blind                  # 대조군 ~50분

python3 eval.py --arena --terrain sand --episodes 30 \
        --model runs/teacher/final --vecnorm runs/teacher/vecnorm.pkl
```

---

## 주의

- **`pkill -f` 로 학습을 죽이지 말 것.** 패턴이 자기 셸까지 잡아 세션이 같이 죽는다
  (이 프로젝트에서 네 번 발생). 실행할 때 PID 를 받아두고 그 PID 로 `kill` 한다.
- `--envs` 는 물리 코어 수를 넘기지 않는다. 16 코어에서 16 이 최대다.
- 학습 중에는 CPU 가 포화되므로 뷰어·평가를 같이 돌리면 둘 다 느려진다.
- 기하(`config.py`)를 바꾸면 이전 정책은 못 쓴다. teacher 부터 다시 돌려야 한다.
