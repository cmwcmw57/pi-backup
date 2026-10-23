# RL 주행보조 — 본학습 계획 (동료 전달용)

작성 2026-10-10. 액션공간 스크리닝 끝나고 **본학습(최종모델)** 들어가기 위한 계획.

> ⚠ **2026-10-11 환경 버그 2건 발견·수정 — 아래 §1 스크리닝 수치는 모두 버그 환경 결과라 무효.**
> 1. 생성 높이 0.15 m < 바퀴 바닥 깊이 0.1618 m → 리셋마다 지형에 박혀 튕김, 에피소드 ~50% 1초 내 전복. → `SPAWN_Z=0.21`.
> 2. 바퀴 drive armature 누락(MuJoCo엔 0.0015) → 명시적 모터 루프 수치불안정, μ≲0.26 바퀴 헛돎 갇힘. → `rover_cfg.py` `armature=0.0015`.
> 수정 후 순수 IK 기준(같은 env·홀드아웃 μ): error_vel xy 0.108→**0.013**, yaw 0.304→**0.033**, 슬립불균형 0.62→**0.054**, 전복 56%→**0%**.
> 즉 현재 env의 좌우 μ 비대칭은 IK도 거의 영향 없음 → **RL 보정 여지를 만들 env 재설계가 먼저** (진단 도구 `scripts/wheel_diag.py`).
실행 방법은 `CLI.md`, 서버 규칙은 `SERVER_RULES.md` 참조.

> **2026-10-11 결정: teacher-student 구조로 간다.** 아래 §1–§3 의 단일단계 계획은 참고용. 코드는 짜 두었고 **아직 한 번도 실행 안 함**(서버 GPU 드라이버 불일치, 재부팅 대기).
>
> | 단계 | 무엇 | 태스크 / 스크립트 | 설정 |
> |---|---|---|---|
> | T1 | teacher 3종 비교 (액션 결정) | `Isaac-KSRC-Rover-Teacher-{Direct8,Res8,Residual3}-v0`, `scripts/chain_teacher_s1.sh` | `DR_STAGE=1` |
> | T2 | 이긴 teacher 를 DR 2단계로 이어 학습 | 같은 태스크 + `--resume --load_run <run> --checkpoint model_1499.pt --set DR_STAGE=2` | `DR_STAGE=2` |
> | S | student 3종 증류·비교 (MLP h5 / MLP h25 / GRU) | `Isaac-KSRC-Rover-Student-{MLP,GRU}-<mode>-v0`, `scripts/chain_student.sh <mode> <teacher run> [ckpt] [stage]` | `STUDENT_HISTORY` |
> | E | 배포 | `scripts/export_student.py` → TorchScript + 입력 배치 JSON | |
> | X | **sim2sim** (MuJoCo, 노트북 CPU) | `sim/rl/sim2sim_isaac.py`, `scripts/chain_sim2sim.sh [episodes]` → `logs/sim2sim/` | 시나리오 flat_mu / flat_sand(terramech) / rough_mu / rough_sand |
>
> sim2sim 은 런의 `params/env.yaml`·`agent.yaml` 에서 관측 그룹·항목 순서·history·액션·명령 주기를 읽어 재현한다(단일단계·teacher·student MLP/GRU 모두, 합성 체크포인트로 확인). 사용법은 `CLI.md` §10. 같은 seed 면 IK·정책이 같은 명령·μ·지형 열을 본다(에피소드가 전복으로 일찍 끝나면 이후 열이 어긋날 수 있음).
>
> - 관측 그룹: `policy`(실차 센서 29×2, teacher 입력), `privileged`(22: 실속도3·바퀴μ4·슬립4·질량/CoM4·DR값7, teacher 전용), `student`(실차 센서 29×`STUDENT_HISTORY`).
> - teacher = MLP, actor·critic 둘 다 `policy+privileged`. student = 실차 센서만, rsl-rl `Distillation`(student 가 몰고 teacher 가 정답 라벨, MSE).
> - DR 1단계(`mdp/events.py`): μ patch 모델, 질량 0.95–1.2 + CoM, 명령 2–6s·에피소드 16s, wide 지형(0–20°, 30° 제외: Nav2 경사상한 21°).
> - DR 2단계(`mdp/dr.py`, CONFIG `DR2`/`NOISE`): 배터리 전압·모터 개체차·개루프 기울기·바퀴 armature, 서보 최고속도·강성, 행동지연 0–1·관측지연 1–2 스텝(펌웨어 50Hz·직전주기 텔레메트리 근거), 자이로 바이어스, 센서 가우시안 노이즈. `privileged` 크기가 1·2단계 같아서 T1 → T2 resume 가능.
> - CONFIG 상수는 `scripts/train.py`/`eval_log.py` 의 `--set KEY=VALUE` 로 실행마다 덮어쓸 수 있다.
> - 주의: 명령 재샘플(2–6s) < 에피소드(16s)라 TensorBoard `error_vel_*` 는 m/s 평균이 아님(≈×2.7). 판정은 `eval_log`+`eval_analyze`.

---

## 0. 배포 맥락 (왜 이렇게 고르는가)
실제 파이프라인: **Nav2(위치계획) → cmd_vel → [RL 속도추종] → 바퀴 → 카메라 → SLAM(위치) → Nav2**.
- **Nav2+SLAM이 위치 루프를 닫는다** → RL의 **속도추종 정밀도는 덜 중요**(Nav2가 오차 보정).
- 실지형 = **모래 ~5cm, 빠질 수 있음** → **슬립/견인(안 빠지기)이 하드 실패모드**.
- SLAM은 **매끄러운 카메라 모션** 선호(단 로버가 느려 ~0.16 m/s라 영향 작을 것). 슬립은 SLAM 위치추정을 직접 망치진 않음(카메라는 실제모션을 봄), 단 **물리적으로 빠지면 끝**.

## 1. 결정: 액션공간 = **Direct8 (주력)**, Res3는 조건부 폴백
### 스크리닝 결과 (1500iter, 보상B, 비대칭μ, vs 순수 IK 베이스라인)
```
policy      xy오차  yaw오차  슬립   잔차  r~dmu  오버슈트(v/w)  vsIK(xy/yaw/slip%)
IK기준     0.108  0.304  0.619  0.00  +0.00  1.15/1.13   —
Res3       0.086  0.270  0.634  0.61  -0.08  1.20/1.25   +19.9/+11.1/-2.3
Direct8    0.101  0.226  0.548  2.86  +0.00  1.12/1.22   +6.4/+25.8/+11.5
Res8_old   0.104  0.320  0.621  3.81  -0.00  1.19/1.26   +3.7/-5.3/-0.3
Res8_new   0.095  0.297  0.624  4.35  +0.05  1.15/1.25   +12.3/+2.4/-0.8
Res8_slip  0.099  0.305  0.610  4.05  -0.06  1.13/1.29   +8.3/-0.4/+1.5
```
### 근거
- **Direct8만 좌우 슬립불균형을 줄인다(+11.5%)** + yaw 보정 최고(+26%). = **모래/비대칭 견인에 유일하게 유효.**
- **Res3는 추종 1위지만** 그 강점(속도정밀)은 **Nav2가 메워 희석**되고, **슬립을 방치**(−2.3%) → 모래 전이 리스크.
- **Res8은 3번 시도(스케일↓, 슬립보상) 모두 실패** — IK 바닥+제한 잔차 구조가 슬립 균형을 방해(계속 railing, 슬립 거의 못 줄임). "per-wheel+IK안전" 절충은 **구조적으로 안 됨.**
- Direct8 약점은 **엔지니어링으로 보완 가능**(아래 §4), Res3 약점(박힘)은 **구조적으로 불가**.

### ⚠ 결정 게이트 (Direct8 → Res3로 바꿀 조건)
실모래/렌더 테스트에서 **(a) Direct8 모션에 SLAM(ORBSLAM3)이 깨지고 매끄러움 튜닝으로도 안 잡히며, 동시에 (b) 실제 모래가 생각보다 단단해 박힘이 거의 없다** → 그땐 **Res3**로 전환.

## 2. 보상 (CONFIG 상수)
- **보상 B 기반**: `STD_V=0.05`, `STD_OM=0.20`, `LINEAR=False` (exp-only, 선형항 불필요 — 검증됨).
- **`SLIP_W` 슬립 페널티**: 모래 견인용. 시작 0.5 → **슬립이 충분히 안 떨어지면 1.0~2.0까지 상향**(단 추종 붕괴 감시).
- **`action_rate` 가중 상향**: SLAM용 매끄러움 ↑ (현재 −0.01 → −0.03~−0.05 실험). Direct8 jerk 억제.
- 판정은 reward 아니라 **`error_vel`/슬립** (`eval_analyze`/`compare_all`).

## 3. 환경 커리큘럼 (점진 현실화 — env가 액션보다 큰 레버)
현재 env는 **단순화**(에피소드 고정 + 좌우페어 비대칭 + 비공간적). 아래로 점진 확장:
1. **기준**: 현재 env (`rough_slope`/`narrow` + `WHEEL_MU=wide`) 재현.
2. **바퀴 독립 μ**: `mdp/events.py`의 좌/우 페어 → **4바퀴 독립** 할당. (지금은 fl,rl 같은값 / fr,rr 같은값)
3. **시간가변/패치 μ**: `mode="reset"` → **에피소드 중 주기 재샘플**(순간 패치 proxy). 가능하면 공간적 μ(위치의존, 동료 지형 패키지 지원여부 확인).
4. **미배선 DR 켜기**: `sample_episode`가 계산하는 **질량·모터응답·토크한계·액션/관측 지연** + **IMU 노이즈/바이어스**(현재 None) → `events.py`에 배선.
5. **경사 확대**: `TERRAIN_MODE` narrow(0–10°) → **wide(0–20°)**.
→ 각 단계는 이전 체크포인트에서 이어가되(커리큘럼), **env 바뀌면 action/obs 정의가 안 바뀌는 한 resume 가능**. 슬립/추종 지표로 단계별 검증.

## 4. 실차 보완 (엔지니어링, 학습 밖)
- **안전 래퍼**: Direct8 출력에 **조향 rate·속도 한계 클램프** → IK 안전망 proxy(생 명령 폭주 방지).
- **매끄러움**: action_rate 튜닝(§2)로 SLAM 보호.
- **ik_validate**: (Res3 폴백 쓰게 되면) `ik_torch` ↔ 펌웨어 C IK 패리티 확인 필요. Direct8은 IK 안 쓰므로 해당 없음.

## 5. 검증 게이트
- **시뮬**: `compare_all.py`로 vs IK 베이스라인 — error_vel↓, 슬립↓, 오버슈트 적정, (가능하면 잔차↔|Δμ| 양상관).
- **실모래 (필수)**: 시뮬이 **침하를 모델 안 함** → 슬립→박힘은 **실모래라야 검증**. 안 빠지고 완주하나.
- **SLAM 품질**: Direct8 모션에서 **ORBSLAM3 포즈가 유지**되나(특징추적·드리프트). 실차 또는 렌더 필요.

## 6. 미해결 / 주의
- **Isaac RTX 렌더 서버 고장**(호스트 드라이버 이슈, 모든 isaac 이미지 세그폴트) → 영상/WebRTC/SLAM용 렌더 불가. **드라이버 롤백/수정 필요**(별도). 학습·eval(헤드리스)은 정상.
- **침하 시뮬 없음**(강체 heightfield + PhysX μ) → 진짜 모래 견인은 terramech/입자 시뮬 or 실차. 현재 슬립은 **대리지표**.
- **좌우페어 비대칭 = 단순화** — 실모래의 공간/전이/바퀴독립성 미반영(§3-2,3에서 보완).

## 7. 요약 한 줄
**Direct8 + (보상B + 슬립페널티 + action_rate 매끄러움) + env 점진 현실화 커리큘럼**, **안전래퍼**로 실차 보완, **실모래+SLAM 테스트로 최종 검증**. SLAM이 Direct8 모션에 깨지고 모래가 단단하면 Res3로 전환.
