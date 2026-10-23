#!/bin/bash
# Sim2sim 체인 (노트북, CPU만 씀 — 서버 GPU 와 무관): Isaac 정책을 MuJoCo 에서 돌려 비교.
#   1) 서버에서 체크포인트+params 가져오기 (VPN 필요, 이미 있으면 건너뜀)
#   2) 모델 × 시나리오를 하나씩 순서대로 실행 → npz (eval_log.py 와 같은 형식)
#   3) 시나리오마다 compare_all 통합표 + IK 대비 짝비교
# 사용: bash sim/isaac/ksrc_rover/scripts/chain_sim2sim.sh [episodes=32]   (저장소 루트에서)
# 결과: sim/isaac/ksrc_rover/logs/sim2sim/<시나리오>_<모델>.npz, 표는 표준출력(로그로 남기려면 | tee)
set -u
cd "$(dirname "$0")/../../../.."           # 저장소 루트
EP=${1:-32}
O=sim/isaac/ksrc_rover/logs/sim2sim
C=$O/ckpt
PY=python3
RUN=sim/rl/sim2sim_isaac.py

# 모델 목록: 이름|서버 실험폴더|런|체크포인트|추가 인자(쉼표 구분, 없으면 비움)   (IK 기준은 자동 포함)
# 관측 구성·액션 종류·명령 주기·μ 방식은 런의 params/*.yaml 에서 자동으로 읽는다 (teacher/student/GRU 모두).
# 추가 인자 예: DR 2단계 모델이면  --obs-delay,1,--act-delay,1,--noise
MODELS="
s6_direct8|ksrc_rover_direct8|2026-10-11_01-27-51|model_1499.pt|
s6_res8|ksrc_rover_res8|2026-10-11_05-32-31|model_1499.pt|
"
# 시나리오: 이름|MuJoCo 지형|μ 샘플러|μ 재추첨 간격 최소,최대 s  (공백 금지: for 가 줄을 쪼갬)
#   flat_mu    평지 + Isaac S6 와 같은 μ(바퀴별 0.1–0.6, 1s)  → 접촉·액추에이터 차이만
#   flat_sand  평지 + terramech 규사(슬립-침하, 견인 피크)     → 학습에 없던 흙 모델
#   rough_mu   계단식 램프 지형 + S6 μ                         → 지형 + 접촉 차이
#   rough_sand 램프 지형 + 규사                                 → 가장 실제에 가까움
SCENARIOS="
flat_mu|flat_mu|s6|1,1
flat_sand|flat_sand|patch|1,5
rough_mu|rough_mu|s6|1,1
rough_sand|rough_sand|patch|1,5
"

echo "=== 1) 체크포인트 $(date +%F_%T) ==="
for L in $MODELS; do
  IFS='|' read NAME EXP RUNDIR CK XARGS <<< "$L"
  D=$C/$NAME
  if [ -f "$D/$CK" ]; then echo "  $NAME 있음"; continue; fi
  mkdir -p "$D"
  ssh -o ConnectTimeout=15 hpi-lab "cd ~/gyusa/ksrc_rover/logs/rsl_rl/$EXP/$RUNDIR && tar cf - $CK params" \
    | tar -C "$D" -xf - && echo "  $NAME 받음" || { echo "  $NAME 받기 실패(VPN?) — 건너뜀"; rm -rf "$D"; }
done

echo "=== 2) 실행 (episodes=$EP, seed 777 공통 → 같은 명령·μ·지형 열) ==="
for S in $SCENARIOS; do
  IFS='|' read SN TERR MU RS <<< "$S"
  $PY $RUN --action Res8 --policy ik --scenario $TERR --mu $MU --mu-resample-s ${RS/,/ } \
    --episodes $EP --out $O/${SN}_ik.npz | tail -1
  for L in $MODELS; do
    IFS='|' read NAME EXP RUNDIR CK XARGS <<< "$L"
    [ -f "$C/$NAME/$CK" ] || continue
    $PY $RUN --policy $C/$NAME/$CK ${XARGS//,/ } --scenario $TERR --mu $MU --mu-resample-s ${RS/,/ } \
      --episodes $EP --out $O/${SN}_${NAME}.npz | tail -1
  done
done

echo "=== 3) 표 ==="
for S in $SCENARIOS; do
  IFS='|' read SN TERR MU RS <<< "$S"
  ARGS="IK=${SN}_ik"
  for L in $MODELS; do
    IFS='|' read NAME _ <<< "$L"
    [ -f $O/${SN}_${NAME}.npz ] && ARGS="$ARGS $NAME=${SN}_${NAME}"
  done
  echo; echo "##### $SN  (지형 $TERR, μ $MU) #####"
  $PY sim/isaac/ksrc_rover/scripts/compare_all.py $O $ARGS
done
echo "SIM2SIM_ALL_DONE $(date +%F_%T)"
