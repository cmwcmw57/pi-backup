#!/bin/bash
# 본학습: 한 teacher 에서 student 3종 증류 → 평가. 순차, 절대 병렬 없음.
#   MLP history 5 (0.1s) / MLP history 25 (0.5s) / GRU (history 1)
# 사용: bash scripts/chain_student.sh <Direct8|Res8|Residual3> <teacher run dir> [teacher ckpt=model_1499.pt] [DR_STAGE=1]
# 서버: cd ~/gyusa/ksrc_rover && setsid nohup bash scripts/chain_student.sh Res8 2026-10-12_10-00-00 </dev/null > logs/student.master.log 2>&1 & disown
# student 는 teacher 폴더(logs/rsl_rl/ksrc_rover_teacher_<mode>/)에 run_name student_mlp/student_gru 로 쌓인다.
cd ~/gyusa/ksrc_rover
MODE=${1:?mode}; TRUN=${2:?teacher run}; TCK=${3:-model_1499.pt}; STAGE=${4:-1}
ITERS=1000
E=/workspace/logs/eval
LOW=$(echo "$MODE" | tr 'A-Z' 'a-z')
TDIR=logs/rsl_rl/ksrc_rover_teacher_$LOW
n_train() { docker ps --filter ancestor=ksrc-rover:lab2.3.2-rsl312 --format '{{.Names}}' | grep -vc ksrc_tb; }
wait_gpu1() {
  while :; do
    used=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    [[ "$used" =~ ^[0-9]+$ ]] && [ "$(n_train)" -eq 0 ] && [ "$used" -lt 1000 ] && break
    sleep 60
  done
  echo "[gpu] GPU1 ${used}MiB, 학습 컨테이너 0 — $(date +%F_%T)"
}
[ -f "$TDIR/$TRUN/$TCK" ] || { echo "teacher 체크포인트 없음: $TDIR/$TRUN/$TCK"; exit 1; }
# 이름  아키텍처  STUDENT_HISTORY
VARIANTS="mlp_h5:MLP:5 mlp_h25:MLP:25 gru:GRU:1"
declare -A RUN
for V in $VARIANTS; do
  IFS=: read NAME ARCH H <<< "$V"
  wait_gpu1
  echo "=== DISTILL $MODE $NAME start $(date +%F_%T) ==="
  bash run.sh /workspace/scripts/train.py --task Isaac-KSRC-Rover-Student-$ARCH-$MODE-v0 --headless \
    --num_envs 4096 --max_iterations $ITERS --seed 42 --run_name student_$NAME \
    --load_run "$TRUN" --checkpoint "$TCK" --set STUDENT_HISTORY=$H --set DR_STAGE=$STAGE \
    > logs/st_${LOW}_$NAME.log 2>&1
  RUN[$NAME]=$(basename "$(ls -dt $TDIR/*_student_$NAME/ | head -1)")
  echo "=== DISTILL $MODE $NAME done  $(date +%F_%T) run=${RUN[$NAME]} last: $(grep -a 'Learning iteration' logs/st_${LOW}_$NAME.log | tail -1 | tr -s ' ')"
done
wait_gpu1
echo "=== EVAL teacher $(date +%T) ==="
bash run.sh /workspace/scripts/eval_log.py --task Isaac-KSRC-Rover-Teacher-$MODE-v0 \
  --checkpoint /workspace/$TDIR/$TRUN/$TCK --num_envs 64 --steps 1500 --wheel-seed 777 \
  --set DR_STAGE=$STAGE --out $E/st_${LOW}_teacher.npz --headless > logs/st_${LOW}_eval_teacher.log 2>&1
for V in $VARIANTS; do
  IFS=: read NAME ARCH H <<< "$V"
  wait_gpu1
  CK=/workspace/$TDIR/${RUN[$NAME]}/model_$((ITERS-1)).pt
  echo "=== EVAL $NAME ckpt=$CK $(date +%T) ==="
  bash run.sh /workspace/scripts/eval_log.py --task Isaac-KSRC-Rover-Student-$ARCH-$MODE-v0 --checkpoint $CK \
    --num_envs 64 --steps 1500 --wheel-seed 777 --set STUDENT_HISTORY=$H --set DR_STAGE=$STAGE \
    --out $E/st_${LOW}_$NAME.npz --headless > logs/st_${LOW}_eval_$NAME.log 2>&1
done
wait_gpu1
echo "===== $MODE teacher vs student 통합표 ====="
bash run.sh /workspace/scripts/compare_all.py $E Teacher=st_${LOW}_teacher MLP5=st_${LOW}_mlp_h5 MLP25=st_${LOW}_mlp_h25 GRU=st_${LOW}_gru 2>&1 | grep -vE "^\[|Warning|warn"
echo "STUDENT_ALL_DONE $(date +%F_%T)"
