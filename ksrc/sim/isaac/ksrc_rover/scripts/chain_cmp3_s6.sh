#!/bin/bash
# S6 환경 액션공간 재비교: Direct8 → Res8 → Residual3, 각 1500iter·4096env·seed42, 절대 병렬 없음.
# run.sh 는 포그라운드라 학습이 끝나야 다음 줄로 넘어간다. 매 실행 전 GPU1 비었는지 확인.
# 학습 3개 모두 끝난 뒤에 eval(홀드아웃 wheel-seed 777) + IK 기준 + 통합표.
# 서버: cd ~/gyusa/ksrc_rover && setsid nohup bash scripts/chain_cmp3_s6.sh </dev/null > logs/cmp3_s6.master.log 2>&1 & disown
cd ~/gyusa/ksrc_rover
ITERS=1500
E=/workspace/logs/eval
n_train() { docker ps --filter ancestor=ksrc-rover:lab2.3.2-rsl312 --format '{{.Names}}' | grep -vc ksrc_tb; }
wait_gpu1() {
  while :; do
    used=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$(n_train)" -eq 0 ] && [ "$used" -lt 1000 ] && break
    sleep 60
  done
  echo "[gpu] GPU1 ${used}MiB, 학습 컨테이너 0 — $(date +%F_%T)"
}
echo "CONFIG:"; grep -E "^(TERRAIN_|WHEEL_MU|MU_|MASS_|SPAWN_Z|SLIP_W|RATE_|JERK_|YAW_ACC_W)" source/ksrc_rover/rover_env_cfg.py
declare -A DIR=([Direct8]=direct8 [Res8]=res8 [Residual3]=residual3)
declare -A RUN
for T in Direct8 Res8 Residual3; do
  wait_gpu1
  echo "=== TRAIN $T start $(date +%F_%T) ==="
  bash run.sh /workspace/scripts/train.py --task Isaac-KSRC-Rover-$T-v0 --headless \
    --num_envs 4096 --max_iterations $ITERS --seed 42 > logs/s6_$T.log 2>&1
  RUN[$T]=$(basename "$(ls -dt logs/rsl_rl/ksrc_rover_${DIR[$T]}/*/ | head -1)")
  echo "=== TRAIN $T done  $(date +%F_%T) run=${RUN[$T]} last: $(grep -a 'Learning iteration' logs/s6_$T.log | tail -1 | tr -s ' ')"
done
wait_gpu1
echo "=== EVAL IK baseline $(date +%T) ==="
bash run.sh /workspace/scripts/eval_log.py --task Isaac-KSRC-Rover-Res8-v0 --num_envs 64 --steps 1500 \
  --wheel-seed 777 --out $E/s6_ik.npz --headless > logs/s6_eval_ik.log 2>&1
for T in Direct8 Res8 Residual3; do
  wait_gpu1
  CK=/workspace/logs/rsl_rl/ksrc_rover_${DIR[$T]}/${RUN[$T]}/model_$((ITERS-1)).pt
  echo "=== EVAL $T ckpt=$CK $(date +%T) ==="
  bash run.sh /workspace/scripts/eval_log.py --task Isaac-KSRC-Rover-$T-v0 --checkpoint $CK --num_envs 64 \
    --steps 1500 --wheel-seed 777 --out $E/s6_${DIR[$T]}.npz --headless > logs/s6_eval_$T.log 2>&1
done
wait_gpu1
echo "===== S6 통합표 ====="
bash run.sh /workspace/scripts/compare_all.py $E IK=s6_ik Direct8=s6_direct8 Res8=s6_res8 Res3=s6_residual3 2>&1 | grep -vE "^\[|Warning|warn"
for T in direct8 res8 residual3; do
  echo "===== IK vs $T ====="
  bash run.sh /workspace/scripts/eval_analyze.py $E/s6_ik.npz $E/s6_$T.npz 2>&1 | sed -n '/짝비교/,$p'
done
echo "CMP3_S6_ALL_DONE $(date +%F_%T)"
