#!/bin/bash
# 본학습 teacher 3종: Res8 1500 → Direct8 2500 → Residual3 1500 iter, 4096env·seed42, DR 1–8 전부(DR_STAGE=2).
# 하나씩 순서대로(절대 병렬 없음). 각 학습이 끝나면 바로 그 teacher 평가(홀드아웃 wheel-seed 777) — 결과가 일찍 나오게.
# 마지막에 IK 기준 + 통합표. 맵 밖 이탈은 off_map 종료(truncation).
# 서버: cd ~/gyusa/ksrc_rover && setsid nohup bash scripts/chain_teacher_main.sh </dev/null > logs/teacher_main.master.log 2>&1 & disown
cd ~/gyusa/ksrc_rover
declare -A ITER=([Res8]=1500 [Direct8]=2500 [Residual3]=1500)   # Direct8 은 IK 없이 배워 느리게 오름
E=/workspace/logs/eval
n_train() { docker ps --filter ancestor=ksrc-rover:lab2.3.2-rsl312 --format '{{.Names}}' | grep -vc ksrc_tb; }
wait_gpu1() {
  while :; do
    used=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    [[ "$used" =~ ^[0-9]+$ ]] && [ "$(n_train)" -eq 0 ] && [ "$used" -lt 1000 ] && break
    sleep 60
  done
  echo "[gpu] GPU1 ${used}MiB, 학습 컨테이너 0 — $(date +%F_%T)"
}
echo "CONFIG:"; grep -E "^(TERRAIN_|WHEEL_MU|MU_PATCH|MASS_|COM_|CMD_|EPISODE_|DR_STAGE|STUDENT_|SPAWN_Z|SLIP_W|RATE_|JERK_|YAW_ACC_W)" source/ksrc_rover/rover_env_cfg.py
grep -n "STEER_LIMIT_RAD =\|RES8_.*=" source/ksrc_rover/utils/contract.py
declare -A DIR=([Direct8]=teacher_direct8 [Res8]=teacher_res8 [Residual3]=teacher_residual3)
for T in Res8 Direct8 Residual3; do
  wait_gpu1
  echo "=== TRAIN Teacher-$T start $(date +%F_%T) ==="
  bash run.sh /workspace/scripts/train.py --task Isaac-KSRC-Rover-Teacher-$T-v0 --headless \
    --num_envs 4096 --max_iterations ${ITER[$T]} --seed 42 --run_name main_dr2 > logs/tmain_$T.log 2>&1
  RUN=$(basename "$(ls -dt logs/rsl_rl/ksrc_rover_${DIR[$T]}/*_main_dr2/ | head -1)")
  echo "=== TRAIN Teacher-$T done  $(date +%F_%T) run=$RUN last: $(grep -a 'Learning iteration' logs/tmain_$T.log | tail -1 | tr -s ' ')"
  wait_gpu1
  CK=/workspace/logs/rsl_rl/ksrc_rover_${DIR[$T]}/$RUN/model_$((ITER[$T]-1)).pt
  bash run.sh /workspace/scripts/eval_log.py --task Isaac-KSRC-Rover-Teacher-$T-v0 --checkpoint $CK --num_envs 64 \
    --steps 1500 --wheel-seed 777 --out $E/tmain_${DIR[$T]}.npz --headless > logs/tmain_eval_$T.log 2>&1
  echo "=== EVAL Teacher-$T: $(grep -a EVAL_SAVED logs/tmain_eval_$T.log)"
done
wait_gpu1
bash run.sh /workspace/scripts/eval_log.py --task Isaac-KSRC-Rover-Res8-v0 --num_envs 64 --steps 1500 \
  --wheel-seed 777 --out $E/tmain_ik.npz --headless > logs/tmain_eval_ik.log 2>&1
echo "===== teacher 본학습 통합표 ====="
bash run.sh /workspace/scripts/compare_all.py $E IK=tmain_ik Res8=tmain_teacher_res8 Direct8=tmain_teacher_direct8 Res3=tmain_teacher_residual3 2>&1 | grep -vE "^\[|Warning|warn"
for T in teacher_res8 teacher_direct8 teacher_residual3; do
  echo "===== IK vs $T ====="
  bash run.sh /workspace/scripts/eval_analyze.py $E/tmain_ik.npz $E/tmain_$T.npz 2>&1 | sed -n '/짝비교/,$p'
done
echo "TEACHER_MAIN_ALL_DONE $(date +%F_%T)"
