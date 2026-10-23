#!/bin/bash
# 본학습 1단계: teacher 3종(Direct8 → Res8 → Residual3) 비교. 각 1500iter·4096env·seed42, 절대 병렬 없음.
# 환경 = CONFIG 현재값(WHEEL_MU patch, 질량 0.95–1.2 + CoM, 명령 2–6s·에피소드 16s, wide 지형).
# 학습 3개 끝난 뒤 eval(홀드아웃 wheel-seed 777, 노이즈 없음) + IK 기준 + 통합표.
# 서버: cd ~/gyusa/ksrc_rover && setsid nohup bash scripts/chain_teacher_s1.sh </dev/null > logs/teacher_s1.master.log 2>&1 & disown
cd ~/gyusa/ksrc_rover
ITERS=1500
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
echo "CONFIG:"; grep -E "^(TERRAIN_|WHEEL_MU|MU_|MASS_|COM_|CMD_|EPISODE_|SPAWN_Z|SLIP_W|RATE_|JERK_|YAW_ACC_W)" source/ksrc_rover/rover_env_cfg.py
declare -A DIR=([Direct8]=teacher_direct8 [Res8]=teacher_res8 [Residual3]=teacher_residual3)
declare -A RUN
for T in Direct8 Res8 Residual3; do
  wait_gpu1
  echo "=== TRAIN Teacher-$T start $(date +%F_%T) ==="
  bash run.sh /workspace/scripts/train.py --task Isaac-KSRC-Rover-Teacher-$T-v0 --headless \
    --num_envs 4096 --max_iterations $ITERS --seed 42 > logs/t1_$T.log 2>&1
  RUN[$T]=$(basename "$(ls -dt logs/rsl_rl/ksrc_rover_${DIR[$T]}/*/ | head -1)")
  echo "=== TRAIN Teacher-$T done  $(date +%F_%T) run=${RUN[$T]} last: $(grep -a 'Learning iteration' logs/t1_$T.log | tail -1 | tr -s ' ')"
done
wait_gpu1
echo "=== EVAL IK baseline $(date +%T) ==="
bash run.sh /workspace/scripts/eval_log.py --task Isaac-KSRC-Rover-Res8-v0 --num_envs 64 --steps 1500 \
  --wheel-seed 777 --out $E/t1_ik.npz --headless > logs/t1_eval_ik.log 2>&1
for T in Direct8 Res8 Residual3; do
  wait_gpu1
  CK=/workspace/logs/rsl_rl/ksrc_rover_${DIR[$T]}/${RUN[$T]}/model_$((ITERS-1)).pt
  echo "=== EVAL Teacher-$T ckpt=$CK $(date +%T) ==="
  bash run.sh /workspace/scripts/eval_log.py --task Isaac-KSRC-Rover-Teacher-$T-v0 --checkpoint $CK --num_envs 64 \
    --steps 1500 --wheel-seed 777 --out $E/t1_${DIR[$T]}.npz --headless > logs/t1_eval_$T.log 2>&1
done
wait_gpu1
echo "===== Teacher 1단계 통합표 ====="
bash run.sh /workspace/scripts/compare_all.py $E IK=t1_ik Direct8=t1_teacher_direct8 Res8=t1_teacher_res8 Res3=t1_teacher_residual3 2>&1 | grep -vE "^\[|Warning|warn"
for T in teacher_direct8 teacher_res8 teacher_residual3; do
  echo "===== IK vs $T ====="
  bash run.sh /workspace/scripts/eval_analyze.py $E/t1_ik.npz $E/t1_$T.npz 2>&1 | sed -n '/짝비교/,$p'
done
echo "TEACHER_S1_ALL_DONE $(date +%F_%T)"
