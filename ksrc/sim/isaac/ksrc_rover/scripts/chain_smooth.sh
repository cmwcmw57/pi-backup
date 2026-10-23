#!/bin/bash
# 기준(d8_base) 학습 종료 대기 → 기준 eval → 매끄러움 학습(d8_smooth) → eval → 통합표.
# 서버에서: cd ~/gyusa/ksrc_rover && setsid nohup bash scripts/chain_smooth.sh </dev/null > logs/chain_smooth.master.log 2>&1 & disown
# 매끄러움 가중치는 실행 시점의 rover_env_cfg.py CONFIG(RATE_*/JERK_*/YAW_ACC_W)를 따른다.
cd ~/gyusa/ksrc_rover
BASE_TS=2026-10-10_11-19-22          # d8_base(4096env) 런dir (UTC)
ITERS=4000
E=/workspace/logs/eval

n_train() { docker ps --filter ancestor=ksrc-rover:lab2.3.2-rsl312 --format '{{.Names}}' | grep -vc ksrc_tb; }
# 실행 전 항상 GPU 확인: 학습 컨테이너 0개 + GPU1 사용 < 1000 MiB 일 때까지 대기.
wait_gpu1() {
  while :; do
    used=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$(n_train)" -eq 0 ] && [ "$used" -lt 1000 ] && break
    sleep 60
  done
  echo "[gpu] GPU1 ${used}MiB, 학습 컨테이너 0 — 실행 $(date +%F_%T)"
}

wait_gpu1
CK=/workspace/logs/rsl_rl/ksrc_rover_direct8/$BASE_TS/model_$((ITERS-1)).pt
echo "=== EVAL d8_base ckpt=$CK $(date +%T) ==="
bash run.sh /workspace/scripts/eval_log.py --task Isaac-KSRC-Rover-Direct8-v0 --checkpoint $CK \
  --num_envs 64 --steps 1500 --wheel-seed 777 --out $E/d8_base.npz --headless

wait_gpu1
echo "=== TRAIN d8_smooth $(date +%T) ==="
grep -E "^(SLIP_W|RATE_|JERK_|YAW_ACC_W)" source/ksrc_rover/rover_env_cfg.py
bash run.sh /workspace/scripts/train.py --task Isaac-KSRC-Rover-Direct8-v0 --headless \
  --num_envs 4096 --max_iterations $ITERS --seed 42 > logs/d8_smooth.log 2>&1
echo "=== TRAIN DONE $(date +%T) ==="

wait_gpu1
RD=$(ls -dt logs/rsl_rl/ksrc_rover_direct8/*/ | head -1); TS=$(basename "$RD")
CK=/workspace/logs/rsl_rl/ksrc_rover_direct8/$TS/model_$((ITERS-1)).pt
echo "=== EVAL d8_smooth ckpt=$CK $(date +%T) ==="
bash run.sh /workspace/scripts/eval_log.py --task Isaac-KSRC-Rover-Direct8-v0 --checkpoint $CK \
  --num_envs 64 --steps 1500 --wheel-seed 777 --out $E/d8_smooth.npz --headless

wait_gpu1
echo "===== 통합표 ====="
bash run.sh /workspace/scripts/compare_all.py 2>&1 | grep -vE "^\[|Warning|warn"
echo "CHAIN_SMOOTH_DONE $(date +%F_%T)"
