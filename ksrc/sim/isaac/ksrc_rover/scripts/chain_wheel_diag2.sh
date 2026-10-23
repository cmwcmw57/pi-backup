#!/bin/bash
# 바퀴 요동 진단 2차(μ별 상세 + 지형 분해)를 순차 실행(매 실행 전 GPU1 확인). 결과 요약은 마스터 로그의 WHEEL_DIAG 블록.
# 서버: cd ~/gyusa/ksrc_rover && setsid nohup bash scripts/chain_wheel_diag2.sh </dev/null > logs/wheel_diag2.master.log 2>&1 & disown
cd ~/gyusa/ksrc_rover
n_train() { docker ps --filter ancestor=ksrc-rover:lab2.3.2-rsl312 --format '{{.Names}}' | grep -vc ksrc_tb; }
wait_gpu1() {
  while :; do
    used=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$(n_train)" -eq 0 ] && [ "$used" -lt 1000 ] && break
    sleep 30
  done
}
run() {  # tag terrain mu damping vx wz
  wait_gpu1
  echo "=== $1 $(date +%T) ==="
  bash run.sh -u /workspace/scripts/wheel_diag.py --headless --tag "$1" --terrain "$2" --mu "$3" \
    --damping "$4" --vx "$5" --wz "$6" > "logs/wheel_diag_$1.log" 2>&1
  grep -aA30 "WHEEL_DIAG\|\[warn\]\|Traceback" "logs/wheel_diag_$1.log" | grep -v "^--$"
}
run C2_plane_mu   plane       wide 0.2 0.15 0.0   # env별 μ ↔ 헛돎
run G_slope      slope       off  0.2 0.15 0.0   # 경사만 (요철 없음)
run H_roughflat  rough_flat  off  0.2 0.15 0.0   # 요철만 (경사 없음)
run I_roughslope rough_slope off  0.2 0.15 0.0   # 현재 지형, μ 대칭
run J_slope_mu   slope       wide 0.2 0.15 0.0   # 경사 + μ 비대칭
echo "WHEEL_DIAG_ALL_DONE $(date +%T)"
