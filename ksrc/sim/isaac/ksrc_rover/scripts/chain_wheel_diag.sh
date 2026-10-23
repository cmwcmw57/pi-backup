#!/bin/bash
# 바퀴 요동 진단 변형들을 순차 실행(매 실행 전 GPU1 확인). 결과 요약은 마스터 로그의 WHEEL_DIAG 블록.
# 서버: cd ~/gyusa/ksrc_rover && setsid nohup bash scripts/chain_wheel_diag.sh </dev/null > logs/wheel_diag.master.log 2>&1 & disown
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
  grep -aA6 "WHEEL_DIAG\|\[warn\]\|Traceback" "logs/wheel_diag_$1.log" | grep -v "^--$"
}
run A_plane        plane       off  0.2  0.15 0.0   # 최소 조건: 평지·μ대칭·직진 (확인됨: 요동 0)
run B_plane_turn   plane       off  0.2  0.10 0.5   # 조향 scrub
run C_plane_mu     plane       wide 0.2  0.15 0.0   # 바퀴 μ 비대칭
run D_rough        rough_slope wide 0.2  0.15 0.0   # 현재 학습 환경
run E_damp005      plane       off  0.05 0.15 0.0   # 제어기 이득 ↓
run F_damp05       plane       off  0.5  0.15 0.0   # 제어기 이득 ↑
echo "WHEEL_DIAG_ALL_DONE $(date +%T)"
