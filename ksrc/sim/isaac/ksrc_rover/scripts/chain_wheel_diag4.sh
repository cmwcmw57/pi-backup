#!/bin/bash
# 바퀴 요동 진단 4차: 해결책 검증 — 내장(implicit) 드라이브 / DCMotor+armature, 저μ·비대칭μ·현재지형. 매 실행 전 GPU1 확인.
cd ~/gyusa/ksrc_rover
n_train() { docker ps --filter ancestor=ksrc-rover:lab2.3.2-rsl312 --format '{{.Names}}' | grep -vc ksrc_tb; }
wait_gpu1() {
  while :; do
    used=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$(n_train)" -eq 0 ] && [ "$used" -lt 1000 ] && break
    sleep 30
  done
}
run() {
  local tag=$1; shift
  wait_gpu1
  echo "=== $tag $(date +%T) ==="
  bash run.sh -u /workspace/scripts/wheel_diag.py --headless --tag "$tag" --vx 0.15 --wz 0.0 "$@" > "logs/wheel_diag_$tag.log" 2>&1
  grep -aA6 "WHEEL_DIAG\|\[warn\]\|Traceback" "logs/wheel_diag_$tag.log" | grep -v "^--$\|env0 바퀴"
}
run L1_dc_arm0015_mu02   --terrain plane --mu off  --mu-fixed 0.2 --armature 0.0015
run L2_impl_planewide    --terrain plane --mu wide --actuator implicit
run L3_impl_rough_wide   --terrain rough_slope --mu wide --actuator implicit
run L4_dc_arm_rough_wide --terrain rough_slope --mu wide --armature 0.0015
run L5_impl_turn_wide    --terrain plane --mu wide --actuator implicit --vx 0.10 --wz 0.5
echo "WHEEL_DIAG_ALL_DONE $(date +%T)"
