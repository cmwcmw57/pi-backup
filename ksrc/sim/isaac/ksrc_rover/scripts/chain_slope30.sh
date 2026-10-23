#!/bin/bash
# 30° 오르막 한계 시험: 경사 고정(+x 오르막), yaw 0, IK 직진 0.15 m/s, 바퀴 μ 고정 스윕. 매 실행 전 GPU1 확인.
cd ~/gyusa/ksrc_rover
n_train() { docker ps --filter ancestor=ksrc-rover:lab2.3.2-rsl312 --format '{{.Names}}' | grep -vc ksrc_tb; }
wait_gpu1() {
  while :; do
    used=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$(n_train)" -eq 0 ] && [ "$used" -lt 1000 ] && break
    sleep 30
  done
}
run() {  # tag slope mu
  wait_gpu1
  echo "=== $1 $(date +%T) ==="
  bash run.sh -u /workspace/scripts/wheel_diag.py --headless --tag "$1" --terrain slope --mu off \
    --slope-deg "$2" --mu-fixed "$3" --spawn-z 0.30 --vx 0.15 --wz 0.0 > "logs/slope_$1.log" 2>&1
  grep -aE "SLOPE|목표ω|몸체 vx|slip|Traceback" "logs/slope_$1.log"
}
for mu in 0.4 0.5 0.55 0.6 0.65 0.7 0.8; do run s30_mu$mu 30 $mu; done
for mu in 0.3 0.4; do run s20_mu$mu 20 $mu; done
echo "SLOPE_ALL_DONE $(date +%T)"
