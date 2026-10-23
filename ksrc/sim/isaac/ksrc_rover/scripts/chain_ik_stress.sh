#!/bin/bash
# IK 스트레스 스윕: 어떤 조건에서 순수 IK(Res8 zero-action)가 무너지나. 매 실행 전 GPU1 확인.
# 서버: cd ~/gyusa/ksrc_rover && setsid nohup bash scripts/chain_ik_stress.sh </dev/null > logs/ik_stress.master.log 2>&1 & disown
cd ~/gyusa/ksrc_rover
E=/workspace/logs/eval
n_train() { docker ps --filter ancestor=ksrc-rover:lab2.3.2-rsl312 --format '{{.Names}}' | grep -vc ksrc_tb; }
wait_gpu1() {
  while :; do
    used=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$(n_train)" -eq 0 ] && [ "$used" -lt 1000 ] && break
    sleep 30
  done
}
run() {  # tag, 나머지는 --set 인자들
  local tag=$1; shift
  wait_gpu1
  echo "=== $tag $(date +%T) $* ==="
  bash run.sh /workspace/scripts/eval_log.py --task Isaac-KSRC-Rover-Res8-v0 --num_envs 64 --steps 1500 \
    --wheel-seed 777 --out $E/ik_$tag.npz --headless "$@" > logs/ik_stress_$tag.log 2>&1
  grep -aE "EVAL_SAVED|Traceback|Error" logs/ik_stress_$tag.log | head -3
  bash run.sh /workspace/scripts/eval_analyze.py $E/ik_$tag.npz 2>&1 | grep -aE "error_vel|slip|매끄러움|μmin|구간" | grep -v Warning
}
C='--set WHEEL_MU="custom"'
run S1_pair02   $C --set 'MU_PER="pair"'  --set 'MU_RANGE=(0.2,0.6)'
run S2_pair01   $C --set 'MU_PER="pair"'  --set 'MU_RANGE=(0.1,0.6)'
run S3_wheel01  $C --set 'MU_PER="wheel"' --set 'MU_RANGE=(0.1,0.6)'
run S4_wheel01_rs1 $C --set 'MU_PER="wheel"' --set 'MU_RANGE=(0.1,0.6)' --set MU_RESAMPLE_S=1.0
run S8_wheel_low $C --set 'MU_PER="wheel"' --set 'MU_RANGE=(0.05,0.3)' --set MU_RESAMPLE_S=1.0
run S5_terrwide --set 'TERRAIN_MODE="wide"' --set SPAWN_Z=0.24
run S6_terrwide_wheel01_rs1 $C --set 'MU_PER="wheel"' --set 'MU_RANGE=(0.1,0.6)' --set MU_RESAMPLE_S=1.0 --set 'TERRAIN_MODE="wide"' --set SPAWN_Z=0.24
run S7_wheel01_rs1_mass $C --set 'MU_PER="wheel"' --set 'MU_RANGE=(0.1,0.6)' --set MU_RESAMPLE_S=1.0 --set 'MASS_SCALE=(0.8,1.5)'
echo "IK_STRESS_ALL_DONE $(date +%T)"
