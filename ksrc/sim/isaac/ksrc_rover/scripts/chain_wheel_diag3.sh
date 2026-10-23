#!/bin/bash
# 바퀴 요동 진단 3차: 저μ(0.2) 평지 직진 "헛돎 갇힘" 원인 — 모터 모델 / 솔버 / dt 분리. 매 실행 전 GPU1 확인.
# 서버: cd ~/gyusa/ksrc_rover && setsid nohup bash scripts/chain_wheel_diag3.sh </dev/null > logs/wheel_diag3.master.log 2>&1 & disown
cd ~/gyusa/ksrc_rover
n_train() { docker ps --filter ancestor=ksrc-rover:lab2.3.2-rsl312 --format '{{.Names}}' | grep -vc ksrc_tb; }
wait_gpu1() {
  while :; do
    used=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$(n_train)" -eq 0 ] && [ "$used" -lt 1000 ] && break
    sleep 30
  done
}
run() {  # tag, 나머지는 wheel_diag.py 인자
  local tag=$1; shift
  wait_gpu1
  echo "=== $tag $(date +%T) ==="
  bash run.sh -u /workspace/scripts/wheel_diag.py --headless --tag "$tag" --terrain plane --mu off \
    --vx 0.15 --wz 0.0 "$@" > "logs/wheel_diag_$tag.log" 2>&1
  grep -aA45 "TRACE\|WHEEL_DIAG\|\[warn\]\|Traceback" "logs/wheel_diag_$tag.log" | grep -v "^--$\|env0 바퀴"
}
run K0_dc_mu02       --mu-fixed 0.2 --trace
run K7_dc_mu03       --mu-fixed 0.3
run K6_dc_mu02_d08   --mu-fixed 0.2 --mu-dyn-ratio 0.8
run K1_impl_d02      --mu-fixed 0.2 --actuator implicit --damping 0.2 --trace
run K2_impl_d2       --mu-fixed 0.2 --actuator implicit --damping 2.0
run K3_dc_iters16_4  --mu-fixed 0.2 --pos-iters 16 --vel-iters 4
run K4_dc_pgs        --mu-fixed 0.2 --solver-type 0
run K5_dc_dt2ms      --mu-fixed 0.2 --sim-dt 0.002
echo "WHEEL_DIAG_ALL_DONE $(date +%T)"
