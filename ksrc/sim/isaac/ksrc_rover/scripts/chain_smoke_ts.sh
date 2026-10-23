#!/bin/bash
# teacher-student 코드 첫 실행 시험 (짧게, 64env). 실제 학습 체인 전에 한 번.
#  0) check_dr.py: DC모터 텐서 토크제한 / 관측지연 / 행동지연
#  1) teacher Res8 3iter (DR1) → 2) 같은 teacher 를 DR2 로 resume 2iter
#  3) student MLP / GRU 증류 2iter (teacher 가중치 로드 확인) → 4) eval_log 짧게
# 서버: cd ~/gyusa/ksrc_rover && setsid nohup bash scripts/chain_smoke_ts.sh </dev/null > logs/smoke_ts.master.log 2>&1 & disown
cd ~/gyusa/ksrc_rover
E=/workspace/logs/eval
n_train() { docker ps --filter ancestor=ksrc-rover:lab2.3.2-rsl312 --format '{{.Names}}' | grep -vc ksrc_tb; }
wait_gpu1() {
  while :; do
    used=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    [[ "$used" =~ ^[0-9]+$ ]] && [ "$(n_train)" -eq 0 ] && [ "$used" -lt 1000 ] && break
    sleep 30
  done
}
step() { echo "=== $1 $(date +%T) ==="; }
TD=logs/rsl_rl/ksrc_rover_teacher_res8
wait_gpu1; step "0 check_dr"
bash run.sh /workspace/scripts/check_dr.py --headless > logs/smoke_check_dr.log 2>&1
grep -aE "^\[[123]\]|CHECK_DR|Error|error" logs/smoke_check_dr.log | tail -20
wait_gpu1; step "1 teacher DR1"
bash run.sh /workspace/scripts/train.py --task Isaac-KSRC-Rover-Teacher-Res8-v0 --headless \
  --num_envs 64 --max_iterations 3 --seed 1 --run_name smoke > logs/smoke_t1.log 2>&1
T=$(basename "$(ls -dt $TD/*_smoke/ | head -1)"); echo "teacher run $T: $(ls $TD/$T | grep model | tr '\n' ' ')"
grep -aE "Traceback|Error" logs/smoke_t1.log | head -5
grep -aE "obs|Actor MLP|privileged" logs/smoke_t1.log | grep -ai "shape\|dim\|input" | head -5
CK=$(ls $TD/$T | grep model | sort -V | tail -1)
wait_gpu1; step "2 teacher DR2 resume"
bash run.sh /workspace/scripts/train.py --task Isaac-KSRC-Rover-Teacher-Res8-v0 --headless \
  --num_envs 64 --max_iterations 2 --seed 1 --run_name smoke_dr2 --resume --load_run $T --checkpoint $CK \
  --set DR_STAGE=2 > logs/smoke_t2.log 2>&1
grep -aE "\[set\]|Loading model|Traceback|Error" logs/smoke_t2.log | head -6
for V in MLP:5 GRU:1; do
  IFS=: read A H <<< "$V"
  wait_gpu1; step "3 student $A H=$H"
  bash run.sh /workspace/scripts/train.py --task Isaac-KSRC-Rover-Student-$A-Res8-v0 --headless \
    --num_envs 64 --max_iterations 2 --seed 1 --run_name smoke_student_$A --load_run $T --checkpoint $CK \
    --set STUDENT_HISTORY=$H > logs/smoke_s_$A.log 2>&1
  grep -aE "Loading model|Student|Teacher MLP|Teacher model|Traceback|Error|Learning iteration" logs/smoke_s_$A.log | head -8
done
wait_gpu1; step "4 eval"
bash run.sh /workspace/scripts/eval_log.py --task Isaac-KSRC-Rover-Teacher-Res8-v0 --checkpoint /workspace/$TD/$T/$CK \
  --num_envs 16 --steps 100 --out $E/smoke_teacher.npz --headless > logs/smoke_eval_t.log 2>&1
grep -aE "EVAL_SAVED|Traceback|Error" logs/smoke_eval_t.log | head -3
for V in MLP:5 GRU:1; do
  IFS=: read A H <<< "$V"
  S=$(basename "$(ls -dt $TD/*_smoke_student_$A/ | head -1)"); SC=$(ls $TD/$S | grep model | sort -V | tail -1)
  bash run.sh /workspace/scripts/eval_log.py --task Isaac-KSRC-Rover-Student-$A-Res8-v0 --checkpoint /workspace/$TD/$S/$SC \
    --num_envs 16 --steps 100 --set STUDENT_HISTORY=$H --out $E/smoke_student_$A.npz --headless > logs/smoke_eval_$A.log 2>&1
  grep -aE "EVAL_SAVED|Traceback|Error" logs/smoke_eval_$A.log | head -3
done
echo "SMOKE_TS_DONE $(date +%F_%T) teacher=$T"
