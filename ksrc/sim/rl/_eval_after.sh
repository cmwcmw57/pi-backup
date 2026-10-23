#!/usr/bin/env bash
cd "$(dirname "$0")"
while kill -0 33000 2>/dev/null; do sleep 20; done
M=runs/balanced/final; V=runs/balanced/vecnorm.pkl
echo "=== 학습 종료 ==="
awk '/total_timesteps/{ts=$4} /curriculum\]/{last=$NF; lastts=ts} END{print "  최종 난이도 d="last"  @ "lastts" 스텝"}' runs_balanced.log
echo; echo "########## 실제 대회장 STL (36 에피소드) ##########"
stdbuf -oL python3 eval.py --model $M --vecnorm $V --arena --episodes 36 --episode-s 20 2>&1 | grep -viE "warn|torch"
echo; echo "########## 랜덤 지형 d=0.8 (36 에피소드) ##########"
stdbuf -oL python3 eval.py --model $M --vecnorm $V --difficulty 0.8 --episodes 36 --episode-s 20 2>&1 | grep -viE "warn|torch"
