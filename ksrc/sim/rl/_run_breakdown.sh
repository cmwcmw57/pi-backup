#!/usr/bin/env bash
cd "$(dirname "$0")"
M="runs/balanced/final"; V="runs/balanced/vecnorm.pkl"
echo "########## 원인 분해: 실제 대회장 STL (36 에피소드) ##########"
stdbuf -oL python3 eval.py --model $M --vecnorm $V --arena --episodes 36 --episode-s 20 2>&1 | grep -viE "warn|torch"
echo
echo "########## 원인 분해: 랜덤 지형 d=0.8 (36 에피소드) ##########"
stdbuf -oL python3 eval.py --model $M --vecnorm $V --difficulty 0.8 --episodes 36 --episode-s 20 2>&1 | grep -viE "warn|torch"
