#!/usr/bin/env bash
# 학습이 끝날 때까지 기다렸다가 평가를 돌린다.
cd "$(dirname "$0")"
while pgrep -f "train.py --steps" >/dev/null 2>&1; do sleep 20; done
echo "=== 학습 종료. 최종 커리큘럼 ==="
awk '/total_timesteps/{ts=$4} /curriculum\]/{last=$NF; lastts=ts} END{print "  최종 난이도 d="last"  @ "lastts" 스텝"}' runs_balanced.log
ls -la runs/balanced/final.zip runs/balanced/vecnorm.pkl 2>&1 | sed 's/^/  /'
echo
echo "############ 평가 1: 실제 대회장 STL ############"
python3 eval.py --model runs/balanced/final --vecnorm runs/balanced/vecnorm.pkl \
                --arena --episodes 24 --episode-s 20 2>&1 | grep -viE "warn|torch"
echo
echo "############ 평가 2: 랜덤 지형 d=0.8 (학습 분포) ############"
python3 eval.py --model runs/balanced/final --vecnorm runs/balanced/vecnorm.pkl \
                --difficulty 0.8 --episodes 24 --episode-s 20 2>&1 | grep -viE "warn|torch"
