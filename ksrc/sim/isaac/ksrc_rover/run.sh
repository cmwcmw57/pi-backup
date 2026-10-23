#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU=${KSRC_GPU:-1}   # 1 고정 사용(사용자 지시: GPU0 쓰지 말 것). 돌리기 전 nvidia-smi 확인.
echo "[run.sh] GPU=$GPU" >&2
docker run --rm --gpus "\"device=${GPU}\"" \
  -v "$ROOT:/workspace" -w /workspace \
  -v "$HOME/gyusa/runs/fast_sensor_replay/20261008/reference/assets:/assets:ro" \
  -v "$HOME/gyusa/runs/rover_training_terrains/npzcheck20261007c:/terrain:ro" \
  -e PYTHONPATH=/workspace/source:/terrain \
  -e GIT_PYTHON_REFRESH=quiet \
  -e TZ=KST-9 \
  -e KSRC_RES8_STEER_DEG -e KSRC_RES8_WHEEL \
  "${KSRC_IMAGE:-ksrc-rover:lab2.3.2-rsl312}" "$@"
