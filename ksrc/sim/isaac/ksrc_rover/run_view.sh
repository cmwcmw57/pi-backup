#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU=1   # 쓸 GPU 번호. 돌리기 전에 `ssh hpi-lab nvidia-smi` 로 빈 GPU 확인하고 필요시 이 줄만 바꿔라.
echo "[run.sh] GPU=$GPU (고정)" >&2
docker run --rm --gpus "\"device=${GPU}\"" \
  -p 8211:8211 -p 49100:49100/udp \
  -v "$ROOT:/workspace" -w /workspace \
  -v "$HOME/gyusa/runs/fast_sensor_replay/20261008/reference/assets:/assets:ro" \
  -v "$HOME/gyusa/runs/rover_training_terrains/npzcheck20261007c:/terrain:ro" \
  -e PYTHONPATH=/workspace/source:/terrain \
  -e GIT_PYTHON_REFRESH=quiet \
  -e TZ=KST-9 \
  "${KSRC_IMAGE:-ksrc-rover:lab2.3.2-rsl312}" "$@"
