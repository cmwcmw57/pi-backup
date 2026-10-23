#!/usr/bin/env bash
# Isaac Sim 6.1 WebRTC 재생 보기 (노트북에서 실행). GPU1 만 쓴다 (GPU0 금지).
# 방화벽을 안 건드리고 SSH 터널(49100/tcp) + UDP 브리지(47998/udp <-> 47999/tcp)로 잇는다.
#
#   bash tools/webrtc.sh replay [npz] [follow]  서버에 6.1 재생 컨테이너 띄우기 (기본 최신 녹화, follow 0 / -1=자유시점)
#     (REPLAY_LIGHT=0.5 밝기 배율, REPLAY_ARROW=2.0 화살표 m per m/s, 0=끄기)
#   bash tools/webrtc.sh up                     서버 브리지 + 터널 + 노트북 브리지 + 클라이언트 (클라이언트에서 127.0.0.1 Connect)
#   bash tools/webrtc.sh status
#   bash tools/webrtc.sh down                   노트북 쪽 + 서버 브리지 + 재생 컨테이너 전부 끄기 (GPU1 반납)
#
# 녹화는 서버에서: KSRC 런 → bash run.sh /workspace/scripts/record_play.py ... (CLI.md §11)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
S=${XDG_CACHE_HOME:-$HOME/.cache}/ksrc_gui
mkdir -p "$S"
CLIENT="$HOME/.local/opt/isaacsim-webrtc-client/opt/Isaac Sim WebRTC Streaming Client/isaacsim-webrtc-streaming-client"
IMAGE="nvcr.io/nvidia/isaac-sim:6.1.0@sha256:af1d2b4e75d553bfa27beb5a401198654aa8d607f3b7a6749196e9ce253def20"

remote() { ssh -o ConnectTimeout=15 -o BatchMode=yes hpi-lab "$@"; }
SRV=$(remote 'echo $HOME')/gyusa/ksrc_rover   # server-side path (not the laptop ~)
alive() { [[ -f $S/$1.pid ]] && kill -0 "$(cat "$S/$1.pid")" 2>/dev/null; }
# start a detached process and record its real pid (no setsid fork, no pkill -f self-match)
spawn() { local n=$1; shift; nohup "$@" > "$S/$n.log" 2>&1 < /dev/null & echo $! > "$S/$n.pid"; disown; }
stop_local() { if alive "$1"; then kill "$(cat "$S/$1.pid")"; echo "$1: stopped"; fi; rm -f "$S/$1.pid"; }

server_bridge() {
  remote "cd $SRV; if [ -f /tmp/ksrc_gui_bridge.pid ] && kill -0 \$(cat /tmp/ksrc_gui_bridge.pid) 2>/dev/null; then echo 'server bridge: running'; \
    else nohup python3 tools/udp_tcp_bridge.py server 127.0.0.1:47999 127.0.0.1:47998 > /tmp/ksrc_gui_bridge.log 2>&1 < /dev/null & echo \$! > /tmp/ksrc_gui_bridge.pid; echo 'server bridge: started'; fi"
}

up() {
  [[ -n $(remote 'docker ps -q -f name=^ksrc-gui$') ]] || echo "주의: 서버 재생 컨테이너(ksrc-gui)가 없음 → 먼저 'replay'"
  server_bridge
  if alive tunnel; then echo "tunnel: running"; else
    spawn tunnel ssh -o BatchMode=yes -o ControlPath=none -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
      -N -L 127.0.0.1:49100:127.0.0.1:49100 -L 127.0.0.1:47999:127.0.0.1:47999 hpi-lab
    sleep 3; alive tunnel && echo "tunnel: started" || { echo "tunnel 실패:"; cat "$S/tunnel.log"; exit 1; }
  fi
  if alive bridge; then echo "local bridge: running"; else
    spawn bridge python3 "$HERE/udp_tcp_bridge.py" client 127.0.0.1:47998 127.0.0.1:47999
    sleep 1; alive bridge && echo "local bridge: started" || { echo "local bridge 실패:"; cat "$S/bridge.log"; exit 1; }
  fi
  if alive client; then echo "client: running"; else
    spawn client env DISPLAY="${DISPLAY:-:1}" "$CLIENT" --no-sandbox
    echo "client: started → 창에서 서버 127.0.0.1 입력 후 Connect"
  fi
}

replay() {
  local npz=${1:-} follow=${2:-0}
  [[ -z $npz ]] && npz=$(remote "ls -t $SRV/logs/play/*.npz | head -1 | sed 's#$SRV#/workspace#'")
  echo "replay: $npz (follow=$follow)"
  remote "docker stop ksrc-gui >/dev/null 2>&1; docker rm ksrc-gui >/dev/null 2>&1; \
    docker run -d --name ksrc-gui --user 0:0 --gpus device=1 --shm-size=2g \
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e HOME=/isaac-sim -e OMNI_KIT_ALLOW_ROOT=1 -e ISAACSIM_HOST=127.0.0.1 -e CUDA_VISIBLE_DEVICES=0 \
    -e REPLAY_NPZ=$npz -e REPLAY_USD=/workspace/assets/usd/rover.usd -e REPLAY_FOLLOW=$follow \
    -e REPLAY_LIGHT=${REPLAY_LIGHT:-1.0} -e REPLAY_ARROW=${REPLAY_ARROW:-2.0} \
    -p 127.0.0.1:49100:49100/tcp -p 127.0.0.1:47998:47998/udp -v $SRV:/workspace -w /workspace \
    --entrypoint /isaac-sim/runheadless.sh $IMAGE \
    --/exts/omni.services.livestream.session/quitOnSessionEnded=false \
    --/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/activeGpu=0 \
    --exec /workspace/scripts/replay61.py > /dev/null" || exit 1
  echo -n "app ready 대기"
  for _ in $(seq 60); do
    sleep 5; echo -n .
    remote 'docker logs ksrc-gui 2>&1 | grep -q "app ready"' && { echo " ok"; remote 'docker logs ksrc-gui 2>&1 | grep -a "\[replay61\]" | head -4'; return; }
    [[ -z $(remote 'docker ps -q -f name=^ksrc-gui$') ]] && { echo " 종료됨"; remote 'docker logs --tail 20 ksrc-gui 2>&1'; exit 1; }
  done
  echo " timeout"
}

down() {
  stop_local client; stop_local bridge; stop_local tunnel
  remote 'if [ -f /tmp/ksrc_gui_bridge.pid ]; then kill $(cat /tmp/ksrc_gui_bridge.pid) 2>/dev/null; rm -f /tmp/ksrc_gui_bridge.pid; echo "server bridge: stopped"; fi
    docker stop ksrc-gui >/dev/null 2>&1 && echo "ksrc-gui: stopped"; docker rm ksrc-gui >/dev/null 2>&1
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'
}

status() {
  echo "ksrc-gui: $(remote 'docker ps -a -f name=^ksrc-gui$ --format "{{.Status}}"')"
  remote 'if [ -f /tmp/ksrc_gui_bridge.pid ] && kill -0 $(cat /tmp/ksrc_gui_bridge.pid) 2>/dev/null; then echo "server bridge: running ($(tail -1 /tmp/ksrc_gui_bridge.log))"; else echo "server bridge: stopped"; fi'
  for n in tunnel bridge client; do alive $n && echo "$n: running" || echo "$n: stopped"; done
  [[ -f $S/bridge.log ]] && echo "local bridge: $(tail -1 "$S/bridge.log")"
}

case "${1:-}" in
  up) up ;; replay) shift; replay "$@" ;; down) down ;; status) status ;;
  *) sed -n '2,11p' "$0"; exit 2 ;;
esac
