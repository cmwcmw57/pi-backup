#!/usr/bin/env bash
# ================================================================
# record_load.sh
# 설명: 제어 명령 없이 CPU·메모리·스레드·Pi 온도를 시험 폴더에 기록한다.
# 사용법: bash record_load.sh state/S1_01 600
# ================================================================
set -euo pipefail
state_dir="${1:?사용법: bash record_load.sh STATE_DIR [SECONDS]}"
duration="${2:-600}"
[[ "$duration" =~ ^[0-9]+$ ]] && ((duration >= 1 && duration <= 3600)) || {
    echo 'SECONDS는 1~3600 정수여야 합니다.' >&2; exit 2;
}
export LC_ALL=C
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$state_dir/load"
load_dir="$(cd "$state_dir/load" && pwd)"
{
    date -Is
    uname -a
    getconf _NPROCESSORS_ONLN
    uptime
    df -h "$state_dir"
    sha256sum "$script_dir/cli.py" "$script_dir/runner.py" "$script_dir/record_load.sh"
    for name in supervisor processes; do
        if [[ -f "$state_dir/$name.json" ]]; then cat "$state_dir/$name.json"; fi
    done
} > "$load_dir/meta.txt"
children=()
cleanup() {
    if ((${#children[@]})); then
        kill "${children[@]}" 2>/dev/null || true
        wait "${children[@]}" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# top 첫 프레임과 vmstat 첫 데이터 행은 부팅 이후 평균이므로 분석에서 제외한다.
top -b -H -1 -w 200 -d 1 -n "$((duration + 1))" > "$load_dir/top_threads.log" &
children+=("$!")
vmstat -w -t 1 "$((duration + 1))" > "$load_dir/vmstat.log" &
children+=("$!")
for ((elapsed=0; elapsed<duration; elapsed++)); do
    if ((elapsed % 5 == 0)); then
        {
            date -Is
            if [[ -r /sys/class/thermal/thermal_zone0/temp ]]; then
                printf 'temperature_mC='; cat /sys/class/thermal/thermal_zone0/temp
            fi
            if command -v vcgencmd >/dev/null; then
                vcgencmd get_throttled || true
                vcgencmd measure_clock arm || true
            fi
        } >> "$load_dir/thermal.log"
        {
            date -Is
            ps -eLo pid,ppid,lwp,psr,cls,rtprio,pcpu,pmem,rss,comm
        } >> "$load_dir/processes.log"
    fi
    sleep 1
done
for child in "${children[@]}"; do wait "$child"; done
children=()
date -Is > "$load_dir/finished.txt"
echo "load logs: $load_dir"
