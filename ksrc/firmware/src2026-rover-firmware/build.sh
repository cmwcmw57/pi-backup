#!/usr/bin/env bash
# 빌드 (+ --flash 주면 보드에 바로 굽기)
#
#   ./build.sh            빌드만
#   ./build.sh --flash    빌드 후 ST-Link 로 플래시
#   ./build.sh --clean    전체 재빌드
#
# arm-none-eabi-gcc 가 PATH 에 없으면 CubeIDE 안에 번들된 툴체인을 찾아 씀.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v arm-none-eabi-gcc >/dev/null 2>&1; then
  # 디렉터리 이름("bin")으로 찾으면 tools/arm-none-eabi/bin 처럼 gcc 가 없는
  # 곳이 먼저 걸린다. 실행 파일 자체를 찾아서 그 디렉터리를 쓴다.
  # 존재하는 디렉터리만 넘긴다. 없는 경로를 주면 find 가 non-zero 로 끝나고,
  # pipefail + set -e 조합 때문에 스크립트가 아무 메시지 없이 죽는다.
  SEARCH_DIRS=()
  for d in /opt/st "$HOME/st" "$HOME/.local/share/stm32cubeide"; do
    [ -d "$d" ] && SEARCH_DIRS+=("$d")
  done
  GCC_BIN=""
  if [ ${#SEARCH_DIRS[@]} -gt 0 ]; then
    GCC_BIN="$(find "${SEARCH_DIRS[@]}" -maxdepth 8 -type f -name arm-none-eabi-gcc 2>/dev/null | head -1 || true)"
  fi
  if [ -z "$GCC_BIN" ]; then
    echo "arm-none-eabi-gcc 를 찾을 수 없습니다. CubeIDE 설치 경로를 확인하세요." >&2
    exit 1
  fi
  export PATH="$(dirname "$GCC_BIN"):$PATH"
fi

DO_FLASH=0
for arg in "$@"; do
  case "$arg" in
    --flash) DO_FLASH=1 ;;
    --clean) make clean ;;
    *) echo "알 수 없는 옵션: $arg" >&2; exit 1 ;;
  esac
done

make -j"$(nproc)"

if [ "$DO_FLASH" = "1" ]; then
  PROG_DIRS=()
  for d in /opt/st "$HOME/st" "$HOME/.local/share/stm32cubeide"; do
    [ -d "$d" ] && PROG_DIRS+=("$d")
  done
  PROG=""
  if [ ${#PROG_DIRS[@]} -gt 0 ]; then
    PROG="$(find "${PROG_DIRS[@]}" -maxdepth 8 -type f -name STM32_Programmer_CLI 2>/dev/null | head -1 || true)"
  fi
  if [ -z "$PROG" ]; then
    echo "STM32_Programmer_CLI 를 찾을 수 없습니다. CubeIDE 에서 플래시하세요." >&2
    exit 1
  fi
  "$PROG" -c port=SWD -w build/rover_motor.elf -rst
fi
