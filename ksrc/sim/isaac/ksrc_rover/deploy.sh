#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER="${KSRC_SERVER:-hpi-lab}"
tar --exclude='__pycache__' -C "$ROOT" -cf - \
  README.md run.sh run_view.sh scripts source benchmarks tools |
  ssh -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=yes "$SERVER" \
    'tar -C ~/gyusa/ksrc_rover -xf -'
echo "Deployed KSRC Isaac task to $SERVER:~/gyusa/ksrc_rover"
