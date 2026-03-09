#!/usr/bin/env bash
set -e

TF_DIR="/opt/venv/lib/python3.10/site-packages/tensorflow"

if [ -d "$TF_DIR" ]; then
  export LD_LIBRARY_PATH="$TF_DIR:${LD_LIBRARY_PATH:-}"

  if [ -f "$TF_DIR/libtensorflow_framework.so.2" ]; then
    export LD_PRELOAD="$TF_DIR/libtensorflow_framework.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
  fi

  if [ -f "$TF_DIR/libtensorflow_cc.so.2" ]; then
    export LD_PRELOAD="$TF_DIR/libtensorflow_cc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
  fi
fi

exec "$@"
