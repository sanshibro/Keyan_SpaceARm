#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
V2_ROOT="$ROOT/.v2_deps/root"
export PYTHONPATH="$V2_ROOT/usr/lib/python3/dist-packages${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$V2_ROOT/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
