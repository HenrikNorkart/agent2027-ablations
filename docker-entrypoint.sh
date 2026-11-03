#!/usr/bin/env bash
set -euo pipefail

if [[ -t 0 || -t 1 ]]; then
    terminal_size="$(stty size 2>/dev/null || true)"
    if [[ -n "$terminal_size" ]]; then
        lines="${terminal_size%% *}"
        columns="${terminal_size##* }"

        if [[ "$columns" =~ ^[0-9]+$ ]] && [[ "$columns" -gt 0 ]]; then
            export COLUMNS="$columns"
        fi

        if [[ "$lines" =~ ^[0-9]+$ ]] && [[ "$lines" -gt 0 ]]; then
            export LINES="$lines"
        fi
    fi
fi

if [[ "$#" -eq 0 ]]; then
    exec /bin/bash -lc "python3 -u /workspace/rogue_one_main.py || tail -f /dev/null"
fi

exec /bin/bash -lc "$*"
