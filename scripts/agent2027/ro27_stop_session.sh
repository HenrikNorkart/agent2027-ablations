#!/usr/bin/env bash
# Stop one AGENT2027 session cleanly (installed as ~/ro27_stop_session.sh):
#   bash ~/ro27_stop_session.sh <SID> "<reason>"
# Killing ro27_clients_manifest.sh alone is NOT enough: its `xargs -P` row workers keep
# starting rows (seen 2026-10-20 on S1019g). This kills every process whose environment
# has SESSION_ID=<SID> (launcher, vllm_session, clients, xargs, run_row workers, docker
# run clients), then stops the session's containers (ro27-<sid>-*: clients, energy
# sidecar) and its vLLM server, and appends a STOPPED line to ~/ro27_<SID>.job.log.
# It never touches a process or container of any other session.
set -uo pipefail
SID="${1:?usage: ro27_stop_session.sh <SID> <reason>}"; REASON="${2:?reason required}"
sid_lc="$(tr 'A-Z' 'a-z' <<<"$SID")"
log="$HOME/ro27_${SID}.job.log"
for round in 1 2 3; do
  pids=""
  for p in $(pgrep -u "$(id -u)" -f .); do
    tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -qx "SESSION_ID=$SID" && pids="$pids $p"
  done
  [ -z "${pids// /}" ] && break
  kill $pids 2>/dev/null; sleep 3
done
rows="$(docker ps --format '{{.Names}}' | grep "^ro27-${sid_lc}-" | grep -v -- '-energy$' | sed "s/^ro27-${sid_lc}-//" | tr '\n' ' ')"
for c in $(docker ps --format '{{.Names}}' | grep -E "^ro27-${sid_lc}-|^rogueone-vllm-.*-${sid_lc}$"); do
  docker stop -t 20 "$c" >/dev/null && echo "stopped $c"
done
echo "[$(date -u +%FT%TZ)] === RO27_SESSION_STOPPED $SID: $REASON; rows stopped mid-run (not results): ${rows:-none} ===" | tee -a "$log"
