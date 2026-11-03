#!/usr/bin/env bash
# Start each A18 replay row as a CPU client of a GPU 7 chain server once that host session's
# server is serving (its first START ROW line), and before the host ends. Installed as
# ~/ro27_a18_riders.sh; uses ~/ro27_rp1_clients.sh (mode b, HOST_SID). One rider per row, in order.
#   nohup bash ~/ro27_a18_riders.sh > ~/ro27_a18_riders.log 2>&1 &
set -uo pipefail
PAIRS="${PAIRS:-A18-003:S1019g A18-001:S1020h A18-002:S1020i A18-004:S1020j}"
log() { echo "[$(date -u +%FT%TZ)] [a18-riders] $*"; }
for p in $PAIRS; do
  row="${p%%:*}"; host="${p##*:}"; hl="$HOME/ro27_${host}.job.log"; out="$HOME/ro27_a18_${row}.job.log"
  [ -e "$out" ] && { log "$row already started ($out); skip"; continue; }
  log "waiting for $host (row $row)"
  until [ -f "$hl" ] && grep -q "START ROW" "$hl"; do
    [ -e "$HOME/ro27_queue_STOP" ] && { log "STOP file; exiting"; exit 0; }
    sleep 120
  done
  if grep -q "RO27_SESSION_END" "$hl"; then log "$host already ended; $row NOT run (own-server form needed)"; continue; fi
  log "START $row on $host"
  RP1_ROW="$row" HOST_SID="$host" nohup bash "$HOME/ro27_rp1_clients.sh" > "$out" 2>&1 < /dev/null &
  sleep 60
done
log "all riders started; exiting"
