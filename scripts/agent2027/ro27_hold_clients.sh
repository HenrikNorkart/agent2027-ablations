#!/usr/bin/env bash
# AGENT 2027 multi-server sessions without ladder_session.sh (SR4 fallback, 2026-10-19).
# Installed as ~/ro27_hold_clients.sh and passed as CLIENTS_SCRIPT to
# ~/ro27_session_launch.sh, so every server still has its own vllm_session.sh owner
# (start, health, teardown, energy sidecar) and its own job log.
#
# A mixed-routing session (A04 session C, A19 under SR4) needs several servers up at once,
# e.g. Q122 TP2 + Q4 TP1 + Q0.8 TP1. E7's planned ladder_session.sh was never built, so the
# session is run as one MAIN launch plus one HOLDER launch per extra server:
#
#   MAIN    (owns e.g. Q122): CLIENTS_SCRIPT=~/ro27_hold_clients.sh WAIT_PORTS="8032 8030"
#           -> waits until every port in WAIT_PORTS answers /health, then runs
#              INNER_CLIENTS (default ~/ro27_clients_manifest.sh) on its SESSION_FILE.
#   HOLDER  (owns e.g. Q4):   CLIENTS_SCRIPT=~/ro27_hold_clients.sh HOLD_FOR=<MAIN SID>
#           [INNER_SESSION_FILE=<file>]  -> optionally runs its own uniform rows first
#              (INNER_CLIENTS on INNER_SESSION_FILE, e.g. A20 + sentinel_Q4 on the Q4
#              server), then BLOCKS until the MAIN job log shows RO27_SESSION_END (or
#              HOLD_MAX_H passes), so its server stays up for the main session's roles.
#
# Launch holders first, then the main session; every launch needs
# ROGUEONE_ALLOW_SECOND_SERVER=1 after a pre-flight that shows the total GPU count is
# within what Henrik allowed. The main session's END marker is written even on FATAL
# (ro27_session_launch.sh), so a holder never waits on a dead main session for long.
#
# Env:
#   WAIT_PORTS          space-separated ports that must answer GET /health (200) before
#                       INNER_CLIENTS starts (default empty: no wait)
#   WAIT_PORTS_MAX_MIN  give up after this long (default 60); exit 3, no row runs
#   INNER_CLIENTS       clients script to run (default ~/ro27_clients_manifest.sh);
#                       "none" = run nothing (pure holder)
#   INNER_SESSION_FILE  session file for INNER_CLIENTS (default $SESSION_FILE). A holder
#                       without own rows sets INNER_CLIENTS=none.
#   INNER_SESSION_ID    session id for INNER_CLIENTS (default: from INNER_SESSION_FILE's
#                       first row, else $SESSION_ID)
#   HOLD_FOR            main session id to wait for after the inner rows (default empty)
#   HOLD_MAX_H          longest hold (default 48)
#   HOLD_POLL_S         poll interval (default 60)
#   DRY_RUN=1           print the plan; no wait, no clients, no hold
# Host-side tools only: bash, curl, grep, awk, sleep.
set -uo pipefail

WAIT_PORTS="${WAIT_PORTS:-}"
WAIT_PORTS_MAX_MIN="${WAIT_PORTS_MAX_MIN:-60}"
INNER_CLIENTS="${INNER_CLIENTS:-$HOME/ro27_clients_manifest.sh}"
INNER_SESSION_FILE="${INNER_SESSION_FILE:-${SESSION_FILE:-}}"
HOLD_FOR="${HOLD_FOR:-}"
HOLD_MAX_H="${HOLD_MAX_H:-48}"
HOLD_POLL_S="${HOLD_POLL_S:-60}"
DRY_RUN="${DRY_RUN:-0}"
SID="${SESSION_ID:-unknown}"

log() { echo "[$(date -u +%FT%TZ)] [hold $SID] $*"; }

inner_sid() {
  if [ -n "${INNER_SESSION_ID:-}" ]; then echo "$INNER_SESSION_ID"; return; fi
  if [ -n "$INNER_SESSION_FILE" ] && [ -f "$INNER_SESSION_FILE" ]; then
    awk -F',' 'NR == 2 { gsub(/["\r]/, "", $10); print $10; exit }' "$INNER_SESSION_FILE"
    return
  fi
  echo "$SID"
}

log "plan: wait_ports=[${WAIT_PORTS}] inner=${INNER_CLIENTS} inner_file=${INNER_SESSION_FILE:-none} hold_for=${HOLD_FOR:-none} port=${PORT:-?} model=${MODEL:-?}"
if [ "$DRY_RUN" = "1" ]; then
  log "DRY_RUN: no wait, no clients, no hold"
  exit 0
fi

# --- 1. other servers of this session must be healthy before any row starts ----------
if [ -n "$WAIT_PORTS" ]; then
  waited=0
  for p in $WAIT_PORTS; do
    while :; do
      code=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:${p}/health" || true)
      [ "$code" = "200" ] && { log "port $p healthy"; break; }
      if [ "$waited" -ge $((WAIT_PORTS_MAX_MIN * 60)) ]; then
        log "FATAL: port $p not healthy after ${WAIT_PORTS_MAX_MIN} min; no row started"
        exit 3
      fi
      sleep 30; waited=$((waited + 30))
    done
  done
fi

# --- 2. this server's own rows ---------------------------------------------------------
rc=0
if [ "$INNER_CLIENTS" != "none" ]; then
  [ -n "$INNER_SESSION_FILE" ] && [ -f "$INNER_SESSION_FILE" ] || { log "FATAL: INNER_SESSION_FILE missing"; exit 1; }
  isid="$(inner_sid)"
  log "running $INNER_CLIENTS on $INNER_SESSION_FILE (session $isid)"
  SESSION_ID="$isid" SESSION_FILE="$INNER_SESSION_FILE" bash "$INNER_CLIENTS"
  rc=$?
  log "inner clients returned rc=$rc"
fi

# --- 3. keep the server up for the main session --------------------------------------
if [ -n "$HOLD_FOR" ]; then
  main_log="$HOME/ro27_${HOLD_FOR}.job.log"
  log "holding until RO27_SESSION_END of $HOLD_FOR ($main_log), at most ${HOLD_MAX_H} h"
  held=0
  until grep -q "RO27_SESSION_END ${HOLD_FOR} " "$main_log" 2>/dev/null; do
    if [ "$held" -ge $((HOLD_MAX_H * 3600)) ]; then
      log "WARNING: HOLD_MAX_H=${HOLD_MAX_H} reached before $HOLD_FOR ended; releasing the server"
      break
    fi
    sleep "$HOLD_POLL_S"; held=$((held + HOLD_POLL_S))
  done
  log "hold released after ${held}s"
fi
exit "$rc"
