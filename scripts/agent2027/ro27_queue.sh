#!/usr/bin/env bash
# AGENT2027 session queue for one GPU set (Henrik 2026-10-19: keep the GPUs busy with the
# pre-registered sessions without waiting for a prompt). Installed as ~/ro27_queue.sh.
#
#   QUEUE_FILE=~/ro27_queue_56.txt GPUS=5,6 [WAIT_FOR=<SID>] nohup bash ~/ro27_queue.sh > ~/ro27_queue_56.log 2>&1 &
#
# QUEUE_FILE: one session per line, "SID|IMAGE_TAG|PORT|EXTRA_ENV" (EXTRA_ENV: space-separated
# K=V, optional). Lines are read fresh before each launch, so sessions can be appended or
# removed (by editing the file) while the queue runs; a line starting with '#' is skipped,
# and a session whose ~/ro27_<SID>.job.log already exists is skipped (never relaunched).
# Before each launch: wait until no AGENT2027 session holds GPUS (the previous one's
# RO27_SESSION_END), then GPU pre-flight: every GPU in GPUS must be < 2000 MiB, else wait.
# Stops when no unlaunched line is left, or when ~/ro27_queue_STOP exists.
#
# Non-O120 rungs (SR4 fallback, 2026-10-19): put RUNG=<rung> in EXTRA_ENV (e.g. RUNG=Q4).
# The queue then reads MODEL, EXTRA_ARGS (which contains spaces, so it cannot be passed
# as K=V) and the server name from RUNGS_CSV (default
# $ROGUEONE_CAIR_PATH/scripts/agent2027/rungs.csv), and refuses a rung whose tp differs
# from the number of GPUS. Without RUNG= the behaviour is unchanged (O120).
set -uo pipefail
QF="${QUEUE_FILE:?QUEUE_FILE unset}"
GPUS="${GPUS:?GPUS unset}"
NEED=$(awk -F, '{print NF}' <<<"$GPUS")
log() { echo "[$(date -u +%FT%TZ)] [queue $GPUS] $*"; }

gpus_free() {
  local busy
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' -v want=",$GPUS," 'index(want, ","$1",") && $2+0 >= 2000 {print $1}')
  [ -z "$busy" ]
}

next_line() {
  local sid
  while IFS= read -r line; do
    case "$line" in ''|\#*) continue ;; esac
    sid="${line%%|*}"
    [ -e "$HOME/ro27_${sid}.job.log" ] && continue
    echo "$line"; return 0
  done < "$QF"
  return 1
}

current="${WAIT_FOR:-}"   # a session already running on GPUS (not launched by this queue)
while :; do
  [ -e "$HOME/ro27_queue_STOP" ] && { log "STOP file present; exiting"; exit 0; }
  if [ -n "$current" ] && ! grep -q "RO27_SESSION_END" "$HOME/ro27_${current}.job.log" 2>/dev/null; then
    sleep 60; continue
  fi
  line="$(next_line)" || { log "queue empty; exiting"; exit 0; }
  if ! gpus_free; then sleep 120; continue; fi
  IFS='|' read -r sid tag port extra <<<"$line"
  grep -qi covtype "$HOME/ro27_manifests/sessions/${sid}.csv" && { log "FATAL: covtype in $sid; refusing"; exit 1; }
  [ -f "$HOME/ro27_manifests/sessions/${sid}.csv" ] || { log "FATAL: no session file for $sid"; exit 1; }
  sid_lc="$(tr 'A-Z' 'a-z' <<<"$sid")"
  rung=""; for kv in $extra; do case "$kv" in RUNG=*) rung="${kv#RUNG=}" ;; esac; done
  q_model="${MODEL:-openai/gpt-oss-120b}"; q_extra_args="${EXTRA_ARGS:-}"; q_server="rogueone-vllm-o120-${sid_lc}"
  if [ -n "$rung" ]; then
    rcsv="${RUNGS_CSV:-${ROGUEONE_CAIR_PATH:-$HOME/code/RougeOne}/scripts/agent2027/rungs.csv}"
    rline="$(awk -F',' -v r="$rung" 'NR > 1 && $1 == r { print; exit }' "$rcsv" 2>/dev/null)"
    [ -n "$rline" ] || { log "FATAL: rung $rung not in $rcsv"; exit 1; }
    IFS=',' read -r _r q_model _rev r_tp _port q_extra_args _verdict <<<"$rline"
    [ "$r_tp" = "$NEED" ] || { log "FATAL: rung $rung has tp=$r_tp but GPUS=$GPUS"; exit 1; }
    q_server="rogueone-vllm-$(tr 'A-Z' 'a-z' <<<"$rung")-${sid_lc}"
  fi
  log "LAUNCH $sid image=$tag port=$port gpus=$GPUS rung=${rung:-O120} model=$q_model extra=[$extra]"
  (
    export SESSION_ID="$sid" SESSION_FILE="$HOME/ro27_manifests/sessions/${sid}.csv" GPUS NEED \
      MODEL="$q_model" EXTRA_ARGS="$q_extra_args" PORT="$port" SERVER_NAME="$q_server" \
      LOG="$HOME/vllm_session_${sid}.log" IMAGE_RO="rogue_one:${tag}" PARALLEL="${PARALLEL:-6}" CPUS=12 \
      RUN_CAP_HOURS=6 RUNROOT="$HOME/rogueone-runs-agent2027" WANDB_PROJECT_NAME=RogueOne-AGENT2027 \
      WAIT_MAX_MIN=0 ROGUEONE_ALLOW_SECOND_SERVER=1
    for kv in $extra; do export "$kv"; done
    nohup bash "$HOME/ro27_session_launch.sh" > "$HOME/ro27_${sid}.job.log" 2>&1 < /dev/null
  ) &
  current="$sid"
  sleep 120
done
