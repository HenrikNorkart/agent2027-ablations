#!/usr/bin/env bash
# Amendment 19c (Henrik 2026-10-29 /unblock: "Dedicated Q122 on GPU 1-2"): inf-guard re-runs A19-503..508.
#   1. Wait until GPU 6 or 7 is empty (S1028a ends about 16:00Z -> GPU 7; S1027j's Q0.8 holder frees GPU 6
#      about 22:00Z). On that GPU start the holders, as S1027k/S1027l were set up on GPU 6:
#      S1029b Q4 (port 8049, 0.50) and S1029c Q0.8 (port 8050, 0.30), no rows, both hold for S1029a.
#   2. Main S1029a: Q122 TP2 on GPUs 1,2 (port 8048), WAIT_PORTS 8049 8050, PARALLEL 6, 18 h row cap.
# L-LAST: nothing is launched after Thu 29 23:30Z (the main must start before 24:00).
# Stop file: ~/ro27_queue_STOP.
set -uo pipefail
LOGF=~/ro27_launch_amend19c.log
exec >> "$LOGF" 2>&1
log() { echo "[$(date -u +%FT%TZ)] $*"; }
DEADLINE=$(date -u -d '2026-10-29 23:30' +%s)
RC=$HOME/code/RougeOne/scripts/agent2027/rungs.csv
xa() { awk -F',' -v r="$1" 'NR > 1 && $1 == r { print $6; exit }' "$RC"; }
Q4X="$(xa Q4)"; Q08X="$(xa Q0.8)"; Q122X="$(xa Q122)"
[ -n "$Q4X" ] && [ -n "$Q08X" ] && [ -n "$Q122X" ] || { log "FATAL: extra_args missing"; exit 1; }
for s in S1029a S1029b S1029c; do [ ! -e "$HOME/ro27_$s.job.log" ] || { log "FATAL: $s already launched"; exit 1; }; done
S=$HOME/ro27_manifests/sessions
[ -s "$S/S1029a.csv" ] || { log "FATAL: $S/S1029a.csv missing"; exit 1; }
grep -qi covtype "$S/S1029a.csv" && { log "FATAL: covtype in S1029a"; exit 1; }
IMG=rogue_one:agent2027-v2.3-qfix6
docker image inspect "$IMG" > /dev/null 2>&1 || { log "FATAL: $IMG missing"; exit 1; }
COMMON="ROGUEONE_ALLOW_SECOND_SERVER=1 IMAGE_RO=$IMG RUNROOT=$HOME/rogueone-runs-agent2027 WANDB_PROJECT_NAME=RogueOne-AGENT2027 WAIT_MAX_MIN=0 MAX_MODEL_LEN=131072 EAGER=0 CLIENTS_SCRIPT=$HOME/ro27_hold_clients.sh"

healthy() { curl -s -o /dev/null -w '%{http_code}' "http://localhost:$1/health" | grep -q 200; }
wait_healthy() {  # $1 port, $2 sid
  for i in $(seq 1 80); do healthy "$1" && { log "$2 healthy on :$1"; return 0; }
    grep -q "RO27_SESSION_END" "$HOME/ro27_$2.job.log" 2>/dev/null && { log "FATAL: $2 ended before healthy"; return 1; }
    sleep 15; done
  log "FATAL: $2 not healthy on :$1 in 20 min"; return 1
}
gpu_mem() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g="$1" '$1 == g { print $2 }'; }
release_holders() {  # $1 reason; holders would otherwise hold their GPU for HOLD_MAX_H
  for h in S1029b S1029c; do [ -e "$HOME/ro27_$h.job.log" ] && bash ~/ro27_stop_session.sh "$h" "$1"; done
}
launch() {  # $1 sid, then env assignments
  local sid=$1; shift
  log "launch $sid: $*"
  env $COMMON "$@" SESSION_ID=$sid LOG=$HOME/vllm_session_$sid.log \
    setsid nohup bash ~/ro27_session_launch.sh > ~/ro27_$sid.job.log 2>&1 < /dev/null &
}

for g in 1 2; do [ "$(gpu_mem $g)" -lt 2000 ] || { log "FATAL: GPU $g not free"; exit 1; }; done
for p in 8048 8049 8050; do ss -ltn | grep -q ":$p " && { log "FATAL: port $p in use"; exit 1; }; done

# 1. holders on the first of GPU 7 / GPU 6 that is empty
log "waiting for GPU 7 or 6 to be empty (deadline 23:30Z)"
HG=""
while [ -z "$HG" ]; do
  [ -e "$HOME/ro27_queue_STOP" ] && { log "STOP file; exiting"; exit 0; }
  [ "$(date +%s)" -lt "$DEADLINE" ] || { log "FATAL: no holder GPU before L-LAST; nothing launched"; exit 1; }
  for g in 7 6; do [ "$(gpu_mem $g)" -lt 2000 ] && { HG=$g; break; }; done
  [ -n "$HG" ] || sleep 60
done
log "GPU $HG empty"
launch S1029b SESSION_FILE=$S/S1029a.csv GPUS=$HG NEED=1 MODEL=Qwen/Qwen3.5-4B PORT=8049 SERVER_NAME=rogueone-vllm-q4-s1029b \
  EXTRA_ARGS="$Q4X --gpu-memory-utilization 0.50" INNER_CLIENTS=none HOLD_FOR=S1029a HOLD_MAX_H=30
wait_healthy 8049 S1029b || { release_holders "amend19c: Q4 holder not healthy"; exit 1; }
launch S1029c SESSION_FILE=$S/S1029a.csv GPUS=$HG NEED=1 MODEL=Qwen/Qwen3.5-0.8B PORT=8050 SERVER_NAME=rogueone-vllm-q0.8-s1029c \
  EXTRA_ARGS="$Q08X --gpu-memory-utilization 0.30" INNER_CLIENTS=none HOLD_FOR=S1029a HOLD_MAX_H=30
wait_healthy 8050 S1029c || { release_holders "amend19c: Q0.8 holder not healthy"; exit 1; }

# 2. main Q122 on GPUs 1,2
[ "$(date +%s)" -lt "$DEADLINE" ] || { log "FATAL: past L-LAST before the main"; release_holders "amend19c: past L-LAST"; exit 1; }
for g in 1 2; do [ "$(gpu_mem $g)" -lt 2000 ] || { log "FATAL: GPU $g taken meanwhile"; release_holders "amend19c: GPU $g taken"; exit 1; }; done
launch S1029a SESSION_FILE=$S/S1029a.csv GPUS=1,2 NEED=2 MODEL=Qwen/Qwen3.5-122B-A10B-FP8 PORT=8048 SERVER_NAME=rogueone-vllm-q122-s1029a \
  EXTRA_ARGS="$Q122X" WAIT_PORTS="8049 8050" PARALLEL=6 CPUS=12 RUN_CAP_HOURS=18
log "launcher done"
