#!/usr/bin/env bash
# A19 + A20 relaunch with the Qwen tool fixes (Henrik 2026-10-26 "I approve A19 and A20"; Amendment 18).
#   1. After S1027f (A17 Q0.8, GPU 6) ends and GPU 6 is free: on GPU 6 start
#      the Q4 holder S1027k (port 8044, 0.50) -- runs A20-401..410 as inner rows, then holds for S1027i;
#      the Q0.8 holder S1027l (port 8045, 0.30) -- no rows, holds for S1027j.
#   2. After S1027c (A17 Q122, GPUs 1,2) ends: main S1027i (A19 mix_esmall, Q122 TP2, port 8041, WAIT_PORTS 8044).
#   3. After S1027d (A17 Q122, GPUs 3,4) ends: main S1027j (A19 mix_1b,  Q122 TP2, port 8042, WAIT_PORTS 8045).
# Pattern as ~/ro27_launch_A19c.sh (A19 first wave). Stop file: ~/ro27_queue_STOP.
set -uo pipefail
LOGF=~/ro27_launch_A19r.log
exec >> "$LOGF" 2>&1
log() { echo "[$(date -u +%FT%TZ)] $*"; }
RC=$HOME/code/RougeOne/scripts/agent2027/rungs.csv
xa() { awk -F',' -v r="$1" 'NR > 1 && $1 == r { print $6; exit }' "$RC"; }
Q4X="$(xa Q4)"; Q08X="$(xa Q0.8)"; Q122X="$(xa Q122)"
[ -n "$Q4X" ] && [ -n "$Q08X" ] && [ -n "$Q122X" ] || { log "FATAL: extra_args missing"; exit 1; }
for s in S1027i S1027j S1027k S1027l; do [ ! -e "$HOME/ro27_$s.job.log" ] || { log "FATAL: $s already launched"; exit 1; }; done
S=$HOME/ro27_manifests/sessions
for f in S1027i S1027j S1027k; do
  [ -s "$S/$f.csv" ] || { log "FATAL: $S/$f.csv missing"; exit 1; }
  grep -qi covtype "$S/$f.csv" && { log "FATAL: covtype in $f"; exit 1; }
done
IMG=rogue_one:agent2027-v2.3-qfix5
docker image inspect "$IMG" > /dev/null 2>&1 || { log "FATAL: $IMG missing"; exit 1; }
COMMON="ROGUEONE_ALLOW_SECOND_SERVER=1 IMAGE_RO=$IMG RUNROOT=$HOME/rogueone-runs-agent2027 WANDB_PROJECT_NAME=RogueOne-AGENT2027 WAIT_MAX_MIN=0 MAX_MODEL_LEN=131072 EAGER=0 CLIENTS_SCRIPT=$HOME/ro27_hold_clients.sh"

healthy() { curl -s -o /dev/null -w '%{http_code}' "http://localhost:$1/health" | grep -q 200; }
wait_healthy() {  # $1 port, $2 sid
  for i in $(seq 1 80); do healthy "$1" && { log "$2 healthy on :$1"; return 0; }
    grep -q "RO27_SESSION_END" "$HOME/ro27_$2.job.log" 2>/dev/null && { log "FATAL: $2 ended before healthy"; return 1; }
    sleep 15; done
  log "FATAL: $2 not healthy on :$1 in 20 min"; return 1
}
wait_free() {  # $1 session to wait for, $2 awk GPU condition, $3 label; polls up to 72 h
  log "waiting for $1 to end and GPUs $3 to free"
  for i in $(seq 1 4320); do
    [ -e "$HOME/ro27_queue_STOP" ] && { log "STOP file; exiting"; exit 0; }
    if grep -q "RO27_SESSION_END $1 " "$HOME/ro27_$1.job.log" 2>/dev/null; then
      busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' "($2) && \$2+0 >= 2000")
      [ -z "$busy" ] && { log "$1 ended, GPUs $3 free"; return 0; }
    fi
    sleep 60
  done
  log "FATAL: $1 / GPUs $3 never freed"; return 1
}
launch() {  # $1 sid, then env assignments
  local sid=$1; shift
  log "launch $sid: $*"
  env $COMMON "$@" SESSION_ID=$sid LOG=$HOME/vllm_session_$sid.log \
    setsid nohup bash ~/ro27_session_launch.sh > ~/ro27_$sid.job.log 2>&1 < /dev/null &
}

# 1. holders on GPU 6
wait_free S1027f '$1==6' 6 || exit 1
for p in 8044 8045; do ss -ltn | grep -q ":$p " && { log "FATAL: port $p in use"; exit 1; }; done
launch S1027k SESSION_FILE=$S/S1027k.csv GPUS=6 NEED=1 MODEL=Qwen/Qwen3.5-4B PORT=8044 SERVER_NAME=rogueone-vllm-q4-s1027k \
  EXTRA_ARGS="$Q4X --gpu-memory-utilization 0.50" HOLD_FOR=S1027i HOLD_MAX_H=72 PARALLEL=5 CPUS=12 RUN_CAP_HOURS=12
wait_healthy 8044 S1027k || exit 1
launch S1027l SESSION_FILE=$S/S1027j.csv GPUS=6 NEED=1 MODEL=Qwen/Qwen3.5-0.8B PORT=8045 SERVER_NAME=rogueone-vllm-q0.8-s1027l \
  EXTRA_ARGS="$Q08X --gpu-memory-utilization 0.30" INNER_CLIENTS=none HOLD_FOR=S1027j HOLD_MAX_H=72
wait_healthy 8045 S1027l || exit 1

# 2./3. Q122 mains, each as soon as its A17 Q122 session ends
main() {  # sid, wait-for, gpu cond, gpus, port, holder port
  wait_free "$2" "$3" "$4" || return 1
  ss -ltn | grep -q ":$5 " && { log "FATAL: port $5 in use"; return 1; }
  launch "$1" SESSION_FILE=$S/$1.csv GPUS=$4 NEED=2 MODEL=Qwen/Qwen3.5-122B-A10B-FP8 PORT=$5 SERVER_NAME=rogueone-vllm-q122-$(echo "$1" | tr A-Z a-z) \
    EXTRA_ARGS="$Q122X" WAIT_PORTS=$6 PARALLEL=7 CPUS=12 RUN_CAP_HOURS=12
}
main S1027i S1027c '$1==1||$1==2' 1,2 8041 8044 &
main S1027j S1027d '$1==3||$1==4' 3,4 8042 8045 &
wait
log "launcher done"
