#!/usr/bin/env bash
# AGENT 2027 A08 GPU part (A08-T12): the clients script for an A08 LLM-FE session.
#
# Launched exactly like any ro27 session, through E1's wrapper with a clients override:
#   SESSION_ID=S1020l ... IMAGE_RO=rogue_one:agent2027-v2.N \
#   CLIENTS_SCRIPT=~/a08_comparators_clients.sh nohup bash ~/ro27_session_launch.sh > ~/ro27_S1020l.job.log 2>&1 &
# ro27_session_launch.sh logs both image IDs, runs the E3 energy sidecar and hands off to
# ro27_vllm_session.sh, which owns the vLLM server and exports PORT, API_KEY, MODEL before it
# calls this script. This script BLOCKS until every row (and the retry pass) is done.
#
#   1. Splits $SESSION_FILE by row: A08 `llmfe` rows -> the LLM-FE queue (order_key order);
#      rows of other ablations (the A06 sentinel) -> a filtered session file, run in the
#      background by ~/ro27_clients_manifest.sh with PARALLEL=1 straight on $PORT (Rogue One
#      logs its own tokens, E3). Any other A08 arm (caafe: CL4; CPU rows) is refused.
#   2. Starts the X5 token proxy ro27-<sid>-proxy on $PROXY_PORT -> 127.0.0.1:$PORT, in
#      $IMAGE_RO with the repo's token_proxy.py mounted; LLM-FE talks only to the proxy, with
#      a per-run key a08-llmfe-<ds>-r<rep> that the proxy uses as the attribution label.
#   3. LLM-FE per row (<= PARALLEL clients in total; one slot goes to the sentinel while it runs):
#        API_KEY=a08-llmfe-<ds>-r<rep> SUFFIX=oss120_a08r<rep> PORT=$PROXY_PORT \
#        timeout $RUN_CAP_S bash ~/llmfe_launch.sh --one <ds>   then  docker rm -f llmfe-<suffix>-<ds>
#      Regression tasks run with LLMFE_PROBLEM_NAME=crab and LLMFE_DATA_MOUNT=$A08_DIR/llmfe_alias/<ds>
#      (LLM-FE's main.py decides regression by name; see scripts/agent2027/llmfe_specs/).
#      rc comes from the `--- done <ds> rc=<rc>` line (llmfe_launch.sh --one always exits 0);
#      OK = rc 0 AND 5 split_<i>/samples dirs each with >= 1 samples_*.json.
#      timeout rc 124 = CAP_HIT (ITT, never retried). Anything else FAILED -> one solo retry.
#   4. Waits for the sentinel, stops the proxy (SIGTERM: totals.json + W&B run <SID>__proxy).
#   One line per attempt: `ROW <row_id> attempt=<n> status=<OK|FAILED|CAP_HIT> rc=<rc> wall=<s>s ...`
#   on stdout (vLLM session log) and in $RO27_JOB_LOG; table in $A08_DIR/<SID>/status.tsv.
#   Ends with `=== RO27_CLIENTS_DONE session=<SID> ... ===`; exit 0 iff every row is OK.
#
# Host-side tools only: docker, timeout, coreutils, awk, grep, sed, find, xargs, curl. No python.
#
# Env: SESSION_ID (required)  SESSION_FILE (~/ro27_manifests/sessions/<SID>.csv)
#   PORT API_KEY MODEL        from ro27_vllm_session.sh (API_KEY = the server key; never logged)
#   PROXY_PORT (8090)         one proxy per session; a second concurrent session uses 8091
#   IMAGE_RO (required)       rogue_one:agent2027-v2.N (proxy image and the sentinel's image)
#   PARALLEL (6)  RUN_CAP_S (21600 = 6 h)  THREADS (8, LLM-FE container CPUs)
#   LLMFE_DIR (~/llmfe)  LLMFE_LAUNCH (~/llmfe_launch.sh)  LLMFE_IMAGE (llmfe:oss120)
#   A08_DIR (~/ro27_a08)  CLIENTS_MANIFEST (~/ro27_clients_manifest.sh)
#   ROGUEONE_CAIR_PATH (~/code/RougeOne)  WANDB_ENV_FILE (PlanBench .env)  RO27_JOB_LOG
#   DRY_RUN=1  print the plan only (no docker, no proxy, no LLM-FE)
set -uo pipefail

SESSION_ID="${SESSION_ID:?SESSION_ID is required (S<MMDD><a-z>)}"
SID_LC="$(printf '%s' "$SESSION_ID" | tr 'A-Z' 'a-z')"
SESSION_FILE="${SESSION_FILE:-$HOME/ro27_manifests/sessions/${SESSION_ID}.csv}"
PORT="${PORT:-}"
API_KEY="${API_KEY:-}"
PROXY_PORT="${PROXY_PORT:-8090}"
IMAGE_RO="${IMAGE_RO:-}"
PARALLEL="${PARALLEL:-6}"
RUN_CAP_S="${RUN_CAP_S:-21600}"
THREADS="${THREADS:-8}"
LLMFE_DIR="${LLMFE_DIR:-$HOME/llmfe}"
LLMFE_LAUNCH="${LLMFE_LAUNCH:-$HOME/llmfe_launch.sh}"
LLMFE_IMAGE="${LLMFE_IMAGE:-llmfe:oss120}"
A08_DIR="${A08_DIR:-$HOME/ro27_a08}"
CLIENTS_MANIFEST="${CLIENTS_MANIFEST:-$HOME/ro27_clients_manifest.sh}"
ROGUEONE_CAIR_PATH="${ROGUEONE_CAIR_PATH:-$HOME/code/RougeOne}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-/raid/home/henrbr16/code/PlanBench/.env}"
RO27_JOB_LOG="${RO27_JOB_LOG:-}"
DRY_RUN="${DRY_RUN:-0}"
REGRESSION_ALIAS="crab"   # see scripts/agent2027/llmfe_specs/make_llmfe_inputs.py
PROXY_NAME="ro27-${SID_LC}-proxy"
SESSDIR="$A08_DIR/$SESSION_ID"
STATUS="$SESSDIR/status.tsv"
RETRYQ="$SESSDIR/retry_queue.txt"
US=$'\x1f'
R7_HEADER="row_id,ablation_id,arm_id,dataset_id,task_type,fold_seed,llm_seed,replicate,is_sentinel,session_id,order_key,image_tag,rung_S,rung_E,rung_T,port_S,port_E,port_T,env_overrides,cap_hours,expected_u,notes"

log() { echo "[$(date -u +%FT%TZ)] $*"; }
row_log() {
  local line="[$(date -u +%FT%TZ)] $*"
  echo "$line"
  if [ -n "$RO27_JOB_LOG" ]; then echo "$line" >> "$RO27_JOB_LOG"; fi
}
fatal() { row_log "FATAL (A08 clients $SESSION_ID): $*"; exit 1; }

# --------------------------------------------------------------------------- parse + split
[ -f "$SESSION_FILE" ] || fatal "SESSION_FILE=$SESSION_FILE does not exist"
mkdir -p "$SESSDIR" || fatal "mkdir $SESSDIR"
first="$(head -1 "$SESSION_FILE" | tr -d '\r')"
[ "$first" = "$R7_HEADER" ] || fatal "session file header is not the R7 header"

# CSV -> US-separated rows (fields 1-12 are never quoted-with-commas in R7 files except
# env_overrides (19) and notes (22); a small RFC 4180 parser keeps it exact).
awk -v us="$US" '
  function parse_csv(line, out,    n, i, c, field, inq) {
    n = 0; field = ""; inq = 0
    for (i = 1; i <= length(line); i++) {
      c = substr(line, i, 1)
      if (inq) { if (c == "\"") { if (substr(line, i + 1, 1) == "\"") { field = field "\""; i++ } else inq = 0 } else field = field c }
      else if (c == "\"") inq = 1
      else if (c == ",") { out[++n] = field; field = "" }
      else field = field c
    }
    out[++n] = field
    return n
  }
  NR == 1 { next }
  { sub(/\r$/, "") }
  /^[ ]*$/ { next }
  { n = parse_csv($0, f); if (n != 22) { printf "BAD %d\n", NR; next }
    line = f[1]; for (i = 2; i <= 22; i++) line = line us f[i]; print line }
' "$SESSION_FILE" | sort -t "$US" -k11,11n -k1,1 > "$SESSDIR/rows.usv"
grep -q '^BAD ' "$SESSDIR/rows.usv" && fatal "a session-file line does not have 22 fields"

covt=$(awk -F "$US" 'tolower($4) ~ /covtype/ {print $1}' "$SESSDIR/rows.usv" | tr '\n' ' ')
[ -z "${covt// /}" ] || fatal "covtype rows are excluded from every Rogue One study (Henrik, 2026-10-16): $covt"
bad_sid=$(awk -F "$US" -v s="$SESSION_ID" '$10 != s {print $1}' "$SESSDIR/rows.usv" | tr '\n' ' ')
[ -z "${bad_sid// /}" ] || fatal "rows not in session $SESSION_ID: $bad_sid"

awk -F "$US" '$2 == "A08" && $3 == "llmfe"' "$SESSDIR/rows.usv" > "$SESSDIR/llmfe.usv"
other_a08=$(awk -F "$US" '$2 == "A08" && $3 != "llmfe" {print $1 "(" $3 ")"}' "$SESSDIR/rows.usv" | tr '\n' ' ')
[ -z "${other_a08// /}" ] || fatal "A08 rows this script does not run (CAAFE is dropped under CL4; CPU rows run in classicfe): $other_a08"
# Rows of other ablations (the A06 sentinel), verbatim, into a filtered session file.
FILTERED="$SESSDIR/${SESSION_ID}.rogueone.csv"
{ echo "$R7_HEADER"; tail -n +2 "$SESSION_FILE" | tr -d '\r' | grep -v -E '^A08-' | grep -v -E '^[ ]*$'; } > "$FILTERED"
N_LLMFE=$(wc -l < "$SESSDIR/llmfe.usv" | tr -d ' ')
N_RO=$(( $(wc -l < "$FILTERED") - 1 ))
log "=== A08 clients: session=$SESSION_ID llmfe_rows=$N_LLMFE rogueone_rows=$N_RO parallel=$PARALLEL proxy=$PROXY_PORT->${PORT:-?} cap_s=$RUN_CAP_S dry_run=$DRY_RUN ==="
[ "$N_LLMFE" -gt 0 ] || [ "$N_RO" -gt 0 ] || fatal "no rows"
awk -F "$US" '{ printf "  %s\t%s\t%s\t%s\tr%s\n", $11, $1, $3, $4, $8 }' "$SESSDIR/rows.usv"

# --------------------------------------------------------------------------- checks
[ -n "$IMAGE_RO" ] || fatal "IMAGE_RO is required (proxy image; the sentinel's image)"
if [ "$DRY_RUN" != "1" ]; then
  [ -n "$PORT" ] && [ -n "$API_KEY" ] || fatal "PORT/API_KEY not exported (run under ro27_vllm_session.sh)"
  [ -f "$LLMFE_LAUNCH" ] || fatal "$LLMFE_LAUNCH missing (cp <P>/paper/scripts/llmfe_launch.sh ~/)"
  docker image inspect "$LLMFE_IMAGE" >/dev/null 2>&1 || fatal "image $LLMFE_IMAGE missing (A08-T6)"
  log "llmfe image $LLMFE_IMAGE id=$(docker image inspect "$LLMFE_IMAGE" --format '{{.Id}}')"
  [ -f "$ROGUEONE_CAIR_PATH/scripts/agent2027/token_proxy/token_proxy.py" ] || fatal "token_proxy.py not in $ROGUEONE_CAIR_PATH (fetch agent2027 first)"
  if [ "$N_RO" -gt 0 ]; then [ -f "$CLIENTS_MANIFEST" ] || fatal "$CLIENTS_MANIFEST missing"; fi
  missing=""
  while IFS="$US" read -r rid _ _ ds ttype _; do
    [ -f "$LLMFE_DIR/specs/specification_${ds}.txt" ] || missing="$missing $LLMFE_DIR/specs/specification_${ds}.txt"
    [ -f "$LLMFE_DIR/data/${ds}.csv" ] || missing="$missing $LLMFE_DIR/data/${ds}.csv"
    if [ "$ttype" = "regression" ]; then
      [ -f "$A08_DIR/llmfe_alias/${ds}/${REGRESSION_ALIAS}.csv" ] || missing="$missing $A08_DIR/llmfe_alias/${ds}/${REGRESSION_ALIAS}.csv"
    fi
  done < "$SESSDIR/llmfe.usv"
  [ -z "$missing" ] || fatal "LLM-FE inputs missing (X4 install step):$missing"
  # Never mix a new run into an old run's sample directories.
  stale=""
  while IFS="$US" read -r rid _ _ ds _ _ _ rep _; do
    [ -e "$LLMFE_DIR/logs/${ds}_oss120_a08r${rep}_split_1" ] && stale="$stale ${ds}_oss120_a08r${rep}"
  done < "$SESSDIR/llmfe.usv"
  [ -z "$stale" ] || fatal "LLM-FE log dirs already exist (an earlier attempt?): $stale -- move them aside first"
fi

[ -f "$STATUS" ] || printf 'row_id\tattempt\tstatus\trc\twall_s\tstart_epoch\tend_epoch\tsplits_ok\tclient_key\treason\n' > "$STATUS"
: > "$RETRYQ"

# --------------------------------------------------------------------------- proxy
proxy_up=0
stop_proxy() {
  [ "$proxy_up" = "1" ] || return 0
  log "stopping token proxy $PROXY_NAME (SIGTERM: totals + W&B upload)"
  docker stop -t 120 "$PROXY_NAME" >/dev/null 2>&1 || true
  docker logs --tail 5 "$PROXY_NAME" 2>&1 | sed 's/^/  proxy: /'
  docker rm -f "$PROXY_NAME" >/dev/null 2>&1 || true
  proxy_up=0
}
RO_PID=""
cleanup() {
  # Interrupted: stop the sentinel's clients script (its own trap stops its container),
  # then our LLM-FE containers, the proxy last.
  if [ -n "$RO_PID" ] && kill -0 "$RO_PID" 2>/dev/null; then
    kill -TERM "$RO_PID" 2>/dev/null; wait "$RO_PID" 2>/dev/null
  fi
  for n in $(docker ps --format '{{.Names}}' 2>/dev/null | grep -E '^llmfe-oss120_a08r[01]-' || true); do
    grep -qx "$n" "$SESSDIR/started.txt" 2>/dev/null && docker rm -f "$n" >/dev/null 2>&1
  done
  stop_proxy
}

if [ "$DRY_RUN" != "1" ]; then
  if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$WANDB_ENV_FILE" ]; then
    WANDB_API_KEY=$(grep -E '^WANDB_API_KEY=' "$WANDB_ENV_FILE" | head -1 | cut -d= -f2- | tr -d '"'"'"'\r')
  fi
  export WANDB_API_KEY="${WANDB_API_KEY:-}"
  docker ps -a --format '{{.Names}}' | grep -qx "$PROXY_NAME" && fatal "$PROXY_NAME already exists"
  mkdir -p "$A08_DIR/proxy/$SESSION_ID"
  trap 'cleanup' EXIT
  trap 'exit 143' INT TERM
  UPSTREAM_API_KEY="$API_KEY" docker run -d --name "$PROXY_NAME" --network host \
      -e UPSTREAM_API_KEY -e WANDB_API_KEY -e WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-RogueOne-AGENT2027}" \
      -v "$ROGUEONE_CAIR_PATH/scripts/agent2027/token_proxy:/proxy:ro" \
      -v "$A08_DIR/proxy/$SESSION_ID:/out" \
      --entrypoint python3 "$IMAGE_RO" /proxy/token_proxy.py \
      --listen-port "$PROXY_PORT" --upstream "http://127.0.0.1:$PORT" --log-dir /out \
      --session-id "$SESSION_ID" --wandb >/dev/null || fatal "token proxy failed to start"
  proxy_up=1
  ok=0
  for i in $(seq 1 30); do
    code=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:${PROXY_PORT}/health" || true)
    [ "$code" = "200" ] && { ok=1; break; }
    sleep 2
  done
  [ "$ok" = "1" ] || { docker logs --tail 20 "$PROXY_NAME" 2>&1 | sed 's/^/  proxy: /'; fatal "proxy :$PROXY_PORT never healthy"; }
  log "token proxy $PROXY_NAME healthy on :$PROXY_PORT -> :$PORT (log $A08_DIR/proxy/$SESSION_ID)"
fi

# --------------------------------------------------------------------------- sentinel
SLOTS="$PARALLEL"
if [ "$N_RO" -gt 0 ]; then
  SLOTS=$(( PARALLEL - 1 ))
  [ "$SLOTS" -ge 1 ] || SLOTS=1
  log "Rogue One rows (sentinel) -> $CLIENTS_MANIFEST, PARALLEL=1 on :${PORT:-?}; LLM-FE slots $SLOTS"
  SESSION_FILE="$FILTERED" PARALLEL=1 DRY_RUN="$DRY_RUN" bash "$CLIENTS_MANIFEST" &
  RO_PID=$!
fi

# --------------------------------------------------------------------------- llmfe rows
run_llmfe() {
  local rid="$1" attempt="$2" line
  line="$(awk -F "$US" -v id="$rid" '$1 == id' "$SESSDIR/llmfe.usv")"
  IFS="$US" read -r R_ROW R_AID R_ARM R_DS R_TYPE R_FSEED R_LSEED R_REP R_SENT R_SID R_OKEY R_TAG _rest <<< "$line"
  local key="a08-llmfe-${R_DS}-r${R_REP}" suffix="oss120_a08r${R_REP}"
  local cname="llmfe-${suffix}-${R_DS}" out="$SESSDIR/${rid}.attempt${attempt}.out"
  local logdir="$A08_DIR/llmfe_logs_r${R_REP}"
  local -a envs=(API_KEY="$key" SUFFIX="$suffix" PORT="$PROXY_PORT" CLIENT_LOGDIR="$logdir"
                 IMAGE="$LLMFE_IMAGE" THREADS="$THREADS")
  if [ "$R_TYPE" = "regression" ]; then
    envs+=(LLMFE_PROBLEM_NAME="$REGRESSION_ALIAS" LLMFE_DATA_MOUNT="$A08_DIR/llmfe_alias/${R_DS}")
  fi
  local t0; t0=$(date +%s)
  if [ "$DRY_RUN" = "1" ]; then
    row_log "DRYRUN ROW $rid attempt=$attempt order_key=$R_OKEY: env ${envs[*]} timeout $RUN_CAP_S bash $LLMFE_LAUNCH --one $R_DS"
    printf '%s\t%s\tDRY\t-\t0\t%s\t%s\t-\t%s\tdry-run\n' "$rid" "$attempt" "$t0" "$t0" "$key" >> "$STATUS"
    return 0
  fi
  if [ "$attempt" != "1" ]; then
    # A retry starts from empty sample dirs; the failed attempt is kept beside them.
    for i in 1 2 3 4 5; do
      d="$LLMFE_DIR/logs/${R_DS}_${suffix}_split_${i}"
      [ -e "$d" ] && mv "$d" "${d}.attempt$((attempt - 1)).$(date +%s)" 2>/dev/null
    done
  fi
  echo "$cname" >> "$SESSDIR/started.txt"
  row_log "START ROW $rid attempt=$attempt dataset=$R_DS task_type=$R_TYPE rep=$R_REP key=$key container=$cname"
  env "${envs[@]}" timeout "$RUN_CAP_S" bash "$LLMFE_LAUNCH" --one "$R_DS" > "$out" 2>&1
  local trc=$?
  docker rm -f "$cname" >/dev/null 2>&1 || true   # timeout kills the launcher, not the container
  local t1; t1=$(date +%s)
  local rc; rc="$(grep -oE -- "--- done ${R_DS} rc=[0-9]+" "$out" | tail -1 | grep -oE '[0-9]+$' || true)"
  local splits=0 i
  for i in 1 2 3 4 5; do
    ls "$LLMFE_DIR/logs/${R_DS}_${suffix}_split_${i}/samples/"samples_*.json >/dev/null 2>&1 && splits=$((splits + 1))
  done
  local status reason=ok
  if [ "$trc" = "124" ] || [ "$trc" = "137" ]; then
    status=CAP_HIT; reason="cap_${RUN_CAP_S}s"; rc="${rc:-timeout}"
  elif [ "$rc" = "0" ] && [ "$splits" = "5" ]; then
    status=OK
  else
    status=FAILED; reason="rc=${rc:-none},splits=$splits/5"
  fi
  row_log "ROW $rid attempt=$attempt status=$status rc=${rc:-none} wall=$((t1 - t0))s splits=$splits/5 key=$key${reason:+ reason=$reason}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$rid" "$attempt" "$status" "${rc:-none}" "$((t1 - t0))" \
    "$t0" "$t1" "$splits" "$key" "$reason" >> "$STATUS"
  if [ "$status" = "FAILED" ] && [ "$attempt" = "1" ]; then echo "$rid" >> "$RETRYQ"; fi
  return 0
}
export -f run_llmfe row_log log
export SESSDIR STATUS RETRYQ US PROXY_PORT LLMFE_IMAGE THREADS LLMFE_LAUNCH LLMFE_DIR A08_DIR \
  RUN_CAP_S REGRESSION_ALIAS DRY_RUN RO27_JOB_LOG

if [ "$N_LLMFE" -gt 0 ]; then
  NP="$SLOTS"; [ "$DRY_RUN" = "1" ] && NP=1
  { cut -d "$US" -f1 "$SESSDIR/llmfe.usv" | xargs -P "$NP" -I{} bash -c 'run_llmfe "$1" 1' _ {}; } &
  wait $!
  log "=== LLM-FE concurrent pass done ==="
  RETRY=$(sort -u "$RETRYQ" | tr '\n' ' ')
  if [ -n "${RETRY// /}" ]; then
    log "=== LLM-FE retry pass (solo): $RETRY==="
    for rid in $RETRY; do run_llmfe "$rid" 2; done
  else
    log "=== no LLM-FE first-pass failures, no retry pass ==="
  fi
fi

ro_rc=0
if [ -n "$RO_PID" ]; then
  log "waiting for the Rogue One rows (pid $RO_PID)"
  wait "$RO_PID"; ro_rc=$?
  log "Rogue One rows returned rc=$ro_rc"
fi

# --------------------------------------------------------------------------- summary
final=$(awk -F '\t' 'NR > 1 { last[$1] = $3 } END { for (r in last) print r "\t" last[r] }' "$STATUS")
n_ok=$(printf '%s\n' "$final" | grep -c $'\tOK$' || true)
n_cap=$(printf '%s\n' "$final" | grep -c $'\tCAP_HIT$' || true)
n_fail=$(printf '%s\n' "$final" | grep -c $'\tFAILED$' || true)
n_dry=$(printf '%s\n' "$final" | grep -c $'\tDRY$' || true)
still=$(printf '%s\n' "$final" | awk -F '\t' '$2 == "FAILED" {print $1}' | sort | tr '\n' ' ')
row_log "=== RO27_CLIENTS_DONE session=$SESSION_ID llmfe_ok=$n_ok llmfe_failed=$n_fail llmfe_cap_hit=$n_cap dry=$n_dry failed_rows=[${still% }] rogueone_rc=$ro_rc ==="
stop_proxy
[ "$n_fail" = "0" ] && [ "$n_cap" = "0" ] && [ "$ro_rc" = "0" ] && exit 0
exit 2
