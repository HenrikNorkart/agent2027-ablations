#!/usr/bin/env bash
# X7 / A17-T7, A17-T13: A17 F0 pools (rows of manifests/agent2027/A17_F0.csv) as CPU-only
# f0_generate.py containers against an OpenAI-compatible vLLM server. One container per row,
# rows one after another; each row is 5 folds x 128 single-shot drafts.
# Installed on CAIR as ~/ro27_f0_clients.sh. Two ways to run it (as ~/ro27_rp1_clients.sh):
#
#  (a) CLIENTS_SCRIPT of ~/ro27_session_launch.sh (own server). vllm_session.sh exports PORT,
#      API_KEY, MODEL; SESSION_ID and IMAGE_RO come from the launch environment.
#
#  (b) Stand-alone piggy-back: a client of a server that already runs for another session of
#      the same rung. HOST_SID=<sid> reads port, model and server name from ~/ro27_<HOST_SID>.job.log:
#        F0_ROWS="A17-164 A17-165" HOST_SID=S1022h nohup bash ~/ro27_f0_clients.sh > ~/ro27_f0_S1022h.job.log 2>&1 &
#      The host owns the server; if it is torn down mid-pool the draws become infra errors and
#      the row stops; a rerun resumes from draws.jsonl.
#
# Per row: refuse covtype and non-F0 rows; check the server serves the row's rung; skip a row
# whose summary.json says complete; run the container under a RUN_CAP_HOURS watchdog; a row that
# ends incomplete is re-run once (resume; A17 §4 "a pool job that dies is re-run once").
#
# Env:
#   F0_ROWS (required)   space-separated A17-1xx row ids
#   F0_MANIFEST          default ~/ro27_manifests/A17_F0.csv
#   HOST_SID             mode (b)
#   PORT MODEL API_KEY   server (API_KEY default rogueone-nestedcv-key)
#   SESSION_ID           run tag / container prefix (default HOST_SID, else F0)
#   F0_SCRIPTS           scripts/agent2027 snapshot mounted into the container,
#                        default ~/ro27_f0_code/scripts/agent2027 (its F0_VERSION file is logged)
#   F0_TASKS             default $ROGUEONE_CAIR_PATH/tasks (~/code/RougeOne/tasks), mounted read-only
#   F0_WORK              default ~/ro27_f0_work (one folder per row)
#   F0_CONCURRENCY       in-flight requests per row (default 16)    F0_CPUS (default 4)
#   F0_IMAGE             default rogue_one:<row image_tag>
#   RUN_CAP_HOURS        per-row cap (default 6)
#   RUNGS_CSV            default $F0_SCRIPTS/rungs.csv
#   WANDB_ENV_FILE       default /raid/home/henrbr16/code/PlanBench/.env
#   DRY_RUN=1            per row: print the docker command; no health check, no container
# Host-side tools only: bash, awk, grep, curl, docker. Never python on the host.
set -uo pipefail

DRY_RUN="${DRY_RUN:-0}"
ROWS="${F0_ROWS:?F0_ROWS unset (A17-1xx row ids)}"
F0_SCRIPTS="${F0_SCRIPTS:-$HOME/ro27_f0_code/scripts/agent2027}"
F0_MANIFEST="${F0_MANIFEST:-$HOME/ro27_manifests/A17_F0.csv}"
F0_TASKS="${F0_TASKS:-${ROGUEONE_CAIR_PATH:-$HOME/code/RougeOne}/tasks}"
F0_WORK="${F0_WORK:-$HOME/ro27_f0_work}"
F0_CONCURRENCY="${F0_CONCURRENCY:-16}"
F0_CPUS="${F0_CPUS:-4}"
RUN_CAP_HOURS="${RUN_CAP_HOURS:-6}"
RUNGS_CSV="${RUNGS_CSV:-$F0_SCRIPTS/rungs.csv}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-/raid/home/henrbr16/code/PlanBench/.env}"
API_KEY="${API_KEY:-rogueone-nestedcv-key}"
PROJECT="RogueOne-AGENT2027"

log() { echo "[$(date -u +%FT%TZ)] [f0] $*"; }
fatal() { log "FATAL: $*"; echo "F0_END rc=1"; exit 1; }

case "$(tr 'A-Z' 'a-z' <<<"$ROWS")" in *covtype*) fatal "covtype is never run (Henrik, 2026-10-16)" ;; esac
[ -f "$F0_MANIFEST" ] || fatal "manifest $F0_MANIFEST missing"
grep -qi covtype "$F0_MANIFEST" && fatal "covtype in $F0_MANIFEST; refusing"

# --- the server: mode (b) reads the host session's job log -------------------------------
if [ -n "${HOST_SID:-}" ]; then
  hl="$HOME/ro27_${HOST_SID}.job.log"
  [ -f "$hl" ] || [ "$DRY_RUN" = "1" ] || fatal "host job log $hl missing"
  if [ -f "$hl" ]; then
    hline="$(grep -m1 -E 'server=[^ ]+ model=[^ ]+ port=[0-9]+' "$hl" || true)"
    PORT="${PORT:-$(sed -E 's/.* port=([0-9]+).*/\1/' <<<"$hline")}"
    MODEL="${MODEL:-$(sed -E 's/.* model=([^ ]+).*/\1/' <<<"$hline")}"
    SERVER_NAME="${SERVER_NAME:-$(sed -E 's/.*server=([^ ]+).*/\1/' <<<"$hline")}"
    grep -q "RO27_SESSION_END" "$hl" && [ "$DRY_RUN" != "1" ] && fatal "host session $HOST_SID already ended"
  fi
  SESSION_ID="${SESSION_ID:-$HOST_SID}"
fi
SESSION_ID="${SESSION_ID:-F0}"
[ -n "${PORT:-}" ] || fatal "PORT unset (give HOST_SID or PORT)"
[ -n "${MODEL:-}" ] || fatal "MODEL unset (give HOST_SID or MODEL)"
SID_LC="$(tr 'A-Z' 'a-z' <<<"$SESSION_ID")"
CODE_VERSION="$(cat "$F0_SCRIPTS/F0_VERSION" 2>/dev/null || echo unknown)"
CAP_S=$(awk -v h="$RUN_CAP_HOURS" 'BEGIN { printf "%d", h * 3600 }')

if [ "$DRY_RUN" != "1" ]; then
  if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$WANDB_ENV_FILE" ]; then
    WANDB_API_KEY=$(grep -E '^WANDB_API_KEY=' "$WANDB_ENV_FILE" | head -1 | cut -d= -f2- | tr -d '"'"'"'\r')
  fi
  [ -n "${WANDB_API_KEY:-}" ] || fatal "no WANDB_API_KEY (env or $WANDB_ENV_FILE); W&B is the system of record"
  export WANDB_API_KEY
  export LLM_API_KEY="$API_KEY"
  [ -f "$F0_SCRIPTS/f0_generate.py" ] || fatal "$F0_SCRIPTS/f0_generate.py missing (install step, LAUNCH.md §7)"
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:${PORT}/health" || true)
  [ "$code" = "200" ] || fatal "server on port $PORT not healthy (HTTP $code)"
  curl -s -H "Authorization: Bearer $API_KEY" "http://localhost:${PORT}/v1/models" | grep -q "\"$MODEL\"" \
    || fatal "port $PORT does not serve $MODEL"
fi
log "plan: rows=[$ROWS] model=$MODEL port=$PORT server=${SERVER_NAME:-?} session=$SESSION_ID concurrency=$F0_CONCURRENCY cap_h=$RUN_CAP_HOURS code=$CODE_VERSION"

run_row() {  # $1 row id, $2 attempt
  local row="$1" attempt="$2" line rid abl arm ds tt fs ls rep _s _sid _ok tag rung _rest
  line="$(awk -F',' -v r="$row" 'NR > 1 { gsub(/\r/, ""); if ($1 == r) { print; exit } }' "$F0_MANIFEST")"
  [ -n "$line" ] || { log "REFUSE $row: not in $F0_MANIFEST"; return 3; }
  IFS=',' read -r rid abl arm ds tt fs ls rep _s _sid _ok tag rung _rest <<<"$line"
  [ "$abl" = "A17" ] && [ "$arm" = "f0pool_$rung" ] || { log "REFUSE $row: not an F0 pool row ($abl/$arm)"; return 3; }
  local want; want="$(awk -F',' -v r="$rung" 'NR > 1 && $1 == r { print $2; exit }' "$RUNGS_CSV" 2>/dev/null)"
  [ -n "$want" ] || { log "REFUSE $row: rung $rung not in $RUNGS_CSV"; return 3; }
  [ "$want" = "$MODEL" ] || { log "REFUSE $row: server model $MODEL is not rung $rung ($want)"; return 3; }
  local out="$F0_WORK/$row" image="${F0_IMAGE:-rogue_one:${tag}}" cname
  cname="ro27-${SID_LC}-f0-$(tr 'A-Z' 'a-z' <<<"$row")"
  if grep -qs '"complete": true' "$out/summary.json"; then log "SKIP $row: $out/summary.json is complete"; return 0; fi
  local -a dargs=(run --rm --init --name "$cname" --network host --cpus "$F0_CPUS"
    -e WANDB_API_KEY -e WANDB_ENTITY=PlanBench -e LLM_API_KEY
    -e "SESSION_ID=$SESSION_ID" -e "ROGUE_ONE_IMAGE_TAG=$tag" -e "F0_CODE_VERSION=$CODE_VERSION"
    -e "ROGUE_ONE_GIT_SHA=${GIT_SHA:-}"
    -v "$F0_SCRIPTS:/workspace/scripts/agent2027:ro" -v "$(dirname "$F0_MANIFEST"):/f0manifest:ro"
    -v "$F0_TASKS:/f0tasks:ro" -v "$out:/out"
    --entrypoint python3 "$image" /workspace/scripts/agent2027/f0_generate.py
    --manifest "/f0manifest/$(basename "$F0_MANIFEST")" --row "$row" --tasks /f0tasks
    --model "$MODEL" --port "$PORT" --concurrency "$F0_CONCURRENCY" --out /out --wandb
    --project "$PROJECT")
  if [ "$DRY_RUN" = "1" ]; then
    log "DRYRUN ROW $row rung=$rung dataset=$ds fold_seed=$fs image=$image: docker ${dargs[*]}  (watchdog ${CAP_S}s)"
    return 0
  fi
  docker image inspect "$image" >/dev/null 2>&1 || { log "REFUSE $row: image $image not found"; return 3; }
  docker ps --format '{{.Names}}' | grep -qx "$cname" && { log "REFUSE $row: $cname already running"; return 3; }
  mkdir -p "$out" || { log "REFUSE $row: cannot create $out"; return 3; }
  log "START ROW $row attempt=$attempt rung=$rung dataset=$ds fold_seed=$fs container=$cname"
  local t0 rc; t0=$(date +%s)
  ( sleep "$CAP_S"; log "CAP: ${CAP_S}s reached, docker stop $cname"; docker stop -t 30 "$cname" >/dev/null 2>&1 ) &
  local wd=$!
  docker "${dargs[@]}" > "$out/container_attempt${attempt}.log" 2>&1
  rc=$?
  kill "$wd" 2>/dev/null; wait "$wd" 2>/dev/null
  local wall=$(( $(date +%s) - t0 ))
  local status=FAILED
  grep -qs '"complete": true' "$out/summary.json" && [ "$rc" = "0" ] && status=OK
  [ "$wall" -ge "$CAP_S" ] && [ "$status" != "OK" ] && status=CAP_HIT
  log "ROW $row attempt=$attempt status=$status rc=$rc wall=${wall}s container=$cname"
  [ "$status" = "OK" ] && return 0
  [ "$status" = "CAP_HIT" ] && return 4
  return 1
}

ok=0; failed=0; cap=0; refused=0; failed_rows=""
for row in $ROWS; do
  run_row "$row" 1; rc=$?
  if [ "$rc" = "1" ] && [ "$DRY_RUN" != "1" ]; then run_row "$row" 2; rc=$?; fi
  case "$rc" in
    0) ok=$((ok + 1)) ;;
    3) refused=$((refused + 1)); failed_rows="$failed_rows $row" ;;
    4) cap=$((cap + 1)); failed_rows="$failed_rows $row" ;;
    *) failed=$((failed + 1)); failed_rows="$failed_rows $row" ;;
  esac
done
log "=== RO27_F0_DONE session=$SESSION_ID ok=$ok failed=$failed cap_hit=$cap refused=$refused failed_rows=[${failed_rows# }] dry=$DRY_RUN ==="
[ "$((failed + cap + refused))" = "0" ] && { echo "F0_END rc=0"; exit 0; }
echo "F0_END rc=2"; exit 2
