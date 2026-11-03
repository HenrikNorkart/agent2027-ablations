#!/usr/bin/env bash
# A18-T7: one RP1 role-wise replay target (row RP1_ROW of manifests/agent2027/A18_RP1.csv)
# as a CPU-only replay_tool.py container against an OpenAI-compatible vLLM server.
# Installed on CAIR as ~/ro27_rp1_clients.sh. Two ways to run it:
#
#  (a) CLIENTS_SCRIPT of ~/ro27_session_launch.sh (own one-GPU/TP2 server; A18-T11 form).
#      vllm_session.sh exports PORT, API_KEY, MODEL and runs this script in the foreground,
#      then tears its server down when it returns. SESSION_ID, SERVER_NAME, IMAGE_RO and
#      RUN_CAP_HOURS come from the launch environment.
#
#  (b) Stand-alone piggy-back (SR4 fallback, 2026-10-19; LAUNCH.md §6): a client of a
#      server that already runs for another session of the same rung. Give HOST_SID=<sid>
#      and the port, model and server name are read from ~/ro27_<HOST_SID>.job.log, or set
#      PORT, MODEL, SERVER_NAME yourself. Run it detached:
#        RP1_ROW=A18-003 HOST_SID=S1019g nohup bash ~/ro27_rp1_clients.sh > ~/ro27_a18_A18-003.job.log 2>&1 &
#      The host owns the server; if it is torn down mid-replay the calls become infra
#      errors (A18 §4 re-queue rule; rerun the target once).
#
# Steps (A18-T7):
#   1. docker run --rm --init --name ro27-<sid>-<row> --network host --cpus $RP1_CPUS with
#      the W&B key and the server key passed by name (never on the command line), image
#      rogue_one:<row image_tag> (the frozen wrapper the source prompts went through;
#      RP1_IMAGE overrides), the $RP1_SCRIPTS snapshot mounted read-only, and
#      replay_tool.py --validity a18 --seed 7042 --concurrency ${RP1_CONCURRENCY:-8}
#      --clip-max-tokens 131072 --clip-min-tokens 1024 --timeout 600, run name
#      A18__replay_<rung>__rp1set__f42__r0 (group A18, job_type replay_<rung>), replies
#      artifact rp1-replies-<rung>. A watchdog `docker stop`s it after RUN_CAP_HOURS.
#   2. Infra errors are re-queued once at the end with the same seed (inside replay_tool).
#   3. docker logs $SERVER_NAME (since the replay started) > ~/ro27_a18_logs/vllm_server_<sid>_<row>.log,
#      uploaded to the same W&B run (rp1_score.py attach-log, artifact rp1-server-log-<rung>).
#
# Env:
#   RP1_ROW (required)         A18-001 … A18-007 (A18-005 is dropped: SR4)
#   RP1_MANIFEST               default ~/ro27_manifests/A18_RP1.csv (SESSION_FILE is tried first)
#   HOST_SID                   mode (b): host session id (reads its job log)
#   PORT MODEL API_KEY         server (API_KEY default rogueone-nestedcv-key)
#   SERVER_NAME                vLLM container name (for its log)
#   SESSION_ID                 run tag / container name (default HOST_SID, else A18)
#   RP1_PROMPTS                default rp1-prompts:v0     RP1_CONCURRENCY  default 8
#   RP1_CPUS                   default 4                  RP1_IMAGE        default rogue_one:<row image_tag>
#   RUN_CAP_HOURS              default 6
#   RP1_SCRIPTS                scripts/agent2027 snapshot mounted into the container, default
#                              ~/ro27_a18_code/scripts/agent2027 (installed with `git archive FETCH_HEAD`,
#                              so ~/code/RougeOne is never checked out while sessions run)
#   RUNGS_CSV                  default $RP1_SCRIPTS/rungs.csv
#   WANDB_ENV_FILE             default /raid/home/henrbr16/code/PlanBench/.env
#   DRY_RUN=1                  print the three steps; no health check, no container
# Host-side tools only: bash, awk, grep, curl, docker. Never python on the host.
set -uo pipefail

DRY_RUN="${DRY_RUN:-0}"
ROW="${RP1_ROW:?RP1_ROW unset (A18-001 … A18-007)}"
RP1_SCRIPTS="${RP1_SCRIPTS:-$HOME/ro27_a18_code/scripts/agent2027}"
RUNGS_CSV="${RUNGS_CSV:-$RP1_SCRIPTS/rungs.csv}"
RP1_MANIFEST="${RP1_MANIFEST:-$HOME/ro27_manifests/A18_RP1.csv}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-/raid/home/henrbr16/code/PlanBench/.env}"
RP1_PROMPTS="${RP1_PROMPTS:-rp1-prompts:v0}"
RP1_CONCURRENCY="${RP1_CONCURRENCY:-8}"
RP1_CPUS="${RP1_CPUS:-4}"
RUN_CAP_HOURS="${RUN_CAP_HOURS:-6}"
API_KEY="${API_KEY:-rogueone-nestedcv-key}"
PROJECT="RogueOne-AGENT2027"
LOGDIR="$HOME/ro27_a18_logs"
WORKDIR="$HOME/ro27_a18_work"

log() { echo "[$(date -u +%FT%TZ)] [rp1 $ROW] $*"; }
fatal() { log "FATAL: $*"; echo "RP1_END $ROW rc=1"; exit 1; }

# --- the manifest row ------------------------------------------------------------------
row_line() {
  local f
  for f in "${SESSION_FILE:-}" "$RP1_MANIFEST"; do
    [ -n "$f" ] && [ -f "$f" ] || continue
    awk -F',' -v r="$ROW" 'NR > 1 { gsub(/["\r]/, ""); if ($1 == r) { print; exit } }' "$f" | grep . && return 0
  done
  return 1
}
LINE="$(row_line)" || fatal "row $ROW not found in ${SESSION_FILE:-<no SESSION_FILE>} or $RP1_MANIFEST"
grep -qi covtype <<<"$LINE" && fatal "covtype in row $ROW; refusing"
IFS=',' read -r _rid ABL ARM DATASET _tt _fs LLM_SEED _rep _sent _sid _ok IMAGE_TAG RUNG _rest <<<"$LINE"
[ "$ABL" = "A18" ] || fatal "row $ROW is not an A18 row (ablation_id=$ABL)"
[ "$DATASET" = "rp1set" ] || fatal "row $ROW: dataset_id=$DATASET, expected rp1set"
[ "$ARM" = "replay_$RUNG" ] || fatal "row $ROW: arm_id=$ARM does not match rung_S=$RUNG"
[ "$LLM_SEED" = "7042" ] || fatal "row $ROW: llm_seed=$LLM_SEED, expected 7042"

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
SESSION_ID="${SESSION_ID:-A18}"
[ -n "${PORT:-}" ] || fatal "PORT unset (give HOST_SID or PORT)"
[ -n "${MODEL:-}" ] || fatal "MODEL unset (give HOST_SID or MODEL)"

# the server must serve this row's rung (never replay_Q4 against a Q0.8 server)
want_model="$(awk -F',' -v r="$RUNG" 'NR > 1 && $1 == r { print $2; exit }' "$RUNGS_CSV" 2>/dev/null)"
[ -n "$want_model" ] || fatal "rung $RUNG not in $RUNGS_CSV"
[ "$want_model" = "$MODEL" ] || fatal "server model $MODEL is not rung $RUNG ($want_model)"

SID_LC="$(tr 'A-Z' 'a-z' <<<"$SESSION_ID")"
ROW_LC="$(tr 'A-Z' 'a-z' <<<"$ROW")"
CNAME="ro27-${SID_LC}-${ROW_LC}"
RUNG_ID="$(tr 'A-Z.' 'a-zp' <<<"$RUNG")"
RUN_ID="a18-${RUNG_ID}-${SID_LC}-$(date -u +%s)"
RUN_NAME="A18__${ARM}__rp1set__f42__r0"
IMAGE="${RP1_IMAGE:-rogue_one:${IMAGE_TAG}}"
OUT="$WORKDIR/replies_${RUNG}_${SESSION_ID}_${ROW}"
SRVLOG="$LOGDIR/vllm_server_${SESSION_ID}_${ROW}.log"
CAP_S=$(awk -v h="$RUN_CAP_HOURS" 'BEGIN { printf "%d", h * 3600 }')

REPLAY_ARGS=(--prompts "$RP1_PROMPTS" --prompts-project "$PROJECT" --project "$PROJECT"
  --model "$MODEL" --port "$PORT" --seed 7042 --concurrency "$RP1_CONCURRENCY"
  --clip-max-tokens 131072 --clip-min-tokens 1024 --timeout 600 --validity a18
  --wandb --ablation-id A18 --group A18 --arm-id "$ARM" --job-type "$ARM" --rung "$RUNG"
  --dataset-slot rp1set --task-type mixed --run-name "$RUN_NAME" --summary-prefix rp1
  --replies-artifact "rp1-replies-${RUNG}" --session-id "$SESSION_ID" --image-tag "$IMAGE_TAG"
  --tags "v2,${ROW}" --run-id "$RUN_ID" --out /out)
DOCKER_ARGS=(run --rm --init --name "$CNAME" --network host --cpus "$RP1_CPUS"
  -e WANDB_API_KEY -e WANDB_ENTITY=PlanBench -e "WANDB_PROJECT_NAME=$PROJECT" -e LLM_API_KEY
  -e ROGUE_ONE_LLM_SEED=7042 -e "ROGUE_ONE_IMAGE_TAG=$IMAGE_TAG"
  -v "$RP1_SCRIPTS:/workspace/scripts/agent2027:ro" -v "$OUT:/out"
  --entrypoint python3 "$IMAGE" /workspace/scripts/agent2027/replay_tool.py "${REPLAY_ARGS[@]}")
ATTACH_ARGS=(run --rm --name "${CNAME}-log" --cpus 1 -e WANDB_API_KEY
  -v "$RP1_SCRIPTS:/workspace/scripts/agent2027:ro" -v "$LOGDIR:/logs:ro"
  --entrypoint python3 "$IMAGE" /workspace/scripts/agent2027/analysis/rp1_score.py attach-log
  --run-id "$RUN_ID" --file "/logs/$(basename "$SRVLOG")" --artifact "rp1-server-log-${RUNG}"
  --project "$PROJECT")

log "plan: row=$ROW arm=$ARM rung=$RUNG model=$MODEL port=$PORT server=${SERVER_NAME:-?} session=$SESSION_ID image=$IMAGE concurrency=$RP1_CONCURRENCY cap_h=$RUN_CAP_HOURS run=$RUN_NAME id=$RUN_ID"
if [ "$DRY_RUN" = "1" ]; then
  log "DRY_RUN step 1: docker ${DOCKER_ARGS[*]}   (watchdog: docker stop $CNAME after ${CAP_S}s)"
  log "DRY_RUN step 2: infra errors re-queued once at the end, same seed (replay_tool.py)"
  log "DRY_RUN step 3: docker logs --since <start> ${SERVER_NAME:-<server>} > $SRVLOG; docker ${ATTACH_ARGS[*]}"
  echo "RP1_END $ROW rc=0 dry_run=1"
  exit 0
fi

# --- pre-flight: key, image, healthy server serving MODEL, no duplicate container -------
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$WANDB_ENV_FILE" ]; then
  WANDB_API_KEY=$(grep -E '^WANDB_API_KEY=' "$WANDB_ENV_FILE" | head -1 | cut -d= -f2- | tr -d '"'"'"'\r')
fi
[ -n "${WANDB_API_KEY:-}" ] || fatal "no WANDB_API_KEY (env or $WANDB_ENV_FILE); W&B is the system of record"
export WANDB_API_KEY
export LLM_API_KEY="$API_KEY"
docker image inspect "$IMAGE" >/dev/null 2>&1 || fatal "image $IMAGE not found"
grep -q -- '--validity' "$RP1_SCRIPTS/replay_tool.py" 2>/dev/null \
  && [ -f "$RP1_SCRIPTS/analysis/rp1_score.py" ] \
  || fatal "$RP1_SCRIPTS lacks the A18 replay_tool.py / rp1_score.py (install step, LAUNCH.md §6)"
code=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:${PORT}/health" || true)
[ "$code" = "200" ] || fatal "server on port $PORT not healthy (HTTP $code)"
curl -s -H "Authorization: Bearer $API_KEY" "http://localhost:${PORT}/v1/models" | grep -q "\"$MODEL\"" \
  || fatal "port $PORT does not serve $MODEL"
docker ps --format '{{.Names}}' | grep -qx "$CNAME" && fatal "container $CNAME already running"
mkdir -p "$LOGDIR" "$OUT" || fatal "cannot create $LOGDIR / $OUT"

# --- step 1 (+2 inside replay_tool): the replay container under a watchdog ----------------
START_ISO="$(date -u +%FT%TZ)"
log "START replay container=$CNAME image=$IMAGE (cap ${CAP_S}s)"
( sleep "$CAP_S"; log "CAP: ${CAP_S}s reached, docker stop $CNAME"; docker stop -t 30 "$CNAME" >/dev/null 2>&1 ) &
WATCHDOG=$!
docker "${DOCKER_ARGS[@]}"
rc=$?
kill "$WATCHDOG" 2>/dev/null; wait "$WATCHDOG" 2>/dev/null
log "replay container exited rc=$rc (replies: $OUT)"

# --- step 3: server log to ~/ro27_a18_logs and to the same W&B run ---------------------
if [ -n "${SERVER_NAME:-}" ]; then
  docker logs --since "$START_ISO" "$SERVER_NAME" > "$SRVLOG" 2>&1 || log "WARNING: docker logs $SERVER_NAME failed"
  if [ "$rc" = "0" ] && [ -s "$SRVLOG" ]; then
    docker "${ATTACH_ARGS[@]}" || log "WARNING: server log upload failed (scratch copy: $SRVLOG)"
  fi
else
  log "WARNING: SERVER_NAME unset; no server log"
fi
echo "RP1_END $ROW rc=$rc"
exit "$rc"
