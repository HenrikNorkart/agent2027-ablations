#!/usr/bin/env bash
# A04 Stage-0 clients script (A04-T7), installed as ~/ro27_stage0_clients.sh.
# ~/ro27_session_launch.sh ... CLIENTS_SCRIPT=$HOME/ro27_stage0_clients.sh runs it once the
# session's vLLM server is healthy, with PORT, API_KEY, MODEL, SERVER_NAME, SESSION_ID,
# SESSION_FILE and IMAGE_RO exported. It BLOCKS until its work is done, then:
#   1. (check b) runs X2's replay tool on the frozen stage0-prompts set against this server,
#      in container ro27-<sid>-replay50 (skipped with a logged line when STAGE0_REPLAY=0 or
#      the tool / prompt file is absent; A04 then replays in a later boot of the same rung);
#      STAGE0_KV_PROBE=1 (Q122 only) also sends the 6 longest prompts concurrently;
#   2. (check c) runs ~/ro27_clients_manifest.sh on $SESSION_FILE (the balance-scale fold);
#   3. dumps the server log to ~/ro27_a04_logs/vllm_server_<SID>_<rung>.log before the
#      session tears the server down (vllm_session.sh's exit removes it, log included).
# Env: STAGE0_RUNG (row's rung, for names), STAGE0_ARM (arm_id; default stage0_<rung>),
#      STAGE0_REPLAY (1), STAGE0_KV_PROBE (0), STAGE0_PROMPTS (~/ro27_a04_prompts/stage0-prompts-v0.jsonl),
#      REPLAY_DIR (~/code/RougeOne/scripts/agent2027; holds X2's replay_tool.py, mounted read-only
#      over /workspace/scripts/agent2027), REPLAY_PROJECT (RogueOne-AGENT2027-prefreeze),
#      WANDB_ENV_FILE (/raid/home/henrbr16/code/PlanBench/.env), CLIENTS_MANIFEST (~/ro27_clients_manifest.sh),
#      DRY_RUN=1 prints the three steps with resolved names and runs nothing.
set -uo pipefail

SID="${SESSION_ID:?SESSION_ID unset}"
SID_LC="$(printf '%s' "$SID" | tr 'A-Z' 'a-z')"
RUNG="${STAGE0_RUNG:?STAGE0_RUNG unset (e.g. Q0.8)}"
RUNG_LC="$(printf '%s' "$RUNG" | tr 'A-Z' 'a-z')"
ARM="${STAGE0_ARM:-stage0_${RUNG}}"
REPLAY="${STAGE0_REPLAY:-1}"
KV_PROBE="${STAGE0_KV_PROBE:-0}"
PROMPTS="${STAGE0_PROMPTS:-$HOME/ro27_a04_prompts/stage0-prompts-v0.jsonl}"
REPLAY_DIR="${REPLAY_DIR:-$HOME/code/RougeOne/scripts/agent2027}"
REPLAY_TOOL="$REPLAY_DIR/replay_tool.py"
REPLAY_PROJECT="${REPLAY_PROJECT:-RogueOne-AGENT2027-prefreeze}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-/raid/home/henrbr16/code/PlanBench/.env}"
CLIENTS_MANIFEST="${CLIENTS_MANIFEST:-$HOME/ro27_clients_manifest.sh}"
LOGDIR="$HOME/ro27_a04_logs"
REPLAY_OUT="$LOGDIR/replay_${SID}_${RUNG}"
REPLAY_NAME="ro27-${SID_LC}-replay50"
SERVER_LOG="$LOGDIR/vllm_server_${SID}_${RUNG}.log"
DRY="${DRY_RUN:-0}"

say() {
  local line="[$(date -u +%FT%TZ)] [stage0 $SID $RUNG] $*"
  echo "$line"
  [ -n "${RO27_JOB_LOG:-}" ] && echo "$line" >> "$RO27_JOB_LOG"
  return 0
}

mkdir -p "$LOGDIR" "$REPLAY_OUT"

dump_server_log() {
  if [ "$DRY" = "1" ]; then say "DRY_RUN: step 3: docker logs ${SERVER_NAME:-?} > $SERVER_LOG"; return; fi
  docker logs "${SERVER_NAME:?SERVER_NAME unset}" > "$SERVER_LOG" 2>&1 \
    && say "step 3: server log -> $SERVER_LOG ($(wc -l < "$SERVER_LOG") lines)" \
    || say "step 3: WARNING could not dump the log of $SERVER_NAME"
}
trap dump_server_log EXIT

# --- step 1: replay (check b) --------------------------------------------------------------
replay_cmd=(docker run --rm --name "$REPLAY_NAME" --network host --cpus 4 --memory 8g
  --env-file "$WANDB_ENV_FILE"
  -e LLM_MODEL="${MODEL:-}" -e LLM_PORT="${PORT:-}" -e LLM_API_KEY="${API_KEY:-}" -e ROGUE_ONE_LLM_SEED=42
  -v "$REPLAY_DIR":/workspace/scripts/agent2027:ro
  -v "$PROMPTS":/in/prompts.jsonl:ro -v "$REPLAY_OUT":/out
  --entrypoint python3 "${IMAGE_RO:-rogue_one:agent2027-pre.3}"
  /workspace/scripts/agent2027/replay_tool.py --prompts /in/prompts.jsonl
  --model "${MODEL:-}" --port "${PORT:-}" --seed 42 --concurrency 1
  --arm-id "$ARM" --rung "$RUNG" --session-id "$SID" --image-tag "${IMAGE_RO#*:}"
  --project "$REPLAY_PROJECT" --wandb)
if [ "$REPLAY" != "1" ]; then
  say "step 1: replay skipped (STAGE0_REPLAY=$REPLAY)"
elif [ ! -f "$REPLAY_TOOL" ] || [ ! -f "$PROMPTS" ]; then
  say "step 1: replay skipped (tool $REPLAY_TOOL or prompts $PROMPTS missing); replay in a later boot"
elif [ "$DRY" = "1" ]; then
  shown="${replay_cmd[*]}"; [ -n "${API_KEY:-}" ] && shown="${shown//$API_KEY/<key>}"
  say "DRY_RUN: step 1: $shown --out /out/replay50"
  [ "$KV_PROBE" = "1" ] && say "DRY_RUN: step 1b: same container, --kv-probe 6"
else
  say "step 1: replay50 -> $REPLAY_OUT"
  "${replay_cmd[@]}" --out /out/replay50 > "$REPLAY_OUT/replay.log" 2>&1; say "step 1: replay rc=$?"
  if [ "$KV_PROBE" = "1" ]; then
    "${replay_cmd[@]}" --out /out/kvprobe6 --kv-probe 6 > "$REPLAY_OUT/kv_probe.log" 2>&1; say "step 1b: kv probe rc=$?"
  fi
fi

# --- step 2: the fold-run (check c) ----------------------------------------------------------
if [ "$DRY" = "1" ]; then
  say "DRY_RUN: step 2: bash $CLIENTS_MANIFEST (SESSION_FILE=${SESSION_FILE:-?})"
  DRY_RUN=1 bash "$CLIENTS_MANIFEST"; crc=$?
else
  say "step 2: bash $CLIENTS_MANIFEST"
  bash "$CLIENTS_MANIFEST"; crc=$?
  say "step 2: clients rc=$crc"
fi
exit "$crc"
