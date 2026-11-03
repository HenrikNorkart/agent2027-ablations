#!/usr/bin/env bash
# AGENT 2027 clients script (E1c, 01 §5.3): runs every row of ONE session file against
# the vLLM server that ro27_vllm_session.sh (= paper/scripts/vllm_session.sh) owns.
#
# Installed as ~/ro27_clients_manifest.sh and normally invoked by
# ~/ro27_session_launch.sh -> ~/ro27_vllm_session.sh ~/ro27_clients_manifest.sh, which
# exports PORT, API_KEY and MODEL once the server is healthy. It BLOCKS until every row
# (and the retry pass) is done, so "clients finished" and "server torn down" stay the
# same event (vllm_session.sh's design).
#
# Per row, in order_key order, at most PARALLEL rows at a time:
#   1. ~/rogueone-runs-agent2027/<SESSION_ID>/<row_id>/tasks/<task_type>/<dataset_id>/
#      gets its own copy of config.yml, data.csv, labels.csv (outputs are written inside
#      the task folder, so two rows of one dataset would otherwise collide on output/).
#   2. env = v2_defaults.env (read from INSIDE the image, so the defaults are the ones the
#      image was built with) -> the row's env_overrides -> per-row keys (dataset, seeds,
#      task registry, models, ports, W&B project/entity) -> run-identity keys. Merged
#      last-wins into <row_dir>/env.list and passed with --env-file. Secrets
#      (WANDB_API_KEY, LLM_API_KEY) are passed by name (-e KEY) and never written to disk.
#   3. docker run -d --name ro27-<session_id>-<row_id> (lower-case) --network host
#      --ipc=host --cpus $CPUS, tasks mount at /workspace/tasks, Serper cache at
#      /workspace/cache, and only for the arms that need them the RT3 focus file / the
#      KT3 corpus. Explicit command `python3 -u /workspace/rogue_one_main.py`, so a crash
#      exits instead of hanging in the image's `|| tail -f /dev/null` default.
#   4. Watchdog: min(cap_hours, RUN_CAP_HOURS). On timeout `docker stop`, status CAP_HIT,
#      never retried.
#   5. Success = rc 0 AND fold_aggregation_summary.csv exists under the row's tasks dir
#      (clients_t16_rerun.sh lesson 3: rc=0 without it is a known silent failure).
#      Otherwise FAILED, queued for ONE solo retry after the concurrent pass.
#   6. One line per attempt: `ROW <row_id> attempt=<n> status=<OK|FAILED|CAP_HIT>
#      rc=<rc> wall=<s>s ...` on stdout (the vLLM session log) and, if RO27_JOB_LOG is
#      set, in the job log too; plus <RUNROOT>/<SESSION_ID>/status.tsv.
#
# Host-side tools only: docker, coreutils, awk, sed, grep, find, xargs. No python.
#
# Env (launcher-level keys, 01 §7 R4):
#   SESSION_ID (required)         SESSION_FILE (default ~/ro27_manifests/sessions/<SID>.csv)
#   IMAGE_RO   (default rogue_one:<row image_tag>; when set, every row's image_tag must match)
#   RUNROOT    (default ~/rogueone-runs-agent2027)   PARALLEL (6)   CPUS (12)
#   RUN_CAP_HOURS (6)  POLL_S (30)  STOP_TIMEOUT_S (120)
#   PORT, API_KEY, MODEL          from vllm_session.sh (the server this session owns)
#   ROGUEONE_CAIR_PATH (default ~/code/RougeOne)   TASKS_SRC (default <path>/tasks)
#   SERPER_CACHE_DIR (~/rogueone-serper-cache)     RT3_FOCUS_DIR (~/rogueone-rt3-focus)
#   CORPORA_DIR (<path>/scripts/agent2027/corpora)
#   WANDB_ENV_FILE (/raid/home/henrbr16/code/PlanBench/.env; WANDB_API_KEY in env wins)
#   WANDB_PROJECT_NAME (default by code phase: RogueOne-AGENT2027 for v2, else -prefreeze)
#   WANDB_ENTITY (PlanBench)      CODE_PHASE (default from image_tag/notes)
#   GIT_SHA, ROGUE_ONE_IMAGE_ID, ROGUE_ONE_VLLM_IMAGE_ID, PREREG_TAG  (identity metadata)
#   V2_DEFAULTS (a local file instead of the image's copy; DRY_RUN falls back to the repo)
#   RO27_JOB_LOG (also append ROW lines there)     DRY_RUN=1 (print only; no docker call)
set -uo pipefail

SESSION_ID="${SESSION_ID:?SESSION_ID is required (S<MMDD><a-z>)}"
SESSION_FILE="${SESSION_FILE:-$HOME/ro27_manifests/sessions/${SESSION_ID}.csv}"
RUNROOT="${RUNROOT:-$HOME/rogueone-runs-agent2027}"
PARALLEL="${PARALLEL:-6}"
CPUS="${CPUS:-12}"
RUN_CAP_HOURS="${RUN_CAP_HOURS:-6}"
POLL_S="${POLL_S:-30}"
STOP_TIMEOUT_S="${STOP_TIMEOUT_S:-120}"
IMAGE_RO="${IMAGE_RO:-}"
ROGUEONE_CAIR_PATH="${ROGUEONE_CAIR_PATH:-$HOME/code/RougeOne}"
TASKS_SRC="${TASKS_SRC:-$ROGUEONE_CAIR_PATH/tasks}"
SERPER_CACHE_DIR="${SERPER_CACHE_DIR:-$HOME/rogueone-serper-cache}"
RT3_FOCUS_DIR="${RT3_FOCUS_DIR:-$HOME/rogueone-rt3-focus}"
CORPORA_DIR="${CORPORA_DIR:-$ROGUEONE_CAIR_PATH/scripts/agent2027/corpora}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-/raid/home/henrbr16/code/PlanBench/.env}"
WANDB_ENTITY="${WANDB_ENTITY:-PlanBench}"
DRY_RUN="${DRY_RUN:-0}"
RO27_JOB_LOG="${RO27_JOB_LOG:-}"
V2_DEFAULTS="${V2_DEFAULTS:-}"
PORT="${PORT:-}"
API_KEY="${API_KEY:-}"
MODEL="${MODEL:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SESSDIR="$RUNROOT/$SESSION_ID"
PLAN="$SESSDIR/_plan"
STATUS="$SESSDIR/status.tsv"
RETRYQ="$SESSDIR/retry_queue.txt"
STARTED="$PLAN/started_containers.txt"
US=$'\x1f'   # field separator of the parsed plan (never appears in a manifest)

R7_HEADER="row_id,ablation_id,arm_id,dataset_id,task_type,fold_seed,llm_seed,replicate,is_sentinel,session_id,order_key,image_tag,rung_S,rung_E,rung_T,port_S,port_E,port_T,env_overrides,cap_hours,expected_u,notes"

log() { echo "[$(date -u +%FT%TZ)] $*"; }
row_log() {
  local line="[$(date -u +%FT%TZ)] $*"
  echo "$line"
  if [ -n "$RO27_JOB_LOG" ]; then echo "$line" >> "$RO27_JOB_LOG"; fi
}
lower() { printf '%s' "$1" | tr 'A-Z' 'a-z'; }

# R9 rung -> HF id (used for per-role routing of mixed rows; uniform rows use $MODEL).
rung_model() {
  case "$1" in
    Q0.8) echo "Qwen/Qwen3.5-0.8B" ;;  Q2) echo "Qwen/Qwen3.5-2B" ;;
    Q4) echo "Qwen/Qwen3.5-4B" ;;      Q9) echo "Qwen/Qwen3.5-9B" ;;
    Q122) echo "Qwen/Qwen3.5-122B-A10B-FP8" ;;
    O20) echo "openai/gpt-oss-20b" ;;  O120) echo "openai/gpt-oss-120b" ;;
    *) echo "" ;;
  esac
}

# --------------------------------------------------------------------------- parsing
# CSV (RFC 4180 quoting, no embedded newlines) -> one US-separated line per row in
# $PLAN/rows.usv, plus $PLAN/<row_id>.overrides (KEY=VALUE lines from env_overrides).
parse_session_file() {
  awk -v plan="$PLAN" -v want_header="$R7_HEADER" -v us="$US" '
    function parse_csv(line, out,    n, i, c, field, inq) {
      n = 0; field = ""; inq = 0
      for (i = 1; i <= length(line); i++) {
        c = substr(line, i, 1)
        if (inq) {
          if (c == "\"") {
            if (substr(line, i + 1, 1) == "\"") { field = field "\""; i++ }
            else inq = 0
          } else field = field c
        } else if (c == "\"") inq = 1
        else if (c == ",") { out[++n] = field; field = "" }
        else field = field c
      }
      out[++n] = field
      return n
    }
    function parse_overrides(rid, s,    file, key, val, m, tok) {
      file = plan "/" rid ".overrides"
      printf "" > file
      gsub(/^[ ]+|[ ]+$/, "", s)
      if (s == "" || s == "{}") { close(file); return 1 }
      if (substr(s, 1, 1) != "{" || substr(s, length(s), 1) != "}") return 0
      s = substr(s, 2, length(s) - 2)
      while (length(s) > 0) {
        if (!match(s, /^[ ]*"[A-Za-z0-9_]+"[ ]*:[ ]*("[^"]*"|[^,"]*)[ ]*(,|$)/)) return 0
        tok = substr(s, RSTART, RLENGTH); s = substr(s, RSTART + RLENGTH)
        match(tok, /"[A-Za-z0-9_]+"/); key = substr(tok, RSTART + 1, RLENGTH - 2)
        val = tok; sub(/^[ ]*"[A-Za-z0-9_]+"[ ]*:[ ]*/, "", val); sub(/[ ]*,?$/, "", val)
        if (substr(val, 1, 1) == "\"") val = substr(val, 2, length(val) - 2)
        if (val == "true") val = "True"; if (val == "false") val = "False"
        print key "=" val > file
      }
      close(file)
      return 1
    }
    {
      sub(/\r$/, "")
      if (NR == 1) {
        if ($0 != want_header) { print "FATAL: session file header is not the R7 header" > "/dev/stderr"; bad = 1; exit 3 }
        next
      }
      if ($0 ~ /^[ ]*$/) next
      n = parse_csv($0, f)
      if (n != 22) { printf "FATAL: line %d has %d fields, expected 22\n", NR, n > "/dev/stderr"; bad = 1; exit 3 }
      if (f[1] in seen) { printf "FATAL: duplicate row_id %s\n", f[1] > "/dev/stderr"; bad = 1; exit 3 }
      seen[f[1]] = 1
      if (!parse_overrides(f[1], f[19])) { printf "FATAL: %s env_overrides is not a flat JSON object: %s\n", f[1], f[19] > "/dev/stderr"; bad = 1; exit 3 }
      line = f[1]
      for (i = 2; i <= 22; i++) { v = f[i]; if (i == 19) v = "json"; line = line us v }
      print line
    }
  ' "$SESSION_FILE" > "$PLAN/rows.unsorted.usv" || return 1
  # order_key is column 11; sentinels are not reordered here (make_manifest decides).
  sort -t "$US" -k11,11n -k1,1 "$PLAN/rows.unsorted.usv" > "$PLAN/rows.usv"
}

# Fields of one row into the R_* globals.
load_row() {
  local rid="$1" line
  line="$(awk -F "$US" -v id="$rid" '$1 == id' "$PLAN/rows.usv")"
  [ -n "$line" ] || return 1
  IFS="$US" read -r R_ROW R_AID R_ARM R_DS R_TYPE R_FSEED R_LSEED R_REP R_SENT R_SID \
    R_OKEY R_TAG R_RS R_RE R_RT R_PS R_PE R_PT R_OV R_CAP R_EU R_NOTES <<< "$line"
  return 0
}

# min(row cap_hours, RUN_CAP_HOURS) in whole seconds (>= 1).
cap_seconds() {
  awk -v a="${1:-}" -v b="$RUN_CAP_HOURS" 'BEGIN {
    if (a == "") a = b; h = (a + 0 < b + 0) ? a + 0 : b + 0; s = int(h * 3600 + 0.5);
    if (s < 1) s = 1; print s }'
}

code_phase_for() {
  if [ -n "${CODE_PHASE:-}" ]; then echo "$CODE_PHASE"; return; fi
  case ",$R_NOTES," in *smoke*) echo "smoke"; return ;; esac
  case "$R_TAG" in
    *-pre.*) echo "prefreeze" ;;
    *-v2.*) echo "v2" ;;
    *) echo "prefreeze" ;;
  esac
}

project_for() {
  if [ -n "${WANDB_PROJECT_NAME:-}" ]; then echo "$WANDB_PROJECT_NAME"; return; fi
  case "$1" in v2) echo "RogueOne-AGENT2027" ;; *) echo "RogueOne-AGENT2027-prefreeze" ;; esac
}

image_for() {
  local want="rogue_one:$R_TAG"
  if [ -n "$IMAGE_RO" ]; then
    if [ -n "$R_TAG" ] && [ "$IMAGE_RO" != "$want" ]; then echo ""; return 1; fi
    echo "$IMAGE_RO"; return 0
  fi
  [ -n "$R_TAG" ] || { echo ""; return 1; }
  echo "$want"
}

# v2_defaults.env for an image, cached per image (cleaned: KEY=VALUE lines only).
defaults_for_image() {
  local img="$1" cache
  cache="$PLAN/v2_defaults.$(printf '%s' "$img" | tr -c 'A-Za-z0-9._-' '_').env"
  if [ ! -s "$cache" ]; then
    local src=""
    if [ -n "$V2_DEFAULTS" ]; then
      src="$V2_DEFAULTS"
    elif [ "$DRY_RUN" = "1" ]; then
      for c in "$SCRIPT_DIR/v2_defaults.env" "$ROGUEONE_CAIR_PATH/scripts/agent2027/v2_defaults.env"; do
        [ -f "$c" ] && { src="$c"; break; }
      done
    fi
    if [ -z "$src" ] && [ "$DRY_RUN" = "1" ]; then
      log "FATAL: DRY_RUN needs V2_DEFAULTS or a repo v2_defaults.env"; return 1
    fi
    local raw="$cache.raw.$$.$RANDOM"
    if [ -n "$src" ]; then
      [ -f "$src" ] || { log "FATAL: V2_DEFAULTS=$src does not exist"; return 1; }
      cat "$src" > "$raw"
    else
      docker run --rm --entrypoint cat "$img" /workspace/scripts/agent2027/v2_defaults.env \
        > "$raw" 2>/dev/null || { log "FATAL: $img has no /workspace/scripts/agent2027/v2_defaults.env"; rm -f "$raw"; return 1; }
    fi
    # Written to a private temp file and renamed, so concurrent rows never read half a file.
    tr -d '\r' < "$raw" | grep -E '^[A-Za-z_][A-Za-z0-9_]*=' > "$raw.clean" || true
    mv -f "$raw.clean" "$cache"
    rm -f "$raw"
    [ -s "$cache" ] || { log "FATAL: v2_defaults for $img is empty"; return 1; }
  fi
  echo "$cache"
}

# Merge KEY=VALUE files, last occurrence wins, first-seen order kept.
merge_env() {
  awk '{ i = index($0, "="); if (i < 2) next; k = substr($0, 1, i - 1);
         if (!(k in v)) o[++n] = k; v[k] = $0 }
       END { for (j = 1; j <= n; j++) print v[o[j]] }' "$@"
}

ov_get() {  # value of KEY in the row's overrides file (empty if absent)
  awk -v k="$1" 'index($0, k "=") == 1 { v = substr($0, length(k) + 2) } END { print v }' \
    "$PLAN/${R_ROW}.overrides"
}

record_status() {  # row attempt status rc wall start end container reason
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "$STATUS"
}

# --------------------------------------------------------------------------- one row
run_row() {
  local rid="$1" attempt="${2:-1}"
  load_row "$rid" || { row_log "ROW $rid attempt=$attempt status=FAILED rc=setup reason=row-not-in-plan"; return 0; }
  local name; name="$(lower "ro27-${SESSION_ID}-${rid}")"
  local rowdir="$SESSDIR/$rid"
  local t0; t0=$(date +%s)

  refuse() {  # reason (not retried: a setup fault does not go away on a retry)
    row_log "ROW $rid attempt=$attempt status=FAILED rc=setup wall=0s container=$name reason=$1"
    record_status "$rid" "$attempt" FAILED setup 0 "$t0" "$(date +%s)" "$name" "$1"
    return 0
  }

  [ "$R_SID" = "$SESSION_ID" ] || { refuse "session_id=$R_SID"; return 0; }
  case "$(lower "$R_DS")" in covtype) refuse "covtype-excluded"; return 0 ;; esac
  [ -n "$R_RS" ] && [ -n "$R_RE" ] && [ -n "$R_RT" ] || { refuse "no-rung(cpu/comparator row)"; return 0; }
  local image; image="$(image_for)" || { refuse "image_tag=$R_TAG vs IMAGE_RO=$IMAGE_RO"; return 0; }

  # Fresh row directory per attempt; the previous attempt is kept for the post-mortem.
  if [ -d "$rowdir" ] && [ "$DRY_RUN" != "1" ]; then
    mv "$rowdir" "$rowdir.attempt$((attempt - 1)).$(date +%s)" 2>/dev/null || true
  fi
  local tdir="$rowdir/tasks/$R_TYPE/$R_DS" src="$TASKS_SRC/$R_TYPE/$R_DS"
  if [ "$DRY_RUN" != "1" ]; then
    [ -f "$src/config.yml" ] && [ -f "$src/data.csv" ] || { refuse "missing-task:$src"; return 0; }
    mkdir -p "$tdir/output" "$rowdir/logs" || { refuse "mkdir:$tdir"; return 0; }
    cp "$src/config.yml" "$src/data.csv" "$tdir/" || { refuse "copy:$src"; return 0; }
    [ -f "$src/labels.csv" ] && cp "$src/labels.csv" "$tdir/"
    # output/ is pre-created so a config with the default ./output (SYN-P) resolves to
    # the task folder (config_dir), i.e. onto the mount, not to /workspace/output.
    printf -- '- /workspace/tasks/%s/%s/config.yml\n' "$R_TYPE" "$R_DS" > "$rowdir/tasks/task_registry.yml"
  else
    mkdir -p "$rowdir"
  fi

  # ---- env: defaults -> overrides -> per-row keys -> identity
  local defaults; defaults="$(defaults_for_image "$image")" || { refuse "no-v2-defaults"; return 0; }
  local phase; phase="$(code_phase_for)"
  local uniform=0
  [ "$R_RS" = "$R_RE" ] && [ "$R_RE" = "$R_RT" ] && [ "$R_PS" = "$R_PE" ] && [ "$R_PE" = "$R_PT" ] && uniform=1
  # Mixed rows: the global LLM_MODEL/LLM_PORT default to the Extractor's rung, unless the
  # row's env_overrides name a global (make_manifest admits the pair on mixed rows only:
  # A04 session C's "third model", A19's big rung). Written into env.perrow, which wins.
  local g_model="" g_port=""
  if [ "$uniform" != "1" ]; then
    g_model="$(ov_get LLM_MODEL)"; g_port="$(ov_get LLM_PORT)"
    [ -n "$g_model" ] && [ -n "$g_port" ] || { g_model="$(rung_model "$R_RE")"; g_port="$R_PE"; }
  fi
  local perrow="$rowdir/env.perrow"
  {
    echo "ROGUE_ONE_DATASETS=$R_DS"
    echo "ROGUE_ONE_FOLD_SEED=$R_FSEED"
    echo "ROGUE_ONE_LLM_SEED=$R_LSEED"
    echo "ROGUE_ONE_TASK_REGISTRY=/workspace/tasks/task_registry.yml"
    echo "ROGUE_ONE_START_VLLM=False"
    if [ "$uniform" = "1" ]; then
      # Uniform rows talk to the server this session owns ($PORT/$MODEL); the row's
      # planned port is only a planning field (TP2 siblings share manifests).
      echo "LLM_MODEL=${MODEL:-$(rung_model "$R_RE")}"
      echo "LLM_PORT=${PORT:-$R_PE}"
    else
      echo "LLM_MODEL=$g_model"
      echo "LLM_PORT=$g_port"
      echo "ROGUE_ONE_SCIENTIST_MODEL=$(rung_model "$R_RS")"
      echo "ROGUE_ONE_SCIENTIST_PORT=$R_PS"
      echo "ROGUE_ONE_EXTRACTOR_MODEL=$(rung_model "$R_RE")"
      echo "ROGUE_ONE_EXTRACTOR_PORT=$R_PE"
      echo "ROGUE_ONE_TESTER_MODEL=$(rung_model "$R_RT")"
      echo "ROGUE_ONE_TESTER_PORT=$R_PT"
    fi
    echo "WANDB_PROJECT_NAME=$(project_for "$phase")"
    echo "WANDB_ENTITY=$WANDB_ENTITY"
    echo "ROGUE_ONE_ABLATION_ID=$R_AID"
    echo "ROGUE_ONE_ARM_ID=$R_ARM"
    echo "ROGUE_ONE_REPLICATE=$R_REP"
    echo "ROGUE_ONE_SESSION_ID=$SESSION_ID"
    echo "ROGUE_ONE_MANIFEST_ROW=$R_ROW"
    echo "ROGUE_ONE_IS_SENTINEL=$R_SENT"
    echo "ROGUE_ONE_CODE_PHASE=$phase"
    echo "ROGUE_ONE_IMAGE_TAG=${R_TAG:-${image#*:}}"
    echo "ROGUE_ONE_GIT_SHA=${GIT_SHA:-}"
    if [ "$uniform" = "1" ]; then
      echo "ROGUE_ONE_SERVER_TOPOLOGY=uniform:${R_RE}@${PORT:-$R_PE}"
    else
      echo "ROGUE_ONE_SERVER_TOPOLOGY=mixed:S=${R_RS}@${R_PS};E=${R_RE}@${R_PE};T=${R_RT}@${R_PT};G=${g_model}@${g_port}"
    fi
    echo "ROGUE_ONE_RUNG_S=$R_RS"
    echo "ROGUE_ONE_RUNG_E=$R_RE"
    echo "ROGUE_ONE_RUNG_T=$R_RT"
    echo "ROGUE_ONE_IMAGE_ID=${ROGUE_ONE_IMAGE_ID:-}"
    echo "ROGUE_ONE_VLLM_IMAGE_ID=${ROGUE_ONE_VLLM_IMAGE_ID:-}"
    echo "ROGUE_ONE_PREREG_TAG=${PREREG_TAG:-}"
    echo "ROGUE_ONE_PARALLEL=$PARALLEL"
  } > "$perrow"

  # ---- arm-specific mounts (RT3 focus file, KT3 corpus): only when the row asks.
  local -a mounts=(-v "$rowdir/tasks:/workspace/tasks" -v "$SERPER_CACHE_DIR:/workspace/cache")
  local planner; planner="$(ov_get ROGUE_ONE_PLANNER_MODE)"
  if [ "$planner" = "oracle" ] || [ "$planner" = "misdirected" ]; then
    local focus="$RT3_FOCUS_DIR/${rid}.json"
    [ "$DRY_RUN" = "1" ] || [ -f "$focus" ] || { refuse "missing-focus-file:$focus"; return 0; }
    mounts+=(-v "$focus:/workspace/focus/focus.json:ro")
    [ -n "$(ov_get ROGUE_ONE_PLANNER_FOCUS_FILE)" ] || echo "ROGUE_ONE_PLANNER_FOCUS_FILE=/workspace/focus/focus.json" >> "$perrow"
  fi
  if [ "$(ov_get ROGUE_ONE_KNOWLEDGE_BACKEND)" = "local" ]; then
    local corpus; corpus="$(ov_get ROGUE_ONE_LOCAL_CORPUS)"
    [ -n "$corpus" ] || { refuse "local-backend-without-ROGUE_ONE_LOCAL_CORPUS"; return 0; }
    [ "$DRY_RUN" = "1" ] || [ -d "$CORPORA_DIR/$corpus" ] || { refuse "missing-corpus:$CORPORA_DIR/$corpus"; return 0; }
    mounts+=(-v "$CORPORA_DIR/$corpus:/workspace/corpora/$corpus:ro")
  fi

  merge_env "$defaults" "$PLAN/${R_ROW}.overrides" "$perrow" > "$rowdir/env.list"
  rm -f "$perrow"

  local cap_s; cap_s="$(cap_seconds "$R_CAP")"
  local -a cmd=(docker run -d --name "$name" --network host --ipc=host --cpus "$CPUS"
    --env-file "$rowdir/env.list" -e WANDB_API_KEY -e LLM_API_KEY
    "${mounts[@]}" "$image" python3 -u /workspace/rogue_one_main.py)

  if [ "$DRY_RUN" = "1" ]; then
    # One write per row, so the block of a row is never interleaved with another's.
    printf '%s\n  cmd: %s\n%s\n' \
      "[$(date -u +%FT%TZ)] DRYRUN ROW $rid order_key=$R_OKEY sentinel=$R_SENT container=$name image=$image cap_s=$cap_s" \
      "${cmd[*]}" "$(sed 's/^/  env: /' "$rowdir/env.list")"
    record_status "$rid" "$attempt" DRY - 0 "$t0" "$t0" "$name" dry-run
    return 0
  fi

  # ---- ownership guard: never remove a container that is not this row's.
  if docker ps -a --format '{{.Names}}' | grep -qx "$name"; then
    local owner
    owner="$(docker inspect "$name" --format '{{range .Mounts}}{{.Source}} {{end}}' 2>/dev/null || true)"
    case " $owner " in
      *" $rowdir/tasks "*|*" $rowdir/tasks/ "*) docker rm -f "$name" >/dev/null 2>&1 || true ;;
      *) refuse "container-name-owned-elsewhere:$owner"; return 0 ;;
    esac
  fi

  export LLM_API_KEY="$API_KEY"
  row_log "START ROW $rid attempt=$attempt container=$name image=$image dataset=$R_DS arm=$R_ARM cap_s=$cap_s"
  if ! "${cmd[@]}" > "$rowdir/logs/docker_run.out" 2>&1; then
    local t1; t1=$(date +%s)
    row_log "ROW $rid attempt=$attempt status=FAILED rc=docker-run wall=$((t1 - t0))s container=$name"
    record_status "$rid" "$attempt" FAILED docker-run $((t1 - t0)) "$t0" "$t1" "$name" docker-run-failed
    [ "$attempt" = "1" ] && echo "$rid" >> "$RETRYQ"
    return 0
  fi
  echo "$name" >> "$STARTED"

  local cap_hit=0 now
  while :; do
    [ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" = "true" ] || break
    now=$(date +%s)
    if [ $((now - t0)) -ge "$cap_s" ]; then
      cap_hit=1
      row_log "CAP ROW $rid: ${cap_s}s cap reached, docker stop $name"
      docker stop -t "$STOP_TIMEOUT_S" "$name" >/dev/null 2>&1 || true
      break
    fi
    sleep "$POLL_S"
  done
  local rc; rc="$(docker inspect -f '{{.State.ExitCode}}' "$name" 2>/dev/null || echo unknown)"
  docker logs "$name" > "$rowdir/logs/container.log" 2>&1 || true
  docker rm "$name" >/dev/null 2>&1 || true
  local t1; t1=$(date +%s)
  local wall=$((t1 - t0))
  local summary; summary="$(find "$rowdir/tasks" -name fold_aggregation_summary.csv 2>/dev/null | head -1)"

  # A run whose every LLM call failed still completes on the deterministic fallbacks and
  # writes a summary (S1012a, 2026-10-12: openai/pydantic drift broke usage parsing, 0
  # LLM calls, fallback_share 1.0, reported OK). E3's summary records the call count, so
  # zero logical LLM calls is a failure, not a result. A row without the file is not judged.
  local instr llm_calls=""
  instr="$(find "$rowdir/tasks" -name instrumentation_summary.json 2>/dev/null | head -1)"
  [ -n "$instr" ] && llm_calls="$(grep -oE '"llm_calls_total": *[0-9]+' "$instr" | grep -oE '[0-9]+$' || true)"

  local status reason=""
  if [ "$cap_hit" = "1" ]; then
    status=CAP_HIT; reason="cap_hit"
  elif [ "$rc" = "0" ] && [ -n "$summary" ] && [ "$llm_calls" = "0" ]; then
    status=FAILED; reason="no-llm-calls"
  elif [ "$rc" = "0" ] && [ -n "$summary" ]; then
    status=OK
  else
    status=FAILED
    if [ "$rc" = "0" ]; then reason="rc0-no-fold_aggregation_summary"; else reason="rc=$rc"; fi
  fi
  row_log "ROW $rid attempt=$attempt status=$status rc=$rc wall=${wall}s container=$name${reason:+ reason=$reason}"
  record_status "$rid" "$attempt" "$status" "$rc" "$wall" "$t0" "$t1" "$name" "${reason:-ok}"
  if [ "$status" = "FAILED" ]; then
    tail -6 "$rowdir/logs/container.log" 2>/dev/null | sed 's/^/    /'
    [ "$attempt" = "1" ] && echo "$rid" >> "$RETRYQ"
  fi
  return 0
}

# --------------------------------------------------------------------------- session
stop_ours() {
  [ -f "$STARTED" ] || return 0
  log "signal: stopping this session's row containers"
  while read -r n; do
    docker ps --format '{{.Names}}' | grep -qx "$n" && docker stop -t 30 "$n" >/dev/null 2>&1
  done < "$STARTED"
}
if [ "$DRY_RUN" != "1" ]; then trap 'stop_ours; exit 143' INT TERM; fi

[ -f "$SESSION_FILE" ] || { log "FATAL: SESSION_FILE=$SESSION_FILE does not exist"; exit 1; }
mkdir -p "$PLAN" "$SERPER_CACHE_DIR" 2>/dev/null || mkdir -p "$PLAN"
: > "$RETRYQ"
[ -f "$STATUS" ] || printf 'row_id\tattempt\tstatus\trc\twall_s\tstart_epoch\tend_epoch\tcontainer\treason\n' > "$STATUS"
cp "$SESSION_FILE" "$PLAN/session.csv"
parse_session_file || { log "FATAL: could not parse $SESSION_FILE"; exit 1; }
NROWS=$(wc -l < "$PLAN/rows.usv" | tr -d ' ')
[ "$NROWS" -gt 0 ] || { log "FATAL: $SESSION_FILE has no rows"; exit 1; }

# Every row must belong to this session, and container names must be unique (they are
# lower-cased, so two row ids differing only in case would collide).
bad_sid=$(awk -F "$US" -v s="$SESSION_ID" '$10 != s {print $1}' "$PLAN/rows.usv" | tr '\n' ' ')
[ -z "${bad_sid// /}" ] || { log "FATAL: rows not in session $SESSION_ID: $bad_sid"; exit 1; }
dups=$(cut -d "$US" -f1 "$PLAN/rows.usv" | tr 'A-Z' 'a-z' | sort | uniq -d | tr '\n' ' ')
[ -z "${dups// /}" ] || { log "FATAL: container-name collision for rows: $dups"; exit 1; }
covt=$(awk -F "$US" 'tolower($4) == "covtype" {print $1}' "$PLAN/rows.usv" | tr '\n' ' ')
[ -z "${covt// /}" ] || { log "FATAL: covtype rows are excluded from AGENT 2027 and every other Rogue One study (Henrik, 2026-10-16): $covt"; exit 1; }

if [ "$DRY_RUN" != "1" ]; then
  if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$WANDB_ENV_FILE" ]; then
    WANDB_API_KEY=$(grep -E '^WANDB_API_KEY=' "$WANDB_ENV_FILE" | head -1 | cut -d= -f2- | tr -d '"'"'"'\r')
  fi
  [ -n "${WANDB_API_KEY:-}" ] || { log "FATAL: no WANDB_API_KEY (env or $WANDB_ENV_FILE); W&B is the system of record"; exit 1; }
fi
export WANDB_API_KEY="${WANDB_API_KEY:-}"

# Resolve each image's v2_defaults.env once, before any row starts.
if [ -n "$IMAGE_RO" ]; then
  images="$IMAGE_RO"
else
  images=$(awk -F "$US" '$12 != "" { print "rogue_one:" $12 }' "$PLAN/rows.usv" | sort -u)
fi
for img in $images; do
  d="$(defaults_for_image "$img")" || { log "FATAL: no v2_defaults.env for $img"; exit 1; }
  log "v2 defaults for $img: $(wc -l < "$d" | tr -d ' ') keys ($d)"
done

log "=== RO27 clients: session=$SESSION_ID rows=$NROWS parallel=$PARALLEL cpus=$CPUS cap_h=$RUN_CAP_HOURS image=${IMAGE_RO:-per-row} port=${PORT:-per-row} dry_run=$DRY_RUN ==="
log "run order (order_key from make_manifest.py --plan-session; the RNG seed is in the session's .plan.txt):"
awk -F "$US" '{ printf "  %s\t%s\t%s\t%s\t%s%s\n", $11, $1, $3, $4, "ro27-" sid "-", tolower($1) }' sid="$(lower "$SESSION_ID")" "$PLAN/rows.usv"
sentinels=$(awk -F "$US" '$9 == "1" {print $1 "(order_key " $11 ")"}' "$PLAN/rows.usv" | tr '\n' ' ')
log "sentinel slot: ${sentinels:-none}"

export -f run_row load_row cap_seconds code_phase_for project_for image_for defaults_for_image \
  merge_env ov_get record_status row_log log lower rung_model
export SESSION_ID SESSION_FILE RUNROOT PARALLEL CPUS RUN_CAP_HOURS POLL_S STOP_TIMEOUT_S IMAGE_RO \
  ROGUEONE_CAIR_PATH TASKS_SRC SERPER_CACHE_DIR RT3_FOCUS_DIR CORPORA_DIR WANDB_ENTITY DRY_RUN \
  RO27_JOB_LOG V2_DEFAULTS PORT API_KEY MODEL SCRIPT_DIR SESSDIR PLAN STATUS RETRYQ STARTED US

# --- pass 1: concurrent, in order_key order -----------------------------------------
# Backgrounded + wait, so a TERM to this script runs the trap (stop our containers)
# immediately instead of after xargs returns.
NP="$PARALLEL"; [ "$DRY_RUN" = "1" ] && NP=1   # print-only: keep the output in row order
{ cut -d "$US" -f1 "$PLAN/rows.usv" \
    | xargs -P "$NP" -I{} bash -c 'run_row "$1" 1' _ {}; } &
wait $!
log "=== concurrent pass done ==="

# --- pass 2: every crash / no-result gets ONE solo attempt (CAP_HIT is never retried) --
RETRY=$(sort -u "$RETRYQ" 2>/dev/null | tr '\n' ' ')
if [ -n "${RETRY// /}" ]; then
  log "=== retry pass (solo): $RETRY==="
  for rid in $RETRY; do run_row "$rid" 2; done
else
  log "=== no first-pass failures, no retry pass ==="
fi

# --- summary: final status = the row's last attempt ----------------------------------
final=$(awk -F '\t' 'NR > 1 { last[$1] = $3 } END { for (r in last) print r "\t" last[r] }' "$STATUS" \
  | awk -F '\t' -v plan="$PLAN/rows.usv" -v us="$US" 'BEGIN { while ((getline l < plan) > 0) { split(l, a, us); want[a[1]] = 1 } }
      ($1 in want) { print }')
n_ok=$(printf '%s\n' "$final" | grep -c $'\tOK$' || true)
n_cap=$(printf '%s\n' "$final" | grep -c $'\tCAP_HIT$' || true)
n_fail=$(printf '%s\n' "$final" | grep -c $'\tFAILED$' || true)
n_dry=$(printf '%s\n' "$final" | grep -c $'\tDRY$' || true)
still=$(printf '%s\n' "$final" | awk -F '\t' '$2 == "FAILED" {print $1}' | sort | tr '\n' ' ')
caps=$(printf '%s\n' "$final" | awk -F '\t' '$2 == "CAP_HIT" {print $1}' | sort | tr '\n' ' ')
row_log "=== RO27_CLIENTS_DONE session=$SESSION_ID ok=$n_ok failed=$n_fail cap_hit=$n_cap dry=$n_dry failed_rows=[${still% }] cap_hit_rows=[${caps% }] ==="
[ "$n_fail" = "0" ] && [ "$n_cap" = "0" ] && exit 0
exit 2
