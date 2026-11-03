#!/usr/bin/env bash
# AGENT 2027 detached session wrapper (E1c, 01 §5.2 step 4). Installed as
# ~/ro27_session_launch.sh and launched with nohup, its stdout being the job log:
#
#   SESSION_ID=S1016a SESSION_FILE=~/ro27_manifests/sessions/S1016a.csv GPUS=0,1,2,3 \
#   MODEL=openai/gpt-oss-120b PORT=8010 SERVER_NAME=rogueone-vllm-o120-s1016a \
#   LOG=~/vllm_session_S1016a.log IMAGE_RO=rogue_one:agent2027-v2.0 PARALLEL=6 CPUS=12 \
#   RUN_CAP_HOURS=6 RUNROOT=~/rogueone-runs-agent2027 WAIT_MAX_MIN=0 \
#   nohup bash ~/ro27_session_launch.sh > ~/ro27_S1016a.job.log 2>&1 &
#
# It:
#   1. refuses to start when this session's containers already exist, when GPUS is not
#      explicit (01 §5.1 rule 2; ALLOW_GPU_AUTO=1 re-enables vllm_session.sh's auto-wait),
#      or when the client image carries a /workspace/.dev.env (config.py loads it with
#      override=True, so it would silently beat every docker -e flag);
#   2. asserts the vLLM version in-container and logs both image IDs (t17_rerun_launch.sh);
#   3. starts the energy sidecar ro27-<session_id>-energy (E3 energy_sampler.py, W&B run
#      <SESSION_ID>__energy) and stops it on exit (EXIT trap; SIGTERM lets it upload);
#   4. calls  ~/ro27_vllm_session.sh $CLIENTS_SCRIPT  (default ~/ro27_clients_manifest.sh),
#      which owns the server lifecycle and tears it down on exit;
#   5. ends with  === RO27_SESSION_END <SESSION_ID> rc=<rc> ===.
#
# Host-side tools only: docker, nvidia-smi (via vllm_session.sh), curl (idem), coreutils.
#
# Env (all exported to the session and clients scripts):
#   SESSION_ID (required), SESSION_FILE (default ~/ro27_manifests/sessions/<SID>.csv)
#   GPUS (explicit ids), NEED, MODEL, PORT (8010), API_KEY, IMAGE (vLLM, v0.22.0),
#   SERVER_NAME (default rogueone-vllm-<rung>-<sid>, rung from the session's first row),
#   LOG (~/vllm_session_<SID>.log), WAIT_MAX_MIN (0), EAGER (0), EXTRA_ARGS, MAX_MODEL_LEN,
#   ROGUEONE_ALLOW_SECOND_SERVER, IMAGE_RO (required), PARALLEL, CPUS, RUN_CAP_HOURS,
#   RUNROOT, WANDB_PROJECT_NAME, CODE_PHASE, GIT_SHA (default: git rev-parse in
#   ROGUEONE_CAIR_PATH), PREREG_TAG
#   VLLM_SESSION (~/ro27_vllm_session.sh), CLIENTS_SCRIPT (~/ro27_clients_manifest.sh)
#   EXPECT_VLLM (0.22.0; empty = log only), ENERGY (1), ENERGY_DIR (~/rogueone-energy)
#   JOB_LOG (~/ro27_<SID>.job.log; exported as RO27_JOB_LOG so per-row lines land here too)
#   DRY_RUN=1: print the plan and run the clients script with DRY_RUN=1, no server,
#   no sidecar, no docker call from this script.
set -uo pipefail

SESSION_ID="${SESSION_ID:?SESSION_ID is required (S<MMDD><a-z>)}"
SID_LC="$(printf '%s' "$SESSION_ID" | tr 'A-Z' 'a-z')"
export SESSION_ID
export SESSION_FILE="${SESSION_FILE:-$HOME/ro27_manifests/sessions/${SESSION_ID}.csv}"
export IMAGE="${IMAGE:-vllm/vllm-openai:v0.22.0}"
export MODEL="${MODEL:-openai/gpt-oss-120b}"
export PORT="${PORT:-8010}"
export GPUS="${GPUS:-}"
export LOG="${LOG:-$HOME/vllm_session_${SESSION_ID}.log}"
export WAIT_MAX_MIN="${WAIT_MAX_MIN:-0}"
export EAGER="${EAGER:-0}"
export EXTRA_ARGS="${EXTRA_ARGS:-}" MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
export RUNROOT="${RUNROOT:-$HOME/rogueone-runs-agent2027}"
export PARALLEL="${PARALLEL:-6}"
export CPUS="${CPUS:-12}"
export RUN_CAP_HOURS="${RUN_CAP_HOURS:-6}"
export ROGUEONE_CAIR_PATH="${ROGUEONE_CAIR_PATH:-$HOME/code/RougeOne}"
VLLM_SESSION="${VLLM_SESSION:-$HOME/ro27_vllm_session.sh}"
CLIENTS_SCRIPT="${CLIENTS_SCRIPT:-$HOME/ro27_clients_manifest.sh}"
EXPECT_VLLM="${EXPECT_VLLM-0.22.0}"
ENERGY="${ENERGY:-1}"
ENERGY_DIR="${ENERGY_DIR:-$HOME/rogueone-energy}"
ENERGY_NAME="ro27-${SID_LC}-energy"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-/raid/home/henrbr16/code/PlanBench/.env}"
export RO27_JOB_LOG="${JOB_LOG:-$HOME/ro27_${SESSION_ID}.job.log}"
DRY_RUN="${DRY_RUN:-0}"

say() { echo "[$(date -u +%FT%TZ)] $*"; }
RC=1
end_marker() { say "=== RO27_SESSION_END ${SESSION_ID} rc=${RC} ==="; }

# Server name: rogueone-vllm-<rung>-<session_id>, rung from the first data row's rung_E.
if [ -z "${SERVER_NAME:-}" ] && [ -f "$SESSION_FILE" ]; then
  rung=$(awk -F',' 'NR == 2 { print $14; exit }' "$SESSION_FILE" | tr -d '\r"' | tr 'A-Z' 'a-z')
  [ -n "$rung" ] && SERVER_NAME="rogueone-vllm-${rung}-${SID_LC}"
fi
export SERVER_NAME="${SERVER_NAME:-rogueone-vllm-${SID_LC}}"

say "=== RO27 session $SESSION_ID: job start (dry_run=$DRY_RUN) ==="
say "session file=$SESSION_FILE rows=$( [ -f "$SESSION_FILE" ] && echo $(( $(wc -l < "$SESSION_FILE") - 1 )) || echo missing)"
say "server=$SERVER_NAME model=$MODEL port=$PORT gpus=${GPUS:-unset} vllm image=$IMAGE"
say "clients=$CLIENTS_SCRIPT image_ro=${IMAGE_RO:-unset} parallel=$PARALLEL cpus=$CPUS cap_h=$RUN_CAP_HOURS runroot=$RUNROOT"

fatal() { say "FATAL: $*"; RC=1; end_marker; exit 1; }

[ -f "$SESSION_FILE" ] || fatal "SESSION_FILE=$SESSION_FILE does not exist"
[ -n "${IMAGE_RO:-}" ] || fatal "IMAGE_RO is required (rogue_one:<image_tag>)"
export IMAGE_RO
if [ -z "$GPUS" ] || [ "$GPUS" = "auto" ]; then
  if [ "${ALLOW_GPU_AUTO:-0}" = "1" ]; then
    export GPUS=auto
    say "WARNING: GPUS=auto (ALLOW_GPU_AUTO=1): vllm_session.sh waits up to ${WAIT_MAX_MIN} min for free GPUs"
  else
    fatal "GPUS must list explicit ids from the pre-flight (01 §5.1 rule 2)"
  fi
fi
if [ -z "${GIT_SHA:-}" ] && [ -d "$ROGUEONE_CAIR_PATH/.git" ]; then
  GIT_SHA=$(git -C "$ROGUEONE_CAIR_PATH" rev-parse HEAD 2>/dev/null || true)
  say "GIT_SHA from the checkout at $ROGUEONE_CAIR_PATH (not read from the image): ${GIT_SHA:-unknown}"
fi
export GIT_SHA="${GIT_SHA:-}"
export PREREG_TAG="${PREREG_TAG:-}" CODE_PHASE="${CODE_PHASE:-}" WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-}"

if [ "$DRY_RUN" = "1" ]; then
  say "DRY_RUN: would assert vLLM ${EXPECT_VLLM:-any} in $IMAGE, check $IMAGE_RO for /workspace/.dev.env,"
  say "DRY_RUN: start $ENERGY_NAME (energy=$ENERGY) and run: bash $VLLM_SESSION $CLIENTS_SCRIPT"
  DRY_RUN=1 bash "$CLIENTS_SCRIPT"
  RC=$?
  end_marker
  exit "$RC"
fi

# --- 1. collisions: one launch per session id ----------------------------------------
ours=$(docker ps -a --format '{{.Names}}' | grep -E "^ro27-${SID_LC}-" || true)
[ -z "$ours" ] || fatal "containers of session $SESSION_ID already exist: $(echo $ours)"

# --- 2. image checks -----------------------------------------------------------------
IMAGE_ID=$(docker image inspect "$IMAGE_RO" --format '{{.Id}}' 2>/dev/null) || fatal "client image $IMAGE_RO not found"
VLLM_IMAGE_ID=$(docker image inspect "$IMAGE" --format '{{.Id}}' 2>/dev/null || true)
export ROGUE_ONE_IMAGE_ID="$IMAGE_ID" ROGUE_ONE_VLLM_IMAGE_ID="$VLLM_IMAGE_ID"
say "rogueone image=$IMAGE_RO id=$IMAGE_ID"
say "vllm image=$IMAGE id=${VLLM_IMAGE_ID:-not-pulled-yet}"
if ! docker run --rm --entrypoint /bin/sh "$IMAGE_RO" -c 'test ! -e /workspace/.dev.env'; then
  fatal "$IMAGE_RO contains /workspace/.dev.env (config.py loads it with override=True); rebuild from a context without it"
fi
say "image has no /workspace/.dev.env: ok"
docker run --rm --entrypoint /bin/sh "$IMAGE_RO" -c 'test -f /workspace/scripts/agent2027/v2_defaults.env' \
  || fatal "$IMAGE_RO has no /workspace/scripts/agent2027/v2_defaults.env"
vver=$(docker run --rm --entrypoint python3 "$IMAGE" -c 'import vllm; print(vllm.__version__)' 2>/dev/null | tail -1)
say "in-container vllm == ${vver:-unknown}"
if [ -n "$EXPECT_VLLM" ] && [ "$vver" != "$EXPECT_VLLM" ]; then
  fatal "vLLM in $IMAGE is '${vver:-unknown}', expected $EXPECT_VLLM"
fi

# --- 3. W&B key for the sidecar (the clients script reads it itself as well) ------------
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$WANDB_ENV_FILE" ]; then
  WANDB_API_KEY=$(grep -E '^WANDB_API_KEY=' "$WANDB_ENV_FILE" | head -1 | cut -d= -f2- | tr -d '"'"'"'\r')
fi
export WANDB_API_KEY="${WANDB_API_KEY:-}"

# --- 4. energy sidecar ---------------------------------------------------------------
cleanup() {
  if [ "$ENERGY" = "1" ]; then
    if docker ps -a --format '{{.Names}}' | grep -qx "$ENERGY_NAME"; then
      say "stopping energy sidecar $ENERGY_NAME (SIGTERM, then upload)"
      docker stop -t 120 "$ENERGY_NAME" >/dev/null 2>&1 || true
      docker logs --tail 5 "$ENERGY_NAME" 2>&1 | sed 's/^/  energy: /'
      docker rm -f "$ENERGY_NAME" >/dev/null 2>&1 || true
    fi
  fi
  end_marker
}
trap cleanup EXIT
trap 'RC=143; exit 143' INT TERM

if [ "$ENERGY" = "1" ]; then
  if [ "$GPUS" = "auto" ]; then
    say "WARNING: GPUS=auto, the energy sidecar is not started (it needs the session's GPU ids)"
  else
    mkdir -p "$ENERGY_DIR"
    case "${CODE_PHASE:-}" in
      v2) eproj="RogueOne-AGENT2027" ;;
      "") case "$IMAGE_RO" in *-v2.*) eproj="RogueOne-AGENT2027" ;; *) eproj="RogueOne-AGENT2027-prefreeze" ;; esac ;;
      *) eproj="RogueOne-AGENT2027-prefreeze" ;;
    esac
    [ -n "$WANDB_PROJECT_NAME" ] && eproj="$WANDB_PROJECT_NAME"
    if docker run -d --name "$ENERGY_NAME" --gpus all --network host \
        -v "$ENERGY_DIR:/workspace/energy" \
        -e WANDB_API_KEY -e WANDB_ENTITY=PlanBench -e WANDB_PROJECT_NAME="$eproj" \
        --entrypoint python3 "$IMAGE_RO" /workspace/scripts/agent2027/energy_sampler.py \
        --session-id "$SESSION_ID" --gpus "$GPUS" \
        --out "/workspace/energy/${SESSION_ID}__energy.csv" >/dev/null; then
      say "energy sidecar $ENERGY_NAME started (gpus $GPUS, W&B PlanBench/$eproj ${SESSION_ID}__energy)"
    else
      say "WARNING: energy sidecar failed to start; the session continues without it"
    fi
  fi
fi

# --- 5. server + clients ---------------------------------------------------------------
say "handing off: bash $VLLM_SESSION $CLIENTS_SCRIPT (server log $LOG)"
bash "$VLLM_SESSION" "$CLIENTS_SCRIPT"
RC=$?
say "session script returned rc=$RC"
grep -E 'ROW .* status=|RO27_CLIENTS_DONE' "$LOG" 2>/dev/null | tail -3 | sed 's/^/  /'
exit "$RC"
