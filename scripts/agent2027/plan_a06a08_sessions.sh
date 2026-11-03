#!/usr/bin/env bash
# Re-plan the A08 LLM-FE sessions with their A06 O120 sentinel once G2 has named the task
# (developer machine; never on the CAIR host). A06-T9 O120 block + A06-T12 placement.
#
#   bash scripts/agent2027/plan_a06a08_sessions.sh <SENTINEL_TASK> <K> <IMAGE_TAG> [guard3]
#     SENTINEL_TASK  the G2 memo's task (synp_p1 … synp_p6b; A06 §4.3)
#     K              the lowest unused O120 sentinel k (A06-00K; check every committed session
#                    file: grep -l 'A06-0' manifests/agent2027/sessions/*.csv)
#     IMAGE_TAG      the image the A08 session will run its sentinel on (= its IMAGE_RO tag),
#                    e.g. agent2027-v2.2
#     guard3         pass "guard3" if this is the first qualifying session on IMAGE_TAG
#                    (sentinel in the first wave, A06 §4.5)
#
# The sentinel goes into S1020l (port 8020): S1020l + S1020m launched together are ONE A06
# session (TP2 pair), so only one sentinel. S1020m is re-written unchanged (no sentinel).
set -euo pipefail
TASK="${1:?SENTINEL_TASK}"; K="${2:?K}"; TAG="${3:?IMAGE_TAG}"; G3="${4:-}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON:-python}"
MM="$REPO/scripts/agent2027/make_manifest.py"
ROW="A06-$(printf '%03d' "$K")"
if grep -l "^${ROW}," "$REPO"/manifests/agent2027/sessions/*.csv 2>/dev/null; then
  echo "FATAL: $ROW is already planned in the session file(s) above; use the next free k" >&2; exit 1
fi
first=()
[ "$G3" = "guard3" ] && first=(--sentinel-first)
"$PY" "$MM" --ablation A06 --param "SENTINEL_TASK=$TASK"
"$PY" "$MM" --plan-session S1020l --rows A08-082 A08-084 A08-086 A08-088 A08-090 A08-092 A08-094 "$ROW" \
  --port 8020 --proxy-port 8090 --sentinel-image-tag "$TAG" ${first[@]+"${first[@]}"}
"$PY" "$MM" --plan-session S1020m --rows A08-083 A08-085 A08-087 A08-089 A08-091 A08-093 A08-095 \
  --port 8021 --proxy-port 8091 --no-sentinel
"$PY" "$MM" --check "$REPO/manifests/agent2027/A06_M0-S.csv" "$REPO/manifests/agent2027/A08_M5.csv" \
  "$REPO/manifests/agent2027/sessions/S1020l.csv" "$REPO/manifests/agent2027/sessions/S1020m.csv"
cat "$REPO/manifests/agent2027/sessions/S1020l.plan.txt"
