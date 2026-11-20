#!/usr/bin/env bash
# X4 install + A08-T4 partition check, on the CAIR host (docker / cp / grep only; CPU containers,
# no GPU). Run from anywhere after <P> = ~/code/RougeOne has fetched this branch:
#
#   bash ~/code/RougeOne/scripts/agent2027/llmfe_specs/install_and_check.sh
#
#   1. builds the LLM-FE inputs for SYN-P 7 in a container (--network none, rogue_one image,
#      as the caller's uid): make_llmfe_inputs.py data -> ~/ro27_a08/llmfe_build/
#   2. installs them: specs -> ~/llmfe/specs/, data + metadata -> ~/llmfe/data/,
#      regression alias dirs -> ~/ro27_a08/llmfe_alias/<ds>/ (crab.csv = the task's data)
#   3. checks, per task, that LLM-FE's own generation split (llmfe:oss120, main.py logic, under
#      the problem_name it will run with) equals Rogue One's seed-42 outer partition
#      (rogue_one image, deterministic_kfold_indices; stratified for classification, D4).
# Exit 0 iff every task's partitions are identical (fold order included).
set -uo pipefail
P="${ROGUEONE_CAIR_PATH:-$HOME/code/RougeOne}"
IMG="${IMAGE_RO:-rogue_one:agent2027-v2.0}"
LLMFE_DIR="${LLMFE_DIR:-$HOME/llmfe}"
A08_DIR="${A08_DIR:-$HOME/ro27_a08}"
BUILD="$A08_DIR/llmfe_build"
X="$P/scripts/agent2027"
TASKS="synp_p1 synp_p2 synp_p3 synp_p4 synp_p5 synp_p6a synp_p6b"
case "$(printf '%s' "$TASKS ${ONLY:-}" | tr 'A-Z' 'a-z')" in *covtype*) echo "FATAL: covtype is never used" >&2; exit 1 ;; esac
[ -n "${ONLY:-}" ] && TASKS="$ONLY"

mkdir -p "$BUILD" "$A08_DIR/llmfe_alias" "$A08_DIR/partition"
echo "=== 1. build ($IMG) -> $BUILD"
docker run --rm --network none --cpus 2 --user "$(id -u):$(id -g)" \
  -v "$P/tasks:/tasks:ro" -v "$X/llmfe_specs:/specs:ro" -v "$BUILD:/out" \
  --entrypoint python3 "$IMG" /specs/make_llmfe_inputs.py data --tasks-root /tasks --specs-dir /specs \
  --out /out --tasks $TASKS || { echo "FATAL: build failed"; exit 1; }

echo "=== 2. install"
for ds in $TASKS; do
  cp "$BUILD/specs/specification_${ds}.txt" "$LLMFE_DIR/specs/" || exit 1
  cp "$BUILD/data/${ds}.csv" "$BUILD/data/${ds}-metadata.json" "$LLMFE_DIR/data/" || exit 1
  if [ -d "$BUILD/alias/$ds" ]; then
    mkdir -p "$A08_DIR/llmfe_alias/$ds" && cp "$BUILD/alias/$ds/"* "$A08_DIR/llmfe_alias/$ds/" || exit 1
  fi
done
ls -la "$LLMFE_DIR/specs/" | grep synp_ ; ls "$A08_DIR/llmfe_alias/"

echo "=== 3. partition check (LLM-FE split vs Rogue One seed-42 folds)"
bad=0
for ds in $TASKS; do
  if [ -d "$P/tasks/regression/$ds" ]; then
    ttype=regression; csvdir="$A08_DIR/llmfe_alias/$ds"; csv=crab.csv; pname=crab
  else
    ttype=classification; csvdir="$LLMFE_DIR/data"; csv="${ds}.csv"; pname="$ds"
  fi
  lf="$A08_DIR/partition/llmfe_${ds}.json"; ro="$A08_DIR/partition/rogueone_${ds}.json"
  docker run --rm --network none --cpus 2 -v "$P/tasks:/tasks:ro" -v "$csvdir:/d:ro" -v "$X:/x:ro" \
    llmfe:oss120 python /x/check_fold_partition.py llmfe "/tasks/$ttype/$ds" \
    --llmfe-csv "/d/$csv" --llmfe-problem-name "$pname" > "$lf" || { echo "$ds: llmfe side failed"; bad=1; continue; }
  docker run --rm --network none --cpus 2 -v "$P/tasks:/tasks:ro" -v "$X:/x:ro" \
    --entrypoint python3 "$IMG" /x/check_fold_partition.py rogueone "/tasks/$ttype/$ds" --seed 42 \
    > "$ro" || { echo "$ds: rogueone side failed"; bad=1; continue; }
  res=$(docker run --rm --network none -v "$A08_DIR/partition:/p:ro" -v "$X:/x:ro" --entrypoint python3 "$IMG" \
    /x/check_fold_partition.py compare "/p/llmfe_${ds}.json" "/p/rogueone_${ds}.json")
  rc=$?
  echo "$ds ($ttype, problem_name=$pname): $res"
  [ "$rc" = "0" ] || bad=1
done
[ "$bad" = "0" ] && echo "=== X4_INSTALL_OK all partitions identical ===" || echo "=== X4_INSTALL_FAILED (see above) ==="
exit "$bad"
