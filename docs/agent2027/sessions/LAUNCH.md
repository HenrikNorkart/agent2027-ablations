# LAUNCH — small-model arms under the SR4 fallback (A17, A19, A20, A18, A04 session C)

Prepared 2026-10-19 (Mon) on branch `agent2027/cap`. Nothing here has been launched. Every line
assumes the branch is merged into `agent2027/v2` and pushed (step 0). Pre-registration: Part
VII.1 item 14 (SR4), Part VIII.1, Part VIII.2, Amendment 2 (τ = 0.60, CL1, CL2, sentinel
`synp_p2`). Hard rules as always: containers only, GPU pre-flight before every launch, never
covtype (every launcher refuses it), W&B is the record.

`CAIR` below = `ssh -o BatchMode=yes -o ConnectTimeout=30 CAIR-gpu14`.

## Rungs (from `scripts/agent2027/rungs.csv`)

| Rung | MODEL | GPUs | EXTRA_ARGS |
|---|---|---|---|
| Q0.8 | `Qwen/Qwen3.5-0.8B` | 1 | `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --language-model-only --revision 2fc06364715b967f1860aea9cf38778875588b17` |
| Q2 | `Qwen/Qwen3.5-2B` | 1 | `… --language-model-only --revision 15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Q4 | `Qwen/Qwen3.5-4B` | 1 | `… --language-model-only --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Q9 | `Qwen/Qwen3.5-9B` | 1 | `… --language-model-only --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Q122 | `Qwen/Qwen3.5-122B-A10B-FP8` | 2 (TP2) | `… --language-model-only --max-num-seqs 256 --revision a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9` |

`…` = `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3`.
`MAX_MODEL_LEN=131072` everywhere. Weights at these revisions are in `~/.cache/huggingface`
(checked 2026-10-19). O20 is not used (SR4 FAIL).

## Sessions

| Session | Arm(s) | Rows | Server(s) · port | GPUs | Image | Sentinel | Gate |
|---|---|---|---|---|---|---|---|
| `S1019g` | A17 `f2_Q4` + A06 | A17-030…044, -047 (incl. Q4 anchor A17-031), A06-102 | Q4 · 8034 | 1 (GPU 7) | v2.0 | `sentinel_Q4` k=2 | **none: runnable now** (after `S1019q` frees GPU 7) |
| `S1019h` | A17 `f2_Q0.8` | A17-001…015, -018 | Q0.8 · 8035 | 1 (GPU 7) | v2.0 | none (no Q4 server) | **none: runnable now** (queued after `S1019g`) |
| `S1019i` | A17 `f2_Q2` | A17-019…026, -029 | Q2 · 8036 | 1 (GPU 7) | v2.0 | none | **none: runnable now** (queued) |
| `S1019j` | A17 `f2_Q9` | A17-048…055, -058 | Q9 · 8037 | 1 (GPU 7) | v2.0 | none | **none: runnable now** (queued) |
| `S1021m` (+ holders `S1021n`, `S1021o`) | A04 session C (SR6) | A04-007, A04-008 | Q122 · 8040 + Q4 · 8032 + Q0.8 · 8030 | 4 | v2.3 | none (smoke) | **G0b-1** (Wed 21: `S1021b` switch smokes pass, Full byte-identical to `prompt-baseline-v2.0`) |
| `S1022c` | A20 smoke | A20-008 | Q4 · 8038 | 1 | v2.3 | none (smoke) | **G0b-1** (A20-T4) and the A19/A20 slot (Thu 22) |
| `S1022d` (holder) | A20 `fx_pajM_Q4` + A06 | A20-001…007, A06-103 (first wave) | Q4 · 8032 | 1 | v2.3 | `sentinel_Q4` k=3 | **Thu 22**, G0b-1, `S1022c` checks passed; as A19's Q4 server also SR6 |
| `S1022f` (holder) | — (A19's Q0.8 server) | none | Q0.8 · 8030 | 1 | v2.3 | — | with `S1022a` |
| `S1022a` | A19 `mix_esmall`, `mix_1b` | A19-001…014, A19-019…032 | Q122 · 8040 (+ 8032, 8030 above) | 2 (+2) | v2.3 | via `S1022d` | **Thu 22**, G0b-1, **SR6 pass** (A04-T19 on `S1021m`), A19-T9a (`misrouted_calls`); τ addendum done |
| `S1024a` | A17 `f2_Q122` | A17-070…087 | Q122 · 8041 | 2 | v2.0 | none (single Q122 session) | **Sat 24** (A17 Q122 window; launch by Wed 28) |
| A18 replays | A18-001…004, 006, 007 | — | clients of the servers above (§6) | 0 extra | v2.0 | none | **Thu 22 – Tue 27** (Henrik); tooling built 2026-10-19 (`agent2027/a18`), `rp1-prompts:v0` logged; needs the merge and the §6.0 install |
| F0 pools | A17-101…177 (minus 143…149), `manifests/agent2027/A17_F0.csv` | — | clients of a server of the row's rung (§7) | 0 extra | v2.0 | none | prompt frozen (Amendment 6); X7 code built 2026-10-22 (`f0_generate.py`, `f0_clients.sh`); ED3 Wed 28 |

GPU totals when everything overlaps (Thu 22 – Sat 24): GPU 7 chain 1 + A19 trio 4 (+ A17 Q122
2 from Sat 24) on top of the O120 sessions. That exceeds 4; it needs Henrik's Oct 2026 8-GPU
allowance (agreed runs only), and GPU 0 is dead.

## 0. Install (once per change; safe while other sessions run)

Never `git checkout` in `~/code/RougeOne` while a session runs, and never overwrite a running
script in place: `~/ro27_queue.sh` is running the GPU 5,6 queue, so the new queue is installed
under a new name, and the clients script is replaced by an atomic `mv`.

```
$CAIR "cd ~/code/RougeOne && GIT_SSH_COMMAND='ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15' git fetch git@github.com:HenrikNorkart/RougeOne.git agent2027/v2 && git log -1 --format='FETCH_HEAD: %h %s' FETCH_HEAD \
  && git show FETCH_HEAD:scripts/agent2027/ro27_queue.sh > ~/ro27_queue_rung.sh \
  && git show FETCH_HEAD:scripts/agent2027/queue_gpu7.txt > ~/ro27_queue_gpu7.txt \
  && git show FETCH_HEAD:scripts/agent2027/ro27_hold_clients.sh > ~/ro27_hold_clients.sh \
  && git show FETCH_HEAD:scripts/agent2027/ro27_clients_manifest.sh > ~/ro27_clients_manifest.sh.new && mv ~/ro27_clients_manifest.sh.new ~/ro27_clients_manifest.sh \
  && for f in A04_CAP-0 A17_CAP-1 A18_RP1 A19_CAP-2 A20_CAP-3; do git show FETCH_HEAD:manifests/agent2027/\$f.csv > ~/ro27_manifests/\$f.csv; done \
  && for s in S1019g S1019h S1019i S1019j S1021m S1022a S1022c S1022d S1024a; do git show FETCH_HEAD:manifests/agent2027/sessions/\$s.csv > ~/ro27_manifests/sessions/\$s.csv; done \
  && grep -l covtype ~/ro27_manifests/sessions/S10*.csv || echo no-covtype"
```

`RUNGS_CSV` defaults to `~/code/RougeOne/scripts/agent2027/rungs.csv` (unchanged since `747b514`,
present in the current checkout).

Pre-flight (before every launch below):
```
$CAIR "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader && echo '--- compute apps ---' && nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader && echo '--- containers ---' && docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'"
```

## 1. Runnable now on GPU 7 — A17 small rungs (no date gate)

A17 small rungs were cleared from Sun 18 night once A05/A07 ended and G2 was GO (A17-T5); the
docs place no Thu gate on them. They are plain Full-v2 on `agent2027-v2.0`. The A06 Q4 sentinel
`S1019q` (A06-101, port 8022) holds GPU 7 until it ends (6 h cap, started 07:28Z, so at the
latest ≈ 13:28Z). The queue waits for its `RO27_SESSION_END`, then for GPU 7 < 2,000 MiB, then
runs `S1019g` → `S1019h` → `S1019i` → `S1019j`, one at a time, 6 clients each.

```
$CAIR "QUEUE_FILE=\$HOME/ro27_queue_gpu7.txt GPUS=7 WAIT_FOR=S1019q PARALLEL=6 nohup bash ~/ro27_queue_rung.sh > ~/ro27_queue_gpu7.log 2>&1 < /dev/null & echo launched; sleep 3; tail -3 ~/ro27_queue_gpu7.log"
```

Equivalent single launch of the first session only (if the queue is not wanted):
```
$CAIR "SESSION_ID=S1019g SESSION_FILE=\$HOME/ro27_manifests/sessions/S1019g.csv GPUS=7 NEED=1 MODEL=Qwen/Qwen3.5-4B PORT=8034 SERVER_NAME=rogueone-vllm-q4-s1019g EXTRA_ARGS='--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --language-model-only --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a' MAX_MODEL_LEN=131072 EAGER=0 ROGUEONE_ALLOW_SECOND_SERVER=1 LOG=\$HOME/vllm_session_S1019g.log IMAGE_RO=rogue_one:agent2027-v2.0 PARALLEL=6 CPUS=12 RUN_CAP_HOURS=6 RUNROOT=\$HOME/rogueone-runs-agent2027 WANDB_PROJECT_NAME=RogueOne-AGENT2027 WAIT_MAX_MIN=0 nohup bash ~/ro27_session_launch.sh > ~/ro27_S1019g.job.log 2>&1 < /dev/null & echo launched"
```
(`S1019h`/`i`/`j`: same form with the rung's MODEL, EXTRA_ARGS, port 8035/8036/8037 and
`SERVER_NAME=rogueone-vllm-<rung>-<sid>`.)

Checks after ≈ 1 h (A17-T11): W&B `flags/LLM_PORT` = 8034, `flags/LLM_MODEL` = `Qwen/Qwen3.5-4B`,
non-zero `tok_compl_*` for S, E, T. Expect many 6 h cap hits (A04 SR2: Q0.8/Q4/Q9 fold-runs hit
the cap); they are data. Rough wall time: 17 + 16 + 9 + 9 rows at 6 clients ≈ 18 + 18 + 12 + 12 h
≈ 60 h, so the chain ends around Thu 22 morning. Stop the queue with `touch ~/ro27_queue_STOP`
(note: the GPU 5,6 queue reads the same STOP file).

## 2. Gated: A04 session C (SR6), from G0b-1 (Wed 21)

Holders first, then the main session (4 free GPUs `<q122a>,<q122b>`, `<q4>`, `<q08>`):
```
$CAIR "SESSION_ID=S1021n SESSION_FILE=\$HOME/ro27_manifests/sessions/S1021m.csv GPUS=<q4> NEED=1 MODEL=Qwen/Qwen3.5-4B PORT=8032 SERVER_NAME=rogueone-vllm-q4-s1021n EXTRA_ARGS='<Q4 extra_args>' MAX_MODEL_LEN=131072 EAGER=0 ROGUEONE_ALLOW_SECOND_SERVER=1 LOG=\$HOME/vllm_session_S1021n.log IMAGE_RO=rogue_one:agent2027-v2.3 CLIENTS_SCRIPT=\$HOME/ro27_hold_clients.sh INNER_CLIENTS=none HOLD_FOR=S1021m HOLD_MAX_H=8 RUNROOT=\$HOME/rogueone-runs-agent2027 WAIT_MAX_MIN=0 nohup bash ~/ro27_session_launch.sh > ~/ro27_S1021n.job.log 2>&1 < /dev/null & echo launched"
$CAIR "SESSION_ID=S1021o SESSION_FILE=\$HOME/ro27_manifests/sessions/S1021m.csv GPUS=<q08> NEED=1 MODEL=Qwen/Qwen3.5-0.8B PORT=8030 SERVER_NAME=rogueone-vllm-q0.8-s1021o EXTRA_ARGS='<Q0.8 extra_args>' MAX_MODEL_LEN=131072 EAGER=0 ROGUEONE_ALLOW_SECOND_SERVER=1 LOG=\$HOME/vllm_session_S1021o.log IMAGE_RO=rogue_one:agent2027-v2.3 CLIENTS_SCRIPT=\$HOME/ro27_hold_clients.sh INNER_CLIENTS=none HOLD_FOR=S1021m HOLD_MAX_H=8 RUNROOT=\$HOME/rogueone-runs-agent2027 WAIT_MAX_MIN=0 nohup bash ~/ro27_session_launch.sh > ~/ro27_S1021o.job.log 2>&1 < /dev/null & echo launched"
$CAIR "SESSION_ID=S1021m SESSION_FILE=\$HOME/ro27_manifests/sessions/S1021m.csv GPUS=<q122a>,<q122b> NEED=2 MODEL=Qwen/Qwen3.5-122B-A10B-FP8 PORT=8040 SERVER_NAME=rogueone-vllm-q122-s1021m EXTRA_ARGS='<Q122 extra_args>' MAX_MODEL_LEN=131072 EAGER=0 ROGUEONE_ALLOW_SECOND_SERVER=1 LOG=\$HOME/vllm_session_S1021m.log IMAGE_RO=rogue_one:agent2027-v2.3 CLIENTS_SCRIPT=\$HOME/ro27_hold_clients.sh WAIT_PORTS='8032 8030' PARALLEL=2 CPUS=12 RUN_CAP_HOURS=6 RUNROOT=\$HOME/rogueone-runs-agent2027 WANDB_PROJECT_NAME=RogueOne-AGENT2027 WAIT_MAX_MIN=0 nohup bash ~/ro27_session_launch.sh > ~/ro27_S1021m.job.log 2>&1 < /dev/null & echo launched"
```
Then A04-T19 (`stage0_table.py`, SR6 verdict). No `stage0_O120` replay (no O120 server).

## 3. Gated: A20 smoke (G0b-1; Thu 22 slot)

```
$CAIR "SESSION_ID=S1022c SESSION_FILE=\$HOME/ro27_manifests/sessions/S1022c.csv GPUS=<g> NEED=1 MODEL=Qwen/Qwen3.5-4B PORT=8038 SERVER_NAME=rogueone-vllm-q4-s1022c EXTRA_ARGS='<Q4 extra_args>' MAX_MODEL_LEN=131072 EAGER=0 ROGUEONE_ALLOW_SECOND_SERVER=1 LOG=\$HOME/vllm_session_S1022c.log IMAGE_RO=rogue_one:agent2027-v2.3 PARALLEL=1 CPUS=12 RUN_CAP_HOURS=6 RUNROOT=\$HOME/rogueone-runs-agent2027 WANDB_PROJECT_NAME=RogueOne-AGENT2027 WAIT_MAX_MIN=0 nohup bash ~/ro27_session_launch.sh > ~/ro27_S1022c.job.log 2>&1 < /dev/null & echo launched"
```
A20-T9 checks: `llm_calls_scientist = 0`, `llm_calls_tester = 0`, model_S/E/T/W = Q4, E and W on
port 8038, `fx_digest.jsonl` present, `focus_line_missing` logged.

## 4. Gated: A19 trio + A20 rows (Thu 22; G0b-1, SR6, A19-T9a, smoke passed)

Holders first (each on 1 GPU), then the main session on 2 GPUs:
```
$CAIR "SESSION_ID=S1022d SESSION_FILE=\$HOME/ro27_manifests/sessions/S1022d.csv GPUS=<q4> NEED=1 MODEL=Qwen/Qwen3.5-4B PORT=8032 SERVER_NAME=rogueone-vllm-q4-s1022d EXTRA_ARGS='<Q4 extra_args>' MAX_MODEL_LEN=131072 EAGER=0 ROGUEONE_ALLOW_SECOND_SERVER=1 LOG=\$HOME/vllm_session_S1022d.log IMAGE_RO=rogue_one:agent2027-v2.3 CLIENTS_SCRIPT=\$HOME/ro27_hold_clients.sh HOLD_FOR=S1022a HOLD_MAX_H=48 PARALLEL=4 CPUS=12 RUN_CAP_HOURS=6 RUNROOT=\$HOME/rogueone-runs-agent2027 WANDB_PROJECT_NAME=RogueOne-AGENT2027 WAIT_MAX_MIN=0 nohup bash ~/ro27_session_launch.sh > ~/ro27_S1022d.job.log 2>&1 < /dev/null & echo launched"
$CAIR "SESSION_ID=S1022f SESSION_FILE=\$HOME/ro27_manifests/sessions/S1022a.csv GPUS=<q08> NEED=1 MODEL=Qwen/Qwen3.5-0.8B PORT=8030 SERVER_NAME=rogueone-vllm-q0.8-s1022f EXTRA_ARGS='<Q0.8 extra_args>' MAX_MODEL_LEN=131072 EAGER=0 ROGUEONE_ALLOW_SECOND_SERVER=1 LOG=\$HOME/vllm_session_S1022f.log IMAGE_RO=rogue_one:agent2027-v2.3 CLIENTS_SCRIPT=\$HOME/ro27_hold_clients.sh INNER_CLIENTS=none HOLD_FOR=S1022a HOLD_MAX_H=48 RUNROOT=\$HOME/rogueone-runs-agent2027 WAIT_MAX_MIN=0 nohup bash ~/ro27_session_launch.sh > ~/ro27_S1022f.job.log 2>&1 < /dev/null & echo launched"
$CAIR "SESSION_ID=S1022a SESSION_FILE=\$HOME/ro27_manifests/sessions/S1022a.csv GPUS=<q122a>,<q122b> NEED=2 MODEL=Qwen/Qwen3.5-122B-A10B-FP8 PORT=8040 SERVER_NAME=rogueone-vllm-q122-s1022a EXTRA_ARGS='<Q122 extra_args>' MAX_MODEL_LEN=131072 EAGER=0 ROGUEONE_ALLOW_SECOND_SERVER=1 LOG=\$HOME/vllm_session_S1022a.log IMAGE_RO=rogue_one:agent2027-v2.3 CLIENTS_SCRIPT=\$HOME/ro27_hold_clients.sh WAIT_PORTS='8032 8030' PARALLEL=6 CPUS=12 RUN_CAP_HOURS=6 RUNROOT=\$HOME/rogueone-runs-agent2027 WANDB_PROJECT_NAME=RogueOne-AGENT2027 WAIT_MAX_MIN=0 nohup bash ~/ro27_session_launch.sh > ~/ro27_S1022a.job.log 2>&1 < /dev/null & echo launched"
```
`SERVER_NAME` must be given explicitly for `S1022a` and `S1021m` (the launcher's default takes
the rung of the session file's first row, which may be Q4 or Q0.8 on a mixed row). If SR6 fails,
run `S1022d` alone without `CLIENTS_SCRIPT`/`HOLD_FOR` (A20 does not depend on A19). A19-T14:
`for s in q122-s1022a q4-s1022d q0.8-s1022f; do docker logs --since 40m rogueone-vllm-$s …`.
`mix_1b` stop rule at 6 completed `mix_1b` rows (A19 §6).

## 5. Gated: A17 Q122 (from Sat 24, launch by Wed 28)

```
$CAIR "SESSION_ID=S1024a SESSION_FILE=\$HOME/ro27_manifests/sessions/S1024a.csv GPUS=<a>,<b> NEED=2 MODEL=Qwen/Qwen3.5-122B-A10B-FP8 PORT=8041 SERVER_NAME=rogueone-vllm-q122-s1024a EXTRA_ARGS='<Q122 extra_args>' MAX_MODEL_LEN=131072 EAGER=0 ROGUEONE_ALLOW_SECOND_SERVER=1 LOG=\$HOME/vllm_session_S1024a.log IMAGE_RO=rogue_one:agent2027-v2.0 PARALLEL=6 CPUS=12 RUN_CAP_HOURS=6 RUNROOT=\$HOME/rogueone-runs-agent2027 WANDB_PROJECT_NAME=RogueOne-AGENT2027 WAIT_MAX_MIN=0 nohup bash ~/ro27_session_launch.sh > ~/ro27_S1024a.job.log 2>&1 < /dev/null & echo launched"
```
(or as a queue line `S1024a|agent2027-v2.0|8041|RUNG=Q122` with `GPUS=<a>,<b>`).

## 6. A18 replays (RP1; slot Thu 22 – Tue 27, L-LAST Thu 29)

Built 2026-10-19 on branch `agent2027/a18` (to merge into `agent2027/v2`):
`scripts/agent2027/analysis/rp1_score.py` (`sample`, `fetch`, `score`, `attach-log`),
`scripts/agent2027/rp1_clients.sh` (installed as `~/ro27_rp1_clients.sh`) and
`replay_tool.py --validity a18`. The prompt set is drawn and frozen:

- **`PlanBench/RogueOne-AGENT2027/rp1-prompts:v0`** (type `prompt-set`, logged by run
  `ANALYSIS__rp1_score__20261019`, id `j58hagpv`): 800 records (S 200, E 300, T 200, W 100;
  every §4 cell at its target n, no fill needed, no ITT substitution), file
  `rp1-prompts-v0.jsonl.gz` sha256 `a2b11c09…d678c` (in the artifact metadata), sampler
  `rp1_score.py` sha256 `a2df4266…b9a2a` at `30ed810`, seed 42. The A18-T5 heavy QC passed on all
  18 source runs (trace records = `llm_calls_total`, A18-T2 fields in 100% of records, T2 tables
  for every dumped trial). Largest recorded prompt 67,640 tokens. Never re-drawn
  (`sample --wandb` refuses while `rp1-prompts` exists).

A replay is one CPU container (`replay_tool.py`, concurrency 8, `--cpus 4`) against a vLLM server
that already runs for another session of the same rung. It uses no GPU, so no GPU pre-flight is
needed for it; the host session's own pre-flight covered its server. The replay adds load to the
host server (wall time only, disclosed, A18 §4). `rp1_clients.sh` refuses a server whose model is
not the row's rung (`rungs.csv`), an unhealthy server, a host session that already logged
`RO27_SESSION_END`, and a running container of the same name. Client image
`rogue_one:agent2027-v2.0` (from the manifest row; the frozen `clients.py` wrapper the source
prompts went through, so the A18-T10 wrapper check holds by construction).

### 6.0 Install (once, after `agent2027/a18` is merged into `agent2027/v2` and pushed)

Safe while other sessions run: `~/code/RougeOne` is not checked out; the scripts go to a
snapshot directory that the replay container mounts. Do not reinstall while an A18 replay runs.
```
$CAIR "cd ~/code/RougeOne && GIT_SSH_COMMAND='ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15' git fetch git@github.com:HenrikNorkart/RougeOne.git agent2027/v2 && git log -1 --format='FETCH_HEAD: %h %s' FETCH_HEAD \
  && rm -rf ~/ro27_a18_code.new && mkdir -p ~/ro27_a18_code.new ~/ro27_a18_logs ~/ro27_a18_work \
  && git archive FETCH_HEAD scripts/agent2027 | tar -x -C ~/ro27_a18_code.new && rm -rf ~/ro27_a18_code && mv ~/ro27_a18_code.new ~/ro27_a18_code \
  && git show FETCH_HEAD:scripts/agent2027/rp1_clients.sh > ~/ro27_rp1_clients.sh.new && mv ~/ro27_rp1_clients.sh.new ~/ro27_rp1_clients.sh \
  && git show FETCH_HEAD:manifests/agent2027/A18_RP1.csv > ~/ro27_manifests/A18_RP1.csv \
  && grep -c -- '--validity' ~/ro27_a18_code/scripts/agent2027/replay_tool.py && RP1_ROW=A18-003 PORT=8034 MODEL=Qwen/Qwen3.5-4B SERVER_NAME=rogueone-vllm-q4-s1019g SESSION_ID=S1019g DRY_RUN=1 bash ~/ro27_rp1_clients.sh"
```
The dry run prints the three A18-T7 steps and `RP1_END A18-003 rc=0 dry_run=1`.

### 6.1 Hosts and timing

| Row | Target | Host session · server · port | Host up (estimate) |
|---|---|---|---|
| A18-003 | `replay_Q4` | `S1019g` · `rogueone-vllm-q4-s1019g` · 8034 | ≈ Mon 19 13:30Z – Tue 20 07:30Z |
| A18-001 | `replay_Q0.8` | `S1019h` · `rogueone-vllm-q0.8-s1019h` · 8035 | ≈ Tue 20 07:30Z – Wed 21 01:30Z |
| A18-002 | `replay_Q2` | `S1019i` · `rogueone-vllm-q2-s1019i` · 8036 | ≈ Wed 21 01:30Z – 13:30Z |
| A18-004 | `replay_Q9` | `S1019j` · `rogueone-vllm-q9-s1019j` · 8037 | ≈ Wed 21 13:30Z – Thu 22 01:30Z |
| A18-006 | `replay_Q122` | `S1022a` · `rogueone-vllm-q122-s1022a` · 8040, or `S1024a` · `…-q122-s1024a` · 8041 | Thu 22 (A19) / from Sat 24 (A17) |
| A18-007 | `replay_O120` | any O120 TP2 session (GPUs 5,6 queue: `S1019a` 8023, `S1021b` 8024, `S1020a` 8018, `S1020b` 8019, …; A12/A16) | continuous |

GPU 7 chain timing (`~/ro27_queue_gpu7.log`, checked Mon 19 09:39Z): the queue is waiting for
`S1019q` (A06-101, started 07:28Z, 6 h cap, so it ends by ≈ 13:30Z); each link then runs its rows
at 6 clients with 6 h caps (≈ 18 + 18 + 12 + 12 h, §1). **So the GPU 7 Q-rung servers are gone
before Thu 22**: on the planned dates only Q9 (`S1019j`) may still be up in the early hours of
Thu 22. Either Henrik lets A18-001…004 ride the chain from Mon 19/Tue 20 (each target needs
≈ 1–2 h of its host, A18 §4; start it in the first hours of the host session), or, from Thu 22:
Q4 on A19's holder `S1022d` (8032), Q0.8 on `S1022f` (8030), and Q2 and Q9 as their own
one-GPU sessions (6.4). Check the host before each launch:
```
$CAIR "date -u; tail -3 ~/ro27_queue_gpu7.log; for s in S1019g S1019h S1019i S1019j; do echo \$s: \$(grep -m1 -o 'server=[^ ]* model=[^ ]* port=[0-9]*' ~/ro27_\$s.job.log 2>/dev/null) \$(grep -c RO27_SESSION_END ~/ro27_\$s.job.log 2>/dev/null); done"
```
(`… 0` = running, `… 1` = ended, empty = not started.)

### 6.2 Launch lines (stand-alone piggy-back; one per target, detached)

`HOST_SID` makes the script read port, model and server name from `~/ro27_<HOST_SID>.job.log`.
```
# A18-003 replay_Q4 on the GPU 7 Q4 server (8034); Thu 22 alternative: HOST_SID=S1022d (8032)
$CAIR "RP1_ROW=A18-003 HOST_SID=S1019g nohup bash ~/ro27_rp1_clients.sh > ~/ro27_a18_A18-003.job.log 2>&1 < /dev/null & echo launched; sleep 30; tail -5 ~/ro27_a18_A18-003.job.log"
# A18-001 replay_Q0.8 on the Q0.8 server (8035); Thu 22 alternative: HOST_SID=S1022f (8030)
$CAIR "RP1_ROW=A18-001 HOST_SID=S1019h nohup bash ~/ro27_rp1_clients.sh > ~/ro27_a18_A18-001.job.log 2>&1 < /dev/null & echo launched; sleep 30; tail -5 ~/ro27_a18_A18-001.job.log"
# A18-002 replay_Q2 on the Q2 server (8036)
$CAIR "RP1_ROW=A18-002 HOST_SID=S1019i nohup bash ~/ro27_rp1_clients.sh > ~/ro27_a18_A18-002.job.log 2>&1 < /dev/null & echo launched; sleep 30; tail -5 ~/ro27_a18_A18-002.job.log"
# A18-004 replay_Q9 on the Q9 server (8037)
$CAIR "RP1_ROW=A18-004 HOST_SID=S1019j nohup bash ~/ro27_rp1_clients.sh > ~/ro27_a18_A18-004.job.log 2>&1 < /dev/null & echo launched; sleep 30; tail -5 ~/ro27_a18_A18-004.job.log"
# A18-006 replay_Q122 on the A19 Q122 TP2 server (8040); or HOST_SID=S1024a (8041, A17, from Sat 24)
$CAIR "RP1_ROW=A18-006 HOST_SID=S1022a nohup bash ~/ro27_rp1_clients.sh > ~/ro27_a18_A18-006.job.log 2>&1 < /dev/null & echo launched; sleep 30; tail -5 ~/ro27_a18_A18-006.job.log"
# A18-007 replay_O120 (control) on a running O120 TP2 server: pick the host first
$CAIR "for f in \$(grep -l 'model=openai/gpt-oss-120b' ~/ro27_S10*.job.log); do grep -q RO27_SESSION_END \$f || echo \$f \$(grep -m1 -o 'port=[0-9]*' \$f); done"
$CAIR "RP1_ROW=A18-007 HOST_SID=<running O120 sid> nohup bash ~/ro27_rp1_clients.sh > ~/ro27_a18_A18-007.job.log 2>&1 < /dev/null & echo launched; sleep 30; tail -5 ~/ro27_a18_A18-007.job.log"
```
Explicit form (if the host job log has no `server=… model=… port=…` line): replace `HOST_SID=…`
with `PORT=<port> MODEL=<HF id> SERVER_NAME=<server> SESSION_ID=<host sid>`.
`RP1_CONCURRENCY=32` for a target that A18-T4 projects > 6 h; `RUN_CAP_HOURS` (default 6)
caps the container. Start a target only when its host has ≥ 2 h left (Q122/O120: ≥ 3 h): if the
host tears its server down mid-replay the remaining calls become `infra_error`, and a target
with > 5% infra errors is rerun once (A18 §4).

### 6.3 Monitor, then check

```
ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 CAIR-gpu14 "while ! grep -q RP1_END ~/ro27_a18_A18-00n.job.log; do sleep 120; done; tail -30 ~/ro27_a18_A18-00n.job.log"
$CAIR "docker ps --format '{{.Names}}' | grep -E -- '-a18-' || echo none"
```
Done when the job log ends `RP1_END A18-00n rc=0`, the W&B run `A18__replay_<rung>__rp1set__f42__r0`
(group `A18`) is `finished` with `rp1_valid_call_raw__<role>` / `rp1_valid_call_rescued__<role>`
(rates over the target's own scored prompts) and artifacts `rp1-replies-<rung>` and
`rp1-server-log-<rung>`. After `replay_O120`: RP1-D5(a), every `rp1_valid_call_rescued__<role>`
≥ 0.9. Replies and server logs also stay in `~/ro27_a18_work/replies_<rung>_<sid>_<row>/` and
`~/ro27_a18_logs/`.

### 6.4 Own-server form (no host up: Q2/Q9 on Thu 22, or any target)

GPU pre-flight first (§0); then the A18-T11 form, one GPU (Q122: 2), `rp1_clients.sh` as the
clients script, `SERVER_NAME` explicit (the manifest's first row is Q0.8):
```
$CAIR "SESSION_ID=<SID> SESSION_FILE=\$HOME/ro27_manifests/A18_RP1.csv RP1_ROW=A18-002 GPUS=<g> NEED=1 MODEL=Qwen/Qwen3.5-2B PORT=8031 SERVER_NAME=rogueone-vllm-q2-<sid> EXTRA_ARGS='--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --language-model-only --revision 15852e8c16360a2fea060d615a32b45270f8a8fc' MAX_MODEL_LEN=131072 EAGER=0 ROGUEONE_ALLOW_SECOND_SERVER=1 LOG=\$HOME/vllm_session_<SID>.log IMAGE_RO=rogue_one:agent2027-v2.0 RUN_CAP_HOURS=6 CLIENTS_SCRIPT=\$HOME/ro27_rp1_clients.sh WANDB_PROJECT_NAME=RogueOne-AGENT2027 WAIT_MAX_MIN=0 nohup bash ~/ro27_session_launch.sh > ~/ro27_<SID>.job.log 2>&1 < /dev/null & echo launched"
```
(Q9: `RP1_ROW=A18-004 MODEL=Qwen/Qwen3.5-9B PORT=8033`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`,
`SERVER_NAME=rogueone-vllm-q9-<sid>`.) The server is torn down when the replay returns.

### 6.5 Scoring (A18-T16, Sun 1 – Mon 2 Nov; CPU containers)

```
$CAIR "docker run --rm --name ro27-a18-fetch --cpus 4 --env-file /raid/home/henrbr16/code/PlanBench/.env -v \$HOME/ro27_a18_work:/work -v \$HOME/ro27_a18_code/scripts/agent2027:/workspace/scripts/agent2027:ro --entrypoint python3 rogue_one:agent2027-v2.3 /workspace/scripts/agent2027/analysis/rp1_score.py fetch --out /work/score"
$CAIR "nohup docker run --rm --name ro27-a18-score --network none --cpus 16 --memory 64g -v \$HOME/ro27_a18_work:/work -v \$HOME/code/RougeOne/tasks:/workspace/tasks:ro -v \$HOME/rogueone-synp-meta:/synp_meta:ro -v \$HOME/ro27_a18_code/scripts/agent2027:/workspace/scripts/agent2027:ro --entrypoint python3 rogue_one:agent2027-v2.3 /workspace/scripts/agent2027/analysis/rp1_score.py score --work /work/score --synp-meta /synp_meta --exec-timeout 60 --jobs 16 > ~/ro27_a18_score.log 2>&1 < /dev/null & echo launched"
```
(v2.3 for E9d's `ja_rules`; the generated code runs in subprocesses with the E13 sandbox.)
Outputs `/work/score/scored/rp1_scored.csv`, `rp1_rates.csv`, `rp1_score_summary.json`.
Floors, bootstrap, McNemar and the D1–D5 verdicts (`floors`, A18-T18) are not built yet.

Trial run 2026-10-19 (`fetch` + `score` on `rp1-prompts:v0`, pseudo-target `recorded_O120`
only, 37 min fetch + 4 min score, `f04ddc4`): `rp1_valid_call_rescued` S 1.00, E 0.89, T 0.995,
W 1.00; `rp1_executes` S 1.00 (87), E 0.93 (172; 6 hit the 60 s cap); `rp1_usable` S 0.985,
E 0.79, T 0.95, W 0.93. A18-T9's "recorded O120 rescued ≥ 0.95" fails only for E: 33 of 300
recorded E calls omit a nested required key (`args.argumentation` 21, `args.explanation` 12),
which the A04 strict (nested) reading counts as invalid; the top-level reading would pass them.
Henrik to choose the reading for A18 before any score is read (both are in `replay_tool`).

## 7. A17 F0 pools (X7 / A17-T7, A17-T13)

F0 rows live in `manifests/agent2027/A17_F0.csv` (70 rows: A17-101…177 minus the dropped O20 pool
143…149), never in a session file read by `ro27_clients_manifest.sh` (it would run them as Full-v2).
One row = one pool = 5 outer folds × 128 single-shot drafts of the frozen X7 prompt (Amendment 6),
through Rogue One's client wrapper, seeds `llm_seed·1000 + 128·f + j`, no tools, T 0.7,
`max_tokens` 32768. Folds are built as `rogue_one_main.py` builds them (D4 stratification on
`synp_p3/p4/p5`). Each row logs W&B run `A17__f0pool_<rung>__<ds>__f<seed>__r0` (group A17) with
an `f0-pool` artifact (`draws.jsonl`, `folds.json`, `prompts.jsonl`, `summary.json`). The generator
never executes model output; execution and scoring are A17-T8 (`f0_score.py`, not built).

Install (snapshot, so `~/code/RougeOne` is never checked out; image `agent2027-v2.0` has every
import, no rebuild needed):
```
scp scripts/agent2027/f0_generate.py scripts/agent2027/rungs.csv CAIR-gpu14:ro27_f0_code/scripts/agent2027/  # as *.new + mv
scp scripts/agent2027/f0_clients.sh CAIR-gpu14:ro27_f0_clients.sh                                            # as *.new + mv
scp manifests/agent2027/A17_F0.csv CAIR-gpu14:ro27_manifests/A17_F0.csv                                     # as *.new + mv
$CAIR "git -C ~/code/RougeOne rev-parse --short HEAD >/dev/null; echo <commit> > ~/ro27_f0_code/scripts/agent2027/F0_VERSION"
```
Dry run (host bash only):
```
$CAIR "F0_ROWS='A17-164 A17-171' HOST_SID=S1022h DRY_RUN=1 bash ~/ro27_f0_clients.sh"
```
O120 pools ride any running O120 TP2 session (here A16 Core-10 `S1022h`, Amendment 8):
```
$CAIR "F0_ROWS='A17-164 A17-165 A17-166 A17-167 A17-168 A17-169 A17-170 A17-171 A17-172 A17-173 A17-174 A17-175 A17-176 A17-177' HOST_SID=S1022h F0_CONCURRENCY=16 nohup bash ~/ro27_f0_clients.sh > ~/ro27_f0_S1022h.job.log 2>&1 < /dev/null & echo launched"
```
Rows run one after another; a row whose `summary.json` is complete is skipped; an incomplete row
is re-run once and resumes from `draws.jsonl`; `RUN_CAP_HOURS` (6) caps each row. The script
refuses a row whose rung does not match the server's model, so Q122 pools need a Q122 server
(e.g. `HOST_SID=S1022m` or A19's `S1022a`) and the small-rung pools their own TP1 servers.
Progress: `grep -E 'START ROW|ROW .* status=|F0_DONE' ~/ro27_f0_S1022h.job.log`; per-row
`~/ro27_f0_work/<row>/draws.jsonl` (one line per finished draw, 640 per row).

