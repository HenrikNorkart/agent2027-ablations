# RQ3 Capacity: do the roles have different capacity needs, and is per-role sizing what the split buys?

(Saved by the main session from the CAPACITY agent's final report; the agent could not write files. Scripts:
capacity_analysis.py, live_iteration_profile.py; tables in derived/; CAIR read-only copies in cair_a18/, cair_live/.)

**Everything here is exploratory** (post hoc, outside the pre-registered Holm family). Only the quoted C2a, C2b,
C5a, C5b p-values are confirmatory. The A18 replay was registered but sits outside the family.

Verdict: half supported.
- Supported: roles differ in what they need; the Tester can be cheap (strongly as code, C2a/C2b; weakly as a small
  LLM, format-level replay).
- Not supported as a measured result: (1) no live run had big Scientist+Extractor with a small LLM Tester; (2) both
  live mixed configurations lost recovery; (3) the Extractor has the largest token volume and the highest floor;
  (4) a single agent could also switch models between phases; what the split provides is short role contexts.

## 1. Role x evidence
| Role | Completion share Full | Prompt share | Calls; tokens/call | A18 floor (rp1_usable) | Live evidence | Error profile Full |
|---|---|---|---|---|---|---|
| Scientist | 0.086 (syn .089, real .082) | 0.086 | 14%; 352 | 4B, stability 1.00; non-monotone (0.8B passes .82 vs .77 threshold, 2B fails .58) | mix_1b (S+T on 0.8B): 0/60, 0/14 runs finished; 0.8B Scientist 10.0 runner errors/iter (context overflow) vs 0.07 at 122B; ladder S errors/iter 8.5/14.5/21.1/19.2/0.07 | 0.4% tool calls fail (14/3,539) + 300 runner 400s in 30 runs |
| Extractor | 0.442 (syn .431, real .443) | 0.743 | 69%; 355 | above 122B (best .21 vs gpt-oss .74); with decoded args 4B but stability .44; executed share .00-.02 -> .52-.92 decoded | mix_esmall (E on 4B): 11/60 vs 20/60 (C5b Holm p=.035), 7/12 crashed/capped vs 2/12; 72% duplicate names (13% at 122B); 82% of 20,182 errors "already exists" | 2.7% fail (957/34,955) |
| Tester | 0.476 (assess .210, prune .257) | 0.170 | 17%; 1,549 | <=0.8B on format/coverage; prune Jaccard vs gpt-oss recorded: re-sample .62, 0.8B .39, 2B .46, 4B .48, 9B .53, 122B .44 | No live arm moves only the Tester. In mix_1b 0.8B Tester prunes nothing in 35% of iters (7% at 122B), active table 234 vs 35, 122B Extractor prompt x8.7. As code (A12 fx_Paj*): 44/100 vs 42/100 | 7.2% fail (669/9,315) |

Compute (Full, 30 synthetic traces): completion/prompt/LLM-time shares — Scientist focus .044/.057/.045; web search
(S) .039/.030/.040; Extractor append .458/.736/.452; web search (E) .002/.007/.002; Tester assessment .204/.087/.207;
Tester pruning .254/.083/.254. Wall shares incl. tools/XGBoost: S .10, E .43, T .48.

## 2. Floors
A18 replay: 800 recorded gpt-oss-120b turns from 18 Full runs (seed 42); S 200, E 300, T 100+100, search 100.
Single next message, no tools run; code executed offline. Measures validity/usability, not agreement.
Floor rule RP1-D1: rung passes if rate >= 0.8 x gpt-oss self-replay; floor = smallest passing rung with all larger passing.

| Role | recorded | replay | 0.8B | 2B | 4B | 9B | 122B | Floor (stab.) |
|---|---|---|---|---|---|---|---|---|
| Scientist | 197/200 | 193 | 165 | 117 | 180 | 191 | 177 | 4B (1.00) |
| Extractor | 236/300 | 223 | 13 | 64 | 40 | 45 | 39 | >122B (1.00) |
| Extractor decoded (hq315) | 236 | 223 | 101 | 177 | 204 | 187 | 224 | 4B (0.44) |
| Tester | 190/200 | 191 | 185 | 180 | 190 | 195 | 188 | <=0.8B (1.00) |
| Web search | 93/100 | 91 | 90 | 92 | 88 | 88 | 89 | <=0.8B |

Limits: Tester score format/coverage only (Jaccard .39 vs .62); single-turn replay hides compounding failures;
Scientist floor rests on 2B failing while 0.8B passes. PAPER FLAG: v2_results.tex line 70 omits the Extractor floor.

A17 ladder recovery: 0.8B 0/70, 2B 0/35, 4B 11/70, 9B 0/35, 122B 20/70; Full 30/90. C5a 0/14, 1/7, 8/14, 0/7, 10/14.
All three roles degrade at small rungs; the ladder alone does not separate them.
PARSER FLAG: the "1-2% -> 52-92%" figure is from the replay (v2.0 wrapper). Live 4xx runs had
ROGUE_ONE_INLINE_TOOL_SCHEMA_REFS on (Amendment 17); live mix_esmall Extractor errors are 82% duplicate names, 5%
context length, no string-argument errors among top kinds. Parser explanation supported for replay, not shown for
the live C5b gap.

A19: mix_esmall 11/60 vs 20/60 (exact p=.0038); E wall share .67 (.24 at 122B), E prompt x3.9.
mix_1b 0/60 (p=3.1e-9), never generated, 0/14 finished; 0.8B Tester under-prunes -> 122B Extractor prompt x8.7,
915 s/iter (x12); 0.8B Scientist overflows 131k context (~800 errors). A small Tester is not role-local.
A20 cells Full gpt-oss / minimal gpt-oss / Full 4B / minimal 4B: 9/30, 5/30, 5/30, 0/30; interaction p=.29.
Gap check over 41 ablation:arm values: no live arm runs big S+E with small LLM Tester. Closest: Tester as code.

## 3. Full vs single agent (30 vs 27 synthetic runs)
| Measure | Full | Single | Ratio |
|---|---|---|---|
| Max prompt per run, median (traces) | 28,326 | 47,654 | 1.68 |
| Median prompt/call Extractor | 7,920 | 13,487 | 1.70 |
| Median prompt/call Tester | 7,529 | 23,773 (p90 35,418; max 101,107) | 3.16 |
| Median prompt/call Scientist | 3,515 | 3,172 | 0.90 |
| Prompt tokens per run | 18.9M | 32.0M | 1.69 |
| Completion tokens per run | 1.33M | 0.93M | 0.70 |
| Tester completion per run | 633k | 228k | 0.36 |
| Tester tool-error rate | 7.2% | 0.2% | |
| Wall share S/E/T | .10/.43/.48 | .22/.49/.28 | |
Single agent is not cheaper on FLOP proxy: 3.4e17 vs 2.0e17.

## 4. Cost projection (projection, not measurement)
cost = sum over roles N(model) x tokens; N = active params (pareto.py line 55; gpt-oss-120b 5.1B, Qwen 122B-A10B 10B)
or total params. Code costs 0. Assumes tokens unchanged by model size — FALSE in data (run prompt totals 18.3M
gpt-oss, 89M 122B, 369M 4B).
| Scenario | Tokens off big model | Saving active P+C | active C only | total params P+C |
|---|---|---|---|---|
| A floors on gpt-oss (S 4B, E gpt-oss, T 0.8B) | 27.6% | 17.9% | 40.4% | 27.2% |
| D Tester 0.8B only | 19.0% | 16.0% | 38.6% | 18.9% |
| E Tester as code | 19.0% | 19.0% | 45.8% | 19.0% |
| F Tester code + S 4B | 27.6% | 20.9% | 47.6% | 27.3% |
| B floors on 122B | 27.6% | 22.7% | 47.1% | 27.2% |
| C decoded floors, E 4B (unstable) | 100% | 66.1% | 74.6% | 97.2% |
Measured check: Tester as code completion 0.80M vs 1.41M (-43%), FLOP proxy -17% (projected -46%, -19%).
mix_esmall vs all-122B FLOP -31% (projected -43%) with recovery 11/60 vs 20/60; Qwen arms cost 6-9x Full.

## 5. Literature (verified on arXiv 2026-11-07)
- Belcak et al. "Small Language Models are the Future of Agentic AI." arXiv:2506.02153, 2025.
- Chen, Zaharia, Zou. "FrugalGPT." arXiv:2305.05176, 2023.
- Ong et al. "RouteLLM." arXiv:2406.18665, 2024.
Difference: those route per query; we assign per role and roles share state (mix_1b shows coupling).

## 6. Shown / not shown
Shown: role workloads differ; replay floors order E > S > T; Tester replaceable by code (-43% completion, -17% FLOP);
both live mixes lost recovery; split keeps role contexts short (Tester prompts x3.2 in single agent).
Not shown: split enables per-role sizing live; sizing requires a split; 0.8B Tester makes the same decisions;
sizing saves much (~18% projected; live costs rose); live C5b gap is a parser artefact.

## 7. Candidate sentences
1. "The roles load the model differently: in Full the Tester writes about half of all completion tokens (0.48) in few,
long calls, while the Extractor makes 69% of the calls and reads 74% of the prompt tokens. Replaying recorded prompts
role by role puts the Tester's floor at or below 0.8B and the Scientist's at 4B; no Qwen size reaches the Extractor's
(exploratory)."
2. "We could not show that per-role sizing pays in the live loop: both mixed configurations we ran lowered recovery
(Extractor on 4B: 11/60 vs 20/60; Scientist and Tester on 0.8B: 0/60), and no run paired a small Tester with large
Scientist and Extractor. The cheaper option we did confirm is to replace the Tester's LLM stages with code (A4, A5),
which cut completion tokens by 43% at no detectable loss."
3. "What separate agents do provide is short role contexts: in the single agent the Tester phase sees prompts 3.2
times as long (median 23.8k vs 7.5k tokens), a context in which small models failed in our ladder runs (exploratory)."

## 8. Sources
T1 derived/t1_full_role_shares_rows_final.csv (rows_final A05 full OK, 69); T2 t2_live_arm_role_profile.csv; T3-T5,
S1 from local traces and iteration_log; T6b/c agent_toolcall_errors + traces; T9 t9_cost_projection.csv; T10
t10_measured_flop_proxy_live.csv; R1-R3 cair_a18/final_1102/a18_verdicts/*.csv (CAIR ~/ro27_analysis/final_1102/
a18_verdicts/); F1 cair_a18/funnel_episodes.csv; L1-L4 cair_live/ (S1027*/A17-4*, A19-4*); P1 final_readout.json;
docs/ablations/A18_RP1_rolewise_replay.md (69-76, s4), A19_CAP2_role_mixing.md (22-27), preregistration.md
(1372-1394, 1476-1480), pareto.py (55), v2_results.tex (62, 70).
Coverage limits: trace numbers synthetic only; live profiles pool all attempt dirs; mix_1b has no token totals.
