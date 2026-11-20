# MAST codebook for Rogue One hypothesis cycles

Source: Cemri et al., "Why Do Multi-Agent LLM Systems Fail?" (arXiv 2503.13657), MAST taxonomy, 14 failure
modes in 3 categories. Definitions below paraphrase the paper; the operationalisation lines are ours, for this
system, and must be applied identically by every annotator.

Unit of annotation: ONE hypothesis cycle (one iteration of one fold): the planner turn (Scientist: reads results,
proposes a focus), the coder turn (Extractor: writes features as pandas code via tools), the evaluator turn
(Tester: written assessment + pruning). Packet roles are labelled PLANNER / CODER / EVALUATOR.
Label each mode 1 (clearly present, with a quoted piece of evidence) or 0. When unsure, 0. Several modes can be 1.

Task objective known to all roles: engineer features from a raw table that improve cross-validated prediction.
The packet also states (for the annotator only, never shown to the agents) the planted function phi. Use phi only
for the "missed signal" notes, not to decide MAST labels, except where a mode's operationalisation says so.

## FC1 System design issues
- FM-1.1 Disobey task specification: output violates task constraints/requirements.
  Op: coder writes features that break stated rules (uses the label/target, leaks test data, ignores required tool
  format), or planner proposes something outside the task (e.g. model tuning).
- FM-1.2 Disobey role specification: an agent acts outside its role, behaving like another agent.
  Op: planner writes feature code itself; coder chooses its own new focus instead of implementing the given one;
  evaluator proposes new hypotheses instead of assessing.
- FM-1.3 Step repetition: needlessly redoes completed steps.
  Op: re-creates features already in the pool, repeats the same tool call/query with no change, proposes a focus
  essentially identical to an earlier one without new reason.
- FM-1.4 Loss of conversation history: recent context dropped, reverting to an earlier state.
  Op: an agent acts as if unaware of something stated earlier IN THIS CYCLE or the provided history (e.g. coder
  re-asks/ignores the focus details visible in its input; planner contradicts the latest results it was given).
- FM-1.5 Unaware of termination conditions: does not recognise when to stop, continues unnecessarily.
  Op: loops of tool calls far beyond need, keeps appending after its stated plan is done, hits round/iteration caps.

## FC2 Inter-agent misalignment (apply to hand-offs between roles, also when roles share one conversation)
- FM-2.1 Conversation reset: restarts dialogue unexpectedly, losing context/progress.
  Op: a role starts over from scratch mid-cycle (e.g. coder discards its own progress and restarts).
- FM-2.2 Fail to ask for clarification: proceeds on unclear/incomplete input instead of resolving it.
  Op: focus is ambiguous/underspecified (missing columns or operation) and the coder guesses without checking the
  data or history. (In this system agents cannot message each other back; count it when the receiving role neither
  resolves the ambiguity with available tools nor flags it.)
- FM-2.3 Task derailment: drifts from the intended objective into irrelevant work.
  Op: the cycle's work goes to things unrelated to the focus or to prediction (cosmetic recoding, spending the cycle
  on errors/debugging unrelated to the hypothesis).
- FM-2.4 Information withholding: an agent has information others need and does not pass it on.
  Op: the planner's own reasoning/tool output contains a specific, promising lead (named columns, operation,
  threshold, periodicity, ratio, interaction) that is absent or blurred in the focus message the coder receives; or
  the evaluator's report omits a key result the planner then lacks.
- FM-2.5 Ignored other agent's input: disregards input from another agent.
  Op: coder does not implement (some of) the focus it received; planner ignores the evaluator's explicit
  recommendation without reason.
- FM-2.6 Reasoning-action mismatch: stated reasoning does not match actions.
  Op: an agent says it will build X and builds Y; evaluator says a feature is strong and prunes it (or vice versa).

## FC3 Task verification
- FM-3.1 Premature termination: ends before objectives/necessary exchange are met.
  Op: coder stops after few/no successful features while the focus still has unimplemented parts; evaluator stops
  without assessing.
- FM-3.2 No or incomplete verification: checks skipped or partial.
  Op: features accepted without checking they compute (NaN/constant/inf), evaluator assesses without looking at
  the metrics provided.
- FM-3.3 Incorrect verification: verification done but wrong.
  Op: evaluator misreads metrics (calls a weak feature strong or the reverse against the numbers it was given),
  prunes a feature that its own numbers support.

## Additional (not MAST, record separately)
- MISSED_SIGNAL (0/1): the cycle contained a concrete pointer toward phi (in data summaries, results, or an agent's
  reasoning) that was not acted on. Quote it.
- HANDOFF_FIDELITY (0-2): how faithfully the coder's actual features implement the planner's focus
  (0 = unrelated, 1 = partial, 2 = complete).

Output per unit: JSON object {unit_id, labels: {FM-1.1: 0/1, ..., FM-3.3: 0/1}, evidence: {mode: "quote"},
missed_signal: 0/1, missed_signal_quote, handoff_fidelity: 0/1/2, notes}.
