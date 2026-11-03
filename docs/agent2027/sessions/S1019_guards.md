# Guard 1 on real data, images v2.1–v2.4 (2026-10-19)

Rule (01 §2.2 guard 1): with every post-freeze switch at its default, the Full-v2 prompts
rendered by a real run on the live O120 server equal `prompt-baseline-v2.0` (the `S1016t`
A05-901 dump, `synp_p3`, one fold) byte for byte. Compared: every LLM-independent prompt file
present in both dumps (all role system prompts and `tools.json` files) plus the Scientist's
round-1 user message. Later user messages depend on LLM output and are not compared.

| Image | Guard row | Session | W&B run | Files compared | Identical | Scientist round-1 user |
|---|---|---|---|---|---|---|
| `agent2027-v2.1` (`d21b8fa`, E9c) | A09-000 | S1019a | `wo6jlzmg` | 69 | 69 | identical |
| `agent2027-v2.2` (`3754252`, + E9a/E9b) | A11-015 | S1020s | `k9hx13m7` | 80 | 80 | identical |
| `agent2027-v2.3` (`4d4adcc`, + E9d/E9e/E7/E8b) | A12-065 | S1021b | `actdu72j` | 80 | 80 | identical |
| `agent2027-v2.4` (`ca92424`, + E9f, detector) | A16-027 | S1022b | `6niht5l8` | 80 | 80 | identical |

All four pass. (A09-000's run-output download was partial on the Windows W&B cache, hence 69
files; every file present matched.) A16-028 (`heart`) is the second v2.4 guard; the local
baseline copy holds only the `synp_p3` dump, so it is compared when the `heart` baseline is
fetched.

Smoke sessions, all rows OK, 0 failed, 0 cap hits: `S1020s` (3/3, v2.2), `S1021b` (14/14,
v2.3: 8 FX cells, Full guard, A13 off, A14 off/parametric/local-correct/local-wrong on the
DRAFT corpus), `S1022b` (5/5, v2.4: A16 smoke, 2 Full guards, crash-only pilot `blood` and
`communities`).
