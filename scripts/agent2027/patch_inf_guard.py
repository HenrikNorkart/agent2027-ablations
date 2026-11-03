"""Apply the ROGUE_ONE_INF_GUARD change (repo commit) to a qfix source tree: python patch_inf_guard.py <root>."""
import sys
from pathlib import Path

root = Path(sys.argv[1])
tester = root / "rogueone/llm/agents/core/tester/agent.py"
config = root / "rogueone/utils/config.py"

HELPER = '''#: Run-total counter of fold matrices the inf guard had to clean.
INF_GUARD_COUNTER = "inf_guard"


def _inf_guard(x: Any) -> np.ndarray:
    """Clean a fold matrix the way the scoring fits already do (A17 relaunch).

    A feature replayed on one partition can be infinite there although it was finite on
    the full frame, and a float64 value beyond float32 range becomes infinite inside
    XGBoost. The scoring fits pass every matrix through ``np.nan_to_num(float32,
    posinf=1e6, neginf=-1e6)``, but the assessment fit and the pruning tools received
    the raw matrix, so one such value raised ``XGBoostError: Input data contains inf``
    and ended the run without a result. This applies the same clipping (NaN is left
    alone: XGBoost treats it as missing). Only under ``ROGUE_ONE_INF_GUARD``.
    """
    arr = np.asarray(x)
    try:
        arr = arr.astype(np.float32)
    except (TypeError, ValueError):
        return np.asarray(x)
    if np.isinf(arr).any():
        instrumentation.count(INF_GUARD_COUNTER)
        arr = np.nan_to_num(arr, nan=np.nan, posinf=1e6, neginf=-1e6)
    return arr


def _inner_cv_score('''

XF_OLD = "                x_folds = [d[feature_names].to_numpy() for d in df_attributes_folds]\n"
XF_NEW = XF_OLD + (
    "                if rouge_one_cfg.inf_guard:\n"
    "                    x_folds = [_inf_guard(x) for x in x_folds]\n"
)

CFG_OLD = '        self.loop_guard_repeats: int = int(os.getenv("ROGUE_ONE_LOOP_GUARD", "0") or 0)\n'
CFG_NEW = CFG_OLD + (
    "        # Inf guard (A17 relaunch: Q4/Q122 rows ended on XGBoostError \"Input data\n"
    "        # contains inf\"). Clips +-inf in the Tester's fold matrices to +-1e6, as the\n"
    "        # scoring fits already do. Off by default so O120 behaviour is unchanged.\n"
    '        self.inf_guard: bool = env_as_bool("ROGUE_ONE_INF_GUARD", False)\n'
)


def sub_once(text, old, new, what):
    if text.count(old) != 1:
        sys.exit(f"anchor not unique in {what}: {old[:60]!r} ({text.count(old)})")
    return text.replace(old, new)


t = tester.read_text(encoding="utf-8")
t = sub_once(t, "def _inner_cv_score(", HELPER, "tester")
t = sub_once(t, XF_OLD, XF_NEW, "tester")
tester.write_text(t, encoding="utf-8", newline="\n")
c = config.read_text(encoding="utf-8")
c = sub_once(c, CFG_OLD, CFG_NEW, "config")
config.write_text(c, encoding="utf-8", newline="\n")
print("patched", root)
