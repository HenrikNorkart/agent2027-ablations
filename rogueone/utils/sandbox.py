"""Least-privilege exec scopes for LLM-authored code (AGENT 2027 E13, ``ROGUE_ONE_SANDBOX``).

With ``ROGUE_ONE_SANDBOX=on`` the generation-side ``exec`` scopes (Extractor
``exec_user_command``, Scientist ``generic_pandas_tool``, Tester
``generic_python_executor_tool``) run with:

* a restricted ``__builtins__``: no ``open``, ``exec``, ``eval``, ``compile``,
  ``input``, ``breakpoint`` (and no ``exit`` / ``quit`` / ``help``, which are
  ``site`` helpers whose ``SystemExit`` would bypass the tools' ``except
  Exception`` handlers);
* an allow-listed ``__import__``: ``import`` statements still work for the
  numerical stack (pandas, numpy, sklearn, scipy, statsmodels, shap, xgboost) and
  a few pure-computation stdlib modules, but ``os``, ``sys``, ``io``, ``pathlib``,
  ``subprocess``, ``builtins``, ``pickle`` ... raise ``ImportError``;
* the injected modules (``pd``, ``np``, ``spy``, ``sm``, ``sklearn``) replaced by
  thin proxies that hide their file-reading entry points (``pd.read_*``,
  ``pd.io``, ``np.load``/``np.loadtxt``/``np.genfromtxt``/``np.fromfile`` ...,
  ``scipy.io``, ``statsmodels.iolib``, ``sklearn.datasets``). Everything else is
  delegated to the real module, so ``pd.DataFrame``, ``pd.concat``,
  ``np.where``, ``df.groupby('id')...`` behave exactly as before.

With ``ROGUE_ONE_SANDBOX=off`` (the code default) :func:`sandbox_scope` returns
the scope dict untouched, so the exec call is byte-identical to the pre-E13 code.

**This is not a security boundary.** In-process Python cannot be sandboxed
against a determined adversary: attribute traversal (``f.__globals__``,
``().__class__.__mro__[1].__subclasses__()``), modules reachable as attributes of
allowed modules (e.g. ``np.lib.npyio``, joblib under sklearn), and APIs that take
a path (``xgb.DMatrix('<path>')``) remain open. The measure removes the direct,
obvious channels (``open``, ``pd.read_csv``, ``import os``) so that the
least-privilege claim of 01 §E13 holds for ordinary model output, and
``scan_leakage.py`` still audits outputs.
"""

from __future__ import annotations

import builtins
import os
import types
from typing import Any, Iterable

#: Environment flag (01 §7 R4). Code default ``off``.
SANDBOX_ENV = "ROGUE_ONE_SANDBOX"
_ON_VALUES = frozenset({"on", "1", "true", "yes"})

#: Builtins removed from generation-side scopes when the sandbox is on.
BLOCKED_BUILTINS: frozenset[str] = frozenset(
    {"open", "exec", "eval", "compile", "input", "breakpoint", "exit", "quit", "help"}
)

#: Top-level packages that ``import`` may load inside a sandboxed scope.
ALLOWED_IMPORT_ROOTS: frozenset[str] = frozenset(
    {
        # numerical stack (already injected into one or more scopes)
        "pandas", "numpy", "sklearn", "scipy", "statsmodels", "shap", "xgboost",
        # pure-computation stdlib
        "math", "cmath", "statistics", "random", "re", "string", "collections",
        "itertools", "functools", "operator", "warnings", "typing", "datetime",
        "decimal", "fractions", "heapq", "bisect", "copy", "dataclasses", "enum",
        "numbers", "json",
    }
)

#: Per-package attributes (and submodules) hidden by the module proxies.
#: ``prefixes`` hide every attribute that starts with one of the strings.
_BLOCKED_MODULE_ATTRS: dict[str, dict[str, tuple[str, ...]]] = {
    "pandas": {
        "names": ("io", "HDFStore", "ExcelFile", "ExcelWriter"),
        "prefixes": ("read_",),
    },
    "numpy": {
        "names": (
            "load", "loadtxt", "genfromtxt", "fromfile", "fromregex", "memmap",
            "DataSource", "save", "savez", "savez_compressed", "savetxt",
        ),
        "prefixes": (),
    },
    "scipy": {"names": ("io",), "prefixes": ()},
    "statsmodels": {"names": ("iolib",), "prefixes": ()},
    "sklearn": {"names": ("datasets",), "prefixes": ()},
}


class SandboxViolation(AttributeError):
    """Raised when sandboxed code reaches for a blocked module attribute.

    An ``AttributeError`` so that ``hasattr`` is False and ``from pandas import
    read_csv`` surfaces as the usual ``ImportError: cannot import name``.
    """


def sandbox_enabled(environ: Any = None) -> bool:
    """True iff ``ROGUE_ONE_SANDBOX`` is ``on`` (also accepts 1/true/yes)."""
    env = os.environ if environ is None else environ
    return str(env.get(SANDBOX_ENV, "off")).strip().lower() in _ON_VALUES


def _is_blocked(root: str, attr: str) -> bool:
    rule = _BLOCKED_MODULE_ATTRS.get(root)
    if rule is None:
        return False
    return attr in rule["names"] or any(attr.startswith(p) for p in rule["prefixes"])


class GuardedModule(types.ModuleType):
    """Read-only proxy for a module that hides its file-I/O entry points."""

    def __init__(self, module: types.ModuleType, root: str):
        super().__init__(module.__name__, getattr(module, "__doc__", None))
        object.__setattr__(self, "_sandbox_module", module)
        object.__setattr__(self, "_sandbox_root", root)

    def __getattr__(self, name: str) -> Any:
        if _is_blocked(self._sandbox_root, name):
            raise SandboxViolation(
                f"'{self._sandbox_root}.{name}' is not available in this sandboxed "
                f"environment (file access is disabled; use the provided DataFrames)."
            )
        return getattr(self._sandbox_module, name)

    def __setattr__(self, name: str, value: Any) -> None:
        raise SandboxViolation(f"module '{self._sandbox_root}' is read-only in the sandbox")

    def __delattr__(self, name: str) -> None:
        raise SandboxViolation(f"module '{self._sandbox_root}' is read-only in the sandbox")

    def __dir__(self) -> Iterable[str]:
        return [n for n in dir(self._sandbox_module) if not _is_blocked(self._sandbox_root, n)]

    def __repr__(self) -> str:
        return repr(self._sandbox_module)


_GUARD_CACHE: dict[int, GuardedModule] = {}


def guard_module(obj: Any) -> Any:
    """Return a :class:`GuardedModule` proxy for packages with blocked attributes."""
    if isinstance(obj, GuardedModule) or not isinstance(obj, types.ModuleType):
        return obj
    root = obj.__name__.partition(".")[0]
    if root not in _BLOCKED_MODULE_ATTRS or obj.__name__ != root:
        return obj
    key = id(obj)
    proxy = _GUARD_CACHE.get(key)
    if proxy is None or proxy._sandbox_module is not obj:
        proxy = GuardedModule(obj, root)
        _GUARD_CACHE[key] = proxy
    return proxy


_REAL_IMPORT = builtins.__import__


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level != 0:
        raise ImportError("relative imports are not available in the sandbox")
    parts = name.split(".")
    root = parts[0]
    if root not in ALLOWED_IMPORT_ROOTS:
        raise ImportError(
            f"import of '{name}' is not available in this sandboxed environment"
        )
    if len(parts) > 1 and _is_blocked(root, parts[1]):
        raise ImportError(
            f"import of '{name}' is not available in this sandboxed environment"
        )
    module = _REAL_IMPORT(name, globals, locals, fromlist, level)
    # ``import pkg.sub`` / ``import pkg`` return the top-level package: hand back
    # the proxy so ``pkg.read_csv`` etc. stay hidden. ``from pkg.sub import x``
    # returns ``pkg.sub`` itself, which the check above already vetted.
    return guard_module(module)


def restricted_builtins() -> dict:
    """A fresh ``__builtins__`` dict without the blocked names."""
    safe = {k: v for k, v in vars(builtins).items() if k not in BLOCKED_BUILTINS}
    safe["__import__"] = _guarded_import
    return safe


def sandbox_scope(scope: dict, environ: Any = None) -> dict:
    """Apply the sandbox to an exec scope in place (and return it).

    ``off``: the dict is returned unchanged (no ``__builtins__`` key is added, so
    ``exec`` inserts the real builtins exactly as before E13).
    """
    if not sandbox_enabled(environ):
        return scope
    for key, value in list(scope.items()):
        scope[key] = guard_module(value)
    scope["__builtins__"] = restricted_builtins()
    return scope
