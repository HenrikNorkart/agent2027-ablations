"""Import-time compatibility shims for the CAAFE image (docker/Dockerfile.caafe).

Installed as sitecustomize.py so every shim is in place before ANY other import.

Both shims below exist for the same reason and both are NUMERICALLY INERT: CAAFE
(2023) and tabpfn 0.1.11 (2023) call two library APIs that were later renamed or
removed. The alternative to shimming them is to downgrade numpy/scikit-learn below
the pins in docker/Dockerfile.classicfe -- and those pins are the reason
paper/scripts/caafe_arm.py can import classicfe_arm.evaluate() and produce cells
scored by the identical estimator as the openfe/autofeat/raw cells. Keeping the
scoring identical is worth more than avoiding two aliases, so the aliases are
restored instead. Neither shim changes a number, a default, or a code path.
"""

import warnings as _warnings

# ---------------------------------------------------------------------------------
# 1. numpy removed the `np.warnings` alias in 1.24. tabpfn 0.1.11's
#    scripts/tabular_baselines.py still calls np.warnings.filterwarnings(...) at
#    import time -- and that module is where CAAFE's own default internal evaluator
#    (iterative_method="logistic" -> logistic_metric) lives. `np.warnings` was never
#    anything but a re-export of the stdlib warnings module, so restoring it is
#    exactly the original semantics.
# ---------------------------------------------------------------------------------
try:
    import numpy as _np

    if not hasattr(_np, "warnings"):
        _np.warnings = _warnings
except Exception:  # pragma: no cover - never block interpreter startup
    pass

# ---------------------------------------------------------------------------------
# 2. scikit-learn renamed OneHotEncoder's `sparse` argument to `sparse_output` in
#    1.2 and deleted the old name in 1.4. tabpfn 0.1.11 constructs
#    OneHotEncoder(sparse=False) inside its preprocessing path, so under our pinned
#    sklearn 1.5.2 the first internal evaluation dies with
#    "OneHotEncoder.__init__() got an unexpected keyword argument 'sparse'".
#
#    `sparse=X` and `sparse_output=X` select the same behaviour -- the rename was
#    cosmetic, to disambiguate from other estimators' `sparse` -- so forwarding one
#    to the other reproduces the 2023 behaviour exactly. Only the keyword is
#    translated; everything else goes through untouched, and if the running sklearn
#    still accepts `sparse` natively the shim does not install itself at all.
# ---------------------------------------------------------------------------------
try:
    import inspect as _inspect

    import sklearn.preprocessing as _skp

    _OHE = _skp.OneHotEncoder
    if "sparse" not in _inspect.signature(_OHE.__init__).parameters:

        class _OneHotEncoderSparseCompat(_OHE):  # noqa: D101
            def __init__(self, *args, sparse="__unset__", **kwargs):
                if sparse != "__unset__":
                    kwargs["sparse_output"] = sparse
                super().__init__(*args, **kwargs)

        _OneHotEncoderSparseCompat.__name__ = "OneHotEncoder"
        _OneHotEncoderSparseCompat.__qualname__ = "OneHotEncoder"
        _skp.OneHotEncoder = _OneHotEncoderSparseCompat
except Exception:  # pragma: no cover - never block interpreter startup
    pass
