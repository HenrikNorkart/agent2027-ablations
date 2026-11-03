"""Shared setup for the AGENT 2027 (E8a) unit tests."""

import os
import sys
from pathlib import Path

# rogueone.utils instantiates config objects at import time and refuses to start
# without these variables; the values are irrelevant for the CPU-only tests.
for _k, _v in {"EMBEDD_API_KEY": "x", "EMBEDD_MODEL": "x", "EMBEDD_PORT": "1", "LLM_MODEL": "x",
           "LLM_API_KEY": "x", "LLM_PORT": "1"}.items():
    os.environ.setdefault(_k, _v)

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
