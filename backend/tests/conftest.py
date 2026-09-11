"""Make ``backend/`` importable for every test module, not just the lucky ones.

Most test modules import bare backend packages (``services``, ``compute_policy``,
``historical_regimes``) while only some of them put ``backend/`` on ``sys.path``
themselves.  That made collection order decide which modules could import at
all, so the same suite failed on a different set of tests depending on what ran
first.  Doing it once here removes the ordering dependency.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT / "backend"

for path in (ROOT, BACKEND_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Tests import kernels as ``backend.<pkg>.<mod>`` while ``uvicorn app:app`` run
# from ``backend/`` imports the same sources as ``<pkg>.<mod>``.  Numba records
# the importing module name inside each cached environment, so a cache written
# under one naming makes the other entry point fail with a misleading
# ``ModuleNotFoundError: No module named 'backend'`` raised from deep inside
# numba's cache loader.  Give the test run its own cache directory so the two
# namings can never poison each other.  Must be set before numba is imported.
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache" / "tests"))
