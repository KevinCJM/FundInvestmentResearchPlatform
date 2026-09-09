"""Point-in-time discipline: declarations, measurement, releases and context.

This package is reachable under two names — `pit` when the app runs with
`backend/` on the path, and `backend.pit` from the test suite and any
`backend.*` module. Left alone, Python would build a *separate* copy per
spelling, which breaks two things that matter:

* `audit._CACHE` is per copy, so a cache warmed at startup is a cold 37M-row
  rescan for the other copy;
* `PitContextError` is a different class per copy, so `except PitContextError`
  written against one spelling silently fails to catch the other's exception.

Aliasing every submodule under both names at import time keeps exactly one
instance in the process. `__init__` always executes before any submodule import,
so the first spelling in wins and the second resolves to the same objects.
"""

import sys

from . import audit, catalog, clock, context, frame, guard, release, settings, universe  # noqa: F401  (imported to alias)
from .catalog import (
    DATASETS,
    SNAPSHOT_FIELD,
    DATASETS_BY_ID,
    GRADE_APPROXIMATE,
    GRADE_NONE,
    GRADE_STRICT,
    RUN_MODE_RESEARCH,
    RUN_MODE_STRICT,
    RUN_MODES,
    DatasetPitDeclaration,
    grade,
    is_usable_under,
)
from .context import (
    PitContextError,
    ResearchContext,
    build_context,
    parse_view_override,
    require_usable,
    reset_view_override,
    resolve,
    resolve_request_context,
    set_view_override,
    view_override,
)
from .clock import DecisionClock, availability_from_rows, visible_at
from .frame import LATEST_ONLY, REPLAYED, PitFrame, read_pit
from .guard import assert_no_universe_lookahead, check_universe, universe_lineage
from .settings import PitSettingsRepository, release_as_of
from .universe import INTERVAL, UniverseView, universe_as_of

_PACKAGE_ALIASES = ("pit", "backend.pit")
_SUBMODULES = (
    "audit",
    "catalog",
    "clock",
    "context",
    "frame",
    "guard",
    "release",
    "settings",
    "universe",
)

for _alias in _PACKAGE_ALIASES:
    sys.modules.setdefault(_alias, sys.modules[__name__])
    for _submodule in _SUBMODULES:
        sys.modules.setdefault(f"{_alias}.{_submodule}", sys.modules[f"{__name__}.{_submodule}"])

__all__ = [
    "DATASETS",
    "DecisionClock",
    "INTERVAL",
    "LATEST_ONLY",
    "PitFrame",
    "REPLAYED",
    "SNAPSHOT_FIELD",
    "UniverseView",
    "assert_no_universe_lookahead",
    "availability_from_rows",
    "check_universe",
    "clock",
    "frame",
    "guard",
    "read_pit",
    "universe",
    "universe_as_of",
    "universe_lineage",
    "visible_at",
    "DATASETS_BY_ID",
    "GRADE_APPROXIMATE",
    "GRADE_NONE",
    "GRADE_STRICT",
    "RUN_MODES",
    "RUN_MODE_RESEARCH",
    "RUN_MODE_STRICT",
    "DatasetPitDeclaration",
    "PitContextError",
    "ResearchContext",
    "audit",
    "build_context",
    "catalog",
    "context",
    "grade",
    "is_usable_under",
    "parse_view_override",
    "PitSettingsRepository",
    "release",
    "release_as_of",
    "require_usable",
    "reset_view_override",
    "resolve",
    "resolve_request_context",
    "set_view_override",
    "settings",
    "view_override",
]
