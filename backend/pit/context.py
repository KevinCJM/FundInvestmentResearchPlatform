"""The one research context every downstream computation reads.

`as_of` says which day the research pretends to stand on, `run_mode` says how
hard that pretence is enforced, and `data_release_id` pins which vintage of the
data answered the question.  Every page used to carry its own date box; this is
the single object they all share instead.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

# Relative so this module always shares one audit cache with its own package
# copy; see the note in `audit.py`.
from .audit import audit_all
from .catalog import (
    GRADE_LABELS,
    GRADE_NONE,
    RUN_MODE_RESEARCH,
    RUN_MODE_STRICT,
    RUN_MODES,
    is_usable_under,
)


class PitContextError(ValueError):
    """Raised when a research context cannot be honoured as asked."""


@dataclass(frozen=True)
class ResearchContext:
    """as_of=None means 'use everything on disk' — the pre-PIT behaviour."""

    as_of: Optional[str] = None
    run_mode: str = RUN_MODE_RESEARCH
    data_release_id: Optional[str] = None

    @property
    def strict(self) -> bool:
        return self.run_mode == RUN_MODE_STRICT


def parse_as_of(value: Any) -> Optional[pd.Timestamp]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = pd.to_datetime(text)
    except Exception as exc:  # noqa: BLE001 - surfaced as a 400
        raise PitContextError(f"研究日格式错误，应为 YYYY-MM-DD：{text}") from exc
    if pd.isna(parsed):
        raise PitContextError(f"研究日格式错误，应为 YYYY-MM-DD：{text}")
    return parsed.normalize()


def build_context(
    as_of: Any = None,
    run_mode: Any = None,
    data_release_id: Any = None,
) -> ResearchContext:
    mode = str(run_mode or RUN_MODE_RESEARCH).strip().upper() or RUN_MODE_RESEARCH
    if mode not in RUN_MODES:
        raise PitContextError(f"不支持的运行模式：{run_mode}")
    parsed = parse_as_of(as_of)
    release = str(data_release_id or "").strip() or None
    if mode == RUN_MODE_STRICT and parsed is None:
        # Strict mode without an as_of is a contradiction: there is no cut-off to
        # enforce, so it would quietly behave exactly like research mode.
        raise PitContextError("严格 PIT 模式必须指定研究日。")
    return ResearchContext(
        as_of=parsed.strftime("%Y-%m-%d") if parsed is not None else None,
        run_mode=mode,
        data_release_id=release,
    )


# --------------------------------------------------------------------------- #
# per-tab viewing override
# --------------------------------------------------------------------------- #

# The口径 one browser tab asked to *view* under, for the current request only.
#
# The system setting is the default, not a cage: an analyst reading a page has to
# be able to switch vintage or turn PIT off to look around, without changing what
# anyone else sees. It rides on request headers rather than every request body
# because the alternative is threading three fields through every endpoint that
# reads data — and forgetting one silently computes on the wrong口径.
_VIEW_OVERRIDE: contextvars.ContextVar[Optional[ResearchContext]] = contextvars.ContextVar(
    "pit_view_override", default=None
)

VIEW_OFF_VALUES = {"1", "true", "yes", "on"}


def view_override() -> Optional[ResearchContext]:
    """The current request's viewing口径, or None if it asked for none."""

    return _VIEW_OVERRIDE.get()


def set_view_override(override: Optional[ResearchContext]) -> contextvars.Token:
    """Returns the token to hand back to :func:`reset_view_override`."""

    return _VIEW_OVERRIDE.set(override)


def reset_view_override(token: contextvars.Token) -> None:
    _VIEW_OVERRIDE.reset(token)


def parse_view_override(
    data_dir: Path, headers: Mapping[str, str]
) -> Optional[ResearchContext]:
    """Read the per-tab viewing口径 out of request headers.

    `None` means the request said nothing and so inherits the system setting.
    `X-Pit-Off` is the explicit "show me every row on disk", which saying nothing
    can never mean — that distinction is the whole reason this is a header and
    not an absent value.
    """

    from .settings import PitSettingsRepository, release_as_of

    if str(headers.get("x-pit-off") or "").strip().lower() in VIEW_OFF_VALUES:
        return ResearchContext()
    release = str(headers.get("x-pit-release") or "").strip()
    mode = str(headers.get("x-pit-run-mode") or "").strip().upper()
    as_of = str(headers.get("x-pit-as-of") or "").strip()
    if not (release or mode or as_of):
        return None
    if release and not as_of:
        # A release already knows the last day it can answer for; making the tab
        # carry the date as well would let the two drift apart.
        as_of = release_as_of(data_dir, release) or ""
        if not as_of:
            raise PitContextError(f"数据版本 {release} 没有可得截止日，无法作为查看口径。")
    if not mode:
        # Never quietly relax a strict system口径 just because a tab switched
        # vintage and said nothing about the mode.
        mode = PitSettingsRepository(data_dir).effective_context().run_mode
    return build_context(as_of or None, mode, release or None)


def resolve_request_context(
    data_dir: Path,
    as_of: Any = None,
    run_mode: Any = None,
    data_release_id: Any = None,
) -> ResearchContext:
    """The context a request actually runs under.

    Three layers, weakest first: the system-level PIT setting, the viewing
    override this tab asked for (see :func:`parse_view_override`), then whatever
    the caller states here. A request that says nothing still computes on the
    right口径 — the safe default, and the reason this is resolved server-side
    rather than assembled by each page — while a stated value always wins, which
    is what lets a backtest sweep `as_of` past either default.
    """

    # Imported here: settings imports context for ResearchContext/PitContextError,
    # so a module-level import would close the cycle.
    from .settings import PitSettingsRepository

    base = view_override()
    if base is None:
        base = PitSettingsRepository(data_dir).effective_context()
    stated_as_of = str(as_of or "").strip()
    stated_mode = str(run_mode or "").strip().upper()
    stated_release = str(data_release_id or "").strip()
    if not stated_as_of and not stated_mode and not stated_release:
        return base
    return build_context(
        stated_as_of or base.as_of,
        stated_mode or base.run_mode,
        stated_release or base.data_release_id,
    )


def resolve(data_dir: Path, context: ResearchContext) -> dict[str, Any]:
    """Answer 'what can I actually use under this context?' before any run."""

    audit = audit_all(data_dir)
    blocked: list[dict[str, Any]] = []
    warned: list[dict[str, Any]] = []
    for item in audit["datasets"]:
        if not item["present"]:
            continue
        entry = {
            "dataset_id": item["dataset_id"],
            "label": item["label"],
            "grade": item["grade"],
            "grade_label": item["grade_label"],
            "available_through": item.get("available_through"),
            "reason": item["note"],
        }
        if not is_usable_under(context.run_mode, item["grade"]):
            blocked.append(entry)
        elif item["grade"] != "A":
            warned.append(entry)

    stale: list[dict[str, Any]] = []
    if context.as_of:
        cutoff = pd.Timestamp(context.as_of)
        for item in audit["datasets"]:
            end = item.get("available_through")
            if item["present"] and end and pd.Timestamp(end) < cutoff:
                stale.append(
                    {
                        "dataset_id": item["dataset_id"],
                        "label": item["label"],
                        "available_through": end,
                    }
                )

    return {
        "context": {
            "as_of": context.as_of,
            "run_mode": context.run_mode,
            "run_mode_label": RUN_MODES[context.run_mode],
            "data_release_id": context.data_release_id,
        },
        "summary": audit["summary"],
        "blocked_datasets": blocked,
        "degraded_datasets": warned,
        # Not an error: a dataset whose latest data predates the research day
        # still answers correctly, it just cannot see the last few sessions.
        "stale_datasets": stale,
        "usable": not blocked,
    }


def require_usable(data_dir: Path, context: ResearchContext, dataset_ids: list[str]) -> None:
    """Gate a run on the datasets it is about to read. Strict mode fails closed."""

    if not context.strict:
        return
    audit = audit_all(data_dir)
    by_id = {item["dataset_id"]: item for item in audit["datasets"]}
    offenders = [
        by_id[dataset_id]
        for dataset_id in dataset_ids
        if dataset_id in by_id
        and by_id[dataset_id]["present"]
        and not is_usable_under(context.run_mode, by_id[dataset_id]["grade"])
    ]
    if offenders:
        names = "、".join(f"{item['label']}（{GRADE_LABELS[item['grade']]}）" for item in offenders)
        raise PitContextError(
            f"严格 PIT 模式禁止使用无时点能力的数据集：{names}。"
            "请改用研究模式，或为该数据集补充可得时间列。"
        )


__all__ = [
    "GRADE_NONE",
    "VIEW_OFF_VALUES",
    "parse_view_override",
    "reset_view_override",
    "resolve_request_context",
    "set_view_override",
    "view_override",
    "PitContextError",
    "ResearchContext",
    "build_context",
    "parse_as_of",
    "require_usable",
    "resolve",
]
