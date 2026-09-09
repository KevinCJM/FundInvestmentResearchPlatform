"""The one system-level PIT setting the whole platform reads.

PIT口径 is governance, not a per-page filter: it belongs next to the benchmark
convention and the risk-free-rate convention, set centrally and obeyed
everywhere. Two consequences follow, and both are the point of this module.

**It lives on the server.** Keeping it in each browser's `localStorage` means
two analysts open the same page, see different numbers, and neither can tell.

**The data release implies the research day.** A sealed release already knows
the last date it can honestly answer for (`available_through`), so pinning a
release fixes `as_of` too. That keeps one knob where users expect one knob,
while leaving `as_of` free to be overridden per request — a backtest sweeps it
by nature, and a system-wide single value would make that impossible.

No release applied means no PIT: every row on disk is fair game, exactly the
pre-PIT behaviour, and results say so rather than pretending otherwise.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

try:  # pragma: no cover - platform specific
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

from .catalog import RUN_MODE_RESEARCH, RUN_MODE_STRICT, RUN_MODES
from .context import PitContextError, ResearchContext
from .release import DataReleaseError, DataReleaseRepository

PIT_SETTINGS_STORE = "pit_settings.json"
SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class PitSettingsStore:
    """Atomic write plus an advisory lock, matching the release store."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock_path = path.with_suffix(path.suffix + ".lock")
        self._thread_lock = threading.RLock()

    @contextmanager
    def locked(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._thread_lock:
            with self.lock_path.open("a+", encoding="utf-8") as lock_file:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    if fcntl is not None:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def read_unlocked(self) -> dict[str, Any]:
        default = {
            "schema_version": SCHEMA_VERSION,
            "active_release_id": None,
            "run_mode": RUN_MODE_RESEARCH,
            "updated_at": None,
            "note": "",
        }
        if not self.path.exists():
            return default
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PitContextError("PIT 系统设置无法读取，请检查 data/pit_settings.json。") from exc
        if not isinstance(payload, dict):
            raise PitContextError("PIT 系统设置格式无效。")
        return {**default, **payload}

    def write_unlocked(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent)
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)


class PitSettingsRepository:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.store = PitSettingsStore(data_dir / PIT_SETTINGS_STORE)
        self.releases = DataReleaseRepository(data_dir / "data_releases.json")

    # ------------------------------------------------------------------ read

    def raw(self) -> dict[str, Any]:
        with self.store.locked():
            return self.store.read_unlocked()

    def describe(self) -> dict[str, Any]:
        """Everything the UI and the request layer need in one payload."""

        stored = self.raw()
        release_id = stored.get("active_release_id") or None
        release: Optional[dict[str, Any]] = None
        release_error: Optional[str] = None
        if release_id:
            try:
                release = self.releases.get(release_id)
            except DataReleaseError as exc:
                # A deleted or hand-edited release must degrade to no-PIT loudly,
                # never silently keep a stale as_of.
                release_error = str(exc)

        available = self.releases.list_releases()
        no_pit = release is None
        run_mode = str(stored.get("run_mode") or RUN_MODE_RESEARCH).upper()
        if run_mode not in RUN_MODES or no_pit:
            run_mode = RUN_MODE_RESEARCH
        as_of = _release_as_of(release) if release else None

        return {
            "settings": {
                "active_release_id": release_id,
                "run_mode": str(stored.get("run_mode") or RUN_MODE_RESEARCH).upper(),
                "updated_at": stored.get("updated_at"),
                "note": stored.get("note") or "",
            },
            "effective": {
                "as_of": as_of,
                "run_mode": run_mode,
                "run_mode_label": RUN_MODES[run_mode],
                "data_release_id": release["id"] if release else None,
                "no_pit": no_pit,
                "label": _effective_label(release, run_mode, as_of),
            },
            "release": _release_summary(release),
            "release_error": release_error,
            "available_releases": [_release_summary(item) for item in available],
            "can_apply": bool(available),
        }

    def effective_context(self) -> ResearchContext:
        effective = self.describe()["effective"]
        return ResearchContext(
            as_of=effective["as_of"],
            run_mode=effective["run_mode"],
            data_release_id=effective["data_release_id"],
        )

    # ----------------------------------------------------------------- write

    def update(
        self,
        active_release_id: Optional[str],
        run_mode: Optional[str],
        note: str = "",
    ) -> dict[str, Any]:
        wanted_release = str(active_release_id or "").strip() or None
        wanted_mode = str(run_mode or RUN_MODE_RESEARCH).strip().upper() or RUN_MODE_RESEARCH
        if wanted_mode not in RUN_MODES:
            raise PitContextError(f"不支持的运行模式：{run_mode}")
        if wanted_release is None and wanted_mode == RUN_MODE_STRICT:
            # Strict mode needs a cut-off to enforce; without a release there is
            # none, so it would quietly behave exactly like research mode.
            raise PitContextError("未应用数据版本时无法启用严格 PIT，请先封版并选择一个版本。")
        if wanted_release is not None:
            release = self.releases.get(wanted_release)  # raises if unknown
            if _release_as_of(release) is None:
                raise PitContextError(
                    f"数据版本 {release['name']} 没有可得截止日，无法作为 PIT 口径应用。"
                )

        with self.store.locked():
            payload = self.store.read_unlocked()
            payload.update(
                {
                    "schema_version": SCHEMA_VERSION,
                    "active_release_id": wanted_release,
                    "run_mode": wanted_mode,
                    "updated_at": _utc_now(),
                    "note": str(note or "").strip()[:500],
                }
            )
            self.store.write_unlocked(payload)
        return self.describe()


def _release_as_of(release: dict[str, Any]) -> Optional[str]:
    """A release's research day is the last date its A/B tables can answer for."""

    summary = release.get("summary") or {}
    value = summary.get("available_through")
    return str(value) if value else None


def release_as_of(data_dir: Path, release_id: str) -> Optional[str]:
    """The research day a sealed release can honestly answer for.

    Used when a tab asks to *view* a vintage other than the applied one: the
    release implies its own research day, so the caller never has to carry both.
    Raises `DataReleaseError` for an unknown id rather than degrading to no PIT.
    """

    repository = DataReleaseRepository(data_dir / "data_releases.json")
    return _release_as_of(repository.get(release_id))


def _release_summary(release: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not release:
        return None
    summary = release.get("summary") or {}
    return {
        "id": release.get("id"),
        "name": release.get("name"),
        "note": release.get("note") or "",
        "created_at": release.get("created_at"),
        "sequence": release.get("sequence"),
        "available_through": summary.get("available_through"),
        "grade_a": summary.get("grade_a"),
        "grade_b": summary.get("grade_b"),
        "grade_c": summary.get("grade_c"),
        "table_count": len(release.get("tables") or []),
        "total_rows": summary.get("total_rows"),
        "release_fingerprint": release.get("release_fingerprint"),
    }


def _effective_label(
    release: Optional[dict[str, Any]], run_mode: str, as_of: Optional[str]
) -> str:
    """One short string every result footnote can print verbatim."""

    if release is None:
        return "无 PIT 口径 · 使用全部磁盘数据"
    mode = "严格 PIT" if run_mode == RUN_MODE_STRICT else "研究模式"
    return f"{release.get('name')} · 研究日 {as_of} · {mode}"


__all__ = ["PIT_SETTINGS_STORE", "PitSettingsRepository", "PitSettingsStore", "release_as_of"]
