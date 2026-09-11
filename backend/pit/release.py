"""A named, reusable PIT口径: which day you stand on, over which vintage.

A release is the *whole* setting, not half of it. It carries three things:

* `as_of` — the day the platform stands on while this version is applied.
* the table fingerprints — which vintage of the files answered.
* `run_mode` — research or strict.

The vintage half matters because an `as_of` alone does not make research
reproducible: re-run the same day next month and the provider may have restated
the rows underneath you. The research day is stored *here* rather than beside
it, because "which day" and "which copy" were never two decisions a user wanted
to make separately — they wanted one saved口径 they could name and re-apply.
`as_of` may not run past what the vintage can answer for; unset means the last
day it can.

**What can be edited.** The口径 half — name, note, `as_of`, `run_mode` — is the
user's own choice and is editable; getting the research day wrong and having to
re-seal 1.5GB of fingerprints to fix a date is not integrity, it is friction.
The evidence half — the table list and their fingerprints — is not editable,
because a version whose vintage can be rewritten proves nothing. An edit stamps
`updated_at` so a changed口径 is visible rather than silent.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

try:  # pragma: no cover - platform specific
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

from .audit import audit_all
from .catalog import RUN_MODE_RESEARCH, RUN_MODE_STRICT, RUN_MODES
from .context import parse_as_of

try:
    from backend.market_data import resolve_market_data_file
except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
    from market_data import resolve_market_data_file


RELEASE_STORE = "data_releases.json"
SCHEMA_VERSION = 1


class DataReleaseError(ValueError):
    """Raised when a release cannot be created or found."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class DataReleaseStore:
    """Same atomic-write discipline as the product-pool store: temp file,
    fsync, rename, plus an advisory lock so two封版 cannot interleave."""

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
        if not self.path.exists():
            return {"schema_version": SCHEMA_VERSION, "releases": []}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DataReleaseError("数据版本存储无法读取，请检查 data/data_releases.json。") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("releases"), list):
            raise DataReleaseError("数据版本存储格式无效。")
        return payload

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


def _table_fingerprint(item: dict[str, Any], size: Optional[int]) -> str:
    """A metadata digest, deliberately not a row-level content hash.

    Hashing 1.5M rows on every封版 buys almost nothing: any parquet rewrite moves
    the byte count, the row count or the date range. The field is named
    `fingerprint` rather than `content_hash` so nobody reads more into it than
    it can carry.
    """

    material = json.dumps(
        {
            "dataset_id": item["dataset_id"],
            "file": item["file"],
            "rows": item["rows"],
            "bytes": size,
            "event_range": item["event_range"],
            "availability_range": item["availability_range"],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


class DataReleaseRepository:
    def __init__(self, path: Path) -> None:
        self.store = DataReleaseStore(path)

    @staticmethod
    def _order_key(item: dict[str, Any]) -> tuple[int, str]:
        """Newest-first ordering that does not depend on the clock.

        Two releases sealed inside the same wall-clock tick used to sort
        arbitrarily, which then chained `parent_release_id` to the wrong parent.
        The sequence number is monotonic by construction, so it survives both
        coarse timestamps and an NTP step backwards.
        """

        return (int(item.get("sequence") or 0), str(item.get("created_at") or ""))

    def list_releases(self) -> list[dict[str, Any]]:
        with self.store.locked():
            payload = self.store.read_unlocked()
        releases = list(payload["releases"])
        releases.sort(key=self._order_key, reverse=True)
        return releases

    def get(self, release_id: str) -> dict[str, Any]:
        wanted = str(release_id or "").strip()
        for release in self.list_releases():
            if release.get("id") == wanted:
                return release
        raise DataReleaseError(f"未找到数据版本 {wanted}。")

    def latest(self) -> Optional[dict[str, Any]]:
        releases = self.list_releases()
        return releases[0] if releases else None

    def update(
        self,
        release_id: str,
        *,
        name: str,
        note: str = "",
        as_of: Any = None,
        run_mode: Optional[str] = None,
    ) -> dict[str, Any]:
        """Rewrite a version's口径. The vintage it pins is left alone.

        `as_of` is checked against the ceiling recorded *in this release*, not
        against today's files: the version answers for the vintage it sealed, and
        that vintage did not grow because the disk did.
        """

        label = str(name or "").strip()
        if not label:
            raise DataReleaseError("数据版本名称不能为空。")
        if len(label) > 120:
            raise DataReleaseError("数据版本名称不能超过 120 个字符。")
        parsed = parse_as_of(as_of)
        wanted_as_of = parsed.strftime("%Y-%m-%d") if parsed is not None else None
        wanted_mode = str(run_mode or RUN_MODE_RESEARCH).strip().upper()
        if wanted_mode not in RUN_MODES:
            raise DataReleaseError(f"不支持的运行模式：{run_mode}")
        if wanted_mode == RUN_MODE_STRICT and wanted_as_of is None:
            raise DataReleaseError("严格 PIT 需要一个研究日：请填写「站在哪一天」。")

        wanted = str(release_id or "").strip()
        with self.store.locked():
            payload = self.store.read_unlocked()
            for release in payload["releases"]:
                if release.get("id") != wanted:
                    continue
                ceiling = (release.get("summary") or {}).get("available_through")
                if wanted_as_of is not None and ceiling and wanted_as_of > str(ceiling):
                    raise DataReleaseError(
                        f"研究日 {wanted_as_of} 晚于这个版本的可得截止日 {ceiling}；请前移研究日，或新建一个基于更新数据的版本。"
                    )
                release.update(
                    {
                        "name": label,
                        "note": str(note or "").strip()[:500],
                        "as_of": wanted_as_of,
                        "run_mode": wanted_mode,
                        "updated_at": _utc_now(),
                    }
                )
                self.store.write_unlocked(payload)
                return dict(release)
        raise DataReleaseError(f"未找到数据版本 {wanted}。")

    def delete(self, release_id: str) -> str:
        """Drop a version. Callers guard against deleting one that is in use."""

        wanted = str(release_id or "").strip()
        with self.store.locked():
            payload = self.store.read_unlocked()
            remaining = [item for item in payload["releases"] if item.get("id") != wanted]
            if len(remaining) == len(payload["releases"]):
                raise DataReleaseError(f"未找到数据版本 {wanted}。")
            payload["releases"] = remaining
            self.store.write_unlocked(payload)
        return wanted

    def create(
        self,
        data_dir: Path,
        name: str,
        note: str = "",
        as_of: Any = None,
        run_mode: Optional[str] = None,
    ) -> dict[str, Any]:
        label = str(name or "").strip()
        if not label:
            raise DataReleaseError("数据版本名称不能为空。")
        if len(label) > 120:
            raise DataReleaseError("数据版本名称不能超过 120 个字符。")
        parsed = parse_as_of(as_of)
        wanted_as_of = parsed.strftime("%Y-%m-%d") if parsed is not None else None
        wanted_mode = str(run_mode or RUN_MODE_RESEARCH).strip().upper()
        if wanted_mode not in RUN_MODES:
            raise DataReleaseError(f"不支持的运行模式：{run_mode}")
        if wanted_mode == RUN_MODE_STRICT and wanted_as_of is None:
            raise DataReleaseError("严格 PIT 需要一个研究日：请填写「站在哪一天」。")

        audit = audit_all(data_dir)
        tables: list[dict[str, Any]] = []
        for item in audit["datasets"]:
            if not item["present"]:
                continue
            # The active Tushare manifest can point the real files at a snapshot
            # subdirectory; sizing `data_dir / file` there silently yields None
            # and drops the byte count out of the fingerprint.
            try:
                path = resolve_market_data_file(item["file"], data_dir)
            except Exception:  # noqa: BLE001 - a broken manifest must not block封版
                path = data_dir / item["file"]
            size = path.stat().st_size if path.exists() else None
            tables.append(
                {
                    "dataset_id": item["dataset_id"],
                    "label": item["label"],
                    "file": item["file"],
                    "rows": item["rows"],
                    "bytes": size,
                    "grade": item["grade"],
                    "event_range": item["event_range"],
                    "availability_range": item["availability_range"],
                    "available_through": item.get("available_through"),
                    "fingerprint": _table_fingerprint(item, size),
                }
            )
        if not tables:
            raise DataReleaseError("没有任何可封版的数据集，请先完成数据下载。")
        ceiling = audit["summary"].get("available_through")
        if wanted_as_of is not None and ceiling and wanted_as_of > str(ceiling):
            # One-way, and the only constraint between the two halves: a vintage
            # cannot answer for days it does not contain.
            raise DataReleaseError(
                f"研究日 {wanted_as_of} 晚于这批数据的可得截止日 {ceiling}；请前移研究日。"
            )

        with self.store.locked():
            payload = self.store.read_unlocked()
            existing = sorted(payload["releases"], key=self._order_key, reverse=True)
            parent = existing[0]["id"] if existing else None
            next_sequence = (int(existing[0].get("sequence") or 0) + 1) if existing else 1
            release = {
                "id": f"release-{uuid.uuid4().hex[:12]}",
                "sequence": next_sequence,
                "name": label,
                "note": str(note or "").strip()[:500],
                "as_of": wanted_as_of,
                "run_mode": wanted_mode,
                "created_at": _utc_now(),
                "updated_at": None,
                "parent_release_id": parent,
                # The tables below, not the口径 above: `update` rewrites the
                # name/day/mode and never touches a fingerprint.
                "immutable": True,
                "tables": tables,
                "summary": audit["summary"],
                "release_fingerprint": hashlib.sha256(
                    "|".join(sorted(table["fingerprint"] for table in tables)).encode("utf-8")
                ).hexdigest(),
            }
            payload["releases"].append(release)
            payload["schema_version"] = SCHEMA_VERSION
            self.store.write_unlocked(payload)
        return release


__all__ = ["DataReleaseError", "DataReleaseRepository", "DataReleaseStore", "RELEASE_STORE"]
