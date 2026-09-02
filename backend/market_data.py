"""Resolve and atomically activate versioned Tushare data snapshots."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_DATA_DIR = PROJECT_ROOT / "data"
ACTIVE_MANIFEST_NAME = "tushare_active.json"
CORE_SNAPSHOT_FILES = (
    "etf_info_df.parquet",
    "etf_daily_df.parquet",
    "etf_daily_candle_df.parquet",
    "fund_info_df.parquet",
    "fund_nav_df.parquet",
    "trade_day_df.parquet",
    "instrument_metrics_snapshot.parquet",
)


class MarketDataManifestError(RuntimeError):
    """Raised when an active Tushare snapshot manifest is unsafe or invalid."""


def _normalise_base_dir(base_dir: Path | None = None) -> Path:
    return (base_dir or LEGACY_DATA_DIR).expanduser().resolve()


def _resolve_snapshot_path(base_dir: Path, raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (base_dir / candidate).resolve()
    if resolved != base_dir and base_dir not in resolved.parents:
        raise MarketDataManifestError("Tushare 快照目录必须位于项目 data 目录内。")
    if not resolved.is_dir():
        raise MarketDataManifestError(f"Tushare 快照目录不存在: {resolved}")
    return resolved


def read_active_manifest(base_dir: Path | None = None) -> dict[str, object] | None:
    """Read and validate the active manifest without changing filesystem state."""

    base = _normalise_base_dir(base_dir)
    manifest_path = base / ACTIVE_MANIFEST_NAME
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MarketDataManifestError("Tushare 活跃数据 manifest 无法读取。") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1 or not isinstance(payload.get("snapshot_dir"), str):
        raise MarketDataManifestError("Tushare 活跃数据 manifest 格式无效。")
    _resolve_snapshot_path(base, str(payload["snapshot_dir"]))
    return payload


def resolve_tushare_data_dir(base_dir: Path | None = None, *, strict: bool = False) -> Path:
    """Return the active snapshot directory, falling back to legacy ``data/``.

    ``TUSHARE_DATA_DIR`` is an explicit operator override.  An invalid manifest
    falls back only for compatibility unless ``strict`` is requested; snapshot
    activation itself always validates strictly.
    """

    base = _normalise_base_dir(base_dir)
    configured = os.getenv("TUSHARE_DATA_DIR", "").strip()
    if configured:
        path = Path(configured).expanduser().resolve()
        if path.is_dir():
            return path
        if strict:
            raise MarketDataManifestError(f"TUSHARE_DATA_DIR 不存在: {path}")
        return base
    try:
        payload = read_active_manifest(base)
        if payload is None:
            return base
        return _resolve_snapshot_path(base, str(payload["snapshot_dir"]))
    except MarketDataManifestError:
        if strict:
            raise
        return base


def resolve_market_data_file(filename: str, base_dir: Path | None = None) -> Path:
    """Resolve one Tushare dataset with a legacy-file fallback."""

    base = _normalise_base_dir(base_dir)
    active_path = resolve_tushare_data_dir(base) / filename
    if active_path.exists():
        return active_path
    return base / filename


def validate_snapshot_directory(
    snapshot_dir: Path,
    *,
    required_files: Iterable[str] = CORE_SNAPSHOT_FILES,
) -> dict[str, int]:
    """Validate the minimum file contract before activation."""

    snapshot = snapshot_dir.expanduser().resolve()
    missing = [name for name in required_files if not (snapshot / name).is_file()]
    if missing:
        raise MarketDataManifestError(f"Tushare 快照缺少必要文件: {', '.join(missing)}")
    empty = [name for name in required_files if (snapshot / name).stat().st_size <= 0]
    if empty:
        raise MarketDataManifestError(f"Tushare 快照包含空文件: {', '.join(empty)}")
    return {name: (snapshot / name).stat().st_size for name in required_files}


def activate_tushare_snapshot(
    snapshot_dir: Path,
    *,
    base_dir: Path | None = None,
    required_files: Iterable[str] = CORE_SNAPSHOT_FILES,
    validation_report: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Atomically switch the manifest after a complete local validation.

    This operation performs no network access.  The previous manifest remains
    intact until the replacement JSON has been fully written and fsynced.
    """

    base = _normalise_base_dir(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    snapshot = _resolve_snapshot_path(base, str(snapshot_dir))
    inventory = validate_snapshot_directory(snapshot, required_files=required_files)
    if validation_report is None:
        try:
            from backend.market_data_validation import validate_tushare_snapshot
        except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
            from market_data_validation import validate_tushare_snapshot

        baseline = resolve_tushare_data_dir(base)
        validation_report = validate_tushare_snapshot(
            snapshot,
            strict=base == LEGACY_DATA_DIR.resolve(),
            baseline_dir=baseline if baseline != snapshot else None,
        )
    if validation_report.get("status") != "passed":
        raise MarketDataManifestError("Tushare 快照尚未通过完整数据验收，拒绝激活。")
    try:
        report_dir = Path(str(validation_report["snapshot_dir"])).expanduser().resolve()
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise MarketDataManifestError("Tushare 快照验收报告格式无效。") from exc
    if report_dir != snapshot:
        raise MarketDataManifestError("Tushare 快照验收报告与待激活目录不一致。")
    reported_inventory = validation_report.get("inventory")
    if not isinstance(reported_inventory, Mapping):
        raise MarketDataManifestError("Tushare 快照验收报告缺少文件指纹清单。")
    for filename in required_files:
        record = reported_inventory.get(filename)
        path = snapshot / filename
        if not isinstance(record, Mapping) or (
            record.get("size") != path.stat().st_size
            or record.get("mtime_ns") != path.stat().st_mtime_ns
        ):
            raise MarketDataManifestError(
                f"Tushare 快照在验收后发生变化，拒绝激活: {filename}"
            )
    payload: dict[str, object] = {
        "schema_version": 1,
        "snapshot_dir": snapshot.relative_to(base).as_posix(),
        "activated_at": datetime.now(timezone.utc).isoformat(),
        "files": inventory,
        "validation": {
            "status": "passed",
            "datasets": validation_report.get("datasets", {}),
        },
    }

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{ACTIVE_MANIFEST_NAME}.", suffix=".tmp", dir=base
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, base / ACTIVE_MANIFEST_NAME)
        try:
            directory_fd = os.open(base, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        temporary_path.unlink(missing_ok=True)
    return payload
