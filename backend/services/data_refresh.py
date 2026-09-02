"""Process-safe coordinator for Tushare refreshes and analytics snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import pyarrow.parquet as parquet

try:
    from backend.market_data import resolve_tushare_data_dir
except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
    from market_data import resolve_tushare_data_dir

from .refresh_runtime import (
    InterProcessFileLock,
    atomic_write_json,
    is_file_lock_held,
    read_json_object,
)
from .index_data import (
    DEFAULT_INDEX_SCOPES,
    INDEX_DATASET_SPECS,
    INDEX_SCOPES,
    index_required_files,
    normalise_index_scopes,
    validate_index_snapshot,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
REFRESH_SCRIPT = PROJECT_ROOT / "T01_get_data.py"
TRUE_VALUES = {"1", "true", "yes", "on"}
REFRESH_MODULES = {"base", "etf", "fund", "index"}
REFRESH_MODES = {"incremental", "full"}
REFRESH_LOCK_FILENAME = ".tushare_refresh.lock"
REFRESH_STATE_FILENAME = ".tushare_refresh_status.json"
TUSHARE_TOKEN_FILENAME = ".tushare_token"
TOKEN_CONFIG_LOCK_FILENAME = ".tushare_token.lock"
MAX_LOG_TAIL_CHARS = 8000
LOG_STATE_FLUSH_INTERVAL_SECONDS = 1.0
LEGACY_CHECKPOINT_ACTIVE_SECONDS = 30 * 60
DATASET_SPECS = {
    "etf_info": ("etf_info_df.parquet", None),
    "etf_nav": ("etf_daily_df.parquet", "date"),
    "etf_candle": ("etf_daily_candle_df.parquet", "date"),
    "fund_info": ("fund_info_df.parquet", None),
    "fund_nav": ("fund_nav_df.parquet", "date"),
    "fund_company": ("fund_company_df.parquet", None),
    "calendar": ("trade_day_df.parquet", "cal_date"),
    "instrument_metrics": ("instrument_metrics_snapshot.parquet", "latest_date"),
    **{
        key: (filename, date_column)
        for key, (filename, date_column, _scope) in INDEX_DATASET_SPECS.items()
    },
}
ANALYTICS_SOURCE_FILENAMES = (
    "etf_daily_df.parquet",
    "fund_nav_df.parquet",
    "etf_daily_candle_df.parquet",
    "trade_day_df.parquet",
)

STAGING_COPY_EXCLUDES = {
    REFRESH_LOCK_FILENAME,
    REFRESH_STATE_FILENAME,
    "tushare_active.json",
}

MODULE_FLAGS = {
    "base": ["--calendar", "--stock-basic", "--fund-company"],
    "etf": ["--etf-info", "--nav", "--candle"],
    "fund": ["--fund-info", "--fund-nav"],
    "index": [],
}
INDEX_SCOPE_FLAGS = {scope: f"--index-{scope}" for scope in INDEX_SCOPES}

TUSHARE_TOKEN_VALUE = re.compile(r"^[A-Za-z0-9._-]{16,256}$")
LOG_LEVEL_ONLY = re.compile(r"^\[(?:INFO|OK|WARN|ERROR)\]$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _latest_complete_status_line(value: str) -> str | None:
    """Return one complete progress/success line from a bounded log tail.

    Subprocess stdout is read in fixed-size chunks, so one chunk may start and
    end in the middle of a line.  A user-facing status must never expose that
    arbitrary chunk boundary as if it were the current progress.
    """

    lines = value.splitlines()
    if value and not value.endswith(("\n", "\r")) and lines:
        lines.pop()
    for line in reversed(lines):
        stripped = line.strip()
        if not stripped or LOG_LEVEL_ONLY.fullmatch(stripped):
            continue
        if "进度" in stripped or stripped.startswith("[OK]"):
            return stripped[-500:]
    return None


def data_refresh_enabled() -> bool:
    configured = os.getenv("DATA_REFRESH_ENABLED")
    if configured is not None:
        return configured.strip().lower() in TRUE_VALUES
    return os.getenv("APP_ENV", "development").strip().lower() != "production"


def full_refresh_enabled() -> bool:
    return os.getenv("DATA_FULL_REFRESH_ENABLED", "false").strip().lower() in TRUE_VALUES


def tushare_token_configuration_enabled() -> bool:
    """Allow local token management whenever browser-triggered refresh is enabled."""

    configured = os.getenv("TUSHARE_TOKEN_CONFIG_ENABLED")
    if configured is not None:
        return data_refresh_enabled() and configured.strip().lower() in TRUE_VALUES
    return data_refresh_enabled()


def _tushare_token_path(data_dir: Path | None = None) -> Path:
    return Path(data_dir) / TUSHARE_TOKEN_FILENAME if data_dir is not None else DATA_DIR / TUSHARE_TOKEN_FILENAME


def _read_local_tushare_token(data_dir: Path | None = None) -> str:
    token_path = _tushare_token_path(data_dir)
    if token_path.is_symlink() or not token_path.is_file():
        return ""
    try:
        return token_path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def tushare_token_source(data_dir: Path | None = None) -> str:
    if _read_local_tushare_token(data_dir):
        return "frontend_local"
    return "none"


def tushare_token_configured(data_dir: Path | None = None) -> bool:
    return tushare_token_source(data_dir) != "none"


def tushare_token_status(data_dir: Path | None = None) -> dict[str, Any]:
    source = tushare_token_source(data_dir)
    configuration_enabled = tushare_token_configuration_enabled()
    return {
        "token_configured": source != "none",
        "token_source": source,
        "token_configuration_enabled": configuration_enabled,
        "token_editable": configuration_enabled,
    }


def _normalise_tushare_token(value: str) -> str:
    token = value.strip()
    if not TUSHARE_TOKEN_VALUE.fullmatch(token):
        raise ValueError("Tushare TOKEN 格式无效。")
    return token


def _atomic_write_token(path: Path, token: str) -> None:
    if path.is_symlink():
        raise PermissionError("拒绝写入符号链接形式的 Token 凭据文件。")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".tushare_token.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(token)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        path.chmod(0o600)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_local_tushare_token(token: str | None, data_dir: Path | None = None) -> dict[str, Any]:
    root = Path(data_dir) if data_dir is not None else DATA_DIR
    root.mkdir(parents=True, exist_ok=True)
    refresh_lock = InterProcessFileLock(root / REFRESH_LOCK_FILENAME)
    owner = f"token_config=1\npid={os.getpid()}\nupdated_at={utc_now()}\n"
    if not refresh_lock.acquire(owner=owner):
        raise RuntimeError("数据刷新正在运行，暂不能修改 Tushare Token。")
    lock = InterProcessFileLock(root / TOKEN_CONFIG_LOCK_FILENAME)
    try:
        if not lock.acquire(owner=owner):
            raise RuntimeError("另一个 Token 配置操作正在进行，请稍后重试。")
        try:
            token_path = _tushare_token_path(root)
            if token_path.is_symlink():
                raise PermissionError("拒绝修改符号链接形式的 Token 凭据文件。")
            if token is None:
                token_path.unlink(missing_ok=True)
            else:
                _atomic_write_token(token_path, _normalise_tushare_token(token))
            return tushare_token_status(root)
        finally:
            lock.release()
    finally:
        refresh_lock.release()


def save_local_tushare_token(token: str, data_dir: Path | None = None) -> dict[str, Any]:
    """Persist a frontend-supplied token locally without returning its value."""

    return _write_local_tushare_token(_normalise_tushare_token(token), data_dir)


def remove_local_tushare_token(data_dir: Path | None = None) -> dict[str, Any]:
    """Remove the frontend-managed local token."""

    return _write_local_tushare_token(None, data_dir)


def _configured_token_values(data_dir: Path | None = None) -> list[str]:
    values: list[str] = []
    file_token = _read_local_tushare_token(data_dir)
    if file_token:
        values.append(file_token)
    return values


def normalise_refresh_request(modules: list[str], mode: str) -> tuple[list[str], str]:
    clean_modules = list(dict.fromkeys(str(item).strip().lower() for item in modules if str(item).strip()))
    unknown = set(clean_modules) - REFRESH_MODULES
    if unknown:
        raise ValueError(f"不支持的数据模块: {', '.join(sorted(unknown))}")
    if not clean_modules:
        raise ValueError("至少选择一个数据模块。")
    clean_mode = str(mode).strip().lower()
    if clean_mode not in REFRESH_MODES:
        raise ValueError(f"不支持的更新模式: {mode}")
    if clean_mode == "full" and not full_refresh_enabled():
        raise PermissionError("全量更新未开启；请设置 DATA_FULL_REFRESH_ENABLED=true。")
    return clean_modules, clean_mode


def refresh_request_fingerprint(
    modules: list[str], mode: str, index_scopes: list[str] | None = None
) -> str:
    modules, mode = normalise_refresh_request(modules, mode)
    scopes = normalise_index_scopes(index_scopes) if "index" in modules else []
    payload = {
        "modules": modules,
        "index_scopes": scopes,
        "mode": mode,
        "start_date": os.getenv("TUSHARE_FULL_START_DATE", "20100101"),
        "end_date": os.getenv("TUSHARE_FULL_END_DATE", "").strip()
        or datetime.now(timezone.utc).strftime("%Y%m%d"),
        "history_chunk_days": os.getenv("TUSHARE_HISTORY_CHUNK_DAYS", "3650"),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def build_refresh_command(
    modules: list[str],
    mode: str,
    *,
    index_scopes: list[str] | None = None,
    data_dir: Path | None = None,
    resume: bool = False,
) -> list[str]:
    modules, mode = normalise_refresh_request(modules, mode)
    scopes = normalise_index_scopes(index_scopes) if "index" in modules else []
    output_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    max_calls = os.getenv("TUSHARE_MAX_CALLS_PER_MINUTE", "450")
    min_interval = os.getenv("TUSHARE_MIN_CALL_INTERVAL_SECONDS", "0.13")
    max_days = os.getenv("TUSHARE_MAX_LATEST_DAYS", "120")
    command = [
        sys.executable,
        str(REFRESH_SCRIPT),
        "--output-dir",
        str(output_dir),
        "--max-calls-per-minute",
        max_calls,
        "--min-call-interval-sec",
        min_interval,
        "--max-retries",
        os.getenv("TUSHARE_MAX_RETRIES", "5"),
        "--backoff-sec",
        os.getenv("TUSHARE_RETRY_BACKOFF_SECONDS", "2.0"),
        "--wait-on-rate-limit-sec",
        os.getenv("TUSHARE_RATE_LIMIT_WAIT_SECONDS", "15.0"),
        "--retry-jitter-sec",
        os.getenv("TUSHARE_RETRY_JITTER_SECONDS", "0.5"),
        "--max-workers",
        os.getenv("TUSHARE_MAX_WORKERS", "16"),
        "--max-fund-basic-pages",
        os.getenv("TUSHARE_MAX_FUND_BASIC_PAGES", "20"),
        "--max-fund-nav-pages",
        os.getenv("TUSHARE_MAX_FUND_NAV_PAGES", "20"),
        "--fund-nav-page-size",
        os.getenv("TUSHARE_FUND_NAV_PAGE_SIZE", "10000"),
        "--incremental-batch-days",
        os.getenv("TUSHARE_INCREMENTAL_BATCH_DAYS", "20"),
    ]
    if mode == "incremental":
        command.extend(["--latest", "--max-latest-days", max_days])
    else:
        full_end_date = os.getenv("TUSHARE_FULL_END_DATE", "").strip()
        command.extend(
            [
                "--start-date",
                os.getenv("TUSHARE_FULL_START_DATE", "20100101"),
                "--history-chunk-days",
                os.getenv("TUSHARE_HISTORY_CHUNK_DAYS", "3650"),
            ]
        )
        if full_end_date:
            command.extend(["--end-date", full_end_date])
        if resume:
            command.append("--resume")
    flags: list[str] = []
    if mode == "incremental" and any(module in {"etf", "fund", "index"} for module in modules):
        flags.append("--calendar")
    for module in modules:
        flags.extend(MODULE_FLAGS[module])
    flags.extend(INDEX_SCOPE_FLAGS[scope] for scope in scopes)
    command.extend(dict.fromkeys(flags))
    return command


def _normalise_stat_date(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if hasattr(value, "isoformat"):
        return value.isoformat()[:10]
    text = str(value).replace("-", "")
    if len(text) >= 8 and text[:8].isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    return str(value)


@lru_cache(maxsize=16)
def _parquet_summary_cached(path_text: str, mtime_ns: int, size: int, date_column: str | None) -> dict[str, Any]:
    del mtime_ns, size
    file = parquet.ParquetFile(path_text)
    metadata = file.metadata
    latest: Any = None
    earliest: Any = None
    if date_column:
        names = metadata.schema.names
        if date_column in names:
            column_index = names.index(date_column)
            for group_index in range(metadata.num_row_groups):
                statistics = metadata.row_group(group_index).column(column_index).statistics
                if statistics is not None and statistics.has_min_max:
                    candidate = statistics.max
                    if latest is None or candidate > latest:
                        latest = candidate
                    minimum = statistics.min
                    if earliest is None or minimum < earliest:
                        earliest = minimum
    return {
        "rows": metadata.num_rows,
        "earliest_date": _normalise_stat_date(earliest),
        "latest_date": _normalise_stat_date(latest),
    }


def dataset_summaries(data_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    directory = Path(data_dir) if data_dir is not None else DATA_DIR
    summaries: dict[str, dict[str, Any]] = {}
    for dataset, (filename, date_column) in DATASET_SPECS.items():
        path = directory / filename
        if not path.exists():
            summaries[dataset] = {
                "file": filename,
                "exists": False,
                "status": "missing",
                "rows": 0,
                "earliest_date": None,
                "latest_date": None,
            }
            continue
        stat = path.stat()
        try:
            summary = _parquet_summary_cached(str(path), stat.st_mtime_ns, stat.st_size, date_column)
            summaries[dataset] = {
                "file": filename,
                "exists": True,
                "status": "ready",
                "updated_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                **summary,
            }
        except Exception as exc:  # noqa: BLE001
            summaries[dataset] = {
                "file": filename,
                "exists": True,
                "status": "error",
                "rows": None,
                "earliest_date": None,
                "latest_date": None,
                "error": f"无法读取 parquet 元数据: {exc}",
            }
    return summaries


def _analytics_source_state(data_dir: Path) -> tuple[tuple[str, int, int], ...]:
    state = []
    for filename in ANALYTICS_SOURCE_FILENAMES:
        path = data_dir / filename
        if not path.exists():
            continue
        stat = path.stat()
        state.append((filename, stat.st_mtime_ns, stat.st_size))
    return tuple(state)


def _reusable_analytics_snapshot(
    data_dir: Path,
    before: tuple[tuple[str, int, int], ...],
    after: tuple[tuple[str, int, int], ...],
) -> dict[str, Any] | None:
    if not before or before != after:
        return None
    snapshot_path = data_dir / "instrument_metrics_snapshot.parquet"
    if not snapshot_path.exists():
        return None
    snapshot_stat = snapshot_path.stat()
    if snapshot_stat.st_mtime_ns < max(item[1] for item in after):
        return None
    summary = dataset_summaries(data_dir).get("instrument_metrics", {})
    if summary.get("status") != "ready":
        return None
    return {
        "status": "succeeded",
        "reused": True,
        "path": str(snapshot_path),
        "rows": summary.get("rows"),
        "as_of": summary.get("latest_date"),
        "updated_at": summary.get("updated_at"),
        "source_files": [item[0] for item in after],
    }


def prepare_full_refresh_staging(
    *,
    data_root: Path,
    source_dir: Path,
    job_id: str,
) -> Path:
    """Create an isolated full-refresh version seeded from the active snapshot."""

    root = Path(data_root).expanduser().resolve()
    source = Path(source_dir).expanduser().resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = root / f"tushare_snapshot_{stamp}_{job_id[:8]}"
    target.mkdir(parents=True, exist_ok=False)
    for path in source.iterdir():
        if not path.is_file() or path.name in STAGING_COPY_EXCLUDES or path.name.startswith("."):
            continue
        shutil.copy2(path, target / path.name)
    return target


def validate_and_activate_full_refresh(
    staging_dir: Path,
    *,
    data_root: Path,
    index_scopes: list[str] | None = None,
) -> dict[str, Any]:
    """Run the read-only promotion gate and atomically move the manifest."""

    try:
        from backend.market_data import activate_tushare_snapshot
        from backend.market_data_validation import validate_tushare_snapshot
    except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
        from market_data import activate_tushare_snapshot
        from market_data_validation import validate_tushare_snapshot

    baseline_dir = resolve_tushare_data_dir(data_root)
    report = validate_tushare_snapshot(
        staging_dir,
        strict=True,
        baseline_dir=baseline_dir if baseline_dir.resolve() != staging_dir.resolve() else None,
    )
    required_files: tuple[str, ...] | None = None
    if index_scopes:
        index_report = validate_index_snapshot(staging_dir, index_scopes)
        report.setdefault("datasets", {}).update(index_report["datasets"])
        required_files = tuple(
            dict.fromkeys((*report["inventory"].keys(), *index_required_files(index_scopes)))
        )
    manifest = activate_tushare_snapshot(
        staging_dir,
        base_dir=data_root,
        **({"required_files": required_files} if required_files else {}),
        validation_report=report,
    )
    return {"validation": report, "manifest": manifest}


def _sanitise_output(
    value: str,
    *,
    data_dir: Path | None = None,
    extra_tokens: tuple[str, ...] = (),
) -> str:
    output = value
    for token in [*_configured_token_values(data_dir), *extra_tokens]:
        if not token:
            continue
        output = output.replace(token, "[REDACTED]")
    return re.sub(r"(?i)(TUSHARE_TOKEN\s*=\s*)\S+", r"\1[REDACTED]", output)[-MAX_LOG_TAIL_CHARS:]


def rebuild_local_analytics_snapshot(data_dir: Path | None = None) -> dict[str, Any]:
    """Invoke the local-only analytics builder without touching Tushare."""

    from . import instrument_analytics

    builder = getattr(instrument_analytics, "rebuild_analytics_snapshot", None)
    if not callable(builder):
        raise RuntimeError("instrument_analytics 未提供可调用的本地快照 builder。")
    directory = Path(data_dir) if data_dir is not None else DATA_DIR
    result = builder(data_dir=directory)
    if result is None:
        return {"status": "succeeded"}
    if not isinstance(result, dict):
        raise RuntimeError("分析快照 builder 必须返回字典结果。")
    return result


def _empty_job() -> dict[str, Any]:
    return {
        "job_id": None,
        "status": "idle",
        "started_at": None,
        "finished_at": None,
        "modules": [],
        "index_scopes": [],
        "mode": None,
        "message": "尚未启动更新",
        "log_tail": "",
        "owner_pid": None,
        "worker_pid": None,
        "heartbeat_at": None,
        "analytics_snapshot": None,
        "warnings": [],
        "request_fingerprint": None,
        "staging_data_dir": None,
        "fetch_complete": False,
        "resumed": False,
    }


class DataRefreshManager:
    def __init__(self, *, data_dir: Path | None = None) -> None:
        # State and the cross-process lock deliberately stay in the stable data
        # root.  Dataset reads/writes resolve the active version on each
        # operation so an atomic manifest switch is visible without restarting
        # the API process.
        self.data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
        self._resolve_active_version = data_dir is None
        self.state_path = self.data_dir / REFRESH_STATE_FILENAME
        self.process_lock_path = self.data_dir / REFRESH_LOCK_FILENAME
        self._lock = threading.Lock()
        self._job: dict[str, Any] = _empty_job()
        self._active_process_lock: InterProcessFileLock | None = None
        self._active_job_id: str | None = None
        self._active_token: str | None = None
        self._last_log_state_flush = 0.0
        persisted = self._read_persisted_job()
        if persisted is not None:
            self._job.update(persisted)

    def _current_data_dir(self) -> Path:
        if not self._resolve_active_version:
            return self.data_dir
        return resolve_tushare_data_dir(self.data_dir)

    def _sanitise(self, value: str) -> str:
        tokens = (self._active_token,) if self._active_token else ()
        return _sanitise_output(value, data_dir=self.data_dir, extra_tokens=tokens)

    def _safe_staging_dir(self, value: Any) -> Path | None:
        if not value:
            return None
        try:
            root = self.data_dir.expanduser().resolve()
            candidate = Path(str(value)).expanduser().resolve()
        except (OSError, RuntimeError, TypeError, ValueError):
            return None
        if root not in candidate.parents or not candidate.name.startswith("tushare_snapshot_"):
            return None
        return candidate if candidate.is_dir() else None

    def _resumable_candidate_locked(
        self,
        *,
        modules: list[str],
        index_scopes: list[str],
        mode: str,
        request_fingerprint: str,
    ) -> tuple[Path | None, bool]:
        previous = self._job
        if mode != "full" or previous.get("mode") != "full":
            return None, False
        if list(previous.get("modules") or []) != modules:
            return None, False
        if list(previous.get("index_scopes") or []) != index_scopes:
            return None, False
        previous_fingerprint = previous.get("request_fingerprint")
        if previous_fingerprint and previous_fingerprint != request_fingerprint:
            return None, False
        analytics = previous.get("analytics_snapshot")
        analytics_failed = isinstance(analytics, dict) and analytics.get("status") == "failed"
        if previous.get("status") != "failed" and not analytics_failed:
            return None, False
        candidate = self._safe_staging_dir(previous.get("staging_data_dir"))
        return candidate, bool(previous.get("fetch_complete")) if candidate is not None else False

    def _legacy_external_checkpoint(self) -> Path | None:
        """Detect a pre-lock validation run by its recently growing checkpoint.

        This compatibility bridge is intentionally narrow.  New CLI and Web
        runs share the global flock; it only protects validation jobs started
        with an older script before that lock existed.
        """

        now = time.time()
        for checkpoint in self.data_dir.glob(
            "tushare_full_validation_*/.fund_nav_df_parts_*"
        ):
            final_path = checkpoint.parent / "fund_nav_df.parquet"
            if final_path.exists() or not checkpoint.is_dir():
                continue
            try:
                age = now - checkpoint.stat().st_mtime
            except OSError:
                continue
            if 0 <= age <= LEGACY_CHECKPOINT_ACTIVE_SECONDS:
                return checkpoint
        return None

    def _read_persisted_job(self) -> dict[str, Any] | None:
        payload = read_json_object(self.state_path)
        if payload is None:
            return None
        job = payload.get("job")
        if not isinstance(job, dict):
            return None
        if job.get("status") not in {"idle", "running", "succeeded", "failed"}:
            return None
        return dict(job)

    def _persist_locked(self) -> None:
        atomic_write_json(
            self.state_path,
            {
                "schema_version": 1,
                "updated_at": utc_now(),
                "job": self._job,
            },
        )

    def _local_job_is_running_locked(self, job_id: Any) -> bool:
        return bool(
            job_id
            and job_id == self._active_job_id
            and self._active_process_lock is not None
            and self._active_process_lock.acquired
        )

    def _refresh_from_disk_locked(self) -> None:
        if self._local_job_is_running_locked(self._job.get("job_id")):
            return
        persisted = self._read_persisted_job()
        if persisted is not None:
            self._job = {**_empty_job(), **persisted}

    def _mark_interrupted_if_stale(self) -> None:
        with self._lock:
            self._refresh_from_disk_locked()
            job_id = self._job.get("job_id")
            should_probe = self._job.get("status") == "running" and not self._local_job_is_running_locked(job_id)
        if not should_probe or is_file_lock_held(self.process_lock_path):
            return
        with self._lock:
            self._refresh_from_disk_locked()
            job_id = self._job.get("job_id")
            if self._job.get("status") != "running" or self._local_job_is_running_locked(job_id):
                return
            if is_file_lock_held(self.process_lock_path):
                return
            self._job.update(
                {
                    "status": "failed",
                    "finished_at": utc_now(),
                    "message": "数据更新进程已中断；检查点仍保留，可稍后重新启动继续。",
                }
            )
            self._persist_locked()

    def snapshot(self) -> dict[str, Any]:
        self._mark_interrupted_if_stale()
        with self._lock:
            self._refresh_from_disk_locked()
            job = dict(self._job)
        if job.get("status") == "running":
            latest_status = _latest_complete_status_line(str(job.get("log_tail") or ""))
            if latest_status:
                job["message"] = latest_status
        legacy_checkpoint = self._legacy_external_checkpoint()
        if job.get("status") != "running" and legacy_checkpoint is not None:
            job = {
                **_empty_job(),
                "job_id": f"legacy-{legacy_checkpoint.parent.name}",
                "status": "running",
                "message": "检测到旧版外部全量任务仍在更新检查点；网页不会重复启动抓取。",
                "mode": "full",
                "modules": ["fund"],
                "legacy_checkpoint": str(legacy_checkpoint),
            }
        return {
            "source": "tushare",
            "enabled": data_refresh_enabled(),
            "full_refresh_enabled": full_refresh_enabled(),
            "available_modules": sorted(REFRESH_MODULES),
            "available_index_scopes": list(INDEX_SCOPES),
            "default_index_scopes": list(DEFAULT_INDEX_SCOPES),
            **tushare_token_status(self.data_dir),
            "job": job,
            "data_dir": str(self._current_data_dir()),
            "datasets": dataset_summaries(self._current_data_dir()),
        }

    def start(
        self,
        modules: list[str],
        mode: str,
        index_scopes: list[str] | None = None,
    ) -> dict[str, Any]:
        modules, mode = normalise_refresh_request(modules, mode)
        scopes = normalise_index_scopes(index_scopes) if "index" in modules else []
        request_fingerprint = refresh_request_fingerprint(modules, mode, scopes)
        self._mark_interrupted_if_stale()
        legacy_checkpoint = self._legacy_external_checkpoint()
        if legacy_checkpoint is not None:
            raise RuntimeError(
                "检测到旧版外部全量任务仍在更新检查点；为避免重复抓取，本次网页任务未启动。"
            )
        job_id = uuid.uuid4().hex
        process_lock = InterProcessFileLock(self.process_lock_path)
        owner = f"job_id={job_id}\npid={os.getpid()}\nstarted_at={utc_now()}\n"
        if not process_lock.acquire(owner=owner):
            with self._lock:
                self._refresh_from_disk_locked()
                message = self._job.get("message") or "已有数据刷新任务正在运行。"
            raise RuntimeError(f"数据更新任务正在运行，请勿重复启动。{message}")
        token = _read_local_tushare_token(self.data_dir)
        if not token:
            process_lock.release()
            raise PermissionError("尚未配置 Tushare Token，请先在主界面的数据管理中保存。")
        with self._lock:
            self._refresh_from_disk_locked()
            if self._local_job_is_running_locked(self._job.get("job_id")):
                process_lock.release()
                raise RuntimeError("数据更新任务正在运行，请勿重复启动。")
            resume_dir, fetch_complete = self._resumable_candidate_locked(
                modules=modules,
                index_scopes=scopes,
                mode=mode,
                request_fingerprint=request_fingerprint,
            )
            self._job = {
                **_empty_job(),
                "job_id": job_id,
                "status": "running",
                "started_at": utc_now(),
                "finished_at": None,
                "modules": modules,
                "index_scopes": scopes,
                "mode": mode,
                "message": (
                    "正在续跑已保留的全量候选，仅处理未完成项"
                    if resume_dir is not None
                    else f"正在从 Tushare 执行{'增量' if mode == 'incremental' else '全量'}更新"
                ),
                "log_tail": "",
                "owner_pid": os.getpid(),
                "heartbeat_at": utc_now(),
                "request_fingerprint": request_fingerprint,
                "staging_data_dir": str(resume_dir) if resume_dir is not None else None,
                "fetch_complete": fetch_complete,
                "resumed": resume_dir is not None,
            }
            self._active_process_lock = process_lock
            self._active_job_id = job_id
            self._active_token = token
            self._last_log_state_flush = 0.0
            try:
                self._persist_locked()
            except Exception:
                self._active_process_lock = None
                self._active_job_id = None
                self._active_token = None
                process_lock.release()
                raise
        thread = threading.Thread(
            target=self._run,
            args=(job_id, modules, mode, scopes),
            name=f"tushare-refresh-{job_id[:8]}",
            daemon=True,
        )
        try:
            thread.start()
        except Exception:
            self._finish_job(
                job_id,
                {
                    "status": "failed",
                    "message": "无法启动数据更新工作线程",
                    "log_tail": "",
                },
            )
            raise
        return self.snapshot()

    def _append_log(self, job_id: str, text: str, *, force_persist: bool = False) -> None:
        if not text:
            return
        now = time.monotonic()
        with self._lock:
            if self._job.get("job_id") != job_id:
                return
            combined = f"{self._job.get('log_tail') or ''}{text}"
            self._job["log_tail"] = self._sanitise(combined)
            self._job["heartbeat_at"] = utc_now()
            latest_status = _latest_complete_status_line(self._job["log_tail"])
            if latest_status:
                self._job["message"] = latest_status
            if force_persist or now - self._last_log_state_flush >= LOG_STATE_FLUSH_INTERVAL_SECONDS:
                try:
                    self._persist_locked()
                    self._last_log_state_flush = now
                except OSError:
                    # The in-memory status and bounded log remain usable. A final
                    # state write is attempted after the child exits.
                    pass

    def _stream_process_output(self, job_id: str, process: subprocess.Popen[str]) -> None:
        stream = process.stdout
        if stream is None:
            return
        try:
            # Iterate by line so a short progress record becomes visible
            # immediately instead of waiting for an arbitrary 1 KiB buffer to
            # fill. Split exceptionally long lines to retain bounded handling.
            for line in stream:
                for offset in range(0, len(line), 1024):
                    self._append_log(job_id, line[offset:offset + 1024])
        finally:
            stream.close()

    def _touch_heartbeat(self, job_id: str) -> None:
        with self._lock:
            if self._job.get("job_id") != job_id or self._job.get("status") != "running":
                return
            self._job["heartbeat_at"] = utc_now()
            try:
                self._persist_locked()
            except OSError:
                pass

    def _postprocess_heartbeat_loop(
        self,
        job_id: str,
        stop_event: threading.Event,
    ) -> None:
        while not stop_event.wait(5.0):
            self._touch_heartbeat(job_id)

    @staticmethod
    def _stop_process(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)

    def _finish_job(self, job_id: str, final: dict[str, Any]) -> None:
        process_lock: InterProcessFileLock | None = None
        clear_active_token = False
        try:
            with self._lock:
                if self._job.get("job_id") == job_id:
                    self._job.update(final)
                    self._job["finished_at"] = utc_now()
                    self._job["log_tail"] = self._sanitise(str(self._job.get("log_tail") or ""))
                if self._active_job_id == job_id:
                    process_lock = self._active_process_lock
                    self._active_process_lock = None
                    self._active_job_id = None
                    clear_active_token = True
                try:
                    self._persist_locked()
                except Exception as exc:  # noqa: BLE001
                    self._job.setdefault("warnings", []).append(
                        {
                            "code": "REFRESH_STATE_PERSIST_FAILED",
                            "message": self._sanitise(str(exc))[-500:],
                        }
                    )
                if clear_active_token:
                    self._active_token = None
        finally:
            if process_lock is not None:
                process_lock.release()

    def rebuild_analytics(self, *, candidate: bool = False) -> dict[str, Any]:
        """Rebuild analytics locally while excluding concurrent data writes."""

        process_lock = InterProcessFileLock(self.process_lock_path)
        owner = f"analytics_rebuild=1\npid={os.getpid()}\nstarted_at={utc_now()}\n"
        if not process_lock.acquire(owner=owner):
            raise RuntimeError("数据刷新正在运行，暂不能重建分析快照。")
        try:
            target = self._current_data_dir()
            if candidate:
                with self._lock:
                    self._refresh_from_disk_locked()
                    target = self._safe_staging_dir(self._job.get("staging_data_dir"))
                    fetch_complete = bool(self._job.get("fetch_complete"))
                    candidate_index_scopes = list(self._job.get("index_scopes") or [])
                if target is None:
                    raise RuntimeError("没有可重建的全量候选目录。")
                if not fetch_complete:
                    raise RuntimeError("候选数据尚未抓取完成，请先续跑缺失项。")
            result = rebuild_local_analytics_snapshot(target)
            response: dict[str, Any] = {
                "status": "succeeded",
                "rebuilt_at": utc_now(),
                "data_dir": str(target),
                **result,
            }
            if candidate:
                promotion = validate_and_activate_full_refresh(
                    target,
                    data_root=self.data_dir,
                    **({"index_scopes": candidate_index_scopes} if candidate_index_scopes else {}),
                )
                response["validation_status"] = promotion["validation"]["status"]
                response["activated_snapshot_dir"] = promotion["manifest"]["snapshot_dir"]
            with self._lock:
                self._refresh_from_disk_locked()
                analytics_snapshot = {
                    "status": "succeeded",
                    "rebuilt_at": response["rebuilt_at"],
                }
                for key in ("path", "rows", "by_kind", "as_of", "source_files", "updated_at"):
                    if key in result:
                        analytics_snapshot[key] = result[key]
                if candidate:
                    analytics_snapshot["validation_status"] = response["validation_status"]
                    analytics_snapshot["activated_snapshot_dir"] = response[
                        "activated_snapshot_dir"
                    ]
                    self._job["message"] = "已在本地重建、验收并原子接入候选快照"
                self._job["analytics_snapshot"] = analytics_snapshot
                self._job["warnings"] = [
                    warning
                    for warning in self._job.get("warnings", [])
                    if not isinstance(warning, dict)
                    or warning.get("code")
                    not in {"SNAPSHOT_PROMOTION_FAILED", "ANALYTICS_REBUILD_FAILED"}
                ]
                self._persist_locked()
            return response
        finally:
            process_lock.release()

    def _run(
        self,
        job_id: str,
        modules: list[str],
        mode: str,
        index_scopes: list[str],
    ) -> None:
        process: subprocess.Popen[str] | None = None
        reader: threading.Thread | None = None
        postprocess_heartbeat_stop: threading.Event | None = None
        postprocess_heartbeat: threading.Thread | None = None
        timeout = 0
        target_data_dir: Path | None = None
        analytics_sources_before: tuple[tuple[str, int, int], ...] = ()
        try:
            timeout_env = "DATA_FULL_REFRESH_TIMEOUT_SECONDS" if mode == "full" else "DATA_REFRESH_TIMEOUT_SECONDS"
            timeout_default = "86400" if mode == "full" else "1800"
            timeout = int(os.getenv(timeout_env, timeout_default))
            if timeout < 1:
                raise ValueError("DATA_REFRESH_TIMEOUT_SECONDS 必须大于 0。")
            child_env = os.environ.copy()
            child_env.pop("TUSHARE_TOKEN", None)
            child_env["PYTHONUNBUFFERED"] = "1"
            child_env["TUSHARE_REFRESH_LOCK_HELD_BY_PARENT"] = "1"
            source_data_dir = self._current_data_dir()
            target_data_dir = source_data_dir
            resumed = False
            fetch_complete = False
            if mode == "full":
                with self._lock:
                    if self._job.get("job_id") == job_id:
                        resumed_dir = self._safe_staging_dir(self._job.get("staging_data_dir"))
                        resumed = bool(self._job.get("resumed") and resumed_dir is not None)
                        fetch_complete = bool(self._job.get("fetch_complete")) if resumed else False
                    else:
                        resumed_dir = None
                target_data_dir = resumed_dir if resumed_dir is not None else prepare_full_refresh_staging(
                    data_root=self.data_dir,
                    source_dir=source_data_dir,
                    job_id=job_id,
                )
                with self._lock:
                    if self._job.get("job_id") == job_id:
                        self._job["staging_data_dir"] = str(target_data_dir)
                        self._persist_locked()
                self._append_log(
                    job_id,
                    (
                        f"[INFO] 续跑已保留隔离版本: {target_data_dir}\n"
                        if resumed
                        else f"[INFO] 全量更新写入隔离版本目录: {target_data_dir}\n"
                    ),
                    force_persist=True,
                )
            if mode == "incremental":
                analytics_sources_before = _analytics_source_state(target_data_dir)
            with self._lock:
                if self._active_job_id != job_id or self._active_process_lock is None:
                    raise RuntimeError("数据刷新锁状态异常，拒绝启动抓取子进程。")
                inherited_lock_fd = self._active_process_lock.fileno()
            if fetch_complete:
                self._append_log(
                    job_id,
                    "[INFO] 候选数据抓取已完成，本次仅重建并验收分析快照，不调用 Tushare。\n",
                    force_persist=True,
                )
            else:
                process = subprocess.Popen(
                    build_refresh_command(
                        modules,
                        mode,
                        index_scopes=index_scopes,
                        data_dir=target_data_dir,
                        resume=resumed,
                    ),
                    cwd=PROJECT_ROOT,
                    env=child_env,
                    pass_fds=(inherited_lock_fd,),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                with self._lock:
                    if self._job.get("job_id") == job_id:
                        self._job["worker_pid"] = process.pid
                        self._persist_locked()
                reader = threading.Thread(
                    target=self._stream_process_output,
                    args=(job_id, process),
                    name=f"tushare-log-{job_id[:8]}",
                    daemon=True,
                )
                reader.start()
                deadline = time.monotonic() + timeout
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(process.args, timeout)
                    try:
                        returncode = process.wait(timeout=min(5.0, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        self._touch_heartbeat(job_id)
                reader.join(timeout=10)
                if returncode != 0:
                    raise RuntimeError(f"更新命令退出码 {returncode}")
                with self._lock:
                    if self._job.get("job_id") == job_id:
                        self._job["fetch_complete"] = True
                        self._job["worker_pid"] = None
                        self._persist_locked()
            try:
                from trading_calendar import _load_calendar

                _load_calendar.cache_clear()
            except Exception:
                pass
            warnings: list[dict[str, str]] = []
            postprocess_heartbeat_stop = threading.Event()
            postprocess_heartbeat = threading.Thread(
                target=lambda: self._postprocess_heartbeat_loop(
                    job_id, postprocess_heartbeat_stop
                ),
                name=f"tushare-postprocess-heartbeat-{job_id[:8]}",
                daemon=True,
            )
            postprocess_heartbeat.start()
            try:
                reusable_snapshot = (
                    _reusable_analytics_snapshot(
                        target_data_dir,
                        analytics_sources_before,
                        _analytics_source_state(target_data_dir),
                    )
                    if mode == "incremental"
                    else None
                )
                if reusable_snapshot is not None:
                    analytics_snapshot = reusable_snapshot
                    message = "Tushare 增量更新完成；业绩指标数据源无变化，沿用现有分析快照"
                    self._append_log(
                        job_id,
                        "[INFO] 业绩指标数据源内容无变化，跳过本地分析快照重建。\n",
                        force_persist=True,
                    )
                else:
                    analytics_result = rebuild_local_analytics_snapshot(target_data_dir)
                    analytics_snapshot = {
                        "status": "succeeded",
                        "rebuilt_at": utc_now(),
                    }
                    for key in (
                        "path",
                        "rows",
                        "by_kind",
                        "as_of",
                        "source_files",
                        "updated_at",
                        "message",
                    ):
                        if key in analytics_result:
                            analytics_snapshot[key] = analytics_result[key]
                    message = "Tushare 数据更新及分析快照重建完成"
                if mode == "full":
                    promotion = validate_and_activate_full_refresh(
                        target_data_dir,
                        data_root=self.data_dir,
                        **({"index_scopes": index_scopes} if index_scopes else {}),
                    )
                    analytics_snapshot["validation_status"] = promotion["validation"]["status"]
                    analytics_snapshot["activated_snapshot_dir"] = promotion["manifest"]["snapshot_dir"]
                    message = "Tushare 全量数据、分析快照验收完成并已原子切换"
            except Exception as exc:  # noqa: BLE001
                warning_message = self._sanitise(str(exc))[-1000:]
                analytics_snapshot = {
                    "status": "failed",
                    "failed_at": utc_now(),
                    "message": warning_message,
                }
                warning_code = "SNAPSHOT_PROMOTION_FAILED" if mode == "full" else "ANALYTICS_REBUILD_FAILED"
                warnings.append({"code": warning_code, "message": warning_message})
                self._append_log(
                    job_id,
                    f"\n[WARN] Tushare 数据已更新，但分析快照验收/接入失败: {warning_message}\n",
                    force_persist=True,
                )
                message = (
                    "Tushare 全量抓取完成，但候选版本未通过分析快照验收/接入；"
                    "旧版本继续服务，检查点与候选目录已保留，无需重新拉取数据"
                    if mode == "full"
                    else "Tushare 数据更新完成；分析快照重建失败，可单独重建，无需重新拉取数据"
                )
            final = {
                "status": "succeeded",
                "message": message,
                "analytics_snapshot": analytics_snapshot,
                "warnings": warnings,
            }
        except subprocess.TimeoutExpired:
            if process is not None:
                self._stop_process(process)
            final = {
                "status": "failed",
                "message": f"数据更新超过 {timeout} 秒，已终止",
            }
        except Exception as exc:  # noqa: BLE001
            if process is not None:
                self._stop_process(process)
            error_text = self._sanitise(str(exc))
            self._append_log(job_id, f"\n[ERROR] {error_text}\n", force_persist=True)
            final = {"status": "failed", "message": "Tushare 数据更新失败"}
        finally:
            if postprocess_heartbeat_stop is not None:
                postprocess_heartbeat_stop.set()
            if postprocess_heartbeat is not None and postprocess_heartbeat.ident is not None:
                postprocess_heartbeat.join(timeout=10)
            if reader is not None and reader.is_alive():
                reader.join(timeout=10)
            self._finish_job(job_id, final)


refresh_manager = DataRefreshManager()
