"""Small process-safe primitives shared by refresh entry points."""

from __future__ import annotations

import errno
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional, TextIO

try:  # pragma: no cover - unavailable only on Windows
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


class InterProcessFileLock:
    """Hold a non-blocking advisory lock for the lifetime of this object."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._handle: Optional[TextIO] = None

    @property
    def acquired(self) -> bool:
        return self._handle is not None

    def fileno(self) -> int:
        """Return the locked descriptor so a refresh child can inherit it."""

        if self._handle is None:
            raise RuntimeError("刷新锁尚未获取。")
        return self._handle.fileno()

    def acquire(self, *, owner: str = "") -> bool:
        if self._handle is not None:
            return True
        if fcntl is None:  # pragma: no cover
            raise RuntimeError("当前平台不支持 fcntl，无法提供可靠的跨进程刷新锁。")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                return False
            raise
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(owner)
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
            raise
        self._handle = handle
        return True

    def release(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()

    def __enter__(self) -> "InterProcessFileLock":
        if not self.acquire():
            raise RuntimeError("已有数据刷新进程持有全局锁。")
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()


def is_file_lock_held(path: Path) -> bool:
    """Return whether another open file description currently owns the lock."""

    if fcntl is None:  # pragma: no cover
        return Path(path).exists()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                return True
            raise
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return False
    finally:
        handle.close()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Durably replace a small JSON state file without exposing partial writes."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def read_json_object(path: Path) -> Optional[dict[str, Any]]:
    """Read an object state file; malformed files are treated as unavailable."""

    path = Path(path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None
