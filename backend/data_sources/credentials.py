"""Private, destination-bound credentials shared by every data source."""
from __future__ import annotations
import hashlib
import json
import os
import tempfile
from pathlib import Path

from .models import CenterError
from .store import SourceStore


def credential_path(store: SourceStore, source_id: str) -> Path:
    store.get("source", source_id)
    # Compatibility location only; the UI and authorization rules are identical.
    return store.root / ".tushare_token" if source_id == "tushare" else store.root / ".source_credentials" / source_id


def _signature(config: dict) -> str:
    body = {key: config.get(key) for key in ("transport", "base_url", "auth_mode", "auth_header")}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def destination_matches(store: SourceStore, source_id: str) -> bool:
    config = store.get("source", source_id)["config"]
    binding = store.root / ".source_credentials" / (source_id + ".binding.json")
    if binding.exists():
        try:
            return json.loads(binding.read_text())["signature"] == _signature(config)
        except (OSError, ValueError, KeyError):
            return False
    # Existing credentials are bound to the original saved destination, not
    # whatever address was most recently edited in the browser.
    with store.connection() as db:
        first = db.execute("SELECT body FROM source_config_revision WHERE kind='source' AND id=? ORDER BY revision LIMIT 1", (source_id,)).fetchone()
    return first is not None and _signature(json.loads(first[0])) == _signature(config)


def configured(store: SourceStore, source_id: str) -> bool:
    path = credential_path(store, source_id)
    return path.is_file() and path.stat().st_size > 0 and destination_matches(store, source_id)


def read_credential(store: SourceStore, source_id: str) -> str:
    if not configured(store, source_id):
        raise CenterError("CREDENTIAL_REQUIRED", "请保存访问凭据；修改来源地址或认证方式后需重新确认凭据。")
    value = credential_path(store, source_id).read_text().strip()
    if not value:
        raise CenterError("CREDENTIAL_REQUIRED", "请先配置数据源访问凭据。")
    return value


def _atomic_private(path: Path, value: str) -> None:
    path.parent.mkdir(mode=0o700, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def save_credential(store: SourceStore, source_id: str, value: str | None) -> None:
    path = credential_path(store, source_id)
    binding = store.root / ".source_credentials" / (source_id + ".binding.json")
    if value is None:
        path.unlink(missing_ok=True)
        binding.unlink(missing_ok=True)
        return
    if not value.strip() or len(value) > 4096 or "\n" in value or "\r" in value:
        raise CenterError("INVALID_CREDENTIAL", "凭据为空、过长或包含换行。")
    config = store.get("source", source_id)["config"]
    _atomic_private(path, value.strip())
    _atomic_private(binding, json.dumps({"signature": _signature(config)}))
