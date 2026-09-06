"""One-request acceptance of the configured Tushare transport and mapping.

Explicit network opt-in; share the workspace quota/refresh lock, but write all
raw/Parquet candidates into a temporary directory, never an active snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.data_sources.batches import capture_batch
from backend.data_sources.models import CenterError
from backend.data_sources.presets import default_interfaces, default_source
from backend.data_sources.runtime import fetch_once
from backend.data_sources.store import DEFAULT_ROOT, SourceStore
from backend.services.refresh_runtime import InterProcessFileLock
from config import require_tushare_token


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", required=True)
    parser.add_argument("--allow-config-token", action="store_true", required=True)
    args = parser.parse_args()
    del args
    files = [ROOT / "T01_get_data.py", *sorted((ROOT / "backend/data_sources").glob("*.py"))]
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.relative_to(ROOT).as_posix().encode())
        digest.update(file.read_bytes())
    report = {"implementation_sha256": digest.hexdigest(), "api": "fund_daily", "max_requests": 1}
    try:
        token = require_tushare_token()
        live_store = SourceStore(DEFAULT_ROOT)
        lock = InterProcessFileLock(DEFAULT_ROOT / ".tushare_refresh.lock")
        if not lock.acquire(owner="source-center-one-request-smoke"):
            raise CenterError("REFRESH_RUNNING", "已有更新任务，未发起真实请求。")
        try:
            source = default_source()
            interface = next(item for item in default_interfaces() if item.api_name == "fund_daily")
            params = {"ts_code": "510300.SH", "trade_date": "20260828"}
            _, rows = fetch_once(live_store, source, interface, params, credential=token)
            if not rows:
                raise CenterError("EMPTY_SMOKE_SAMPLE", "固定日期样本为空，未通过验收。")
            with tempfile.TemporaryDirectory(prefix="source-center-smoke-") as directory:
                store = SourceStore(Path(directory))
                batch = capture_batch(store, interface, rows, params, report["implementation_sha256"])
                if batch["status"] != "VALIDATED_CANDIDATE":
                    raise CenterError("SMOKE_MAPPING_FAILED", "真实样本映射未通过。")
                import pyarrow.parquet as pq
                for entry in batch["tables"]:
                    schema = pq.read_schema(store.root / entry["artifact"])
                    if schema.metadata[b"contract_version"] != b"1.2.0":
                        raise CenterError("SMOKE_SCHEMA_FAILED", "候选数据版本错误。")
                report.update(status="passed", requests=1, source_rows=len(rows),
                              mapped_tables=len(batch["tables"]), published=False,
                              outputs="temporary_directory_removed")
        finally:
            lock.release()
    except Exception as exc:
        report.update(status="not_passed", error_code=exc.code if isinstance(exc, CenterError) else type(exc).__name__)
        print(json.dumps(report, ensure_ascii=False))
        return 1
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
