"""Workspace configuration for indicator values precomputed after data refresh."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


SNAPSHOT_CONFIG_SCHEMA_VERSION = 1
MAX_SNAPSHOT_INDICATORS = 30


# These aliases preserve the public product filter/ranking contract while the
# value itself is now owned by the corresponding Indicator Center definition.
DEFAULT_SNAPSHOT_INDICATORS: tuple[dict[str, Any], ...] = (
    {"indicator_id": "builtin-total-return-v2", "indicator_revision": 1, "period": "1M", "field": "return_1m"},
    {"indicator_id": "builtin-total-return-v2", "indicator_revision": 1, "period": "3M", "field": "return_3m"},
    {"indicator_id": "builtin-total-return-v2", "indicator_revision": 1, "period": "1Y", "field": "return_1y"},
    {"indicator_id": "builtin-total-return-v2", "indicator_revision": 1, "period": "3Y", "field": "return_3y"},
    {"indicator_id": "builtin-annualized-volatility-v2", "indicator_revision": 1, "period": "1Y", "field": "annual_volatility_1y"},
    {"indicator_id": "builtin-maximum-drawdown-v2", "indicator_revision": 1, "period": "3Y", "field": "max_drawdown_3y"},
    {"indicator_id": "builtin-annualized-sharpe-v2", "indicator_revision": 1, "period": "1Y", "field": "sharpe_1y"},
    {"indicator_id": "builtin-calmar-ratio-v2", "indicator_revision": 1, "period": "3Y", "field": "calmar_3y"},
)


def snapshot_field_name(indicator_id: str, indicator_revision: int, period: str) -> str:
    """Return a deterministic, column-safe name for a configured metric."""

    readable = re.sub(r"[^a-z0-9]+", "_", indicator_id.lower()).strip("_")[-36:]
    digest = hashlib.sha256(
        f"{indicator_id}@{int(indicator_revision)}:{period.upper()}".encode("utf-8")
    ).hexdigest()[:10]
    return f"metric_{readable}_{period.lower()}_{digest}"


def _legacy_normalized_snapshot_item(item: dict[str, Any]) -> dict[str, Any]:
    indicator_id = str(item.get("indicator_id") or "").strip()
    revision = int(item.get("indicator_revision") or 0)
    period = str(item.get("period") or "").strip().upper()
    field = str(item.get("field") or "").strip() or snapshot_field_name(
        indicator_id, revision, period
    )
    return {
        "indicator_id": indicator_id,
        "indicator_revision": revision,
        "period": period,
        "field": field,
    }


def normalized_snapshot_item(item: dict[str, Any]) -> dict[str, Any]:
    """Normalize scalar or explicit time-series-channel snapshot configuration."""

    indicator_id = str(item.get("indicator_id") or "").strip()
    revision = int(item.get("indicator_revision") or 0)
    period = str(item.get("period") or "").strip().upper()
    channel_id = str(item.get("channel_id") or "").strip() or None
    reducer = str(item.get("reducer") or "").strip() or None
    if channel_id and reducer is None:
        reducer = "last_finite"
    if reducer not in {None, "last_finite"}:
        reducer = str(reducer)
    payload = {
        "indicator_id": indicator_id,
        "indicator_revision": revision,
        "period": period,
        "channel_id": channel_id,
        "reducer": reducer,
    }
    field = str(item.get("field") or "").strip()
    if not field:
        if channel_id is None:
            # Preserve the established scalar snapshot field contract.
            field = snapshot_field_name(indicator_id, revision, period)
        else:
            readable = re.sub(
                r"[^a-z0-9]+",
                "_",
                indicator_id.lower(),
            ).strip("_")[-24:]
            channel = re.sub(
                r"[^a-z0-9]+",
                "_",
                channel_id.lower(),
            ).strip("_")[:20]
            digest = hashlib.sha256(
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()[:10]
            field = (
                f"metric_{readable}_{channel}_{period.lower()}_{digest}"
            )
    return {**payload, "field": field}
