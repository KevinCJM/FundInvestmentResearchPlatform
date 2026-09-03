"""Workspace configuration for indicator values precomputed after data refresh."""

from __future__ import annotations

import hashlib
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


def normalized_snapshot_item(item: dict[str, Any]) -> dict[str, Any]:
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
