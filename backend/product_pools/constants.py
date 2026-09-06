"""Stable product-pool domain constants."""

from pathlib import Path


DEFAULT_DATA_DIR = (Path(__file__).resolve().parents[2] / "data").resolve()

# Universe snapshots are appended to the product-pool store by
# ``ProductPoolRepository``; a dedicated file would just be an empty store.
UNIVERSE_SNAPSHOT_STORE = "product_pools.json"

POOL_STATUSES = {"draft", "active", "archived"}
RESEARCH_STATUSES = {"candidate", "under_review", "approved", "watch", "excluded"}
USAGE_STATUSES = {"normal", "limited", "no_new", "hold_only", "unavailable"}
FINAL_RESEARCH_STATUSES = {"approved", "watch", "excluded"}
ELIGIBLE_RESEARCH_STATUSES = {"approved", "watch"}
ELIGIBLE_USAGE_STATUSES = {"normal", "limited"}
USAGE_SEVERITY = {
    "normal": 0,
    "limited": 1,
    "no_new": 2,
    "hold_only": 3,
    "unavailable": 4,
}


__all__ = [
    "DEFAULT_DATA_DIR",
    "ELIGIBLE_RESEARCH_STATUSES",
    "ELIGIBLE_USAGE_STATUSES",
    "FINAL_RESEARCH_STATUSES",
    "POOL_STATUSES",
    "RESEARCH_STATUSES",
    "USAGE_SEVERITY",
    "USAGE_STATUSES",
    "UNIVERSE_SNAPSHOT_STORE",
]
