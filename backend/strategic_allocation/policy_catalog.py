"""Display saved SAA lineage from frozen records, without loading live inputs."""
from copy import deepcopy


def policy_summary(baseline: dict) -> dict:
    policy = baseline["policy"]
    assumptions = policy.get("assumptions") or {}
    mandate = policy.get("mandate") or {}
    universe = baseline.get("strategic_universe_snapshot") or {}
    strategic_id = baseline.get("strategic_universe_id") or assumptions.get("strategic_universe_id")
    scope = {
        "research_path": "strategy_first" if strategic_id else "product_first",
        "id": strategic_id or baseline.get("universe_snapshot_id"),
        "name": (universe.get("name") or universe.get("definition", {}).get("name"))
                if strategic_id else baseline.get("alloc_name"),
    }
    sources = (policy.get("multi_cma") or {}).get("sources")
    cmas = ([{"id": source["cma_id"], "name": source.get("name")} for source in sources]
            if sources else [{"id": policy.get("cma_id"), "name": assumptions.get("name")}])
    return {
        **{key: baseline[key] for key in ("id", "name", "as_of", "created_at")},
        "mode": policy.get("mode", "single"),
        "mandate": {"id": policy.get("mandate_id"), "name": mandate.get("name"),
                    "definition": deepcopy(mandate)},
        "scope": scope, "cmas": cmas,
    }
