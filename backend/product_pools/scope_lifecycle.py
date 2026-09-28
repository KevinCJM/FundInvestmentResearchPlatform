"""Shared research-scope lifecycle: one name space across both scope stores.

Product-pool snapshots and strategic universes live in different files, but a
user sees one library. Name checks therefore read both stores and serialize on
one governance lock file; frozen records are never rewritten by a rename or a
removal (supersession and retirement are recorded separately).
"""

from __future__ import annotations

from pathlib import Path

from backend.custom_indicators.errors import ConflictError, IndicatorDomainError, ValidationError
from backend.sensitivity.repository import ArtifactRepository

from .errors import ProductPoolError
from .repository import AtomicProductPoolStore

SCOPE_GOVERNANCE_FILE = "scope_governance.json"


def normalize_scope_name(name: str) -> str:
    return str(name or "").strip().casefold()


def strategic_artifacts_root(root: Path) -> Path:
    """The strategic service stores its artifacts under this fixed sub-path."""

    return Path(root) / "strategic_allocation" / "artifacts"


def scope_governance_lock(artifacts_root: Path) -> AtomicProductPoolStore:
    """One lock file serializes name checks and lifecycle writes across stores."""

    return AtomicProductPoolStore(Path(artifacts_root) / SCOPE_GOVERNANCE_FILE)


def scope_mandate_fields(root: Path, kind: str, record: dict) -> dict:
    """Project the exact bound mandate, without rewriting frozen scope bytes."""
    if record.get("mandate_id"):
        return {key: record[key] for key in ("mandate_id", "mandate_hash") if key in record}
    store = scope_governance_lock(root)
    try:
        bindings = store.read_unlocked().get("scope_mandates", {})
    except ProductPoolError as exc:
        raise ValidationError("SCOPE_BINDING_UNREADABLE", "研究范围与投资目标的关联暂时无法读取，请重试。") from exc
    if not isinstance(bindings, dict):
        raise ValidationError("SCOPE_BINDING_CORRUPT", "研究范围关联记录无法读取，请检查存储。")
    binding = bindings.get(f"{kind}:{record['id']}")
    if binding is None:
        return {}
    if not isinstance(binding, dict) or not binding.get("mandate_id") or not binding.get("mandate_hash") or binding.get("scope_hash") != record.get("content_hash"):
        raise ValidationError("SCOPE_BINDING_CORRUPT", "研究范围关联与保存版本不一致，请检查存储。")
    return {key: binding[key] for key in ("mandate_id", "mandate_hash")}


def _mandate_upgrade(root: Path, bound: str, requested: str) -> bool:
    """同一目标系列的当前版本可替换旧版本，见 docs/pre-investment/versioning.md。"""
    from backend.strategic_allocation.versioning import CURRENT, ResearchVersions

    index = ResearchVersions(ArtifactRepository(root)).index["mandate"]
    old, new = index.get(bound), index.get(requested)
    return bool(old and new and old["lineage_id"] == new["lineage_id"] and new["status"] == CURRENT)


def checked_scope_mandate(root: Path, requested: str | None, existing: dict | None = None,
                          allow_upgrade: bool = False) -> dict:
    """An edit inherits its original mandate or upgrades it within the same lineage;
    a different goal needs a new scope."""
    bound = (existing or {}).get("mandate_id")
    upgrade = bool(bound and requested and bound != requested)
    if upgrade and not (allow_upgrade and _mandate_upgrade(root, bound, requested)):
        raise ConflictError("SCOPE_MANDATE_CONFLICT", "该研究范围已绑定其他投资目标；如需更换目标，请复制为新研究。")
    identifier = requested if upgrade else bound or requested
    if not identifier:
        return {}
    repository = ArtifactRepository(root)
    mandate = repository.get(identifier, "series")
    if mandate.get("artifact_type") != "investment_mandate":
        raise ValidationError("SCOPE_MANDATE_INVALID", "请选择已保存的投资目标与约束。")
    if not upgrade and (existing or {}).get("mandate_hash") not in (None, mandate["content_hash"]):
        raise ValidationError("SCOPE_MANDATE_CHANGED", "绑定的投资目标版本与保存记录不一致，请检查存储。")
    return {"mandate_id": identifier, "mandate_hash": mandate["content_hash"]}


def bind_scope_mandate(root: Path, kind: str, record: dict, mandate_id: str) -> dict:
    """Fill a missing legacy association once, separately from immutable data."""
    store = scope_governance_lock(root)
    with store.locked():
        existing = scope_mandate_fields(root, kind, record)
        fields = checked_scope_mandate(root, mandate_id, existing)
        if not existing:
            payload = store.read_unlocked()
            payload.setdefault("scope_mandates", {})[f"{kind}:{record['id']}"] = {
                **fields, "scope_hash": record["content_hash"],
            }
            store.write_unlocked(payload)
        return fields


def product_scope_names(store_path: Path) -> dict[str, str]:
    """Active product-scope names (id -> name) from the shared product store."""

    path = Path(store_path)
    if not path.is_file():
        return {}
    try:
        payload = AtomicProductPoolStore(path).read_unlocked()
    except ProductPoolError as exc:
        # Strategic routes only speak the indicator error contract; keep the
        # typed status/code instead of leaking an unhandled 500.
        raise IndicatorDomainError(
            exc.code, exc.message, status_code=exc.status_code, field=exc.field,
        ) from exc
    records = list(payload.get("universe_snapshots") or [])
    retirements = {
        str(item.get("snapshot_id") or "")
        for item in payload.get("universe_retirements") or []
        if isinstance(item, dict)
    }
    superseded = {
        str(item.get("supersedes_snapshot_id"))
        for item in records
        if item.get("supersedes_snapshot_id")
    }
    return {
        str(item.get("id")): str(item.get("name") or "")
        for item in records
        if item.get("id")
        and str(item.get("id")) not in superseded
        and str(item.get("id")) not in retirements
    }


def strategic_scope_names(root: Path) -> dict[str, str]:
    """Active strategic-scope names (id -> name) via the guarded artifact store.

    A missing store means no saved scopes; an existing but corrupt store raises
    the repository's typed error instead of silently reporting an empty library.
    """

    repository = ArtifactRepository(strategic_artifacts_root(root))
    if not repository.root.exists():
        return {}
    require_artifact_index(repository)
    superseded: set[str] = set()
    retired: set[str] = set()
    active: dict[str, str] = {}
    for summary in repository.list():
        artifact_type = summary.get("artifact_type")
        if artifact_type not in {"strategic_universe", "strategic_universe_retirement"}:
            continue
        item = repository.get(str(summary.get("id")), summary.get("kind"))
        if artifact_type == "strategic_universe_retirement":
            if item.get("universe_id"):
                retired.add(str(item["universe_id"]))
            continue
        if item.get("supersedes_universe_id"):
            superseded.add(str(item["supersedes_universe_id"]))
        active[str(item.get("id"))] = str(item.get("name") or "")
    for object_id in superseded | retired:
        active.pop(object_id, None)
    return active


def require_artifact_index(repository: ArtifactRepository) -> None:
    """Fail closed when research artifacts exist but their index is missing.

    A root created only by the governance lock has no artifact folders and is a
    valid empty library; missing index next to real manifests is corruption.
    """

    if repository.index.path.exists() or not repository.root.exists():
        return
    for entry in repository.root.iterdir():
        if entry.is_dir() and (entry / "manifest.json").is_file():
            raise ValidationError(
                "RESEARCH_INDEX_MISSING",
                "研究成果索引缺失，已停止使用；不会自动重建替代原结果。",
            )


__all__ = [
    "SCOPE_GOVERNANCE_FILE",
    "normalize_scope_name",
    "product_scope_names",
    "require_artifact_index",
    "scope_governance_lock",
    "strategic_artifacts_root",
    "strategic_scope_names",
]
