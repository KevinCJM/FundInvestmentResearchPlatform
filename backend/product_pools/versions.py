"""Publication, effective-version selection, and version comparison."""

from __future__ import annotations

import copy
from datetime import date
from typing import Any

from .constants import FINAL_RESEARCH_STATUSES
from .domain import content_hash, member_eligibility, parse_date, product_key, today, trimmed, utc_now
from .errors import ProductPoolValidationError
from .repository import ProductPoolRepository


class ProductPoolVersionService:
    def __init__(self, pools: ProductPoolRepository) -> None:
        self.pools = pools

    @staticmethod
    def _validate_publishable_members(members: list[dict[str, Any]]) -> None:
        if not members:
            raise ProductPoolValidationError(
                "PRODUCT_POOL_HAS_NO_MEMBERS",
                "产品池没有候选产品。",
            )
        unresolved = [
            member
            for member in members
            if member.get("research_status") not in FINAL_RESEARCH_STATUSES
        ]
        if unresolved:
            raise ProductPoolValidationError(
                "PRODUCT_POOL_REVIEW_INCOMPLETE",
                "仍有候选或研究中的产品，完成研究结论后才能发布。",
                diagnostics=[
                    {
                        "kind": item.get("kind"),
                        "product_id": item.get("product_id"),
                        "name": item.get("name"),
                    }
                    for item in unresolved[:20]
                ],
            )
        if any(not str(member.get("decision_reason") or "").strip() for member in members):
            raise ProductPoolValidationError(
                "PRODUCT_POOL_DECISION_REASON_MISSING",
                "每个产品都必须保存准入、观察或排除原因。",
                field="decision_reason",
            )

    @staticmethod
    def _source_snapshot(bindings: list[dict[str, Any]]) -> tuple[str, str | None]:
        if not bindings:
            raise ProductPoolValidationError(
                "PRODUCT_POOL_HAS_NO_PLANS",
                "请至少导入一套评价方案。",
            )
        as_of_values = {
            str((binding.get("source_run") or {}).get("as_of") or "")
            for binding in bindings
        }
        if "" in as_of_values or len(as_of_values) != 1:
            raise ProductPoolValidationError(
                "MIXED_PRODUCT_POOL_AS_OF",
                "同一产品池版本下的评价方案必须使用同一数据截止日。",
                field="as_of",
            )
        generations = {
            str((binding.get("source_run") or {}).get("data_generation") or "")
            for binding in bindings
            if (binding.get("source_run") or {}).get("data_generation")
        }
        if len(generations) > 1:
            raise ProductPoolValidationError(
                "MIXED_PRODUCT_POOL_DATA_GENERATION",
                "同一产品池版本不能混用不同市场数据快照。请重新运行相关评价方案。",
            )
        return next(iter(as_of_values)), next(iter(generations), None)

    @staticmethod
    def _publish_dates(fields: dict[str, Any], data_as_of: date) -> tuple[date, date | None]:
        effective = parse_date(fields.get("effective_date"), field="effective_date")
        assert effective is not None
        if effective < data_as_of:
            raise ProductPoolValidationError(
                "EFFECTIVE_DATE_BEFORE_DATA",
                "生效日不能早于产品评价截止日。",
                field="effective_date",
            )
        expires = parse_date(fields.get("expires_at"), field="expires_at", required=False)
        if expires is not None and expires < effective:
            raise ProductPoolValidationError(
                "INVALID_EXPIRY_DATE",
                "失效日不能早于生效日。",
                field="expires_at",
            )
        return effective, expires

    def publish(
        self,
        pool: dict[str, Any],
        revision: int,
        fields: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        bindings = list(pool.get("evaluation_plans") or [])
        members = list(pool.get("members") or [])
        self._validate_publishable_members(members)
        data_as_of_text, data_generation = self._source_snapshot(bindings)
        data_as_of = parse_date(data_as_of_text, field="as_of")
        assert data_as_of is not None
        effective, expires = self._publish_dates(fields, data_as_of)
        eligible_count = sum(member_eligibility(member, effective)[0] for member in members)
        if eligible_count == 0:
            raise ProductPoolValidationError(
                "PRODUCT_POOL_HAS_NO_ELIGIBLE_MEMBERS",
                "产品池没有可供新增配置使用的准入产品。",
            )
        snapshot = {
            "pool_id": pool["id"],
            "name": pool["name"],
            "description": pool.get("description", ""),
            "owner": pool.get("owner", ""),
            "data_as_of": data_as_of.isoformat(),
            "data_generation": data_generation,
            "evaluation_plans": copy.deepcopy(bindings),
            "members": copy.deepcopy(members),
            "summary": {
                "evaluation_plan_count": len(bindings),
                "member_count": len(members),
                "eligible_count": eligible_count,
                "approved_count": sum(member.get("research_status") == "approved" for member in members),
                "watch_count": sum(member.get("research_status") == "watch" for member in members),
                "excluded_count": sum(member.get("research_status") == "excluded" for member in members),
            },
        }
        digest = content_hash(snapshot)
        return self.pools.publish(
            pool["id"],
            revision,
            version_fields={
                "effective_date": effective.isoformat(),
                "expires_at": expires.isoformat() if expires else None,
                "published_by": trimmed(
                    fields.get("published_by"), field="published_by", maximum=80
                ),
                "change_note": trimmed(
                    fields.get("change_note"), field="change_note", maximum=500
                ),
                "content_hash": digest,
                "snapshot": snapshot,
            },
            current_fields={
                "status": "active",
                "last_published_at": utc_now(),
                "last_published_hash": digest,
            },
        )

    def _effective_versions(self, on_date: date) -> dict[str, dict[str, Any]]:
        active_pool_ids = {
            pool["id"] for pool in self.pools.list() if pool.get("status") == "active"
        }
        selected: dict[str, dict[str, Any]] = {}
        for version in self.pools.list_all_versions():
            if version.get("pool_id") not in active_pool_ids:
                continue
            effective = date.fromisoformat(version["effective_date"])
            expiry = date.fromisoformat(version["expires_at"]) if version.get("expires_at") else None
            if effective > on_date or (expiry and expiry < on_date):
                continue
            previous = selected.get(version["pool_id"])
            if previous is None or (
                effective,
                int(version["version_number"]),
            ) > (
                date.fromisoformat(previous["effective_date"]),
                int(previous["version_number"]),
            ):
                selected[version["pool_id"]] = version
        return selected

    def decorate(
        self,
        version: dict[str, Any],
        *,
        on_date: date | None = None,
        active_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        reference_date = on_date or today()
        effective = date.fromisoformat(version["effective_date"])
        expiry = date.fromisoformat(version["expires_at"]) if version.get("expires_at") else None
        if expiry and expiry < reference_date:
            lifecycle_status = "expired"
        elif effective > reference_date:
            lifecycle_status = "scheduled"
        else:
            ids = active_ids if active_ids is not None else {
                item["id"] for item in self._effective_versions(reference_date).values()
            }
            lifecycle_status = "active" if version["id"] in ids else "superseded"
        snapshot = version.get("snapshot") or {}
        return {
            **copy.deepcopy(version),
            "lifecycle_status": lifecycle_status,
            "name": snapshot.get("name"),
            "data_as_of": snapshot.get("data_as_of"),
            "data_generation": snapshot.get("data_generation"),
            "summary": copy.deepcopy(snapshot.get("summary") or {}),
        }

    def list(self, pool_id: str) -> dict[str, Any]:
        reference_date = today()
        active_ids = {item["id"] for item in self._effective_versions(reference_date).values()}
        items = [
            self.decorate(version, on_date=reference_date, active_ids=active_ids)
            for version in self.pools.list_versions(pool_id)
        ]
        return {"items": items, "total": len(items)}

    def get(self, version_id: str) -> dict[str, Any]:
        return self.decorate(self.pools.get_version(version_id))

    def available(self, as_of: str) -> dict[str, Any]:
        on_date = parse_date(as_of, field="as_of")
        assert on_date is not None
        selected = self._effective_versions(on_date)
        active_ids = {item["id"] for item in selected.values()}
        items = [
            self.decorate(version, on_date=on_date, active_ids=active_ids)
            for version in selected.values()
        ]
        items.sort(key=lambda item: str(item.get("name") or ""))
        return {"items": items, "total": len(items), "as_of": on_date.isoformat()}

    def compare(self, base_version_id: str, target_version_id: str) -> dict[str, Any]:
        base = self.pools.get_version(base_version_id)
        target = self.pools.get_version(target_version_id)
        if base["pool_id"] != target["pool_id"]:
            raise ProductPoolValidationError(
                "PRODUCT_POOL_VERSION_POOL_MISMATCH",
                "只能比较同一个产品池的两个版本。",
            )
        base_members = {
            product_key(item.get("kind"), item.get("product_id")): item
            for item in (base.get("snapshot") or {}).get("members") or []
        }
        target_members = {
            product_key(item.get("kind"), item.get("product_id")): item
            for item in (target.get("snapshot") or {}).get("members") or []
        }
        tracked = (
            "evaluation_plan_id",
            "evaluation_plan_revision",
            "research_status",
            "usage_status",
            "decision_reason",
            "max_weight",
            "valid_until",
            "next_review_date",
        )
        changed: list[dict[str, Any]] = []
        for key in sorted(base_members.keys() & target_members.keys()):
            fields = [
                field
                for field in tracked
                if base_members[key].get(field) != target_members[key].get(field)
            ]
            if fields:
                changed.append(
                    {
                        "kind": key[0],
                        "product_id": key[1],
                        "name": target_members[key].get("name") or base_members[key].get("name"),
                        "changed_fields": fields,
                        "before": {field: base_members[key].get(field) for field in fields},
                        "after": {field: target_members[key].get(field) for field in fields},
                    }
                )
        added_keys = target_members.keys() - base_members.keys()
        removed_keys = base_members.keys() - target_members.keys()
        return {
            "pool_id": base["pool_id"],
            "base_version": self.decorate(base),
            "target_version": self.decorate(target),
            "added": [copy.deepcopy(target_members[key]) for key in sorted(added_keys)],
            "removed": [copy.deepcopy(base_members[key]) for key in sorted(removed_keys)],
            "changed": changed,
            "summary": {
                "added": len(added_keys),
                "removed": len(removed_keys),
                "changed": len(changed),
            },
        }


__all__ = ["ProductPoolVersionService"]
