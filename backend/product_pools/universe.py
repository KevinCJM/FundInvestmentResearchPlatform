"""Immutable pre-investment universe construction and product search."""

from __future__ import annotations

import copy
from datetime import date
from typing import Any

from .constants import USAGE_SEVERITY
from .domain import content_hash, member_eligibility, parse_date, product_key, trimmed
from .errors import ProductPoolConflictError, ProductPoolValidationError
from .repository import InvestableUniverseRepository, ProductPoolRepository
from .versions import ProductPoolVersionService


class InvestableUniverseService:
    def __init__(
        self,
        pools: ProductPoolRepository,
        universes: InvestableUniverseRepository,
        versions: ProductPoolVersionService,
    ) -> None:
        self.pools = pools
        self.universes = universes
        self.versions = versions

    def _resolve_versions(
        self,
        version_ids: list[str],
        research_date: date,
    ) -> list[dict[str, Any]]:
        if not version_ids:
            raise ProductPoolValidationError(
                "PRODUCT_POOL_VERSION_REQUIRED",
                "请至少选择一个产品池版本。",
                field="version_ids",
            )
        if len(version_ids) > 20:
            raise ProductPoolValidationError(
                "TOO_MANY_PRODUCT_POOL_VERSIONS",
                "一次最多合并 20 个产品池版本。",
                field="version_ids",
            )
        resolved = [self.pools.get_version(version_id) for version_id in version_ids]
        if len({version["pool_id"] for version in resolved}) != len(resolved):
            raise ProductPoolValidationError(
                "DUPLICATE_PRODUCT_POOL",
                "同一个产品池只能选择一个有效版本。",
                field="version_ids",
            )
        effective = {
            item["pool_id"]: item["id"]
            for item in self.versions.available(research_date.isoformat())["items"]
        }
        for version in resolved:
            if effective.get(version["pool_id"]) != version["id"]:
                raise ProductPoolConflictError(
                    "PRODUCT_POOL_VERSION_NOT_EFFECTIVE",
                    "所选产品池版本在研究日期不是当前有效版本。",
                    field="version_ids",
                )
        return resolved

    @staticmethod
    def _source_record(
        version: dict[str, Any],
        snapshot: dict[str, Any],
        member: dict[str, Any],
    ) -> dict[str, Any]:
        plan_names = {
            binding.get("plan_id"): binding.get("plan_name")
            for binding in snapshot.get("evaluation_plans") or []
        }
        return {
            "pool_id": version["pool_id"],
            "pool_name": snapshot.get("name"),
            "pool_version_id": version["id"],
            "pool_version_number": version["version_number"],
            "evaluation_plan_id": member.get("evaluation_plan_id"),
            "evaluation_plan_revision": member.get("evaluation_plan_revision"),
            "evaluation_plan_name": plan_names.get(member.get("evaluation_plan_id")),
            "source_rank": member.get("source_rank"),
            "source_score": member.get("source_score"),
        }

    @staticmethod
    def _new_merged_member(
        member: dict[str, Any],
        source: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "kind": member.get("kind"),
            "product_id": member.get("product_id"),
            "code": member.get("product_id"),
            "name": member.get("name") or member.get("product_id"),
            "research_status": member.get("research_status"),
            "usage_status": member.get("usage_status"),
            "decision_reasons": [member.get("decision_reason")]
            if member.get("decision_reason")
            else [],
            "max_weight": member.get("max_weight"),
            "valid_until": member.get("valid_until"),
            "substitute_groups": [member.get("substitute_group")]
            if member.get("substitute_group")
            else [],
            "evaluation_sources": [source],
        }

    @staticmethod
    def _merge_research_status(current: str | None, incoming: str | None) -> str:
        statuses = {current, incoming}
        if "excluded" in statuses:
            return "excluded"
        if "watch" in statuses:
            return "watch"
        return "approved"

    @staticmethod
    def _merge_member(
        current: dict[str, Any],
        member: dict[str, Any],
        source: dict[str, Any],
    ) -> None:
        current["evaluation_sources"].append(source)
        if member.get("decision_reason"):
            current["decision_reasons"].append(member["decision_reason"])
        if member.get("substitute_group"):
            current["substitute_groups"].append(member["substitute_group"])
        current["research_status"] = InvestableUniverseService._merge_research_status(
            current.get("research_status"),
            member.get("research_status"),
        )
        current_usage = str(current.get("usage_status") or "normal")
        incoming_usage = str(member.get("usage_status") or "normal")
        if USAGE_SEVERITY[incoming_usage] > USAGE_SEVERITY[current_usage]:
            current["usage_status"] = incoming_usage
        weights = [
            float(value)
            for value in (current.get("max_weight"), member.get("max_weight"))
            if value is not None
        ]
        current["max_weight"] = min(weights) if weights else None
        dates = [
            str(value)
            for value in (current.get("valid_until"), member.get("valid_until"))
            if value
        ]
        current["valid_until"] = min(dates) if dates else None

    def _merge_versions(
        self,
        versions: list[dict[str, Any]],
        research_date: date,
    ) -> list[dict[str, Any]]:
        merged: dict[tuple[str, str], dict[str, Any]] = {}
        for version in versions:
            snapshot = version.get("snapshot") or {}
            for member in snapshot.get("members") or []:
                key = product_key(member.get("kind"), member.get("product_id"))
                source = self._source_record(version, snapshot, member)
                if key not in merged:
                    merged[key] = self._new_merged_member(member, source)
                else:
                    self._merge_member(merged[key], member, source)
        result: list[dict[str, Any]] = []
        for item in merged.values():
            eligible, reasons = member_eligibility(item, research_date)
            warnings: list[str] = []
            if item.get("research_status") == "watch":
                warnings.append("产品处于观察状态")
            if item.get("usage_status") == "limited":
                warnings.append("产品存在使用限额")
            result.append(
                {
                    **item,
                    "decision_reasons": list(dict.fromkeys(item.get("decision_reasons") or [])),
                    "substitute_groups": list(dict.fromkeys(item.get("substitute_groups") or [])),
                    "eligible": eligible,
                    "eligibility_reasons": reasons,
                    "warnings": warnings,
                }
            )
        result.sort(
            key=lambda item: (
                not bool(item["eligible"]),
                str(item.get("kind")),
                str(item.get("name")),
                str(item.get("product_id")),
            )
        )
        return result

    @staticmethod
    def _version_references(versions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "pool_id": version["pool_id"],
                "pool_name": (version.get("snapshot") or {}).get("name"),
                "version_id": version["id"],
                "version_number": version["version_number"],
                "effective_date": version["effective_date"],
                "data_as_of": (version.get("snapshot") or {}).get("data_as_of"),
                "content_hash": version.get("content_hash"),
            }
            for version in versions
        ]

    def create(self, fields: dict[str, Any]) -> dict[str, Any]:
        name = trimmed(fields.get("name"), field="name", maximum=100, required=True)
        research_date = parse_date(fields.get("research_date"), field="research_date")
        assert research_date is not None
        version_ids = list(
            dict.fromkeys(
                str(item).strip()
                for item in fields.get("version_ids") or []
                if str(item).strip()
            )
        )
        versions = self._resolve_versions(version_ids, research_date)
        members = self._merge_versions(versions, research_date)
        payload = {
            "name": name,
            "research_date": research_date.isoformat(),
            "version_refs": self._version_references(versions),
            "members": members,
            "summary": {
                "pool_count": len(versions),
                "member_count": len(members),
                "eligible_count": sum(bool(item["eligible"]) for item in members),
                "restricted_count": sum(not bool(item["eligible"]) for item in members),
                "watch_count": sum(item.get("research_status") == "watch" for item in members),
            },
        }
        payload["content_hash"] = content_hash(payload)
        return self.universes.create(payload)

    def list(self) -> dict[str, Any]:
        keys = (
            "id",
            "name",
            "research_date",
            "version_refs",
            "summary",
            "content_hash",
            "created_at",
            "immutable",
        )
        items = [
            {key: copy.deepcopy(item.get(key)) for key in keys}
            for item in self.universes.list()
        ]
        return {"items": items, "total": len(items)}

    def get(self, snapshot_id: str) -> dict[str, Any]:
        return self.universes.get(snapshot_id)

    def search_products(
        self,
        snapshot_id: str,
        *,
        query: str = "",
        kind: str | None = None,
        eligible_only: bool = True,
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        if kind is not None and kind not in {"etf", "fund"}:
            raise ProductPoolValidationError(
                "INVALID_PRODUCT_KIND",
                "产品类型无效。",
                field="kind",
            )
        snapshot = self.universes.get(snapshot_id)
        members = list(snapshot.get("members") or [])
        if eligible_only:
            members = [item for item in members if item.get("eligible")]
        if kind:
            members = [item for item in members if item.get("kind") == kind]
        normalized_query = query.strip().lower()
        if normalized_query:
            members = [
                item
                for item in members
                if normalized_query
                in " ".join(
                    [
                        str(item.get("product_id") or ""),
                        str(item.get("name") or ""),
                        *[
                            str(source.get("evaluation_plan_name") or "")
                            for source in item.get("evaluation_sources") or []
                        ],
                    ]
                ).lower()
            ]
        normalized_page = max(1, int(page))
        normalized_size = max(1, min(int(page_size), 100))
        total = len(members)
        start = (normalized_page - 1) * normalized_size
        return {
            "snapshot_id": snapshot_id,
            "snapshot_name": snapshot.get("name"),
            "research_date": snapshot.get("research_date"),
            "items": copy.deepcopy(members[start : start + normalized_size]),
            "total": total,
            "page": normalized_page,
            "page_size": normalized_size,
        }


__all__ = ["InvestableUniverseService"]
