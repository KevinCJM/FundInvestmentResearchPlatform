"""Shared validation of products against immutable investable-universe snapshots."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable

from .domain import member_eligibility, parse_date, product_key, today
from .errors import ProductPoolValidationError
from .repository import InvestableUniverseRepository


@dataclass(frozen=True)
class UniverseMembershipResult:
    reference: dict[str, Any]
    members: list[dict[str, Any]]


class InvestableUniverseMembership:
    """Single source of truth for downstream product-universe membership checks."""

    def __init__(self, repository: InvestableUniverseRepository) -> None:
        self.repository = repository

    @staticmethod
    def _version_refs(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
        refs = snapshot.get("version_refs")
        if isinstance(refs, list) and refs:
            return copy.deepcopy(refs)
        # Snapshots written by ProductPoolService keep the lineage as parallel
        # ``version_ids``/``pool_ids`` lists; without this the audit trail on a
        # real snapshot would come back empty.
        pool_ids = list(snapshot.get("pool_ids") or [])
        return [
            {
                "version_id": str(version_id),
                "pool_id": str(pool_ids[index]) if index < len(pool_ids) else "",
            }
            for index, version_id in enumerate(snapshot.get("version_ids") or [])
        ]

    @classmethod
    def _reference(cls, snapshot: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": snapshot["id"],
            "name": snapshot.get("name"),
            "research_date": snapshot.get("research_date"),
            "content_hash": snapshot.get("content_hash"),
            "version_refs": cls._version_refs(snapshot),
            "created_at": snapshot.get("created_at"),
            "immutable": bool(snapshot.get("immutable")),
        }

    @staticmethod
    def _members(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
        """Read both stored snapshot shapes.

        ``ProductPoolService`` persists a snapshot as ``products`` and maps it to
        ``members`` only in its read API, so validating against the stored record
        has to do the same mapping -- otherwise every product in a real snapshot
        looks absent from its own universe.
        """

        raw = snapshot.get("members")
        if isinstance(raw, list) and raw:
            return [item for item in raw if isinstance(item, dict)]
        on_date = parse_date(
            snapshot.get("research_date"),
            field="research_date",
            required=False,
        ) or today()
        members: list[dict[str, Any]] = []
        for product in snapshot.get("products") or []:
            if not isinstance(product, dict):
                continue
            member = {
                "kind": product.get("kind"),
                "product_id": product.get("product_id"),
                "code": product.get("code") or product.get("product_id"),
                "name": product.get("name") or product.get("code") or product.get("product_id"),
                # A product only reaches a universe snapshot once its pool
                # approved it; the snapshot no longer carries the pool status.
                "research_status": "approved",
                "usage_status": product.get("usage_status") or "normal",
                "max_weight": product.get("max_weight"),
                "valid_until": product.get("valid_until"),
                "decision_reasons": list(product.get("reasons") or []),
            }
            eligible, reasons = member_eligibility(member, on_date)
            member["eligible"] = eligible
            member["eligibility_reasons"] = reasons
            members.append(member)
        return members

    @staticmethod
    def _base_product_id(product_id: str) -> str:
        return product_id.split(".", 1)[0].upper()

    @classmethod
    def _aliases(
        cls,
        members: list[dict[str, Any]],
    ) -> tuple[
        dict[tuple[str, str], dict[str, Any]],
        dict[tuple[str, str], list[dict[str, Any]]],
        dict[str, list[dict[str, Any]]],
    ]:
        exact: dict[tuple[str, str], dict[str, Any]] = {}
        by_kind_base: dict[tuple[str, str], list[dict[str, Any]]] = {}
        by_base: dict[str, list[dict[str, Any]]] = {}
        for member in members:
            key = product_key(member.get("kind"), member.get("product_id"))
            exact[key] = member
            base = cls._base_product_id(key[1])
            by_kind_base.setdefault((key[0], base), []).append(member)
            by_base.setdefault(base, []).append(member)
        return exact, by_kind_base, by_base

    @classmethod
    def _resolve_member(
        cls,
        *,
        kind: str | None,
        product_id: str,
        exact: dict[tuple[str, str], dict[str, Any]],
        by_kind_base: dict[tuple[str, str], list[dict[str, Any]]],
        by_base: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any] | None:
        normalized_id = str(product_id or "").strip()
        normalized_kind = str(kind or "").strip().lower()
        if normalized_kind:
            member = exact.get(product_key(normalized_kind, normalized_id))
            if member is not None:
                return member
            candidates = by_kind_base.get(
                (normalized_kind, cls._base_product_id(normalized_id)),
                [],
            )
            return candidates[0] if len(candidates) == 1 else None
        candidates = by_base.get(cls._base_product_id(normalized_id), [])
        return candidates[0] if len(candidates) == 1 else None

    def validate(
        self,
        snapshot_id: str,
        products: Iterable[dict[str, Any]],
        *,
        require_eligible: bool = True,
    ) -> UniverseMembershipResult:
        snapshot = self.repository.get(snapshot_id)
        if snapshot.get("immutable") is not True:
            raise ProductPoolValidationError(
                "INVESTABLE_UNIVERSE_NOT_IMMUTABLE",
                "可投资域必须是不可变快照。",
                field="universe_snapshot_id",
            )
        universe_members = self._members(snapshot)
        exact, by_kind_base, by_base = self._aliases(universe_members)
        resolved: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        for product in products:
            requested_kind = str(product.get("kind") or "").strip().lower() or None
            requested_id = str(
                product.get("product_id") or product.get("code") or ""
            ).strip()
            member = self._resolve_member(
                kind=requested_kind,
                product_id=requested_id,
                exact=exact,
                by_kind_base=by_kind_base,
                by_base=by_base,
            )
            if member is None:
                diagnostics.append(
                    {
                        "kind": requested_kind,
                        "product_id": requested_id,
                        "reason": "not_in_universe",
                    }
                )
                continue
            if require_eligible and member.get("eligible") is not True:
                diagnostics.append(
                    {
                        "kind": member.get("kind"),
                        "product_id": member.get("product_id"),
                        "reason": "not_eligible",
                        "details": list(member.get("eligibility_reasons") or []),
                    }
                )
                continue
            resolved.append(copy.deepcopy(member))
        if diagnostics:
            raise ProductPoolValidationError(
                "PRODUCT_OUTSIDE_INVESTABLE_UNIVERSE",
                "存在不在可投资域内或当前不可用的产品。",
                field="products",
                diagnostics=diagnostics[:50],
            )
        return UniverseMembershipResult(
            reference=self._reference(snapshot),
            members=resolved,
        )


__all__ = ["InvestableUniverseMembership", "UniverseMembershipResult"]
