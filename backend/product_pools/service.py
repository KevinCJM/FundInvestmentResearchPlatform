"""Application service for product-pool research, publication, and selection."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from datetime import date
from typing import Any, Protocol

from .errors import ProductPoolValidationError
from .repository import ProductPoolRepository, utc_now

RESEARCH_STATUSES = {"pending", "approved", "watch", "rejected"}
USAGE_STATUSES = {"normal", "limited", "no_new", "hold_only", "unavailable"}
SELECTION_MODES = {"all_ranked", "top_n", "top_percent"}
INVESTABLE_USAGE_STATUSES = {"normal", "limited"}


class EvaluationPlanGateway(Protocol):
    """Narrow dependency boundary to the existing evaluation-plan subsystem."""

    def get_plan(self, plan_id: str) -> dict[str, Any]: ...

    def run_plan(self, plan_id: str, as_of: str | None = None) -> dict[str, Any]: ...

    def get_run_page(
        self,
        result_id: str,
        *,
        page: int,
        page_size: int,
    ) -> dict[str, Any]: ...


def _trimmed(value: Any, *, maximum: int = 500) -> str:
    return str(value or "").strip()[:maximum]


def _required_text(value: Any, field: str, *, maximum: int = 100) -> str:
    result = _trimmed(value, maximum=maximum)
    if not result:
        raise ProductPoolValidationError(
            "REQUIRED_FIELD",
            f"{field} 不能为空。",
            field=field,
        )
    return result


def _iso_date(value: Any, field: str, *, required: bool = True) -> str | None:
    text = _trimmed(value, maximum=10)
    if not text:
        if required:
            raise ProductPoolValidationError("INVALID_DATE", f"{field} 不能为空。", field=field)
        return None
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise ProductPoolValidationError(
            "INVALID_DATE",
            f"{field} 必须为 YYYY-MM-DD。",
            field=field,
        ) from exc


def _finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    number = _finite_number(value)
    if number is None or number < 1 or not float(number).is_integer():
        return None
    return int(number)


def _member_key(kind: str, product_id: str) -> str:
    return f"{kind}:{product_id}"


def _evaluation_result_id(
    run: dict[str, Any],
    *,
    plan_id: str,
    plan_revision: int,
    rows: list[dict[str, Any]],
) -> str:
    """Return the evaluation repository id or a stable inline-result fingerprint."""

    persisted_id = _trimmed(run.get("result_id"), maximum=120)
    if persisted_id:
        return persisted_id
    payload = {
        "plan_id": plan_id,
        "plan_revision": plan_revision,
        "run_at": run.get("run_at"),
        "as_of": run.get("as_of"),
        "ranked_count": run.get("ranked_count"),
        "excluded_count": run.get("excluded_count"),
        "rows": rows,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return f"inline-{hashlib.sha256(encoded).hexdigest()}"


def _dedupe_text(values: Any, *, maximum_items: int = 20, maximum_length: int = 300) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = _trimmed(raw, maximum=maximum_length)
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
        if len(result) >= maximum_items:
            break
    return result


class ProductPoolService:
    """Coordinates evaluation evidence; it never recalculates plan scores itself."""

    def __init__(
        self,
        repository: ProductPoolRepository,
        evaluation_gateway: EvaluationPlanGateway,
    ) -> None:
        self.repository = repository
        self.evaluation_gateway = evaluation_gateway

    @staticmethod
    def _normalize_metadata(fields: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": _required_text(fields.get("name"), "产品池名称", maximum=80),
            "description": _trimmed(fields.get("description"), maximum=500),
            "purpose": _trimmed(fields.get("purpose"), maximum=200),
            "owner": _trimmed(fields.get("owner"), maximum=80),
        }

    def list_pools(self) -> dict[str, Any]:
        items = self.repository.list_pools()
        return {"items": items, "total": len(items)}

    def get_pool(self, pool_id: str) -> dict[str, Any]:
        return self.repository.get_pool(pool_id)

    def create_pool(self, fields: dict[str, Any]) -> dict[str, Any]:
        return self.repository.create_pool(
            {
                **self._normalize_metadata(fields),
                "state": "draft",
                "evaluation_plans": [],
                "members": [],
            }
        )

    def update_pool(
        self,
        pool_id: str,
        expected_revision: int,
        fields: dict[str, Any],
    ) -> dict[str, Any]:
        return self.repository.update_pool(
            pool_id,
            expected_revision,
            self._normalize_metadata(fields),
        )

    def archive_pool(self, pool_id: str, expected_revision: int) -> dict[str, Any]:
        return self.repository.archive_pool(pool_id, expected_revision)

    @staticmethod
    def _run_row_target(
        row: dict[str, Any],
        default_kind: str,
    ) -> tuple[str, str, str, str] | None:
        target = row.get("target") if isinstance(row.get("target"), dict) else {}
        kind = _trimmed(
            row.get("kind")
            or row.get("product_kind")
            or target.get("kind")
            or default_kind,
            maximum=20,
        ).lower()
        product_id = _trimmed(
            row.get("product_id")
            or row.get("id")
            or target.get("product_id")
            or target.get("id"),
            maximum=100,
        )
        if kind not in {"etf", "fund"} or not product_id:
            return None
        code = _trimmed(
            row.get("code")
            or row.get("ts_code")
            or target.get("code")
            or product_id,
            maximum=100,
        )
        name = _trimmed(
            row.get("name")
            or row.get("product_name")
            or target.get("name")
            or code,
            maximum=160,
        )
        return kind, product_id, code, name

    @staticmethod
    def _row_rank(row: dict[str, Any]) -> int | None:
        for key in ("rank", "ranking", "rank_no"):
            rank = _integer(row.get(key))
            if rank is not None:
                return rank
        return None

    @staticmethod
    def _row_score(row: dict[str, Any]) -> float | None:
        for key in ("score", "total_score", "composite_score", "normalized_score"):
            score = _finite_number(row.get(key))
            if score is not None:
                return score
        return None

    @staticmethod
    def _row_exclusion_reason(row: dict[str, Any]) -> str | None:
        for key in ("exclusion_reason", "reason", "message"):
            value = _trimmed(row.get(key), maximum=300)
            if value:
                return value
        reasons = row.get("exclusion_reasons")
        if isinstance(reasons, list):
            joined = "；".join(_dedupe_text(reasons, maximum_items=5, maximum_length=100))
            return joined or None
        return None

    def _collect_run_rows(self, run: dict[str, Any]) -> list[dict[str, Any]]:
        rows = [dict(item) for item in run.get("rows", []) if isinstance(item, dict)]
        result_id = _trimmed(run.get("result_id"), maximum=120)
        total = _integer(run.get("total"))
        if total is None:
            ranked_count = int(_finite_number(run.get("ranked_count")) or 0)
            excluded_count = int(_finite_number(run.get("excluded_count")) or 0)
            total = ranked_count + excluded_count
        if not result_id or total <= len(rows):
            return rows
        page_size = min(500, max(1, int(_finite_number(run.get("page_size")) or 500)))
        page = 2 if rows else 1
        while len(rows) < total:
            payload = self.evaluation_gateway.get_run_page(
                result_id,
                page=page,
                page_size=page_size,
            )
            page_rows = [dict(item) for item in payload.get("rows", []) if isinstance(item, dict)]
            if not page_rows:
                break
            rows.extend(page_rows)
            page += 1
        return rows[:total]

    @staticmethod
    def _select_ranked_rows(
        rows: list[dict[str, Any]],
        mode: str,
        value: float | None,
    ) -> list[dict[str, Any]]:
        ranked = [row for row in rows if ProductPoolService._row_rank(row) is not None]
        ranked.sort(
            key=lambda row: (
                ProductPoolService._row_rank(row) or 2**31,
                -(ProductPoolService._row_score(row) or float("-inf")),
            )
        )
        if mode == "all_ranked":
            return ranked
        if mode == "top_n":
            count = int(value or 0)
            if count < 1:
                raise ProductPoolValidationError(
                    "INVALID_SELECTION_VALUE",
                    "Top N 必须大于 0。",
                    field="selection_value",
                )
            return ranked[:count]
        percent = float(value or 0)
        if not 0 < percent <= 100:
            raise ProductPoolValidationError(
                "INVALID_SELECTION_VALUE",
                "Top 百分比必须在 0 至 100 之间。",
                field="selection_value",
            )
        count = max(1, math.ceil(len(ranked) * percent / 100)) if ranked else 0
        return ranked[:count]

    @staticmethod
    def _binding_by_id(pool: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {
            str(item.get("plan_id")): item
            for item in pool.get("evaluation_plans", [])
            if isinstance(item, dict) and item.get("plan_id")
        }

    def attach_evaluation_plan(
        self,
        pool_id: str,
        expected_revision: int,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        pool = self.repository.get_pool(pool_id)
        if int(pool["revision"]) != expected_revision:
            return self.repository.update_pool(pool_id, expected_revision, {})
        plan_id = _required_text(request.get("plan_id"), "评价方案", maximum=120)
        selection_mode = _trimmed(request.get("selection_mode") or "all_ranked", maximum=30)
        if selection_mode not in SELECTION_MODES:
            raise ProductPoolValidationError(
                "INVALID_SELECTION_MODE",
                "评价方案入池方式无效。",
                field="selection_mode",
            )
        selection_value = _finite_number(request.get("selection_value"))
        as_of = _iso_date(request.get("as_of"), "评价截止日", required=False)
        plan = self.evaluation_gateway.get_plan(plan_id)
        run = self.evaluation_gateway.run_plan(plan_id, as_of)
        rows = self._collect_run_rows(run)
        selected_rows = self._select_ranked_rows(rows, selection_mode, selection_value)
        plan_revision = int(plan.get("revision") or 1)
        plan_name = _required_text(plan.get("name"), "评价方案名称", maximum=120)
        product_kind = _trimmed(plan.get("product_kind"), maximum=20).lower()
        if product_kind not in {"etf", "fund"}:
            raise ProductPoolValidationError(
                "INVALID_EVALUATION_PLAN",
                "评价方案缺少有效产品类型。",
                field="plan_id",
            )
        result_id = _evaluation_result_id(
            run,
            plan_id=plan_id,
            plan_revision=plan_revision,
            rows=rows,
        )
        binding = {
            "plan_id": plan_id,
            "plan_revision": plan_revision,
            "plan_name": plan_name,
            "product_kind": product_kind,
            "result_id": result_id,
            "as_of": run.get("as_of") or as_of,
            "data_generation": run.get("data_generation"),
            "selection_mode": selection_mode,
            "selection_value": selection_value,
            "ranked_count": int(_finite_number(run.get("ranked_count")) or len(selected_rows)),
            "excluded_count": int(_finite_number(run.get("excluded_count")) or 0),
            "imported_count": len(selected_rows),
            "attached_at": utc_now(),
        }

        plans = [
            copy.deepcopy(item)
            for item in pool.get("evaluation_plans", [])
            if item.get("plan_id") != plan_id
        ]
        plans.append(binding)

        members_by_key: dict[str, dict[str, Any]] = {}
        for raw_member in pool.get("members", []):
            member = copy.deepcopy(raw_member)
            member["evidences"] = [
                evidence
                for evidence in member.get("evidences", [])
                if not (
                    evidence.get("source") == "evaluation_plan"
                    and evidence.get("plan_id") == plan_id
                )
            ]
            if member.get("primary_plan_id") == plan_id:
                member["primary_plan_id"] = None
            if member["evidences"] or member.get("manual_exception"):
                members_by_key[str(member["key"])] = member

        for row in selected_rows:
            target = self._run_row_target(row, product_kind)
            if target is None:
                continue
            kind, product_id, code, name = target
            key = _member_key(kind, product_id)
            member = members_by_key.get(key)
            if member is None:
                member = {
                    "key": key,
                    "kind": kind,
                    "product_id": product_id,
                    "code": code,
                    "name": name,
                    "research_status": "pending",
                    "usage_status": "normal",
                    "primary_plan_id": plan_id,
                    "max_weight": None,
                    "reasons": [],
                    "owner": "",
                    "review_due_date": None,
                    "valid_until": None,
                    "substitute_group": "",
                    "manual_exception": False,
                    "evidences": [],
                }
                members_by_key[key] = member
            if not member.get("primary_plan_id"):
                member["primary_plan_id"] = plan_id
            evidence = {
                "source": "evaluation_plan",
                "plan_id": plan_id,
                "plan_revision": plan_revision,
                "plan_name": plan_name,
                "result_id": result_id,
                "as_of": binding["as_of"],
                "rank": self._row_rank(row),
                "score": self._row_score(row),
                "percentile": _finite_number(row.get("percentile") or row.get("rank_percentile")),
                "result_status": _trimmed(row.get("status"), maximum=40) or "ranked",
                "exclusion_reason": self._row_exclusion_reason(row),
            }
            member.setdefault("evidences", []).append(evidence)

        valid_plan_ids = {item["plan_id"] for item in plans}
        members: list[dict[str, Any]] = []
        for member in members_by_key.values():
            available = [
                evidence.get("plan_id")
                for evidence in member.get("evidences", [])
                if evidence.get("plan_id") in valid_plan_ids
            ]
            if member.get("primary_plan_id") not in available:
                member["primary_plan_id"] = available[0] if available else None
            members.append(member)
        members.sort(key=lambda item: (str(item.get("primary_plan_id")), str(item.get("code"))))
        return self.repository.update_pool(
            pool_id,
            expected_revision,
            {"evaluation_plans": plans, "members": members, "state": "draft"},
        )

    def remove_evaluation_plan(
        self,
        pool_id: str,
        expected_revision: int,
        plan_id: str,
    ) -> dict[str, Any]:
        pool = self.repository.get_pool(pool_id)
        plans = [
            copy.deepcopy(item)
            for item in pool.get("evaluation_plans", [])
            if item.get("plan_id") != plan_id
        ]
        if len(plans) == len(pool.get("evaluation_plans", [])):
            raise ProductPoolValidationError(
                "EVALUATION_PLAN_NOT_ATTACHED",
                "该评价方案未关联到当前产品池。",
                field="plan_id",
            )
        valid_plan_ids = {item["plan_id"] for item in plans}
        members: list[dict[str, Any]] = []
        for raw_member in pool.get("members", []):
            member = copy.deepcopy(raw_member)
            member["evidences"] = [
                evidence
                for evidence in member.get("evidences", [])
                if evidence.get("plan_id") != plan_id
            ]
            member["manual_exception"] = any(
                evidence.get("source") == "manual_exception"
                for evidence in member["evidences"]
            )
            if not member["evidences"]:
                continue
            available = [
                evidence.get("plan_id")
                for evidence in member["evidences"]
                if evidence.get("plan_id") in valid_plan_ids
            ]
            if member.get("primary_plan_id") not in available:
                member["primary_plan_id"] = available[0] if available else None
            members.append(member)
        return self.repository.update_pool(
            pool_id,
            expected_revision,
            {"evaluation_plans": plans, "members": members, "state": "draft"},
        )

    def add_manual_member(
        self,
        pool_id: str,
        expected_revision: int,
        fields: dict[str, Any],
    ) -> dict[str, Any]:
        pool = self.repository.get_pool(pool_id)
        plan_id = _required_text(fields.get("plan_id"), "所属评价方案", maximum=120)
        binding = self._binding_by_id(pool).get(plan_id)
        if binding is None:
            raise ProductPoolValidationError(
                "EVALUATION_PLAN_NOT_ATTACHED",
                "人工例外产品必须归属于产品池中的评价方案。",
                field="plan_id",
            )
        kind = _trimmed(fields.get("kind") or binding.get("product_kind"), maximum=20).lower()
        if kind not in {"etf", "fund"}:
            raise ProductPoolValidationError("INVALID_PRODUCT_KIND", "产品类型无效。", field="kind")
        product_id = _required_text(fields.get("product_id"), "产品代码", maximum=100)
        reason = _required_text(fields.get("reason"), "人工例外原因", maximum=300)
        key = _member_key(kind, product_id)
        members = [copy.deepcopy(item) for item in pool.get("members", [])]
        if any(item.get("key") == key for item in members):
            raise ProductPoolValidationError(
                "PRODUCT_ALREADY_IN_POOL",
                "该产品已经在产品池候选列表中。",
                field="product_id",
            )
        members.append(
            {
                "key": key,
                "kind": kind,
                "product_id": product_id,
                "code": _trimmed(fields.get("code") or product_id, maximum=100),
                "name": _trimmed(fields.get("name") or product_id, maximum=160),
                "research_status": "pending",
                "usage_status": "normal",
                "primary_plan_id": plan_id,
                "max_weight": None,
                "reasons": [reason],
                "owner": "",
                "review_due_date": None,
                "valid_until": None,
                "substitute_group": "",
                "manual_exception": True,
                "evidences": [
                    {
                        "source": "manual_exception",
                        "plan_id": plan_id,
                        "plan_revision": binding["plan_revision"],
                        "plan_name": binding["plan_name"],
                        "result_id": None,
                        "as_of": binding.get("as_of"),
                        "rank": None,
                        "score": None,
                        "percentile": None,
                        "result_status": "manual_exception",
                        "exclusion_reason": reason,
                    }
                ],
            }
        )
        return self.repository.update_pool(
            pool_id,
            expected_revision,
            {"members": members, "state": "draft"},
        )

    @staticmethod
    def _apply_member_update(
        member: dict[str, Any],
        fields: dict[str, Any],
        plans: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        research_status = _trimmed(fields.get("research_status"), maximum=30)
        usage_status = _trimmed(fields.get("usage_status"), maximum=30)
        if research_status not in RESEARCH_STATUSES:
            raise ProductPoolValidationError(
                "INVALID_RESEARCH_STATUS",
                "产品研究状态无效。",
                field="research_status",
            )
        if usage_status not in USAGE_STATUSES:
            raise ProductPoolValidationError(
                "INVALID_USAGE_STATUS",
                "产品使用状态无效。",
                field="usage_status",
            )
        primary_plan_id = _required_text(
            fields.get("primary_plan_id"),
            "所属评价方案",
            maximum=120,
        )
        if primary_plan_id not in plans:
            raise ProductPoolValidationError(
                "INVALID_PRIMARY_PLAN",
                "产品所属评价方案不在当前产品池中。",
                field="primary_plan_id",
            )
        evidence_plan_ids = {
            evidence.get("plan_id") for evidence in member.get("evidences", [])
        }
        if primary_plan_id not in evidence_plan_ids:
            raise ProductPoolValidationError(
                "INVALID_PRIMARY_PLAN",
                "该产品没有来自所选评价方案的证据。",
                field="primary_plan_id",
            )
        max_weight = _finite_number(fields.get("max_weight"))
        if max_weight is not None and not 0 < max_weight <= 1:
            raise ProductPoolValidationError(
                "INVALID_MAX_WEIGHT",
                "产品最大权重必须大于 0 且不超过 1。",
                field="max_weight",
            )
        reasons = _dedupe_text(fields.get("reasons"))
        if (
            research_status in {"watch", "rejected"}
            or usage_status != "normal"
        ) and not reasons:
            raise ProductPoolValidationError(
                "REVIEW_REASON_REQUIRED",
                "观察、排除或受限状态必须填写原因。",
                field="reasons",
            )
        updated = copy.deepcopy(member)
        updated.update(
            {
                "research_status": research_status,
                "usage_status": usage_status,
                "primary_plan_id": primary_plan_id,
                "max_weight": max_weight,
                "reasons": reasons,
                "owner": _trimmed(fields.get("owner"), maximum=80),
                "review_due_date": _iso_date(
                    fields.get("review_due_date"),
                    "复审日期",
                    required=False,
                ),
                "valid_until": _iso_date(
                    fields.get("valid_until"),
                    "有效截止日",
                    required=False,
                ),
                "substitute_group": _trimmed(
                    fields.get("substitute_group"),
                    maximum=80,
                ),
                "reviewed_at": utc_now(),
            }
        )
        return updated

    def update_members(
        self,
        pool_id: str,
        expected_revision: int,
        updates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not updates:
            raise ProductPoolValidationError(
                "EMPTY_MEMBER_BATCH",
                "批量保存至少需要一个产品。",
                field="items",
            )
        if len(updates) > 1000:
            raise ProductPoolValidationError(
                "MEMBER_BATCH_TOO_LARGE",
                "单次最多保存 1000 个产品。",
                field="items",
            )
        pool = self.repository.get_pool(pool_id)
        plans = self._binding_by_id(pool)
        members = [copy.deepcopy(item) for item in pool.get("members", [])]
        positions = {
            str(member.get("key")): position
            for position, member in enumerate(members)
        }
        seen: set[str] = set()
        for update in updates:
            kind = _trimmed(update.get("kind"), maximum=20).lower()
            if kind not in {"etf", "fund"}:
                raise ProductPoolValidationError(
                    "INVALID_PRODUCT_KIND",
                    "产品类型无效。",
                    field="kind",
                )
            product_id = _required_text(
                update.get("product_id"),
                "产品代码",
                maximum=100,
            )
            key = _member_key(kind, product_id)
            if key in seen:
                raise ProductPoolValidationError(
                    "DUPLICATE_MEMBER_BATCH_ITEM",
                    "批量保存中存在重复产品。",
                    field="items",
                )
            seen.add(key)
            position = positions.get(key)
            if position is None:
                raise ProductPoolValidationError(
                    "PRODUCT_POOL_MEMBER_NOT_FOUND",
                    "未找到指定产品池成员。",
                    field="product_id",
                )
            members[position] = self._apply_member_update(
                members[position],
                update,
                plans,
            )
        return self.repository.update_pool(
            pool_id,
            expected_revision,
            {"members": members, "state": "draft"},
        )

    def update_member(
        self,
        pool_id: str,
        expected_revision: int,
        kind: str,
        product_id: str,
        fields: dict[str, Any],
    ) -> dict[str, Any]:
        return self.update_members(
            pool_id,
            expected_revision,
            [{**fields, "kind": kind, "product_id": product_id}],
        )

    def publish_pool(
        self,
        pool_id: str,
        expected_revision: int,
        fields: dict[str, Any],
    ) -> dict[str, Any]:
        pool = self.repository.get_pool(pool_id)
        if int(pool["revision"]) != expected_revision:
            return self.repository.update_pool(pool_id, expected_revision, {})
        plans = [copy.deepcopy(item) for item in pool.get("evaluation_plans", [])]
        members = [copy.deepcopy(item) for item in pool.get("members", [])]
        if not plans:
            raise ProductPoolValidationError(
                "PRODUCT_POOL_PLAN_REQUIRED",
                "发布前至少需要关联一个评价方案。",
            )
        pending = [item for item in members if item.get("research_status") == "pending"]
        if pending:
            raise ProductPoolValidationError(
                "PRODUCT_POOL_REVIEW_INCOMPLETE",
                f"仍有 {len(pending)} 个产品未完成人工复核。",
            )
        plan_ids = {item["plan_id"] for item in plans}
        for member in members:
            if member.get("research_status") == "approved" and member.get("primary_plan_id") not in plan_ids:
                raise ProductPoolValidationError(
                    "PRODUCT_POOL_GROUP_MISSING",
                    f"产品 {member.get('code') or member.get('product_id')} 缺少有效评价方案归属。",
                )
        investable = [
            item
            for item in members
            if item.get("research_status") == "approved"
            and item.get("usage_status") in INVESTABLE_USAGE_STATUSES
        ]
        if not investable:
            raise ProductPoolValidationError(
                "PRODUCT_POOL_EMPTY",
                "产品池没有可供投前研究使用的已准入产品。",
            )
        effective_from = _iso_date(fields.get("effective_from"), "生效日")
        effective_to = _iso_date(fields.get("effective_to"), "失效日", required=False)
        if effective_to and effective_from and effective_to < effective_from:
            raise ProductPoolValidationError(
                "INVALID_EFFECTIVE_RANGE",
                "失效日不能早于生效日。",
                field="effective_to",
            )
        counts = {
            status: sum(item.get("research_status") == status for item in members)
            for status in sorted(RESEARCH_STATUSES)
        }
        current, version = self.repository.publish_pool(
            pool_id,
            expected_revision,
            {
                "pool_name": pool["name"],
                "description": pool.get("description", ""),
                "purpose": pool.get("purpose", ""),
                "owner": pool.get("owner", ""),
                "effective_from": effective_from,
                "effective_to": effective_to,
                "publication_note": _trimmed(fields.get("publication_note"), maximum=500),
                "evaluation_plans": plans,
                "members": members,
                "member_counts": counts,
                "investable_count": len(investable),
            },
        )
        return {"pool": current, "version": version}

    def list_versions(
        self,
        *,
        pool_id: str | None = None,
        active_on: str | None = None,
    ) -> dict[str, Any]:
        normalized_active_on = _iso_date(active_on, "有效日期", required=False) if active_on else None
        items = self.repository.list_versions(pool_id=pool_id, active_on=normalized_active_on)
        return {"items": items, "total": len(items)}

    def get_version(self, version_id: str) -> dict[str, Any]:
        return self.repository.get_version(version_id)

    def diff_versions(self, version_id: str, against_id: str) -> dict[str, Any]:
        current = self.repository.get_version(version_id)
        against = self.repository.get_version(against_id)
        current_members = {item["key"]: item for item in current.get("members", [])}
        against_members = {item["key"]: item for item in against.get("members", [])}
        added_keys = sorted(current_members.keys() - against_members.keys())
        removed_keys = sorted(against_members.keys() - current_members.keys())
        changed: list[dict[str, Any]] = []
        tracked_fields = (
            "research_status",
            "usage_status",
            "primary_plan_id",
            "max_weight",
            "valid_until",
            "substitute_group",
        )
        for key in sorted(current_members.keys() & against_members.keys()):
            before = against_members[key]
            after = current_members[key]
            changes = {
                field: {"before": before.get(field), "after": after.get(field)}
                for field in tracked_fields
                if before.get(field) != after.get(field)
            }
            if changes:
                changed.append({"key": key, "name": after.get("name"), "changes": changes})
        return {
            "version_id": version_id,
            "against_id": against_id,
            "added": [current_members[key] for key in added_keys],
            "removed": [against_members[key] for key in removed_keys],
            "changed": changed,
        }

    def create_universe_snapshot(self, fields: dict[str, Any]) -> dict[str, Any]:
        name = _required_text(fields.get("name"), "可投资域名称", maximum=100)
        research_date = _iso_date(fields.get("research_date"), "研究日期")
        raw_version_ids = fields.get("version_ids")
        if not isinstance(raw_version_ids, list):
            raw_version_ids = []
        version_ids = list(dict.fromkeys(_trimmed(value, maximum=120) for value in raw_version_ids if _trimmed(value, maximum=120)))
        if not version_ids:
            raise ProductPoolValidationError(
                "PRODUCT_POOL_VERSION_REQUIRED",
                "请至少选择一个产品池版本。",
                field="version_ids",
            )
        excluded_keys = {
            _trimmed(value, maximum=220)
            for value in (fields.get("excluded_product_keys") or [])
            if _trimmed(value, maximum=220)
        }
        versions = [self.repository.get_version(version_id) for version_id in version_ids]
        for version in versions:
            starts = str(version.get("effective_from") or "")
            ends = str(version.get("effective_to") or "")
            if not starts or starts > research_date or (ends and ends < research_date):
                raise ProductPoolValidationError(
                    "PRODUCT_POOL_VERSION_NOT_ACTIVE",
                    f"产品池版本 {version['id']} 在研究日期 {research_date} 未生效。",
                    field="version_ids",
                )

        products: dict[str, dict[str, Any]] = {}
        for version in versions:
            bindings = {
                item["plan_id"]: item
                for item in version.get("evaluation_plans", [])
                if item.get("plan_id")
            }
            for raw_member in version.get("members", []):
                if raw_member.get("research_status") != "approved":
                    continue
                if raw_member.get("usage_status") not in INVESTABLE_USAGE_STATUSES:
                    continue
                key = str(raw_member.get("key") or "")
                if not key or key in excluded_keys:
                    continue
                primary_plan_id = raw_member.get("primary_plan_id")
                binding = bindings.get(primary_plan_id)
                if binding is None:
                    raise ProductPoolValidationError(
                        "PRODUCT_GROUP_REFERENCE_BROKEN",
                        f"产品 {raw_member.get('code') or key} 的评价方案引用无效。",
                    )
                assignment = (
                    str(binding["plan_id"]),
                    int(binding["plan_revision"]),
                    str(binding["plan_name"]),
                )
                existing = products.get(key)
                if existing is not None:
                    existing_assignment = (
                        existing["evaluation_plan_id"],
                        existing["evaluation_plan_revision"],
                        existing["evaluation_plan_name"],
                    )
                    if existing_assignment != assignment:
                        raise ProductPoolValidationError(
                            "PRODUCT_GROUP_CONFLICT",
                            f"产品 {raw_member.get('code') or key} 在所选产品池版本中归属于不同评价方案。",
                            field="version_ids",
                        )
                    existing["source_version_ids"].append(version["id"])
                    existing["source_pool_ids"].append(version["pool_id"])
                    existing["usage_status"] = (
                        "limited"
                        if "limited" in {existing["usage_status"], raw_member.get("usage_status")}
                        else "normal"
                    )
                    candidate_limit = _finite_number(raw_member.get("max_weight"))
                    if candidate_limit is not None:
                        current_limit = _finite_number(existing.get("max_weight"))
                        existing["max_weight"] = candidate_limit if current_limit is None else min(current_limit, candidate_limit)
                    existing["reasons"] = _dedupe_text(existing["reasons"] + list(raw_member.get("reasons", [])))
                    continue
                products[key] = {
                    "key": key,
                    "kind": raw_member["kind"],
                    "product_id": raw_member["product_id"],
                    "code": raw_member.get("code"),
                    "name": raw_member.get("name"),
                    "evaluation_plan_id": assignment[0],
                    "evaluation_plan_revision": assignment[1],
                    "evaluation_plan_name": assignment[2],
                    "usage_status": raw_member.get("usage_status"),
                    "max_weight": raw_member.get("max_weight"),
                    "valid_until": raw_member.get("valid_until"),
                    "substitute_group": raw_member.get("substitute_group", ""),
                    "reasons": list(raw_member.get("reasons", [])),
                    "source_version_ids": [version["id"]],
                    "source_pool_ids": [version["pool_id"]],
                }

        if not products:
            raise ProductPoolValidationError(
                "INVESTABLE_UNIVERSE_EMPTY",
                "所选产品池版本没有可用产品。",
            )
        grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
        for product in products.values():
            group_key = (
                product["evaluation_plan_id"],
                product["evaluation_plan_revision"],
                product["evaluation_plan_name"],
            )
            grouped.setdefault(group_key, []).append(copy.deepcopy(product))
        groups = [
            {
                "evaluation_plan_id": plan_id,
                "evaluation_plan_revision": revision,
                "evaluation_plan_name": plan_name,
                "products": sorted(items, key=lambda item: str(item.get("code") or "")),
                "product_count": len(items),
            }
            for (plan_id, revision, plan_name), items in grouped.items()
        ]
        groups.sort(key=lambda item: item["evaluation_plan_name"])
        product_items = sorted(
            products.values(),
            key=lambda item: (item["evaluation_plan_name"], str(item.get("code") or "")),
        )
        return self.repository.create_universe_snapshot(
            {
                "name": name,
                "research_date": research_date,
                "version_ids": version_ids,
                "pool_ids": list(dict.fromkeys(version["pool_id"] for version in versions)),
                "excluded_product_keys": sorted(excluded_keys),
                "groups": groups,
                "products": product_items,
                "product_count": len(product_items),
            }
        )

    def get_universe_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        return self.repository.get_universe_snapshot(snapshot_id)
