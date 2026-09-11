"""Import locked evaluation-plan results into product-pool drafts."""

from __future__ import annotations

import copy
import math
import uuid
from dataclasses import dataclass
from datetime import date
from typing import Any

from .domain import content_hash, parse_date, product_key, today, trimmed, utc_now
from .errors import ProductPoolConflictError, ProductPoolValidationError


_PRESERVED_MEMBER_FIELDS = {
    "research_status",
    "usage_status",
    "decision_reason",
    "max_weight",
    "valid_until",
    "next_review_date",
    "substitute_group",
    "notes",
    "reviewed_by",
    "reviewed_at",
    "created_at",
}


@dataclass(frozen=True)
class EvaluationImportResult:
    bindings: list[dict[str, Any]]
    members: list[dict[str, Any]]


class EvaluationPlanImporter:
    def __init__(self, indicator_service: Any) -> None:
        self.indicator_service = indicator_service

    def _require_service(self) -> Any:
        if self.indicator_service is None:
            raise ProductPoolConflictError(
                "INDICATOR_SERVICE_UNAVAILABLE",
                "评价方案服务不可用。",
            )
        return self.indicator_service

    @staticmethod
    def _normalize_criteria(fields: dict[str, Any]) -> tuple[str, date, int | None, float | None]:
        plan_id = trimmed(fields.get("plan_id"), field="plan_id", maximum=120, required=True)
        as_of = parse_date(fields.get("as_of"), field="as_of")
        assert as_of is not None
        if as_of > today():
            raise ProductPoolValidationError(
                "FUTURE_AS_OF",
                "评价截止日不能晚于当前日期。",
                field="as_of",
            )
        max_rank_raw = fields.get("max_rank")
        max_rank = int(max_rank_raw) if max_rank_raw is not None else None
        if max_rank is not None and max_rank < 1:
            raise ProductPoolValidationError(
                "INVALID_MAX_RANK",
                "最大排名必须大于 0。",
                field="max_rank",
            )
        min_score_raw = fields.get("min_score")
        min_score = float(min_score_raw) if min_score_raw is not None else None
        if min_score is not None and (
            not math.isfinite(min_score) or not 0 <= min_score <= 100
        ):
            raise ProductPoolValidationError(
                "INVALID_MIN_SCORE",
                "最低得分必须在 0 至 100 之间。",
                field="min_score",
            )
        return plan_id, as_of, max_rank, min_score

    def _collect_plan_rows(
        self,
        result: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        result_id = result.get("result_id")
        if not result_id:
            return result, list(result.get("rows") or [])
        service = self._require_service()
        rows: list[dict[str, Any]] = []
        page = 1
        summary = result
        while True:
            current = service.get_plan_run_result(str(result_id), page=page, page_size=500)
            if page == 1:
                summary = current
            rows.extend(current.get("rows") or [])
            if not (current.get("pagination") or {}).get("has_next"):
                return summary, rows
            page += 1

    @staticmethod
    def _select_rows(
        rows: list[dict[str, Any]],
        max_rank: int | None,
        min_score: float | None,
    ) -> list[dict[str, Any]]:
        selected = [
            row
            for row in rows
            if row.get("status") == "ranked"
            and row.get("rank") is not None
            and (max_rank is None or int(row["rank"]) <= max_rank)
            and (min_score is None or float(row.get("score") or 0.0) >= min_score)
        ]
        if not selected:
            raise ProductPoolValidationError(
                "NO_PRODUCTS_MATCH_IMPORT_RULE",
                "没有产品满足本次排名导入条件。",
                field="max_rank",
            )
        return selected

    @staticmethod
    def _compact_metric_evidence(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "indicator_id": value.get("indicator_id"),
                "indicator_revision": value.get("indicator_revision"),
                "indicator_name": value.get("indicator_name"),
                "period": value.get("period"),
                "value": value.get("value"),
                "status": value.get("status"),
                "normalized_score": value.get("normalized_score"),
                "weighted_contribution": value.get("weighted_contribution"),
                "warning_codes": [
                    warning.get("code")
                    for warning in value.get("warnings") or []
                    if warning.get("code")
                ],
            }
            for value in values
        ]

    @staticmethod
    def _binding_id(pool: dict[str, Any], plan_id: str) -> str:
        existing = next(
            (
                item
                for item in pool.get("evaluation_plans") or []
                if item.get("plan_id") == plan_id
            ),
            None,
        )
        return str(existing.get("binding_id")) if existing else f"pool-plan-{uuid.uuid4().hex}"

    @staticmethod
    def _assert_no_cross_plan_collision(
        pool: dict[str, Any],
        plan_id: str,
        selected_rows: list[dict[str, Any]],
    ) -> None:
        selected_keys = {
            product_key(row.get("target", {}).get("kind"), row.get("target", {}).get("product_id"))
            for row in selected_rows
        }
        collisions = [
            member
            for member in pool.get("members") or []
            if product_key(member.get("kind"), member.get("product_id")) in selected_keys
            and member.get("evaluation_plan_id") != plan_id
        ]
        if collisions:
            raise ProductPoolConflictError(
                "PRODUCT_ALREADY_ASSIGNED_TO_EVALUATION_PLAN",
                "同一产品在一个产品池中只能归属于一个评价方案。",
                field="plan_id",
            )

    def _build_members(
        self,
        pool: dict[str, Any],
        plan: dict[str, Any],
        binding_id: str,
        source_run_id: str,
        selected_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        plan_id = str(plan["id"])
        existing = {
            product_key(member.get("kind"), member.get("product_id")): member
            for member in pool.get("members") or []
            if member.get("evaluation_plan_id") == plan_id
        }
        now = utc_now()
        members: list[dict[str, Any]] = []
        for row in selected_rows:
            target = row["target"]
            key = product_key(target.get("kind"), target.get("product_id"))
            previous = existing.get(key, {})
            manual = {
                field: copy.deepcopy(previous.get(field))
                for field in _PRESERVED_MEMBER_FIELDS
                if field in previous
            }
            members.append(
                {
                    "kind": key[0],
                    "product_id": key[1],
                    "name": str(target.get("name") or key[1]),
                    "evaluation_plan_binding_id": binding_id,
                    "evaluation_plan_id": plan_id,
                    "evaluation_plan_revision": int(plan["revision"]),
                    "source_run_id": source_run_id,
                    "source_rank": int(row["rank"]),
                    "source_score": float(row["score"]),
                    "metric_evidence": self._compact_metric_evidence(row.get("values") or []),
                    "research_status": manual.get("research_status") or "candidate",
                    "usage_status": manual.get("usage_status") or "normal",
                    "decision_reason": manual.get("decision_reason") or "",
                    "max_weight": manual.get("max_weight"),
                    "valid_until": manual.get("valid_until"),
                    "next_review_date": manual.get("next_review_date"),
                    "substitute_group": manual.get("substitute_group") or "",
                    "notes": manual.get("notes") or "",
                    "reviewed_by": manual.get("reviewed_by") or "",
                    "reviewed_at": manual.get("reviewed_at"),
                    "created_at": manual.get("created_at") or now,
                    "updated_at": now,
                }
            )
        return members

    @staticmethod
    def _build_binding(
        plan: dict[str, Any],
        binding_id: str,
        source_run_id: str,
        run_summary: dict[str, Any],
        selected_rows: list[dict[str, Any]],
        *,
        as_of: date,
        max_rank: int | None,
        min_score: float | None,
    ) -> dict[str, Any]:
        execution = run_summary.get("execution") or {}
        source_run = {
            "id": source_run_id,
            "run_at": run_summary.get("run_at"),
            "as_of": as_of.isoformat(),
            "data_generation": execution.get("data_generation"),
            "engine_version": execution.get("engine_version"),
            "normalization": copy.deepcopy(run_summary.get("normalization")),
            "ranked_count": int(run_summary.get("ranked_count") or 0),
            "excluded_count": int(run_summary.get("excluded_count") or 0),
            "imported_count": len(selected_rows),
            "criteria": {"max_rank": max_rank, "min_score": min_score},
            "result_hash": content_hash(
                {
                    "plan_id": plan["id"],
                    "plan_revision": plan["revision"],
                    "as_of": as_of.isoformat(),
                    "rows": selected_rows,
                }
            ),
        }
        return {
            "binding_id": binding_id,
            "plan_id": plan["id"],
            "plan_revision": int(plan["revision"]),
            "plan_name": str(plan.get("name") or plan["id"]),
            "product_kind": plan.get("product_kind"),
            "description": str(plan.get("description") or ""),
            "indicators": copy.deepcopy(plan.get("indicators") or []),
            "source_run": source_run,
            "updated_at": utc_now(),
        }

    def import_into_pool(
        self,
        pool: dict[str, Any],
        fields: dict[str, Any],
    ) -> EvaluationImportResult:
        service = self._require_service()
        plan_id, as_of, max_rank, min_score = self._normalize_criteria(fields)
        plan = service.get_plan(plan_id)
        requested_revision = fields.get("plan_revision")
        if requested_revision is not None and int(requested_revision) != int(plan["revision"]):
            raise ProductPoolConflictError(
                "EVALUATION_PLAN_REVISION_CONFLICT",
                "评价方案已更新，请刷新后重新选择。",
                field="plan_revision",
            )
        run_summary, rows = self._collect_plan_rows(
            service.run_plan(plan_id, as_of.isoformat())
        )
        if int(run_summary.get("plan_revision") or 0) != int(plan["revision"]):
            raise ProductPoolConflictError(
                "EVALUATION_RUN_REVISION_MISMATCH",
                "评价运行结果与所选方案版本不一致。",
            )
        selected_rows = self._select_rows(rows, max_rank, min_score)
        self._assert_no_cross_plan_collision(pool, plan_id, selected_rows)
        binding_id = self._binding_id(pool, plan_id)
        source_run_id = f"pool-run-{uuid.uuid4().hex}"
        imported = self._build_members(
            pool,
            plan,
            binding_id,
            source_run_id,
            selected_rows,
        )
        binding = self._build_binding(
            plan,
            binding_id,
            source_run_id,
            run_summary,
            selected_rows,
            as_of=as_of,
            max_rank=max_rank,
            min_score=min_score,
        )
        bindings = [
            item
            for item in pool.get("evaluation_plans") or []
            if item.get("plan_id") != plan_id
        ] + [binding]
        bindings.sort(
            key=lambda item: (str(item.get("product_kind")), str(item.get("plan_name")))
        )
        members = [
            item
            for item in pool.get("members") or []
            if item.get("evaluation_plan_id") != plan_id
        ] + imported
        members.sort(
            key=lambda item: (
                str(item.get("evaluation_plan_id")),
                int(item.get("source_rank") or 10**9),
                str(item.get("product_id")),
            )
        )
        return EvaluationImportResult(bindings=bindings, members=members)


__all__ = ["EvaluationImportResult", "EvaluationPlanImporter"]
