"""Stable presentation metadata for metric catalogs and calculation results."""

from __future__ import annotations

from typing import Any


INDICATOR_TYPE_LABELS = {
    "return": "收益型指标",
    "risk": "风险型指标",
    "risk_adjusted": "收益风险性价比指标",
    "path": "路径与回撤指标",
    "market_liquidity": "行情与流动性指标",
    "other": "其他指标",
}

CATEGORY_LABELS = INDICATOR_TYPE_LABELS

_LEGACY_CATEGORY_TYPES = {
    "return_statistics": "return",
    "distribution_risk": "risk",
    "risk_path": "path",
    "etf_market": "market_liquidity",
    "portfolio_return": "return",
    "portfolio_risk": "risk",
    "custom": "other",
    "compatibility": "other",
}


def indicator_type(definition: dict[str, Any]) -> str:
    requested = str(
        definition.get("indicator_type")
        or definition.get("category_id")
        or "other"
    )
    normalized = _LEGACY_CATEGORY_TYPES.get(requested, requested)
    return normalized if normalized in INDICATOR_TYPE_LABELS else "other"


def catalog_status(definition: dict[str, Any]) -> str:
    """Classify frozen DSL v1 definitions without hiding custom history."""

    explicit = definition.get("catalog_status_override")
    if explicit in {"current", "compatibility"}:
        return str(explicit)
    return (
        "compatibility"
        if str(definition.get("dsl_version") or "1.0.0") == "1.0.0"
        else "current"
    )


def ui_exposed(definition: dict[str, Any]) -> bool:
    """New selectors hide compatibility built-ins but keep custom definitions visible."""

    if definition.get("ui_exposed_override") is False:
        return False
    return not (
        definition.get("source") == "built_in"
        and catalog_status(definition) == "compatibility"
    )


def metric_presentation(definition: dict[str, Any]) -> dict[str, Any]:
    """Build the immutable display contract embedded in every result.

    The contract is derived from the exact definition revision used for the
    calculation, so clients never need to join a result with today's catalog.
    """

    status = catalog_status(definition)
    output_measure = str(definition.get("output_measure") or "dimensionless")
    display_format = str(definition.get("display_format") or "number")
    category = indicator_type(definition)
    category_label = str(
        definition.get("category_label")
        or CATEGORY_LABELS.get(category)
        or category
    )
    notation = (
        "compact"
        if output_measure in {"volume", "currency_amount", "count"}
        else "standard"
    )
    unit = str(definition.get("unit") or "")
    if not unit and output_measure == "volume":
        unit = "份"
    elif not unit and output_measure == "currency_amount":
        unit = "元"
    elif not unit and output_measure == "raw_market_price":
        unit = "元"

    return {
        "indicator_id": definition.get("id"),
        "revision": definition.get("revision"),
        "name": str(definition.get("name") or "未命名指标"),
        "source": str(definition.get("source") or "inline"),
        "indicator_type": category,
        "category": category,
        "category_label": category_label,
        "context_kind": str(definition.get("context_kind") or "single_product"),
        "catalog_status": status,
        "display_format": display_format,
        "precision": int(definition.get("precision", 2)),
        "unit": unit,
        "notation": notation,
        "value_scale": 100.0 if display_format == "percent" else 1.0,
        "output_measure": output_measure,
        "direction": str(definition.get("direction") or "higher_better"),
        "description": str(definition.get("description") or ""),
        "methodology": str(
            definition.get("methodology")
            or definition.get("description")
            or ""
        ),
        "data_basis": str(
            definition.get("data_basis")
            or (
                "legacy v1 锁定计算口径"
                if status == "compatibility"
                else "真实数据、严格窗口、缺失不填充"
            )
        ),
        "minimum_observations": int(definition.get("minimum_observations", 1)),
        "applicable_product_kinds": list(
            definition.get("applicable_product_kinds") or ["etf", "fund"]
        ),
    }
