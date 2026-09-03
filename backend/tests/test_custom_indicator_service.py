from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from custom_indicators.errors import ConflictError, ValidationError
from custom_indicators.service import CustomIndicatorService


def _write_market_data(data_dir: Path, periods: int = 70) -> None:
    dates = pd.bdate_range("2026-01-02", periods=periods)
    pd.DataFrame(
        [{"ts_code": "510050.SH", "code": "510050", "name": "上证50ETF"}]
    ).to_parquet(data_dir / "etf_info_df.parquet", index=False)
    pd.DataFrame(
        [{"ts_code": "000001.OF", "code": "000001", "name": "华夏成长"}]
    ).to_parquet(data_dir / "fund_info_df.parquet", index=False)
    etf_rows = [
        {
            "ts_code": "510050.SH",
            "name": "上证50ETF",
            "date": date,
            "adj_nav": 1.0 + index * 0.01,
        }
        for index, date in enumerate(dates)
    ]
    # The provider must keep the last value for duplicate dates.
    etf_rows.append({**etf_rows[-1], "adj_nav": etf_rows[-1]["adj_nav"] + 0.01})
    pd.DataFrame(etf_rows).to_parquet(data_dir / "etf_daily_df.parquet", index=False)
    pd.DataFrame(
        [
            {
                "ts_code": "000001.OF",
                "name": "华夏成长",
                "date": date,
                "adj_nav": 2.0 + index * 0.005,
            }
            for index, date in enumerate(dates)
        ]
    ).to_parquet(data_dir / "fund_nav_df.parquet", index=False)


def _draft(**overrides):
    payload = {
        "name": "自定义累计收益",
        "description": "测试指标",
        "expression": r"\left(\prod\left(\mathbf{r}+1\right)\right)-1",
        "periods": ["1W", "1M"],
        "unit": "%",
        "display_format": "percent",
        "precision": 2,
        "direction": "higher_better",
        "indicator_type": "return",
        "annual_risk_free_rate_percent": 1.5,
    }
    payload.update(overrides)
    return payload


def test_meta_operator_templates_round_trip_through_validator(tmp_path: Path) -> None:
    service = CustomIndicatorService(tmp_path, tmp_path)

    meta = service.meta()

    assert {item["id"] for item in meta["indicator_types"]} == {
        "return",
        "risk",
        "risk_adjusted",
        "path",
        "market_liquidity",
        "other",
    }
    assert meta["numeric_backend"]["policy"][
        "one_dimensional_reductions_and_scans"
    ] == "numba_njit_fixed_signature"
    assert meta["math_notation_version"] == "1.2.0"

    operator_names = {item["name"] for item in meta["operators"]}
    assert {"add", "multiply", "product", "dot", "matmul"}.issubset(operator_names)
    assert "cumulative_return" not in operator_names
    assert meta["predefined_calculations"] == []
    assert meta["templates"] == []
    assert meta["template_composition_deprecated"] is True
    assert "formula_fragments" not in meta
    assert "indicator_templates" not in meta
    operators = {item["name"]: item for item in meta["operators"]}
    assert operators["mean"]["label"] == "全元素算术平均值"
    assert operators["mean"]["execution_backend"] == "numba_njit_fixed_signature"
    assert operators["matmul"]["execution_backend"] == "numba_njit_fixed_signature"
    assert operators["matmul"]["execution_lane"] == "numba_blas"
    assert operators["clip"]["label"] == "逐元素数值限幅"
    assert operators["drawdown_series"]["label"] == "回撤序列"
    assert operators["drawdown_series"]["category_label"] == "序列与路径"
    assert operators["drawdown_series"]["execution_backend"] == (
        "numba_njit_fixed_signature"
    )
    assert operators["new_high_mask"]["label"] == "严格创新高判断"
    assert operators["new_high_mask"]["parameters"][0]["label"] == (
        "净值或价格序列"
    )
    assert "不改变计算周期或样本窗口" in operators["clip"][
        "mathematical_essence"
    ]
    assert operators["mean_time"]["label"] == "时间轴算术平均值"
    assert operators["mean_asset"]["label"] == "资产轴算术平均值"
    assert operators["mean"]["parameters"][0]["label"] == "输入值"
    assert operators["mean"]["parameters"][0]["allowed_shapes"] == [
        "series",
        "vector",
        "matrix",
    ]
    assert operators["absolute"]["output_shape"] == "unknown"
    assert operators["absolute"]["return_type"].startswith("same(")
    assert operators["covariance"]["output_shape"] == "unknown"
    assert operators["diag"]["output_shape"] == "unknown"
    assert operators["dot"]["parameters"][1]["allowed_shapes"] == [
        "series",
        "vector",
    ]
    assert operators["mean"]["domains"] == ["single_product", "portfolio"]
    assert operators["mean_asset"]["domains"] == ["portfolio"]
    assert operators["trace"]["domains"] == ["portfolio"]
    assert all("收益" not in item["label"] for item in meta["operators"])
    assert all("收益" not in item["label"] for item in meta["legacy_operators"])
    assert "cumulative_return" not in {
        item["name"] for item in meta["legacy_operators"]
    }

    composed = service.compose(
        {
            "template_id": "cumulative-return",
            "context": "single_product",
            "arguments": [{"parameter": "values", "source": "variable", "value": "returns"}],
        }
    )
    result = service.validate(
        {
            "name": "累计收益率",
            "expression": composed["latex"],
            "periods": ["1M"],
            "dsl_version": "2.0.0",
            "context_kind": "single_product",
        }
    )
    assert result["valid"], result["diagnostics"]

    built_ins = service.list_indicators()["items"]
    payoff = next(item for item in built_ins if item["id"] == "builtin-payoff-ratio-v2")
    assert "mean_where" not in payoff["display_latex"]
    assert r"\mathbb{E}" in payoff["display_latex"]
    portfolio_built_ins = [
        item
        for item in built_ins
        if item.get("source") == "built_in"
        and item.get("context_kind") == "portfolio"
    ]
    assert {item["name"] for item in portfolio_built_ins} == {
        "组合累计收益率",
        "组合波动率",
    }
    for definition in portfolio_built_ins:
        validation = service.validate(definition)
        assert validation["valid"], validation["diagnostics"]


def test_portfolio_indicators_use_realized_daily_weight_path(tmp_path: Path) -> None:
    service = CustomIndicatorService(tmp_path, tmp_path)
    asset_returns = np.asarray(
        [[0.10, 0.00], [0.00, 0.10], [0.05, -0.02]], dtype=np.float64
    )
    daily_weights = np.asarray(
        [[0.50, 0.50], [11.0 / 21.0, 10.0 / 21.0], [0.60, 0.40]],
        dtype=np.float64,
    )
    realized_returns = np.sum(asset_returns * daily_weights, axis=1)
    snapshot = {
        "id": "run-dynamic-weights",
        "target_name": "动态权重组合",
        "context_schema": "portfolio-v2",
        "asset_order": ["etf:A", "fund:B"],
        "data_fingerprints": {"etf:A": "one", "fund:B": "two"},
        "common_date_hash": "dates-hash",
        "requested_as_of": None,
        "effective_as_of": "2026-01-06",
        "actual_start_date": "2026-01-02",
        "actual_end_date": "2026-01-06",
        "observation_count": 3,
        "asset_returns": asset_returns,
        "daily_weights": daily_weights,
        "portfolio_returns": realized_returns,
        "benchmark_returns": None,
        "warnings": [],
    }

    response = service.evaluate_portfolio_snapshot(
        [
            "builtin-portfolio-realized-cumulative-return",
            "builtin-portfolio-realized-volatility",
        ],
        snapshot,
    )
    values = {item["indicator_name"]: item["value"] for item in response["results"]}

    assert values["组合累计收益率"] == pytest.approx(
        float(np.prod(1.0 + realized_returns) - 1.0), rel=1e-12, abs=1e-12
    )
    assert values["组合波动率"] == pytest.approx(
        float(np.std(realized_returns, ddof=1)), rel=1e-12, abs=1e-12
    )
    terminal_weight_history = np.prod(1.0 + asset_returns @ daily_weights[-1]) - 1.0
    assert values["组合累计收益率"] != pytest.approx(terminal_weight_history)


def test_portfolio_context_rejects_inconsistent_realized_returns(tmp_path: Path) -> None:
    service = CustomIndicatorService(tmp_path, tmp_path)
    snapshot = {
        "asset_returns": [[0.01, 0.02], [0.03, -0.01]],
        "daily_weights": [[0.60, 0.40], [0.61, 0.39]],
        "portfolio_returns": [0.018, 0.50],
    }

    with pytest.raises(ValidationError, match="组合收益序列与每日生效权重"):
        service._portfolio_context(snapshot, {})


def test_existing_indicator_composition_locks_revision_and_protocol(
    tmp_path: Path,
) -> None:
    service = CustomIndicatorService(tmp_path, tmp_path)
    definition = service.get_indicator("builtin-total-return-v2")

    composed = service.compose(
        {
            "indicator_id": definition["id"],
            "indicator_revision": definition["revision"],
            "context": definition["context_kind"],
            "dsl_version": definition["dsl_version"],
            "operator_registry_version": definition["operator_registry_version"],
            "variable_registry_version": definition["variable_registry_version"],
            "data_contract_version": definition["data_contract_version"],
            "context_schema_version": definition["context_schema_version"],
            "arguments": [],
        }
    )

    assert composed["expression"] == definition["expression"]
    assert composed["shape"] == "scalar"
    assert composed["indicator_origin"] == {
        "indicator_id": definition["id"],
        "indicator_revision": definition["revision"],
        "name": definition["name"],
        "source": "built_in",
    }

    with pytest.raises(ValidationError, match="版本协议不一致") as mismatch:
        service.compose(
            {
                "indicator_id": definition["id"],
                "indicator_revision": definition["revision"],
                "context": definition["context_kind"],
                "dsl_version": definition["dsl_version"],
                "operator_registry_version": "2.0.0",
                "arguments": [],
            }
        )
    assert mismatch.value.code == "INDICATOR_PROTOCOL_MISMATCH"


def test_validation_returns_dependencies_and_safe_dag(tmp_path: Path) -> None:
    service = CustomIndicatorService(tmp_path, tmp_path)

    valid = service.validate(_draft())
    invalid = service.validate({"name": "非法", "expression": "returns.__class__", "periods": ["1M"]})

    assert valid["valid"] is True
    assert valid["dependencies"] == ["returns"]
    assert valid["dag"]["nodes"]
    assert valid["dag"]["edges"]
    assert valid["dag"]["roots"].keys() == {"result"}
    assert invalid["valid"] is False
    assert invalid["diagnostics"][0]["code"] == "INVALID_EXPRESSION"


def test_typed_validation_exposes_node_level_semantic_contract(tmp_path: Path) -> None:
    service = CustomIndicatorService(tmp_path, tmp_path)

    result = service.validate(
        {
            "name": "非法量纲混算",
            "expression": "returns + volume",
            "dsl_version": "2.1.0",
            "context_kind": "single_product",
        }
    )

    assert result["valid"] is False
    diagnostic = result["diagnostics"][0]
    assert diagnostic["code"] == "SEMANTIC_DIMENSION_MISMATCH"
    assert diagnostic["node_id"] >= 0
    assert diagnostic["expected"]
    assert diagnostic["actual"]


def test_validation_accepts_redundant_mathbf_on_log_returns(tmp_path: Path) -> None:
    service = CustomIndicatorService(tmp_path, tmp_path)

    result = service.validate(
        {
            "name": "对数收益波动率",
            "expression": r"\operatorname{std}\left(\mathbf{\mathbf{\ell}},1\right)",
            "periods": ["1M"],
        }
    )

    assert result["valid"] is True
    assert result["dependencies"] == ["log_returns"]
    assert result["python_expression"] == "sequence_std(log_returns,1)"


def test_indicator_repository_persists_versions_and_detects_conflicts(tmp_path: Path) -> None:
    service = CustomIndicatorService(tmp_path, tmp_path)
    created = service.create_indicator(_draft())
    updated = service.update_indicator(
        created["id"],
        created["revision"],
        _draft(name="更新后指标", expression=r"\operatorname{mean}(\mathbf{r})"),
    )

    restarted = CustomIndicatorService(tmp_path, tmp_path)
    assert restarted.get_indicator(created["id"])["revision"] == 2
    assert restarted.get_indicator(created["id"])["indicator_type"] == "return"
    assert restarted.get_indicator(created["id"])["presentation"]["indicator_type"] == "return"
    assert restarted.indicators.get(created["id"], 1)["expression"] == created["expression"]
    assert updated["created_at"] == created["created_at"]
    with pytest.raises(ConflictError, match="刷新后重试"):
        restarted.update_indicator(created["id"], 1, _draft())
    payload = json.loads((tmp_path / "custom_indicators.json").read_text(encoding="utf-8"))
    assert payload["items"][0]["history"][0]["revision"] == 1


def test_historical_definition_loads_as_legacy_without_rewriting_file(tmp_path: Path) -> None:
    legacy = {
        **_draft(name="历史指标"),
        "id": "indicator-legacy",
        "revision": 1,
        "source": "custom",
        "read_only": False,
        "created_at": "2025-01-01T00:00:00+00:00",
        "updated_at": "2025-01-01T00:00:00+00:00",
    }
    legacy.pop("indicator_type")
    path = tmp_path / "custom_indicators.json"
    path.write_text(
        json.dumps({"schema_version": 1, "items": [{"current": legacy, "history": []}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    before = path.read_text(encoding="utf-8")

    loaded = CustomIndicatorService(tmp_path, tmp_path).get_indicator("indicator-legacy")

    assert loaded["dsl_version"] == "1.0.0"
    assert loaded["operator_registry_version"] == "legacy-v1"
    assert loaded["context_kind"] == "single_product"
    assert loaded["indicator_type"] == "other"
    assert path.read_text(encoding="utf-8") == before


def test_real_etf_and_fund_evaluation_has_fixed_week_window_and_cache(tmp_path: Path) -> None:
    _write_market_data(tmp_path)
    service = CustomIndicatorService(tmp_path, tmp_path)
    request = {
        "indicator_ids": ["builtin-cumulative-return"],
        "inline_definition": None,
        "targets": [
            {"kind": "etf", "product_id": "510050"},
            {"kind": "fund", "product_id": "000001.OF"},
        ],
        "period": "1W",
        "include_series": False,
    }

    first = service.evaluate(**request)
    second = service.evaluate(**request)

    assert first["summary"] == {"total": 2, "ok": 2, "warning": 0, "error": 0}
    assert all(item["value"] > 0 for item in first["results"])
    assert all(item["window"]["observation_count"] == 5 for item in first["results"])
    assert {item["target"]["name"] for item in first["results"]} == {"上证50ETF", "华夏成长"}
    assert second["cache"] == {"hits": 2, "misses": 0}


def test_natural_month_window_includes_anchor_and_uses_effective_as_of(tmp_path: Path) -> None:
    _write_market_data(tmp_path)
    service = CustomIndicatorService(tmp_path, tmp_path)

    result = service.evaluate(
        indicator_ids=["builtin-cumulative-return"],
        inline_definition=None,
        targets=[{"kind": "etf", "product_id": "510050.SH"}],
        period="1M",
        as_of="2026-03-22",  # Sunday: effective date must be the prior real data date.
    )["results"][0]

    assert result["window"]["requested_as_of"] == "2026-03-22"
    assert result["window"]["effective_as_of"] == "2026-03-20"
    assert result["window"]["start_date"] <= "2026-02-20"
    assert result["window"]["end_date"] == "2026-03-20"
    assert result["window"]["data_latest_date"] > result["window"]["effective_as_of"]


def test_non_finite_formula_result_is_null_with_explicit_warning(tmp_path: Path) -> None:
    _write_market_data(tmp_path)
    service = CustomIndicatorService(tmp_path, tmp_path)

    result = service.evaluate(
        indicator_ids=[],
        inline_definition=_draft(name="除零指标", expression="1/0", periods=["1W"]),
        targets=[{"kind": "etf", "product_id": "510050.SH"}],
        period="1W",
    )["results"][0]

    assert result["value"] is None
    assert result["status"] == "warning"
    assert result["warnings"][-1]["code"] == "DIVIDE_BY_ZERO"


def test_insufficient_sample_and_missing_data_return_null_with_warning(tmp_path: Path) -> None:
    _write_market_data(tmp_path, periods=4)
    service = CustomIndicatorService(tmp_path, tmp_path)

    result = service.evaluate(
        indicator_ids=["builtin-cumulative-return"],
        inline_definition=None,
        targets=[
            {"kind": "etf", "product_id": "510050.SH"},
            {"kind": "fund", "product_id": "missing.OF"},
        ],
        period="1W",
    )

    assert result["results"][0]["value"] is None
    assert result["results"][0]["warnings"][0]["code"] == "INSUFFICIENT_SAMPLE"
    assert result["results"][1]["value"] is None
    assert result["results"][1]["warnings"][0]["code"] == "DATA_NOT_FOUND"


def test_plan_run_uses_locked_indicator_revision_and_protects_references(tmp_path: Path) -> None:
    _write_market_data(tmp_path)
    service = CustomIndicatorService(tmp_path, tmp_path)
    created = service.create_indicator(
        _draft(name="收益求和", expression=r"\sum\left(\mathbf{r}\right)", periods=["1W"])
    )
    plan = service.create_plan(
        {
            "name": "锁定版本方案",
            "description": "",
            "indicators": [
                {
                    "indicator_id": created["id"],
                    "indicator_revision": created["revision"],
                    "period": "1W",
                    "weight": 100,
                    "direction": "higher_better",
                }
            ],
            "targets": [{"kind": "etf", "product_id": "510050.SH"}],
            "missing_policy": "strict",
        }
    )
    service.update_indicator(
        created["id"],
        1,
        _draft(name="平均收益", expression=r"\operatorname{mean}(\mathbf{r})", periods=["1W"]),
    )

    locked = service.evaluate(
        indicator_ids=[created["id"]],
        inline_definition=None,
        targets=plan["targets"],
        period="1W",
        indicator_versions={created["id"]: 1},
    )["results"][0]["value"]
    current = service.evaluate(
        indicator_ids=[created["id"]],
        inline_definition=None,
        targets=plan["targets"],
        period="1W",
    )["results"][0]["value"]
    run = service.run_plan(plan["id"])

    assert run["rows"][0]["values"][0]["indicator_revision"] == 1
    assert run["rows"][0]["values"][0]["value"] == pytest.approx(locked)
    assert locked != pytest.approx(current)
    with pytest.raises(ConflictError, match="正在被评价方案引用"):
        service.delete_indicator(created["id"], 2)


def test_strict_plan_excludes_target_with_missing_indicator_data(tmp_path: Path) -> None:
    _write_market_data(tmp_path)
    service = CustomIndicatorService(tmp_path, tmp_path)
    plan = service.create_plan(
        {
            "name": "严格样本方案",
            "description": "",
            "indicators": [
                {
                        "indicator_id": "builtin-total-return-v2",
                    "period": "1W",
                    "weight": 1,
                }
            ],
            "targets": [
                {"kind": "etf", "product_id": "510050.SH"},
                {"kind": "etf", "product_id": "missing.SH"},
            ],
            "missing_policy": "strict",
        }
    )
    run = service.run_plan(plan["id"])

    assert run["ranked_count"] == 1
    assert run["excluded_count"] == 1
    excluded = next(row for row in run["rows"] if row["status"] == "excluded")
    assert excluded["score"] is None
    assert excluded["missing_indicators"] == ["累计收益率"]
    assert excluded["exclusion_reasons"][0]["code"] == "PRODUCT_DATA_NOT_FOUND"


def test_plan_product_kinds_and_catalog_are_isolated(tmp_path: Path) -> None:
    service = CustomIndicatorService(tmp_path, tmp_path)
    etf_plan = service.create_plan(
        {
            "name": "ETF 方案",
            "product_kind": "etf",
            "indicators": [
                {"indicator_id": "builtin-total-return-v2", "period": "1Y", "weight": 100}
            ],
            "targets": [{"kind": "etf", "product_id": "510050.SH"}],
            "missing_policy": "strict",
        }
    )
    fund_plan = service.create_plan(
        {
            "name": "基金方案",
            "product_kind": "fund",
            "indicators": [
                {"indicator_id": "builtin-total-return-v2", "period": "1Y", "weight": 100}
            ],
            "targets": [{"kind": "fund", "product_id": "000001.OF"}],
            "missing_policy": "strict",
        }
    )

    assert etf_plan["product_kind"] == "etf"
    assert fund_plan["product_kind"] == "fund"
    assert etf_plan["product_selection"] == {
        "query": "",
        "filters": {
            "fund_type": [],
            "invest_type": [],
            "market": [],
            "status": [],
            "management": [],
            "custodian": [],
        },
        "conditions": [],
        "selection_mode": "manual",
    }
    assert [item["id"] for item in service.list_plans("etf")["items"]] == [etf_plan["id"]]
    assert [item["id"] for item in service.list_plans("fund")["items"]] == [fund_plan["id"]]

    with pytest.raises(ValidationError) as exc_info:
        service.create_plan(
            {
                "name": "混合方案",
                "product_kind": "etf",
                "indicators": [
                    {"indicator_id": "builtin-total-return-v2", "period": "1Y", "weight": 100}
                ],
                "targets": [
                    {"kind": "etf", "product_id": "510050.SH"},
                    {"kind": "fund", "product_id": "000001.OF"},
                ],
                "missing_policy": "strict",
            }
        )
    assert exc_info.value.code == "MIXED_PRODUCT_KINDS"


def test_large_typed_plan_scans_target_set_once(monkeypatch, tmp_path: Path) -> None:
    service = CustomIndicatorService(tmp_path, tmp_path)
    targets = [
        {"kind": "etf", "product_id": f"{index:06d}.SH"}
        for index in range(120)
    ]
    dates = pd.bdate_range("2024-01-02", periods=270)
    pd.DataFrame(
        [
            {"ts_code": target["product_id"], "name": target["product_id"]}
            for target in targets
        ]
    ).to_parquet(tmp_path / "etf_info_df.parquet", index=False)
    pd.DataFrame(
        [
            {
                "ts_code": target["product_id"],
                "date": date,
                "adj_nav": 1.0 + date_index * 0.001 + target_index * 0.00001,
            }
            for target_index, target in enumerate(targets)
            for date_index, date in enumerate(dates)
        ]
    ).to_parquet(tmp_path / "etf_daily_df.parquet", index=False)
    plan = service.create_plan(
        {
            "name": "全筛选结果方案",
            "product_kind": "etf",
            "indicators": [
                {"indicator_id": "builtin-total-return-v2", "period": "1Y", "weight": 100}
            ],
            "targets": targets,
            "missing_policy": "strict",
        }
    )
    second_plan = service.create_plan(
        {
            "name": "复用同一批数据",
            "product_kind": "etf",
            "indicators": [
                {
                    "indicator_id": "builtin-total-return-v2",
                    "period": "1Y",
                    "weight": 100,
                }
            ],
            "targets": targets,
            "missing_policy": "strict",
        }
    )
    batch_sizes: list[int] = []
    from custom_indicators import service as service_module

    original_loader = service_module.load_product_variable_series_batch

    def tracked_loader(kind, product_ids, dependencies, data_dir, as_of):
        product_ids = list(product_ids)
        batch_sizes.append(len(product_ids))
        return original_loader(kind, product_ids, dependencies, data_dir, as_of)

    monkeypatch.setattr(
        service_module,
        "load_product_variable_series_batch",
        tracked_loader,
    )

    run = service.run_plan(plan["id"])
    second_run = service.run_plan(second_plan["id"])
    cached_run = service.run_plan(plan["id"])

    assert batch_sizes == [120]
    assert run["ranked_count"] == 120
    assert run["excluded_count"] == 0
    assert run["execution"]["execution_lanes"] == {
        "numba_fused": 1,
        "numba_blas": 0,
        "python_fallback": 0,
    }
    assert run["execution"]["typed_batch_fallback"] == 0
    assert run["execution"]["python_operator_calls"] == 0
    assert second_run["execution"]["cache"]["data_hits"] == 1
    assert second_run["execution"]["cache"]["window_hits"] == 1
    assert cached_run["execution"]["cache"] == {
        "plan_hits": 1,
        "plan_misses": 0,
        "data_hits": 0,
        "data_misses": 0,
        "window_hits": 0,
        "window_misses": 0,
        "cell_hits": 120,
        "cell_misses": 0,
    }
    assert cached_run["execution"]["timings_ms"]["total"] == cached_run[
        "execution"
    ]["timings_ms"]["cache_lookup"]
    assert cached_run["execution"]["timings_ms"]["compute"] == 0.0


def test_plan_rejects_duplicate_locked_indicator_period(tmp_path: Path) -> None:
    service = CustomIndicatorService(tmp_path, tmp_path)

    with pytest.raises(ValidationError) as exc_info:
        service.create_plan(
            {
                "name": "重复配置",
                "indicators": [
                    {"indicator_id": "builtin-total-return-v2", "period": "1Y", "weight": 60},
                    {"indicator_id": "builtin-total-return-v2", "period": "1Y", "weight": 40},
                ],
                "targets": [{"kind": "etf", "product_id": "510050.SH"}],
                "missing_policy": "strict",
            }
        )

    assert exc_info.value.code == "DUPLICATE_PLAN_INDICATOR"


def test_shadow_comparison_checks_values_windows_and_ranks() -> None:
    row = {
        "rank": 1,
        "status": "ranked",
        "target": {"kind": "etf", "product_id": "510050.SH"},
        "values": [
            {
                "value": 0.123,
                "status": "ok",
                "window": {"start_date": "2025-01-01", "end_date": "2026-01-01"},
            }
        ],
    }
    equivalent = CustomIndicatorService._compare_shadow_results(
        {"rows": [row], "ranked_count": 1},
        {"rows": [copy.deepcopy(row)], "ranked_count": 1},
    )
    changed = copy.deepcopy(row)
    changed["values"][0]["value"] = 0.2
    mismatch = CustomIndicatorService._compare_shadow_results(
        {"rows": [row], "ranked_count": 1},
        {"rows": [changed], "ranked_count": 1},
    )

    assert equivalent["equivalent"] is True
    assert mismatch["equivalent"] is False
    assert mismatch["mismatch_count"] == 1


def test_large_result_contract_returns_first_page_and_result_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("INDICATOR_INLINE_RESULT_LIMIT", "1")
    service = CustomIndicatorService(tmp_path, tmp_path)
    dates = pd.bdate_range("2024-01-02", periods=270)
    products = ["510001.SH", "510002.SH"]
    pd.DataFrame(
        [{"ts_code": product, "name": product} for product in products]
    ).to_parquet(tmp_path / "etf_info_df.parquet", index=False)
    pd.DataFrame(
        [
            {
                "ts_code": product,
                "date": date,
                "adj_nav": 1.0 + date_index * (0.001 + product_index * 0.0001),
            }
            for product_index, product in enumerate(products)
            for date_index, date in enumerate(dates)
        ]
    ).to_parquet(tmp_path / "etf_daily_df.parquet", index=False)
    plan = service.create_plan(
        {
            "name": "分页方案",
            "product_kind": "etf",
            "indicators": [
                {
                    "indicator_id": "builtin-total-return-v2",
                    "period": "1Y",
                    "weight": 100,
                }
            ],
            "targets": [
                {"kind": "etf", "product_id": product} for product in products
            ],
            "missing_policy": "strict",
        }
    )

    first_page = service.run_plan(plan["id"])
    second_page = service.get_plan_run_result(
        first_page["result_id"], page=2, page_size=1
    )

    assert first_page["pagination"]["total"] == 2
    assert first_page["pagination"]["page"] == 1
    assert second_page["pagination"]["page"] == 2
    assert second_page["rows"][0]["rank"] == 2


def test_snapshot_indicator_config_locks_versions_and_protects_references(tmp_path: Path) -> None:
    service = CustomIndicatorService(tmp_path, tmp_path)
    initial = service.get_snapshot_config()

    assert initial["revision"] == 1
    assert initial["items"]
    assert all(item["status"] == "ready" for item in initial["items"])

    created = service.create_indicator(_draft(periods=None))
    updated = service.update_snapshot_config(
        initial["revision"],
        [
            {
                "indicator_id": created["id"],
                "indicator_revision": created["revision"],
                "period": "1Y",
            }
        ],
    )

    assert updated["revision"] == 2
    assert updated["items"][0]["indicator_id"] == created["id"]
    assert updated["items"][0]["field"].startswith("metric_")
    with pytest.raises(ConflictError) as stale:
        service.update_snapshot_config(initial["revision"], [])
    assert stale.value.code == "REVISION_CONFLICT"
    with pytest.raises(ConflictError) as referenced:
        service.delete_indicator(created["id"], created["revision"])
    assert referenced.value.code == "INDICATOR_IN_SNAPSHOT_CONFIG"
