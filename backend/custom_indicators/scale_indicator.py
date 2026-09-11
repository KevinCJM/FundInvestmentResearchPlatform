"""规模与份额：ETF 的存量指标，随研究日变化。

规模不是维表属性，而是一个算出来的量——份额 × 单位净值。所以它和收益率一样
受 PIT 约束：站在 2014 年末，看到的应当是那天的份额和那天的净值，而不是今天的。
产品页头部的「当前规模」原先读的是快照表最后一行，因此在任何历史研究日下都是错的。

`total_size` 直接取数据源同排给出的乘积，而不是自己乘一遍：同一行里份额和净值
已经对齐，跨数据集再对齐一次只会多出一个失败点。
"""
from __future__ import annotations

from cal_indicators.typed_numba_kernels import NUMERIC_KERNEL_VERSION
from cal_indicators.typed_operators import TYPED_DSL_VERSION, TYPED_OPERATOR_REGISTRY_VERSION
from .periods import SUPPORTED_PERIODS
from .variable_registry import CONTEXT_SCHEMA_VERSION, DATA_CONTRACT_VERSION, VARIABLE_REGISTRY_VERSION

FUND_SIZE_ID = "builtin-fund-size-latest"
FUND_SHARE_ID = "builtin-fund-share-latest"


def scale_indicators():
    common = {
        "revision": 1, "source": "built_in", "read_only": True,
        "created_at": "2026-09-09T00:00:00+00:00", "updated_at": "2026-09-09T00:00:00+00:00",
        "context_kind": "single_product", "indicator_type": "scale",
        "result_kind": "scalar", "output_contract": "scalar",
        "series_outputs": [], "parameter_schema": [], "fixed_parameters": [],
        "annual_risk_free_rate_percent": 0.0, "periods": list(SUPPORTED_PERIODS),
        "period_policy": "all_supported", "minimum_observations": 1,
        # 场外基金没有 etf_share_size 数据源，份额与规模在那里不可算。
        "applicable_product_kinds": ["etf"],
        "availability_policy": "runtime_required", "availability_status": "runtime_check",
        "dsl_version": TYPED_DSL_VERSION, "operator_registry_version": TYPED_OPERATOR_REGISTRY_VERSION,
        "numeric_kernel_version": NUMERIC_KERNEL_VERSION, "variable_registry_version": VARIABLE_REGISTRY_VERSION,
        "data_contract_version": DATA_CONTRACT_VERSION, "context_schema_version": CONTEXT_SCHEMA_VERSION,
        "template_origin": None,
        "data_basis": "etf_share_size 同排份额与单位净值；按事件日期截断，不填充，不外推。",
        "methodology": "取研究窗口内最后一个有效观察值。份额变动只在披露当日生效，不向前回填。",
        "direction": "neutral",
    }
    specs = (
        (FUND_SIZE_ID, "规模（期末）", "last(fund_size)", "currency_amount", "万元", "number", 2,
         "研究窗口最后一个交易日的规模＝份额 × 单位净值。随研究日变化，不是维表上的最新值。"),
        (FUND_SHARE_ID, "份额（期末）", "last(total_share)", "count", "万份", "number", 2,
         "研究窗口最后一个交易日的 ETF 总份额。"),
    )
    return [
        {
            **common, "id": key, "name": name, "expression": expression,
            "output_measure": measure, "unit": unit, "display_format": display,
            "precision": precision, "description": description,
            "required_variables": [expression[len("last(") : -1]],
        }
        for key, name, expression, measure, unit, display, precision, description in specs
    ]
