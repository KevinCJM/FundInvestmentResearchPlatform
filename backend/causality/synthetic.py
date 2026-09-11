"""内置合成时序数据集：不联网、不落盘、可复现。

因果性审计需要一份随时可用的输入。真实行情不合适：要下载、会变、
且平稳的真实走势反而是**弱探针**——泄露发生时数值可能只差一点点，看不出来。
这里的数据是按「让泄露显影」设计的，不是按「像行情」设计的。

每条约束都对应一种会掩盖泄露的失败：

* 逐点唯一——若序列里有重复值，「把未来某点抄到过去」可能碰巧算出同一个数。
* 末段结构性突变——任何偷看尾部的算子，误差会被放大到肉眼可见，而不是淹没在噪声里。
* 净值恒正、收益率有界——``log``/``sqrt``/``drawdown_series`` 的定义域安全，
  ``prod(1+r)`` 不会塌到零以下；否则算子直接抛错，审计结果变成一片 UNKNOWN。
* 协方差满秩——``solve``/``covariance``/``quadratic_form`` 不退化。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from cal_indicators.typed_dsl import DEFAULT_VARIABLE_TYPES

CAUSALITY_DATASET_VERSION = "1.0.0"
DEFAULT_SEED = 20260908
DEFAULT_PERIODS = 512
DEFAULT_ASSETS = 6

#: 末段多长比例注入结构性突变。
BREAK_FRACTION = 0.15


@dataclass(frozen=True)
class SyntheticPanel:
    """一份确定性的合成面板，键名与 typed DSL 的上下文变量一一对应。"""

    periods: int
    assets: int
    seed: int
    variables: Mapping[str, Any]
    break_index: int
    version: str = CAUSALITY_DATASET_VERSION

    def context(self) -> dict[str, Any]:
        """完整窗口的上下文。"""

        return {name: _copy(value) for name, value in self.variables.items()}

    def context_asof(self, decision_index: int) -> dict[str, Any]:
        """截断到决策日 ``decision_index``（含）的上下文。

        全模块只有这一处处理 ``T`` 与 ``L`` 的长度差：``adjusted_nav`` 的类型
        符号是 ``L``（净值路径比收益率多一个起点），所以它要多留一个点。写错这里
        不会报错，只会让所有探针悄悄失效——所以它必须只有一份实现。
        """

        if not 0 <= decision_index < self.periods:
            raise ValueError(f"决策日下标越界: {decision_index} 不在 [0, {self.periods}) 内")
        sliced: dict[str, Any] = {}
        for name, value in self.variables.items():
            keep = _keep_count(name, decision_index)
            sliced[name] = _copy(value) if keep is None else _copy(value[:keep])
        return sliced

    def time_length(self, name: str) -> int | None:
        """变量沿时间轴的长度；无时间轴时为 ``None``。"""

        value = self.variables[name]
        return None if _time_symbol(name) is None else int(np.shape(value)[0])


def _copy(value: Any) -> Any:
    return value.copy() if isinstance(value, np.ndarray) else value


def _time_symbol(name: str) -> str | None:
    """变量沿时间轴的符号维度（``"T"`` / ``"L"``），无时间轴时为 ``None``。"""

    value_type = DEFAULT_VARIABLE_TYPES.get(name)
    if value_type is None or "time" not in value_type.axes:
        return None
    axis = value_type.axes.index("time")
    if axis != 0:
        raise AssertionError(f"{name} 的时间轴不在 0 号位置，截断逻辑需要重写")
    return str(value_type.shape[axis])


def _keep_count(name: str, decision_index: int) -> int | None:
    symbol = _time_symbol(name)
    if symbol is None:
        return None
    # ``L`` 比 ``T`` 多一个起点，截断时必须同步多留一个，否则两者长度关系被破坏，
    # 同时引用净值与收益率的公式会直接编译失败。
    return decision_index + 2 if symbol == "L" else decision_index + 1


def synthetic_panel(
    periods: int = DEFAULT_PERIODS,
    assets: int = DEFAULT_ASSETS,
    seed: int = DEFAULT_SEED,
) -> SyntheticPanel:
    """生成确定性合成面板。相同参数永远得到相同数据。"""

    if periods < 32:
        raise ValueError("periods 至少 32，否则滚动窗口与预热探针无从展开")
    if assets < 2:
        raise ValueError("assets 至少 2，否则协方差与截面算子退化")

    rng = np.random.default_rng(seed)
    break_index = int(periods * (1.0 - BREAK_FRACTION))

    # 三因子 + 特异噪声：载荷各不相同，保证资产间协方差满秩。
    factors = rng.normal(0.0, 0.006, size=(periods, 3))
    loadings = rng.uniform(0.4, 1.6, size=(3, assets)) * np.sign(
        rng.uniform(-1.0, 1.0, size=(3, assets))
    )
    idiosyncratic = rng.normal(0.0, 0.004, size=(periods, assets))
    asset_returns = factors @ loadings + idiosyncratic

    # 末段结构性突变：波动放大 + 一次跳空。偷看尾部的算子在这里会被放大显影。
    asset_returns[break_index:] *= 3.0
    asset_returns[break_index] += 0.07

    # 有界化后 P1 的 1.9 倍扰动仍然远离 -1，``prod(1+r)`` 不会塌。
    asset_returns = np.clip(asset_returns, -0.2, 0.2)

    returns = asset_returns[:, 0].copy()
    log_returns = np.log1p(returns)
    asset_log_returns = np.log1p(asset_returns)

    asset_weights = _normalized(rng.uniform(0.5, 2.0, size=assets))
    weight_path = _normalized(rng.uniform(0.5, 2.0, size=(periods, assets)), axis=1)
    portfolio_returns = np.sum(weight_path * asset_returns, axis=1)
    benchmark_returns = asset_returns @ np.full(assets, 1.0 / assets)

    # 复权净值走 ``L = T + 1``：多出来的是起点 1.0。
    adjusted_nav = np.concatenate(([1.0], np.cumprod(1.0 + returns)))
    unit_nav = adjusted_nav[1:] * 0.97
    accumulated_nav = unit_nav + np.cumsum(np.full(periods, 0.0012))

    close = adjusted_nav[1:] * 10.0
    previous_close = np.concatenate(([close[0] / (1.0 + returns[0])], close[:-1]))
    open_ = previous_close * (1.0 + rng.normal(0.0, 0.001, size=periods))
    spread = np.abs(rng.normal(0.0, 0.004, size=periods)) + 1e-4
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    volume = np.abs(rng.lognormal(12.0, 0.35, size=periods))
    turnover_amount = volume * close

    variables: dict[str, Any] = {
        "observation_dates": np.arange(periods + 1, dtype=np.float64) + 20_000.0,
        "returns": returns,
        "log_returns": log_returns,
        "asset_returns": asset_returns,
        "asset_log_returns": asset_log_returns,
        "portfolio_returns": portfolio_returns,
        "asset_weights": asset_weights,
        "weight_path": weight_path,
        "benchmark_returns": benchmark_returns,
        "annual_risk_free_rate_decimal": 0.025,
        "risk_free_rate_per_period": 0.025 / 252.0,
        "periods_per_year": 252.0,
        "adjusted_nav": adjusted_nav,
        "unit_nav": unit_nav,
        "accumulated_nav": accumulated_nav,
        "market_open": open_,
        "market_high": high,
        "market_low": low,
        "market_close": close,
        "previous_close": previous_close,
        "price_change": close - previous_close,
        "price_return": close / previous_close - 1.0,
        "volume": volume,
        "turnover_amount": turnover_amount,
        # 旧公式仍在用的行情别名，与上面的 market_* 指向同一条序列。
        "open_price": open_,
        "high_price": high,
        "low_price": low,
        "close_price": close,
    }

    missing = set(DEFAULT_VARIABLE_TYPES) - set(variables)
    if missing:
        raise AssertionError(
            f"合成数据集缺少 DSL 变量: {sorted(missing)}；新增变量必须同步在这里生成，"
            "否则引用它的公式无法审计。"
        )

    return SyntheticPanel(
        periods=periods,
        assets=assets,
        seed=seed,
        variables=variables,
        break_index=break_index,
    )


def _normalized(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    total = values.sum(axis=axis, keepdims=axis is not None)
    return values / total


# --------------------------------------------------------------------------
# 对照组：探针自己坏掉时，这些必须先响。
# --------------------------------------------------------------------------

def _rolling_mean_20(values: np.ndarray) -> np.ndarray:
    window = 20
    padded = np.concatenate((np.full(window - 1, np.nan), values))
    return np.asarray(
        [np.mean(padded[index : index + window]) for index in range(values.size)]
    )


def _drawdown(values: np.ndarray) -> np.ndarray:
    path = np.cumprod(1.0 + values)
    return path / np.maximum.accumulate(path) - 1.0


def _rank_over_time(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(values))
    return order / max(values.size - 1, 1)


#: 已知因果的样本函数：探针必须放行。
KNOWN_CAUSAL = {
    "identity": lambda values: values.copy(),
    "rolling_mean_20": _rolling_mean_20,
    "cumulative_return": lambda values: np.cumprod(1.0 + values) - 1.0,
    "expanding_mean": lambda values: np.cumsum(values) / np.arange(1, values.size + 1),
    "running_max": np.maximum.accumulate,
    "drawdown": _drawdown,
}

#: 已知带未来函数的样本函数：探针必须抓住。
KNOWN_LEAKY = {
    # 最直白的一类：t 日直接取 t+1 日的值。
    "shift_minus_1": lambda values: np.append(values[1:], values[-1]),
    # 作用域错误：全期均值/标准差回头改写了每一个历史点。
    "full_window_zscore": lambda values: (values - values.mean()) / values.std(),
    # 用全期最高点做归一化——回测里永远不会在山顶发信号。
    "normalize_by_global_max": lambda values: values / np.abs(values).max(),
    # 拿今天跟未来比名次。
    "rank_over_time": _rank_over_time,
    # 整条序列都等于最后一个值。
    "peek_last_value": lambda values: np.full(values.size, values[-1]),
}


def hostile_variants(periods: int = 128) -> dict[str, np.ndarray]:
    """退化输入：探针在这些上面应当报 UNKNOWN，而不是误报 LEAK。"""

    return {
        "constant": np.full(periods, 0.01),
        "monotonic": np.linspace(0.001, 0.05, periods),
        "with_nan": np.concatenate((np.full(8, np.nan), np.linspace(0.01, 0.03, periods - 8))),
        "very_short": np.asarray([0.01, -0.02, 0.03]),
        "all_zero": np.zeros(periods),
    }
