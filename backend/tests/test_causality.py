"""因果性审计的自测。

第一组（对照组）是其余所有断言的前提：一个不会误报也不会漏报的探针，才有资格
去给 104 个算子和用户公式下裁决。探针自己坏掉时，这一组必须先响。
"""

from __future__ import annotations

import numpy as np
import pytest

from causality import (
    KNOWN_CAUSAL,
    KNOWN_LEAKY,
    Verdict,
    audit_expression,
    audit_operators,
    compare_prefix,
    hostile_variants,
    load_baseline,
    perturb_tail,
    synthetic_panel,
    tail_perturbation,
)
from causality.audit import compare_to_baseline
from cal_indicators.typed_dsl import DEFAULT_VARIABLE_TYPES


@pytest.fixture(scope="module")
def panel():
    return synthetic_panel()


@pytest.fixture(scope="module")
def operator_findings():
    return audit_operators()


# --------------------------------------------------------------------------
# 对照组：探针本身可信吗
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(KNOWN_CAUSAL))
def test_known_causal_functions_pass(panel, name: str) -> None:
    outcome = tail_perturbation(KNOWN_CAUSAL[name], panel.variables["returns"])
    assert outcome.verdict is Verdict.CAUSAL, f"{name} 被误报：{outcome.detail}"


@pytest.mark.parametrize("name", sorted(KNOWN_LEAKY))
def test_known_leaky_functions_are_caught(panel, name: str) -> None:
    outcome = tail_perturbation(KNOWN_LEAKY[name], panel.variables["returns"])
    assert outcome.verdict is Verdict.LEAK, f"{name} 未被抓住：{outcome.detail}"
    assert outcome.first_mismatch is not None
    assert outcome.baseline_value != outcome.perturbed_value


@pytest.mark.parametrize("name", sorted(hostile_variants()))
def test_degenerate_inputs_never_report_leak(name: str) -> None:
    """退化输入下宁可报 UNKNOWN，也不能凭空诬告一个因果函数。"""

    outcome = tail_perturbation(KNOWN_CAUSAL["rolling_mean_20"], hostile_variants()[name])
    assert outcome.verdict in (Verdict.CAUSAL, Verdict.UNKNOWN)


# --------------------------------------------------------------------------
# 数据集
# --------------------------------------------------------------------------

def test_panel_is_deterministic() -> None:
    left, right = synthetic_panel(periods=128), synthetic_panel(periods=128)
    for name, value in left.variables.items():
        assert np.array_equal(np.asarray(value), np.asarray(right.variables[name])), name


def test_panel_covers_every_dsl_variable(panel) -> None:
    """新增 DSL 变量却忘了在合成数据里生成，引用它的公式就无法审计。"""

    assert set(DEFAULT_VARIABLE_TYPES) <= set(panel.variables)


def test_panel_values_do_not_mask_leaks(panel) -> None:
    returns = panel.variables["returns"]
    # 逐点唯一：有重复值时，「把未来某点抄到过去」可能碰巧算出同一个数。
    assert len(np.unique(returns)) == returns.size
    # 定义域安全：净值恒正，收益率远离 -1，扰动放大 6 倍后依然成立。
    assert np.all(panel.variables["adjusted_nav"] > 0)
    assert returns.min() > -1.0 / 6.0
    # 末段结构性突变，让偷看尾部的算子被放大显影。
    tail = returns[panel.break_index :]
    assert tail.std() > returns[: panel.break_index].std() * 1.5


def test_covariance_is_full_rank(panel) -> None:
    covariance = np.cov(panel.variables["asset_returns"], rowvar=False)
    assert np.linalg.matrix_rank(covariance) == panel.assets


def test_weights_are_valid(panel) -> None:
    assert panel.variables["asset_weights"].min() >= 0
    assert panel.variables["asset_weights"].sum() == pytest.approx(1.0)
    assert np.allclose(panel.variables["weight_path"].sum(axis=1), 1.0)


def test_context_asof_keeps_nav_one_point_longer(panel) -> None:
    """``adjusted_nav`` 的长度符号是 ``L`` = T + 1；截断时必须同步多留一个点。

    写错这里不会报错，只会让扰动落在决策日之内，把因果公式误判成泄露。
    """

    sliced = panel.context_asof(99)
    assert sliced["returns"].size == 100
    assert sliced["asset_returns"].shape == (100, panel.assets)
    assert sliced["adjusted_nav"].size == 101
    assert sliced["periods_per_year"] == panel.variables["periods_per_year"]
    assert np.array_equal(sliced["returns"], panel.variables["returns"][:100])


def test_context_asof_rejects_out_of_range(panel) -> None:
    with pytest.raises(ValueError):
        panel.context_asof(panel.periods)


# --------------------------------------------------------------------------
# 探针机制
# --------------------------------------------------------------------------

def test_perturb_tail_leaves_the_prefix_untouched(panel) -> None:
    values = panel.variables["returns"]
    perturbed = perturb_tail(values, 50)
    assert perturbed is not None
    assert np.array_equal(perturbed[:51], values[:51])
    assert not np.array_equal(perturbed[51:], values[51:])


def test_perturb_tail_reports_when_it_cannot_perturb() -> None:
    assert perturb_tail(np.zeros(20), 5) is None, "全零序列缩放后没变，必须如实返回 None"
    assert perturb_tail(np.arange(10.0), 9) is None, "尾部为空"


def test_perturb_tail_handles_masks() -> None:
    mask = np.asarray([True, False, True, True, False])
    perturbed = perturb_tail(mask, 1)
    assert perturbed is not None
    assert np.array_equal(perturbed[:2], mask[:2])
    assert perturbed.dtype == np.bool_


def test_shortened_output_is_right_aligned() -> None:
    """``difference(x, n)`` 返回 ``x[n:] - x[:-n]``，第 j 项属于时点 j+n。

    按左对齐比较会把它误判成读取未来——本模块最容易出的假阳性。
    """

    values = np.linspace(1.0, 2.0, 64)
    outcome = tail_perturbation(lambda array: array[2:] - array[:-2], values)
    assert outcome.verdict is Verdict.CAUSAL


def test_compare_prefix_flags_grey_zone_as_unknown() -> None:
    baseline = np.ones(10)
    perturbed = baseline.copy()
    perturbed[3] += 1e-7  # 大于 tight、小于 loose
    outcome = compare_prefix(baseline, perturbed, decision_index=8, input_length=10)
    assert outcome is not None and outcome.verdict is Verdict.UNKNOWN


def test_compare_prefix_flags_macroscopic_change_as_leak() -> None:
    baseline = np.ones(10)
    perturbed = baseline.copy()
    perturbed[3] = 5.0
    outcome = compare_prefix(baseline, perturbed, decision_index=8, input_length=10)
    assert outcome is not None and outcome.verdict is Verdict.LEAK
    assert outcome.first_mismatch == 3


def test_compare_prefix_follows_the_time_axis() -> None:
    """转置后时间轴不在 0 号位；按 0 号轴切会得到荒谬的偏移和沉默的假阴性。"""

    baseline = np.zeros((4, 40))
    perturbed = baseline.copy()
    perturbed[:, 2] = 9.0
    outcome = compare_prefix(
        baseline, perturbed, decision_index=20, input_length=40, time_axis=1
    )
    assert outcome is not None and outcome.verdict is Verdict.LEAK


# --------------------------------------------------------------------------
# 算子级审计
# --------------------------------------------------------------------------

def test_no_builtin_operator_leaks(operator_findings) -> None:
    leaking = [f.operator_id for f in operator_findings if f.verdict is Verdict.LEAK]
    assert not leaking, f"内置算子出现未来函数: {leaking}"


def test_every_operator_gets_a_verdict(operator_findings) -> None:
    unknown = [f.operator_id for f in operator_findings if f.verdict is Verdict.UNKNOWN]
    assert not unknown, f"以下算子无法构造合法输入，需人工核定: {unknown}"
    assert all(f.cases for f in operator_findings)


@pytest.mark.parametrize(
    ("operator_id", "expected"),
    [
        # 截面聚合只用当日数据，天然 PIT-safe；沿时间轴聚合则吃掉整个窗口。
        ("mean_asset", Verdict.CAUSAL),
        ("mean_time", Verdict.WINDOW_CONSUMING),
        ("max_asset", Verdict.CAUSAL),
        ("max_time", Verdict.WINDOW_CONSUMING),
        # lag 丢弃末尾且拒绝负 periods，difference 右对齐——两者都是因果的。
        ("lag", Verdict.CAUSAL),
        ("difference", Verdict.CAUSAL),
        ("rolling_mean", Verdict.CAUSAL),
        ("recursive_smooth", Verdict.CAUSAL),
        ("drawdown_series", Verdict.CAUSAL),
        # first 只读窗口头部，last 与 max_consecutive_true 真的依赖尾部。
        ("first", Verdict.CAUSAL),
        ("last", Verdict.WINDOW_CONSUMING),
        ("max_consecutive_true", Verdict.WINDOW_CONSUMING),
        ("argmax", Verdict.WINDOW_CONSUMING),
    ],
)
def test_representative_operator_verdicts(operator_findings, operator_id, expected) -> None:
    finding = next(f for f in operator_findings if f.operator_id == operator_id)
    assert finding.verdict is expected, finding.detail


def test_warmup_is_reported_separately_from_leakage(operator_findings) -> None:
    """预热敏感是可复现性性质，不是泄露；混进裁决会让因果算子看起来有问题。"""

    by_id = {f.operator_id: f for f in operator_findings}
    assert by_id["recursive_smooth"].warmup_sensitive is True
    assert by_id["recursive_smooth"].verdict is Verdict.CAUSAL
    assert by_id["cumulative_sum"].warmup_sensitive is True
    assert by_id["rolling_mean"].warmup_sensitive is False


def test_baseline_covers_every_operator(operator_findings) -> None:
    """新增算子没有裁决即失败——与 NJIT 预热同样的 fail-closed 约定。"""

    baseline = load_baseline()
    assert baseline, "baseline.json 缺失；用 python -m causality.audit 生成"
    drift = compare_to_baseline(operator_findings, baseline)
    assert not drift, (
        f"算子裁决与基线不一致: {drift}。"
        "verdict 变化必须人工复核后再用 python -m causality.audit 更新基线。"
    )


# --------------------------------------------------------------------------
# 公式级审计
# --------------------------------------------------------------------------

def test_full_window_zscore_is_blocked(panel) -> None:
    """经典泄露：全期均值/标准差回头改写了每一个历史点。"""

    report = audit_expression("(returns - mean(returns)) / std(returns)", panel=panel)
    assert report.verdict is Verdict.LEAK
    assert report.blocked is True
    # 报错必须指出责任节点，「有偏/无偏」这种布尔结论对用户没有用。
    assert "mean(returns)" in report.detail


def test_normalize_by_global_max_is_blocked(panel) -> None:
    report = audit_expression("adjusted_nav / max_value(adjusted_nav)", panel=panel)
    assert report.blocked is True


def test_rolling_formula_is_causal(panel) -> None:
    report = audit_expression(
        "rolling_mean(returns, 20, 1) - rolling_mean(returns, 60, 1)", panel=panel
    )
    assert report.verdict is Verdict.CAUSAL
    assert report.blocked is False


def test_nav_formula_survives_the_length_offset(panel) -> None:
    """净值走 ``L`` 轴。偏移写错会让这条完全因果的公式被误判成泄露。"""

    report = audit_expression("difference(adjusted_nav, 1) / lag(adjusted_nav, 1)", panel=panel)
    assert report.verdict is Verdict.CAUSAL


def test_drawdown_is_causal(panel) -> None:
    assert audit_expression("drawdown_series(adjusted_nav)", panel=panel).verdict is Verdict.CAUSAL


def test_scalar_reduction_is_window_consuming(panel) -> None:
    report = audit_expression("mean(returns) / std(returns)", panel=panel)
    assert report.verdict is Verdict.WINDOW_CONSUMING
    assert report.blocked is False
    assert len(report.window_consuming) == 2


def test_cross_sectional_reduction_is_causal(panel) -> None:
    """截面聚合只用当日数据，不该被当成消费整窗。"""

    assert audit_expression("mean_asset(asset_returns)", panel=panel).verdict is Verdict.CAUSAL


def test_uncompilable_expression_reports_unknown(panel) -> None:
    report = audit_expression("这不是公式(", panel=panel)
    assert report.verdict is Verdict.UNKNOWN
    assert report.blocked is False


def test_report_is_json_serializable(panel) -> None:
    import json

    payload = audit_expression("rolling_mean(returns, 20, 1)", panel=panel).as_dict()
    assert json.loads(json.dumps(payload, ensure_ascii=False))["verdict"] == "causal"
