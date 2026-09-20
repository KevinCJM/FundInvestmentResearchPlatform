"""Fee coverage and dated cash orchestration, separate from numeric cost solving."""

from datetime import date
import numpy as np
from backend.custom_indicators.errors import ValidationError
from . import cost_kernels as kernels


def fee_arrays(candidate):
    kernels.require_ready()
    missing = []
    for p in candidate.products:
        if p.kind == "cash":
            continue
        if (
            p.buy_rate is None
            or p.sell_rate is None
            or len(p.fee_source) < 3
            or p.fee_basis != "each_side_notional"
            or p.fee_valid_until is None
            or p.fee_valid_until <= max(candidate.as_of, date.today())
            or p.nav_includes_management_fee is None
        ):
            missing.append(p.product_id)
        if p.kind == "fund" and (p.holding_days is None or not p.channel):
            missing.append(p.product_id)
    if missing:
        raise ValidationError(
            "IMPLEMENTATION_FEES_MISSING",
            "请补齐有效买卖费率、收费基数、来源及 NAV 费用覆盖："
            + "、".join(dict.fromkeys(missing)),
        )
    if candidate.annual_additional_fee and len(candidate.annual_fee_source) < 3:
        raise ValidationError(
            "IMPLEMENTATION_ANNUAL_FEE",
            "附加年费须注明来源，并确认未被 NAV 或原资金预算包含。",
        )
    return (
        np.asarray(
            [0.0 if p.kind == "cash" else p.buy_rate for p in candidate.products],
            dtype=np.float64,
        ),
        np.asarray(
            [0.0 if p.kind == "cash" else p.sell_rate for p in candidate.products],
            dtype=np.float64,
        ),
        np.asarray([int(p.kind == "cash") for p in candidate.products], dtype=np.int64),
    )


def balance(candidate):
    state = candidate.state
    product_values = np.asarray(
        [p.current_value for p in candidate.products if p.kind != "cash"],
        dtype=np.float64,
    )
    value, valid = kernels.balance_kernel(
        product_values,
        state.settled_cash,
        state.restricted_cash,
        state.receivables,
        state.payables,
        state.confirmed_investable_value,
    )
    if not valid:
        raise ValidationError(
            "IMPLEMENTATION_BALANCE",
            "产品市值＋可支付现金＋受限现金＋应收－应付与确认的可投资总价值不一致。组合外储备不再扣除。",
        )
    for p in candidate.products:
        if p.kind == "cash" and abs(p.current_value - state.settled_cash) > 1e-6:
            raise ValidationError(
                "IMPLEMENTATION_CASH_BALANCE",
                "现金项目的当前金额须与可支付现金一致，不能重复计入。",
            )
    return float(value)


def transition(candidate):
    wealth = balance(candidate)
    state = candidate.state
    if state.restricted_cash or state.receivables or state.payables:
        raise ValidationError(
            "IMPLEMENTATION_TRANSITION_BALANCES",
            "含受限现金或未结算应收应付时，请先制定分期实施方案；当前连续目标不能提前使用这些余额。",
        )
    buy, sell, cash = fee_arrays(candidate)
    values = np.asarray([p.current_value for p in candidate.products], dtype=np.float64)
    weights = np.asarray([p.weight for p in candidate.products], dtype=np.float64)
    after, trades, fees, stats = kernels.self_financing_kernel(
        wealth, values, weights, buy, sell, cash
    )
    if state.transition_cost_in_balance and any(
        abs(float(delta)) > 1e-6 for delta in trades
    ):
        raise ValidationError(
            "IMPLEMENTATION_COST_ALREADY_INCLUDED",
            "声明本次调整费用已计入余额时，当前持仓必须已达到目标权重；尚待执行的调整不能视为已付费。",
        )
    return {
        "cost": float(stats[0]),
        "post_cost_value": float(stats[1]),
        "gross_traded_notional": float(stats[2]),
        "half_turnover": float(stats[3]),
        "self_financing_residual": float(stats[4]),
        "post_cost_holdings": after.tolist(),
        "trades": trades.tolist(),
        "fees": fees.tolist(),
        "coverage": [
            {
                "product_id": p.product_id,
                "buy_rate": p.buy_rate,
                "sell_rate": p.sell_rate,
                "included_upstream": p.nav_includes_management_fee,
                "source": p.fee_source,
            }
            for p in candidate.products
        ],
        "scope": "declared_linear_cost_estimate_on_nav_proxy",
        "tax_basis": "pre_tax",
        "capacity_and_execution_price": "unavailable",
    }


def cash_calendar(candidate, trade):
    events = [x.model_dump(mode="json") for x in candidate.state.cash_events]
    unknown = [x["id"] for x in events if not x["date"]]
    valuation = str(candidate.state.valuation_at)
    for i, p in enumerate(candidate.products):
        if p.kind == "cash" or abs(trade["trades"][i]) < 1e-7:
            continue
        delta, fee = trade["trades"][i], trade["fees"][i]
        if delta > 0:
            events.append(
                {
                    "id": "buy:" + p.product_id,
                    "date": valuation,
                    "kind": "payment",
                    "amount": delta + fee,
                }
            )
        elif (
            p.settlement_date is None
            or p.settlement_date < candidate.state.valuation_at
        ):
            unknown.append("sell:" + p.product_id)
        else:
            events.append(
                {
                    "id": "sell:" + p.product_id,
                    "date": str(p.settlement_date),
                    "kind": "receipt",
                    "amount": -delta - fee,
                }
            )
    if any(x["date"] and x["date"] < valuation for x in events):
        raise ValidationError(
            "CASH_EVENT_ALREADY_IN_BALANCE",
            "余额日前的现金事件应先纳入确认余额，不能再次入账。",
        )
    known = sorted(
        (x for x in events if x["date"]),
        key=lambda x: (x["date"], x["kind"] == "payment", x["id"]),
    )
    days = np.asarray([x["date"] for x in known], dtype="datetime64[D]").astype(
        np.int64
    )
    amounts = np.asarray(
        [x["amount"] if x["kind"] == "receipt" else -x["amount"] for x in known],
        dtype=np.float64,
    )
    balances, gaps = kernels.cash_calendar_kernel(
        candidate.state.settled_cash, days, amounts
    )
    return {
        "status": (
            "unavailable" if unknown else "failed" if np.any(gaps > 1e-7) else "passed"
        ),
        "unknown_events": unknown,
        "scope": "confirmed_events_end_of_day_no_bridge_financing",
        "events": [
            {**x, "available_balance": float(balances[i]), "cash_gap": float(gaps[i])}
            for i, x in enumerate(known)
        ],
        "payment_calendar_complete": False,
        "limitation": "只核对已登记的日期事件；尚未认证未来所有赎回、支付及盘中截止时间。",
    }
