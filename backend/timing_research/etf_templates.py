"""Editable ETF adaptations of WF ideas; metadata, never numerical kernels.

Each feature, threshold and event is a real graph step. The adaptation record
states where ETF inputs and the shared train/freeze protocol differ from WF.
"""
from __future__ import annotations


class _Graph:
    """Small definition builder sharing identical named feature dependencies."""

    def __init__(self):
        self.nodes: list[dict] = []
        self._by_id: dict[str, dict] = {}

    def add(self, key, label, op, inputs=None, **parameters):
        node = {"id": key, "label": label, "op": op,
                "inputs": inputs or {}, "parameters": parameters}
        if key in self._by_id:
            if self._by_id[key] != node:
                raise ValueError(f"Conflicting template step: {key}")
        else:
            self._by_id[key] = node
            self.nodes.append(node)
        return key + ".value"

    def source(self, field="close"):
        labels = {"close": "收盘价", "open": "开盘价", "high": "最高价",
                  "low": "最低价", "volume": "成交量"}
        return self.add(field, "ETF " + labels[field], "source", field=field)

    def formula(self, key, label, expression, *refs, panel=False):
        return self.add(key, label, "panel_formula" if panel else "formula",
                        dict(zip("abcd", refs)), expression=expression)

    def lag(self, key, label, ref, periods=1, panel=False):
        return self.add(key, label, "panel_lag" if panel else "indicator.lag",
                        {"values": ref}, periods=periods)

    def compare(self, key, label, ref, operator="gt", threshold=0.0,
                right=None, panel=False):
        inputs = {"left": ref}
        if right is not None:
            inputs["right"] = right
        return self.add(key, label, "panel_compare" if panel else "compare",
                        inputs, operator=operator, threshold=threshold)

    def combine(self, key, label, *refs, op="all"):
        if len(refs) < 2:
            raise ValueError("A logical combination needs at least two inputs")
        result = refs[0]
        for index, ref in enumerate(refs[1:], 1):
            final = index == len(refs) - 1
            result = self.add(key if final else f"{key}_{index}", label if final
                              else f"{label} · 条件 {index + 1}", op,
                              {"left": result, "right": ref})
        return result

    def invert(self, key, label, ref):
        return self.add(key, label, "not", {"value": ref})

    def event(self, key, label, ref, op="first", window=2):
        return self.add(key, label, op, {"value": ref},
                        **({} if op == "first" else {"window": window}))

    def number(self, key, label, ref):
        return self.add(key, label, "condition_value", {"value": ref})

    def returns(self, window):
        price = self.source()
        previous = self.lag(f"close_lag{window}", f"{window} 日前收盘", price, window)
        return self.formula(f"return{window}", f"{window} 日收益率", "a/b-1", price, previous)

    def ma(self, window):
        return self.formula(f"ma{window}", f"{window} 日均价", f"rolling_mean(a,{window},{window})", self.source())

    def basket(self, group="market"):
        return self.add(group, "市场参考篮子" if group == "market" else "同类 ETF 参考篮子",
                        "basket_source", group=group)

    def basket_return(self, window, group="market"):
        price = self.basket(group)
        previous = self.lag(f"{group}_lag{window}", f"参考篮子 {window} 日前价格", price, window, panel=True)
        changes = self.formula(f"{group}_returns{window}", f"参考成员 {window} 日收益", "a/b-1", price, previous, panel=True)
        return self.add(f"{group}_return{window}", f"参考篮子等权 {window} 日收益", "cross_mean", {"value": changes})

    def breadth(self, window, group="market"):
        price = self.basket(group)
        average = self.formula(f"{group}_ma{window}", f"参考成员 {window} 日均价",
                               f"rolling_mean(a,{window},{window})", price, panel=True)
        above = self.compare(f"{group}_above{window}", f"参考成员站上 MA{window}",
                             price, right=average, panel=True)
        return self.add(f"{group}_breadth{window}", f"参考篮子 MA{window} 广度", "breadth", {"value": above})


def _action(key, label, entry):
    return {"id": key, "label": label, "entry": entry}


def _search(label, node, parameter, values):
    return {"label": label, "choices": [[{"node": node, "parameter": parameter, "value": value}]
                                         for value in values]}


def _training(actions, *, mode="global", states=(), search=(), embargo=0):
    return {"mode": mode, "state_refs": list(states), "actions": list(actions) + [_action("cash", "空仓", None)],
            "search_space": list(search), "min_trades": 5, "confidence": 1.0,
            "risk_penalty": 0.1, "min_utility": 0.0, "embargo_bars": embargo}


def _used_nodes(graph, entry, exit_ref, training):
    """Do not clutter an editable draft with unused helper-created branches."""
    roots = [entry]
    if exit_ref:
        roots.append(exit_ref)
    if training:
        roots.extend(training["state_refs"])
        roots.extend(action["entry"] for action in training["actions"] if action["entry"])
    used = set()

    def mark(reference):
        key = reference.split(".")[0]
        if key not in used:
            used.add(key)
            for upstream in graph._by_id[key]["inputs"].values():
                mark(upstream)

    for reference in roots:
        mark(reference)
    return [node for node in graph.nodes if node["id"] in used]


def _template(source, title, graph, entry, preserved, changed, training=None, exit_ref=None):
    description = "ETF 原生改编；保留研究思想，不代表原股票实验复现或收益承诺。"
    definition = {"schema_version": "1.0", "name": f"{title} · ETF 改编", "description": description,
                  "nodes": _used_nodes(graph, entry, exit_ref, training), "entry": entry, "exit": exit_ref,
                  "execution": {"take_profit": 0.15, "stop_loss": 0.15, "max_holding_bars": 15,
                                "cooldown_bars": 0, "fee_bps": 3.0, "slippage_bps": 5.0},
                  "adaptation": {"source_experiments": [source], "version": "etf-v1",
                                 "preserved": list(preserved), "changed": [
                                     "股票候选母池改为所选 ETF 的真实日频行情；不继承原实验评分。",
                                     "沿用择时中心 T+1、费用、缺失与交易日轴契约，不复制股票日内卖出标签。",
                                     *changed]}}
    if source != "A160S-106" and any(node["id"] == "position20" for node in definition["nodes"]):
        definition["adaptation"]["changed"].append("无振幅区间位置记0，不把恒价解释为强修复。")
    if training is not None:
        definition["training"] = training
        definition["adaptation"]["changed"].append(
            "仅训练区间内已成熟的 ETF 交易学习收益置信下界；样本不足或效用不达标留现金，样本外冻结。")
    return {"id": "etf_" + source.replace("-", "_"), "label": definition["name"],
            "description": description, "definition": definition}


def _repair():
    g = _Graph()
    price = g.source()
    previous = g.lag("previous", "前一日收盘", price)
    positive = g.compare("positive", "当日收盘上涨", price, right=previous)
    first = g.event("first_positive", "下跌或持平后的首个上涨日", positive)
    ret5 = g.lag("prior_return5", "截至前一日的 5 日收益", g.returns(5))
    prior_peak = g.lag("prior_peak20", "前 20 日收盘峰值", g.formula(
        "peak20", "20 日收盘峰值", "rolling_max(a,20,20)", price))
    drawdown = g.formula("prior_drawdown20", "前一日相对前 20 日峰值回撤", "a/b-1", previous, prior_peak)
    location = g.formula("close_position", "当日收盘位置（零振幅为 0.5）", "divide_or_default(a-b,c-b,0.5)", price, g.source("low"), g.source("high"))
    weak = g.compare("weak_ret5", "此前 5 日下跌门槛", ret5, "le", -0.03)
    deep = g.compare("weak_dd20", "此前 20 日回撤门槛", drawdown, "le", -0.08)
    strong = g.compare("close_strong", "收盘位置门槛", location, "ge", 0.5)
    background = g.combine("background", "首阳、下跌背景与收盘确认", first, weak, deep, strong)
    logs = g.formula("log_price", "对数收盘价", "log(a)", price)
    g.add("filter", "低延迟趋势滤波", "alpha_beta", {"value": logs}, alpha=0.5, beta=0.1)
    prior_slope = g.lag("prior_slope", "前一日趋势斜率", "filter.slope")
    up = g.compare("slope_positive", "斜率严格为正", "filter.slope")
    was_down = g.compare("slope_nonpositive", "前一日斜率不为正", prior_slope, "le")
    turn = g.combine("slope_turn", "趋势斜率由非正转正", up, was_down)
    acceleration = g.compare("slope_accel", "斜率较前一日改善", "filter.slope", right=prior_slope)
    innovation = g.compare("innovation_positive", "价格创新严格为正", "filter.innovation")
    accel = g.combine("accel_confirm", "斜率改善且价格创新为正", acceleration, innovation)
    entry = g.combine("entry_turn", "首阳与斜率转正", background, turn)
    alternate = g.combine("entry_accel", "首阳与斜率加速", background, accel)
    pairs = {"label": "Alpha-Beta 联动参数", "choices": [[
        {"node": "filter", "parameter": "alpha", "value": alpha},
        {"node": "filter", "parameter": "beta", "value": beta}]
        for alpha, beta in ((0.35, 0.05), (0.5, 0.1), (0.65, 0.15))]}
    search = [pairs, _search("前期 5 日收益上限", "weak_ret5", "threshold", [0.0, -0.03, -0.06]),
              _search("前期 20 日回撤上限", "weak_dd20", "threshold", [-0.03, -0.08, -0.12]),
              _search("收盘位置下限", "close_strong", "threshold", [0.5, 0.65])]
    training = _training([_action("turn", "斜率转正", entry), _action("accel", "斜率加速", alternate)], search=search, embargo=15)
    return _template("A2552", "首阳低延迟修复", g, entry,
                     ["首阳、T-1 下跌背景、对数 Alpha-Beta 与两种确认。", "完整 108 个规则组合；阈值等号保持。"],
                     ["原70/30日期切分评分改为中心成熟交易置信下界；不是原版优化器。", "缺失保持原交易轴并重置递推，不压缩日期。"], training)


def _reversal_features(g):
    price = g.source()
    high = g.source("high")
    low = g.source("low")
    peak60 = g.lag("prior_high60", "前 60 日最高价", g.formula("high60", "60 日最高价", "rolling_max(a,60,60)", high))
    peak20 = g.lag("prior_high20", "前 20 日最高价", g.formula("high20", "20 日最高价", "rolling_max(a,20,20)", high))
    trough20 = g.lag("prior_low20", "前 20 日最低价", g.formula("low20", "20 日最低价", "rolling_min(a,20,20)", low))
    dd = g.formula("drawdown60", "收盘相对前 60 日最高价回撤", "a/b-1", price, peak60)
    cp = g.formula("position20", "前 20 日区间位置（无振幅为0）", "divide_or_default(a-b,c-b,0)", price, trough20, peak20)
    weak = g.compare("deep_drawdown", "深度回撤门槛", dd, "le", -0.25)
    decline = g.compare("decline20", "20 日跌幅门槛", g.returns(20), "le", -0.10)
    recovery = g.compare("recovery_position", "区间修复位置门槛", cp, "ge", 0.5)
    confirmation = g.event("recovery_confirm", "区间位置连续两日确认", recovery, "confirm", 2)
    previous = g.lag("previous", "前一日收盘", price)
    opening = g.compare("opening_confirm", "当日开盘不低于昨收", g.source("open"), "ge", right=previous)
    return g.combine("reversal_base", "深跌后的连续区间修复", weak, decline, confirmation, opening)


def _volume_features(g):
    price, high, low, volume = g.source(), g.source("high"), g.source("low"), g.source("volume")
    position = g.formula("money_position", "收盘资金位置（零振幅中性）", "divide_or_default(2*a-b-c,b-c,0)", price, high, low)
    pressure = g.formula("money_pressure", "当日收盘量能压力", "a*b", position, volume)
    cmf = g.formula("cmf20", "20 日 CMF（零成交量为0）", "divide_or_default(rolling_mean(a,20,20),rolling_mean(b,20,20),0)", pressure, volume)
    typical = g.formula("typical", "典型价格", "(a+b+c)/3", high, low, price)
    prior = g.lag("prior_typical", "前一日典型价格", typical)
    up = g.number("typical_up_value", "上涨资金流标记", g.compare("typical_up", "典型价格上涨", typical, right=prior))
    down = g.number("typical_down_value", "下跌资金流标记", g.compare("typical_down", "典型价格下跌", typical, "lt", right=prior))
    money = g.formula("money", "典型价格乘成交量", "a*b", typical, volume)
    pos = g.formula("positive_money", "正向资金流", "a*b", money, up)
    neg = g.formula("negative_money", "负向资金流", "a*b", money, down)
    positive_flow = g.formula("positive_flow14", "14 日正向资金流均值", "rolling_mean(a,14,14)", pos)
    total_flow = g.formula("total_flow14", "14 日双向资金流均值", "rolling_mean(a+b,14,14)", pos, neg)
    return cmf, g.formula("mfi14", "14 日 MFI（无方向流量为50）", "100*divide_or_default(a,b,0.5)", positive_flow, total_flow)


def _reversal():
    g = _Graph()
    base = _reversal_features(g)
    weak_market = g.compare("weak_market", "ETF 篮子长期广度不超过 20%", g.breadth(250), "le", 0.2)
    context = g.combine("reversal_context", "弱市中的 ETF 连续修复", base, weak_market)
    cmf, mfi = _volume_features(g)
    soft = g.compare("cmf_soft", "CMF 不低于 -0.05", cmf, "ge", -0.05)
    positive = g.compare("cmf_positive", "CMF 不为负", cmf, "ge")
    flow = g.compare("mfi_confirm", "MFI 不低于 45", mfi, "ge", 45.0)
    both = g.combine("flow_confirm", "MFI 与 CMF 同时确认", flow, soft)
    actions = [_action(key, label, g.combine(key, label, context, ref)) for key, label, ref in (
        ("entry_cmf", "CMF 承接", soft), ("entry_both", "双资金流确认", both),
        ("entry_cmf_positive", "正向 CMF", positive), ("entry_mfi", "MFI 确认", flow))]
    search = [_search("60 日回撤上限", "deep_drawdown", "threshold", [-0.25, -0.22, -0.20]),
              _search("20 日收益上限", "decline20", "threshold", [-0.10, -0.09, -0.08]),
              _search("区间位置下限", "recovery_position", "threshold", [0.50, 0.47, 0.45])]
    return _template("A160S-106", "弱市量价反转", g, actions[0]["entry"],
                     ["前期最高/最低价定义、连续两日区间确认、开盘确认与四组量能过滤。", "108 个结构/量能候选组合。"],
                     ["全股票 MA250 广度改为用户指定 ETF 参考篮子的横截面广度。", "不沿用股票每日28只配额或未来整期零信号回退；零成交保持空仓。",
                      "零区间位置记0不形成强修复；CMF零振幅中性0，MFI无方向资金流为50、单边上涨为100。"],
                     _training(actions, search=search, embargo=15))


def _experts(g):
    price = g.source()
    positive20 = g.compare("momentum20_positive", "20 日收益为正", g.returns(20))
    above20 = g.compare("above_ma20", "收盘高于 20 日均线", price, right=g.ma(20))
    trend = g.combine("momentum_default", "短周期动量专家", positive20, above20)
    positive60 = g.compare("momentum60_positive", "60 日收益为正", g.returns(60))
    above60 = g.compare("above_ma60", "收盘高于 60 日均线", price, right=g.ma(60))
    slow = g.combine("momentum_alternative", "长周期动量专家", positive60, above60)
    reversal = _reversal_features(g)
    return [_action("momentum", "短周期动量", trend), _action("slow", "长周期动量", slow),
            _action("reversal", "深跌反转", reversal)]


def _market_states(g):
    positive = g.compare("market_positive", "市场篮子 60 日收益为正", g.basket_return(60))
    price = g.basket()
    bias = g.formula("market_biases120", "参考成员相对 MA120 偏离", "a/rolling_mean(a,120,120)-1", price, panel=True)
    bias_mean = g.add("market_bias120", "参考篮子等权 MA120 偏离", "cross_mean", {"value": bias})
    above = g.compare("market_above120", "市场篮子高于长期均价", bias_mean)
    bull = g.combine("market_bull", "市场趋势同向向上", positive, above)
    negative = g.compare("market_negative", "市场篮子 60 日收益为负", "market_return60.value", "lt")
    below = g.compare("market_below120", "市场篮子低于长期均价", bias_mean, "lt")
    bear = g.combine("market_bear", "市场趋势同向向下", negative, below)
    return [bull, bear]


def _router():
    g = _Graph()
    actions = _experts(g)
    states = _market_states(g)
    entry = g.combine("default_entry", "两类专家基础候选", actions[0]["entry"], actions[2]["entry"], op="any")
    return _template("A160-MOMO-141", "动量与反转状态路由", g, entry,
                     ["动量/反转专家互补、市场多空状态与标签成熟隔离。"],
                     ["原 A126/A102/A160S74 股票专家改为可展开的 ETF 短趋势、长趋势与反转专家。", "沪深300单指数改为市场参考篮子；原月份×状态层级/Beta硬门控改为状态内专家与现金选择。"],
                     _training(actions, mode="state", states=states, embargo=15))


def _quality_features(g):
    """Replace unavailable stock specialist factors with visible ETF features."""
    relative = g.formula("relative20", "ETF 相对同类篮子的 20 日收益", "a-b",
                         g.returns(20), g.basket_return(20, "category"))
    volatility = g.formula("volatility20", "20 日收益波动", "rolling_std(a,20,1,20)", g.returns(1))
    risk_scaled = g.formula("relative_risk_scaled", "相对收益与波动之比（零波动中性）", "divide_or_default(a,b,0)", relative, volatility)
    volume = g.source("volume")
    participation = g.formula("volume_ratio20", "成交量相对 20 日均量（无成交为0）", "divide_or_default(a,rolling_mean(a,20,20),0)", volume)
    pressure = g.formula("volume_deviation", "量能偏离惩罚", "abs(a-1)", participation)
    score = g.formula("quality_score", "相对强度减去量能偏离", "a-b", risk_scaled, pressure)
    quality = g.compare("quality_gate", "质量分门槛", score, "ge", 0.0)
    liquidity = g.compare("liquidity_gate", "成交参与不低于半数均量", participation, "ge", 0.5)
    return quality, liquidity


def _confidence_router(source, gated):
    g = _Graph()
    experts = _experts(g)
    quality, liquidity = _quality_features(g)
    gate = g.combine("confidence_gate", "相对质量与交易参与确认", quality, liquidity)
    selected = g.combine("qualified_momentum", "高置信 ETF 动量候选", experts[0]["entry"], gate)
    states = _market_states(g) if gated else []
    training = _training([_action("quality", "高置信动量", selected)],
                         mode="state" if gated else "global", states=states,
                         search=[_search("质量过滤强度", "quality_gate", "threshold", [-0.5, 0.0, 0.5])], embargo=15)
    return _template(source, "状态内高置信动量" if gated else "高置信动量筛选", g, selected,
                     ["质量筛选后再训练选择；规则与训练测试边界分离。",
                      "缺少合格历史状态时不强制发出买入信号。" if gated else "只在训练区间选择过滤强度。"],
                     ["原股票五特征/median-IQR/同日 Top5/10/15% 配额改为 ETF 相对篮子强度、波动与量能过滤。",
                      "原市场/行业条件桶改为显式 ETF 篮子状态，合格性按共同训练效用判定。" if gated
                      else "候选是同一 ETF 不同时点，不宣称已实现原跨股票日内排名。"], training)


def _state_actions():
    g = _Graph()
    experts = _experts(g)
    quality, liquidity = _quality_features(g)
    clean = g.combine("quality_entry", "质量保护动作", experts[0]["entry"], quality, liquidity)
    slower = g.event("reduced_entry", "五日冷却降频动作", experts[0]["entry"], "cooldown", 5)
    actions = [_action("pass", "中性通过", experts[0]["entry"]),
               _action("protect", "质量保护", clean), _action("reduce", "降频", slower)]
    return _template("A2536-Momo-053", "保护与降频状态学习", g, experts[0]["entry"],
                     ["明确区分保护、中性、降频与屏蔽，并只在成熟训练数据上选择。"],
                     ["原 A050 母池及11个状态维度改为 ETF 动量/量能与市场篮子多空状态。",
                      "原加权 protect/downweight/gate 规则改为可编辑动作，屏蔽用现金表示。"],
                     _training(actions, mode="state", states=_market_states(g), embargo=15))


def _action_learner():
    g = _Graph()
    actions = _experts(g)
    return _template("ActionLearner-001", "专家与现金动作学习", g, actions[0]["entry"],
                     ["比较多个专家与现金，采用成熟收益、最少样本与保守效用约束。"],
                     ["原 A160S/Momo050/RD012 改为显式 ETF 短趋势、长趋势与深跌反转。",
                      "原训练三分位状态、先验收缩及70/30验证改为显式多空状态与中心训练冻结。",
                      "成熟时点使用真实退出加隔离期，不使用原40自然日代理。"],
                     _training(actions, mode="state", states=_market_states(g), embargo=15))


def _core_and_supplement(g):
    experts = _experts(g)
    quality, liquidity = _quality_features(g)
    core = g.combine("clean_core", "质量确认后的核心动量", experts[0]["entry"], quality, liquidity)
    supplement = g.combine("clean_supplement", "交易参与确认后的反转补位", experts[2]["entry"], liquidity)
    return core, supplement


def _priority(g, core, supplement, key="priority_entry", max_core=12, monthly=4):
    return g.add(key, "主信号优先，有限补位", "priority_quota", {"core": core, "supplement": supplement},
                 max_core_before_supp=max_core, max_supp_per_month=monthly, cooldown=5)


def _fallback_router():
    g = _Graph()
    core, supplement = _core_and_supplement(g)
    entry = _priority(g, core, supplement)
    return _template("A2067", "核心优先与限额补位", g, entry,
                     ["核心信号优先；仅按当时已发信号数量判断是否允许补位。", "月内主信号少于12才补位，补位每月最多4次。"],
                     ["原已清洁股票专家改为 ETF 动量核心与反转补位。", "单产品日频天然每天至多一次；五交易日补位冷却为 ETF 研究默认，不是跨股票配额。"])


def _calendar_router(source, quarterly):
    g = _Graph()
    core, supplement = _core_and_supplement(g)
    actions = [_action("core", "核心来源", core), _action("replacement", "整期替代来源", supplement)]
    return _template(source, "同季整期来源替代" if quarterly else "同月整期来源替代", g, core,
                     ["同自然月/季成熟历史评估，整期固定来源或现金；不为填满信号而补位。"],
                     ["原股票替代源改为 ETF 动量/反转；不继承原信号母池。",
                      "原胜率/收益/止损/路径坏样本硬阈值改为样本数和保守效用约束。",
                      "只用训练内同自然月或季成熟交易，整个样本外冻结；不是每到未来月份再偷看当月结果。"],
                     _training(actions, mode="quarter" if quarterly else "month", embargo=15))


def _market_structure(g):
    """Six observable environment votes, explicit ETF proxies for WF states."""
    breadth = g.breadth(60)
    support = g.compare("breadth_support", "MA60 广度至少 35%", breadth, "ge", 0.35)
    g.basket_return(1)
    up = g.compare("members_up", "参考成员当日上涨", "market_returns1.value", panel=True)
    up_share = g.add("up_share", "上涨成员占比", "breadth", {"value": up})
    broad_up = g.compare("up_support", "上涨占比至少 45%", up_share, "ge", 0.45)
    ret1 = g.compare("return1_nonnegative", "篮子单日收益非负", "market_return1.value", "ge")
    ret3 = g.compare("return3_nonnegative", "篮子三日收益非负", g.basket_return(3), "ge")
    trend = g.combine("market_trend_up", "短周期市场趋势改善", ret1, ret3)
    price = g.basket()
    stress_refs = []
    for window, threshold in ((20, 0.18), (60, 0.12)):
        prior_low = g.lag(f"market_prior_low{window}", f"成员前 {window} 日收盘低点", g.formula(
            f"market_low{window}", f"成员 {window} 日收盘低点", f"rolling_min(a,{window},{window})", price, panel=True), panel=True)
        new_low = g.compare(f"members_new_low{window}", f"成员触及前 {window} 日收盘低点", price, "le", right=prior_low, panel=True)
        share = g.add(f"new_low_share{window}", f"{window} 日新低成员占比", "breadth", {"value": new_low})
        stress_refs.append(g.compare(f"stress{window}", f"{window} 日新低压力门槛", share, "ge", threshold))
    stress = g.combine("market_stress", "新低扩散压力", *stress_refs, op="any")
    no_stress = g.invert("no_market_stress", "没有新低扩散压力", stress)
    prior_breadth = g.lag("prior_breadth3", "三日前 MA60 广度", breadth, 3)
    change = g.formula("breadth_change3", "三日广度改善", "a-b", breadth, prior_breadth)
    repair = g.compare("breadth_repair", "三日广度回升至少5个百分点", change, "ge", 0.05)
    prior_high = g.lag("market_prior_high20", "成员前 20 日收盘高点", g.formula(
        "market_high20", "成员 20 日收盘高点", "rolling_max(a,20,20)", price, panel=True), panel=True)
    new_high = g.compare("members_new_high20", "成员突破前 20 日收盘高点", price, right=prior_high, panel=True)
    high_share = g.add("new_high_share20", "20 日新高成员占比", "breadth", {"value": new_high})
    prior_high_share = g.lag("prior_high_share3", "三日前新高占比", high_share, 3)
    high_delta = g.formula("new_high_delta3", "三日新高占比改善", "a-b", high_share, prior_high_share)
    high_revival = g.compare("new_high_revival", "新高占比改善超过1个百分点", high_delta, "gt", 0.01)
    swing = g.combine("swing_repair", "广度或新高结构修复", repair, high_revival, op="any")
    votes = [g.number(f"vote_{key}", label + "（计1分）", ref) for key, label, ref in (
        ("trend", "短期趋势改善", trend), ("breadth", "长期广度支撑", support),
        ("up", "上涨扩散", broad_up), ("stress", "无新低扩散", no_stress),
        ("repair", "广度修复", repair), ("swing", "结构修复", swing))]
    first = g.formula("state_score_first", "环境前三项得分", "a+b+c", *votes[:3])
    last = g.formula("state_score_last", "环境后三项得分", "a+b+c", *votes[3:])
    score = g.formula("state_score", "六项市场环境得分", "a+b", first, last)
    healthy = g.compare("state_healthy", "环境得分达到门槛", score, "ge", 3.0)
    repair_score = g.compare("repair_score", "修复重入至少两分", score, "ge", 2.0)
    recovery = g.combine("market_recovery", "环境修复且具备最低支撑", repair, repair_score)
    allowed = g.combine("market_allowed", "健康环境或修复通道", healthy, recovery, op="any")
    return {"score": score, "stress": stress, "repair": repair, "swing": swing,
            "healthy": healthy, "recovery": recovery, "allowed": allowed}


_STRUCTURE_CHANGES = [
    "全股票环境改为固定 ETF 篮子，广度缺失不重新加权。",
    "LLT、冰点与原修复分改为篮子1/3日收益、MA60广度三日回升和收盘新高扩散；不是原环境公式。",
    "新高/新低以篮子收盘价计算，不冒用原股票高低价、Hurst或外部状态。",
]


def _environment_gate():
    g = _Graph()
    core, supplement = _core_and_supplement(g)
    market = _market_structure(g)
    candidates = g.combine("candidate_union", "动量与反转候选", core, supplement, op="any")
    entry = g.combine("environment_entry", "市场环境门控后的候选", candidates, market["allowed"])
    return _template("A2140", "独立市场环境门控", g, entry,
                     ["六项状态计分；高分直接通过，修复且最低两分可走恢复通道。"], _STRUCTURE_CHANGES,
                     _training([_action("environment", "环境门控", entry)],
                               search=[_search("环境通过分数", "state_healthy", "threshold", [2.0, 3.0, 4.0])], embargo=15))


def _risk_reentry():
    g = _Graph()
    core, supplement = _core_and_supplement(g)
    market = _market_structure(g)
    unrepaired = g.invert("not_repaired", "尚无广度修复", market["repair"])
    pressure = g.combine("unrepaired_pressure", "压力扩散且尚未修复", market["stress"], unrepaired)
    weak = g.invert("weak_state", "环境得分不足", market["healthy"])
    risk = g.combine("risk_state", "弱态或未修复压力", weak, pressure, op="any")
    safe = g.invert("safe_state", "非风险环境", risk)
    ordinary = g.combine("ordinary_entry", "非风险时核心通过", core, safe)
    reentry = g.combine("risk_reentry", "风险时仅清洁反转和修复重入", supplement, risk, market["recovery"])
    entry = g.combine("controlled_entry", "正常通道或受控重入", ordinary, reentry, op="any")
    return _template("A2143", "风险态隔离与修复重入", g, entry,
                     ["正常态核心通过，风险态只保留质量确认且环境修复的重入。"],
                     [*_STRUCTURE_CHANGES, "原 A2086 风险母池与 clean 来源替换为可见的 ETF 风险条件和量价质量确认。"],
                     _training([_action("controlled", "风险隔离", entry)],
                               search=[_search("环境通过分数", "state_healthy", "threshold", [2.0, 3.0, 4.0])], embargo=15))


def _risk_budget_router():
    g = _Graph()
    core, supplement = _core_and_supplement(g)
    market = _market_structure(g)
    clean = g.combine("clean_repair", "有环境修复确认的补位", supplement, market["recovery"])
    capped = _priority(g, core, clean, max_core=12, monthly=4)
    states = [market["stress"], market["healthy"]]
    return _template("A2276", "状态风控与有限补位", g, capped,
                     ["同态历史决定信号是否值得保留；弱态补位须清洁确认并受配额约束。"],
                     [*_STRUCTURE_CHANGES, "原同态路径坏样本/止损/胜率硬门槛改为真实成熟 ETF 交易效用。",
                      "原多来源日/月配额改为每只 ETF 每月最多4次修复补位，不代表跨产品组合配额。"],
                     _training([_action("core", "仅核心", core), _action("limited", "允许有限修复补位", capped)],
                               mode="state", states=states, embargo=15))


def _negative_veto():
    g = _Graph()
    core, supplement = _core_and_supplement(g)
    market = _market_structure(g)
    sources = g.combine("source_union", "质量确认后的来源并集", core, supplement, op="any")
    negative_return = g.compare("negative_return", "产品单日跌幅超过3%", g.returns(1), "lt", -0.03)
    below = g.compare("below_ma20", "产品低于20日均线", g.source(), "lt", right=g.ma(20))
    negative = g.combine("product_negative", "产品下跌冲击否决", negative_return, below)
    release = g.combine("pressure_release", "新低压力中的短期修复事件", market["stress"], market["repair"])
    veto = g.combine("combined_veto", "产品负信号或压力释放否决", negative, release, op="any")
    clear = g.invert("veto_clear", "没有已知否决", veto)
    entry = g.combine("vetoed_entry", "来源候选经负信号否决", sources, clear)
    return _template("A2296", "强来源合并与负信号否决", g, entry,
                     ["先构造质量来源，再做产品负信号或市场压力事件的 OR 否决。", "压力释放只作否决，不冒作买点。"],
                     [*_STRUCTURE_CHANGES, "原 A2295 同股同日负键与 HMM 审计日期改为可编辑 ETF 下跌冲击/篮子压力修复条件。",
                      "这里没有 HMM 模型或原股票来源优先表；单产品同日布尔并集天然去重。"])


def etf_templates() -> list[dict]:
    """Return independent JSON-compatible drafts; no persistent/shared mutation."""
    return [_repair(), _reversal(), _router(), _confidence_router("A2536-Momo-095", False),
            _confidence_router("A2536-Momo-096", True), _state_actions(), _action_learner(),
            _fallback_router(), _calendar_router("A2074", False), _calendar_router("A2076", True),
            _environment_gate(), _risk_reentry(), _risk_budget_router(), _negative_veto()]
