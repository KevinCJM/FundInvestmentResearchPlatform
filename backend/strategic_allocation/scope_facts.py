"""Economic scope equality, independent of saved-record identity and annotations."""
from backend.sensitivity.repository import digest_json


def proxy_facts(proxy):
    if proxy is None:
        return None
    components = sorted([[c.get(k) for k in ("kind", "series_id", "field")] + [float(c["weight"])]
                         for c in proxy.get("components", [])])
    cash = proxy.get("cash_return")
    return [proxy.get("asset_type"), (float(cash) or 0.) if cash is not None else None, proxy.get("rebalance"), components]


def proxy_difference(left, right):
    if left == right:
        return None
    if left is None or right is None or left[0] != right[0]:
        return "scopeProxyType"
    if left[1] != right[1]:
        return "scopeCashReturn"
    if left[2] != right[2]:
        return "scopeRebalance"
    if [c[:3] for c in left[3]] != [c[:3] for c in right[3]]:
        return "scopeProxySource"
    return "scopeProxyWeights"


def research_proxy_difference(left, right):
    if left is None or right is None:
        return None  # A manual prior has no fitted research proxy to compare.
    if [a[0] for a in left] != [a[0] for a in right]:
        return "scopeAssets"
    return next((issue for a, b in zip(left, right) if (issue := proxy_difference(a[1], b[1]))), None)


def scope_facts(definition):
    assets = definition["assets"]
    return {"currency": definition["currency"], "asset_ids": [a["id"] for a in assets],
            "asset_currencies": [a["currency"] for a in assets],
            "roles": [a["role"] for a in assets], "liquidities": [a["liquidity"] for a in assets],
            "proxies": [proxy_facts(a.get("research_proxy")) for a in assets]}


def scope_fingerprint(definition):
    return digest_json({"contract": "strategic_scope_facts_v1", **scope_facts(definition)})


def scope_weight_limits(assets):
    """Class bounds declared on a strategic scope, keyed by asset id."""
    return {a["id"]: a["weight_limits"] for a in assets if a.get("weight_limits")
            and a["weight_limits"] != {"min_weight": 0, "max_weight": 1}}


def cma_scope_weight_limits(item):
    snapshot = item.get("source_snapshot", {}).get("strategic_universe_snapshot")
    return scope_weight_limits(snapshot["definition"]["assets"]) if snapshot else {}


def cma_scope_facts(item):
    snapshot = item.get("source_snapshot", {}).get("strategic_universe_snapshot")
    return scope_facts(snapshot["definition"]) if snapshot else None


def research_proxy_facts(model):
    inputs = (model or {}).get("proxy_inputs")
    return [[a["id"], proxy_facts(a)] for a in inputs["assets"]] if inputs else None


def scope_difference(left, right):
    if left is None or right is None:
        return "scopeFactsMissing"
    for keys, reason in ((["currency", "asset_currencies"], "scopeCurrency"),
                         (["asset_ids"], "scopeAssets"), (["roles"], "scopeRoles"),
                         (["liquidities"], "scopeLiquidity")):
        if any(left.get(key) != right.get(key) for key in keys):
            return reason
    if len(left['proxies']) != len(right['proxies']):
        return 'scopeProxyType'
    return next((issue for a, b in zip(left['proxies'], right['proxies']) if (issue := proxy_difference(a, b))), None)


def cma_scope_difference(left, right):
    a, b = left["definition"], right["definition"]
    if bool(a.get("strategic_universe_id")) != bool(b.get("strategic_universe_id")):
        return "scopeAssets"
    if not a.get("strategic_universe_id"):
        return None if a.get("alloc_name") == b.get("alloc_name") else "scopeAssets"
    facts_a, facts_b = cma_scope_facts(left), cma_scope_facts(right)
    if facts_a is None or facts_b is None:
        return None if a.get("strategic_universe_id") == b.get("strategic_universe_id") else "scopeFactsMissing"
    return scope_difference(facts_a, facts_b)


SCOPE_MESSAGES = {
    "scopeFactsMissing": "缺少可核验的资产配置事实，无法确认两个范围一致。",
    "scopeCurrency": "资产范围的计价币种不同。",
    "scopeAssets": "资产或资产排列不同。",
    "scopeRoles": "资产的经济角色不同。",
    "scopeLiquidity": "资产的流动性定义不同。",
    "scopeProxyType": "研究代理配置缺失或资产类型不同。",
    "scopeCashReturn": "现金收益率不同。",
    "scopeRebalance": "研究代理的再平衡规则不同。",
    "scopeProxySource": "研究代理的来源或价格字段不同。",
    "scopeProxyWeights": "研究代理的成分权重不同。",
}
