"""Explicit local capabilities and immutable factor templates."""
ENGINE_VERSION = "factor-research-njit-2.0.0"
OPERATORS = {"momentum": 0, "volatility": 1, "drawdown": 2, "reversal": 3}
BUILTINS = [
    {"id": "factor-momentum-126-21", "revision": 1, "name": "中期动量（跳过短期）",
     "operator": "momentum", "window": 126, "skip": 21, "direction": 1,
     "description": "过去126个交易日收益，跳过最近21日；观察中期趋势。"},
    {"id": "factor-low-volatility-63", "revision": 1, "name": "低波动",
     "operator": "volatility", "window": 63, "skip": 0, "direction": -1,
     "description": "63日收益样本标准差年化，低波动获得较高分。"},
    {"id": "factor-drawdown-126", "revision": 1, "name": "回撤控制",
     "operator": "drawdown", "window": 126, "skip": 0, "direction": 1,
     "description": "126日窗口最大回撤（负数），越接近零得分越高。"},
    {"id": "factor-reversal-21", "revision": 1, "name": "短期反转",
     "operator": "reversal", "window": 21, "skip": 0, "direction": 1,
     "description": "最近21日收益的相反数；是待验证假设，不预设有效。"},
]
for _item in BUILTINS:
    _item.update(product_kinds=["etf", "fund"], read_only=True,
                 source="builtin", engine_version=ENGINE_VERSION)
DEFAULT_TARGETS = [
    "510050.SH", "510300.SH", "510500.SH", "512100.SH",
    "159915.SZ", "159949.SZ", "512010.SH", "512480.SH",
    "512660.SH", "512690.SH", "512800.SH", "512880.SH",
]
MODELS = [
    {"id": "characteristic_composite", "name": "特征因子组合", "purpose": "selection", "available": True},
    {"id": "rbsa", "name": "收益风格分析 RBSA", "purpose": "attribution", "available": True},
    {"id": "factor_regression", "name": "通用因子收益回归", "purpose": "attribution", "available": True},
    {"id": "ff3", "name": "Fama–French 三因子", "purpose": "attribution",
     "available": True, "requires": "匹配市场、币种的日频 MKT_RF / SMB / HML / RF 数据集"},
]
CONTEXTS = [
    {"id": "product_research", "name": "产品研究", "path": "/product-research/evaluation"},
    {"id": "saa", "name": "战略资产配置", "path": "/pre-investment/saa/allocation-lab"},
    {"id": "taa", "name": "战术资产配置", "path": "/pre-investment/taa"},
    {"id": "allocation", "name": "类内产品配置", "path": "/pre-investment/product-allocation-timing/construction"},
    {"id": "regime", "name": "历史情景研究", "path": "/settings/scenario-algorithms/workbench"},
    {"id": "portfolio", "name": "组合版本", "path": "/portfolio-center"},
    {"id": "post_investment", "name": "投后复核", "path": "/post-investment/research-diagnosis"},
]
