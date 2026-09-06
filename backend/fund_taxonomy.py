from __future__ import annotations

"""Three-level contract taxonomy for A-share public funds and ETFs.

The tables below follow the classification practice the domestic industry has
converged on:

* level 1 -- asset class (权益/固收/商品/货币/海外/混合), the 银河证券公募基金
  分类体系 top level and the layer SAA actually allocates across;
* level 2 -- for equity the four-way 规模(宽基)/风格因子/行业/主题 split that
  易方达 standardised in 2025, for bonds the 利率债/信用债/可转债/同业存单 split
  the exchanges use, plus 港股/美股/其他海外 offshore and the commodity buckets;
* level 3 -- the eight industry buckets (金融/医药/科技/消费/制造/周期/公用事业/
  房地产), the six theme buckets (数字化与人工智能/高端制造/低碳转型/国企改革/
  人口趋势/ESG), the Smart-Beta factors, the size/board buckets and the bond
  issuer buckets.

Matching is ordered keyword containment over the product's contract metadata,
first row wins.  This is deliberately a lookup table, not a model: the contract
label has to be reproducible and explainable, and it is the layer that keeps a
gold ETF out of an equity class no matter what a correlation window says.
"""

from typing import Iterable, NamedTuple, Optional


TAXONOMY_LEVELS = ("asset_class", "category", "detail")
TAXONOMY_LEVEL_LABELS = {
    "asset_class": "一级·资产大类",
    "category": "二级·细分类型",
    "detail": "三级·风格/行业/主题",
}
FALLBACK_ASSET_CLASS = "其他类"


class TaxonomyLabel(NamedTuple):
    asset_class: str
    category: str
    detail: str
    matched: str

    def at(self, level: str) -> str:
        if level == "asset_class":
            return self.asset_class
        if level == "category":
            return self.category
        if level == "detail":
            return self.detail
        raise ValueError(f"不支持的分类层级：{level}")

    @property
    def path(self) -> str:
        parts = [self.asset_class]
        if self.category != self.asset_class:
            parts.append(self.category)
        if self.detail != self.category:
            parts.append(self.detail)
        return " / ".join(parts)


# Ordered level-1 table; the first matching row wins.  Sector-equity wording that
# reuses a commodity word (黄金股, 有色金属) is caught before the commodity row,
# and every bond word is caught before the equity row so 科创债 does not read as
# 科创板.
_ASSET_CLASS_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("货币类", ("货币", "现金管理", "现金添益")),
    # Equity sector funds that borrow commodity vocabulary.
    ("权益类", ("黄金股", "黄金产业", "有色金属", "稀土", "煤炭", "石油石化", "油气开采")),
    ("商品类", ("黄金", "白银", "贵金属", "原油", "能源化工", "豆粕", "农产品期货", "商品期货", "大宗商品")),
    (
        "海外类",
        (
            "QDII", "恒生", "港股", "香港", "纳斯达克", "标普", "道琼斯", "日经", "德国", "法国",
            "中概", "海外", "亚太", "全球", "美国", "越南", "印度", "沙特", "东南亚", "新兴市场",
        ),
    ),
    ("固收类", ("债", "利率", "信用", "同业存单", "存单", "短融", "政金", "国开", "农发")),
    (
        "权益类",
        (
            "股票", "指数", "沪深", "中证", "上证", "深证", "创业板", "科创", "北证", "国证",
            "红利", "股息", "价值", "成长", "低波", "质量", "动量", "等权", "基本面", "自由现金流",
            "银行", "券商", "证券", "保险", "非银", "金融",
            "医药", "医疗", "生物", "创新药", "中药", "疫苗",
            "半导体", "芯片", "电子", "计算机", "软件", "通信", "传媒", "游戏", "信创", "科技",
            "消费", "白酒", "食品", "饮料", "家电", "农业", "养殖", "旅游", "零售", "美容",
            "机械", "装备", "军工", "国防", "电力设备", "汽车", "电池", "光伏", "风电", "机器人",
            "化工", "钢铁", "建材", "资源", "采掘",
            "电力", "公用", "环保", "水务", "燃气", "运输", "港口", "机场", "高速",
            "地产", "物业", "建筑", "基建",
            "人工智能", "数字经济", "算力", "新能源", "碳中和", "低碳", "绿色", "ESG",
            "央企", "国企", "一带一路", "专精特新", "养老", "红利低波",
        ),
    ),
    ("混合类", ("混合", "平衡", "灵活配置", "FOF", "偏股", "偏债")),
)

# Level 2 + level 3 in one ordered pass per asset class: (category, detail, keywords).
# Within 权益类 the order is factor -> theme -> industry -> size, because a
# 「300红利低波」 is a factor product that happens to name an index, and a
# 「人工智能」 fund is a cross-industry theme rather than a 科技 sector fund.
_SUBCLASS_RULES: dict[str, tuple[tuple[str, str, tuple[str, ...]], ...]] = {
    "权益类": (
        ("风格因子", "红利", ("红利", "股息", "高股息")),
        ("风格因子", "低波", ("低波", "低波动", "最小方差")),
        ("风格因子", "价值", ("价值",)),
        ("风格因子", "成长", ("成长",)),
        ("风格因子", "质量", ("质量", "优质", "盈利")),
        ("风格因子", "动量", ("动量", "趋势")),
        ("风格因子", "等权", ("等权",)),
        ("风格因子", "基本面", ("基本面",)),
        ("风格因子", "自由现金流", ("自由现金流", "现金流")),
        ("主题", "数字化与人工智能", ("人工智能", "数字经济", "算力", "数据中心", "云计算", "大数据", "信创", "软件服务")),
        ("主题", "高端制造", ("高端制造", "机器人", "工业母机", "专精特新", "智能制造", "先进制造")),
        ("主题", "低碳转型", ("碳中和", "低碳", "绿色", "新能源", "光伏", "风电", "储能", "氢能", "电池")),
        ("主题", "国企改革", ("央企", "国企", "国资", "一带一路")),
        ("主题", "人口趋势", ("养老", "银发", "生育", "母婴", "医疗服务")),
        ("主题", "ESG", ("ESG", "可持续", "社会责任")),
        ("行业", "金融", ("银行", "券商", "证券", "保险", "非银", "金融")),
        ("行业", "医药", ("医药", "医疗", "生物", "创新药", "中药", "疫苗", "器械")),
        ("行业", "科技", ("半导体", "芯片", "电子", "计算机", "通信", "传媒", "游戏", "科技")),
        ("行业", "消费", ("消费", "白酒", "食品", "饮料", "家电", "农业", "养殖", "旅游", "零售", "美容", "纺织")),
        ("行业", "制造", ("机械", "装备", "军工", "国防", "电力设备", "汽车", "航空", "船舶")),
        ("行业", "周期", ("有色", "煤炭", "石油", "石化", "化工", "钢铁", "建材", "稀土", "资源", "采掘", "黄金股", "黄金产业")),
        ("行业", "公用事业", ("电力", "公用", "环保", "水务", "燃气", "运输", "港口", "机场", "高速")),
        ("行业", "房地产", ("地产", "物业", "建筑", "基建")),
        ("宽基规模", "科创板", ("科创",)),
        ("宽基规模", "创业板", ("创业板", "创业")),
        ("宽基规模", "北证", ("北证", "北交所")),
        # Longest index names first: 「中证1000」 also contains 「中证100」.
        ("宽基规模", "小盘", ("中证1000", "中证2000", "国证2000", "小盘", "微盘")),
        ("宽基规模", "中盘", ("中证500", "中证200", "中盘")),
        ("宽基规模", "大盘", ("沪深300", "上证50", "上证180", "中证A50", "中证A100", "中证100", "中证800", "深证100", "深证300", "中证A500", "大盘")),
        ("宽基规模", "全市场", ("中证全指", "全指", "万得全A", "中证A股", "全市场", "综合指数", "上证综指")),
    ),
    "固收类": (
        ("可转债", "可转债", ("转债",)),
        ("同业存单", "同业存单", ("同业存单", "存单")),
        ("利率债", "国债", ("国债",)),
        ("利率债", "政金债", ("政金", "国开", "农发", "进出口债")),
        ("利率债", "地方债", ("地方债", "地方政府")),
        ("利率债", "利率债", ("利率",)),
        ("信用债", "城投债", ("城投",)),
        ("信用债", "科创债", ("科创债", "科技创新债")),
        ("信用债", "短融", ("短融", "短期融资")),
        ("信用债", "公司债", ("公司债",)),
        ("信用债", "企业债", ("企业债",)),
        ("信用债", "信用债", ("信用", "做市")),
    ),
    "商品类": (
        ("贵金属", "黄金", ("黄金",)),
        ("贵金属", "白银", ("白银",)),
        ("贵金属", "贵金属", ("贵金属",)),
        ("能源", "原油", ("原油", "石油期货")),
        ("能源", "能源化工", ("能源化工",)),
        ("农产品", "豆粕", ("豆粕",)),
        ("农产品", "农产品", ("农产品", "大豆", "玉米", "白糖")),
        ("有色", "有色金属期货", ("有色金属期货", "铜期货", "铝期货")),
    ),
    "海外类": (
        ("港股", "港股红利", ("恒生红利", "港股通红利", "港股红利", "恒生高股息")),
        ("港股", "港股科技", ("恒生科技", "恒生互联网", "中概互联", "港股科技", "港股互联网")),
        ("港股", "港股医药", ("恒生医疗", "港股创新药", "恒生生物", "港股医药")),
        ("港股", "港股消费", ("恒生消费", "港股消费")),
        ("港股", "港股金融", ("恒生金融", "港股金融", "恒生银行")),
        ("港股", "港股宽基", ("恒生", "港股", "香港")),
        ("美股", "纳斯达克", ("纳斯达克",)),
        ("美股", "标普", ("标普",)),
        ("美股", "道琼斯", ("道琼斯",)),
        ("美股", "美股宽基", ("美国", "中概")),
        ("其他海外", "日本", ("日经", "日本")),
        ("其他海外", "欧洲", ("德国", "法国", "欧洲", "欧元区")),
        ("其他海外", "新兴市场", ("越南", "印度", "沙特", "东南亚", "新兴市场")),
        ("其他海外", "全球", ("全球", "亚太", "海外")),
    ),
    "混合类": (
        ("混合类", "偏股混合", ("偏股",)),
        ("混合类", "偏债混合", ("偏债",)),
        ("混合类", "平衡混合", ("平衡",)),
        ("混合类", "FOF", ("FOF",)),
        ("混合类", "灵活配置", ("灵活配置",)),
    ),
}

# Category used when an asset class matched but no level-2 row did.  Naming the
# residual honestly beats forcing a product into a bucket it did not match.
_DEFAULT_CATEGORY = {
    "权益类": "宽基规模",
    "固收类": "综合债",
    "商品类": "商品",
    "海外类": "其他海外",
    "货币类": "货币",
    "混合类": "混合类",
    FALLBACK_ASSET_CLASS: FALLBACK_ASSET_CLASS,
}


def classify_text(haystack: str) -> TaxonomyLabel:
    """Classify one already-joined contract description."""

    text = (haystack or "").upper()
    asset_class = FALLBACK_ASSET_CLASS
    matched = ""
    for label, keywords in _ASSET_CLASS_RULES:
        hit = next((keyword for keyword in keywords if keyword.upper() in text), None)
        if hit is not None:
            asset_class, matched = label, hit
            break
    category = _DEFAULT_CATEGORY.get(asset_class, asset_class)
    detail = category
    for sub_category, sub_detail, keywords in _SUBCLASS_RULES.get(asset_class, ()):
        hit = next((keyword for keyword in keywords if keyword.upper() in text), None)
        if hit is not None:
            category, detail = sub_category, sub_detail
            matched = f"{matched}+{hit}" if matched and hit != matched else hit
            break
    return TaxonomyLabel(asset_class, category, detail, matched)


def classify(fields: Iterable[Optional[str]]) -> TaxonomyLabel:
    """Classify a product from its contract fields, most specific field first."""

    return classify_text(" ".join(str(field) for field in fields if field))


def taxonomy_tree() -> list[dict[str, object]]:
    """The published taxonomy, for the UI hint and for the meta endpoint."""

    tree: list[dict[str, object]] = []
    for asset_class, _ in _ASSET_CLASS_RULES:
        if any(item["asset_class"] == asset_class for item in tree):
            continue
        categories: dict[str, list[str]] = {}
        for category, detail, _keywords in _SUBCLASS_RULES.get(asset_class, ()):
            bucket = categories.setdefault(category, [])
            if detail not in bucket:
                bucket.append(detail)
        if not categories:
            categories[_DEFAULT_CATEGORY.get(asset_class, asset_class)] = []
        tree.append({
            "asset_class": asset_class,
            "categories": [
                {"category": category, "details": details}
                for category, details in categories.items()
            ],
        })
    return tree


__all__ = [
    "FALLBACK_ASSET_CLASS",
    "TAXONOMY_LEVELS",
    "TAXONOMY_LEVEL_LABELS",
    "TaxonomyLabel",
    "classify",
    "classify_text",
    "taxonomy_tree",
]
