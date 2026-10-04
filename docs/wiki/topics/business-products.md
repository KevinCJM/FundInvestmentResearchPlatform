---
id: "business-products"
type: "topic"
title: "怎样研究产品，并把大类预算落实为产品方案？"
domains: ["business"]
review_state: "partial"
result: "supported"
scope: "本页有源业务概述；当前实现、目标设计及历史证据分别解释；含指定源码静态核对，不代表全模块金融验证、真实投资或生产部署通过"
reviewed_at: "2026-10-04"
reviewed_by: "AI 原文与实现核对"
source_revision: "8ce7c06cc655d027cec6856caa873a1eafa1e51d"
dependencies: ["docs/product-research/README.md::sha256:a84331d90a81b657036ac1efbb0aa8ca8bc023cedaa78260984955c995b45b9f", "docs/product-research/trend-chart.md::sha256:cc39990081fb9ea00b962f0a2f30d353c4a5ad6a19675adac85e6c750ddc79aa", "docs/product-research/scenario-research.md::sha256:90793f8c9c6bb8eb4aaf7c76c970941cfd99858b0d361d2076ec4630334de785", "docs/indicators/README.md::sha256:b6ab4622f2dcb0dc08b2bdfccc5d6feec0125bc34f2f97770ec35fb3964220a9", "docs/factor-research.md::sha256:e1f1a7261e8d082ea8df1f1920c69b8bf594b7a67416ba7cd5d966f8901daca1", "docs/product-research/timing.md::sha256:3a1e503fcde4d9d1676f5700051168aab76937a3bd27a669b41105c2f7ec8e88", "docs/pre-investment/implementation.md::sha256:1c9ba2c90ed50b08f7219bb9908ef71b40cf2de0447b19976f492a05d34b7394", "backend/pre_investment/risk.py::sha256:8ff99c2eb17386515b98a6bc3cca4387b63f01ab3223e2092ef8abe3b9282d02", "docs/pre-investment/funding.md::sha256:7f93a3cc24bcac190c822d15e4707b56133e726de677bf2856f89438494971da"]
evidence_kind: "mixed"
aliases: ["产品研究", "产品实施", "产品池"]
---

# 怎样研究产品，并把大类预算落实为产品方案？

## 这页回答什么

连接持续的 ETF／基金研究、产品池、类内配置与研究包；既看产品本身，也看加入组合后的风险和资金影响。

## 核心认识

产品研究使用保存的指标定义与版本，比较各自明确窗口中的结果。产品详情的“指标×区间”与跨产品的“指标×产品”不是同一种表；窗口不一致要披露。走势、历史情景、收益统计及条件模拟各有用途，不能把片段表现拼成连续策略业绩。

行情指标中心提供定义与共享计算。因子中心进一步区分特征得分、因子收益和统计暴露：评分可以辅助筛选，RBSA／FF3 可以解释已实现收益，但暴露不等于披露持仓，残差也不等于经理能力。发布因子只产生可引用研究证据，进入产品池仍需人工审核。

ETF 择时研究把入场、退出、执行和评价分开，保存定义、数据与运行版本。收盘后信号按契约在下一合法开盘执行；复权价用于收益研究，不能当作实际报价或可下单股数。模板改编也不继承原股票实验绩效。

进入产品实施时，先守住大类预算，再核验经济风险：

- **Q 是资金归属**，要求产品权重按类别合成目标大类预算
- **B 是统计暴露**，产品实际风险可能与类别标签明显不同
- **残差协方差**保留产品之间共同的未解释风险，不能只加单产品残差方差
- 相对 SAA 的总主动风险与相对当前目标的实施跟踪风险分别报告

因此“预算完全匹配”不等于“风险匹配”。例如同类产品共同放大市场暴露且残差高度相关，类内换产品也可能超出总体风险上限。费用、现金可用性与剩余资金验证须使用同一产品候选，不能拿大类层扣费回测代表实际产品净收益。

## 当前与边界

产品指标、因子、ETF 择时和基础产品实施已有真实研究链；完整管理人尽调、持仓穿透及场外基金择时不属于统一完成声明。全站“参考：因子证据”新增绑定面板已因没有真实消费而移除，旧引用可读不代表各页仍可新绑定。

实施基础版覆盖声明条件下的 ETF／现金、产品风险、费用、余款续算与研究包。真实交收、容量、整数份额、未来费率与认证独立审批仍有限制。择时发布是 research_only，静态产品回放也不是完整动态 TAA 策略回放。

## 依据与继续阅读

- [产品指标与比较](../../product-research/README.md#如何使用)、[走势图口径](../../product-research/trend-chart.md)、[历史情景样本与指标](../../product-research/scenario-research.md#样本与指标契约)
- [行情指标职责](../../indicators/README.md#公开对象)、[因子定位和边界](../../factor-research.md#产品定位与边界)、[因子下游入口现状](../../factor-research.md#投研全流程落点)
- [ETF 日频执行](../../product-research/timing.md#日频执行口径)与[历史样本限制](../../product-research/timing.md#历史真实样本与限制)
- [预算与暴露分离](../../pre-investment/implementation.md#产品实施风险预算与暴露必须分开)、[产品风险实际桥接](../../../backend/pre_investment/risk.py)、[剩余资金续算](../../pre-investment/funding.md#剩余资金续算状态期限与路径)

## 复核条件

指标消费、因子入池、择时用途、实施风险、费用或产品执行范围变化时复核。本页未重跑真实产品研究；资格判断须回到具体产品、样本和冻结版本。
