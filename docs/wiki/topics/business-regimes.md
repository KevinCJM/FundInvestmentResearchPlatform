---
id: "business-regimes"
type: "topic"
title: "市场状态、历史事件与压力情景如何使用？"
domains: ["business"]
review_state: "partial"
result: "supported"
scope: "本页有源业务概述；当前实现、目标设计及历史证据分别解释；含指定源码静态核对，不代表全模块金融验证、真实投资或生产部署通过"
reviewed_at: "2026-10-04"
reviewed_by: "AI 原文与实现核对"
source_revision: "8ce7c06cc655d027cec6856caa873a1eafa1e51d"
dependencies: ["docs/wiki/claims/causal-budget-32.md::sha256:319da2d4a5f1ebab8219b3ea69130b8b023dda69be949b25918fc409ff1b788c", "docs/regimes/README.md::sha256:51ff8ea5f087218661f35ec24ad8cd19c5d11fb34725f2bdadfbe471c8839afd", "docs/regimes/events.md::sha256:a5cfedd92c7773cc7d10c41dd46eda15e04ee0a91fd18e051bb4b1ed7c83516b", "docs/regimes/continuous-state.md::sha256:d44494da6d73ab6a8fd8c1a0abd3433490e618f1528ea3249c54672d947d876b", "docs/regimes/peak-trough.md::sha256:7341ccb62b12e1f6ad5b52dfe342bc021903211960970d19c10d8db8f37954d4", "docs/regimes/smoothing.md::sha256:ab343b7acce0276edbe857e3b908ce6a299ce2cc1ec0d8a4efd9a48b26d712e6", "docs/regimes/validation.md::sha256:b9db3dd255b417609a0e121d43707053a392a3b8c721af9c9427bfd30d8f051c", "docs/regimes/risk-models.md::sha256:40b2349d3a42846df5032e4c1291d280b81eced177911c8e035250acb892b8c4", "docs/verification/regimes.md::sha256:515747851f98ae5e01abe3fee74337b19a6a20a5dd1489a99aff30f15bc4d97d"]
evidence_kind: "mixed"
aliases: ["市场状态", "情景研究", "实时识别"]
---

# 市场状态、历史事件与压力情景如何使用？

## 这页回答什么

区分“历史发生了什么”“现在像什么状态”和“假设冲击会怎样”，并解释这些结果怎样进入产品、LTCMA 和 TAA 研究。

## 核心认识

市场状态研究依次是**定义历史参考 → 建立实时识别 → 验证识别能力**。历史参考和实时模型保留不同的定义、运行、发布及 hash；页面把流程连起来，并不把对象合并。

市场状态是在一条状态轴上的互斥标签；历史事件是可重叠的多标签区间。事件的事实日期与研究窗口分别维护，人工事后标签即使不重绘，也不能被解释为当时已经知道。峰谷、双边平滑等未来依赖要沿完整计算图传播，末端增加滞后或比较不能洗掉事后性。

“置信度”不是统一量。one-hot 是规则编码；HMM 后验是模型内部概率；投票是模型一致程度；校准概率是对指定参考的经验匹配。模型输出完整覆盖全部日期，也不自动提高识别可靠性或授予交易资格。

下游按用途消费：历史参考支持 LTCMA 条件统计；实时状态在正式发布、可得时点与对应资格通过后支持 TAA／产品 PIT；历史参考和已校准实时结果可支持条件 CMA 研究，但条件结果目前不能进入 SAA。识别匹配、未来状态预测和配置经济价值是不同证据。

压力研究另有分工：风险模型中心研究产品敏感度，情景中心研究宏观条件传导和冲击路径，业务页只消费已发布成果。普通 OLS、样本外 R² 和时间因果性审计不能单独证明经济因果；事件名称也不能自动生成可信的 GDP 或股价跌幅。

## 当前与边界

历史／实时双轨、事件库、校准、前向捕获协议、已发布风险模型及压力研究已有核心能力；这不表示现实部署已经积累足够新样本。前瞻资格依赖注册之后的实际捕获与成熟参考，历史重放或重存旧数据不能创造前瞻证据。

历史 horizon 按完整 episode 测量，样本日数不等于独立周期数。模拟的频率×期数只是计算网格；终点回基线也不证明现实经济冲击持续相同时间。[因果审计预算核验](../claims/causal-budget-32.md)只回答局部探针覆盖问题，不是全图或投资资格证明。

## 依据与继续阅读

- [三步研究流程](../../regimes/README.md#市场状态研究的人类流程)、[下游消费边界](../../regimes/README.md#ltcma--taa-消费边界)、[Horizon 证据](../../regimes/README.md#horizon不能由用户声明必须有证据)
- [事件模型](../../regimes/events.md#全球事件数据模型)、[连续状态的需求边界](../../regimes/continuous-state.md#需求与边界)；方法细节见[峰谷定界](../../regimes/peak-trough.md)与[平滑流程](../../regimes/smoothing.md)
- [原始证据与最终状态](../../regimes/validation.md#原始证据与最终状态关联)、[真实前向资格](../../regimes/validation.md#语义边界)
- [风险模型量化解释](../../regimes/risk-models.md#量化解释与纠偏)与[情景历史验收限制](../../verification/regimes.md#未解决或未认证的范围)；[证据资格](business-evidence.md)给出跨主题解释

## 复核条件

状态定义、校准、捕获／发布资格、时点、情景传导或 LTCMA／TAA 消费变化时复核。当前概述依据契约和历史证据，未重新认证具体市场模型或生产部署。
