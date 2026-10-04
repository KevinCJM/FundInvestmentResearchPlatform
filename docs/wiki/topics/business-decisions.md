---
id: "business-decisions"
type: "topic"
title: "哪些能力已经接通，关键决定为什么这样做？"
domains: ["business"]
review_state: "partial"
result: "supported"
scope: "本页有源业务概述；当前实现、目标设计及历史证据分别解释；含指定源码静态核对，不代表全模块金融验证、真实投资或生产部署通过"
reviewed_at: "2026-10-04"
reviewed_by: "AI 原文与实现核对"
source_revision: "8ce7c06cc655d027cec6856caa873a1eafa1e51d"
dependencies: ["docs/repo_map.json::sha256:b957aa6b9846d87bb44253533e3a7ad9f2e20dd72d995a8389a8f40efdcb3949", "docs/product/requirements.md::sha256:39e169152d837e65e8891d3489b6b67f3077fa4539a6a528e41926501dfc05c6", "docs/research/allocation-methods.md::sha256:8d2f0a79a54005ad165237a9adf504d1a96b36100361a922b4f2f62aad31bad6", "docs/research/cma-model-decisions.md::sha256:66c5decfaf4d686dc0c791fcda43136347bcdc4bbda65c17212e739f61ee10cd", "docs/research/regime-methods.md::sha256:04dbadaddf4a30b1c1fcb9a6a7a36b9c3f8980e1eca554e379b102f4d6ceb4fd", "docs/pre-investment/implementation.md::sha256:1c9ba2c90ed50b08f7219bb9908ef71b40cf2de0447b19976f492a05d34b7394", "docs/verification/allocation.md::sha256:a017eba29c2672f8ecc283a2d89a38b4bae78699176ca9529caa22481db523c5", "docs/research/pre-investment-institutional-gap-analysis-2026-09-19.md::sha256:b67293ea6e4a71fc755e97fc8abed08a50279153c6325aec415019d64991ecfe", "docs/verification/factors.md::sha256:3d341b85e2e9110633acbfb12431cd5ae6586bff03ca08fe7bab15386b9b356e", "docs/research/csi300-reference-recognition-study-2026-09-15.md::sha256:9e72885204e3b3932bbc1847ecee7abefd6811bdebdb339ade5f832fc34e7602", "docs/verification/engineering.md::sha256:676b4ce2c3e9e77cbcaa07f541beb39c4688971efa8c0087d3f1eeb5a380f80b"]
evidence_kind: "mixed"
aliases: ["业务决定", "当前能力", "规划与实现"]
---

# 哪些能力已经接通，关键决定为什么这样做？

## 这页回答什么

把当前能力、已采纳决定、历史证据和研究提案分开。详细状态仍由产品路线图及专业契约维护，本页提炼跨模块的认识，不另建第二份任务清单。

## 核心认识

产品研究、版本化产品池、RiskScale、核心 Mandate、双路径 LTCMA、SAA／TAA 和基础产品研究包已构成主要研究链。数据、指标、因子、状态与风险模型是横向公共能力。真实主体、账户、运营、核算、正式投后与反馈仍有明显缺口，下一阶段需要把研究目标接到真实事实和可持续复核。

### 已采纳决定与原文理由

- **保留两条研究路径**：战略机会集回答需要什么经济风险，产品池回答当日有什么合格工具；分层有价值，但不能据此删除 Product-first
- **大类研究与产品实施分开**：缺产品映射不应阻断合格 SAA／TAA 研究，具体产品的可投资性在实施边界核验
- **授权先于优化**：所需收益来自投资需求，不是市场承诺；不可达时由人调整目标、投入、期限或授权，优化器不能代改
- **模式 A 与 B 并存**：参数平均和各模型共同通过解决不同问题，不能把平均风险叫作全模型合格
- **保留 Box 并显式增加椭球选项**：Box 是合法区间稳健模型；均值协方差的联合解释需要单独条件。协方差收缩也不天然代表更保守
- **历史参考与实时识别独立**：事后解释、参考匹配和实际前向资格需要不同证据。第一版 Drawdown HMM 曾因统计簇不能支持 Recovery 的经济含义被舍弃
- **验证绑定同一候选**：产品、费用或资金输入改变后，旧报告不能放行新研究；历史成果仍保留

## 当前与边界

[目录元数据](../../repo_map.json)中的 active 表示文档仍承担现行职责，不等于内容全部实现。核算、研究参数等包含目标设计；日期较新的文字也不自动获得采纳权。

09-19 机构差距分析是未采纳草稿，其中“直接 SAA→产品未接通”“统一验证仍原型”“余款续算未完成”等已被后续实施契约与 E0–E5 验收取代，不能照抄为现状。其余机构化建议也未因此全部完成或获开发授权。

奇异协方差前沿有历史反例，但尚缺当前版本对应复验，宜写“未确认关闭”，不能写成当前必现缺陷。历史测试组有重叠，不能相加后称当前全仓通过；外部机构资料只支持公开披露的方法，不证明本项目复刻其内部系统。

## 依据与继续阅读

- [当前能力总表](../../product/requirements.md#当前实现状态总表2026-09-20)与[后续交付顺序](../../product/requirements.md#后续交付顺序)维护现状和方向
- [配置方法与原始理由](../../research/allocation-methods.md#从机构业务到本系统的决定)、[可核验外部资料](../../research/allocation-methods.md#可核验资料)保留 CFA、Vanguard、BlackRock 等来源及其适用范围
- [CMA 最终勘误](../../research/cma-model-decisions.md#数理边界与反驳)、[模型方法依据](../../research/cma-model-decisions.md#方法依据)与[状态方法取舍](../../research/regime-methods.md#长期决定)解释数学和模型选择
- [产品实施现状](../../pre-investment/implementation.md#当前范围)、[配置历史验收](../../verification/allocation.md#历史验收证据)及[奇异前沿复核状态](../../verification/allocation.md#未确认关闭奇异协方差前沿)约束完成声明
- [机构差距草稿](../../research/pre-investment-institutional-gap-analysis-2026-09-19.md)只作历史研究；[因子实证](../../verification/factors.md#历史实验结果)、[沪深300研究边界](../../research/csi300-reference-recognition-study-2026-09-15.md#8-前瞻边界)不可外推；[因子面板移除原因](../../verification/engineering.md#参考因子证据面板确认无真实消费后全部移除2026-09-21)保留有据的产品决定

## 复核条件

能力交付、决定被替代、旧反例获得复验或来源失效时更新对应认识。外部资料沿用仓库原检索日期，本次未联网复核；未从草稿、测试数字或文档日期推定部署与金融资格。

2026-10-04 技能接入复核：逐项核对 repo_map 的本轮差异，变化仅属 documentation_maintenance 的技能入口、测试与回归登记；documentation.documents 的业务文档 active 语义及本页所列业务能力/决定未改变。原业务证据未重跑，保留原范围；只更新实际复核过的目录依赖。
