---
id: "developer-modules"
type: "topic"
title: "各研究模块怎样交接，前端怎样保持同一业务事实？"
domains: ["developer", "business"]
review_state: "partial"
result: "supported"
scope: "2026-10-04当前工作树原文与实现的有源概述；仅支持本页明确边界，不代表全代码审计、业务测试重跑、投资资格或生产部署证明。"
reviewed_at: "2026-10-04"
reviewed_by: "AI 原文与实现核对"
source_revision: "8ce7c06cc655d027cec6856caa873a1eafa1e51d"
dependencies: ["shared/platform-navigation.json::sha256:781673cb6d0b7099bf116383d804247bd4c756b65b18d004670fdf2aa74a1a09", "frontend/src/App.tsx::sha256:87fd236147dda9c10d3936e4befcfed009313ed64e947b759c0be36d0072e922", "frontend/src/app/processRegistry.ts::sha256:b50d1466b8fdb1cf463f3fc5fddf78873b36885098fadae78f29bec6e337eda7", "frontend/src/app/allocationJourney.ts::sha256:4dee10dc4f42c012bc17e89769ef95981bcc518e19ced255495020f3fb9542f2", "frontend/src/app/ResearchContext.tsx::sha256:4a7d84e45c0d174afd76ced2ec66ce5e3131283ac266e24a217fa286eb822358", "frontend/src/layouts/StageLayout.tsx::sha256:6b3521d7e573e4f9646899e837e6f7f4270b90c0f8015875f47d343a96c7b4c0", "frontend/src/components/ui.tsx::sha256:c606865f6581e6cf3ef594f1b072dda18f2ee1a0b5e534cbfa335df2c894b769", "frontend/tailwind.config.js::sha256:8203ae688509a61a1ae887a0e648be1bc3f105ca69acfc3de283ff25fdef043f", "docs/product/requirements.md::sha256:39e169152d837e65e8891d3489b6b67f3077fa4539a6a528e41926501dfc05c6", "docs/product/domain-language.md::sha256:f7d328f84c032a3b42a54cfc543a24f392b8781860ca8acc8dc5d8b379a258ce", "docs/pre-investment/README.md::sha256:2b07c559f9a8760869474181968b278527e1d7a8d713fc8fd91eaa1b960cdc6b", "docs/pre-investment/versioning.md::sha256:71ea867342ee951620957e96cc32b12b80d4ec936735e0726e02673b281568c3", "docs/product-research/README.md::sha256:a84331d90a81b657036ac1efbb0aa8ca8bc023cedaa78260984955c995b45b9f", "docs/product-research/timing.md::sha256:3a1e503fcde4d9d1676f5700051168aab76937a3bd27a669b41105c2f7ec8e88", "docs/factor-research.md::sha256:e1f1a7261e8d082ea8df1f1920c69b8bf594b7a67416ba7cd5d966f8901daca1", "docs/regimes/README.md::sha256:51ff8ea5f087218661f35ec24ad8cd19c5d11fb34725f2bdadfbe471c8839afd", "docs/regimes/risk-models.md::sha256:40b2349d3a42846df5032e4c1291d280b81eced177911c8e035250acb892b8c4", "docs/frontend/README.md::sha256:258fc38478f77af14b79f63bba270da51ea3c1382b874be4d58f28d62d611cb6", "docs/frontend/i18n.md::sha256:bfe17734f4d2e1b4e1bc73429420ae1d46e3df8d15f296e2c4f97b02783b2044", "docs/research/portable-agent-platform-integration.md::sha256:aa7052eab7818c2bb91f0f4a1183c48026ad6e39e682f5980740d6ea393e32fe"]
evidence_kind: "mixed"
aliases: ["前端", "模块交接", "交互状态"]
---

# 各研究模块怎样交接，前端怎样保持同一业务事实？

[开发入口](../navigation/developer.md) · [系统分层](developer-architecture.md) · [设计决定](developer-decisions.md)

## 这页回答什么

页面名称不是模块边界。本页按“产出什么、谁消费、哪里验资格”理解平台，并把前端导航、对象身份和展示规则接到同一条业务链上。

## 核心认识

### 研究模块按成果交接

- **产品研究与产品池**：研究ETF/公募基金、比较和评价，形成有版本的候选范围。产品池回答研究日哪些工具可研究或实施，不等于战略资产机会集
- **投前决策**：目标与约束定义授权边界，范围确定研究对象，LTCMA提供冻结假设，SAA形成战略政策，TAA研究允许的临时偏离；产品实施再处理映射、资金和研究包。缺产品映射不应伪造大类收益，也不必一概阻断大类TAA研究
- **行情指标与组合研究**：指标定义和计算服务被多个页面复用；组合诊断消费不可变组合运行快照，不能把当前页面选择替代成当时持仓
- **状态、风险与情景**：历史参考、实时识别和发布资格分开；风险模型/已发布情景被下游只读消费，预览不等于发布。固定目标压力影响也不等于执行策略重放
- **因子与择时**：因子研究区分特征、标签、收益及归因；产品择时使用可编辑计算图和版本化研究。训练选择、成熟标签与样本外检验是契约，不因下游页面只是展示便可省略
- **共用设置**：数据、PIT、风险标尺、指标、语言及助手配置为研究服务；一个设置页存在不代表全部下游已完成迁移

这些关系只是阅读地图。完整模块文件、测试和扩展路径仍由原 `repo_map.json` 与 `task_routes.json` 管理，不在这里复制一套机器清单。

### 前端统一“在哪儿”和“正在用什么”

`App.tsx` 决定真实路由；`processRegistry.ts` 消费共享导航，首页、阶段布局和服务端能力目录由此保持一致。侧栏是可直达目录，投前流程条是状态引导，后端仍作最终应用准入。按钮可点不能代替业务资格。

URL保存选中对象身份；已保存范围恢复它自己绑定的目标，不能被全局书签覆盖。`ResearchContext` 区分系统口径与标签页临时PIT查看；旧结果的说明应来自该结果血缘，而非当前页面徽章。读取失败必须显示失败与恢复办法，不能退成“没有数据”。

### 视觉系统服务于研究语义

首页与工作台共用 Tailwind 令牌和 `ui.tsx` 原语，构图却不同：工作台优先任务、数值阅读与紧凑操作；首页可有品牌插画。阶段色只表达身份，主操作、错误、成功仍有稳定语义。图表分类色不能被全局换色破坏，缺失指标不可用零或示例曲线补足。

多语言内容受统一术语约束。助手挂载只接线外部组件与页面业务上下文，正式业务编辑器、图表、试算与保存能力仍可独立于聊天使用。

## 当前与边界

当前研究主链已有可用或部分可用能力；投中执行、真实组合中心、基金会计、完整投后和反馈大量入口仍标为原型。`available/partial/prototype` 是共享目录声明，不是本轮逐页验证。前端准则包含后续约束和历史验收，不能据此宣布所有页面已完成视觉收敛；本页未运行浏览器或业务测试。

## 依据与继续阅读

- [产品范围](../../product/requirements.md)、[领域语言](../../product/domain-language.md)、[产品研究](../../product-research/README.md)、[投前研究](../../pre-investment/README.md)、[版本交接](../../pre-investment/versioning.md)
- [状态研究](../../regimes/README.md)、[风险模型](../../regimes/risk-models.md)、[因子研究](../../factor-research.md)、[产品择时](../../product-research/timing.md)：成果及消费者边界
- [真实路由](../../../frontend/src/App.tsx)、[共享导航](../../../shared/platform-navigation.json)、[注册表](../../../frontend/src/app/processRegistry.ts)、[流程引导](../../../frontend/src/app/allocationJourney.ts)、[研究上下文](../../../frontend/src/app/ResearchContext.tsx)
- [阶段布局](../../../frontend/src/layouts/StageLayout.tsx)、[共用UI](../../../frontend/src/components/ui.tsx)、[令牌](../../../frontend/tailwind.config.js)、[前端准则](../../frontend/README.md)、[语言术语](../../frontend/i18n.md)、[助手接入](../../research/portable-agent-platform-integration.md)

## 复核条件

模块输入输出、导航能力状态、URL身份、上游版本、PIT请求头或组件展示契约变化时复核。若将原型提升为真实业务，须同时取得后端持久化、失败路径、业务资格和实际交互证据，不能只改能力标签。
