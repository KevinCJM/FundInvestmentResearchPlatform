# 全项目知识与证据目录

这是从原 repo_map、当前 Git 可见文件和原文入链生成的派生发现视图，不是第二份权威目录。用 `python3 scripts/knowledge_base.py catalog --write` 重建；状态来自原目录，active不等于功能已实现。

[返回主页](README.md) · [业务知识](navigation/business.md) · [开发知识](navigation/developer.md)

覆盖：105份受管文档、36个Hermes模块、92个相关附档/基准入口、4份原技能/操作说明。

## 全部文档

| 分类 | 文档 | 角色 | 生命周期 |
| --- | --- | --- | --- |
| business/developer | [资产配置投研与组合管理平台](../../README.md) | overview | active |
| business/developer | [文档索引](../README.md) | index | active |
| developer | [Repository Guidelines](../../AGENTS.md) | policy | active |
| business/developer | [产品需求与路线图](../product/requirements.md) | topic | active |
| business/developer | [资产主体、账户、组合分摊与投资核算详细设计](../product/accounting.md) | contract | active |
| business/developer | [资产配置投研与组合管理平台领域语言](../product/domain-language.md) | contract | active |
| business/developer | [研究参数中心与运行口径分离](../product/research-parameters.md) | contract | active |
| business/developer | [投前研究](../pre-investment/README.md) | topic | active |
| business/developer | [自动构建资产大类](../pre-investment/asset-classification.md) | contract | active |
| business/developer | [资金目标与剩余资金续算](../pre-investment/funding.md) | contract | active |
| business/developer | [历史配置空间与有效前沿](../pre-investment/historical-frontier.md) | contract | active |
| business/developer | [产品实施、统一验证与研究定稿](../pre-investment/implementation.md) | contract | active |
| business/developer | [LTCMA：长期资本市场假设](../pre-investment/ltcma.md) | contract | active |
| business/developer | [投前产物的版本与依赖管理](../pre-investment/versioning.md) | contract | active |
| business/developer | [投资目标与约束](../pre-investment/mandate.md) | contract | active |
| business/developer | [Universal 风险标尺](../pre-investment/risk-scale.md) | contract | active |
| business/developer | [SAA：战略资产配置](../pre-investment/saa.md) | contract | active |
| business/developer | [TAA 数值接口契约](../pre-investment/taa-numeric-contract.md) | contract | active |
| business/developer | [TAA：战术资产配置](../pre-investment/taa.md) | contract | active |
| business/developer | [数据源、字段映射与多源取值](../data/README.md) | topic | active |
| business/developer | [Tushare 下载与文档同步协议](../data/acquisition-protocol.md) | contract | active |
| business/developer | [复权价格口径：数据、ETL 与指标入参改造设计](../data/adjusted-price.md) | contract | active |
| business/developer | [数据下载与 ETL 编排](../data/etl.md) | contract | active |
| business/developer | [PIT：数据可得时点与决策时钟](../data/pit.md) | contract | active |
| business/developer | [数据存储与目录迁移](../data/storage.md) | contract | active |
| business/developer | [Tushare 下载、文件与数据状态契约](../data/tushare-download.md) | contract | active |
| business/developer | [行情指标中心：独立定义与共享执行](../indicators/README.md) | topic | active |
| business/developer | [指标画布](../indicators/canvas.md) | contract | active |
| business/developer | [因果性审计模块设计（未来函数 / 数据泄露检验）](../indicators/causality.md) | contract | active |
| business/developer | [公用计算图与 ETL 画布](../indicators/computation-graph.md) | contract | active |
| business/developer | [指标 Excel 复现契约](../indicators/excel-export.md) | contract | active |
| business/developer | [指标参数契约](../indicators/parameters.md) | contract | active |
| business/developer | [滚动区间计算图](../indicators/rolling-intervals.md) | contract | active |
| business/developer | [产品研究：指标表与产品对比](../product-research/README.md) | topic | active |
| business/developer | [产品研究：历史情景与模拟](../product-research/scenario-research.md) | contract | active |
| business/developer | [ETF 产品择时研究](../product-research/timing.md) | contract | active |
| business/developer | [产品走势图与时序指标](../product-research/trend-chart.md) | contract | active |
| business/developer | [市场状态研究](../regimes/README.md) | topic | active |
| business/developer | [连续实时市场状态](../regimes/continuous-state.md) | contract | active |
| business/developer | [历史事件库与时点能力](../regimes/events.md) | contract | active |
| business/developer | [峰谷定界法：事后牛熊震荡区间识别](../regimes/peak-trough.md) | contract | active |
| business/developer | [风险模型、情景模拟与压力测试](../regimes/risk-models.md) | contract | active |
| business/developer | [市场状态研究流程与平滑算子](../regimes/smoothing.md) | contract | active |
| business/developer | [市场状态：验证、校准与前瞻资格](../regimes/validation.md) | contract | active |
| business/developer | [因子研究](../factor-research.md) | topic | active |
| developer | [前端设计准则](../frontend/README.md) | topic | active |
| developer | [新版投研首页](../frontend/homepage.md) | contract | active |
| developer | [语言与业务术语](../frontend/i18n.md) | contract | active |
| developer | [分支保护实现](../governance/branch-protection.md) | policy | active |
| developer | [提交、审核与合并规范](../governance/branch-submission-rules.md) | policy | active |
| developer | [文档维护协议](../governance/documentation.md) | policy | active |
| developer | [NJIT 与零拷贝高性能计算约束](../governance/numeric-computing.md) | policy | active |
| developer | [算子颗粒度与组合算法治理](../governance/operator-contracts.md) | policy | active |
| developer | [代码提交、Bot 审核与合并全流程](../governance/submission-workflow.md) | policy | active |
| business/developer | [配置研究：方法来源与选择理由](../research/allocation-methods.md) | research | active |
| business/developer | [CMA 模型：关键决策与勘误](../research/cma-model-decisions.md) | research | active |
| business/developer | [沪深300主趋势：实时识别与 CMA 研究校验](../research/csi300-reference-recognition-study-2026-09-15.md) | research | active |
| business/developer | [市场状态：方法选择与研究结论](../research/regime-methods.md) | research | active |
| business/developer | [配置研究：决策与验收纪要](../verification/allocation.md) | evidence | historical |
| business/developer | [工程、指标与界面：验收纪要](../verification/engineering.md) | evidence | historical |
| business/developer | [因子研究：有效历史证据](../verification/factors.md) | evidence | historical |
| business/developer | [情景研究：决策与验收纪要](../verification/regimes.md) | evidence | historical |
| business/developer | [AI 功能设计](../research/ai-functions-design.md) | research | historical |
| business/developer | [指标中心 × 产品研究：AI 智能体详细设计](../research/ai-agent-indicator-product-research-design-2026-09-18.md) | draft | draft |
| business/developer | [投前研究流程与机构买方投研差距分析](../research/pre-investment-institutional-gap-analysis-2026-09-19.md) | draft | draft |
| developer | [Production deployment](../../deploy/README.md) | topic | active |
| business/developer | [C++ AOT 接入契约](../indicators/cpp-aot-contracts.md) | contract | active |
| business/developer | [C++ AOT 契约验收记录](../verification/cpp-aot-contracts.md) | evidence | active |
| business/developer | [AI 上下文压缩早期设计](../research/ai-agent-context-compaction-design-2026-09-20.md) | research | historical |
| business/developer | [AI Harness 进展检测设计](../research/ai-agent-harness-progress-design-2026-09-20.md) | research | historical |
| business/developer | [AI 指标与产品研究早期实现设计](../research/ai-agent-indicator-product-research-implementation-design-2026-09-19.md) | research | historical |
| business/developer | [AI 助手共用架构与接入设计](../research/ai-agent-reusable-architecture-design-2026-09-21.md) | research | historical |
| business/developer | [AI 对话界面设计依据](../research/ai-assistant-conversation-ui-design-2026-09-20.md) | research | historical |
| business/developer | [投研平台接入独立智能体：整体迁移设计](../research/portable-agent-platform-integration.md) | contract | active |
| business/developer | [项目知识 Wiki](README.md) | topic | active |
| business/developer | [Obsidian LLM Wiki 详细设计](llm-wiki-design.md) | topic | active |
| business/developer | [2026-10-04 全量文档审核报告](documentation-audit-2026-10-04.md) | evidence | historical |
| business/developer | [2026-10-04 文档覆盖与归属台账](audit-files-2026-10-04.md) | evidence | historical |
| business/developer | [全项目知识与证据目录](catalog.md) | index | active |
| business/developer | [复权经济口径：恒等式证据与源码注释冲突](claims/adjustment-economic-evidence.md) | evidence | active |
| business/developer | [因果审计：32节点预算不等于全图通过](claims/causal-budget-32.md) | evidence | active |
| business/developer | [C++ AOT：局部标量适配不等于全平台迁移](claims/cpp-local-coverage.md) | evidence | active |
| business/developer | [NIW：当前日频252与先验信息量边界](claims/niw-frequency.md) | evidence | active |
| business/developer | [当前助手：Portable宿主链与旧Harness边界](claims/portable-current.md) | evidence | active |
| business/developer | [RiskScale：CNY标签不构成来源币种与FX证据](claims/risk-scale-fx.md) | evidence | active |
| business | [业务知识](navigation/business.md) | index | active |
| developer | [开发知识](navigation/developer.md) | index | active |
| business/developer | [项目证据基线与历史审计范围](sources/project-evidence-baseline.md) | evidence | active |
| business/developer | [来源、比较与主张模板](templates.md) | topic | active |
| business | [目标、长期假设与配置决策怎样分工？](topics/business-allocation.md) | topic | active |
| business | [哪些能力已经接通，关键决定为什么这样做？](topics/business-decisions.md) | topic | active |
| business | [怎样读懂金融结果，并判断证据够不够？](topics/business-evidence.md) | topic | active |
| business | [研究定稿之后，真实组合和核算怎样接上？](topics/business-investment.md) | topic | active |
| business | [怎样从一个投资目标完成研究并交接？](topics/business-process.md) | topic | active |
| business | [怎样研究产品，并把大类预算落实为产品方案？](topics/business-products.md) | topic | active |
| business | [这个平台服务谁，解决什么问题？](topics/business-purpose.md) | topic | active |
| business | [市场状态、历史事件与压力情景如何使用？](topics/business-regimes.md) | topic | active |
| developer/business | [这个平台怎样分层，哪些能力属于谁？](topics/developer-architecture.md) | topic | active |
| developer/business | [一份数据怎样成为可追溯的研究结果？](topics/developer-data-compute.md) | topic | active |
| developer/business | [为什么采用这些架构决定，哪些旧设计已经被替代？](topics/developer-decisions.md) | topic | active |
| developer | [一次开发怎样交付，什么证据才算完成？](topics/developer-delivery.md) | topic | active |
| developer/business | [各研究模块怎样交接，前端怎样保持同一业务事实？](topics/developer-modules.md) | topic | active |
| developer | [怎样启动与部署，哪些门槛必须单独验证？](topics/developer-operations.md) | topic | active |
| business/developer | [怎样把研报、论文和竞品方案转成可复用的项目知识？](topics/research-library.md) | topic | active |
| business/developer | [知识库操作手册与验收](workflow.md) | topic | active |

## 模块归属

这里仅展示原模块与文档所有权，不复制源码、测试和回归清单；具体执行继续读取原Hermes路由。

| 原模块 | 知识/契约入口 |
| --- | --- |
| branch_submission_workflow | [AGENTS.md](../../AGENTS.md)；[docs/governance/branch-protection.md](../governance/branch-protection.md)；[docs/governance/branch-submission-rules.md](../governance/branch-submission-rules.md)；[docs/governance/submission-workflow.md](../governance/submission-workflow.md) |
| tactical_allocation_workbench | [docs/pre-investment/README.md](../pre-investment/README.md)；[docs/pre-investment/versioning.md](../pre-investment/versioning.md)；[docs/pre-investment/taa-numeric-contract.md](../pre-investment/taa-numeric-contract.md)；[docs/pre-investment/taa.md](../pre-investment/taa.md)；[docs/verification/allocation.md](../verification/allocation.md)；[docs/verification/regimes.md](../verification/regimes.md) |
| published_risk_scenario_research | [docs/regimes/risk-models.md](../regimes/risk-models.md)；[docs/verification/regimes.md](../verification/regimes.md) |
| product_timing_research | [docs/product-research/timing.md](../product-research/timing.md) |
| backend_data_storage | [docs/data/storage.md](../data/storage.md) |
| factor_research_center | [docs/factor-research.md](../factor-research.md)；[docs/verification/factors.md](../verification/factors.md) |
| project_docs | [README.md](../../README.md)；[docs/README.md](../README.md)；[docs/product/research-parameters.md](../product/research-parameters.md)；[docs/governance/documentation.md](../governance/documentation.md)；[docs/research/ai-functions-design.md](../research/ai-functions-design.md)；[docs/research/ai-agent-indicator-product-research-design-2026-09-18.md](../research/ai-agent-indicator-product-research-design-2026-09-18.md)；[docs/research/pre-investment-institutional-gap-analysis-2026-09-19.md](../research/pre-investment-institutional-gap-analysis-2026-09-19.md)；[docs/research/ai-agent-context-compaction-design-2026-09-20.md](../research/ai-agent-context-compaction-design-2026-09-20.md)；[docs/research/ai-agent-harness-progress-design-2026-09-20.md](../research/ai-agent-harness-progress-design-2026-09-20.md)；[docs/research/ai-agent-indicator-product-research-implementation-design-2026-09-19.md](../research/ai-agent-indicator-product-research-implementation-design-2026-09-19.md)；[docs/research/ai-agent-reusable-architecture-design-2026-09-21.md](../research/ai-agent-reusable-architecture-design-2026-09-21.md)；[docs/research/ai-assistant-conversation-ui-design-2026-09-20.md](../research/ai-assistant-conversation-ui-design-2026-09-20.md)；[docs/wiki/README.md](README.md)；[docs/wiki/llm-wiki-design.md](llm-wiki-design.md)；[docs/wiki/documentation-audit-2026-10-04.md](documentation-audit-2026-10-04.md)；[docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| backend_api_shell | [docs/data/pit.md](../data/pit.md) |
| backend_quant_core | [docs/pre-investment/historical-frontier.md](../pre-investment/historical-frontier.md)；[docs/governance/numeric-computing.md](../governance/numeric-computing.md)；[docs/verification/allocation.md](../verification/allocation.md) |
| backend_strategy_routes | [docs/pre-investment/historical-frontier.md](../pre-investment/historical-frontier.md) |
| backend_analytics_routes | [docs/pre-investment/historical-frontier.md](../pre-investment/historical-frontier.md) |
| backend_etf_product_api | [docs/product-research/README.md](../product-research/README.md) |
| backend_instrument_analytics | [docs/product-research/README.md](../product-research/README.md) |
| market_data_snapshot_lifecycle | [docs/data/README.md](../data/README.md) |
| backend_indicator_runtime | [docs/governance/numeric-computing.md](../governance/numeric-computing.md) |
| backend_custom_indicator_service | [docs/indicators/README.md](../indicators/README.md)；[docs/indicators/excel-export.md](../indicators/excel-export.md)；[docs/indicators/rolling-intervals.md](../indicators/rolling-intervals.md) |
| backend_typed_indicator_v2 | [docs/data/adjusted-price.md](../data/adjusted-price.md)；[docs/indicators/README.md](../indicators/README.md)；[docs/indicators/causality.md](../indicators/causality.md)；[docs/indicators/parameters.md](../indicators/parameters.md)；[docs/indicators/rolling-intervals.md](../indicators/rolling-intervals.md)；[docs/governance/operator-contracts.md](../governance/operator-contracts.md)；[docs/verification/engineering.md](../verification/engineering.md)；[docs/indicators/cpp-aot-contracts.md](../indicators/cpp-aot-contracts.md)；[docs/verification/cpp-aot-contracts.md](../verification/cpp-aot-contracts.md) |
| backend_portfolio_research | [docs/indicators/README.md](../indicators/README.md) |
| backend_regime_research | [docs/data/tushare-download.md](../data/tushare-download.md)；[docs/indicators/computation-graph.md](../indicators/computation-graph.md)；[docs/regimes/README.md](../regimes/README.md)；[docs/regimes/continuous-state.md](../regimes/continuous-state.md)；[docs/regimes/events.md](../regimes/events.md)；[docs/regimes/peak-trough.md](../regimes/peak-trough.md)；[docs/regimes/smoothing.md](../regimes/smoothing.md)；[docs/regimes/validation.md](../regimes/validation.md)；[docs/research/csi300-reference-recognition-study-2026-09-15.md](../research/csi300-reference-recognition-study-2026-09-15.md)；[docs/research/regime-methods.md](../research/regime-methods.md)；[docs/verification/regimes.md](../verification/regimes.md) |
| frontend_regime_research | [docs/regimes/README.md](../regimes/README.md)；[docs/regimes/smoothing.md](../regimes/smoothing.md)；[docs/verification/regimes.md](../verification/regimes.md) |
| frontend_app_shell | [docs/frontend/README.md](../frontend/README.md)；[docs/frontend/homepage.md](../frontend/homepage.md)；[docs/frontend/i18n.md](../frontend/i18n.md)；[docs/verification/engineering.md](../verification/engineering.md) |
| frontend_investment_process_framework | [docs/product/requirements.md](../product/requirements.md)；[docs/product/accounting.md](../product/accounting.md)；[docs/product/domain-language.md](../product/domain-language.md) |
| frontend_indicator_studio | [docs/indicators/README.md](../indicators/README.md)；[docs/indicators/canvas.md](../indicators/canvas.md) |
| frontend_metric_presentation | [docs/product-research/README.md](../product-research/README.md)；[docs/product-research/trend-chart.md](../product-research/trend-chart.md) |
| frontend_etf_research_pages | [docs/product-research/scenario-research.md](../product-research/scenario-research.md) |
| frontend_portfolio_research | [docs/indicators/README.md](../indicators/README.md) |
| frontend_allocation_workflows | [docs/pre-investment/README.md](../pre-investment/README.md)；[docs/pre-investment/asset-classification.md](../pre-investment/asset-classification.md)；[docs/verification/allocation.md](../verification/allocation.md) |
| data_ingestion_config | [docs/product/requirements.md](../product/requirements.md)；[docs/data/acquisition-protocol.md](../data/acquisition-protocol.md)；[docs/data/etl.md](../data/etl.md)；[docs/data/tushare-download.md](../data/tushare-download.md)；[deploy/README.md](../../deploy/README.md) |
| backend_data_model_catalog | [docs/data/README.md](../data/README.md) |
| backend_data_source_etl | [docs/data/README.md](../data/README.md)；[docs/data/etl.md](../data/etl.md)；[docs/indicators/computation-graph.md](../indicators/computation-graph.md) |
| frontend_data_source_center | [docs/indicators/computation-graph.md](../indicators/computation-graph.md) |
| ai_routing_files | [docs/README.md](../README.md) |
| strategic_allocation_policy | [docs/pre-investment/README.md](../pre-investment/README.md)；[docs/pre-investment/funding.md](../pre-investment/funding.md)；[docs/pre-investment/ltcma.md](../pre-investment/ltcma.md)；[docs/pre-investment/versioning.md](../pre-investment/versioning.md)；[docs/pre-investment/mandate.md](../pre-investment/mandate.md)；[docs/pre-investment/risk-scale.md](../pre-investment/risk-scale.md)；[docs/pre-investment/saa.md](../pre-investment/saa.md)；[docs/research/allocation-methods.md](../research/allocation-methods.md)；[docs/research/cma-model-decisions.md](../research/cma-model-decisions.md)；[docs/verification/allocation.md](../verification/allocation.md) |
| pre_investment_implementation | [docs/pre-investment/funding.md](../pre-investment/funding.md)；[docs/pre-investment/implementation.md](../pre-investment/implementation.md)；[docs/pre-investment/versioning.md](../pre-investment/versioning.md)；[docs/verification/allocation.md](../verification/allocation.md) |
| documentation_maintenance | [docs/README.md](../README.md)；[docs/governance/documentation.md](../governance/documentation.md)；[docs/wiki/README.md](README.md)；[docs/wiki/llm-wiki-design.md](llm-wiki-design.md)；[docs/wiki/documentation-audit-2026-10-04.md](documentation-audit-2026-10-04.md)；[docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md)；[docs/wiki/catalog.md](catalog.md)；[docs/wiki/claims/adjustment-economic-evidence.md](claims/adjustment-economic-evidence.md)；[docs/wiki/claims/causal-budget-32.md](claims/causal-budget-32.md)；[docs/wiki/claims/cpp-local-coverage.md](claims/cpp-local-coverage.md)；[docs/wiki/claims/niw-frequency.md](claims/niw-frequency.md)；[docs/wiki/claims/portable-current.md](claims/portable-current.md)；[docs/wiki/claims/risk-scale-fx.md](claims/risk-scale-fx.md)；[docs/wiki/navigation/business.md](navigation/business.md)；[docs/wiki/navigation/developer.md](navigation/developer.md)；[docs/wiki/sources/project-evidence-baseline.md](sources/project-evidence-baseline.md)；[docs/wiki/templates.md](templates.md)；[docs/wiki/topics/business-allocation.md](topics/business-allocation.md)；[docs/wiki/topics/business-decisions.md](topics/business-decisions.md)；[docs/wiki/topics/business-evidence.md](topics/business-evidence.md)；[docs/wiki/topics/business-investment.md](topics/business-investment.md)；[docs/wiki/topics/business-process.md](topics/business-process.md)；[docs/wiki/topics/business-products.md](topics/business-products.md)；[docs/wiki/topics/business-purpose.md](topics/business-purpose.md)；[docs/wiki/topics/business-regimes.md](topics/business-regimes.md)；[docs/wiki/topics/developer-architecture.md](topics/developer-architecture.md)；[docs/wiki/topics/developer-data-compute.md](topics/developer-data-compute.md)；[docs/wiki/topics/developer-decisions.md](topics/developer-decisions.md)；[docs/wiki/topics/developer-delivery.md](topics/developer-delivery.md)；[docs/wiki/topics/developer-modules.md](topics/developer-modules.md)；[docs/wiki/topics/developer-operations.md](topics/developer-operations.md)；[docs/wiki/topics/research-library.md](topics/research-library.md)；[docs/wiki/workflow.md](workflow.md) |
| portable_agent_integration | [docs/research/ai-functions-design.md](../research/ai-functions-design.md)；[docs/research/ai-agent-indicator-product-research-design-2026-09-18.md](../research/ai-agent-indicator-product-research-design-2026-09-18.md)；[deploy/README.md](../../deploy/README.md)；[docs/research/ai-agent-context-compaction-design-2026-09-20.md](../research/ai-agent-context-compaction-design-2026-09-20.md)；[docs/research/ai-agent-harness-progress-design-2026-09-20.md](../research/ai-agent-harness-progress-design-2026-09-20.md)；[docs/research/ai-agent-indicator-product-research-implementation-design-2026-09-19.md](../research/ai-agent-indicator-product-research-implementation-design-2026-09-19.md)；[docs/research/ai-agent-reusable-architecture-design-2026-09-21.md](../research/ai-agent-reusable-architecture-design-2026-09-21.md)；[docs/research/ai-assistant-conversation-ui-design-2026-09-20.md](../research/ai-assistant-conversation-ui-design-2026-09-20.md)；[docs/research/portable-agent-platform-integration.md](../research/portable-agent-platform-integration.md) |

## 原技能与操作说明

这些说明不计入受管文档数，但保留原位置和原Hermes入口。

- [skills/ai-hermes-routing-init/README.md](../../skills/ai-hermes-routing-init/README.md)：项目技能/操作说明；从原Hermes读取，不复制规则
- [skills/ai-hermes-routing-init/SKILL.md](../../skills/ai-hermes-routing-init/SKILL.md)：项目技能/操作说明；从原Hermes读取，不复制规则
- [skills/ai-hermes-self-evolve/README.md](../../skills/ai-hermes-self-evolve/README.md)：项目技能/操作说明；从原Hermes读取，不复制规则
- [skills/ai-hermes-self-evolve/SKILL.md](../../skills/ai-hermes-self-evolve/SKILL.md)：项目技能/操作说明；从原Hermes读取，不复制规则

## 相关附档与基准

逐张图像内容与历史benchmark没有在本轮重新验收；无原文入链的文件也保留可发现性并明确标出。占位文件不作为知识材料。

| 文件 | 类型/大小 | 证据身份 | 原文入链 |
| --- | --- | --- | --- |
| [backend/benchmark_indicator_engine.py](../../backend/benchmark_indicator_engine.py) | py/3698 bytes | 历史基准/方法入口；未在本轮执行 | 未找到正文入链；用途待核 |
| [backend/scripts/benchmark_builtin_series_migration.py](../../backend/scripts/benchmark_builtin_series_migration.py) | py/3794 bytes | 历史基准/方法入口；未在本轮执行 | 未找到正文入链；用途待核 |
| [backend/scripts/benchmark_drawdown_outputs.py](../../backend/scripts/benchmark_drawdown_outputs.py) | py/4562 bytes | 历史基准/方法入口；未在本轮执行 | 未找到正文入链；用途待核 |
| [backend/scripts/benchmark_regime_granularity.py](../../backend/scripts/benchmark_regime_granularity.py) | py/5434 bytes | 历史基准/方法入口；未在本轮执行 | 未找到正文入链；用途待核 |
| [backend/scripts/benchmark_regime_trend.py](../../backend/scripts/benchmark_regime_trend.py) | py/2952 bytes | 历史基准/方法入口；未在本轮执行 | 未找到正文入链；用途待核 |
| [backend/scripts/benchmark_rolling_interval.py](../../backend/scripts/benchmark_rolling_interval.py) | py/6570 bytes | 历史基准/方法入口；未在本轮执行 | 未找到正文入链；用途待核 |
| [backend/scripts/benchmark_timing_research.py](../../backend/scripts/benchmark_timing_research.py) | py/1788 bytes | 历史基准/方法入口；未在本轮执行 | 未找到正文入链；用途待核 |
| [backend/tests/benchmark_implementation.py](../../backend/tests/benchmark_implementation.py) | py/2835 bytes | 历史基准/方法入口；未在本轮执行 | 未找到正文入链；用途待核 |
| [backend/tests/benchmark_ltcma.py](../../backend/tests/benchmark_ltcma.py) | py/2777 bytes | 历史基准/方法入口；未在本轮执行 | 未找到正文入链；用途待核 |
| [backend/tests/benchmark_mandate_reference.py](../../backend/tests/benchmark_mandate_reference.py) | py/2888 bytes | 历史基准/方法入口；未在本轮执行 | 未找到正文入链；用途待核 |
| [backend/tests/benchmark_multi_cma_compatibility.py](../../backend/tests/benchmark_multi_cma_compatibility.py) | py/2451 bytes | 历史基准/方法入口；未在本轮执行 | 未找到正文入链；用途待核 |
| [backend/tests/benchmark_regime_completion.py](../../backend/tests/benchmark_regime_completion.py) | py/2532 bytes | 历史基准/方法入口；未在本轮执行 | 未找到正文入链；用途待核 |
| [backend/tests/benchmark_regime_reliability.py](../../backend/tests/benchmark_regime_reliability.py) | py/1948 bytes | 历史基准/方法入口；未在本轮执行 | 未找到正文入链；用途待核 |
| [backend/tests/benchmark_risk_scale.py](../../backend/tests/benchmark_risk_scale.py) | py/5488 bytes | 历史基准/方法入口；未在本轮执行 | 未找到正文入链；用途待核 |
| [backend/tests/benchmark_smoothing.py](../../backend/tests/benchmark_smoothing.py) | py/1956 bytes | 历史基准/方法入口；未在本轮执行 | 未找到正文入链；用途待核 |
| [backend/tests/benchmark_taa_clocks.py](../../backend/tests/benchmark_taa_clocks.py) | py/1730 bytes | 历史基准/方法入口；未在本轮执行 | 未找到正文入链；用途待核 |
| [docs/ai_routing_evolution_policy.json](../ai_routing_evolution_policy.json) | json/3008 bytes | 原JSON所有者/结构资料；按原协议读取 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/homepage/screenshots/desktop.png](../homepage/screenshots/desktop.png) | png/1243564 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/homepage/screenshots/mobile.png](../homepage/screenshots/mobile.png) | png/395502 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/homepage/screenshots/search.png](../homepage/screenshots/search.png) | png/364515 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/images/frontend-design-20260912/disclosure-unavailable-mobile.png](../images/frontend-design-20260912/disclosure-unavailable-mobile.png) | png/89068 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/verification/engineering.md](../verification/engineering.md)；[docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/images/frontend-design-20260912/home-en-1251.png](../images/frontend-design-20260912/home-en-1251.png) | png/97518 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/verification/engineering.md](../verification/engineering.md)；[docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/images/frontend-design-20260912/indicator-tablet.png](../images/frontend-design-20260912/indicator-tablet.png) | png/55177 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/verification/engineering.md](../verification/engineering.md)；[docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/images/frontend-design-20260912/product-quadrant-tooltip.png](../images/frontend-design-20260912/product-quadrant-tooltip.png) | png/126995 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/verification/engineering.md](../verification/engineering.md)；[docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/images/frontend-design-20260912/report-desktop.png](../images/frontend-design-20260912/report-desktop.png) | png/97261 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/verification/engineering.md](../verification/engineering.md)；[docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/images/frontend-design-20260912/report-mobile.png](../images/frontend-design-20260912/report-mobile.png) | png/35545 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/verification/engineering.md](../verification/engineering.md)；[docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/pitfalls.json](../pitfalls.json) | json/198358 bytes | 原JSON所有者/结构资料；按原协议读取 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md)；[docs/wiki/topics/developer-delivery.md](topics/developer-delivery.md) |
| [docs/qa/data-storage-attach-desktop.png](../qa/data-storage-attach-desktop.png) | png/57905 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/repo_map.json](../repo_map.json) | json/326266 bytes | 原JSON所有者/结构资料；按原协议读取 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md)；[docs/wiki/topics/business-decisions.md](topics/business-decisions.md)；[docs/wiki/topics/developer-delivery.md](topics/developer-delivery.md) |
| [docs/research/ai-assistant-conversation-ui-prototype-2026-09-20.html](../research/ai-assistant-conversation-ui-prototype-2026-09-20.html) | html/18128 bytes | HTML原型/页面源文件；非产品运行验收 | [docs/research/ai-assistant-conversation-ui-design-2026-09-20.md](../research/ai-assistant-conversation-ui-design-2026-09-20.md)；[docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/research/assets/csi300-market-trend-reference-v1.png](../research/assets/csi300-market-trend-reference-v1.png) | png/153273 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/research/assets/regime-math-formula-2026-09-07.png](../research/assets/regime-math-formula-2026-09-07.png) | png/35587 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/research/assets/regime-math-steps-2026-09-07.png](../research/assets/regime-math-steps-2026-09-07.png) | png/209830 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/research/assets/regime-studio-canvas-2026-09-07.png](../research/assets/regime-studio-canvas-2026-09-07.png) | png/417420 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/research/assets/regime-studio-guided-2026-09-07.png](../research/assets/regime-studio-guided-2026-09-07.png) | png/476318 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/research/screenshots/implementation-20260920/desktop-1440.png](../research/screenshots/implementation-20260920/desktop-1440.png) | png/288149 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/research/screenshots/implementation-20260920/mobile-320.png](../research/screenshots/implementation-20260920/mobile-320.png) | png/212137 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/research/screenshots/implementation-20260920/tablet-768.png](../research/screenshots/implementation-20260920/tablet-768.png) | png/219656 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/research/screenshots/mandate-boundaries-20260916/benchmark.json](../research/screenshots/mandate-boundaries-20260916/benchmark.json) | json/2604 bytes | 历史基准/方法入口；未在本轮执行 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/research/screenshots/mandate-boundaries-20260916/desktop-1440/01-cash-task.png](../research/screenshots/mandate-boundaries-20260916/desktop-1440/01-cash-task.png) | png/289609 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/research/screenshots/mandate-boundaries-20260916/desktop-1440/03-risk-authorization.png](../research/screenshots/mandate-boundaries-20260916/desktop-1440/03-risk-authorization.png) | png/359283 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/research/screenshots/mandate-boundaries-20260916/desktop-1440/04-independent-diagnosis.png](../research/screenshots/mandate-boundaries-20260916/desktop-1440/04-independent-diagnosis.png) | png/314543 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/research/screenshots/mandate-boundaries-20260916/desktop-1440/05-frozen-confirmation.png](../research/screenshots/mandate-boundaries-20260916/desktop-1440/05-frozen-confirmation.png) | png/295956 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/research/screenshots/mandate-boundaries-20260916/desktop-1440/06-funding-adjustment-not-approval.png](../research/screenshots/mandate-boundaries-20260916/desktop-1440/06-funding-adjustment-not-approval.png) | png/327655 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/research/screenshots/mandate-boundaries-20260916/desktop-1440/english-keyboard-empty-state.png](../research/screenshots/mandate-boundaries-20260916/desktop-1440/english-keyboard-empty-state.png) | png/205054 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/research/screenshots/mandate-boundaries-20260916/mobile-320/01-cash-task.png](../research/screenshots/mandate-boundaries-20260916/mobile-320/01-cash-task.png) | png/214708 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/research/screenshots/mandate-boundaries-20260916/mobile-320/03-risk-authorization.png](../research/screenshots/mandate-boundaries-20260916/mobile-320/03-risk-authorization.png) | png/277341 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/research/screenshots/mandate-boundaries-20260916/mobile-320/04-independent-diagnosis.png](../research/screenshots/mandate-boundaries-20260916/mobile-320/04-independent-diagnosis.png) | png/221497 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/research/screenshots/mandate-boundaries-20260916/mobile-320/05-frozen-confirmation.png](../research/screenshots/mandate-boundaries-20260916/mobile-320/05-frozen-confirmation.png) | png/215470 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/research/screenshots/mandate-boundaries-20260916/mobile-320/06-funding-adjustment-not-approval.png](../research/screenshots/mandate-boundaries-20260916/mobile-320/06-funding-adjustment-not-approval.png) | png/268592 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/research/screenshots/mandate-boundaries-20260916/mobile-320/english-keyboard-empty-state.png](../research/screenshots/mandate-boundaries-20260916/mobile-320/english-keyboard-empty-state.png) | png/125812 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/research/screenshots/mandate-boundaries-20260916/tablet-768/01-cash-task.png](../research/screenshots/mandate-boundaries-20260916/tablet-768/01-cash-task.png) | png/212818 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/research/screenshots/mandate-boundaries-20260916/tablet-768/03-risk-authorization.png](../research/screenshots/mandate-boundaries-20260916/tablet-768/03-risk-authorization.png) | png/278930 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/research/screenshots/mandate-boundaries-20260916/tablet-768/04-independent-diagnosis.png](../research/screenshots/mandate-boundaries-20260916/tablet-768/04-independent-diagnosis.png) | png/243094 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/research/screenshots/mandate-boundaries-20260916/tablet-768/05-frozen-confirmation.png](../research/screenshots/mandate-boundaries-20260916/tablet-768/05-frozen-confirmation.png) | png/224697 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/research/screenshots/mandate-boundaries-20260916/tablet-768/06-funding-adjustment-not-approval.png](../research/screenshots/mandate-boundaries-20260916/tablet-768/06-funding-adjustment-not-approval.png) | png/273702 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/research/screenshots/mandate-boundaries-20260916/tablet-768/english-keyboard-empty-state.png](../research/screenshots/mandate-boundaries-20260916/tablet-768/english-keyboard-empty-state.png) | png/144199 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/research/screenshots/mandate-fix-20260913/clock-desktop.png](../research/screenshots/mandate-fix-20260913/clock-desktop.png) | png/192526 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/research/screenshots/mandate-fix-20260913/clock-mobile.png](../research/screenshots/mandate-fix-20260913/clock-mobile.png) | png/118167 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/research/screenshots/mandate-fix-20260913/funding-mobile.png](../research/screenshots/mandate-fix-20260913/funding-mobile.png) | png/420774 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/research/screenshots/pr14-merge-review/saved-policy-desktop.png](../research/screenshots/pr14-merge-review/saved-policy-desktop.png) | png/141902 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/research/screenshots/pr14-merge-review/saved-policy-mobile.png](../research/screenshots/pr14-merge-review/saved-policy-mobile.png) | png/67991 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/research/screenshots/saa-taa-20260913/frontier-200-desktop.png](../research/screenshots/saa-taa-20260913/frontier-200-desktop.png) | png/47116 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/research/screenshots/saa-taa-20260913/frontier-200-mobile.png](../research/screenshots/saa-taa-20260913/frontier-200-mobile.png) | png/31336 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/research/screenshots/saa-taa-20260913/policy-desktop.png](../research/screenshots/saa-taa-20260913/policy-desktop.png) | png/229494 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/research/screenshots/saa-taa-20260913/policy-mobile.png](../research/screenshots/saa-taa-20260913/policy-mobile.png) | png/134602 bytes | 历史/示意图或源资产；非当前运行证明 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md) |
| [docs/task_routes.json](../task_routes.json) | json/42999 bytes | 原JSON所有者/结构资料；按原协议读取 | [docs/wiki/audit-files-2026-10-04.md](audit-files-2026-10-04.md)；[docs/wiki/topics/developer-delivery.md](topics/developer-delivery.md) |
| [frontend/index.html](../../frontend/index.html) | html/513 bytes | HTML原型/页面源文件；非产品运行验收 | 未找到正文入链；用途待核 |
| [frontend/public/homepage/images/brand.svg](../../frontend/public/homepage/images/brand.svg) | svg/386 bytes | 历史/示意图或源资产；非当前运行证明 | 未找到正文入链；用途待核 |
| [frontend/public/homepage/images/data-globe-1200.webp](../../frontend/public/homepage/images/data-globe-1200.webp) | webp/76270 bytes | 历史/示意图或源资产；非当前运行证明 | 未找到正文入链；用途待核 |
| [frontend/public/homepage/images/data-globe-1672.webp](../../frontend/public/homepage/images/data-globe-1672.webp) | webp/137180 bytes | 历史/示意图或源资产；非当前运行证明 | 未找到正文入链；用途待核 |
| [frontend/public/homepage/images/data-globe-800.webp](../../frontend/public/homepage/images/data-globe-800.webp) | webp/40192 bytes | 历史/示意图或源资产；非当前运行证明 | 未找到正文入链；用途待核 |
| [frontend/public/homepage/images/hero-bull-1200.webp](../../frontend/public/homepage/images/hero-bull-1200.webp) | webp/46056 bytes | 历史/示意图或源资产；非当前运行证明 | 未找到正文入链；用途待核 |
| [frontend/public/homepage/images/hero-bull-1672.webp](../../frontend/public/homepage/images/hero-bull-1672.webp) | webp/71504 bytes | 历史/示意图或源资产；非当前运行证明 | 未找到正文入链；用途待核 |
| [frontend/public/homepage/images/hero-bull-800.webp](../../frontend/public/homepage/images/hero-bull-800.webp) | webp/28550 bytes | 历史/示意图或源资产；非当前运行证明 | 未找到正文入链；用途待核 |
| [frontend/public/homepage/images/mascot-empty-240.webp](../../frontend/public/homepage/images/mascot-empty-240.webp) | webp/8950 bytes | 历史/示意图或源资产；非当前运行证明 | 未找到正文入链；用途待核 |
| [frontend/public/homepage/images/mascot-error-240.webp](../../frontend/public/homepage/images/mascot-error-240.webp) | webp/7912 bytes | 历史/示意图或源资产；非当前运行证明 | 未找到正文入链；用途待核 |
| [frontend/public/homepage/images/mascot-noresult-240.webp](../../frontend/public/homepage/images/mascot-noresult-240.webp) | webp/7912 bytes | 历史/示意图或源资产；非当前运行证明 | 未找到正文入链；用途待核 |
| [frontend/public/homepage/images/mascot-success-96.webp](../../frontend/public/homepage/images/mascot-success-96.webp) | webp/3166 bytes | 历史/示意图或源资产；非当前运行证明 | 未找到正文入链；用途待核 |
| [frontend/public/homepage/images/mascot-welcome-240.webp](../../frontend/public/homepage/images/mascot-welcome-240.webp) | webp/8542 bytes | 历史/示意图或源资产；非当前运行证明 | 未找到正文入链；用途待核 |
| [frontend/public/homepage/images/mascot-working-160.webp](../../frontend/public/homepage/images/mascot-working-160.webp) | webp/4776 bytes | 历史/示意图或源资产；非当前运行证明 | 未找到正文入链；用途待核 |
| [images/0001.png](../../images/0001.png) | png/1437230 bytes | 历史/示意图或源资产；非当前运行证明 | 未找到正文入链；用途待核 |
| [images/0002.png](../../images/0002.png) | png/1837665 bytes | 历史/示意图或源资产；非当前运行证明 | 未找到正文入链；用途待核 |
| [images/0003.png](../../images/0003.png) | png/1060059 bytes | 历史/示意图或源资产；非当前运行证明 | 未找到正文入链；用途待核 |
| [images/0004.png](../../images/0004.png) | png/724857 bytes | 历史/示意图或源资产；非当前运行证明 | 未找到正文入链；用途待核 |
| [images/0005.png](../../images/0005.png) | png/997237 bytes | 历史/示意图或源资产；非当前运行证明 | 未找到正文入链；用途待核 |
| [images/0006.png](../../images/0006.png) | png/875648 bytes | 历史/示意图或源资产；非当前运行证明 | 未找到正文入链；用途待核 |
| [images/0007.png](../../images/0007.png) | png/989779 bytes | 历史/示意图或源资产；非当前运行证明 | 未找到正文入链；用途待核 |
| [images/0008.png](../../images/0008.png) | png/1047980 bytes | 历史/示意图或源资产；非当前运行证明 | 未找到正文入链；用途待核 |
| [images/0009.png](../../images/0009.png) | png/1046167 bytes | 历史/示意图或源资产；非当前运行证明 | 未找到正文入链；用途待核 |
| [images/0010.png](../../images/0010.png) | png/1052224 bytes | 历史/示意图或源资产；非当前运行证明 | 未找到正文入链；用途待核 |
| [images/0011.png](../../images/0011.png) | png/872068 bytes | 历史/示意图或源资产；非当前运行证明 | 未找到正文入链；用途待核 |
