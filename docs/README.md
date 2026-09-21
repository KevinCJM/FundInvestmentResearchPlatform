# 文档索引

按下面的任务入口读取，再进入必要的专业契约。启动方式见[项目 README](../README.md)，智能体先遵守[AGENTS](../AGENTS.md)。目录元数据由 `repo_map.json` 维护；下表为生成的导航视图，不另存功能或计划状态。

## 从哪里开始

- 了解产品与业务边界：[范围与路线图](product/requirements.md)、[领域语言](product/domain-language.md)。
- 开展配置研究：[投前研究](pre-investment/README.md)；准备数据：[数据管理](data/README.md)。
- 修改计算：[指标中心](indicators/README.md)、[数值规范](governance/numeric-computing.md)、[算子治理](governance/operator-contracts.md)。
- 修改界面：[前端设计](frontend/README.md)；准备提交：[提交规范](governance/branch-submission-rules.md)及[执行流程](governance/submission-workflow.md)。
- 任务收尾、更新计划或文档：[文档维护协议](governance/documentation.md)。

## 完整目录

<!-- DOCUMENT-INDEX:BEGIN -->
| 主题 | 入口／文档 | 用途与读取时机 |
| --- | --- | --- |
| 入口 | [资产配置投研与组合管理平台](../README.md) | 项目定位、快速开始及文档入口 |
| 入口 | [文档索引](README.md) | 查找主题、专业契约、历史证据和草稿 |
| 入口 | [Repository Guidelines](../AGENTS.md) | 智能体执行协议、必读条件与授权边界 |
| 产品与领域 | [产品需求与路线图](product/requirements.md) | 了解产品范围、当前能力和交付顺序 |
| 产品与领域 | [资产主体、账户、组合分摊与投资核算详细设计](product/accounting.md) | 账户、组合分摊、账务与绩效模型变更前 |
| 产品与领域 | [资产配置投研与组合管理平台领域语言](product/domain-language.md) | 业务设计或术语变更前：资产主体、账户、研究与真实组合边界 |
| 产品与领域 | [研究参数中心与运行口径分离](product/research-parameters.md) | 研究参数模板的目标设计；不视为已经交付 |
| 投前研究 | [投前研究](pre-investment/README.md) | 开始或修改投前研究：先看完整链路与承接关系 |
| 投前研究 | [自动构建资产大类](pre-investment/asset-classification.md) | 产品聚类、大类候选、映射及局限 |
| 投前研究 | [资金目标与剩余资金续算](pre-investment/funding.md) | 资金流、月份窗口、负债及剩余资金模型 |
| 投前研究 | [历史配置空间与有效前沿](pre-investment/historical-frontier.md) | 历史有效前沿、求解和诊断 |
| 投前研究 | [产品实施、统一验证与研究定稿](pre-investment/implementation.md) | 产品风险、成本、余款续算、研究包验证和定稿 |
| 投前研究 | [LTCMA：长期资本市场假设](pre-investment/ltcma.md) | 长期资本市场假设、模型与估计口径 |
| 投前研究 | [投资目标与约束](pre-investment/mandate.md) | 目标约束、版本确认及适用边界 |
| 投前研究 | [Universal 风险标尺](pre-investment/risk-scale.md) | C1–C5 风险标尺、冻结输入与诊断 |
| 投前研究 | [SAA：战略资产配置](pre-investment/saa.md) | 单/多 CMA 的战略配置与采纳规则 |
| 投前研究 | [TAA 数值接口契约](pre-investment/taa-numeric-contract.md) | TAA 费用、时钟、持仓递推与数值边界 |
| 投前研究 | [TAA：战术资产配置](pre-investment/taa.md) | 基于 SAA 的战术偏离及用户操作 |
| 数据管理 | [数据源、字段映射与多源取值](data/README.md) | 配置来源、导入映射与标准数据的统一入口 |
| 数据管理 | [Tushare 下载与文档同步协议](data/acquisition-protocol.md) | 下载变更前必读：技能、权限、限流、隔离 smoke 与同步要求 |
| 数据管理 | [复权价格口径：数据、ETL 与指标入参改造设计](data/adjusted-price.md) | 价格复权、收益语义与覆盖 |
| 数据管理 | [数据下载与 ETL 编排](data/etl.md) | 执行图、增量/全量、检查点及恢复 |
| 数据管理 | [PIT：数据可得时点与决策时钟](data/pit.md) | 数据血缘、发布与时点可得性 |
| 数据管理 | [数据存储与目录迁移](data/storage.md) | 共享数据目录、迁移、锁和只读边界 |
| 数据管理 | [Tushare 下载、文件与数据状态契约](data/tushare-download.md) | 下载动作、输出文件、schema 与带日期的数据证据 |
| 指标与计算图 | [指标中心：独立定义与共享执行](indicators/README.md) | 指标中心定义、运行及共享能力入口 |
| 指标与计算图 | [指标画布](indicators/canvas.md) | 画布节点、连线、展开及预览 |
| 指标与计算图 | [因果性审计模块设计（未来函数 / 数据泄露检验）](indicators/causality.md) | 因果性、知识可得时点与审核 |
| 指标与计算图 | [公用计算图与 ETL 画布](indicators/computation-graph.md) | DAG 类型、编译、执行与数据轴 |
| 指标与计算图 | [指标 Excel 复现契约](indicators/excel-export.md) | 指标导出及公式、展示映射 |
| 指标与计算图 | [指标参数契约](indicators/parameters.md) | 运行参数、默认值及定义/实例边界 |
| 指标与计算图 | [滚动区间计算图](indicators/rolling-intervals.md) | 滚动、事件、区间及边界语义 |
| 产品研究 | [产品研究：指标表与产品对比](product-research/README.md) | 产品研究表与指标消费的主要入口 |
| 产品研究 | [产品研究：历史情景与模拟](product-research/scenario-research.md) | 产品在历史情景中的比较及统计 |
| 产品研究 | [ETF 产品择时研究](product-research/timing.md) | ETF 择时图、训练/验证和版本绑定 |
| 产品研究 | [产品走势图与时序指标](product-research/trend-chart.md) | 趋势图、指标叠加、轴和批量请求 |
| 市场状态与情景 | [市场状态研究](regimes/README.md) | 历史/实时状态、研究工作台及证据承接入口 |
| 市场状态与情景 | [连续实时市场状态](regimes/continuous-state.md) | 连续状态与概率序列 |
| 市场状态与情景 | [历史事件库与时点能力](regimes/events.md) | 事件类型、时间能力及区间消费 |
| 市场状态与情景 | [峰谷定界法：事后牛熊震荡区间识别](regimes/peak-trough.md) | 峰谷识别、确认及边界 |
| 市场状态与情景 | [风险模型、情景模拟与压力测试](regimes/risk-models.md) | 发布风险模型、敏感性与压力期限 |
| 市场状态与情景 | [市场状态研究流程与平滑算子](regimes/smoothing.md) | 状态平滑与迟滞规则 |
| 市场状态与情景 | [市场状态：验证、校准与前瞻资格](regimes/validation.md) | 参考置信度、实时可靠性与前瞻资格 |
| 因子研究 | [因子研究](factor-research.md) | 因子、暴露、收益归因与研究证据 |
| 界面与术语 | [前端设计准则](frontend/README.md) | 修改视觉、交互或资产前必读的设计规范 |
| 界面与术语 | [新版投研首页](frontend/homepage.md) | 涉及首页时必读，区分首页与工作台规则 |
| 界面与术语 | [语言与业务术语](frontend/i18n.md) | 语言、术语和业务翻译维护 |
| 开发与治理 | [分支保护实现](governance/branch-protection.md) | 区分仓库规则、工作流配置和远端实际保护 |
| 开发与治理 | [提交、审核与合并规范](governance/branch-submission-rules.md) | 分支、提交、PR、审核、合并和发布前必须完整阅读 |
| 开发与治理 | [文档维护协议](governance/documentation.md) | 文档变更与任务收尾：影响核对、计划结项和检查命令 |
| 开发与治理 | [NJIT 与零拷贝高性能计算约束](governance/numeric-computing.md) | 数值、窗口和内存改动前必读：NJIT、预热、零回退、零拷贝 |
| 开发与治理 | [算子颗粒度与组合算法治理](governance/operator-contracts.md) | 算子/模板变更前必读：颗粒度、类型、时点和等价性 |
| 开发与治理 | [代码提交、Bot 审核与合并全流程](governance/submission-workflow.md) | 提交工作流的执行顺序与异常处理；不另立规则 |
| 方法研究 | [配置研究：方法来源与选择理由](research/allocation-methods.md) | 配置方法的选择依据与尚未完成的扩展 |
| 方法研究 | [CMA 模型：关键决策与勘误](research/cma-model-decisions.md) | CMA 数理勘误与实现取舍 |
| 方法研究 | [沪深300主趋势：实时识别与 CMA 研究校验](research/csi300-reference-recognition-study-2026-09-15.md) | 沪深300特定样本的参考识别研究；不可外推为普遍验证 |
| 方法研究 | [市场状态：方法选择与研究结论](research/regime-methods.md) | 状态识别方法、时点约束与取舍 |
| 验证证据 | [配置研究：决策与验收纪要](verification/allocation.md) | 配置研究历史证据、解析反例与资格边界 |
| 验证证据 | [工程、指标与界面：验收纪要](verification/engineering.md) | 工程、指标与界面验收证据及适用版本 |
| 验证证据 | [因子研究：有效历史证据](verification/factors.md) | 因子研究验收及未闭合资格 |
| 验证证据 | [情景研究：决策与验收纪要](verification/regimes.md) | 情景研究验收、限制与待复验问题 |
| 研究草稿 | [指标中心 × 产品研究：AI 智能体详细设计](research/ai-agent-indicator-product-research-design-2026-09-18.md) | 未采纳研究草稿；实施前重新核对需求、代码和证据 |
| 研究草稿 | [投前研究流程与机构买方投研差距分析](research/pre-investment-institutional-gap-analysis-2026-09-19.md) | 未采纳研究草稿；实施前重新核对需求、代码和证据 |
| 部署 | [Production deployment](../deploy/README.md) | 部署方式、运行环境与持久化要求 |
| 指标与数值计算 | [C++ AOT 接入契约](indicators/cpp-aot-contracts.md) | 接入 C++ AOT：错误隔离、数据参数、历史 DSL、结果所有权及双后端门禁 |
| 验收与证据 | [C++ AOT 契约验收记录](verification/cpp-aot-contracts.md) | 核对 AOT 契约验证的版本、范围、命令及生产迁移边界 |
<!-- DOCUMENT-INDEX:END -->

## 阅读与维护边界

当前专业契约使用稳定主题路径，文件名不再绑定一次开发日期。`research/` 中的独立样本研究保留研究日期，草稿只作为待评估输入；`verification/` 记录带日期/版本的证据，不能当作当前 HEAD 全部通过。

每个长期主题一个主要入口，独立契约按需读取。功能实现、规划、软件验证、投资资格及真实部署分别表达。新增或迁移文档同步维护目录、链接与路由；活动计划结项和坑点沉淀按维护协议执行，不再逐轮新增开发流水。

2026-09-20 历史整理前基线为 `a9dd01bd143e7634a1a6436d69be8d25f2744f85`；旧报告通过 `git show <commit>:<原路径>` 追溯。目录整理不会改变历史证据的适用版本。
