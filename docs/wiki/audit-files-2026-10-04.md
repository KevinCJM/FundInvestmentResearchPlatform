# 文档覆盖与归属台账（2026-10-04）

这是基线 `64e9b5254ad48641bf2645ec23927d4df58506df` 的审核快照。原 `docs/` 共123文件（71 Markdown、45 PNG、5 JSON、1 HTML、1 .gitkeep），另核对根AGENTS/README与部署README。全部保留原路径；没有把技术文档搬根目录，也没有把业务契约搬Wiki。长期目录继续以 `docs/repo_map.json` 为准。

“全文审阅”指71篇Markdown正文均按篇阅读，另3个项目入口也已阅读；主张核验为有界的高风险内容/静态调用链抽查，不代表逐句证明、全量业务回归或生产验收。图片仅逐张解码/尺寸检查和缩略概览，没有全尺寸语义/当前界面像素比对。JSON全部可解析，路由事实并未逐条重新运行；HTML仅静态读取，未在浏览器执行。

## Markdown：原文权威位置不变

| # | 原文件 | 原角色/状态 | 知识视图 | 本轮覆盖与结论 |
| --- | --- | --- | --- | --- |
| 01 | [docs/README.md](../../docs/README.md) | index / active | 开发 | 全文审阅；生成索引、目录角色与原位置规则；修订AI历史引导及新增Wiki入口 |
| 02 | [docs/data/README.md](../../docs/data/README.md) | topic / active | 业务 / 开发 | 全文审阅；多源定义、映射、取值与研究时点边界；未执行真实数据接入 |
| 03 | [docs/data/acquisition-protocol.md](../../docs/data/acquisition-protocol.md) | contract / active | 业务 / 开发 | 全文审阅；下载授权、账号/限频、隔离与激活规则；未验证当前账号权限或实际下载 |
| 04 | [docs/data/adjusted-price.md](../../docs/data/adjusted-price.md) | contract / active | 业务 / 开发 | 全文审阅；修正拆分日期/比例及推导因子证据强度；历史行数与复权样本未重跑 |
| 05 | [docs/data/etl.md](../../docs/data/etl.md) | contract / active | 业务 / 开发 | 全文审阅；执行图、检查点、版本与增量/全量契约；未重跑ETL及恢复流程 |
| 06 | [docs/data/pit.md](../../docs/data/pit.md) | contract / active | 业务 / 开发 | 全文审阅；可得时点、未知状态与决策时钟；不把日期过滤等同PIT认证 |
| 07 | [docs/data/storage.md](../../docs/data/storage.md) | contract / active | 业务 / 开发 | 全文审阅；存储状态、目录迁移与失败关闭契约；未操作真实磁盘或迁移数据 |
| 08 | [docs/data/tushare-download.md](../../docs/data/tushare-download.md) | contract / active | 业务 / 开发 | 全文审阅；接口/文件/schema与带日期样本证据；未下载或核验当前覆盖，文档pytest未运行 |
| 09 | [docs/factor-research.md](../../docs/factor-research.md) | topic / active | 业务 / 开发 | 全文审阅；因子收益与特征、FF3/RBSA边界；修正已移除的全局绑定入口表述 |
| 10 | [docs/frontend/README.md](../../docs/frontend/README.md) | topic / active | 开发 | 全文审阅；当前/历史路由统计、加载态、因子面板与scope清单；保留scroll规范冲突及未完成设计目标 |
| 11 | [docs/frontend/homepage.md](../../docs/frontend/homepage.md) | contract / active | 开发 | 全文审阅；首页入口、导航白名单、本地历史和截图归属；静态代码/测试阅读，未重新浏览器验收 |
| 12 | [docs/frontend/i18n.md](../../docs/frontend/i18n.md) | contract / active | 开发 | 全文审阅；共享导航同源、动态语言、原文/回退边界；去除陈旧17入口数量，运行检查缺依赖 |
| 13 | [docs/governance/branch-protection.md](../../docs/governance/branch-protection.md) | policy / active | 开发 | 全文审阅；本地rulesets/YAML和Bot门禁边界吻合；远端实际保护设置未重验 |
| 14 | [docs/governance/branch-submission-rules.md](../../docs/governance/branch-submission-rules.md) | policy / active | 开发 | 全文审阅；批准、提交、Bot与Owner例外范围保留；本轮无提交/远端动作 |
| 15 | [docs/governance/documentation.md](../../docs/governance/documentation.md) | policy / active | 开发 | 全文审阅；补三处物理归属、四轴证据及核验覆盖规则；检查器与既有协议复核 |
| 16 | [docs/governance/numeric-computing.md](../../docs/governance/numeric-computing.md) | policy / active | 开发 | 全文审阅；NJIT/AOT、零拷贝与启动readiness为规范；未以政策认定全部内核符合 |
| 17 | [docs/governance/operator-contracts.md](../../docs/governance/operator-contracts.md) | policy / active | 开发 | 全文审阅；算子粒度、耦合语义、不可变历史与等价要求保留；未变更算法 |
| 18 | [docs/governance/submission-workflow.md](../../docs/governance/submission-workflow.md) | policy / active | 开发 | 全文审阅；流程与主规范一致；不以文档授权推送/合并或远端规则修改 |
| 19 | [docs/indicators/README.md](../../docs/indicators/README.md) | topic / active | 业务 / 开发 | 全文审阅；核对variable_registry retired过滤；更新新建变量目录与锁定历史兼容 |
| 20 | [docs/indicators/canvas.md](../../docs/indicators/canvas.md) | contract / active | 业务 / 开发 | 全文审阅；画布、类型/连线与真实预览合同；未以静态图确认运行效果 |
| 21 | [docs/indicators/causality.md](../../docs/indicators/causality.md) | contract / active | 业务 / 开发 | 全文审阅；修正P1/P2、节点定位、默认截点、容差和历史算子数量；有限探针非因果证明 |
| 22 | [docs/indicators/computation-graph.md](../../docs/indicators/computation-graph.md) | contract / active | 业务 / 开发 | 全文审阅；类型、轴、DAG共享及ETL边界；不把方案声明等同全部跨中心运行验收 |
| 23 | [docs/indicators/cpp-aot-contracts.md](../../docs/indicators/cpp-aot-contracts.md) | contract / active | 业务 / 开发 | 全文审阅；保留AOT与NJIT并列及失败关闭目标；未确认原生包、凭据或生产后端 |
| 24 | [docs/indicators/excel-export.md](../../docs/indicators/excel-export.md) | contract / active | 业务 / 开发 | 全文审阅；公式/结果映射与复现边界；未实际导出工作簿逐单元格复算 |
| 25 | [docs/indicators/parameters.md](../../docs/indicators/parameters.md) | contract / active | 业务 / 开发 | 全文审阅；定义/实例、默认值、锁定历史与展示分离；静态阅读，未跑参数化回归 |
| 26 | [docs/indicators/rolling-intervals.md](../../docs/indicators/rolling-intervals.md) | contract / active | 业务 / 开发 | 全文审阅；滚动/事件窗口与时间边界；不因参考探针通过推断全部组合有效 |
| 27 | [docs/pre-investment/README.md](../../docs/pre-investment/README.md) | topic / active | 业务 / 开发 | 全文审阅；全链路、九节点与Strategy/Product-first边界；未验证真实研究数据和用户交互 |
| 28 | [docs/pre-investment/asset-classification.md](../../docs/pre-investment/asset-classification.md) | contract / active | 业务 / 开发 | 全文审阅；保留可选合同分层；纠正silhouette经验阈值与无源机构/相关性断言 |
| 29 | [docs/pre-investment/funding.md](../../docs/pre-investment/funding.md) | contract / active | 业务 / 开发 | 全文审阅；现金流逆推、期限、资产/负债与续算；关键公式独立手算，来源编号已恢复，部分来源正文未能获取 |
| 30 | [docs/pre-investment/historical-frontier.md](../../docs/pre-investment/historical-frontier.md) | contract / active | 业务 / 开发 | 全文审阅；历史前沿、约束及研究资格；未复算完整优化器或历史实证 |
| 31 | [docs/pre-investment/implementation.md](../../docs/pre-investment/implementation.md) | contract / active | 业务 / 开发 | 全文审阅；费用自融资、产品映射、残差风险及人工定稿；关键反例手算，不重验真实费用 |
| 32 | [docs/pre-investment/ltcma.md](../../docs/pre-investment/ltcma.md) | contract / active | 业务 / 开发 | 全文审阅；NIW/BL/混合/年化关键口径手算；明确当前日频无模板，外部方法不证明投资效果 |
| 33 | [docs/pre-investment/mandate.md](../../docs/pre-investment/mandate.md) | contract / active | 业务 / 开发 | 全文审阅；目标、诊断与批准边界；保留独立风险授权，不以资金可达性替代 |
| 34 | [docs/pre-investment/risk-scale.md](../../docs/pre-investment/risk-scale.md) | contract / active | 业务 / 开发 | 全文审阅；静态发现固定CNY标签但来源币种/FX未验；限为合成参考并保留待决策 |
| 35 | [docs/pre-investment/saa.md](../../docs/pre-investment/saa.md) | contract / active | 业务 / 开发 | 全文审阅；单/多CMA及稳健风险约束，box/椭球等关键公式手算；不核全量求解 |
| 36 | [docs/pre-investment/taa-numeric-contract.md](../../docs/pre-investment/taa-numeric-contract.md) | contract / active | 业务 / 开发 | 全文审阅；时钟、持仓递推、费用与方向边界；关键自融资关系复算，未执行数值套件 |
| 37 | [docs/pre-investment/taa.md](../../docs/pre-investment/taa.md) | contract / active | 业务 / 开发 | 全文审阅；训练/holdout、信号成熟与应用门禁；不认定所有信号或费用已具实盘资格 |
| 38 | [docs/pre-investment/versioning.md](../../docs/pre-investment/versioning.md) | contract / active | 业务 / 开发 | 全文审阅；版本、来源失效和不可变交接合同；未实际执行删除/恢复或迁移 |
| 39 | [docs/product/accounting.md](../../docs/product/accounting.md) | contract / active | 业务 / 开发 | 全文审阅；IBOR/ABOR/PBOR与核算目标；缩窄交易/交收日会计为适用regular-way，不构成准则合规意见 |
| 40 | [docs/product/domain-language.md](../../docs/product/domain-language.md) | contract / active | 业务 / 开发 | 全文审阅；业务对象与术语/真实组合/研究界限；未把占位业务视为已实施 |
| 41 | [docs/product/requirements.md](../../docs/product/requirements.md) | topic / active | 业务 / 开发 | 全文审阅；范围、路线图和已实现/原型区分；未用代码静默改写批准目标 |
| 42 | [docs/product/research-parameters.md](../../docs/product/research-parameters.md) | contract / active | 业务 / 开发 | 全文审阅；模板中心为目标设计；现有运行参数不等同该中心已交付 |
| 43 | [docs/product-research/README.md](../../docs/product-research/README.md) | topic / active | 业务 / 开发 | 全文审阅；指标表、结果锁定presentation与偏好边界；代码/测试阅读，未真实排名验证 |
| 44 | [docs/product-research/scenario-research.md](../../docs/product-research/scenario-research.md) | contract / active | 业务 / 开发 | 全文审阅；历史情景及模拟解释范围；不把条件统计升级为因果预测 |
| 45 | [docs/product-research/timing.md](../../docs/product-research/timing.md) | contract / active | 业务 / 开发 | 全文审阅；训练/验证、下一开盘执行与不可变版本；未复跑真实价格/策略结果 |
| 46 | [docs/product-research/trend-chart.md](../../docs/product-research/trend-chart.md) | contract / active | 业务 / 开发 | 全文审阅；分批、竞态、实例键和同轴口径；静态测试支持，未浏览器实测 |
| 47 | [docs/regimes/README.md](../../docs/regimes/README.md) | topic / active | 业务 / 开发 | 全文审阅；历史参考/实时状态与发布资格分离；保留研究与前瞻资格边界 |
| 48 | [docs/regimes/continuous-state.md](../../docs/regimes/continuous-state.md) | contract / active | 业务 / 开发 | 全文审阅；连续概率、频率与可得时点；未重训或校准状态模型 |
| 49 | [docs/regimes/events.md](../../docs/regimes/events.md) | contract / active | 业务 / 开发 | 全文审阅；事件分类、知识可得时点与历史修订；未验证外部事件库完整性 |
| 50 | [docs/regimes/peak-trough.md](../../docs/regimes/peak-trough.md) | contract / active | 业务 / 开发 | 全文审阅；峰谷为事后参考、确认与前视限制；不将标签用作实时先知信号 |
| 51 | [docs/regimes/risk-models.md](../../docs/regimes/risk-models.md) | contract / active | 业务 / 开发 | 全文审阅；条件敏感性、OLS与固定现金流限制；官方方法核对不等于因果识别或实盘有效 |
| 52 | [docs/regimes/smoothing.md](../../docs/regimes/smoothing.md) | contract / active | 业务 / 开发 | 全文审阅；平滑、迟滞、事件与窗口语义；未复验所有参数与数值路径 |
| 53 | [docs/regimes/validation.md](../../docs/regimes/validation.md) | contract / active | 业务 / 开发 | 全文审阅；时序校准、独立证据、资格fail-closed；未独立复现实验或授予前瞻资格 |
| 54 | [docs/research/ai-agent-context-compaction-design-2026-09-20.md](../../docs/research/ai-agent-context-compaction-design-2026-09-20.md) | research / historical | 开发 / 业务 | 全文审阅；全文为早期压缩设计/验收；保留正文，目录不再指向旧文作为当前实现 |
| 55 | [docs/research/ai-agent-harness-progress-design-2026-09-20.md](../../docs/research/ai-agent-harness-progress-design-2026-09-20.md) | research / historical | 开发 / 业务 | 全文审阅；全文为迁移前进展与停止检测；当前外部框架未重新验收 |
| 56 | [docs/research/ai-agent-indicator-product-research-design-2026-09-18.md](../../docs/research/ai-agent-indicator-product-research-design-2026-09-18.md) | draft / draft | 开发 / 业务 | 全文审阅；早期整体目标未验收，部分后续采用/替代；不将愿景/旧路径升级为当前运行事实 |
| 57 | [docs/research/ai-agent-indicator-product-research-implementation-design-2026-09-19.md](../../docs/research/ai-agent-indicator-product-research-implementation-design-2026-09-19.md) | research / historical | 开发 / 业务 | 全文审阅；迁移前实现基线；历史固定commit路径存在，不代表当前入口 |
| 58 | [docs/research/ai-agent-reusable-architecture-design-2026-09-21.md](../../docs/research/ai-agent-reusable-architecture-design-2026-09-21.md) | contract / active | 开发 / 业务 | 全文审阅；改为历史研究角色；旧AgentPanel/CopilotKit/standalone路径标历史 |
| 59 | [docs/research/ai-assistant-conversation-ui-design-2026-09-20.md](../../docs/research/ai-assistant-conversation-ui-design-2026-09-20.md) | research / historical | 开发 / 业务 | 全文审阅；历史交互依据及样稿；当前视觉契约在frontend，外部SDK另验 |
| 60 | [docs/research/ai-functions-design.md](../../docs/research/ai-functions-design.md) | contract / active | 开发 / 业务 | 全文审阅；改为历史研究角色；迁移前Python Harness/CopilotKit与当前宿主明确分离 |
| 61 | [docs/research/allocation-methods.md](../../docs/research/allocation-methods.md) | research / active | 业务 / 开发 | 全文审阅；机构官网/原方法有限核实；不从方法采用推断平台复刻程度或收益有效性 |
| 62 | [docs/research/cma-model-decisions.md](../../docs/research/cma-model-decisions.md) | research / active | 业务 / 开发 | 全文审阅；保留方法取舍与数学勘误；关键公式反例支持，不确认全部外部经验主张 |
| 63 | [docs/research/csi300-reference-recognition-study-2026-09-15.md](../../docs/research/csi300-reference-recognition-study-2026-09-15.md) | research / active | 业务 / 开发 | 全文审阅；修正复用选型区间为独立holdout的冲突；保留Insufficient Evidence和历史样本日期 |
| 64 | [docs/research/portable-agent-platform-integration.md](../../docs/research/portable-agent-platform-integration.md) | contract / active | 开发 / 业务 | 全文审阅；当前宿主调用链/ID/工具名/导入归属核对；已推Dev与外部发布/生产未知分开 |
| 65 | [docs/research/pre-investment-institutional-gap-analysis-2026-09-19.md](../../docs/research/pre-investment-institutional-gap-analysis-2026-09-19.md) | draft / draft | 业务 / 开发 | 全文审阅；保留未采纳差距分析/建议；无量化基准不作机构排名或成熟度认证 |
| 66 | [docs/research/regime-methods.md](../../docs/research/regime-methods.md) | research / active | 业务 / 开发 | 全文审阅；方法选择与因果边界；无源效率/效果结论不得扩大为普遍证据 |
| 67 | [docs/verification/allocation.md](../../docs/verification/allocation.md) | evidence / historical | 业务 / 开发 | 全文审阅；历史决策、算术反例与验收范围；原日期/版本保留，未全量重跑 |
| 68 | [docs/verification/cpp-aot-contracts.md](../../docs/verification/cpp-aot-contracts.md) | evidence / active | 业务 / 开发 | 全文审阅；历史契约验收与原生环境边界；未运行AOT引擎或生产切换 |
| 69 | [docs/verification/engineering.md](../../docs/verification/engineering.md) | evidence / historical | 业务 / 开发 | 全文审阅；全文历史证据与明确移除决定；保存未关闭缺口，不将旧绿灯算作当前 |
| 70 | [docs/verification/factors.md](../../docs/verification/factors.md) | evidence / historical | 业务 / 开发 | 全文审阅；历史收益/IC/快照标识保留；本地无原运行数据/NPZ/部分截图，未独立复现 |
| 71 | [docs/verification/regimes.md](../../docs/verification/regimes.md) | evidence / historical | 业务 / 开发 | 全文审阅；历史情景/风险验收；不关闭没有对应复验证据的资格问题 |

## 项目入口及部署说明

| 原文件 | 必须保留位置 | 本轮覆盖 |
| --- | --- | --- |
| [AGENTS.md](../../AGENTS.md) | 原路径 | 全文审阅；固定根入口；补单源/Wiki边界和受控解释器要求，保持授权与读取协议 |
| [README.md](../../README.md) | 原路径 | 全文审阅；固定根入口；Node20、受控Python3.12、Wiki导航及原产品范围 |
| [deploy/README.md](../../deploy/README.md) | 原路径 | 全文审阅；部署原位；补data构建上下文前提，未构建镜像/切换生产 |

## 非 Markdown 附件与机器文件

所有下列文件均保留原路径。附件属于原主题的证据/样稿，不因Obsidian显示而变成当前产品事实。

| # | 原文件 | 归属/用途 | 审核边界 |
| --- | --- | --- | --- |
| 01 | [docs/ai_routing_evolution_policy.json](../../docs/ai_routing_evolution_policy.json) | Hermes：演化治理 | 结构解析及相关归属核对；未将全部路由事实重新验成grounded |
| 02 | [docs/assets/.gitkeep](../../docs/assets/.gitkeep) | assets目录占位 | 目录占位，不属于知识正文 |
| 03 | [docs/homepage/screenshots/desktop.png](../../docs/homepage/screenshots/desktop.png) | frontend/homepage：历史首页截图 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 04 | [docs/homepage/screenshots/mobile.png](../../docs/homepage/screenshots/mobile.png) | frontend/homepage：历史首页截图 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 05 | [docs/homepage/screenshots/search.png](../../docs/homepage/screenshots/search.png) | frontend/homepage：历史首页截图 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 06 | [docs/images/frontend-design-20260912/disclosure-unavailable-mobile.png](../../docs/images/frontend-design-20260912/disclosure-unavailable-mobile.png) | frontend + verification/engineering：历史设计验收 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 07 | [docs/images/frontend-design-20260912/home-en-1251.png](../../docs/images/frontend-design-20260912/home-en-1251.png) | frontend + verification/engineering：历史设计验收 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 08 | [docs/images/frontend-design-20260912/indicator-tablet.png](../../docs/images/frontend-design-20260912/indicator-tablet.png) | frontend + verification/engineering：历史设计验收 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 09 | [docs/images/frontend-design-20260912/product-quadrant-tooltip.png](../../docs/images/frontend-design-20260912/product-quadrant-tooltip.png) | frontend + verification/engineering：历史设计验收 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 10 | [docs/images/frontend-design-20260912/report-desktop.png](../../docs/images/frontend-design-20260912/report-desktop.png) | frontend + verification/engineering：历史设计验收 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 11 | [docs/images/frontend-design-20260912/report-mobile.png](../../docs/images/frontend-design-20260912/report-mobile.png) | frontend + verification/engineering：历史设计验收 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 12 | [docs/pitfalls.json](../../docs/pitfalls.json) | Hermes：隐藏契约和坑点 | 结构解析及相关归属核对；未将全部路由事实重新验成grounded |
| 13 | [docs/qa/data-storage-attach-desktop.png](../../docs/qa/data-storage-attach-desktop.png) | data/storage：历史存储界面 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 14 | [docs/repo_map.json](../../docs/repo_map.json) | Hermes：模块事实、文档目录与回归命令 | 结构解析及相关归属核对；未将全部路由事实重新验成grounded |
| 15 | [docs/research/ai-assistant-conversation-ui-prototype-2026-09-20.html](../../docs/research/ai-assistant-conversation-ui-prototype-2026-09-20.html) | research/ai-assistant-conversation-ui-design：历史交互样稿 | 静态全文读取；消息/模型/保存均为演示，未浏览器运行 |
| 16 | [docs/research/assets/csi300-market-trend-reference-v1.png](../../docs/research/assets/csi300-market-trend-reference-v1.png) | research/csi300：历史样本参考图 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 17 | [docs/research/assets/regime-math-formula-2026-09-07.png](../../docs/research/assets/regime-math-formula-2026-09-07.png) | regimes + verification/engineering：历史数学/工作台截图 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 18 | [docs/research/assets/regime-math-steps-2026-09-07.png](../../docs/research/assets/regime-math-steps-2026-09-07.png) | regimes + verification/engineering：历史数学/工作台截图 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 19 | [docs/research/assets/regime-studio-canvas-2026-09-07.png](../../docs/research/assets/regime-studio-canvas-2026-09-07.png) | regimes + verification/engineering：历史数学/工作台截图 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 20 | [docs/research/assets/regime-studio-guided-2026-09-07.png](../../docs/research/assets/regime-studio-guided-2026-09-07.png) | regimes + verification/engineering：历史数学/工作台截图 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 21 | [docs/research/screenshots/implementation-20260920/desktop-1440.png](../../docs/research/screenshots/implementation-20260920/desktop-1440.png) | pre-investment/implementation：历史实施界面 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 22 | [docs/research/screenshots/implementation-20260920/mobile-320.png](../../docs/research/screenshots/implementation-20260920/mobile-320.png) | pre-investment/implementation：历史实施界面 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 23 | [docs/research/screenshots/implementation-20260920/tablet-768.png](../../docs/research/screenshots/implementation-20260920/tablet-768.png) | pre-investment/implementation：历史实施界面 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 24 | [docs/research/screenshots/mandate-boundaries-20260916/benchmark.json](../../docs/research/screenshots/mandate-boundaries-20260916/benchmark.json) | pre-investment/mandate + verification/allocation：2026-09-16基准 | 完整JSON读取；macOS历史运行/预热/内存数据，未在云环境复跑 |
| 25 | [docs/research/screenshots/mandate-boundaries-20260916/desktop-1440/01-cash-task.png](../../docs/research/screenshots/mandate-boundaries-20260916/desktop-1440/01-cash-task.png) | pre-investment/mandate + verification/allocation：历史目标边界 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 26 | [docs/research/screenshots/mandate-boundaries-20260916/desktop-1440/03-risk-authorization.png](../../docs/research/screenshots/mandate-boundaries-20260916/desktop-1440/03-risk-authorization.png) | pre-investment/mandate + verification/allocation：历史目标边界 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 27 | [docs/research/screenshots/mandate-boundaries-20260916/desktop-1440/04-independent-diagnosis.png](../../docs/research/screenshots/mandate-boundaries-20260916/desktop-1440/04-independent-diagnosis.png) | pre-investment/mandate + verification/allocation：历史目标边界 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 28 | [docs/research/screenshots/mandate-boundaries-20260916/desktop-1440/05-frozen-confirmation.png](../../docs/research/screenshots/mandate-boundaries-20260916/desktop-1440/05-frozen-confirmation.png) | pre-investment/mandate + verification/allocation：历史目标边界 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 29 | [docs/research/screenshots/mandate-boundaries-20260916/desktop-1440/06-funding-adjustment-not-approval.png](../../docs/research/screenshots/mandate-boundaries-20260916/desktop-1440/06-funding-adjustment-not-approval.png) | pre-investment/mandate + verification/allocation：历史目标边界 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 30 | [docs/research/screenshots/mandate-boundaries-20260916/desktop-1440/english-keyboard-empty-state.png](../../docs/research/screenshots/mandate-boundaries-20260916/desktop-1440/english-keyboard-empty-state.png) | pre-investment/mandate + verification/allocation：历史目标边界 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 31 | [docs/research/screenshots/mandate-boundaries-20260916/mobile-320/01-cash-task.png](../../docs/research/screenshots/mandate-boundaries-20260916/mobile-320/01-cash-task.png) | pre-investment/mandate + verification/allocation：历史目标边界 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 32 | [docs/research/screenshots/mandate-boundaries-20260916/mobile-320/03-risk-authorization.png](../../docs/research/screenshots/mandate-boundaries-20260916/mobile-320/03-risk-authorization.png) | pre-investment/mandate + verification/allocation：历史目标边界 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 33 | [docs/research/screenshots/mandate-boundaries-20260916/mobile-320/04-independent-diagnosis.png](../../docs/research/screenshots/mandate-boundaries-20260916/mobile-320/04-independent-diagnosis.png) | pre-investment/mandate + verification/allocation：历史目标边界 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 34 | [docs/research/screenshots/mandate-boundaries-20260916/mobile-320/05-frozen-confirmation.png](../../docs/research/screenshots/mandate-boundaries-20260916/mobile-320/05-frozen-confirmation.png) | pre-investment/mandate + verification/allocation：历史目标边界 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 35 | [docs/research/screenshots/mandate-boundaries-20260916/mobile-320/06-funding-adjustment-not-approval.png](../../docs/research/screenshots/mandate-boundaries-20260916/mobile-320/06-funding-adjustment-not-approval.png) | pre-investment/mandate + verification/allocation：历史目标边界 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 36 | [docs/research/screenshots/mandate-boundaries-20260916/mobile-320/english-keyboard-empty-state.png](../../docs/research/screenshots/mandate-boundaries-20260916/mobile-320/english-keyboard-empty-state.png) | pre-investment/mandate + verification/allocation：历史目标边界 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 37 | [docs/research/screenshots/mandate-boundaries-20260916/tablet-768/01-cash-task.png](../../docs/research/screenshots/mandate-boundaries-20260916/tablet-768/01-cash-task.png) | pre-investment/mandate + verification/allocation：历史目标边界 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 38 | [docs/research/screenshots/mandate-boundaries-20260916/tablet-768/03-risk-authorization.png](../../docs/research/screenshots/mandate-boundaries-20260916/tablet-768/03-risk-authorization.png) | pre-investment/mandate + verification/allocation：历史目标边界 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 39 | [docs/research/screenshots/mandate-boundaries-20260916/tablet-768/04-independent-diagnosis.png](../../docs/research/screenshots/mandate-boundaries-20260916/tablet-768/04-independent-diagnosis.png) | pre-investment/mandate + verification/allocation：历史目标边界 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 40 | [docs/research/screenshots/mandate-boundaries-20260916/tablet-768/05-frozen-confirmation.png](../../docs/research/screenshots/mandate-boundaries-20260916/tablet-768/05-frozen-confirmation.png) | pre-investment/mandate + verification/allocation：历史目标边界 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 41 | [docs/research/screenshots/mandate-boundaries-20260916/tablet-768/06-funding-adjustment-not-approval.png](../../docs/research/screenshots/mandate-boundaries-20260916/tablet-768/06-funding-adjustment-not-approval.png) | pre-investment/mandate + verification/allocation：历史目标边界 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 42 | [docs/research/screenshots/mandate-boundaries-20260916/tablet-768/english-keyboard-empty-state.png](../../docs/research/screenshots/mandate-boundaries-20260916/tablet-768/english-keyboard-empty-state.png) | pre-investment/mandate + verification/allocation：历史目标边界 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 43 | [docs/research/screenshots/mandate-fix-20260913/clock-desktop.png](../../docs/research/screenshots/mandate-fix-20260913/clock-desktop.png) | pre-investment/mandate + verification/allocation：历史目标边界 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 44 | [docs/research/screenshots/mandate-fix-20260913/clock-mobile.png](../../docs/research/screenshots/mandate-fix-20260913/clock-mobile.png) | pre-investment/mandate + verification/allocation：历史目标边界 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 45 | [docs/research/screenshots/mandate-fix-20260913/funding-mobile.png](../../docs/research/screenshots/mandate-fix-20260913/funding-mobile.png) | pre-investment/mandate + verification/allocation：历史目标边界 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 46 | [docs/research/screenshots/pr14-merge-review/saved-policy-desktop.png](../../docs/research/screenshots/pr14-merge-review/saved-policy-desktop.png) | pre-investment/saa + verification/allocation：历史配置界面 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 47 | [docs/research/screenshots/pr14-merge-review/saved-policy-mobile.png](../../docs/research/screenshots/pr14-merge-review/saved-policy-mobile.png) | pre-investment/saa + verification/allocation：历史配置界面 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 48 | [docs/research/screenshots/saa-taa-20260913/frontier-200-desktop.png](../../docs/research/screenshots/saa-taa-20260913/frontier-200-desktop.png) | pre-investment/saa + verification/allocation：历史配置界面 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 49 | [docs/research/screenshots/saa-taa-20260913/frontier-200-mobile.png](../../docs/research/screenshots/saa-taa-20260913/frontier-200-mobile.png) | pre-investment/saa + verification/allocation：历史配置界面 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 50 | [docs/research/screenshots/saa-taa-20260913/policy-desktop.png](../../docs/research/screenshots/saa-taa-20260913/policy-desktop.png) | pre-investment/saa + verification/allocation：历史配置界面 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 51 | [docs/research/screenshots/saa-taa-20260913/policy-mobile.png](../../docs/research/screenshots/saa-taa-20260913/policy-mobile.png) | pre-investment/saa + verification/allocation：历史配置界面 | PNG解码、尺寸及缩略概览；不证明当前实现或截图中的数值 |
| 52 | [docs/task_routes.json](../../docs/task_routes.json) | Hermes：任务匹配和模块扩展 | 结构解析及相关归属核对；未将全部路由事实重新验成grounded |

## 本次新增派生文件

本次只新增以下Wiki文件，未创建Obsidian应用配置、vault、社区插件、数据库或自动化：

| 文件 | 角色 | 权威边界 |
| --- | --- | --- |
| [README](README.md) | 双类导航 | 引用原契约，不保存另一份实现清单 |
| [LLM Wiki详细设计](llm-wiki-design.md) | 待实施设计 | 启用/配置/权限/发布须另行执行授权流程 |
| [审核报告](documentation-audit-2026-10-04.md) | 本轮核验记录 | 仅说明本轮证据与修订，非投资或部署认证 |
| 本台账 | 审核覆盖快照 | 不替代repo_map长期目录 |

## 基线完整性

台账覆盖的原文件清单来自干净HEAD及原路径清点，没有删除、迁移或复制权威文件。二进制附件未修改。历史正文中的固定commit引用即使旧路径已从当前代码删除，也仍按其原commit解释，不机械删除。详细变更与验证见[审核报告](documentation-audit-2026-10-04.md)。
