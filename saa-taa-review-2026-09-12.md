SAA / TAA 最新代码审核与行业对照 · 2026-09-12

**结论：这次优化把“投资目标—长期假设—SAA 政策—TAA”真正接进了计算链，方向正确。但目前仍是研究流程：前轮 4 个已复现问题，加上本次补审的图表入口回退、SLSQP 精炼名实不符、历史实验页路由跳转错误，共 7 项需要整改，尚不足以验收为完整资管配置闭环。**

审核范围：分支 `codex/frontend-design-quality`，HEAD `2329845b345665a9bacc95e36b814868a8e288ea` 加当前 44 个变更文件，包含未跟踪的新实现。按 AGENTS、路由文档与提交审核规则阅读；使用 CodeGraph 追踪调用，并核对当前源码和离线测试。未修改业务代码、提交或推送。行业资料截至 2026-09-12；以下“差距”是结合这些资料对本项目的判断，不表示所有机构必须采用同一模型。

**需要修复的 4 个问题**

1. **P1：政策检查没有计算当前 TAA 权重相对 SAA 的预期主动风险。**

   [policy_gate.py:23](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/backend/strategic_allocation/policy_gate.py:23) 计算总组合波动，但第 26 行只比较“请求的 TE 上限”和“政策的 TE 上限”。训练、留出区的已实现 TE 不能替代当前目标权重的预期 TE。

   真实服务配合合成夹具复现：政策 TE 预算 1%；SAA 权重约 8.774% / 91.226%，TAA 权重约 18.774% / 81.226%。训练 TE 0.9238%、留出 TE 0.9226%，均通过；但按冻结 CMA 协方差计算，当前主动风险为 **1.9157%**，`policy_check.within_limits` 仍为 true。该检查还被[产品应用入口:22](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/backend/tactical_allocation/portfolio_bridge.py:22)复用。

   修复方向：明确区分历史 TE 和预期 TE；对 `delta = w_TAA - w_SAA` 计算 `sqrt(deltaᵀ Σ delta)`，在预览和应用共用的门禁执行。若战术预算使用另一套短期协方差，应显式保存该模型与时点，不能仅比较参数。数值路径继续满足项目固定签名 NJIT 要求。

2. **P2：正常的行情滞后会使刚采纳的政策在 TAA 中被判定“尚未生效”。**

   [TacticalAllocationWorkspace.tsx:36](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/frontend/src/pages/TacticalAllocationWorkspace.tsx:36) 将政策日期、行情结束日等取最小值，同时赋给 `end_date` 和 `as_of`。新增[政策门禁:31](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/backend/strategic_allocation/policy_gate.py:31)要求研究日不得早于政策日期。

   复现相同前端初始化规则并调用真实后端门禁：周六 2026-09-12 建政策，最新行情 09-11，TAA 自动把研究日设为 09-11，随后判定政策无效。基金 T+1 更新也会触发；当前浏览器测试只检查结果表出现，没有检查最终可应用。

   修复方向：将决策时点、行情观察截止日、数据可得时点分别处理。研究日应遵循当前研究上下文与政策有效期；行情截止日可以更早。补周末、节假日、T+1、旧政策继续研究的交接测试。

3. **P2：某一滚动分段违反风险限制，会中断整个预览。**

   [walk_forward.py:108](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/backend/tactical_allocation/walk_forward.py:108) 在固定假设模式强制选 `scale-1`，未处理单段不可行异常。[主预览:315](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/backend/tactical_allocation/service.py:315)直接调用该诊断，异常会连主结果一起丢弃。

   离线复现：整体训练 TE **0.7504% < 1%**，主候选可行；首个 40 期高波动训练段超预算，抛出 `Selected candidate violates training constraints.`，其余分段也不再返回。

   修复方向：把该段记为 blocked，保留原因并继续后续分段；不能偷偷替换用户的固定假设。主研究与诊断应各自保留结果状态。

4. **P2：所有资产同时缺失的日期无法被识别，跨期收益可能按日收益年化。**

   [service.py:111](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/backend/strategic_allocation/service.py:111) 仅用 `excluded_incomplete_dates` 检查缺口。这个指标识别资产之间的日期不齐，无法识别所有资产共同缺失的日期。

   复现：两类资产都只保留每周一个共同观察日，32 个收益观察期、相邻日期间隔 7 天，缺失计数仍为 0，并接受默认年化因子 **252**。若这些确为周收益，相对于 52 的年化频率，波动被放大约 **2.20 倍**，会继续进入 CMA 和政策候选比较。

   修复方向：给风险参考建立明确的频率与交易日历契约；校验共同日期轴。跨期或不规则收益应拒绝或采用明确的重采样规则，不能靠资产交集完整推断日频完整。

上述数值是离线合成输入下的缺陷复现，不是正式组合业绩。原始[复现结果 JSON](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-review-evidence/reproductions.json)与[复现脚本](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-review-evidence/repro.py)已另存到报告证据目录。

**补充审核：有效前沿、离散权重图与精炼功能，要求补回**

结论：前沿/散点计算与图表源码仍在；“没有生成按钮”更准确地说是原按钮改名移位、图表默认折叠，新 SAA 页面又将入口收进历史工具折叠区。另有独立的算法回退：界面继续声明 SLSQP，后端已不再执行该算法。另经路由复现确认，本轮迁移还让历史实验页的加载/切换操作跳错页面。这三项作为 **P2 必须整改项**，加入前轮 4 项，共 7 项。

**5. P2：恢复明确可见的生成入口和默认展示的前沿/离散组合图。**

当前历史配置实验页在[ClassAllocation.tsx:855](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/frontend/src/pages/ClassAllocation.tsx:855)仍有“比较收益与风险候选”，触发 `onCalculate`；成功后在[ClassAllocation.tsx:1081](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/frontend/src/pages/ClassAllocation.tsx:1081)的“查看完整有效前沿图”折叠项里绘图。配置样本点、轮数、步长、分桶、量化等的区域在生成按钮下方很远，操作名称也不再说明会生成图像。

在 URL 已带有效 alloc 参数、方案能自动加载时，当前可用路径是：进入历史配置实验 → 点击“比较收益与风险候选” → 展开“查看完整有效前沿图”。空白入口加载或页面内切换方案另有下述第 7 项路由错误。这不是符合原使用习惯的恢复结果，只是现状说明。新主 SAA 页面[StrategicAllocationWorkspace.tsx:185](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/frontend/src/pages/StrategicAllocationWorkspace.tsx:185)还把“历史有效前沿、风险预算与策略回测”藏在另一个折叠区，进一步降低可发现性。

已核对的真实调用链：

`ClassAllocation.onCalculate → POST /api/efficient-frontier → analytics_routes.post_efficient_frontier → calculate_efficient_frontier_exploration → _run_exploration → explore_portfolios_kernel → scatter/frontier/特征点/权重 → ReactECharts`

[请求组装:720](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/frontend/src/pages/ClassAllocation.tsx:720)、[后端参数传递:272](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/backend/services/analytics_routes.py:272)、[结果生成:913](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/backend/optimizer.py:913)均仍保留。“离散分布图”在当前代码对应风险—收益平面的候选组合散点云，不是概率密度分布。

**6. P2：恢复真实的受约束精炼能力，修正 SLSQP 名称与实现不一致。**

用户提到的 SQLP，对应当前界面实际写的 **SLSQP**。前端[ClassAllocation.tsx:1062](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/frontend/src/pages/ClassAllocation.tsx:1062)仍提供此开关，并传 `refine.use_slsqp` 和 `refine.count`。

但后端[optimizer.py:783](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/backend/optimizer.py:783)只在开启时追加：

`samples=refine_count, step=0.05, buckets=max(2, refine_count)`

之后仍交给同一个随机探索内核。当前这条主链没有 SLSQP 优化调用，也没有 SLSQP 收敛状态或精炼前后证据。因此“参数还在”不等于“原精炼算法仍在”。仅增加一轮随机点不能作为原精炼功能已恢复的验收依据。

**历史定位：两项变化均由 PR #8 带入。**

已比较本地 merge commit 的两侧父提交及合并结果，而不是仅凭提交标题推断：

| 变化 | 引入提交 | PR / 证据 |
|---|---|---|
| 删除原“计算可配置空间与有效前沿”按钮，改为候选比较按钮；将图表包进默认关闭的 details | `b3d403856ca83583651269605cfff4fc7b9dc290`，2026-09-11 | [PR #8](https://github.com/KevinCJM/FundInvestmentResearchPlatform/pull/8)，merge `576e96cd5454c19d5ba98a4c02cf50e6543257c7` |
| 删除实际 `scipy.optimize.minimize(..., method='SLSQP')` 精炼，改成附加随机轮次 | `88e4e75143163759de7084f5a0f935a0f6fad5b0`，2026-09-04 | 同属 PR #8 的新增祖先；其第一父提交仍有原 SLSQP 调用 |
| 最近的样式提交 | `2329845` | 在这两个位置主要调整样式，不是首次改名或折叠的来源 |
| 本轮未提交的 SAA 改造 | 当前工作区 | 保留历史页，但将主 SAA 导向新政策工作区；历史入口位于折叠区 |

PR #8 于 2026-09-11 合并。后续 PR #9、#10 的第一父提交已包含这些变化，不能把首次回退归到它们。[Git 历史核验 JSON](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-review-evidence/frontier-history.json)与[相关 diff 摘录](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-review-evidence/frontier-history-excerpts.txt)已保存。

**7. P2：本轮迁移使历史前沿页“加载/切换方案”跳到新政策页。**

[ClassAllocation.tsx:690](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/frontend/src/pages/ClassAllocation.tsx:690)的加载按钮，以及[ClassAllocation.tsx:805](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/frontend/src/pages/ClassAllocation.tsx:805)的方案选择框，仍使用 `allocationJourneyPath('saa', ...)`。但当前未提交改动已把[allocationJourney.ts:112](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/frontend/src/app/allocationJourney.ts:112)里的 saa 目标从 `/pre-investment/saa/allocation-lab` 改成 `/pre-investment/saa/policy`。

结果是历史实验组件被卸载，原来的前沿按钮和图表离开当前页面。这个错误属于本轮未提交路由迁移，不能全部归到 PR #8。

使用与 App 相同的两条实际 Routes、真实 ClassAllocation、离线接口替身和政策页占位组件复现了两种情况：① 无 alloc 参数进入历史实验页，点击“选择该方案”；② 带方案甲进入，待自动加载完成后切换方案乙。两种情况均跳到政策路径，历史页生成按钮随之消失。2 个复现用例通过表示错误行为被确认，不表示业务行为正确。

现有 ClassAllocation 单测只将组件放在 MemoryRouter 内，没有按路径切换组件，因此即使 URL 跳错仍会保留原组件，12 项通过未发现此回归。修复须区分“进入前瞻 SAA 政策”和“在历史实验页切换当前方案”的导航目的；后者保持在 allocation-lab，并保留 alloc/universe 与相应输入隔离。

[路由复现源码](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-review-evidence/frontier-route-reproduction.test.tsx)与[运行日志](/Users/chenjunming/.codex/visualizations/2026/09/12/01a09417-8630-77b1-add2-c4652594e722/saa-taa-review-evidence/frontier-route-reproduction.log)已保存；源码原本在 pages 同级临时执行，运行后已从仓库移除，报告证据副本仅供审阅。

**补回要求与验收标准**

- **入口必须明确：**在 SAA 主页面提供直接可见的图表研究入口；在对应参数区附近恢复“生成可配置空间与有效前沿”按钮。不得依赖用户先猜中“比较候选”或打开历史工具折叠项。禁用时直接显示未加载方案、日期或约束等原因。
- **生成后直接展示：**默认展开完整散点云、近似有效前沿及较低风险/较高收益风险比/较高收益特征点；清楚显示坐标指标、样本区间、实际候选数及权重明细。候选表和图表共同保留，不能用少数候选行代替可配置空间图。
- **配置真正生效：**样本数、轮次、步长、分桶、权重量化、单项/组约束和精炼参数贯穿同一请求与同一数据口径；输入变化清除旧图，重算后显示对应结果，失败不能无反馈或留下旧结果。
- **恢复真实精炼：**若继续称 SLSQP，必须实际实现并验证该求解过程；若采用另一种受约束精炼，须明确算法、与旧能力的差异以及精度验收，不能继续冒用 SLSQP 名称。按项目 NJIT、固定签名预热、无 Python 回退规则实施，不能直接恢复一份旧 SciPy 数值路径来绕过规范。应提供目标值改善/不恶化、约束满足、收敛或失败状态及可重复证据。
- **新旧数据口径清楚：**历史收益图明确标为历史样本研究；若新 SAA 主工作区生成前瞻图，必须使用当前冻结 CMA 和政策约束。两个入口复用适当的数值与绘图能力，不复制历史算法，不把历史最优点冒充 CMA 政策最优点。
- **必须补浏览器验收：**按 App 的真实路由验证首次加载和切换方案均留在历史实验页，再从主 SAA 入口开始，不预先知道隐藏按钮，修改离散权重数量与精炼选项，点击生成，在桌面和移动端看到真实 ECharts 图；验证参数载荷、非空点集、权重与约束、图表更新及错误反馈。对精炼另外检查实际执行算法，不能只断言请求里有 `use_slsqp=true`。

本次补审实际运行：后端 `test_optimizer_strategy_numba.py` 与 `test_analytics_routes.py` **18 项通过**，包含 5000 个真实 NJIT 候选的前沿输出与约束检查；前端 `ClassAllocation.test.tsx` **12 项通过**，验证当前改名按钮能发请求并处理结果。前端测试使用图表替身，不能据此证明折叠展开后的真实绘图可见；本次另有 2 个路由复现用例确认第 7 项错误，未另跑这条前沿路径的浏览器测试。补审仅更新审核报告，尚未实施上述恢复。


**本次代码实际形成的流程**

`投资目标版本 → 已保存的大类/代理篮子 → CMA 预览与保存 → 政策候选比较与采纳 → 冻结 SAA 基线 → TAA 信号/强度研究 → 决策版本 → 产品配置`

- 目标包含期限、币种、预期收益底线、波动与主动风险预算、流动性比例和复核日期。
- CMA 显式保存预期收益、波动、相关性、均值不确定半宽、来源和经济角色。历史风险仅作参考，历史均值不会自动填成长周期预期收益。
- 采纳政策后，目标、CMA、权重边界、组约束和来源一起冻结到 TAA 基线；采纳是研究员确认，代码明确 `independent_approval=false`。
- 原 ClassAllocation 保留历史优化/回测职责；新增 StrategicAllocationWorkspace 承担前瞻政策配置。这一分工合理。
- 历史策略的 `end_date` 已传至权重、调仓和回测及相关缓存；新增内核具有固定签名与启动预热。这些是实质改进。

**大类拆分：方法丰富，但经济分类还不够可靠**

目前有合同关键词映射、层次聚类、K-medoids、K-means、谱聚类、GMM；特征包括相关性距离、随机矩阵去噪、风险收益画像、PCA 和混合特征。可按合同分类先分组，再在组内聚类。真实计算分派见 [auto_asset_class.py:483](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/backend/auto_asset_class.py:483)。

不过，合同地域、经济资产类型、统计簇不是同一维度。CFA 的分类框架强调类内同质、类间有区分，并单独考察跨资产的共同风险因子。[CFA 2026 资产配置概览](https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/overview-asset-allocation)

存量代码的实际反例：调用 `classify_text("美国国债指数基金QDII")`，得到 **海外类 / 美股 / 美股宽基**。原因是[一级海外规则:71](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/backend/fund_taxonomy.py:71)先于债券规则命中，二级又把“美国”映射成美股。这不属于本次新增 diff，但直接影响大类拆分质量，应该一起纳入改进范围。

建议分别记录：经济资产类型、地区、币种/对冲、久期、信用、流动性、风格；产品资本预算保持唯一归属，风险暴露允许多维。新 CMA 的单一 `role` 标签和理由有帮助，但不能替代暴露测量。统计聚类适合发现重复产品、辅助分组和检验类间稳定性，不能直接决定长期经济资产边界。

**SAA：已经走向前瞻配置，当前算法能力要准确命名**

当前主链是有限候选上的均值—方差与区间稳健比较：

`名义效用 = μᵀw − γ/2 · wᵀΣw`

`稳健效用 = μᵀw − k·uᵀ|w| − γ/2 · wᵀΣw`

从同一批随机候选、等权点和顶点中，挑出最低风险、名义效用、稳健效用和最高收益候选。它是可复现的候选比较；[内核:120](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/backend/strategic_allocation/kernels.py:120)明确不保证全局 QP 最优。只做多时，不确定半宽惩罚等价于对各资产预期收益作保守下调。

CFA 同时讨论 MVO 的输入敏感性、集中度、单期局限，以及 Black–Litterman、约束和重采样等应对方法。这些是可选工具，不是贴上模型名称就能证明配置更好。[CFA 2026 配置原理](https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/principles-asset-allocation)

本次 SAA 主链没有调用 Black–Litterman；历史协方差使用研究员指定的对角收缩系数，也不是自动估计的 Ledoit–Wolf。投资期限用于契约匹配，并未进入多年现金流、目标达成概率或负债匹配求解。

优先补输入口径、敏感性与求解精度验证，再选择是否加入 BL、因子协方差、尾部风险或多期目标模型。尤其不要直接粘贴外部 CMA：AQR 2026 的预测面向 5–10 年，并讨论实际收益和汇率/对冲；项目接收的是年化算术总收益，需明确通胀、算术/几何、币种和对冲口径的转换。[AQR 2026 CMA](https://www.aqr.com/insights/research/alternative-thinking/2026-capital-market-assumptions-for-major-asset-classes)

另一个已验证的口径边界：历史风险参考没有源币种字段，复现中同一参考可被声明为 USD，并继续保留 historical_reference 标记。当前代码明确由研究员确认同币种，所以这里列作尚未自动验证的数据契约，而不是声称系统已做错一次汇率转换。

**TAA 与实施：有研究约束，尚未完成管理闭环**

| 环节 | 当前主链 | 下一步应补的能力 |
|---|---|---|
| 投资目标 | 收益/波动/期限等标量约束 | 按使用者需要增加现金流、风险承受能力、负债或目标达成概率 |
| 战略配置 | 前瞻 CMA、政策权重、风险贡献、人工复核 | 参数敏感性、压力场景、稳定性与求解精度；区分政策版本和日常数据更新 |
| 战术偏离 | SAA 为锚，手工/动量/已发布状态信号，成本与训练/留出约束 | 当前目标的预期主动风险、持续监控、明确不交易区间与失效规则 |
| 滚动检验 | 逐段选择既定信号的偏离强度 | 若要评价自适应策略，需要逐段重训其实际可训练部分；保留完全未用于改规则的最终检验 |
| 产品实施 | 继承大类预算，产品内权重另行决定 | 对实际产品的费用、流动性、跟踪偏差和穿透风险再校验 |
| 再平衡与归因 | 政策记录月/季/年/阈值规则；TAA 比较按日频目标扣费 | 按真实规则模拟现金与交易；拆分 SAA、TAA、产品选择、汇率与成本贡献并反馈复核 |

当前 walk-forward 明确每段独立从 SAA 起步，不能拼接成连续可交易净值，也不自动重训整个信号模型。这个边界写得诚实；不能因此宣称完整的自适应策略已通过样本外验证。政策再平衡字段也尚未驱动 TAA 的实际执行节奏。

长期政策与日常行情的生命周期仍需解耦：[data.py:313](/Users/chenjunming/Desktop/KevinGit/FundInvestmentResearchPlatform/backend/tactical_allocation/data.py:313)要求当前 NAV 哈希等于基线冻结哈希；正常追加行情也可能使旧政策无法继续研究。保留历史快照是对的，但宜分别冻结政策定义、每次研究所用数据快照，并区分“追加新观察”和“改写旧记录”。这是生命周期设计差距，不应通过取消所有哈希校验来解决。

**与机构前沿的关系**

CPP Investments 2026 的 Total Portfolio Approach 从全组合目标出发，将机会映射到经济风险暴露，并统筹货币、融资与流动性；它还区分市场流动性与压力下满足现金义务的能力。[CPP Investments 2026 TPA](https://www.cppinvestments.com/wp-content/uploads/attachments/CPPInvestments_Insights_TPA_EN.pdf)

对本项目可直接借鉴的是“统一风险口径与预算、穿透暴露、实施反馈”，无需现在就照搬大型养老金的杠杆、私募和融资体系。当前同币种、只做多的研究边界可以保留。

建议目标链路是：

`目标/现金流与约束 → 经济资产及基准代理 → CMA与压力情景 → SAA政策及风险预算 → 预算内TAA → 产品实施与再平衡 → 风险/业绩归因 → 定期复核`

产品可投资性应参与上游可行性检查，但替换某只 ETF 不应天然要求重新定义整个长期经济资产体系。

**本次验证与边界**

- 后端定向测试：**180 通过**。覆盖战略配置、滚动检验、研究日期、TAA 数值/数据/服务/应用桥接、状态信号接入、策略 API、窗口和回测输出等。
- 前端定向测试：**8 个文件、74 项通过**；生产构建通过。
- 后端第一次收集测试时命中默认正式数据目录锁权限；改用临时数据根后重跑通过，没有对正式数据加锁或重启服务。
- 4 个问题均以合成输入另行复现；分类错误以当前函数直接复现。
- 浏览器验证：**桌面 1440 与移动端 390 共 2 项通过，耗时 2.9 分钟**。首次启动被沙箱本地端口限制拦截，获准后使用同一隔离测试配置重跑通过。测试使用临时 Parquet 和真实 NJIT，范围不覆盖最终产品应用和持续实盘管理。桌面截图也显示政策日 09-12、研究日 09-11，印证问题 2 的前端交接行为。
- 本次没有重跑全仓测试、使用客户持仓或证明任何策略收益有效；实现文档中的全套验收数字不作为本次实测数字。

**处理顺序：先修 4 个可复现问题、恢复有效前沿/离散组合图和真实精炼、修复历史实验页路由，并修正海外债券分类；再完善数据口径、风险穿透和生命周期，最后按实际研究需求扩展模型。**


审核结束再次核对：44 个变更文件哈希与起始快照一致，HEAD 未变化。
