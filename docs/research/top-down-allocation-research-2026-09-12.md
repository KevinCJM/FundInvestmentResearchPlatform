# 大类构建、SAA 与 TAA：机构方法调研

调研日：2026-09-12。仅使用机构公开资料；公开方法不等于取得其生产模型、参数、数据或业绩。本项目不把公开方法的简化实现标成机构模型复刻。

## 1. 业务顺序与职责

投资目标/约束 → 经济含义明确的资产分类与可投资代理 → 长期资本市场假设（CMA） → 战略政策组合（SAA） → 有期限、受主动风险预算约束的战术偏离（TAA） → 类内产品配置 → 同口径验证与实施 → 归因/复核。

产品池是实施工具的范围，不应该代替长期目标；收益聚类是诊断手段，不能自动证明一个经济资产类别成立。资产类别宜有同质性、互斥性、分散作用与可投资容量。经济风险暴露可能跨类重叠，因此“产品唯一归属”不等于“因子风险互斥”。[1][2]

SAA 的核心不是过去哪个权重涨得最多，而是目标、期限、基准、风险与流动性约束下选择长期敞口。TAA 相对 SAA 表达短期方向或相对价值观点。将漂移的持仓再平衡到长期权重，不等于进行 TAA。[2][3]

## 2. 代表性机构公开方法

| 机构/方法 | 公开事实 | 对本项目的启示与限制 |
| --- | --- | --- |
| J.P. Morgan LTCMA | 2026 版于 2025-10-20 发布；提供跨资产长期收益、波动与相关性假设，适用于 10–15 年政策配置，明确不用于短期战术判断。[4][5] | CMA 必须有期限、币种、收益口径、来源和版本；不能将美元几何收益假设直接塞入人民币算术均值优化。 |
| BlackRock CMA / robust allocation | 公开区分均值估计的不确定性与围绕均值的风险，结合情景、投资者现金流/流动性需求与稳健优化。2026 年页面还强调长期结构可能改变，需持续检视风险目标。[6] | 不把一个预期收益数字视为真值；分别保存收益不确定区间和协方差。实现透明的区间稳健候选比较，不称为 BlackRock 生产算法。 |
| Vanguard VCMM + VAAM | VCMM 的收益分布进入以效用为基础的 VAAM；先定义目标、期限、偏好与风险承受，再界定可投资集合，可整合大类、子类与主动管理。[7] | 目标对象应真正约束计算；历史回测、前瞻分布和实际业绩分开。没有经校准分布时不展示“达标概率”。 |
| Bridgewater / State Street All Weather | 以增长/通胀上升与下降的经济敏感性平衡风险，并可借期货/互换调整风险；不是预测下一种经济环境。[8] | 经济角色与风险贡献是两层。简单逆波动、等权、聚类或零杠杆 ETF 组合均不应命名为完整 All Weather。 |
| State Street quant-anchored TAA | 2026-08-31 公开流程结合宏观与估值、盈利/股息、信用利差、动量等信号；MRI 衡量风险情绪；观点由研究判断复核，主动风险与偏离区间由委托约束。[9] | 先形成可追溯观点，再决定偏离；信号、观点、约束、执行不得揉成不可解释黑盒。 |
| AQR systematic global macro / style research | 宏观策略可用系统化方向/相对价值交易；公开研究覆盖估值、动量、carry、防御等来源，强调构建方式、成本和流动性。[10][11] | 多个相关趋势窗口不等于多个独立 alpha；不能拿美国衍生品模型结果证明 A 股 long-only ETF 的有效性。 |

## 3. 模型选择与工程判断

**大类构建**：经济分类作为政策层，相关聚类/PCA作为证据层，具体 ETF/基金作为实施层。政策类别要记录投资角色、收益计价币种、流动性说明及代理选择理由。自动分类结果仍须人工确认经济解释；历史同涨跌不保证未来同风险。

**SAA**：等权/风险预算可作基线；历史 MVO 对预期收益误差敏感。应先建立可版本化 CMA，再比较名义均值—方差与稳健候选。区间稳健效用可写为 `mu·w - k*u·abs(w) - gamma/2*w' Sigma w`；其中 u 是研究员明确给出的均值不确定半宽，不是样本波动率，也不是自动生成的置信区间。Black–Litterman 用先验与观点形成后验参数，不负责免除下游约束与样本外检验。[2][6]

**TAA**：保留“不偏离”。固定方向+强度网格是可解释起点，但一次留出不足以描述不同市场阶段的稳定性。增加滚动/扩展训练窗口的分段样本外验证；每折选优仅使用当时已成熟训练收益，留出不重选赢家。多次查看留出后调整规则仍有研究者过拟合风险。模型失效、到期、成本超限与观点冲突应有明确结果，而非静默沿用旧权重。

**验证与解释**：区分模型风险、参数误差、历史偏差和实施误差。协方差必须对称/半正定，输入轴/单位必须一致；没有概率模型的确定性压力假设不能冒称 VaR/ES。候选搜索不能称为已证明全局最优；分段独立建仓的回测不能拼成连续实盘净值。

## 4. 当前代码事实与改造取舍

已用 CodeGraph 导航并核对实际源代码：

- `ClassAllocation.tsx → strategy_routes / analytics_routes → strategy / optimizer`：已有历史风险预算、目标优化与受约束候选探索；不是完整前瞻 CMA 工作流。
- `TacticalAllocationData.create_baseline/load_data`：可冻结真实分类/产品映射与净值来源，并严格对齐；可以复用为新政策工作流的来源边界。
- `TacticalAllocationService → numeric.evaluate_candidates → _taa_path_kernel`：已有趋势/人工/发布状态、训练/留出、成本、不可变决策和产品预算守恒，不应重写。
- `ArtifactRepository`：已有受控磁盘、原子写入、内容哈希与只读 NPY，可复用，不另造存储。
- `/pre-investment/objectives` 仍指向演示；SAA 缺独立 CMA 与政策采纳对象；TAA 尚无多折验证。

本轮优先交付真实目标对象、前瞻假设、经济类别说明、CMA 政策候选/确认与 TAA 继承、滚动验证和研究区间一致性。外部自动交易、完整负债模型、私募现金流、真实多币种对冲、专有 MRI/VCMM、自动宏观预测以及全套审批权限不是本轮能由现有数据证实的能力；不以按钮或演示数据冒充实现。

## 来源（均为第一方）

[1] CFA Institute, Overview of Asset Allocation, 2026 curriculum: https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/overview-asset-allocation

[2] CFA Institute, Principles of Asset Allocation, 2026 curriculum: https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/principles-asset-allocation

[3] CFA Institute, Asset Allocation with Real-World Constraints, 2026 curriculum: https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/asset-allocation-with-real-world-constraints

[4] J.P. Morgan, 2026 LTCMA release, 2025-10-20: https://am.jpmorgan.com/us/en/asset-management/adv/about-us/media/press-releases/jp-morgan-releases-2026-long-term-capital-market-assumptions/

[5] J.P. Morgan, LTCMA matrices/methodology: https://am.jpmorgan.com/mx/en/asset-management/adv/market-insights/ltcma/

[6] BlackRock, Capital market assumptions, accessed 2026-09-12: https://www.blackrock.com/us/financial-professionals/insights/capital-market-assumptions

[7] Vanguard, Portfolio construction framework, accessed 2026-09-12: https://www.vanguard.co.uk/professional/vanguard-365/investment-knowledge/portfolio-construction/portfolio-construction-framework

[8] State Street, ALLW brings Bridgewater’s All Weather wisdom to ETF investors, accessed 2026-09-12: https://www.ssga.com/us/en/institutional/insights/allw-brings-bridgewaters-all-weather-wisdom-to-etf-investors

[9] State Street, Expanding the sources of portfolio alpha with quant-anchored TAA, 2026-08-31: https://www.ssga.com/us/en/institutional/insights/portfolio-alpha-through-quant-anchored-taa

[10] AQR, Global Macro Strategies, accessed 2026-09-12: https://funds.aqr.com/Insights/Strategies/Global-Macro

[11] AQR, (Systematic) Investing in Emerging Market Debt, 2020-03-04: https://www.aqr.com/Insights/Research/Journal-Article/Systematic-Investing-in-Emerging-Market-Debt
