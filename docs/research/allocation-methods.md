# 配置研究：方法来源与选择理由

保留影响当前架构的研究决定；资料属于原检索日期记录，本次文档整理没有重新联网核验。[投前研究](../pre-investment/README.md)给出当前流程与能力状态。
## 研究边界与更正

本次以机构官网、CFA课程公开材料及作者论文为依据，不把营销文字、回测业绩或未公开专有模型当作经过独立验证的事实。CFA是专业教育框架，不是对所有资管机构强制适用的法律标准。公开资料只能证明其披露的流程与方法，不能证明复制了机构内部投研系统。

上一轮讨论中的“机构化75%”“超过大量软件”等无量化比较依据，不进入需求与验收。重新检查代码后确认：系统已经有区间稳健效用、资金目标现金流模拟、基准相对约束、收益标签成熟度校验和Walk-forward；本次复用，不重复建设。

“先产品池一定错误”也不成立。CFA说明战略配置与实施通常分开，但允许联合决策。适合本项目的改进是分开战略机会集与产品可投资域，同时保留当前由产品构建大类的路径，而不是删除或强迫重排已有工作流。[S1]

## 可核验资料

以下链接来自原研究阶段的联网核验；本次文档整理未重新联网。原检索日期不代表永久有效。机构观点不自动转为本系统默认参数。

|编号|一手来源|可用于设计的事实|
|---|---|---|
|S1|[CFA 2026 Principles of Asset Allocation](https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/principles-asset-allocation)|战略资产配置与具体证券/基金/账户实施通常分层；同时存在资产单边、负债相对、目标导向等配置框架。|
|S2|[CFA 2026 Overview of Asset Allocation](https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/overview-asset-allocation)|治理需要明确决策权、职责、IPS/SAA审批、报告和复核；战略配置须与整体经济状况匹配。|
|S3|[CFA 2026 Asset Allocation with Real-World Constraints](https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/asset-allocation-with-real-world-constraints)|规模、期限、流动性、税务等影响配置与再平衡；目标、约束和信念变化可以触发战略复核。|
|S4|[BlackRock Capital Market Assumptions](https://www.blackrock.com/us/financial-professionals/insights/capital-market-assumptions)|披露以情景、预期不确定性和多期收益路径支持配置；使用Kalman框架扩展单期观点融合，关注尾部和现金流。|
|S5|[Vanguard: The power of portfolio diversification](https://www.vanguard.co.uk/professional/vanguard-365/investment-knowledge/principles-of-investing/the-power-of-diversification)|VCMM提供收益分布，VAAM结合预期、约束和偏好做配置，并评估投资目标可行性。|
|S6|[PIMCO Investment Process](https://www.pimco.com/us/en/about-us/our-process)|每年一次Secular Forum与三次Cyclical Forum；投委会将宏观观点转为具体风险目标，并结合资产研究与风控。不能笼统说成每年四次Cyclical Forum。|
|S7|[J.P. Morgan: Leveraging the power of cash segmentation](https://am.jpmorgan.com/us/en/asset-management/liq/insights/liquidity-insights/leveraging-the-power-of-cash-segmentation/)|企业现金管理先预测现金流，再区分经营、储备、战略现金；各自有不同可用性、期限、风险与实施工具。|
|S8|[Cambridge Associates: Liquidity Hazard Planning for Families of Wealth](https://www.cambridgeassociates.com/insight/liquidity-hazard-planning-for-families-of-wealth/)|家族资产研究应同时考虑支出、非流动投资、币种和压力下流动性；不能假定退出和分配会准时补足资金。|
|S9|[UBS Global Family Office Report 2026](https://www.ubs.com/global/en/media/display-page-ndp/en-20260528-global-family-office-report-2026.html)|家办实践覆盖战略配置、风险与治理；不同家办治理成熟度存在差异，不存在单一全球家办工作流。|
|S10|[Idzorek: A Step-by-Step Guide to the Black–Litterman Model](https://www.cis.upenn.edu/~mkearns/finread/idzorek.pdf)|市场均衡先验、绝对/相对观点和观点不确定性共同形成后验预期；观点置信设置不是实际成功概率。|
|S11|[AQR / Asness, Moskowitz, Pedersen: Value and Momentum Everywhere](https://www.aqr.com/Insights/Research/Journal-Article/Value-and-Momentum-Everywhere)|跨市场研究支持将价值与动量作为不同信号来源；研究证据不等同于本项目产品集上的有效性认证。|
|S12|[AQR / Koijen et al.: Carry](https://www.aqr.com/Insights/Research/Journal-Article/Carry)|Carry有明确的事前经济定义，不能把价格动量改名成Carry；不同资产需对应收益率、远期或持有收益数据。|
|S13|[Luxenberg et al.: Strategic Asset Allocation with Illiquid Alternatives](https://arxiv.org/abs/2207.07767)|私募承诺、催缴、分配和资产净值不是同一变量；随机现金流与延迟实施适合多期控制，不能把私募视为每日可交易ETF。|
|S14|[Cunha Oliveira et al.: Tactical Asset Allocation with Macroeconomic Regime Detection](https://arxiv.org/abs/2503.11499)|宏观状态分类、状态预测和资产配置是不同环节；论文结果需独立复现，不能把状态识别直接当成交易信号。|

## 从机构业务到本系统的决定

### 01：投资目标、经济状况与边界

共同主链是“资金用途/义务 → 风险承受能力 → 授权边界 → 定量诊断 → 复核确认”。企业经营资金不应因为最大波动率设得较高就被当作长期风险资金；家办不能仅用账户净值代替家庭经济资产负债表。[S2、S3、S7、S8]

实施：保留现有三类目标和现金流诊断；增加机构场景、经济资产/义务说明、明确的现金储备要求与待人工核验事项。能计算的字段进入真实约束；税务、监管、对冲及外部审批仅保存有边界的人工证据，不伪造合规或审批结果。没有数据与模型时明确未评估。

### 02：战略需要与可买产品分开

战略机会集回答“希望持有哪些经济风险”，产品池回答“研究日有什么合格工具”。两者可迭代，但不能通过改产品名字创造战略暴露。[S1]

实施：允许独立定义、预览和确认战略资产范围；继续保留产品池版本、历史回放、手动/自动大类构建。通过显式映射做实施覆盖检查。缺产品的战略资产仍保留在SAA研究里；不填零收益、不自动分给其他产品、不隐藏缺口。进入TAA或产品应用前另行检查真实代理和映射是否齐全。

### 03：CMA生成与政策求解分层

前沿实践把对未来的假设及其不确定性当作模型输入，不把历史最优权重包装成政策结论。[S4、S5、S10]

实施：保留人工CMA、历史风险参考及现有四类政策候选。新增可解释的Black–Litterman预期融合、显式概率情景的矩匹配/比较与风险预算候选。各方法复用相同约束、绩效/风险计算、资金目标门禁和不可变发布。明确有限候选不等于全局最优；均值不确定性不等于波动率；后验均值协方差不等于资产收益协方差。

### 04：信号、决策与交易时钟分开

研究观察可以高频，投资判断和执行则有自己的授权、复核和成本边界。[S1、S3、S6] Value、Carry、Macro不能凭空从NAV推导。[S11、S12、S14]

实施：原有Momentum/Manual/Regime继续可用；添加明确的决策与再平衡策略，非交易日让持仓自然漂移，并保留滞后和成本。支持多个有来源、有可得日的信号组合；没有真实信号数据时不可启用相应来源。训练、Holdout、Walk-forward、当前建议及下游应用使用一致的时点和约束，不保留第二套业务计算引擎。

## 明确不在本次伪造的能力

完整的税法引擎、保险/养老金监管资本、随机负债ALM、FX远期定价及保证金、杠杆/衍生品交易、私募基金逐笔承诺pacing和真实下单，需要独立数据与业务授权。本次提供明确的需求/证据边界和必要门禁，不将说明字段、模拟代理或按钮称为这些能力已经实现。也不引入外部黑盒求解器来绕过固定签名NJIT约束。
## 模型选择与工程判断

**大类构建**：经济分类作为政策层，相关聚类/PCA作为证据层，具体 ETF/基金作为实施层。政策类别要记录投资角色、收益计价币种、流动性说明及代理选择理由。自动分类结果仍须人工确认经济解释；历史同涨跌不保证未来同风险。

**SAA**：等权/风险预算可作基线；历史 MVO 对预期收益误差敏感。应先建立可版本化 CMA，再比较名义均值—方差与稳健候选。区间稳健效用可写为 `mu·w - k*u·abs(w) - gamma/2*w' Sigma w`；其中 u 是研究员明确给出的均值不确定半宽，不是样本波动率，也不是自动生成的置信区间。Black–Litterman 用先验与观点形成后验参数，不负责免除下游约束与样本外检验。[2][6]

**TAA**：保留“不偏离”。固定方向+强度网格是可解释起点，但一次留出不足以描述不同市场阶段的稳定性。增加滚动/扩展训练窗口的分段样本外验证；每折选优仅使用当时已成熟训练收益，留出不重选赢家。多次查看留出后调整规则仍有研究者过拟合风险。模型失效、到期、成本超限与观点冲突应有明确结果，而非静默沿用旧权重。

**验证与解释**：区分模型风险、参数误差、历史偏差和实施误差。协方差必须对称/半正定，输入轴/单位必须一致；没有概率模型的确定性压力假设不能冒称 VaR/ES。候选搜索不能称为已证明全局最优；分段独立建仓的回测不能拼成连续实盘净值。
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

## 授权为何先于优化

投资目标必须区分客户／投资者的事实、愿望与治理授权。资金所需收益来自现金流求解，不作为市场收益预测；风险承受能力和授权上限不能由优化器静默提高。先做参考诊断，再在实际范围和正式 CMA 下验证可行性。无法满足时展示投入、期限、目标或授权的调整情景，由人确认新版本。

历史有效前沿、战略政策与战术偏离分别管理；不把漂亮回测曲线当作已批准政策，不把产品池研究资格直接等同于可交易性。
