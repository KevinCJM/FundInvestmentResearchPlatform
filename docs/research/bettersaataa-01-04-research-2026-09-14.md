# 投前 01–04：机构流程与模型调研

调研日：2026-09-14。开发分支：`ISSUE2609/BetterSaaTaa`。代码基线：`aa8ca7a8a707ed1cc10f4ef624f1ceb5894ebada`（origin/Dev）。

## 1. 研究边界与更正

本次以机构官网、CFA课程公开材料及作者论文为依据，不把营销文字、回测业绩或未公开专有模型当作经过独立验证的事实。CFA是专业教育框架，不是对所有资管机构强制适用的法律标准。公开资料只能证明其披露的流程与方法，不能证明复制了机构内部投研系统。

上一轮讨论中的“机构化75%”“超过大量软件”等无量化比较依据，不进入需求与验收。重新检查代码后确认：系统已经有区间稳健效用、资金目标现金流模拟、基准相对约束、收益标签成熟度校验和Walk-forward；本次复用，不重复建设。

“先产品池一定错误”也不成立。CFA说明战略配置与实施通常分开，但允许联合决策。适合本项目的改进是分开战略机会集与产品可投资域，同时保留当前由产品构建大类的路径，而不是删除或强迫重排已有工作流。[S1]

## 2. 可核验资料

以下链接均在本轮联网核验；日期只代表检索时点，不代表永久有效。机构观点不自动转为本系统默认参数。

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

## 3. 从机构业务到本系统的决定

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

## 4. 明确不在本次伪造的能力

完整的税法引擎、保险/养老金监管资本、随机负债ALM、FX远期定价及保证金、杠杆/衍生品交易、私募基金逐笔承诺pacing和真实下单，需要独立数据与业务授权。本次提供明确的需求/证据边界和必要门禁，不将说明字段、模拟代理或按钮称为这些能力已经实现。也不引入外部黑盒求解器来绕过固定签名NJIT约束。

## 5. 代码事实

- `backend/strategic_allocation/contracts.py`：目标、资金计划、基准、CMA、政策请求。
- `backend/strategic_allocation/kernels.py`：已有有限候选搜索、区间稳健效用、前瞻风险与贡献。
- `backend/strategic_allocation/service.py`：已有目标诊断、确认、CMA与政策保存；当前来源绑定已保存产品大类。
- `backend/tactical_allocation/service.py`：已有标签成熟度、训练/留出隔离、预览哈希复核和下游交接。
- `frontend/src/app/allocationJourney.ts`：已有产品域→大类→基线→TAA的草稿及引用链，需要加入目标和战略范围引用。

开发前已运行相关后端基线：183 passed；结果与命令记录在本次验收文档。具体接口、模型和验收见需求与详细设计文档。
