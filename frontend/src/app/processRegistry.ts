export type StageId =
  | 'product-research'
  | 'pre-investment'
  | 'portfolio-center'
  | 'investment-execution'
  | 'fund-accounting'
  | 'post-investment'
  | 'feedback'
  | 'portfolio-solutions'
  | 'settings'

export type CapabilityStatus = 'available' | 'partial' | 'prototype'

export interface ProcessNodeDefinition {
  id: string
  label: string
  description: string
  path: string
  status: CapabilityStatus
}

export interface StageToolDefinition {
  label: string
  description: string
  path: string
}

export interface StageDefinition {
  id: StageId
  order?: number
  label: string
  eyebrow: string
  description: string
  path: string
  accent: {
    badge: string
    border: string
    button: string
    soft: string
    text: string
  }
  nodes: ProcessNodeDefinition[]
  tools?: StageToolDefinition[]
}

const productResearch: StageDefinition = {
  id: 'product-research',
  order: 1,
  label: '产品研究',
  eyebrow: 'Research foundation',
  description: '独立于单次投资持续运行，沉淀可复用、可追溯的产品研究结论与产品池。',
  path: '/product-research',
  accent: {
    badge: 'bg-sky-100 text-sky-800',
    border: 'border-sky-200',
    button: 'bg-sky-700 hover:bg-sky-600',
    soft: 'bg-sky-50',
    text: 'text-sky-700',
  },
  nodes: [
    { id: 'panorama', label: '基金市场概览', description: '展示 ETF 与场外公募基金的市场规模、产品分布、关键指标和研究入口。', path: '/product-research/panorama', status: 'available' },
    { id: 'products', label: '产品与管理人研究', description: '研究产品业绩、风险、风格、费用、流动性及管理人。', path: '/product-research/products', status: 'partial' },
    { id: 'holding-style', label: '持仓穿透与风格研究', description: '区分披露持仓与估算暴露，研究底层资产、行业风格、因子暴露及披露时滞。', path: '/product-research/holding-style', status: 'prototype' },
    { id: 'evaluation', label: '产品评价与分类', description: '通过版本化指标方案完成产品评价、排名与分类。', path: '/product-research/evaluation', status: 'partial' },
    { id: 'product-backtest', label: '产品与信号回测', description: '验证单产品持有、择时信号与评价逻辑的历史有效性。', path: '/product-research/product-backtest', status: 'prototype' },
    { id: 'pools', label: '产品池构建', description: '直接引用多套产品评价方案，结合人工研究形成可复用产品池。', path: '/product-research/pools', status: 'available' },
    { id: 'pool-lifecycle', label: '产品池版本与生命周期管理', description: '管理不可变版本、生效状态、成员变化与研究证据。', path: '/product-research/pool-lifecycle', status: 'available' },
  ],
}

const preInvestment: StageDefinition = {
  id: 'pre-investment',
  order: 2,
  label: '投前决策',
  eyebrow: 'Portfolio decision',
  description: '针对单个组合，自上而下完成目标定义、大类配置、产品配置研究、方案定稿与外部审批结果记录。',
  path: '/pre-investment',
  accent: {
    badge: 'bg-violet-100 text-violet-800',
    border: 'border-violet-200',
    button: 'bg-violet-700 hover:bg-violet-600',
    soft: 'bg-violet-50',
    text: 'text-violet-700',
  },
  nodes: [
    { id: 'objectives', label: '投资目标与约束', description: '定义收益、风险、基准、期限、流动性和投资范围。', path: '/pre-investment/objectives', status: 'prototype' },
    { id: 'product-pool', label: '选择产品池版本', description: '选择已生效产品池版本并冻结为本次投前可投资域。', path: '/pre-investment/product-pool', status: 'available' },
    { id: 'saa', label: '战略资产配置（SAA）', description: '确定大类资产中枢权重、偏离区间与长期风险预算。', path: '/pre-investment/saa', status: 'partial' },
    { id: 'taa', label: '战术资产配置（TAA）', description: '引用已发布的实时因果情景版本，以状态概率驱动相对 SAA 的权重偏移并完成含成本回测。', path: '/pre-investment/taa', status: 'partial' },
    { id: 'product-allocation-timing', label: '产品配置与择时', description: '在大类资产预算内研究具体基金配置、产品替代关系和产品级择时规则。', path: '/pre-investment/product-allocation-timing', status: 'partial' },
    { id: 'portfolio-synthesis', label: '目标组合合成与风险检查', description: '合成大类与产品权重，检查边际风险贡献、风险预算、集中度及全部投资约束。', path: '/pre-investment/portfolio-synthesis', status: 'prototype' },
    { id: 'validation', label: '统一回测与稳健性验证', description: '通过样本内外、滚动验证、压力和成本模拟检验方案。', path: '/pre-investment/validation', status: 'partial' },
    { id: 'approval', label: '研究方案定稿与外部审批记录', description: '固化研究方案、数据截止日和版本，并记录平台外部审批结果。', path: '/pre-investment/approval', status: 'prototype' },
  ],
  tools: [
    { label: '大类资产构建', description: '定义大类及其 ETF/公募基金代理。', path: '/pre-investment/saa/asset-classes' },
    { label: '自动构建大类', description: '按收益相关性、风险画像或主成分自动划分大类，并给出代表产品、类内权重与分类诊断。', path: '/pre-investment/saa/auto-classification' },
    { label: '大类资产配置与策略回测', description: '效率前沿、风险预算、目标权重、调仓与回测。', path: '/pre-investment/saa/allocation-lab' },
  ],
}

const investmentExecution: StageDefinition = {
  id: 'investment-execution',
  order: 3,
  label: '投中执行',
  eyebrow: 'Execution',
  description: '基于组合中心登记的真实组合，完成方案落地、交易计划测算、执行结果衔接、资金核对和再平衡研究；不提供下单能力。',
  path: '/investment-execution',
  accent: {
    badge: 'bg-amber-100 text-amber-900',
    border: 'border-amber-200',
    button: 'bg-amber-700 hover:bg-amber-600',
    soft: 'bg-amber-50',
    text: 'text-amber-800',
  },
  nodes: [
    { id: 'onboarding', label: '组合落地与启用', description: '把外部已批准的目标组合版本关联至真实组合并登记平台主数据，不代替基金设立、开户或审批。', path: '/investment-execution/onboarding', status: 'prototype' },
    { id: 'trade-plan', label: '目标组合与交易计划测算', description: '结合资金、持仓、容量和交收周期测算组合级交易计划。', path: '/investment-execution/trade-plan', status: 'prototype' },
    { id: 'pre-trade-check', label: '交易前检查', description: '检查现金、申赎、容量、费用、流动性与投资限制。', path: '/investment-execution/pre-trade-check', status: 'prototype' },
    { id: 'trade-allocation', label: '汇总订单与公平交易分配', description: '汇总多个组合和基金经理的独立指令，并把成交按冻结规则拆回组合、投资单元和结算账户。', path: '/investment-execution/trade-allocation', status: 'prototype' },
    { id: 'cash-settlement', label: '现金与交收日历', description: '按 ETF 与场外基金的资金可用日、申赎时滞测算在途资金和现金拖累。', path: '/investment-execution/cash-settlement', status: 'prototype' },
    { id: 'booking', label: '外部账户事实与 Booking 衔接', description: '先拆分执行通道、银行与托管原始记录，再把完成组合归属的事实交由基金会计。', path: '/fund-accounting/account-statements', status: 'prototype' },
    { id: 'reconciliation', label: '投资持仓与资金核对', description: '从投资视角核对现金、份额、成本和目标组合之间的差异。', path: '/investment-execution/reconciliation', status: 'prototype' },
    { id: 'rebalancing', label: '再平衡测算与记录', description: '测算权重偏离与再平衡方案，并记录研究过程。', path: '/investment-execution/rebalancing', status: 'prototype' },
  ],
}

const portfolioCenter: StageDefinition = {
  id: 'portfolio-center',
  label: '组合中心',
  eyebrow: 'Actual portfolio registry',
  description: '统一登记真实组合及其研究方案、账户、核算主体、账簿和生命周期关系，是投中、基金会计与投后的共同主数据入口。',
  path: '/portfolio-center',
  accent: {
    badge: 'bg-indigo-100 text-indigo-900',
    border: 'border-indigo-200',
    button: 'bg-indigo-800 hover:bg-indigo-700',
    soft: 'bg-indigo-50',
    text: 'text-indigo-800',
  },
  nodes: [
    { id: 'register', label: '真实组合登记簿', description: '查看真实组合清单、类型、运行状态及关键责任主体。', path: '/portfolio-center/register', status: 'prototype' },
    { id: 'master', label: '组合主数据', description: '维护组合身份、策略、基准、币种、管理人、托管人和生效日期。', path: '/portfolio-center/master', status: 'prototype' },
    { id: 'relationships', label: '账户主档与组合关系', description: '独立维护执行、资金、托管和 TA 账户，并按账户类型控制是否允许关联多个组合。', path: '/portfolio-center/accounts', status: 'prototype' },
    { id: 'responsibilities', label: '基金经理与投资单元', description: '按有效期关联真实组合、内部 Sleeve、基金经理角色与管理人法人。', path: '/portfolio-center/responsibilities', status: 'prototype' },
    { id: 'versions', label: '研究方案与目标版本', description: '追溯来源研究方案、外部批准记录和历次目标组合版本。', path: '/portfolio-center/versions', status: 'prototype' },
    { id: 'lifecycle', label: '生命周期与状态', description: '记录筹备、待启用、运行、暂停、清算和终止等状态变化。', path: '/portfolio-center/lifecycle', status: 'prototype' },
  ],
}

const fundAccounting: StageDefinition = {
  id: 'fund-accounting',
  label: '基金会计',
  eyebrow: 'Accounting book of record',
  description: '与投资决策线并行并贯穿投中、投后，把业务事实转换为各核算主体独立平衡的复式凭证、账簿、估值、报表与绩效数据；当前仅为研究平台静态框架。',
  path: '/fund-accounting',
  accent: {
    badge: 'bg-cyan-100 text-cyan-900',
    border: 'border-cyan-200',
    button: 'bg-cyan-800 hover:bg-cyan-700',
    soft: 'bg-cyan-50',
    text: 'text-cyan-800',
  },
  nodes: [
    { id: 'policy', label: '核算主体、账簿与记账规则', description: '关联组合中心已有主体与账簿，并定义科目表、正常余额、确认计量、凭证模板和报表政策。', path: '/fund-accounting/policy', status: 'prototype' },
    { id: 'account-statements', label: '外部账户流水、拆分与结算匹配', description: '保存执行通道、银行及托管原始记录，按分配关系拆到组合并处理待认领悬账。', path: '/fund-accounting/account-statements', status: 'prototype' },
    { id: 'booking', label: '组合 Booking 与复式记账', description: '承接已完成账户和组合分配的业务事实，按核算主体、会计时点和政策版本生成借贷平衡凭证草稿。', path: '/fund-accounting/booking', status: 'prototype' },
    { id: 'portfolio-ledger', label: '组合/基金总账与明细账', description: '每只组合或基金独立过账，形成总账、明细账、试算平衡、持仓、现金、净资产及表外备查簿。', path: '/fund-accounting/portfolio-ledger', status: 'prototype' },
    { id: 'manager-ledger', label: '管理人公司账映射', description: '记录管理费、自有资金跟投及管理人自身业务，并与公司财务系统保持接口边界。', path: '/fund-accounting/manager-ledger', status: 'prototype' },
    { id: 'valuation-nav', label: '估值、应计与净资产核算', description: '处理公允价值、利息与费用应计、汇率、份额和净资产，并保留估值调整版本。', path: '/fund-accounting/valuation-nav', status: 'prototype' },
    { id: 'cross-book-reconciliation', label: '双账勾稽、差错与关账', description: '核对 Booking、组合/基金账、管理人账、托管与银行数据，处理差错并冻结各自账期版本。', path: '/fund-accounting/cross-book-reconciliation', status: 'prototype' },
    { id: 'financial-statements', label: '双主体财务报表', description: '分别生成组合/基金报表和管理人公司报表视图，不直接混加；合并仅作为需单独判断的会计事项。', path: '/fund-accounting/financial-statements', status: 'prototype' },
    { id: 'performance-dataset', label: '绩效核算数据集（PBOR）', description: '输出冻结估值、外部资金流、内部交易、费用和基准口径，供组合层 TWR、MWR/XIRR 与归因计算。', path: '/fund-accounting/performance-dataset', status: 'prototype' },
  ],
}

const postInvestment: StageDefinition = {
  id: 'post-investment',
  order: 4,
  label: '投后管理',
  eyebrow: 'Post investment',
  description: '基于会计核算形成的实际组合事实，解释投资结果、识别收益与风险来源并形成可审计的研究结论。',
  path: '/post-investment',
  accent: {
    badge: 'bg-emerald-100 text-emerald-800',
    border: 'border-emerald-200',
    button: 'bg-emerald-700 hover:bg-emerald-600',
    soft: 'bg-emerald-50',
    text: 'text-emerald-700',
  },
  nodes: [
    { id: 'portfolio-data', label: '实际组合数据与三账核对', description: '读取并核对 IBOR、ABOR、PBOR 的交易、持仓、现金、费用、外部资金流和估值快照。', path: '/post-investment/portfolio-data', status: 'prototype' },
    { id: 'performance', label: '绩效计量', description: '基于冻结估值与外部资金流计算 TWR、XIRR、风险和相对基准表现，不把内部买卖当作申赎。', path: '/post-investment/performance', status: 'prototype' },
    { id: 'return-attribution', label: '收益归因', description: '拆解配置、择时、产品选择、交互与执行偏差。', path: '/post-investment/return-attribution', status: 'prototype' },
    { id: 'risk-attribution', label: '风险归因', description: '分析大类、产品、行业风格与风险因子的贡献。', path: '/post-investment/risk-attribution', status: 'prototype' },
    { id: 'monitoring', label: '持仓产品与管理人监控', description: '跟踪业绩、风格漂移、规模、流动性和团队变化。', path: '/post-investment/monitoring', status: 'prototype' },
    { id: 'scenarios', label: '情景分析与压力测试', description: '评估历史或自定义冲击下的组合损失与流动性风险。', path: '/post-investment/scenarios', status: 'prototype' },
    { id: 'conclusions', label: '投后研究结论', description: '固化基于业绩、归因、监控和情景分析形成的研究结论。', path: '/post-investment/conclusions', status: 'prototype' },
    { id: 'reports', label: '投资报告与归档', description: '归档投资结果、原因解释、风险提示和相关版本。', path: '/post-investment/reports', status: 'prototype' },
    { id: 'accounting-reports', label: '组合财务报表与净值核验', description: '查看组合/基金账生成的三张核心报表、附注及报表间勾稽结果。', path: '/fund-accounting/financial-statements', status: 'prototype' },
  ],
  tools: [
    { label: '研究组合诊断', description: '基于不可变研究快照查看净值、贡献、矩阵和历史情景。', path: '/post-investment/research-diagnosis' },
  ],
}

const feedback: StageDefinition = {
  id: 'feedback',
  order: 5,
  label: '反馈与迭代',
  eyebrow: 'Learning loop',
  description: '把实际投资结果反馈给产品研究、配置模型和投资流程，形成可审计的持续改进。',
  path: '/feedback',
  accent: {
    badge: 'bg-rose-100 text-rose-800',
    border: 'border-rose-200',
    button: 'bg-rose-700 hover:bg-rose-600',
    soft: 'bg-rose-50',
    text: 'text-rose-700',
  },
  nodes: [
    { id: 'results', label: '投资结果反馈', description: '汇集实际收益、风险、归因、执行偏差和投后事件。', path: '/feedback/results', status: 'prototype' },
    { id: 'hypothesis-review', label: '研究假设复盘', description: '比较投前假设与实际结果并解释差异来源。', path: '/feedback/hypothesis-review', status: 'prototype' },
    { id: 'product-review', label: '产品复审与产品池更新', description: '重新评价相关产品并发布新的产品池版本。', path: '/feedback/product-review', status: 'prototype' },
    { id: 'model-update', label: '配置与模型更新', description: '评估并更新配置、择时、风险预算和再平衡参数。', path: '/feedback/model-update', status: 'prototype' },
    { id: 'rolling-validation', label: '滚动复验与模型比较', description: '通过样本外和敏感性分析验证改进稳定性。', path: '/feedback/rolling-validation', status: 'prototype' },
    { id: 'process-improvement', label: '投资流程改进', description: '把复盘结论固化为新的流程、规则和控制要求。', path: '/feedback/process-improvement', status: 'prototype' },
  ],
}

const portfolioSolutions: StageDefinition = {
  id: 'portfolio-solutions',
  label: '组合方案展示中心',
  eyebrow: 'Portfolio solutions',
  description: '承接已定稿的组合研究版本，形成面向产品、渠道与客户的标准化组合展示；不参与单个组合的研究决策闭环。',
  path: '/portfolio-solutions',
  accent: {
    badge: 'bg-fuchsia-100 text-fuchsia-800',
    border: 'border-fuchsia-200',
    button: 'bg-fuchsia-700 hover:bg-fuchsia-600',
    soft: 'bg-fuchsia-50',
    text: 'text-fuchsia-700',
  },
  nodes: [
    { id: 'catalog', label: '组合方案目录', description: '按风险等级、期限、策略类型和发布状态管理可展示的组合方案。', path: '/portfolio-solutions/catalog', status: 'prototype' },
    { id: 'profile', label: '组合画像与适用范围', description: '展示组合定位、目标、风险等级、适用客群、配置中枢和服务边界。', path: '/portfolio-solutions/profile', status: 'prototype' },
    { id: 'performance', label: '收益与风险表现', description: '按统一口径展示组合、基准、回撤、风险指标和分阶段表现。', path: '/portfolio-solutions/performance', status: 'prototype' },
    { id: 'backtest', label: '历史回测模拟', description: '展示冻结版本的样本内外回测、成本假设、再平衡规则和稳健性结果。', path: '/portfolio-solutions/backtest', status: 'prototype' },
    { id: 'scenarios', label: '周期与情景模拟', description: '展示不同经济周期、历史危机和自定义冲击下的模拟表现。', path: '/portfolio-solutions/scenarios', status: 'prototype' },
    { id: 'rules', label: '配置与算法说明', description: '说明资产范围、配置中枢、信号输入、约束、调仓频率和算法版本。', path: '/portfolio-solutions/rules', status: 'prototype' },
    { id: 'publishing', label: '披露与展示版本', description: '冻结基准、费用、数据日期、回测标识、风险揭示和对外展示版本。', path: '/portfolio-solutions/publishing', status: 'prototype' },
  ],
}

const settings: StageDefinition = {
  id: 'settings',
  label: '设置',
  eyebrow: 'Platform capabilities',
  description: '统一管理数据、研究口径、指标、情景算法、回测、规则版本和公共参数，不进入单个组合的投资主流程。',
  path: '/settings',
  accent: {
    badge: 'bg-slate-200 text-slate-800',
    border: 'border-slate-300',
    button: 'bg-slate-800 hover:bg-slate-700',
    soft: 'bg-slate-100',
    text: 'text-slate-700',
  },
  nodes: [
    { id: 'data-sources', label: '数据同步与任务', description: '日常更新入口：选择需要的数据，确认增量或全量，查看进度并恢复中断任务。', path: '/settings/data-sources', status: 'available' },
    { id: 'source-center', label: '数据源与接口映射', description: '首次接入或修改配置时使用；Tushare 已预置，支持字段映射、限制设置和样本验证。', path: '/settings/source-center', status: 'available' },
    { id: 'data-model', label: '系统数据模型', description: '查看外部数据需要映射的标准表与字段；系统内部结构在高级入口中查看。', path: '/settings/data-model', status: 'available' },
    { id: 'data-quality', label: '数据质量', description: '监控完整性、及时性、唯一性、产品映射、账户对账和异常处理状态。', path: '/settings/data-quality', status: 'partial' },
    { id: 'research-data-lab', label: '研究数据实验室', description: '搜索并检查指数、宏观、指标版本与上传序列，研究覆盖、缺失、PIT 和时序表现后加入计算图。', path: '/settings/research-data-lab', status: 'available' },
    { id: 'pit-snapshots', label: 'PIT 时点快照', description: '体检每张表的事件时间与可得时间、公告滞后与 PIT 等级，并对数据整体封版，供研究上下文引用。', path: '/settings/pit-snapshots', status: 'available' },
    { id: 'research-parameters', label: '研究口径与参数中心', description: '维护可复用、版本化的基准、无风险利率、数据处理和计算口径模板，供产品与组合研究引用。', path: '/settings/research-parameters', status: 'prototype' },
    { id: 'indicators-models', label: '指标与模型管理', description: '维护标量、矩阵及组合指标的定义、公式、范围和版本。', path: '/settings/indicators-models', status: 'partial' },
    { id: 'factor-research', label: '因子研究中心', description: '构建和检验共享因子，区分比较基准、模型与数据来源，发布研究版本并接入产品池和投研证据。', path: '/settings/factor-research', status: 'available' },
    { id: 'scenario-algorithms', label: '情景算法中心', description: '用版本化 Regime Graph 编排数据、算子、模型和状态处理；情景模拟与压测保持独立入口。', path: '/settings/scenario-algorithms', status: 'available' },
    { id: 'backtest-center', label: '统一回测中心', description: '管理回测引擎、规则模板、运行任务和结果档案。', path: '/settings/backtest-center', status: 'prototype' },
    { id: 'system-parameters', label: '系统参数', description: '管理枚举字典、时区、精度、功能开关和运行环境等技术参数。', path: '/settings/system-parameters', status: 'prototype' },
    { id: 'language-terminology', label: '语言与术语', description: '查看内置系统翻译，维护独立的业务术语与多语言覆盖。', path: '/settings/language-terminology', status: 'available' },
  ],
}

export const businessStages = [productResearch, preInvestment, investmentExecution, postInvestment, feedback]
export const portfolioCenterStage = portfolioCenter
export const accountingStage = fundAccounting
export const supportingStages = [portfolioCenter, portfolioSolutions, settings]
export const allStages = [...businessStages, portfolioCenter, fundAccounting, portfolioSolutions, settings]

export const getStage = (stageId: StageId) => {
  const stage = allStages.find((item) => item.id === stageId)
  if (!stage) throw new Error(`Unknown process stage: ${stageId}`)
  return stage
}

export const statusLabels: Record<CapabilityStatus, string> = {
  available: '已有功能',
  partial: '部分具备',
  prototype: '静态演示',
}
